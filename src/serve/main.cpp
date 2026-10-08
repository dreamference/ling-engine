// ling-serve: the HTTP front of ling-engine (proxygen). One engine worker thread runs requests one at a
// time; the HTTP threads parse requests and stream server-sent events back.
//
//   ling-serve --model DIR [--host 0.0.0.0] [--port 8000] [--served-model-name NAME] [--max-context N]
#include <folly/SocketAddress.h>
#include <folly/init/Init.h>
#include <folly/io/IOBufQueue.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>
#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <proxygen/httpserver/ResponseBuilder.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <ctime>
#include <deque>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include "core/engine.hpp"
#include "serve/api.hpp"
#include "tokenizer/tokenizer.hpp"

namespace ling::serve {
namespace {

using proxygen::HTTPMessage;
using proxygen::ResponseBuilder;

struct ServerContext {
  Engine* engine = nullptr;
  Tokenizer* tok = nullptr;
  std::string model_name;
  int im_end = -1;
};

// What a job reports back, called on the engine thread.
struct JobSink {
  std::function<void(const std::string&)> text;
  std::function<void(const std::string& finish, int prompt_tokens, int completion_tokens, int cached_tokens)> done;
  std::function<void(const std::string&)> error;
};

struct Job {
  Request req;
  std::vector<int> prompt;
  JobSink sink;
  std::shared_ptr<std::atomic<bool>> cancelled;
};

class Worker {
 public:
  explicit Worker(ServerContext& ctx) : ctx_(ctx), thread_([this] { run(); }) {}
  ~Worker() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  void submit(Job job) {
    {
      std::lock_guard<std::mutex> l(mu_);
      queue_.push_back(std::move(job));
    }
    cv_.notify_one();
  }

 private:
  void run() {
    while (true) {
      Job job;
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return stop_ || !queue_.empty(); });
        if (stop_) return;
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      try {
        generate(job);
      } catch (const std::exception& e) {
        job.sink.error(e.what());
      }
    }
  }

  void generate(Job& job) {
    Engine& eng = *ctx_.engine;
    const int ctx_left = eng.max_context() - static_cast<int>(job.prompt.size());
    if (ctx_left <= 0) throw std::invalid_argument("the prompt is longer than the context limit");
    const int limit = job.req.max_tokens > 0 ? std::min(job.req.max_tokens, ctx_left) : ctx_left;
    EngineStats stats;
    std::vector<float> logits = eng.prefill(job.prompt, &stats);
    StreamDecoder dec(*ctx_.tok);
    std::string all;
    size_t emitted = 0, max_stop = 0;
    for (const auto& s : job.req.stop) max_stop = std::max(max_stop, s.size());
    std::string finish = "length";
    int produced = 0;
    for (; produced < limit;) {
      if (job.cancelled->load()) {
        finish = "cancelled";
        break;
      }
      const int t = eng.sample(logits, job.req.sampling, eng.history());
      if (t == eng.config().eos_token || t == ctx_.im_end) {
        finish = "stop";
        break;
      }
      ++produced;
      all += dec.push(t);
      size_t stop_at = std::string::npos;
      for (const auto& s : job.req.stop) {
        size_t p = all.find(s, emitted > s.size() ? emitted - s.size() : 0);
        if (p != std::string::npos) stop_at = std::min(stop_at, p);
      }
      if (stop_at != std::string::npos) {
        if (stop_at > emitted) job.sink.text(all.substr(emitted, stop_at - emitted));
        emitted = all.size();
        finish = "stop";
        break;
      }
      const size_t safe = max_stop > 0 && all.size() >= max_stop - 1 ? all.size() - (max_stop - 1) : (max_stop ? 0 : all.size());
      if (safe > emitted) {
        job.sink.text(all.substr(emitted, safe - emitted));
        emitted = safe;
      }
      if (produced < limit) logits = eng.step(t);
    }
    if (finish != "stop" || emitted < all.size()) {
      all += dec.flush();
      if (all.size() > emitted) job.sink.text(all.substr(emitted));
    }
    job.sink.done(finish == "cancelled" ? "stop" : finish, static_cast<int>(job.prompt.size()), produced,
                  stats.reused_tokens);
  }

  ServerContext& ctx_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> queue_;
  bool stop_ = false;
  std::thread thread_;
};

// Lives as long as either the HTTP handler or the job needs it; touched only on the handler's thread.
struct Exchange {
  proxygen::ResponseHandler* downstream = nullptr;  // null once the client is gone
  folly::EventBase* evb = nullptr;
  std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
};

void post(const std::shared_ptr<Exchange>& ex, std::function<void(proxygen::ResponseHandler*)> fn) {
  ex->evb->runInEventBaseThread([ex, fn = std::move(fn)] {
    if (ex->downstream) fn(ex->downstream);
  });
}

void send_json(proxygen::ResponseHandler* d, int status, const json& body) {
  ResponseBuilder(d)
      .status(status, status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" : "Error")
      .header("Content-Type", "application/json")
      .body(body.dump())
      .sendWithEOM();
}

json error_body(const std::string& msg, const std::string& type) {
  return json{{"error", {{"message", msg}, {"type", type}}}};
}

class Handler : public proxygen::RequestHandler {
 public:
  Handler(ServerContext& ctx, Worker& worker) : ctx_(ctx), worker_(worker) {}

  void onRequest(std::unique_ptr<HTTPMessage> headers) noexcept override {
    method_ = headers->getMethodString();
    path_ = headers->getPath();
    ex_->downstream = downstream_;
    ex_->evb = folly::EventBaseManager::get()->getEventBase();
  }
  void onBody(std::unique_ptr<folly::IOBuf> body) noexcept override {
    body_.append(std::move(body));
    if (body_.chainLength() > (64u << 20)) too_large_ = true;
  }
  void onEOM() noexcept override {
    try {
      route();
    } catch (const std::exception& e) {
      send_json(downstream_, 500, error_body(e.what(), "server_error"));
    }
  }
  void onUpgrade(proxygen::UpgradeProtocol) noexcept override {}
  void requestComplete() noexcept override { finish(); }
  void onError(proxygen::ProxygenError) noexcept override { finish(); }

 private:
  void finish() {
    ex_->downstream = nullptr;
    ex_->cancelled->store(true);
    delete this;
  }

  void route() {
    if (method_ == "GET" && (path_ == "/health" || path_ == "/healthz")) {
      ResponseBuilder(downstream_).status(200, "OK").body("ok").sendWithEOM();
    } else if (method_ == "GET" && path_ == "/v1/models") {
      json model = {{"id", ctx_.model_name}, {"object", "model"}, {"created", std::time(nullptr)},
                    {"owned_by", "ling-engine"}, {"root", ctx_.model_name},
                    {"max_model_len", ctx_.engine->max_context()}};
      send_json(downstream_, 200, json{{"object", "list"}, {"data", json::array({model})}});
    } else if (method_ == "POST" && (path_ == "/v1/chat/completions" || path_ == "/v1/completions")) {
      generate(path_ == "/v1/chat/completions");
    } else {
      send_json(downstream_, 404, error_body("no route for " + method_ + " " + path_, "not_found"));
    }
  }

  void generate(bool chat) {
    if (too_large_) {
      send_json(downstream_, 413, error_body("request body too large", "invalid_request_error"));
      return;
    }
    Job job;
    try {
      std::string text;
      if (auto buf = body_.move()) text = buf->moveToFbString().toStdString();
      json body = json::parse(text);
      job.req = parse_request(body, chat);
    } catch (const std::exception& e) {
      send_json(downstream_, 400, error_body(e.what(), "invalid_request_error"));
      return;
    }
    job.prompt = job.req.prompt_ids.empty() ? ctx_.tok->encode(job.req.prompt_text) : job.req.prompt_ids;
    job.cancelled = ex_->cancelled;
    const std::string id = new_id(chat ? "chatcmpl-" : "cmpl-");
    const long created = std::time(nullptr);
    const std::string model = ctx_.model_name;
    auto ex = ex_;
    const Request req = job.req;

    if (req.stream) {
      ResponseBuilder(downstream_)
          .status(200, "OK")
          .header("Content-Type", "text/event-stream")
          .header("Cache-Control", "no-cache")
          .send();
    }
    // Shared by the job's callbacks (engine thread only).
    struct State {
      OutputParser parser;
      std::string content, reasoning;
      std::vector<ToolCall> calls;
      bool first = true;
      State(bool r, const json& t) : parser(r, t) {}
    };
    auto st = std::make_shared<State>(chat && req.reasoning, req.tools);

    auto sse = [ex](const json& chunk) {
      std::string data = "data: " + chunk.dump() + "\n\n";
      post(ex, [data](proxygen::ResponseHandler* d) { ResponseBuilder(d).body(data).send(); });
    };
    auto chunk_of = [=](json delta, json finish) {
      json c = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                {"model", model}};
      json choice = {{"index", 0}};
      if (chat) choice["delta"] = std::move(delta);
      else choice["text"] = delta.is_string() ? delta : json("");
      choice["finish_reason"] = std::move(finish);
      c["choices"] = json::array({choice});
      return c;
    };
    auto deliver = [=](const OutputParser::Delta& d) {
      st->content += d.content;
      st->reasoning += d.reasoning;
      if (!req.stream) {
        st->calls.insert(st->calls.end(), d.tool_calls.begin(), d.tool_calls.end());
        return;
      }
      if (!chat) {
        if (!d.content.empty()) sse(chunk_of(json(d.content), nullptr));
        return;
      }
      json delta = json::object();
      if (st->first) {
        delta["role"] = "assistant";
        st->first = false;
      }
      if (!d.reasoning.empty()) delta["reasoning_content"] = d.reasoning;
      if (!d.content.empty()) delta["content"] = d.content;
      if (!d.tool_calls.empty()) {
        json calls = json::array();
        for (const auto& c : d.tool_calls) {
          calls.push_back({{"index", static_cast<int>(st->calls.size())},
                           {"id", new_id("call_")},
                           {"type", "function"},
                           {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
          st->calls.push_back(c);
        }
        delta["tool_calls"] = calls;
      }
      if (delta.size() > 0) sse(chunk_of(delta, nullptr));
    };

    job.sink.text = [=](const std::string& piece) { deliver(chat ? st->parser.push(piece) : OutputParser::Delta{{}, piece, {}}); };
    job.sink.error = [=](const std::string& msg) {
      post(ex, [msg, stream = req.stream](proxygen::ResponseHandler* d) {
        if (stream) {
          ResponseBuilder(d).body("data: " + error_body(msg, "server_error").dump() + "\n\n").sendWithEOM();
        } else {
          send_json(d, 500, error_body(msg, "server_error"));
        }
      });
    };
    job.sink.done = [=](const std::string& finish_in, int prompt_tokens, int completion_tokens, int cached) {
      if (chat) deliver(st->parser.finish());
      const std::string finish = !st->calls.empty() && finish_in == "stop" ? "tool_calls" : finish_in;
      json usage = {{"prompt_tokens", prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens", prompt_tokens + completion_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", cached}}}};
      if (req.stream) {
        sse(chunk_of(json::object(), finish));
        if (req.include_usage) {
          json u = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                    {"model", model}, {"choices", json::array()}, {"usage", usage}};
          sse(u);
        }
        post(ex, [](proxygen::ResponseHandler* d) { ResponseBuilder(d).body("data: [DONE]\n\n").sendWithEOM(); });
        return;
      }
      json choice = {{"index", 0}, {"finish_reason", finish}};
      if (chat) {
        json msg = {{"role", "assistant"}, {"content", st->content.empty() && !st->calls.empty() ? json() : json(st->content)}};
        if (!st->reasoning.empty()) msg["reasoning_content"] = st->reasoning;
        if (!st->calls.empty()) {
          json calls = json::array();
          for (const auto& c : st->calls)
            calls.push_back({{"id", new_id("call_")}, {"type", "function"},
                             {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
          msg["tool_calls"] = calls;
        }
        choice["message"] = msg;
      } else {
        choice["text"] = st->content;
      }
      json body = {{"id", id}, {"object", chat ? "chat.completion" : "text_completion"}, {"created", created},
                   {"model", model}, {"choices", json::array({choice})}, {"usage", usage}};
      post(ex, [body](proxygen::ResponseHandler* d) { send_json(d, 200, body); });
    };
    worker_.submit(std::move(job));
  }

  ServerContext& ctx_;
  Worker& worker_;
  std::string method_, path_;
  folly::IOBufQueue body_{folly::IOBufQueue::cacheChainLength()};
  bool too_large_ = false;
  std::shared_ptr<Exchange> ex_ = std::make_shared<Exchange>();
};

class Factory : public proxygen::RequestHandlerFactory {
 public:
  Factory(ServerContext& ctx, Worker& worker) : ctx_(ctx), worker_(worker) {}
  void onServerStart(folly::EventBase*) noexcept override {}
  void onServerStop() noexcept override {}
  proxygen::RequestHandler* onRequest(proxygen::RequestHandler*, HTTPMessage*) noexcept override {
    return new Handler(ctx_, worker_);
  }

 private:
  ServerContext& ctx_;
  Worker& worker_;
};

}  // namespace
}  // namespace ling::serve

int main(int argc, char** argv) {
  std::string model_dir, host = "0.0.0.0", name;
  int port = 8000, max_context = 65536;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model_dir = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--served-model-name") name = next();
    else if (a == "--max-context") max_context = std::stoi(next());
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::cerr << "usage: ling-serve --model DIR [--host H] [--port P] [--served-model-name N] [--max-context N]\n";
    return 2;
  }
  int fake_argc = 1;
  char** fake_argv = argv;
  folly::Init init(&fake_argc, &fake_argv, false);

  ling::Tokenizer tok(model_dir + "/tokenizer.json");
  ling::EngineOptions opts;
  opts.max_context = max_context;
  std::cerr << "loading " << model_dir << " ...\n";
  ling::Engine engine(model_dir, opts);
  std::cerr << "loaded " << engine.model().device_bytes() / 1e9 << " GB of weights\n";

  ling::serve::ServerContext ctx;
  ctx.engine = &engine;
  ctx.tok = &tok;
  ctx.im_end = tok.token_id("<|im_end|>");
  if (name.empty()) {
    name = model_dir;
    while (!name.empty() && name.back() == '/') name.pop_back();
    name = name.substr(name.find_last_of('/') + 1);
  }
  ctx.model_name = name;
  ling::serve::Worker worker(ctx);

  proxygen::HTTPServerOptions options;
  options.threads = 2;
  options.idleTimeout = std::chrono::minutes(30);
  options.shutdownOn = {SIGINT, SIGTERM};
  options.enableContentCompression = false;
  options.handlerFactories =
      proxygen::RequestHandlerChain().addThen<ling::serve::Factory>(ctx, worker).build();
  std::vector<proxygen::HTTPServer::IPConfig> ips = {
      {folly::SocketAddress(host, static_cast<uint16_t>(port), true), proxygen::HTTPServer::Protocol::HTTP}};
  proxygen::HTTPServer server(std::move(options));
  server.bind(ips);
  std::cerr << "ling-serve: serving " << name << " on " << host << ":" << port << "\n";
  std::thread t([&] { server.start(); });
  t.join();
  return 0;
}
