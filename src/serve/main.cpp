// ling-serve: the HTTP front of ling-engine (proxygen). One engine worker thread runs requests one at a
// time; the HTTP threads parse requests and stream server-sent events back.
//
//   ling-serve --model DIR [--host 0.0.0.0] [--port 8000] [--served-model-name NAME] [--max-context N]
//              [--draft DIR [--draft-block N] [--lookup 0|1|2] [--lookup-min N]] [--graphs] [--pdl]
//              [--attn-bulk] [--attn-prefetch N]
//
// --graphs runs each speculative step's draft, verify and commit as captured CUDA graphs (M3); --pdl launches
// the rows path's kernels with programmatic dependent launch (M3); --attn-bulk loads the rows path's attention
// tiles with bulk copies (M3); --attn-prefetch N sets how many KV tiles ahead its attention prefetches into L2
// (default 2, 0 off; M3).
//
// With --draft, requests decode speculatively with the DFlash2 drafter (requests that set penalties
// decode plainly). GET /metrics exports SGLang's counter names, so M0's replay harness reads them as is.
#include <folly/SocketAddress.h>
#include <folly/init/Init.h>
#include <folly/io/IOBufQueue.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/EventBaseManager.h>
#include <proxygen/httpserver/HTTPServer.h>
#include <proxygen/httpserver/RequestHandler.h>
#include <proxygen/httpserver/RequestHandlerFactory.h>
#include <proxygen/httpserver/ResponseBuilder.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
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

// Prometheus counters under SGLang's names (the ones bench/replay/replay.py reads).
struct Metrics {
  std::mutex mu;
  double prompt_tokens = 0, cached_tokens = 0, generation_tokens = 0, verify_calls = 0;
  double e2e_sum = 0, e2e_count = 0, ttft_sum = 0, ttft_count = 0;
  double accepted_drafts = 0, drafted = 0;
  double nonfinite = 0;  // requests ended because the logits were not finite
  SpecStats spec;  // the engine's speculation counters after the last request

  json spec_json() {
    std::lock_guard<std::mutex> l(mu);
    json j = {{"steps", spec.steps}, {"drafted", spec.drafted}, {"accepted", spec.accepted},
              {"accept_histogram", spec.accept_hist}, {"draft_seconds", spec.draft_seconds},
              {"verify_seconds", spec.verify_seconds}, {"commit_seconds", spec.commit_seconds},
              {"lookup_steps", spec.lookup_steps}, {"lookup_accepted", spec.lookup_accepted},
              {"shadow_total_steps", spec.shadow_total_steps}, {"shadow_dflash_all", spec.shadow_dflash_all},
              {"shadow_best", spec.shadow_best}};
    for (int b = 0; b < 5; ++b) {
      j["shadow_steps"].push_back(spec.shadow_steps[b]);
      j["shadow_lookup"].push_back(spec.shadow_lookup[b]);
      j["shadow_dflash"].push_back(spec.shadow_dflash[b]);
    }
    return j;
  }

  std::string render(const std::string& model) {
    std::lock_guard<std::mutex> l(mu);
    const std::string lab = "{model_name=\"" + model + "\"}";
    std::string out;
    auto counter = [&](const char* name, double v, const char* help) {
      out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " counter\n";
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.6f", v);
      out += std::string(name) + lab + " " + buf + "\n";
    };
    counter("sglang:prompt_tokens_total", prompt_tokens, "Prompt tokens.");
    counter("sglang:cached_tokens_total", cached_tokens, "Prompt tokens served from the cache.");
    counter("sglang:generation_tokens_total", generation_tokens, "Generated tokens.");
    counter("sglang:spec_verify_calls_total", verify_calls, "Decode passes (speculative verify steps, or plain steps).");
    counter("sglang:e2e_request_latency_seconds_sum", e2e_sum, "Request latency, sum.");
    counter("sglang:e2e_request_latency_seconds_count", e2e_count, "Requests.");
    counter("sglang:time_to_first_token_seconds_sum", ttft_sum, "Time to first token, sum.");
    counter("sglang:time_to_first_token_seconds_count", ttft_count, "Time to first token, count.");
    counter("ling:spec_drafted_tokens_total", drafted, "Drafted tokens offered to the verify.");
    counter("ling:spec_accepted_tokens_total", accepted_drafts, "Drafted tokens accepted.");
    counter("ling:nonfinite_logits_total", nonfinite, "Requests ended because the logits were NaN or inf.");
    return out;
  }
};

struct ServerContext {
  Engine* engine = nullptr;
  Tokenizer* tok = nullptr;
  std::string model_name;
  int im_end = -1;
  Metrics metrics;
};

// What a job reports back, called on the engine thread.
struct JobSink {
  std::function<void(const std::string&)> text;
  std::function<void(const std::string& finish, int prompt_tokens, int completion_tokens, int cached_tokens)> done;
  // The message, and the code the client sees (server_error, or context_length_exceeded).
  std::function<void(const std::string& message, const std::string& code)> error;
};

struct Job {
  std::chrono::steady_clock::time_point received = std::chrono::steady_clock::now();
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
      } catch (const ContextOverflow& e) {
        job.sink.error(e.what(), kContextLengthExceeded);
      } catch (const NonFiniteLogits& e) {
        {
          std::lock_guard<std::mutex> l(ctx_.metrics.mu);
          ctx_.metrics.nonfinite += 1;
        }
        std::fprintf(stderr, "request failed: %s (%zu prompt tokens)\n", e.what(), job.prompt.size());
        job.sink.error(e.what(), "server_error");
      } catch (const std::exception& e) {
        job.sink.error(e.what(), "server_error");
      }
    }
  }

  void generate(Job& job) {
    Engine& eng = *ctx_.engine;
    if (exceeds_context(job.prompt.size(), eng.max_context())) throw ContextOverflow(job.prompt.size(), eng.max_context());
    const int ctx_left = eng.max_context() - static_cast<int>(job.prompt.size());
    const int limit = job.req.max_tokens > 0 ? std::min(job.req.max_tokens, ctx_left) : ctx_left;
    EngineStats stats;
    std::vector<float> logits = eng.prefill(job.prompt, &stats);
    const bool spec = eng.can_speculate(job.req.sampling);
    const long steps0 = eng.spec_stats().steps, drafted0 = eng.spec_stats().drafted, accepted0 = eng.spec_stats().accepted;
    const auto t_decode = std::chrono::steady_clock::now();
    StreamDecoder dec(*ctx_.tok);
    StopScanner stops(job.req.stop);
    std::string finish = "length";
    int produced = 0, passes = 0;
    bool ended_by_eos = false;
    double ttft = -1;
    // `pending` holds chosen tokens not yet emitted: the first from the prefill's logits, then a
    // speculative step's accepted drafts and its next token, or a plain step's sample.
    std::vector<int> pending = {eng.sample(logits, job.req.sampling, eng.history())};
    bool done = false;
    while (!done) {
      for (int t : pending) {
        if (ttft < 0) ttft = std::chrono::duration<double>(std::chrono::steady_clock::now() - job.received).count();
        if (job.cancelled->load()) {
          finish = "cancelled";
          done = true;
          break;
        }
        if (t == eng.config().eos_token || t == ctx_.im_end) {
          finish = "stop";
          ended_by_eos = true;
          done = true;
          break;
        }
        ++produced;
        const std::string ready = stops.push(dec.push(t));
        if (!ready.empty()) job.sink.text(ready);
        if (stops.stopped()) {
          finish = "stop";
          done = true;
          break;
        }
        if (produced >= limit) {
          done = true;
          break;
        }
      }
      if (done) break;
      ++passes;
      if (spec) {
        pending = eng.speculate(pending.back(), job.req.sampling);
      } else {
        logits = eng.step(pending.back());
        pending = {eng.sample(logits, job.req.sampling, eng.history())};
      }
    }
    if (!stops.stopped()) {  // what the decoder and the stop scanner still hold back
      std::string rest = stops.push(dec.flush());
      if (!stops.stopped()) rest += stops.finish();
      if (!rest.empty()) job.sink.text(rest);
    }
    const auto now = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(now - t_decode).count();
    const long steps = eng.spec_stats().steps - steps0;
    {
      Metrics& m = ctx_.metrics;
      std::lock_guard<std::mutex> l(m.mu);
      m.prompt_tokens += job.prompt.size();
      m.cached_tokens += stats.reused_tokens;
      m.generation_tokens += produced + (ended_by_eos ? 1 : 0);  // SGLang counts the stop token
      m.verify_calls += passes;
      m.e2e_sum += std::chrono::duration<double>(now - job.received).count();
      m.e2e_count += 1;
      if (ttft >= 0) {
        m.ttft_sum += ttft;
        m.ttft_count += 1;
      }
      m.drafted += eng.spec_stats().drafted - drafted0;
      m.accepted_drafts += eng.spec_stats().accepted - accepted0;
      m.spec = eng.spec_stats();
    }
    std::fprintf(stderr,
                 "request: %zu prompt tokens (%d reused), prefill %.2f s; %d tokens in %.2f s (%.1f tok/s), %s; "
                 "%d passes%s\n",
                 job.prompt.size(), stats.reused_tokens, stats.prefill_seconds, produced, decode_s,
                 decode_s > 0 ? produced / decode_s : 0.0, finish.c_str(), passes,
                 spec && steps > 0 ? (", " + std::to_string(double(produced) / std::max(1, passes)).substr(0, 4) +
                                      " tokens/pass").c_str()
                                   : "");
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
      .body(dump_json(body))
      .sendWithEOM();
}

json error_body(const std::string& msg, const std::string& type, const std::string& code = "") {
  json e = {{"message", msg}, {"type", type}};
  if (!code.empty()) e["code"] = code;
  return json{{"error", e}};
}

// The type that goes with an error code.
std::string error_type(const std::string& code) {
  return code == kContextLengthExceeded ? "invalid_request_error" : "server_error";
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
    } else if (method_ == "GET" && path_ == "/metrics") {
      ResponseBuilder(downstream_)
          .status(200, "OK")
          .header("Content-Type", "text/plain; version=0.0.4")
          .body(ctx_.metrics.render(ctx_.model_name))
          .sendWithEOM();
    } else if (method_ == "GET" && path_ == "/spec_stats") {
      send_json(downstream_, 200, ctx_.metrics.spec_json());
    } else if (method_ == "POST" && path_ == "/v1/responses") {
      generate_responses();
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
    if (exceeds_context(job.prompt.size(), ctx_.engine->max_context())) {  // refused before any header
      send_json(downstream_, 400, context_overflow_error(job.prompt.size(), ctx_.engine->max_context()));
      return;
    }
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
      std::string data = "data: " + dump_json(chunk) + "\n\n";
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
    job.sink.error = [=](const std::string& msg, const std::string& code) {
      const json error = error_body(msg, error_type(code), code);
      if (req.stream) {  // the headers are out: end the stream properly after the error
        const std::string tail = stream_error_tail(error, chunk_of(json::object(), kStreamErrorFinish));
        post(ex, [tail](proxygen::ResponseHandler* d) { ResponseBuilder(d).body(tail).sendWithEOM(); });
      } else {
        const int status = code == kContextLengthExceeded ? 400 : 500;
        post(ex, [error, status](proxygen::ResponseHandler* d) { send_json(d, status, error); });
      }
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

  // The Responses API, streamed as the event sequence Mightling's agent parses: response.created, then
  // per output item output_item.added, its text deltas and output_item.done, then response.completed.
  void generate_responses() {
    if (too_large_) {
      send_json(downstream_, 413, error_body("request body too large", "invalid_request_error"));
      return;
    }
    Job job;
    try {
      std::string text;
      if (auto buf = body_.move()) text = buf->moveToFbString().toStdString();
      job.req = parse_responses_request(json::parse(text));
    } catch (const std::exception& e) {
      send_json(downstream_, 400, error_body(e.what(), "invalid_request_error"));
      return;
    }
    job.prompt = ctx_.tok->encode(job.req.prompt_text);
    job.cancelled = ex_->cancelled;
    const Request req = job.req;
    auto ex = ex_;
    const std::string model = ctx_.model_name;
    const long created = std::time(nullptr);

    struct State {
      OutputParser parser;
      std::string id = new_id("resp_");
      int seq = 0;
      json output = json::array();   // finished items, in order
      std::string reasoning, content;
      std::string reasoning_id, message_id;
      int reasoning_index = -1, message_index = -1;
      bool reasoning_open = false, message_open = false;
      std::vector<std::string> custom_tools;
      State(bool r, const json& t, std::vector<std::string> c) : parser(r, t), custom_tools(std::move(c)) {}
      bool custom(const std::string& name) const {
        return std::find(custom_tools.begin(), custom_tools.end(), name) != custom_tools.end();
      }
    };
    auto st = std::make_shared<State>(req.reasoning, req.tools, req.custom_tools);
    auto response_obj = [=](const std::string& status) {
      return json{{"id", st->id}, {"object", "response"}, {"created_at", created}, {"status", status},
                  {"model", model}, {"output", st->output}, {"tools", json::array()}};
    };
    auto event = [=](const std::string& type, json payload) {
      payload["type"] = type;
      payload["sequence_number"] = st->seq++;
      if (!req.stream) return;
      std::string data = "event: " + type + "\ndata: " + dump_json(payload) + "\n\n";
      post(ex, [data](proxygen::ResponseHandler* d) { ResponseBuilder(d).body(data).send(); });
    };
    auto close_reasoning = [=]() {
      if (!st->reasoning_open) return;
      st->reasoning_open = false;
      json item = {{"type", "reasoning"}, {"id", st->reasoning_id}, {"summary", json::array()},
                   {"content", json::array({json{{"type", "reasoning_text"}, {"text", st->reasoning}}})}};
      st->output.push_back(item);
      event("response.output_item.done", {{"output_index", st->reasoning_index}, {"item", item}});
    };
    auto close_message = [=]() {
      if (!st->message_open) return;
      st->message_open = false;
      json item = {{"type", "message"}, {"id", st->message_id}, {"role", "assistant"}, {"status", "completed"},
                   {"content", json::array({json{{"type", "output_text"}, {"text", st->content}, {"annotations", json::array()}}})}};
      st->output.push_back(item);
      event("response.output_item.done", {{"output_index", st->message_index}, {"item", item}});
    };
    auto deliver = [=](const OutputParser::Delta& d) {
      if (!d.reasoning.empty()) {
        if (!st->reasoning_open && st->reasoning.empty()) {
          st->reasoning_open = true;
          st->reasoning_id = new_id("rs_");
          st->reasoning_index = static_cast<int>(st->output.size());
          event("response.output_item.added",
                {{"output_index", st->reasoning_index},
                 {"item", {{"type", "reasoning"}, {"id", st->reasoning_id}, {"summary", json::array()}, {"content", json::array()}}}});
        }
        st->reasoning += d.reasoning;
        event("response.reasoning_text.delta",
              {{"item_id", st->reasoning_id}, {"output_index", st->reasoning_index}, {"content_index", 0}, {"delta", d.reasoning}});
      }
      if (!d.content.empty()) {
        close_reasoning();
        if (!st->message_open && st->content.empty()) {
          st->message_open = true;
          st->message_id = new_id("msg_");
          st->message_index = static_cast<int>(st->output.size());
          event("response.output_item.added",
                {{"output_index", st->message_index},
                 {"item", {{"type", "message"}, {"id", st->message_id}, {"role", "assistant"}, {"status", "in_progress"}, {"content", json::array()}}}});
        }
        st->content += d.content;
        event("response.output_text.delta",
              {{"item_id", st->message_id}, {"output_index", st->message_index}, {"content_index", 0}, {"delta", d.content}});
      }
      for (const ToolCall& c : d.tool_calls) {
        close_reasoning();
        close_message();
        const int index = static_cast<int>(st->output.size());
        if (st->custom(c.name)) {
          // A custom (freeform) tool call: the item is added with its `input` present and empty (Codex
          // parses the item only with that field), the whole input is one custom_tool_call_input.delta
          // keyed by item_id and call_id, then the done events.
          const std::string input = custom_tool_input(c.arguments);
          json item = {{"type", "custom_tool_call"}, {"id", new_id("ctc_")}, {"call_id", new_id("call_")},
                       {"name", c.name}, {"input", ""}, {"status", "in_progress"}};
          event("response.output_item.added", {{"output_index", index}, {"item", item}});
          event("response.custom_tool_call_input.delta",
                {{"item_id", item["id"]}, {"output_index", index}, {"call_id", item["call_id"]}, {"delta", input}});
          event("response.custom_tool_call_input.done",
                {{"item_id", item["id"]}, {"output_index", index}, {"call_id", item["call_id"]}, {"input", input}});
          item["input"] = input;
          item["status"] = "completed";
          st->output.push_back(item);
          event("response.output_item.done", {{"output_index", index}, {"item", item}});
          continue;
        }
        json item = {{"type", "function_call"}, {"id", new_id("fc_")}, {"call_id", new_id("call_")},
                     {"name", c.name}, {"arguments", c.arguments}, {"status", "completed"}};
        event("response.output_item.added", {{"output_index", index}, {"item", item}});
        st->output.push_back(item);
        event("response.output_item.done", {{"output_index", index}, {"item", item}});
      }
    };

    if (req.stream) {
      ResponseBuilder(downstream_)
          .status(200, "OK")
          .header("Content-Type", "text/event-stream")
          .header("Cache-Control", "no-cache")
          .send();
    }
    event("response.created", {{"response", response_obj("in_progress")}});

    // Streamed, an error is response.failed carrying its code (context_length_exceeded is what makes
    // Codex compact; server_error is retried); not streamed, an HTTP error with the same code.
    auto fail = [=](const std::string& msg, const std::string& code) {
      if (req.stream) {
        json failed = response_obj("failed");
        failed["error"] = {{"code", code}, {"message", msg}};
        event("response.failed", {{"response", failed}});
        post(ex, [](proxygen::ResponseHandler* d) { ResponseBuilder(d).sendWithEOM(); });
      } else {
        const json error = error_body(msg, error_type(code), code);
        const int status = code == kContextLengthExceeded ? 400 : 500;
        post(ex, [error, status](proxygen::ResponseHandler* d) { send_json(d, status, error); });
      }
    };
    if (exceeds_context(job.prompt.size(), ctx_.engine->max_context())) {
      fail(context_overflow_message(job.prompt.size(), ctx_.engine->max_context()), kContextLengthExceeded);
      return;
    }

    job.sink.text = [=](const std::string& piece) { deliver(st->parser.push(piece)); };
    job.sink.error = fail;
    job.sink.done = [=](const std::string& finish, int prompt_tokens, int completion_tokens, int cached) {
      deliver(st->parser.finish());
      close_reasoning();
      close_message();
      json resp = response_obj(finish == "length" ? "incomplete" : "completed");
      if (finish == "length") resp["incomplete_details"] = {{"reason", "max_output_tokens"}};
      resp["usage"] = {{"input_tokens", prompt_tokens},
                       {"input_tokens_details", {{"cached_tokens", cached}}},
                       {"output_tokens", completion_tokens},
                       {"output_tokens_details", {{"reasoning_tokens", 0}}},
                       {"total_tokens", prompt_tokens + completion_tokens}};
      if (req.stream) {
        // A response cut at max_output_tokens still ends with response.completed: the agent treats
        // response.incomplete as a failed turn and discards what was written.
        event("response.completed", {{"response", resp}});
        post(ex, [](proxygen::ResponseHandler* d) { ResponseBuilder(d).sendWithEOM(); });
      } else {
        post(ex, [resp](proxygen::ResponseHandler* d) { send_json(d, 200, resp); });
      }
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
  int port = 8000, max_context = 65536, draft_block = 12, lookup = 1, lookup_min = 8, checkpoints = 8;
  std::string draft_dir, pretokenizer = "production";
  bool graphs = false, pdl = false, attn_bulk = false;
  int attn_prefetch = 2;
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
    else if (a == "--draft") draft_dir = next();
    else if (a == "--draft-block") draft_block = std::stoi(next());
    else if (a == "--lookup") lookup = std::stoi(next());
    else if (a == "--lookup-min") lookup_min = std::stoi(next());
    else if (a == "--pretokenizer") pretokenizer = next();
    else if (a == "--prefix-checkpoints") checkpoints = std::stoi(next());
    else if (a == "--graphs") graphs = true;
    else if (a == "--pdl") pdl = true;
    else if (a == "--attn-bulk") attn_bulk = true;
    else if (a == "--attn-prefetch") attn_prefetch = std::stoi(next());
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::cerr << "usage: ling-serve --model DIR [--host H] [--port P] [--served-model-name N] [--max-context N] [--draft DIR [--draft-block N]] [--pretokenizer production|checkpoint] [--prefix-checkpoints N]\n";
    return 2;
  }
  int fake_argc = 1;
  char** fake_argv = argv;
  folly::Init init(&fake_argc, &fake_argv, false);

  // Prompts are tokenized as production tokenizes them (tokenizer.hpp, kProductionPretokenizer) unless
  // --pretokenizer checkpoint asks for the checkpoint's own pattern.
  if (pretokenizer != "production" && pretokenizer != "checkpoint") {
    std::cerr << "--pretokenizer is production or checkpoint\n";
    return 2;
  }
  ling::Tokenizer tok(model_dir + "/tokenizer.json", pretokenizer == "production" ? ling::kProductionPretokenizer : "");
  ling::EngineOptions opts;
  opts.max_context = max_context;
  opts.draft_dir = draft_dir;
  opts.draft_block = draft_block;
  opts.lookup_mode = lookup;
  opts.lookup_min_match = lookup_min;
  opts.prefix_checkpoints = checkpoints;
  opts.boundary_token = tok.token_id("<|im_end|>");
  opts.step_graphs = graphs;
  opts.pdl = pdl;
  opts.attention_bulk = attn_bulk;
  opts.attention_prefetch = attn_prefetch;
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
