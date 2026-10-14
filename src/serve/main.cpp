// ling-serve: the HTTP front of ling-engine (proxygen). One engine thread (the scheduler) runs up to --batch
// requests at once, each in its own resident session, taking turns one prefill chunk or decode step at a
// time; the HTTP threads parse requests and stream server-sent events back.
//
//   ling-serve --model DIR [--host 0.0.0.0] [--port 8000] [--served-model-name NAME] [--max-context N]
//              [--draft DIR [--draft-block N] [--lookup 0|1|2] [--lookup-min N]] [--graphs] [--pdl]
//              [--attn-bulk] [--attn-prefetch N] [--keepalive-seconds N] [--prefill-bf16]
//              [--sessions N] [--session-protect-seconds S] [--batch N] [--prefill-chunk-contended N]
//
// --sessions N (default 2; M4 §1) keeps N resident sessions, each with its own KV cache and state, so N
// agent tasks that alternate keep their histories warm. A request runs in the session it extends (its
// prompt_cache_key's session is looked at first; the token prefix decides), else in a new session, which
// takes a free slot or evicts the least recently used one. A session bound with a prompt_cache_key whose
// last request ran less than --session-protect-seconds ago (default 30, 0 off; the owner's decision of 2026-10-11) is never evicted: a new
// session's request waits in the queue instead, while requests that continue a resident session run.
//
// --batch N (default 2, at most --sessions; M4 §3, stage 2) requests run at once, each in its own session,
// interleaved: the running requests take turns, one prefill chunk or one decode step each. While another
// request decodes, a prefill runs in chunks of --prefill-chunk-contended N tokens (default 256, a multiple
// of 32 up to the prefill chunk of 2,048), so the decoding request's step period stays near twice its
// normal step during a long prefill. A request that continues a running request's session waits for it.
// Nothing numeric changes: every unit runs in its own session's state exactly as it would alone.
//
// --graphs runs each speculative step's draft, verify and commit as captured CUDA graphs (M3); --pdl launches
// the rows path's kernels with programmatic dependent launch (M3); --attn-bulk loads the rows path's attention
// tiles with bulk copies (M3); --attn-prefetch N sets how many KV tiles ahead its attention prefetches into L2
// (default 2, 0 off; M3). --keepalive-seconds N (default 15, 0 off; LING_KEEPALIVE_SECONDS overrides it) is
// how often a streamed request that has produced no token yet is sent a keep-alive (API §3, §4).
// --prefill-bf16 keeps the prefill's DeltaNet intermediates in BF16 as production does (B34; not exact, off
// by default: MILESTONES §14).
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
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
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
  // Gauges the benchmark harness reads (its admission and its parallelism): requests running (up to
  // --batch at once, M4 §3) and queued, and the token pool a night's tasks are budgeted
  // against: the resident sessions times the context size (M4 §1, SESSIONS §11).
  int running = 0, queued = 0, max_total_tokens = 0;
  // The scheduler (M4 §3): requests that run at once, the prefill chunk size beside a decode, and the
  // prefill chunks run between another request's steps so far.
  int batch = 0, prefill_chunk_contended = 0;
  double prefill_chunks_interleaved = 0;
  // Stage 3: the rows of the last decode pass (a batched verify's sum over its sequences, or one sequence's), and
  // the batched steps so far (one per batch, whatever its size).
  int batch_rows = 0;
  double batched_steps = 0;
  // The session pool (M4 §1): slots, slots holding a session, evictions so far, the protection window.
  int sessions = 0, sessions_resident = 0;
  double session_evictions = 0, protect_seconds = 0;

  json spec_json() {
    std::lock_guard<std::mutex> l(mu);
    json j = {{"steps", spec.steps}, {"drafted", spec.drafted}, {"accepted", spec.accepted},
              {"accept_histogram", spec.accept_hist}, {"draft_seconds", spec.draft_seconds},
              {"verify_seconds", spec.verify_seconds}, {"commit_seconds", spec.commit_seconds},
              {"lookup_steps", spec.lookup_steps}, {"lookup_accepted", spec.lookup_accepted},
              {"batched_steps", spec.batched_steps},
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
    counter("ling:session_evictions_total", session_evictions, "Resident sessions replaced by a new session.");
    counter("ling:prefill_chunks_interleaved_total", prefill_chunks_interleaved,
            "Prefill chunks run between another request's decode steps.");
    counter("ling:batched_steps_total", batched_steps,
            "Decode steps that verified two or more requests' rows in one pass.");
    auto gauge = [&](const char* name, double v, const char* help) {
      out += std::string("# HELP ") + name + " " + help + "\n# TYPE " + name + " gauge\n";
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.6f", v);
      out += std::string(name) + lab + " " + buf + "\n";
    };
    gauge("sglang:num_running_reqs", running, "Requests being prefilled or generated (at most --batch).");
    gauge("sglang:num_queue_reqs", queued, "Requests waiting for the engine.");
    gauge("sglang:max_total_num_tokens", max_total_tokens,
          "Tokens the resident sessions can hold (sessions times the context size).");
    gauge("ling:sessions", sessions, "Resident session slots.");
    gauge("ling:sessions_resident", sessions_resident, "Slots holding a session.");
    gauge("ling:batch", batch, "Requests that run at once (each in its own session).");
    gauge("ling:prefill_chunk_contended", prefill_chunk_contended,
          "Prefill chunk size, in tokens, while another request decodes.");
    gauge("ling:batch_rows", batch_rows, "Rows of the last decode pass (all its requests' rows when batched).");
    gauge("ling:session_protect_seconds", protect_seconds,
          "How long a keyed session is protected from eviction after its last request.");
    return out;
  }
};

struct ServerContext {
  Engine* engine = nullptr;
  Tokenizer* tok = nullptr;
  std::string model_name;
  int im_end = -1;
  // The token ids that end the thinking span: `</think>`, then the blank line the template puts after it
  // ("\n\n" as one token, or "\n" twice). -1 when the tokenizer has no such single token.
  int think_end = -1, newline = -1, blank_line = -1;
  // How often a streamed request that has produced no token yet is sent a keep-alive (0: never).
  int keepalive_ms = 15000;
  Metrics metrics;
};

// What a job reports back, called on the engine thread.
struct JobSink {
  std::function<void(const std::string&)> text;
  // reasoning_tokens: the thinking span's share of completion_tokens, counted from the token ids as they
  // are produced (up to and including `</think>` and the blank line after it; every token when the answer
  // was cut inside the span; 0 with thinking off), so an accepted draft block that crosses `</think>` is
  // split exactly and the count never exceeds completion_tokens (vllm#49711, sglang#39826).
  std::function<void(const std::string& finish, int prompt_tokens, int completion_tokens, int cached_tokens,
                     int reasoning_tokens)>
      done;
  // The message, and the code the client sees (server_error, or context_length_exceeded).
  std::function<void(const std::string& message, const std::string& code)> error;
  // Per produced token, only when the request asked for logprobs: the token's own logprob first, then the
  // top k (top_logprobs(), api.hpp). Called before the token's text, which the decoder or the parser may
  // still hold back.
  std::function<void(const std::vector<TokenLogprob>& entries)> logprobs;
};

struct Job {
  std::chrono::steady_clock::time_point received = std::chrono::steady_clock::now();
  Request req;
  std::vector<int> prompt;
  JobSink sink;
  std::shared_ptr<std::atomic<bool>> cancelled;
  int session = -1;    // the session it runs in (bound when the scheduler admits it)
  bool fresh = false;  // a new session in that slot
};

// Step periods of one decoding request: the time from the start of one of its steps (or its first token) to
// the start of the next, split by whether another request was prefilling meanwhile (M4 §3: a prefill chunk
// runs between two of its steps).
struct Periods {
  double sum = 0, max = 0;
  int n = 0;
  void add(double s) {
    sum += s;
    max = std::max(max, s);
    ++n;
  }
  double mean_ms() const { return n ? 1e3 * sum / n : 0.0; }
};

// A request the scheduler is running (M4 §3): its prefill cursor, then what one decode step needs (what
// generate() kept in locals when one request ran at a time).
struct Running {
  Job job;
  Engine::PrefillCursor cur;
  bool prefilling = true;
  int limit = 0;
  bool spec = false;
  std::vector<float> logits;  // the prefill's logits (the first token), then a plain step's
  std::vector<float> rows;    // the verify's rows, when the request asked for logprobs
  std::vector<int> pending;   // chosen tokens not yet emitted
  std::optional<StreamDecoder> dec;
  std::optional<StopScanner> stops;
  std::string finish = "length";
  int produced = 0, passes = 0, reasoning = 0;
  bool in_think = false;
  int newlines_after_think = 0;  // of the blank line after </think>, still to count
  bool ended_by_eos = false;
  double ttft = -1;
  std::chrono::steady_clock::time_point t_decode, last_step;
  long steps0 = 0, drafted0 = 0, accepted0 = 0;
  int chunks_beside = 0;  // prefill chunks run while another request was decoding (contended size)
  int batched = 0;        // its decode steps taken in a batch with another request (M4 stage 3)
  Periods alone, beside;  // step periods without and with another request's prefill in between
};

// The engine thread (M4 §3). Up to `batch` requests run at once, each in its own resident session: a request
// is admitted when it can be bound (its session is not another running request's, or a slot is free or
// evictable), then the running requests take turns, one unit each: a prefill chunk while its prompt is not
// done, else one decode step (speculative, or plain). A prefill chunk that runs while another request is
// decoding is at most `contended` tokens, which bounds the decoding request's step period during a long
// prefill. Nothing numeric changes: each unit runs in its own session's state, as it would alone.
// Stage 3: when the request whose turn it is decodes speculatively and another one does too, they take one
// batched step together (Engine::speculate_batch: one verify pass over both requests' rows), which is bit for bit
// each one's step alone. A request that cannot join (plain decoding, no room for a full draft block) takes its own
// one-row step on its turn; a prefill chunk still alternates with the decode steps.
class Scheduler {
 public:
  Scheduler(ServerContext& ctx, double protect_seconds, int batch, int contended)
      : ctx_(ctx), protect_seconds_(protect_seconds), batch_(batch), contended_(contended), thread_([this] { run(); }) {}
  ~Scheduler() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
  void submit(Job job) {
    std::lock_guard<std::mutex> l(mu_);
    queue_.push_back(std::move(job));
    dirty_ = true;
    {  // under mu_ as well, so the scheduler's decrement cannot precede this increment
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.queued += 1;
    }
    cv_.notify_one();
  }

 private:
  // Under mu_. Drops the jobs whose client has gone, then takes the first job that can run now and binds
  // its session (M4 §1): the resident session it continues, else a slot for a new session. Sessions that a
  // running request holds or a queued job continues are pinned, so a new session never takes them; a job
  // that continues a running request's session waits for it (one request per session at a time), and when
  // every other slot is pinned or protected a new session's job waits while the jobs behind it still run.
  bool pick(Job& out) {
    Engine& eng = *ctx_.engine;
    for (auto it = queue_.begin(); it != queue_.end();) {
      if (!it->cancelled->load()) {
        ++it;
        continue;
      }
      it = queue_.erase(it);
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);  // mu_ then metrics.mu, as submit() takes them
      ctx_.metrics.queued -= 1;
    }
    std::vector<bool> busy(eng.sessions(), false);
    for (const auto& r : running_) busy[r->job.session] = true;
    std::vector<int> match(queue_.size());
    std::vector<bool> pinned = busy;
    for (size_t j = 0; j < queue_.size(); ++j) {
      match[j] = eng.match_session(queue_[j].prompt, queue_[j].req.session_key);
      if (match[j] >= 0) pinned[match[j]] = true;
    }
    for (size_t j = 0; j < queue_.size(); ++j) {
      int s = match[j];
      if (s >= 0 && busy[s]) continue;
      const bool fresh = s < 0;
      if (fresh) s = eng.free_session(protect_seconds_, pinned);
      if (s < 0) continue;
      out = std::move(queue_[j]);
      out.session = s;
      out.fresh = fresh;
      queue_.erase(queue_.begin() + j);
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.queued -= 1;
      return true;
    }
    return false;
  }

  // Waits while nothing runs and nothing can be admitted; admits at most one job. Returns false to stop.
  bool admit() {
    Job job;
    {
      std::unique_lock<std::mutex> l(mu_);
      while (true) {
        if (stop_) return false;
        const auto now = std::chrono::steady_clock::now();
        // A job that could not be placed is looked at again when something changed (a new job, a request
        // ended) or once a second (a protection expires, a waiting client leaves).
        const bool look = static_cast<int>(running_.size()) < batch_ && !queue_.empty() &&
                          (dirty_ || now - last_look_ >= std::chrono::seconds(1));
        if (look) {
          dirty_ = false;
          last_look_ = now;
          if (pick(job)) break;
        }
        if (!running_.empty()) return true;
        if (queue_.empty()) cv_.wait(l);
        else cv_.wait_for(l, std::chrono::seconds(1));
      }
    }
    start(std::move(job));
    return true;
  }

  // Binds the admitted job's session and readies its prefill; a job refused here is answered at once.
  void start(Job job) {
    Engine& eng = *ctx_.engine;
    auto r = std::make_unique<Running>();
    r->job = std::move(job);
    Job& j = r->job;
    try {
      if (exceeds_context(j.prompt.size(), eng.max_context())) throw ContextOverflow(j.prompt.size(), eng.max_context());
      eng.activate_session(j.session, j.req.session_key, j.fresh);
      r->cur = eng.begin_prefill(j.prompt);
    } catch (const std::exception& e) {
      fail(*r, e);
      eng.touch_session(j.session);
      return;
    }
    running_.push_back(std::move(r));
    std::lock_guard<std::mutex> m(ctx_.metrics.mu);
    ctx_.metrics.running = static_cast<int>(running_.size());
    ctx_.metrics.sessions_resident = eng.sessions_resident();
    ctx_.metrics.session_evictions = static_cast<double>(eng.session_evictions());
  }

  void fail(Running& r, const std::exception& e) {
    if (const auto* o = dynamic_cast<const ContextOverflow*>(&e)) {
      r.job.sink.error(o->what(), kContextLengthExceeded);
      return;
    }
    if (dynamic_cast<const NonFiniteLogits*>(&e)) {
      {
        std::lock_guard<std::mutex> l(ctx_.metrics.mu);
        ctx_.metrics.nonfinite += 1;
      }
      std::fprintf(stderr, "request failed: %s (%zu prompt tokens)\n", e.what(), r.job.prompt.size());
    }
    r.job.sink.error(e.what(), "server_error");
  }

  void run() {
    while (admit()) {
      if (running_.empty()) continue;
      const size_t k = turn_++ % running_.size();
      Running& r = *running_[k];
      std::vector<Running*> ended;
      std::vector<Running*> batch = batch_with(r);
      if (batch.size() >= 2) {
        batch_step(batch, ended);
      } else {
        bool done = false;
        try {
          done = unit(r);
        } catch (const std::exception& e) {
          fail(r, e);
          done = true;
        }
        if (done) ended.push_back(&r);
      }
      if (ended.empty()) continue;
      for (Running* e : ended) {
        ctx_.engine->touch_session(e->job.session);  // the session's clock restarts however the request ended
        running_.erase(std::find_if(running_.begin(), running_.end(), [&](const auto& p) { return p.get() == e; }));
      }
      {
        std::lock_guard<std::mutex> l(mu_);
        dirty_ = true;  // its session may be what a queued job waits for
      }
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.running = static_cast<int>(running_.size());
    }
  }

  // Whether r's next decode step can join a batch: past its prompt, speculative, room for a full draft block.
  bool batchable(const Running& r) const {
    return !r.prefilling && r.spec && ctx_.engine->can_batch(r.job.session, r.job.req.sampling);
  }
  // The requests that step with r (r first, then the others in running order) while their rows fit one pass;
  // just r, or nothing, when there is no batch to form.
  std::vector<Running*> batch_with(Running& r) const {
    std::vector<Running*> b;
    if (!batchable(r)) return b;
    b.push_back(&r);
    const int fit = kernels::kMaxStreamRows / ctx_.engine->draft_block();
    for (const auto& o : running_)
      if (o.get() != &r && static_cast<int>(b.size()) < fit && batchable(*o)) b.push_back(o.get());
    return b;
  }

  // One batched speculative step of the requests in `batch`, then each one's tokens emitted as unit() does;
  // the requests that end are added to `ended`. A non-finite row fails only its own request.
  void batch_step(const std::vector<Running*>& batch, std::vector<Running*>& ended) {
    Engine& eng = *ctx_.engine;
    std::vector<Engine::SeqStep> steps(batch.size());
    const auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < batch.size(); ++i) {
      Running& r = *batch[i];
      (other_prefilling(r) ? r.beside : r.alone).add(std::chrono::duration<double>(now - r.last_step).count());
      r.last_step = now;
      ++r.passes;
      ++r.batched;
      steps[i].session = r.job.session;
      steps[i].anchor = r.pending.back();
      steps[i].params = &r.job.req.sampling;
      steps[i].row_logits = r.job.req.logprobs ? &r.rows : nullptr;
    }
    try {
      eng.speculate_batch(steps);
    } catch (const std::exception& e) {  // not one sequence's rows: the step itself failed
      for (Running* r : batch) {
        fail(*r, e);
        ended.push_back(r);
      }
      return;
    }
    {
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.batched_steps += 1;
      ctx_.metrics.batch_rows = static_cast<int>(batch.size()) * eng.draft_block();  // a full block each
    }
    for (size_t i = 0; i < batch.size(); ++i) {
      Running& r = *batch[i];
      bool done = false;
      try {
        if (steps[i].error) std::rethrow_exception(steps[i].error);
        r.pending = std::move(steps[i].out);
        if (r.job.req.logprobs) require_finite(r.rows);  // as unit(): a row reported as logprobs is checked whole
        done = emit(r) && end(r);
      } catch (const std::exception& e) {
        fail(r, e);
        done = true;
      }
      if (done) ended.push_back(&r);
    }
  }

  bool other_decoding(const Running& self) const {
    for (const auto& r : running_)
      if (r.get() != &self && !r->prefilling) return true;
    return false;
  }
  bool other_prefilling(const Running& self) const {
    for (const auto& r : running_)
      if (r.get() != &self && r->prefilling) return true;
    return false;
  }

  // One prefill chunk or one decode step of `r`. Returns true when the request has ended (and was answered).
  bool unit(Running& r) {
    Engine& eng = *ctx_.engine;
    if (r.prefilling) {
      const bool contended = other_decoding(r);
      const bool last = eng.prefill_chunk(r.cur, contended ? contended_ : 0);
      if (contended) count_interleaved(r);
      if (!last) return false;
      r.prefilling = false;
      begin_decode(r);
      return emit(r) && end(r);
    }
    const auto now = std::chrono::steady_clock::now();
    (other_prefilling(r) ? r.beside : r.alone).add(std::chrono::duration<double>(now - r.last_step).count());
    r.last_step = now;
    eng.select_session(r.job.session);
    ++r.passes;
    {
      std::lock_guard<std::mutex> m(ctx_.metrics.mu);
      ctx_.metrics.batch_rows = r.spec ? std::min(eng.draft_block(), eng.max_context() - static_cast<int>(eng.history().size())) : 1;
    }
    if (r.spec) {
      r.pending = eng.speculate(r.pending.back(), r.job.req.sampling, r.job.req.logprobs ? &r.rows : nullptr);
      // The verify checks only each row's top K for NaN and inf (the plain path's sample() checks the
      // whole row); a row reported as logprobs is checked whole, so a NaN never reaches the client.
      if (r.job.req.logprobs) require_finite(r.rows);
    } else {
      r.logits = eng.step(r.pending.back());
      r.pending = {eng.sample(r.logits, r.job.req.sampling, eng.history())};
    }
    return emit(r) && end(r);
  }

  void count_interleaved(Running& r) {
    ++r.chunks_beside;
    std::lock_guard<std::mutex> m(ctx_.metrics.mu);
    ctx_.metrics.prefill_chunks_interleaved += 1;
  }

  // After the last prefill chunk (the engine's active session is the request's, its logits the prompt's).
  void begin_decode(Running& r) {
    Engine& eng = *ctx_.engine;
    const Job& j = r.job;
    const int ctx_left = eng.max_context() - static_cast<int>(j.prompt.size());
    r.limit = j.req.max_tokens > 0 ? std::min(j.req.max_tokens, ctx_left) : ctx_left;
    r.spec = eng.can_speculate(j.req.sampling);
    const SpecStats& s = eng.session_spec_stats(j.session);
    r.steps0 = s.steps;
    r.drafted0 = s.drafted;
    r.accepted0 = s.accepted;
    r.t_decode = r.last_step = std::chrono::steady_clock::now();
    r.dec.emplace(*ctx_.tok);
    r.stops.emplace(j.req.stop);
    r.in_think = j.req.reasoning;  // the prompt ends inside <think>
    r.logits = eng.logits();
    // `pending` holds chosen tokens not yet emitted: the first from the prefill's logits, then a
    // speculative step's accepted drafts and its next token, or a plain step's sample.
    r.pending = {eng.sample(r.logits, j.req.sampling, eng.history())};
    r.rows.clear();
  }

  // Emits r's pending tokens, as generate() did per token. Returns true when the request is done.
  bool emit(Running& r) {
    Job& job = r.job;
    const int eos = ctx_.engine->config().eos_token;
    // Logprobs (API §3), only when asked: each pending token's logprobs come from the row it was chosen
    // from, `logits` on the plain path and the prefill, or the verify's rows that speculate() copies to
    // `rows` (one per pending token, in order). Nothing here runs otherwise.
    const size_t vocab = ctx_.engine->config().vocab;
    for (size_t j = 0; j < r.pending.size(); ++j) {
      const int t = r.pending[j];
      if (r.ttft < 0) r.ttft = std::chrono::duration<double>(std::chrono::steady_clock::now() - job.received).count();
      if (job.cancelled->load()) {
        r.finish = "cancelled";
        return true;
      }
      if (t == eos || t == ctx_.im_end) {
        r.finish = "stop";
        r.ended_by_eos = true;
        return true;
      }
      ++r.produced;
      if (job.req.logprobs) {
        const float* row = r.rows.empty() ? r.logits.data() : r.rows.data() + j * vocab;
        job.sink.logprobs(top_logprobs(std::span<const float>(row, vocab), t, job.req.top_logprobs));
      }
      if (r.in_think) {
        ++r.reasoning;
        if (t == ctx_.think_end) {
          r.in_think = false;
          r.newlines_after_think = 2;
        }
      } else if (r.newlines_after_think > 0) {
        if (t == ctx_.blank_line && r.newlines_after_think == 2) {
          ++r.reasoning;
          r.newlines_after_think = 0;
        } else if (t == ctx_.newline) {
          ++r.reasoning;
          --r.newlines_after_think;
        } else {
          r.newlines_after_think = 0;
        }
      }
      const std::string ready = r.stops->push(r.dec->push(t));
      if (!ready.empty()) job.sink.text(ready);
      if (r.stops->stopped()) {
        r.finish = "stop";
        return true;
      }
      if (r.produced >= r.limit) return true;
    }
    return false;
  }

  // The request's end: what the decoder and the stop scanner still hold, the counters, the log line, done.
  bool end(Running& r) {
    Engine& eng = *ctx_.engine;
    Job& job = r.job;
    if (!r.stops->stopped()) {
      std::string rest = r.stops->push(r.dec->flush());
      if (!r.stops->stopped()) rest += r.stops->finish();
      if (!rest.empty()) job.sink.text(rest);
    }
    const auto now = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(now - r.t_decode).count();
    const SpecStats& s = eng.session_spec_stats(job.session);
    const long steps = s.steps - r.steps0;
    const EngineStats& stats = r.cur.stats;
    {
      Metrics& m = ctx_.metrics;
      std::lock_guard<std::mutex> l(m.mu);
      m.prompt_tokens += job.prompt.size();
      m.cached_tokens += stats.reused_tokens;
      m.generation_tokens += r.produced + (r.ended_by_eos ? 1 : 0);  // SGLang counts the stop token
      m.verify_calls += r.passes;
      m.e2e_sum += std::chrono::duration<double>(now - job.received).count();
      m.e2e_count += 1;
      if (r.ttft >= 0) {
        m.ttft_sum += r.ttft;
        m.ttft_count += 1;
      }
      m.drafted += s.drafted - r.drafted0;
      m.accepted_drafts += s.accepted - r.accepted0;
      m.spec = eng.spec_stats_total();
    }
    // Step periods (M4 §3): mean and max, apart from those with another request's prefill chunk between.
    char periods[200] = "";
    if (r.alone.n + r.beside.n > 0)
      std::snprintf(periods, sizeof periods, "; step period %.1f ms mean, %.1f max over %d; beside a prefill %.1f ms mean, %.1f max over %d",
                    r.alone.mean_ms(), 1e3 * r.alone.max, r.alone.n, r.beside.mean_ms(), 1e3 * r.beside.max, r.beside.n);
    std::fprintf(stderr,
                 "request: session %d%s, %zu prompt tokens (%d reused), prefill %.2f s (%d chunks, %d beside a decode, "
                 "%.0f tok/s of compute); %d tokens in %.2f s (%.1f tok/s), %s; %d passes (%d batched)%s%s\n",
                 job.session, job.fresh ? " (new)" : "", job.prompt.size(), stats.reused_tokens, stats.prefill_seconds,
                 stats.chunks, r.chunks_beside,
                 stats.compute_seconds > 0 ? stats.prefill_tokens / stats.compute_seconds : 0.0, r.produced, decode_s,
                 decode_s > 0 ? r.produced / decode_s : 0.0, r.finish.c_str(), r.passes, r.batched,
                 r.spec && steps > 0 ? (", " + std::to_string(double(r.produced) / std::max(1, r.passes)).substr(0, 4) +
                                        " tokens/pass").c_str()
                                     : "",
                 periods);
    job.sink.done(r.finish == "cancelled" ? "stop" : r.finish, static_cast<int>(job.prompt.size()), r.produced,
                  stats.reused_tokens, r.reasoning);
    return true;
  }

  ServerContext& ctx_;
  double protect_seconds_;
  int batch_, contended_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Job> queue_;
  bool stop_ = false;
  bool dirty_ = false;  // the queue or the running set changed since pick() last looked
  std::chrono::steady_clock::time_point last_look_{};
  // Touched by the engine thread only.
  std::vector<std::unique_ptr<Running>> running_;
  size_t turn_ = 0;
  std::thread thread_;
};

// Lives as long as either the HTTP handler or the job needs it; touched only on the handler's thread.
struct Exchange {
  proxygen::ResponseHandler* downstream = nullptr;  // null once the client is gone
  folly::EventBase* evb = nullptr;
  std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);
  bool keepalive = false;  // a keep-alive timer is armed (arm_keepalive); cleared by stop_keepalive
  int seq = 0;             // the Responses route's sequence numbers, stamped in write order (emit_event)
};

void post(const std::shared_ptr<Exchange>& ex, std::function<void(proxygen::ResponseHandler*)> fn) {
  ex->evb->runInEventBaseThread([ex, fn = std::move(fn)] {
    if (ex->downstream) fn(ex->downstream);
  });
}

// Keep-alives (API §3, §4). While a streamed request has produced nothing (waiting in the worker's queue,
// or in prefill) the request's event base writes one every `interval_ms`, so a client with an idle
// timeout (Codex ends a stream after 300 s without an event) keeps the stream through a long prefill or
// a queue ahead. Armed on the handler's thread right before the job is submitted. Stopped by
// stop_keepalive, which the engine thread posts before its first write (the first token, done, or
// error): posts from one thread run in order, so a tick never follows that write, and since every write
// runs on the event base a tick never interleaves with one. A tick after EOM, or after a disconnect,
// finds `downstream` null (Handler::finish) and does nothing. The timer itself is not cancelled
// (runAfterDelay has no handle): a cleared flag makes the next tick a no-op that does not re-arm, which
// releases its closure within one interval.
void keepalive_tick(std::shared_ptr<Exchange> ex, int interval_ms, std::function<void(proxygen::ResponseHandler*)> write) {
  ex->evb->tryRunAfterDelay(
      [ex, interval_ms, write] {
        if (!ex->keepalive || !ex->downstream) return;
        write(ex->downstream);
        keepalive_tick(ex, interval_ms, write);
      },
      static_cast<uint32_t>(interval_ms));
}

void arm_keepalive(const std::shared_ptr<Exchange>& ex, int interval_ms, std::function<void(proxygen::ResponseHandler*)> write) {
  if (interval_ms <= 0) return;
  ex->keepalive = true;
  keepalive_tick(ex, interval_ms, std::move(write));
}

void stop_keepalive(const std::shared_ptr<Exchange>& ex) {
  ex->evb->runInEventBaseThread([ex] { ex->keepalive = false; });
}

// Writes one Responses event, numbered in write order. The event base's thread only, with a client.
void emit_event(Exchange& ex, const std::string& type, json payload) {
  payload["type"] = type;
  payload["sequence_number"] = ex.seq++;
  ResponseBuilder(ex.downstream).body("event: " + type + "\ndata: " + dump_json(payload) + "\n\n").send();
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
  Handler(ServerContext& ctx, Scheduler& scheduler) : ctx_(ctx), scheduler_(scheduler) {}

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
      bool keepalive = false;  // keep-alives are armed and nothing has been written yet
      // Logprobs in the route's shape (API §3): chat, the `content` entries; completions, the object with
      // `tokens`, `token_logprobs`, `top_logprobs` and `text_offset`. Streamed, what the next chunk takes.
      json logprobs;
      int text_chars = 0;  // characters of output text so far (completions text_offset)
      State(bool r, const json& t) : parser(r, t) {}
    };
    auto st = std::make_shared<State>(chat && req.reasoning, req.tools);
    auto fresh_logprobs = [chat]() {
      return chat ? json(json::array())
                  : json{{"tokens", json::array()}, {"token_logprobs", json::array()}, {"top_logprobs", json::array()},
                         {"text_offset", json::array()}};
    };
    st->logprobs = fresh_logprobs();
    auto pending_logprobs = [=]() { return chat ? !st->logprobs.empty() : !st->logprobs["tokens"].empty(); };
    auto take_logprobs = [=]() {  // choices[0].logprobs from what has been collected, which starts afresh
      json out = chat ? json{{"content", st->logprobs}} : st->logprobs;
      st->logprobs = fresh_logprobs();
      return out;
    };

    auto sse = [ex](const json& chunk) {
      std::string data = "data: " + dump_json(chunk) + "\n\n";
      post(ex, [data](proxygen::ResponseHandler* d) { ResponseBuilder(d).body(data).send(); });
    };
    // Before the first write of any kind (the first token, done, or error): no keep-alive after it.
    auto stop_ticks = [=]() {
      if (!st->keepalive) return;
      st->keepalive = false;
      stop_keepalive(ex);
    };
    auto chunk_of = [=](json delta, json finish, json logprobs = json()) {
      json c = {{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"}, {"created", created},
                {"model", model}};
      json choice = {{"index", 0}};
      if (chat) choice["delta"] = std::move(delta);
      else choice["text"] = delta.is_string() ? delta : json("");
      if (!logprobs.is_null()) choice["logprobs"] = std::move(logprobs);
      choice["finish_reason"] = std::move(finish);
      c["choices"] = json::array({choice});
      return c;
    };
    // A streamed chunk carries the logprobs of the tokens produced since the previous chunk (the text of
    // a token may follow its entry by a chunk when the decoder or the parser held it back).
    auto chunk_logprobs = [=]() { return req.logprobs ? take_logprobs() : json(); };
    auto deliver = [=](const OutputParser::Delta& d) {
      st->content += d.content;
      st->reasoning += d.reasoning;
      if (!req.stream) {
        st->calls.insert(st->calls.end(), d.tool_calls.begin(), d.tool_calls.end());
        return;
      }
      if (!chat) {
        if (!d.content.empty()) sse(chunk_of(json(d.content), nullptr, chunk_logprobs()));
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
      if (delta.size() > 0) sse(chunk_of(delta, nullptr, chunk_logprobs()));
    };

    job.sink.text = [=](const std::string& piece) {
      stop_ticks();
      deliver(chat ? st->parser.push(piece) : OutputParser::Delta{{}, piece, {}});
    };
    if (req.logprobs) {
      // One entry per produced token, in vLLM's shapes. `token` is the token's text (U+FFFD where a
      // character is split across tokens, as dump_json writes it) and `bytes` its exact bytes, so a client
      // can reassemble a character from its pieces. text_offset counts characters of the output text (a
      // byte that is not a continuation byte starts one), so a character split across tokens counts once.
      const Tokenizer* tok = ctx_.tok;
      auto token_json = [tok](const TokenLogprob& e) {
        const std::string text = tok->token_bytes(e.token);
        json bytes = json::array();
        for (const unsigned char c : text) bytes.push_back(static_cast<int>(c));
        return json{{"token", text}, {"logprob", e.logprob}, {"bytes", bytes}};
      };
      const int k = req.top_logprobs;
      job.sink.logprobs = [=](const std::vector<TokenLogprob>& entries) {
        if (entries.empty()) return;
        if (chat) {
          json e = token_json(entries[0]);
          json top = json::array();
          for (size_t i = 0; i < entries.size() && i < static_cast<size_t>(k); ++i) top.push_back(token_json(entries[i]));
          e["top_logprobs"] = std::move(top);
          st->logprobs.push_back(std::move(e));
          return;
        }
        const std::string text = tok->token_bytes(entries[0].token);
        json top = json::object();
        for (const TokenLogprob& e : entries) top[tok->token_bytes(e.token)] = e.logprob;
        st->logprobs["tokens"].push_back(text);
        st->logprobs["token_logprobs"].push_back(entries[0].logprob);
        st->logprobs["top_logprobs"].push_back(std::move(top));
        st->logprobs["text_offset"].push_back(st->text_chars);
        for (const unsigned char c : text) st->text_chars += (c & 0xC0) != 0x80;
      };
    }
    job.sink.error = [=](const std::string& msg, const std::string& code) {
      stop_ticks();
      const json error = error_body(msg, error_type(code), code);
      if (req.stream) {  // the headers are out: end the stream properly after the error
        const std::string tail = stream_error_tail(error, chunk_of(json::object(), kStreamErrorFinish));
        post(ex, [tail](proxygen::ResponseHandler* d) { ResponseBuilder(d).body(tail).sendWithEOM(); });
      } else {
        const int status = code == kContextLengthExceeded ? 400 : 500;
        post(ex, [error, status](proxygen::ResponseHandler* d) { send_json(d, status, error); });
      }
    };
    job.sink.done = [=](const std::string& finish_in, int prompt_tokens, int completion_tokens, int cached,
                        int reasoning_tokens) {
      stop_ticks();
      if (chat) deliver(st->parser.finish());
      const std::string finish = !st->calls.empty() && finish_in == "stop" ? "tool_calls" : finish_in;
      json usage = {{"prompt_tokens", prompt_tokens},
                    {"completion_tokens", completion_tokens},
                    {"total_tokens", prompt_tokens + completion_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", cached}}}};
      if (chat) usage["completion_tokens_details"] = {{"reasoning_tokens", reasoning_tokens}};
      if (req.stream) {
        // Entries still pending (their text held back to the end) ride on the finish chunk.
        sse(chunk_of(json::object(), finish, req.logprobs && pending_logprobs() ? take_logprobs() : json()));
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
      if (req.logprobs) choice["logprobs"] = take_logprobs();
      json body = {{"id", id}, {"object", chat ? "chat.completion" : "text_completion"}, {"created", created},
                   {"model", model}, {"choices", json::array({choice})}, {"usage", usage}};
      post(ex, [body](proxygen::ResponseHandler* d) { send_json(d, 200, body); });
    };
    if (req.stream && ctx_.keepalive_ms > 0) {  // an SSE comment, which every conforming parser ignores (API §3)
      st->keepalive = true;
      arm_keepalive(ex, ctx_.keepalive_ms, [](proxygen::ResponseHandler* d) {
        ResponseBuilder(d).body(std::string(": keep-alive\n\n")).send();
      });
    }
    scheduler_.submit(std::move(job));
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
      json output = json::array();   // finished items, in order
      std::string reasoning, content;
      std::string reasoning_id, message_id;
      int reasoning_index = -1, message_index = -1;
      bool reasoning_open = false, message_open = false;
      bool keepalive = false;  // keep-alives are armed and nothing has been written yet
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
    // Events are numbered when they are written, on the event base (emit_event), so a keep-alive written
    // by the timer and an event posted by the engine thread share one counter and stay consecutive.
    auto event = [=](const std::string& type, json payload) {
      if (!req.stream) return;
      post(ex, [ex, type, payload](proxygen::ResponseHandler*) { emit_event(*ex, type, payload); });
    };
    // Before the first write of any kind (the first token, done, or error): no keep-alive after it.
    auto stop_ticks = [=]() {
      if (!st->keepalive) return;
      st->keepalive = false;
      stop_keepalive(ex);
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
      stop_ticks();
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

    job.sink.text = [=](const std::string& piece) {
      stop_ticks();
      deliver(st->parser.push(piece));
    };
    job.sink.error = fail;
    job.sink.done = [=](const std::string& finish, int prompt_tokens, int completion_tokens, int cached,
                        int reasoning_tokens) {
      stop_ticks();
      deliver(st->parser.finish());
      close_reasoning();
      close_message();
      json resp = response_obj(finish == "length" ? "incomplete" : "completed");
      if (finish == "length") resp["incomplete_details"] = {{"reason", "max_output_tokens"}};
      resp["usage"] = {{"input_tokens", prompt_tokens},
                       {"input_tokens_details", {{"cached_tokens", cached}}},
                       {"output_tokens", completion_tokens},
                       {"output_tokens_details", {{"reasoning_tokens", reasoning_tokens}}},
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
    if (req.stream && ctx_.keepalive_ms > 0) {
      // response.in_progress with the response object as response.created sent it (the output is still
      // empty, and it is taken here, before the engine thread can touch it); Codex ignores the kind.
      st->keepalive = true;
      const json in_progress = {{"response", response_obj("in_progress")}};
      arm_keepalive(ex, ctx_.keepalive_ms, [ex, in_progress](proxygen::ResponseHandler*) {
        emit_event(*ex, "response.in_progress", in_progress);
      });
    }
    scheduler_.submit(std::move(job));
  }

  ServerContext& ctx_;
  Scheduler& scheduler_;
  std::string method_, path_;
  folly::IOBufQueue body_{folly::IOBufQueue::cacheChainLength()};
  bool too_large_ = false;
  std::shared_ptr<Exchange> ex_ = std::make_shared<Exchange>();
};

class Factory : public proxygen::RequestHandlerFactory {
 public:
  Factory(ServerContext& ctx, Scheduler& scheduler) : ctx_(ctx), scheduler_(scheduler) {}
  void onServerStart(folly::EventBase*) noexcept override {}
  void onServerStop() noexcept override {}
  proxygen::RequestHandler* onRequest(proxygen::RequestHandler*, HTTPMessage*) noexcept override {
    return new Handler(ctx_, scheduler_);
  }

 private:
  ServerContext& ctx_;
  Scheduler& scheduler_;
};

}  // namespace
}  // namespace ling::serve

int main(int argc, char** argv) {
  std::string model_dir, host = "0.0.0.0", name;
  std::string draft_head = "full", sparse_head_dir;  // BACKLOG §19.19
  int sparse_head_dynamic = 16384;
  int port = 8000, max_context = 65536, draft_block = 12, lookup = 1, lookup_min = 8, checkpoints = 8;
  std::string draft_dir, pretokenizer = "production";
  bool graphs = false, pdl = false, attn_bulk = false, prefill_bf16 = false;
  int attn_prefetch = 2, keepalive_seconds = 15, sessions = 2, batch = 2, contended = 256;
  double protect_seconds = 30;
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
    else if (a == "--draft-head") draft_head = next();
    else if (a == "--sparse-head-dir") sparse_head_dir = next();
    else if (a == "--sparse-head-dynamic") sparse_head_dynamic = std::stoi(next());
    else if (a == "--lookup") lookup = std::stoi(next());
    else if (a == "--lookup-min") lookup_min = std::stoi(next());
    else if (a == "--pretokenizer") pretokenizer = next();
    else if (a == "--prefix-checkpoints") checkpoints = std::stoi(next());
    else if (a == "--graphs") graphs = true;
    else if (a == "--pdl") pdl = true;
    else if (a == "--attn-bulk") attn_bulk = true;
    else if (a == "--attn-prefetch") attn_prefetch = std::stoi(next());
    else if (a == "--keepalive-seconds") keepalive_seconds = std::stoi(next());
    else if (a == "--prefill-bf16") prefill_bf16 = true;
    else if (a == "--sessions") sessions = std::stoi(next());
    else if (a == "--session-protect-seconds") protect_seconds = std::stod(next());
    else if (a == "--batch") batch = std::stoi(next());
    else if (a == "--prefill-chunk-contended") contended = std::stoi(next());
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::cerr << "usage: ling-serve --model DIR [--host H] [--port P] [--served-model-name N] [--max-context N] [--draft DIR [--draft-block N] [--draft-head full|sparse --sparse-head-dir DIR [--sparse-head-dynamic N]]] [--pretokenizer production|checkpoint] [--prefix-checkpoints N] [--keepalive-seconds N] [--sessions N] [--session-protect-seconds S] [--batch N] [--prefill-chunk-contended N]\n";
    return 2;
  }
  if (const char* e = std::getenv("LING_KEEPALIVE_SECONDS")) {  // overrides the flag, as the engine's variables do
    try {
      keepalive_seconds = std::stoi(e);
    } catch (const std::exception&) {
      std::cerr << "LING_KEEPALIVE_SECONDS must be an integer\n";
      return 2;
    }
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
  if (draft_head == "sparse") {
    if (sparse_head_dir.empty()) {
      std::cerr << "--draft-head sparse needs --sparse-head-dir DIR\n";
      return 2;
    }
    opts.sparse_head_dir = sparse_head_dir;
    opts.sparse_head_dynamic = sparse_head_dynamic;
  } else if (draft_head != "full") {
    std::cerr << "--draft-head is full or sparse\n";
    return 2;
  }
  opts.lookup_mode = lookup;
  opts.lookup_min_match = lookup_min;
  opts.prefix_checkpoints = checkpoints;
  opts.boundary_token = tok.token_id("<|im_end|>");
  opts.step_graphs = graphs;
  opts.pdl = pdl;
  opts.attention_bulk = attn_bulk;
  opts.attention_prefetch = attn_prefetch;
  opts.prefill_bf16 = prefill_bf16;
  opts.sessions = sessions;
  std::cerr << "loading " << model_dir << " ...\n";
  ling::Engine engine(model_dir, opts);
  std::cerr << "loaded " << engine.model().device_bytes() / 1e9 << " GB of weights\n";

  ling::serve::ServerContext ctx;
  ctx.engine = &engine;
  ctx.tok = &tok;
  ctx.im_end = tok.token_id("<|im_end|>");
  ctx.metrics.max_total_tokens = engine.sessions() * max_context;
  ctx.metrics.sessions = engine.sessions();
  ctx.metrics.protect_seconds = protect_seconds;
  if (batch < 1) {
    std::cerr << "--batch must be at least 1\n";
    return 2;
  }
  batch = std::min(batch, engine.sessions());  // one running request per session
  if (contended < 32 || contended > engine.prefill_chunk_size()) {
    std::cerr << "--prefill-chunk-contended must be in [32, " << engine.prefill_chunk_size() << "]\n";
    return 2;
  }
  ctx.metrics.batch = batch;
  ctx.metrics.prefill_chunk_contended = contended;
  ctx.keepalive_ms = keepalive_seconds > 0 ? keepalive_seconds * 1000 : 0;
  ctx.think_end = tok.token_id("</think>");
  {
    const std::vector<int> one = tok.encode("\n"), two = tok.encode("\n\n");
    ctx.newline = one.size() == 1 ? one[0] : -1;
    ctx.blank_line = two.size() == 1 ? two[0] : -1;
  }
  if (name.empty()) {
    name = model_dir;
    while (!name.empty() && name.back() == '/') name.pop_back();
    name = name.substr(name.find_last_of('/') + 1);
  }
  ctx.model_name = name;
  ling::serve::Scheduler scheduler(ctx, protect_seconds, batch, contended);

  proxygen::HTTPServerOptions options;
  options.threads = 2;
  options.idleTimeout = std::chrono::minutes(30);
  options.shutdownOn = {SIGINT, SIGTERM};
  options.enableContentCompression = false;
  options.handlerFactories =
      proxygen::RequestHandlerChain().addThen<ling::serve::Factory>(ctx, scheduler).build();
  std::vector<proxygen::HTTPServer::IPConfig> ips = {
      {folly::SocketAddress(host, static_cast<uint16_t>(port), true), proxygen::HTTPServer::Protocol::HTTP}};
  proxygen::HTTPServer server(std::move(options));
  server.bind(ips);
  std::cerr << "ling-serve: serving " << name << " on " << host << ":" << port << "\n";
  std::thread t([&] { server.start(); });
  t.join();
  return 0;
}
