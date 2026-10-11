// ling-run: loads the model and generates from a prompt; prints tokens, timings and (with --ids) the
// token ids, for checking against a reference server.
//
//   ling-run --model DIR [--text "..." | --chat "..." | --ids 1,2,3] [--max-tokens N] [--temperature T]
//            [--prompts-file F] [--ids-out] [--prompt-ids-out] [--max-context N]
//            [--draft DIR [--block N] [--spec] [--spec-check] [--lookup 0|1|2] [--lookup-min N]]
//            [--graphs] [--pdl] [--attn-bulk] [--attn-prefetch N] [--prefill-bf16] [--graph-check] [--seed N]
//            [--prefix-check [--sessions N]] [--chunk-check N] [--interleave-check [--prefill-chunk-contended N]]
//
// --graphs runs each speculative step's draft, verify and commit as captured CUDA graphs (M3); --pdl launches
// the rows path's kernels with programmatic dependent launch (M3); --attn-bulk loads the rows path's attention
// tiles with bulk copies (M3); --attn-prefetch N sets how many KV tiles ahead its attention prefetches into L2
// (default 2, 0 off; M3); --prefill-bf16 keeps the prefill's DeltaNet intermediates in BF16 as production does
// (B34; not exact, off by default: MILESTONES §14).
//
// --graph-check decodes each prompt speculatively twice from the same prefill, without and with step graphs,
// greedily and then sampling (--temperature, default 1, top-k 20, top-p 0.95, seeded by --seed or 1), and
// checks that the tokens, the recurrent state (bit for bit) and the steps and accepted drafts (which would show a
// wrong drafter context KV) are the same both ways. --seed N seeds sampling.
//
// --spec decodes with the DFlash2 drafter. --spec-check (greedy) decodes each prompt plainly and then
// speculatively, and checks that the tokens are identical and that the recurrent state after the
// speculative run equals the state of a plain run over the same tokens, bit for bit.
//
// --prefix-check prefills each prompt after the one before it (so it resumes from a kept state: the end of
// the previous prompt or a prefix checkpoint), then again from scratch, and checks that the DeltaNet and
// conv state and the next token are the same both ways (M2: prefix checkpoints). With --draft it also
// decodes 64 tokens speculatively (greedy) after each prefill and checks that the tokens, the state, the
// steps and the accepted drafts are the same both ways (the drafter's context ring after a resume, M4 §1).
// --sessions N (M4 §1) keeps N resident sessions and binds each prompt to one as ling-serve does (the
// session it extends, else a free or the least recently used slot), so each prompt resumes from its own
// session's state, from the shared checkpoint store, or from another session's KV.
//
// --chunk-check N (M4 §3) prefills each prompt from a cold engine twice, in full chunks and in chunks of N
// tokens (what ling-serve runs while another request decodes), and checks that the state and the next token
// are the same both ways; with --draft also the 64-token speculative continuation's tokens, state, steps and
// accepted drafts (the drafter's context ring, which the prefill's chunks fill).
//
// --interleave-check (M4 §3, stage 2; needs --draft and --sessions 2) takes the prompts in pairs (A, B). Each
// is first decoded alone (cold engine, full chunks, --max-tokens speculatively); then A is prefilled and
// decodes while B prefills in chunks of --prefill-chunk-contended tokens (default 256) between A's steps, and
// the two then take steps in turn, as ling-serve interleaves two requests. Per sequence the tokens, the state,
// the steps and the accepted drafts must equal the run alone. Greedy and sampled (--temperature, default 1,
// seeded by --seed or 1), and with the pair swapped.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/engine.hpp"
#include "tokenizer/chat_template.hpp"
#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model, text, chat, ids_arg, prompts_file;
  int max_tokens = 64, max_context = 32768;
  float temperature = 0.f;
  bool ids_out = false, prompt_ids_out = false, spec = false, spec_check = false, prefix_check = false;
  bool graphs = false, pdl = false, attn_bulk = false, graph_check = false, prefill_bf16 = false;
  uint64_t seed = 0;
  int attn_prefetch = 2;
  std::string draft;
  int block = 12, lookup = 1, lookup_min = 8, sessions = 1, chunk_check = 0, contended = 256;
  bool interleave_check = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--text") text = next();
    else if (a == "--chat") chat = next();
    else if (a == "--ids") ids_arg = next();
    else if (a == "--prompts-file") prompts_file = next();
    else if (a == "--max-tokens") max_tokens = std::stoi(next());
    else if (a == "--max-context") max_context = std::stoi(next());
    else if (a == "--temperature") temperature = std::stof(next());
    else if (a == "--ids-out") ids_out = true;
    else if (a == "--prompt-ids-out") prompt_ids_out = true;
    else if (a == "--draft") draft = next();
    else if (a == "--block") block = std::stoi(next());
    else if (a == "--spec") spec = true;
    else if (a == "--lookup") lookup = std::stoi(next());
    else if (a == "--lookup-min") lookup_min = std::stoi(next());
    else if (a == "--spec-check") spec_check = true;
    else if (a == "--prefix-check") prefix_check = true;
    else if (a == "--graphs") graphs = true;
    else if (a == "--graph-check") graph_check = true;
    else if (a == "--seed") seed = std::stoull(next());
    else if (a == "--pdl") pdl = true;
    else if (a == "--attn-bulk") attn_bulk = true;
    else if (a == "--attn-prefetch") attn_prefetch = std::stoi(next());
    else if (a == "--prefill-bf16") prefill_bf16 = true;
    else if (a == "--sessions") sessions = std::stoi(next());
    else if (a == "--chunk-check") chunk_check = std::stoi(next());
    else if (a == "--interleave-check") interleave_check = true;
    else if (a == "--prefill-chunk-contended") contended = std::stoi(next());
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model.empty()) {
    std::cerr << "usage: ling-run --model DIR (--text T | --chat T | --ids 1,2,3) [--max-tokens N]\n";
    return 2;
  }
  try {
    ling::Tokenizer tok(model + "/tokenizer.json");
    std::vector<std::vector<int>> prompts;
    if (!prompts_file.empty()) {  // one JSON string per line, each a raw prompt
      std::ifstream f(prompts_file);
      std::string line;
      while (std::getline(f, line))
        if (!line.empty()) prompts.push_back(tok.encode(nlohmann::json::parse(line).get<std::string>()));
    } else if (!ids_arg.empty()) {
      std::vector<int> prompt;
      std::stringstream ss(ids_arg);
      std::string part;
      while (std::getline(ss, part, ',')) prompt.push_back(std::stoi(part));
      prompts.push_back(prompt);
    } else if (!chat.empty()) {
      nlohmann::ordered_json messages = nlohmann::ordered_json::array();
      messages.push_back({{"role", "user"}, {"content", chat}});
      ling::ChatOptions o;
      o.enable_thinking = false;
      prompts.push_back(tok.encode(ling::render_chat(messages, nlohmann::ordered_json(), o)));
    } else {
      prompts.push_back(tok.encode(text));
    }
    auto t0 = std::chrono::steady_clock::now();
    ling::EngineOptions opts;
    opts.max_context = max_context;
    opts.draft_dir = draft;
    opts.draft_block = block;
    opts.lookup_mode = lookup;
    opts.lookup_min_match = lookup_min;
    opts.boundary_token = tok.token_id("<|im_end|>");
    opts.step_graphs = graphs;
    opts.pdl = pdl;
    opts.attention_bulk = attn_bulk;
    opts.attention_prefetch = attn_prefetch;
    opts.prefill_bf16 = prefill_bf16;
    opts.sessions = sessions;
    ling::Engine engine(model, opts);
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "loaded %.1f GB of weights in %.1f s\n", engine.model().device_bytes() / 1e9,
                 std::chrono::duration<double>(t1 - t0).count());
    const int im_end = tok.token_id("<|im_end|>");
    auto is_end = [&](int t) { return t == engine.config().eos_token || t == im_end; };

    // Generates from `prompt` (state reset first): plainly or speculatively. Prints text unless `quiet`.
    auto generate = [&](const std::vector<int>& prompt, const ling::SamplingParams& sp, bool use_spec, bool quiet,
                        double* decode_s) {
      ling::EngineStats stats;
      engine.reset();
      std::vector<float> logits = engine.prefill(prompt, &stats);
      if (!quiet)
        std::fprintf(stderr, "prefill: %d tokens in %.3f s (%.0f tok/s)\n", stats.prefill_tokens,
                     stats.prefill_seconds, stats.prefill_tokens / stats.prefill_seconds);
      ling::StreamDecoder dec(tok);
      std::vector<int> out;
      auto t2 = std::chrono::steady_clock::now();
      std::vector<int> pending = {engine.sample(logits, sp, engine.history())};
      bool done = false;
      while (!done) {
        for (int t : pending) {
          out.push_back(t);
          if (is_end(t) || static_cast<int>(out.size()) >= max_tokens) {
            done = true;
            break;
          }
          if (!quiet && !ids_out) std::cout << dec.push(t) << std::flush;
        }
        if (done) break;
        if (use_spec) {
          pending = engine.speculate(pending.back(), sp);
        } else {
          logits = engine.step(pending.back());
          pending = {engine.sample(logits, sp, engine.history())};
        }
      }
      *decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
      if (!quiet && !ids_out) std::cout << dec.flush() << "\n";
      return out;
    };

    int failures = 0;
    // A sequence decoded speculatively in session `session`, one step at a time (--chunk-check, --interleave-check).
    struct Seq {
      int session = 0;
      ling::SamplingParams sp;
      std::vector<int> out, pending;
      bool done = false;
    };
    // Emits the pending tokens; unless that ends the sequence, runs one speculative step in its session.
    auto advance = [&](Seq& q, int limit) {
      for (int t : q.pending) {
        q.out.push_back(t);
        if (is_end(t) || static_cast<int>(q.out.size()) >= limit) {
          q.done = true;
          return;
        }
      }
      engine.select_session(q.session);
      q.pending = engine.speculate(q.pending.back(), q.sp);
    };
    struct Result {
      std::vector<int> tokens;
      uint64_t hash = 0;
      long steps = 0, accepted = 0;
      bool operator==(const Result& o) const {
        return tokens == o.tokens && hash == o.hash && steps == o.steps && accepted == o.accepted;
      }
    };
    auto result_of = [&](const Seq& q) {
      Result r;
      r.tokens = q.out;
      engine.select_session(q.session);
      r.hash = engine.state_hash();
      r.steps = engine.session_spec_stats(q.session).steps;
      r.accepted = engine.session_spec_stats(q.session).accepted;
      return r;
    };
    if (chunk_check > 0) {
      auto argmax = [](const std::vector<float>& l) { return int(std::max_element(l.begin(), l.end()) - l.begin()); };
      for (size_t i = 0; i < prompts.size(); ++i) {
        Result r[2];
        int next[2];
        ling::EngineStats st[2];
        for (int k = 0; k < 2; ++k) {
          engine.reset_all();
          engine.activate_session(0, "", true);
          ling::Engine::PrefillCursor cur = engine.begin_prefill(prompts[i]);
          while (!engine.prefill_chunk(cur, k ? chunk_check : 0)) {
          }
          st[k] = cur.stats;
          const std::vector<float> logits = engine.logits();
          next[k] = argmax(logits);
          Seq q;
          q.sp.temperature = 0.f;
          if (engine.has_drafter()) {
            q.pending = {engine.sample(logits, q.sp, engine.history())};
            while (!q.done) advance(q, 64);
          }
          r[k] = result_of(q);
        }
        const bool same = r[0] == r[1] && next[0] == next[1];
        failures += !same;
        std::printf("prompt %zu: %zu tokens, %d chunks (%.2f s) against %d chunks of %d (%.2f s): state %s, next token %s, "
                    "continuation %zu tokens %ld steps %ld accepted against %ld steps %ld accepted: %s\n",
                    i, prompts[i].size(), st[0].chunks, st[0].compute_seconds, st[1].chunks, chunk_check,
                    st[1].compute_seconds, r[0].hash == r[1].hash ? "identical" : "DIFFERS",
                    next[0] == next[1] ? "identical" : "DIFFERS", r[1].tokens.size(), r[0].steps, r[0].accepted,
                    r[1].steps, r[1].accepted, same ? "identical" : "DIFFERS");
      }
      std::printf(failures ? "CHUNK CHECK FAILED\n" : "chunk check passed\n");
      return failures ? 1 : 0;
    }
    if (interleave_check) {
      if (!engine.has_drafter() || engine.sessions() < 2 || prompts.size() < 2)
        throw std::runtime_error("--interleave-check needs --draft, --sessions 2 and at least two prompts");
      for (size_t pi = 0; pi + 1 < prompts.size(); pi += 2) {
        for (int swapped = 0; swapped < 2; ++swapped) {
          const std::vector<int>& A = prompts[pi + swapped];
          const std::vector<int>& B = prompts[pi + 1 - swapped];
          for (int sampled = 0; sampled < 2; ++sampled) {
            ling::SamplingParams sp;
            sp.temperature = sampled ? (temperature > 0.f ? temperature : 1.f) : 0.f;
            sp.seed = seed ? seed : 1;
            // Each alone, from a cold engine in session 0.
            Result alone[2];
            for (int k = 0; k < 2; ++k) {
              engine.reset_all();
              engine.activate_session(0, "", true);
              engine.prefill(k ? B : A);
              Seq q;
              q.sp = sp;
              q.pending = {engine.sample(engine.logits(), sp, engine.history())};
              while (!q.done) advance(q, max_tokens);
              alone[k] = result_of(q);
            }
            // Interleaved: A in session 0 prefills and takes two steps alone; then B (session 1) prefills in
            // contended chunks, one between each of A's steps, and the two take steps in turn.
            engine.reset_all();
            Seq a, b;
            a.session = 0;
            b.session = 1;
            a.sp = b.sp = sp;
            engine.activate_session(0, "", true);
            engine.prefill(A);
            a.pending = {engine.sample(engine.logits(), sp, engine.history())};
            for (int k = 0; k < 2 && !a.done; ++k) advance(a, max_tokens);
            engine.activate_session(1, "", true);
            ling::Engine::PrefillCursor cur = engine.begin_prefill(B);
            int chunks_beside = 0;
            while (!a.done || !b.done) {
              if (!cur.done()) {
                if (engine.prefill_chunk(cur, a.done ? 0 : contended)) {
                  b.pending = {engine.sample(engine.logits(), sp, engine.history())};
                }
                chunks_beside += !a.done;
              }
              if (!a.done) advance(a, max_tokens);
              if (cur.done() && !b.done) advance(b, max_tokens);
            }
            const Result ra = result_of(a), rb = result_of(b);
            const bool same_a = ra == alone[0], same_b = rb == alone[1];
            failures += !same_a + !same_b;
            std::printf("pair %zu%s, %s: A %zu prompt tokens, %zu out, %ld steps %ld accepted: %s | B %zu prompt tokens "
                        "(%d chunks, %d beside A's steps), %zu out, %ld steps %ld accepted: %s\n",
                        pi / 2, swapped ? " swapped" : "", sampled ? "sampled" : "greedy", A.size(), ra.tokens.size(),
                        ra.steps, ra.accepted, same_a ? "identical" : "DIFFERS", B.size(), cur.stats.chunks, chunks_beside,
                        rb.tokens.size(), rb.steps, rb.accepted, same_b ? "identical" : "DIFFERS");
            if (!same_a)
              std::printf("  A alone: %zu out, %ld steps %ld accepted, state %s\n", alone[0].tokens.size(), alone[0].steps,
                          alone[0].accepted, alone[0].hash == ra.hash ? "identical" : "differs");
            if (!same_b)
              std::printf("  B alone: %zu out, %ld steps %ld accepted, state %s\n", alone[1].tokens.size(), alone[1].steps,
                          alone[1].accepted, alone[1].hash == rb.hash ? "identical" : "differs");
          }
        }
      }
      std::printf(failures ? "INTERLEAVE CHECK FAILED\n" : "interleave check passed\n");
      return failures ? 1 : 0;
    }
    if (prefix_check) {
      auto argmax = [](const std::vector<float>& l) { return int(std::max_element(l.begin(), l.end()) - l.begin()); };
      // With a drafter: 64 tokens decoded speculatively (greedy) from the state the prefill left.
      struct After {
        std::vector<int> tokens;
        long steps = 0, accepted = 0;
        uint64_t hash = 0;
      };
      auto continue_spec = [&](const std::vector<float>& logits) {
        After a;
        if (!engine.has_drafter()) return a;
        ling::SamplingParams gp;
        gp.temperature = 0.f;
        engine.reset_spec_stats();
        std::vector<int> pending = {engine.sample(logits, gp, engine.history())};
        while (true) {
          a.tokens.insert(a.tokens.end(), pending.begin(), pending.end());
          if (a.tokens.size() >= 64 || is_end(a.tokens.back())) break;
          pending = engine.speculate(pending.back(), gp);
        }
        a.steps = engine.spec_stats().steps;
        a.accepted = engine.spec_stats().accepted;
        a.hash = engine.state_hash();
        return a;
      };
      engine.reset();
      for (size_t i = 0; i < prompts.size(); ++i) {
        // Bind the prompt as ling-serve does: the session it extends, else a new one (a never-used slot,
        // else the least recently used).
        int si = engine.match_session(prompts[i], "");
        const bool fresh_session = si < 0;
        if (fresh_session) si = engine.free_session(0.0, {});
        engine.activate_session(si, "", fresh_session);
        ling::EngineStats st;
        const std::vector<float> resumed_logits = engine.prefill(prompts[i], &st);
        const int resumed = argmax(resumed_logits);
        const uint64_t h_resumed = engine.state_hash();
        const After a_resumed = continue_spec(resumed_logits);
        ling::EngineStats fresh_st;
        const std::vector<float> fresh_logits = engine.prefill(prompts[i], &fresh_st, /*from_scratch=*/true);
        const int fresh = argmax(fresh_logits);
        const uint64_t h_fresh = engine.state_hash();
        const After a_fresh = continue_spec(fresh_logits);
        engine.touch_session();
        const bool same = resumed == fresh && h_resumed == h_fresh;
        const bool spec_same = a_resumed.tokens == a_fresh.tokens && a_resumed.hash == a_fresh.hash &&
                               a_resumed.steps == a_fresh.steps && a_resumed.accepted == a_fresh.accepted;
        failures += !same + !spec_same;
        std::printf("prompt %zu: %zu tokens, session %d%s, resumed at %d (%.2f s) against %.2f s from scratch: state %s, "
                    "next token %s",
                    i, prompts[i].size(), si, fresh_session ? " (new)" : "", st.reused_tokens, st.prefill_seconds,
                    fresh_st.prefill_seconds, h_resumed == h_fresh ? "identical" : "DIFFERS",
                    resumed == fresh ? "identical" : "DIFFERS");
        if (engine.has_drafter())
          std::printf("; speculative continuation %zu tokens, %ld steps %ld accepted against %ld steps %ld accepted: %s",
                      a_resumed.tokens.size(), a_resumed.steps, a_resumed.accepted, a_fresh.steps, a_fresh.accepted,
                      spec_same ? "identical" : "DIFFERS");
        std::printf("\n");
        // The session is left as a server would leave it after prompt i (and its continuation).
      }
      std::printf(failures ? "PREFIX CHECK FAILED\n" : "prefix check passed\n");
      return failures ? 1 : 0;
    }
    for (const std::vector<int>& prompt : prompts) {
      ling::SamplingParams sp;
      sp.temperature = temperature;
      sp.seed = seed;
      if (graph_check) {
        for (int sampled = 0; sampled < 2; ++sampled) {
          ling::SamplingParams gp;
          gp.temperature = sampled ? (temperature > 0.f ? temperature : 1.f) : 0.f;
          gp.seed = seed ? seed : 1;
          double t_eager = 0, t_graph = 0;
          // The steps and accepted drafts must match too: the commit also writes the drafter's context KV, which
          // neither the tokens (the verify corrects every draft) nor the state hash would show if it were wrong.
          engine.set_step_graphs(false);
          engine.reset_spec_stats();
          const std::vector<int> eager = generate(prompt, gp, true, true, &t_eager);
          const uint64_t h_eager = engine.state_hash();
          const long steps_eager = engine.spec_stats().steps, acc_eager = engine.spec_stats().accepted;
          engine.set_step_graphs(true);
          if (!engine.step_graphs()) throw std::runtime_error("--graph-check: step graphs are unavailable");
          engine.reset_spec_stats();
          const std::vector<int> graph = generate(prompt, gp, true, true, &t_graph);
          const uint64_t h_graph = engine.state_hash();
          const long steps_graph = engine.spec_stats().steps, acc_graph = engine.spec_stats().accepted;
          const bool same = eager == graph, state = h_eager == h_graph;
          const bool accept = steps_eager == steps_graph && acc_eager == acc_graph;
          failures += !same + !state + !accept;
          std::printf("prompt %zu tokens, %s: eager %zu tokens %.2f tok/s, %ld steps %ld accepted | graphs %zu tokens "
                      "%.2f tok/s, %ld steps %ld accepted | tokens %s, state %s, acceptance %s\n",
                      prompt.size(), sampled ? "sampled" : "greedy", eager.size(), eager.size() / t_eager, steps_eager,
                      acc_eager, graph.size(), graph.size() / t_graph, steps_graph, acc_graph,
                      same ? "identical" : "DIFFER", state ? "identical" : "DIFFERS", accept ? "identical" : "DIFFERS");
          if (!same) {
            size_t i = 0;
            while (i < eager.size() && i < graph.size() && eager[i] == graph[i]) ++i;
            std::printf("  first difference at output token %zu\n", i);
          }
        }
        continue;
      }
      if (spec_check) {
        sp.temperature = 0.f;
        double t_plain = 0, t_spec = 0;
        const std::vector<int> plain = generate(prompt, sp, false, true, &t_plain);
        engine.reset_spec_stats();
        const std::vector<int> fast = generate(prompt, sp, true, true, &t_spec);
        const uint64_t h_spec = engine.state_hash();
        const std::vector<int> processed(engine.history().begin() + prompt.size(), engine.history().end());
        // The same tokens through plain decoding, from the same prefill.
        engine.reset();
        engine.prefill(prompt);
        for (int t : processed) engine.step(t);
        const uint64_t h_plain = engine.state_hash();
        const ling::SpecStats& st = engine.spec_stats();
        const bool same = plain == fast, state = h_spec == h_plain;
        failures += !same + !state;
        std::printf("prompt %zu tokens: plain %zu tokens %.2f tok/s | spec %zu tokens %.2f tok/s, %ld steps, %.2f "
                    "tokens/step | tokens %s, state %s\n",
                    prompt.size(), plain.size(), plain.size() / t_plain, fast.size(), fast.size() / t_spec, st.steps,
                    st.steps ? double(st.accepted + st.steps) / st.steps : 0.0, same ? "identical" : "DIFFER",
                    state ? "identical" : "DIFFERS");
        if (!same) {
          size_t i = 0;
          while (i < plain.size() && i < fast.size() && plain[i] == fast[i]) ++i;
          std::printf("  first difference at output token %zu\n", i);
        }
        continue;
      }
      double dt = 0;
      engine.reset_spec_stats();
      const std::vector<int> out = generate(prompt, sp, spec, false, &dt);
      if (prompt_ids_out) {
        for (size_t i = 0; i < prompt.size(); ++i) std::cout << (i ? "," : "") << prompt[i];
        std::cout << "\n";
      }
      if (ids_out) {
        for (size_t i = 0; i < out.size(); ++i) std::cout << (i ? "," : "") << out[i];
        std::cout << "\n";
      }
      std::fprintf(stderr, "decode: %zu tokens in %.3f s (%.2f tok/s)\n", out.size(), dt, out.size() / dt);
      if (spec) {
        const ling::SpecStats& st = engine.spec_stats();
        std::fprintf(stderr, "spec: %ld steps, %.2f tokens/step, %.1f ms/step (draft %.1f, verify %.1f, commit %.1f)\n",
                     st.steps, st.steps ? double(st.accepted + st.steps) / st.steps : 0.0,
                     st.steps ? 1e3 * (st.draft_seconds + st.verify_seconds + st.commit_seconds) / st.steps : 0.0,
                     st.steps ? 1e3 * st.draft_seconds / st.steps : 0.0, st.steps ? 1e3 * st.verify_seconds / st.steps : 0.0,
                     st.steps ? 1e3 * st.commit_seconds / st.steps : 0.0);
        if (st.lookup_steps)
          std::fprintf(stderr, "lookup: %ld steps verified its chain, %.2f tokens/step\n", st.lookup_steps,
                       double(st.lookup_accepted + st.lookup_steps) / st.lookup_steps);
        if (st.shadow_total_steps) {
          std::fprintf(stderr, "lookup shadow: drafter alone %.2f accepted/step, best of both %.2f\n",
                       double(st.shadow_dflash_all) / st.shadow_total_steps, double(st.shadow_best) / st.shadow_total_steps);
          const char* names[5] = {"3", "4-7", "8-15", "16-31", "32+"};
          for (int b = 0; b < 5; ++b)
            if (st.shadow_steps[b])
              std::fprintf(stderr, "  match %-5s %5ld steps: lookup %.2f, drafter %.2f accepted/step\n", names[b],
                           st.shadow_steps[b], double(st.shadow_lookup[b]) / st.shadow_steps[b],
                           double(st.shadow_dflash[b]) / st.shadow_steps[b]);
        }
      }
    }
    if (graph_check) {
      std::printf(failures ? "GRAPH CHECK FAILED\n" : "graph check passed\n");
      return failures ? 1 : 0;
    }
    if (spec_check) {
      std::printf(failures ? "SPEC CHECK FAILED\n" : "spec check passed\n");
      return failures ? 1 : 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
