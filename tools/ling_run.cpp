// ling-run: loads the model and generates from a prompt; prints tokens, timings and (with --ids) the
// token ids, for checking against a reference server.
//
//   ling-run --model DIR [--text "..." | --chat "..." | --ids 1,2,3] [--max-tokens N] [--temperature T]
//            [--prompts-file F] [--ids-out] [--prompt-ids-out] [--max-context N]
//            [--draft DIR [--block N] [--spec] [--spec-check] [--lookup 0|1|2] [--lookup-min N]]
//
// --spec decodes with the DFlash2 drafter. --spec-check (greedy) decodes each prompt plainly and then
// speculatively, and checks that the tokens are identical and that the recurrent state after the
// speculative run equals the state of a plain run over the same tokens, bit for bit.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/engine.hpp"
#include "tokenizer/chat_template.hpp"
#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model, text, chat, ids_arg, prompts_file;
  int max_tokens = 64, max_context = 32768;
  float temperature = 0.f;
  bool ids_out = false, prompt_ids_out = false, spec = false, spec_check = false;
  std::string draft;
  int block = 16, lookup = 1, lookup_min = 8;
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
    for (const std::vector<int>& prompt : prompts) {
      ling::SamplingParams sp;
      sp.temperature = temperature;
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
