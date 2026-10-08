// ling-run: loads the model and generates from a prompt; prints tokens, timings and (with --ids) the
// token ids, for checking against a reference server.
//
//   ling-run --model DIR [--text "..." | --chat "..." | --ids 1,2,3] [--max-tokens N] [--temperature T]
//            [--prompts-file F] [--ids-out] [--prompt-ids-out] [--max-context N]
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
  bool ids_out = false, prompt_ids_out = false;
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
    ling::Engine engine(model, opts);
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "loaded %.1f GB of weights in %.1f s\n", engine.model().device_bytes() / 1e9,
                 std::chrono::duration<double>(t1 - t0).count());

    for (const std::vector<int>& prompt : prompts) {
      ling::EngineStats stats;
      ling::SamplingParams sp;
      sp.temperature = temperature;
      engine.reset();
      std::vector<float> logits = engine.prefill(prompt, &stats);
      std::fprintf(stderr, "prefill: %d tokens in %.3f s (%.0f tok/s)\n", stats.prefill_tokens,
                   stats.prefill_seconds, stats.prefill_tokens / stats.prefill_seconds);
      ling::StreamDecoder dec(tok);
      std::vector<int> out;
      auto t2 = std::chrono::steady_clock::now();
      for (int n = 0; n < max_tokens; ++n) {
        int next = engine.sample(logits, sp, engine.history());
        out.push_back(next);
        if (next == engine.config().eos_token || next == tok.token_id("<|im_end|>")) break;
        if (!ids_out) std::cout << dec.push(next) << std::flush;
        logits = engine.step(next);
      }
      auto t3 = std::chrono::steady_clock::now();
      if (prompt_ids_out) {
        for (size_t i = 0; i < prompt.size(); ++i) std::cout << (i ? "," : "") << prompt[i];
        std::cout << "\n";
      }
      if (ids_out) {
        for (size_t i = 0; i < out.size(); ++i) std::cout << (i ? "," : "") << out[i];
        std::cout << "\n";
      } else {
        std::cout << dec.flush() << "\n";
      }
      const double dt = std::chrono::duration<double>(t3 - t2).count();
      std::fprintf(stderr, "decode: %zu tokens in %.3f s (%.2f tok/s)\n", out.size(), dt, out.size() / dt);
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
