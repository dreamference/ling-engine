// ling_force: the engine side of the agreement study (bench/agreement/run.sh), built by hand against any
// ling-engine build's libraries so older builds can be scored too:
//   nvcc -std=c++20 -O2 -gencode arch=compute_121a,code=sm_121a -I$SRC/src ling_force.cpp \
//        $SRC/build/libling_core.a $SRC/build/libling_tokenizer.a -lcublas -lpcre2-8 -licuuc -o ling-force
//   ling-force tokenize MODEL < prompts.jsonl            -> {"ids": [...]} per line (prompt strings, JSON)
//   ling-force score MODEL K < cases.jsonl               -> per case: engine top-K log-probs at each forced position
//        case: {"prompt": [ids], "cont": [ids]}; position j scores the token after prompt + cont[:j]
//   ling-force gen MODEL MAXTOK < cases.jsonl            -> greedy continuation of {"prompt": [ids]}: {"ids", "text"}
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/engine.hpp"
#include "tokenizer/tokenizer.hpp"

using json = nlohmann::json;

static json topk(const std::vector<float>& logits, int K, const std::vector<int>& extra) {
  const int V = static_cast<int>(logits.size());
  double mx = -1e30;
  for (float v : logits) mx = std::max(mx, double(v));
  double sum = 0;
  for (float v : logits) sum += std::exp(double(v) - mx);
  const double lse = mx + std::log(sum);
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                    [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
  json t = json::array();
  for (int i = 0; i < K; ++i) t.push_back({idx[i], double(logits[idx[i]]) - lse});
  json e = json::object();
  for (int id : extra)
    if (id >= 0 && id < V) e[std::to_string(id)] = double(logits[id]) - lse;
  return {{"top", t}, {"lp", e}};
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const std::string mode = argv[1], model = argv[2];
  ling::Tokenizer tok(model + "/tokenizer.json");
  std::string line;
  if (mode == "tokenize") {
    while (std::getline(std::cin, line))
      if (!line.empty()) std::cout << json{{"ids", tok.encode(json::parse(line).get<std::string>())}}.dump() << "\n";
    return 0;
  }
  ling::EngineOptions opts;
  opts.max_context = 40000;
  ling::Engine engine(model, opts);
  const int im_end = tok.token_id("<|im_end|>");
  if (mode == "score") {
    const int K = std::stoi(argv[3]);
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const json c = json::parse(line);
      const std::vector<int> prompt = c["prompt"], cont = c["cont"];
      // Production's top-k ids at each position (optional), so the engine's log-prob of each is reported.
      const json want = c.value("want", json::array());
      engine.reset();
      const std::vector<float>* logits = &engine.prefill(prompt);
      json out = json::array();
      for (size_t j = 0; j < cont.size(); ++j) {
        std::vector<int> extra = {cont[j]};
        if (j < want.size())
          for (const auto& w : want[j]) extra.push_back(w.get<int>());
        out.push_back(topk(*logits, K, extra));
        if (j + 1 < cont.size()) logits = &engine.step(cont[j]);
      }
      std::cout << json{{"pos", out}}.dump() << "\n" << std::flush;
    }
    return 0;
  }
  if (mode == "gen") {
    const int maxtok = std::stoi(argv[3]);
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::vector<int> prompt = json::parse(line)["prompt"];
      engine.reset();
      const std::vector<float>* logits = &engine.prefill(prompt);
      std::vector<int> out;
      for (int i = 0; i < maxtok; ++i) {
        const int t = static_cast<int>(std::max_element(logits->begin(), logits->end()) - logits->begin());
        out.push_back(t);
        if (t == im_end || t == engine.config().eos_token) break;
        logits = &engine.step(t);
      }
      std::cout << json{{"ids", out}, {"text", tok.decode(out)}}.dump() << "\n" << std::flush;
    }
    return 0;
  }
  return 2;
}
