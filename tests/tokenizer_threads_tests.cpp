// ling-tokenizer-threads-tests TOKENIZER_JSON TEXT_FILE: one Tokenizer shared by 16 threads that encode slices of
// a text at once (as ling-serve's HTTP workers do), compared with each slice encoded alone by a fresh Tokenizer.
// Before the cache was locked this crashed or corrupted the BPE cache (node-b, 2026-10-27).
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s TOKENIZER_JSON TEXT_FILE\n", argv[0]);
    return 2;
  }
  std::ifstream f(argv[2]);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  constexpr int kThreads = 16;
  std::vector<std::string> slices;
  const size_t n = text.size() / kThreads;
  for (int i = 0; i < kThreads; ++i) slices.push_back(text.substr(i * n, i + 1 < kThreads ? n : std::string::npos));
  std::vector<std::vector<int>> ref(kThreads);
  for (int i = 0; i < kThreads; ++i) {
    ling::Tokenizer fresh(argv[1]);
    ref[i] = fresh.encode(slices[i]);
  }
  int bad = 0;
  for (int round = 0; round < 3; ++round) {
    ling::Tokenizer shared(argv[1]);
    std::vector<std::vector<int>> got(kThreads);
    std::vector<std::thread> th;
    for (int i = 0; i < kThreads; ++i) th.emplace_back([&, i] { got[i] = shared.encode(slices[i]); });
    for (auto& t : th) t.join();
    for (int i = 0; i < kThreads; ++i) bad += got[i] != ref[i];
  }
  size_t tokens = 0;
  for (const auto& r : ref) tokens += r.size();
  std::printf("%s: %d threads x 3 rounds, %zu tokens a round, %d slices different\n", bad ? "FAILED" : "passed",
              kThreads, tokens, bad);
  return bad ? 1 : 0;
}
