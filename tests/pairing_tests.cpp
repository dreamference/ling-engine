// Fair pairing (serve/pairing.hpp, C3 design A1) and the step trace's line (serve/step_trace.hpp); host only.
// The simulation takes run()'s turns as the scheduler does: the turn goes round the running requests, a batchable
// one steps with the partners select_batch picks, a decoding one that cannot batch steps alone, a prefilling one
// runs a chunk (no decode pass).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "serve/pairing.hpp"
#include "serve/step_trace.hpp"

namespace {

using ling::serve::PairCandidate;
using ling::serve::PassWindow;

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAILED: %s\n", what);
    ++failures;
  }
}

enum class Kind { Batchable, Plain, Prefilling };

struct Sim {
  std::vector<Kind> kinds;
  bool fair;
  int fit = 2;
  std::vector<PassWindow> windows;
  std::vector<long> passes;  // per request, over the counted turns
  long pass = 0;
  size_t turn = 0;
  bool slowest_served = true;  // every batched pass held a request with the fewest recent passes

  Sim(std::vector<Kind> k, bool f) : kinds(std::move(k)), fair(f), windows(kinds.size()), passes(kinds.size(), 0) {}

  void step(bool count) {
    const size_t k = turn++ % kinds.size();
    std::vector<PairCandidate> c(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) c[i] = {kinds[i] == Kind::Batchable, windows[i].count(pass), windows[i].last()};
    std::vector<size_t> b = ling::serve::select_batch(c, k, fit, fair);
    if (b.empty() && kinds[k] == Kind::Plain) b = {k};
    if (b.empty()) return;  // a prefill chunk
    if (b.size() >= 2) {
      int least = 1 << 30, in = 1 << 30;
      for (size_t i = 0; i < c.size(); ++i)
        if (c[i].batchable) least = std::min(least, c[i].recent);
      for (size_t i : b) in = std::min(in, c[i].recent);
      if (in != least) slowest_served = false;
    }
    ++pass;
    for (size_t i : b) {
      windows[i].record(pass);
      if (count) ++passes[i];
    }
  }
  // Warms up for `warm` turns, then counts `turns` turns.
  std::vector<long> run(int warm, int turns) {
    for (int i = 0; i < warm; ++i) step(false);
    for (int i = 0; i < turns; ++i) step(true);
    return passes;
  }
};

bool same(const std::vector<long>& a, const std::vector<long>& b) { return a == b; }
// Within one pass of `b` each (where the window's edge falls on the counted turns moves a pass between requests).
bool even(const std::vector<long>& a, const std::vector<long>& b) {
  if (std::getenv("PAIRING_VERBOSE")) {
    for (long x : a) std::printf("%ld ", x);
    std::printf("\n");
  }
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i] < b[i] - 1 || a[i] > b[i] + 1) {
      for (long x : a) std::printf("%ld ", x);
      std::printf("\n");
      return false;
    }
  return true;
}

void three_decoders_one_prefilling() {
  const std::vector<Kind> k = {Kind::Batchable, Kind::Batchable, Kind::Batchable, Kind::Prefilling};
  // Per cycle of 4 turns: 3 decode passes. Legacy: 3/2/1 per 3 passes; fair: 2/2/2.
  expect(same(Sim(k, false).run(40, 40), {30, 20, 10, 0}), "legacy, 3 decoders + 1 prefilling: 3/2/1");
  Sim fair(k, true);
  expect(even(fair.run(40, 400), {200, 200, 200, 0}), "fair, 3 decoders + 1 prefilling: 2/2/2");
  expect(fair.slowest_served, "fair, 3 decoders: the slowest is in every batched pass");
}

void four_decoders() {
  const std::vector<Kind> k(4, Kind::Batchable);
  expect(same(Sim(k, false).run(40, 40), {40, 20, 10, 10}), "legacy, 4 decoders: 4/2/1/1");
  Sim fair(k, true);
  expect(even(fair.run(40, 400), {200, 200, 200, 200}), "fair, 4 decoders: 2/2/2/2");
  expect(fair.slowest_served, "fair, 4 decoders: the slowest is in every batched pass");
}

void never_pairs() {
  // A plain decoder steps alone on its turn and is never a partner; a prefilling request is not a candidate.
  const std::vector<Kind> k = {Kind::Batchable, Kind::Plain, Kind::Prefilling, Kind::Batchable};
  for (bool f : {false, true}) {
    std::vector<PairCandidate> c = {{true, 0}, {false, 0}, {false, 0}, {true, 5}};
    const auto b = ling::serve::select_batch(c, 0, 2, f);
    expect(b == std::vector<size_t>({0, 3}), "the only batchable partner is chosen, whatever its passes");
    expect(ling::serve::select_batch(c, 1, 2, f).empty(), "a plain decoder forms no batch");
    expect(ling::serve::select_batch(c, 2, 2, f).empty(), "a prefilling request forms no batch");
    Sim s(k, f);
    const auto p = s.run(40, 40);
    expect(p[1] == 10 && p[2] == 0, "the plain decoder steps once per cycle, alone; the prefilling one never");
  }
}

void ties_and_turn() {
  std::vector<PairCandidate> c = {{true, 3}, {true, 1}, {true, 1}, {true, 0}};
  // The request whose turn it is stays first even with the most passes.
  expect(ling::serve::select_batch(c, 0, 2, true) == std::vector<size_t>({0, 3}), "fair: the fewest passes joins");
  expect(ling::serve::select_batch(c, 3, 2, true) == std::vector<size_t>({3, 1}), "fair: a tie goes in running order");
  expect(ling::serve::select_batch(c, 0, 2, false) == std::vector<size_t>({0, 1}), "legacy: running order");
  expect(ling::serve::select_batch(c, 2, 3, true) == std::vector<size_t>({2, 3, 1}), "fit 3: two partners");
  std::vector<PairCandidate> o = {{true, 2, 9}, {true, 0, 0}, {true, 2, 9}, {true, 0, 0}, {true, 2, 7}};
  expect(ling::serve::pair_order(o, true) == std::vector<size_t>({1, 3, 4, 0, 2}), "fewest, then oldest, then order");
}

void window() {
  PassWindow w;
  for (long p = 1; p <= 40; p += 2) w.record(p);  // every other pass
  expect(w.count(40) == 8, "8 of the trailing 16 passes");
  expect(w.count(60) == 0, "nothing after the window has passed");
  expect(w.marks.size() <= 8, "the window is trimmed");
}

void trace_line() {
  ling::serve::StepRecord s;
  s.t = 1.5;
  s.pass = 42;
  s.kind = "decode";
  s.turn = 3;
  s.sessions = {0, 2};
  s.rows = {16, 16};
  s.tokens = {4, 7};
  s.left = {0, 0};
  s.ms = 12.25;
  s.side = true;
  const std::string want =
      "{\"t\":1.500000,\"pass\":42,\"kind\":\"decode\",\"turn\":3,\"sessions\":[0,2],\"rows\":[16,16],"
      "\"tokens\":[4,7],\"left\":[0,0],\"ms\":12.250,\"side\":true}";
  const std::string got = ling::serve::format_step(s);
  if (got != want) std::printf("got  %s\nwant %s\n", got.c_str(), want.c_str());
  expect(got == want, "one trace line");
  // A decode line with the partner choice's view (`cand`); bench/agentlong/fair_gate_test.py parses this literal.
  s.cand = {{3, 11, 2, 1, 40}, {1, 9, 0, 0, 0}, {2, 8, 1, 2, 30}, {0, 7, 1, 3, 41}};
  const std::string want_cand =  // trace line with cand
      "{\"t\":1.500000,\"pass\":42,\"kind\":\"decode\",\"turn\":3,\"sessions\":[0,2],\"rows\":[16,16],"
      "\"tokens\":[4,7],\"left\":[0,0],\"cand\":[[3,11,2,1,40],[1,9,0,0,0],[2,8,1,2,30],[0,7,1,3,41]],"
      "\"ms\":12.250,\"side\":true}";
  const std::string got_cand = ling::serve::format_step(s);
  if (got_cand != want_cand) std::printf("got  %s\nwant %s\n", got_cand.c_str(), want_cand.c_str());
  expect(got_cand == want_cand, "one trace line with the candidates");
}

}  // namespace

int main() {
  three_decoders_one_prefilling();
  four_decoders();
  never_pairs();
  ties_and_turn();
  window();
  trace_line();
  if (failures) {
    std::printf("%d pairing test(s) failed\n", failures);
    return 1;
  }
  std::printf("pairing tests passed\n");
  return 0;
}
