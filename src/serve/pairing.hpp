// Fair pairing (C3, design A1): which other requests join the batched step of the request whose turn it is.
//
// Today (legacy) the others are tried in running order, so with room for two the first running request is in every
// batched pass and the last ones wait: at four decoding conversations the passes per four turns are 4/2/1/1. With
// fair pairing the others are tried in order of the fewest decode passes in the trailing kFairWindow passes, then
// the longest since its last pass, then running order (with running order alone breaking the ties, the first
// running request still took 12% more passes than the last at four). The request whose turn it is stays first. With
// room for two, the request with the fewest recent passes is then in every batched pass (it is either the one whose
// turn it is or the first one tried), so the passes even out (2/2/2/2). Only which requests step together changes: each one's step is bit for bit its step alone.
//
// Host only: no Engine, so tests/pairing_tests.cpp checks the policy without a GPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <numeric>
#include <vector>

namespace ling::serve {

inline constexpr long kFairWindow = 16;  // the trailing decode passes counted

// The decode passes (by index) a request took part in, kept while within the trailing window.
struct PassWindow {
  std::deque<long> marks;
  // Records that the request took part in pass `pass` (the scheduler's decode-pass count after it ran).
  void record(long pass) {
    marks.push_back(pass);
    trim(pass);
  }
  // Drops the passes older than the trailing window ending at pass `now`.
  void trim(long now) {
    while (!marks.empty() && marks.front() <= now - kFairWindow) marks.pop_front();
  }
  // The last pass the request took part in (0: none in the window).
  long last() const { return marks.empty() ? 0 : marks.back(); }
  // How many of the trailing kFairWindow passes ending at `now` the request took part in.
  int count(long now) const {
    int n = 0;
    for (long m : marks)
      if (m > now - kFairWindow) ++n;
    return n;
  }
};

// One running request as the policy sees it.
struct PairCandidate {
  bool batchable = false;  // past its prompt, speculative, room for a full draft block
  int recent = 0;          // its decode passes in the trailing window
  long last = 0;           // its last decode pass (0: none in the window)
};

// The order in which the running requests are tried as partners: running order (legacy), or the fewest recent
// passes first, then the oldest last pass, then running order (fair).
inline std::vector<size_t> pair_order(const std::vector<PairCandidate>& c, bool fair) {
  std::vector<size_t> order(c.size());
  std::iota(order.begin(), order.end(), size_t{0});
  if (fair)
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return c[a].recent != c[b].recent ? c[a].recent < c[b].recent : c[a].last < c[b].last;
    });
  return order;
}

// The batch for the request at `self` (its index first, then partners in pair_order while `fit` allows), as
// Scheduler::batch_with forms it; empty when `self` cannot batch, just `self` when no partner can.
inline std::vector<size_t> select_batch(const std::vector<PairCandidate>& c, size_t self, int fit, bool fair) {
  std::vector<size_t> b;
  if (!c[self].batchable) return b;
  b.push_back(self);
  for (size_t o : pair_order(c, fair))
    if (o != self && static_cast<int>(b.size()) < fit && c[o].batchable) b.push_back(o);
  return b;
}

}  // namespace ling::serve
