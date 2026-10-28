// P4b (specs/DREAMFERENCE_LING_ENGINE_BATCHING.md §5.3, "A full pool"): what ling-serve's scheduler does when a request
// wants KV pages, as arithmetic on page counts, so the policy is tested on the host (tests/kv_pool_tests.cpp) apart
// from the engine and the HTTP server.
//
//  1. Admission: a request's prompt pages plus two of growth (Engine::kv_admit_need) are taken when it is admitted; a
//     request that does not fit waits in the queue (never refused for it: HTTP 503 stays the host-memory rule).
//  2. Reclaim, cheapest first: pages only the prefix KV store holds (none in P4b, where the store keeps its own copies
//     in its arena), then idle sessions' pages, least recently used first, the pool's need overriding
//     --session-protect-seconds. Nothing is evicted unless that makes enough pages.
//  3. Decode growth: a running request's step that needs a page waits while the others run.
//  4. The last resort: when every running request waits for a page and nothing is pending or reclaimable, the request
//     admitted last ends as if it had reached the context limit (finish_reason "length") and its pages are freed.
#pragma once

#include <cstddef>
#include <vector>

namespace ling::serve {

// The pool as the scheduler sees it, in pages: free now; released but free only once the streams that could read
// them have passed the release (the next step boundary); held by idle sessions (no request running in them).
struct KvPoolView {
  int free = 0, pending = 0, idle = 0;
};

enum class KvVerdict {
  Go,       // the pages are free: admit, or step
  Retire,   // they will be after the deferred frees retire: wait a step (or synchronize when nothing runs)
  Reclaim,  // evicting idle sessions, least recently used first, makes enough: evict, then as Retire
  Wait,     // not even that: wait for a running request to end
};

inline KvVerdict kv_verdict(int need, const KvPoolView& v) {
  if (need <= v.free) return KvVerdict::Go;
  if (need <= v.free + v.pending) return KvVerdict::Retire;
  if (need <= v.free + v.pending + v.idle) return KvVerdict::Reclaim;
  return KvVerdict::Wait;
}

// The idle sessions to evict for `need` pages, least recently used first: the shortest such prefix of `lru` (sessions
// in that order, with the pages each holds) that brings free + pending up to need; empty when even all would not.
inline std::vector<int> kv_victims(int need, const KvPoolView& v, const std::vector<int>& lru,
                                   const std::vector<int>& pages) {
  std::vector<int> out;
  int have = v.free + v.pending;
  if (need <= have) return out;
  for (size_t i = 0; i < lru.size() && have < need; ++i) {
    if (pages[i] <= 0) continue;
    out.push_back(lru[i]);
    have += pages[i];
  }
  if (have < need) out.clear();
  return out;
}

// A running request as the last resort sees it: when it was admitted (a counter), whether its next unit waits for
// KV pages (a decode step whose pages are not there), and the pages that step needs.
struct KvRunner {
  long admitted = 0;
  bool waiting = false;
  int need = 0;
};

// The index of the request to end (finish_reason "length"), or -1: only when every running request waits for pages
// and none of them could get its pages from the free, pending or idle ones.
inline int kv_last_resort(const std::vector<KvRunner>& running, const KvPoolView& v) {
  if (running.empty()) return -1;
  int last = -1;
  for (size_t i = 0; i < running.size(); ++i) {
    const KvRunner& r = running[i];
    if (!r.waiting || kv_verdict(r.need, v) != KvVerdict::Wait) return -1;
    if (last < 0 || r.admitted > running[last].admitted) last = static_cast<int>(i);
  }
  return last;
}

}  // namespace ling::serve
