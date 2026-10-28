// P4b (specs/DREAMFERENCE_LING_ENGINE_BATCHING.md §5.3, "A full pool"): what ling-serve's scheduler does when a request
// wants KV pages, as arithmetic on page counts, so the policy is tested on the host (tests/kv_pool_tests.cpp) apart
// from the engine and the HTTP server.
//
//  1. Admission: a request's prompt pages plus two of growth (Engine::kv_admit_need) are taken when it is admitted; a
//     request that does not fit waits in the queue (never refused for it: HTTP 503 stays the host-memory rule).
//  2. Reclaim, cheapest first: the prefix KV store's entries, then idle sessions, least recently used first, the
//     pool's need overriding --session-protect-seconds; an admission's own source (the store entries and the session
//     it resumes from) last, and only when the rest is not enough (kv_reclaim). P4c: pages are shared (between
//     sessions, and with the store), so what evicting frees is counted jointly (a page goes when its last holder
//     does) and the unit's need is planned again after each eviction. Nothing is evicted unless that makes enough
//     pages, except for an admission while nothing runs, whose prompt always fits once everything else is gone.
//  3. Decode growth: a running request's step that needs a page waits while the others run.
//  4. The last resort: when every running request waits for a page and nothing is pending or reclaimable, the request
//     admitted last ends as if it had reached the context limit (finish_reason "length") and its pages are freed.
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace ling::serve {

// The pool as the scheduler sees it, in pages: free now; released but free only once the streams that could read
// them have passed the release (the next step boundary); freed by evicting every idle session (no request running in
// it) and the prefix KV store's entries (P4c: counted jointly, a page shared by two of them counted once).
struct KvPoolView {
  int free = 0, pending = 0, idle = 0;
};

enum class KvVerdict {
  Go,       // the pages are free: admit, or step
  Retire,   // they will be after the deferred frees retire: wait a step (or synchronize when nothing runs)
  Reclaim,  // evicting store entries and idle sessions, least recently used first, makes enough: evict, then as Retire
  Wait,     // not even that: wait for a running request to end
};

inline KvVerdict kv_verdict(int need, const KvPoolView& v) {
  if (need <= v.free) return KvVerdict::Go;
  if (need <= v.free + v.pending) return KvVerdict::Retire;
  if (need <= v.free + v.pending + v.idle) return KvVerdict::Reclaim;
  return KvVerdict::Wait;
}

// One kind of holder to evict, least recently used first (prefix KV store leaves, idle sessions): evicts the next one
// and returns true, or false when none is left.
using KvEvictNext = std::function<bool()>;

// P4c: makes room for a unit (an admission, a decode step) by evicting holders, with pages shared between them.
// need(): the unit's need planned now (it changes as holders go: a page the unit shares with an evicted holder becomes
// its own, an evicted source makes it resume earlier); have(): the free and pending pages now. `kept`: the holders
// that may go without changing where the unit resumes from, in eviction order; `need_kept` the unit's need once all of
// them are gone and `reach_kept` have() then (pages counted jointly). `source`: the unit's own source (its store path,
// its donor); `need_all` the most the unit can need once those are gone too (a prefill from scratch, a step) and
// `reach_all` have() then. Evicts along `kept` while need() > have() when that is enough (need_kept <= reach_kept),
// else along `kept` then `source` when that is (need_all <= reach_all) or `must` (nothing else will free pages: evict
// until it fits or nothing is left); otherwise nothing. Returns need() <= have() at the end.
inline bool kv_reclaim(const std::function<int()>& need, const std::function<int()>& have,
                       const std::vector<KvEvictNext>& kept, int need_kept, int reach_kept,
                       const std::vector<KvEvictNext>& source, int need_all, int reach_all, bool must) {
  if (need() <= have()) return true;
  std::vector<const KvEvictNext*> order;
  for (const KvEvictNext& e : kept) order.push_back(&e);
  if (need_kept > reach_kept) {
    if (!must && need_all > reach_all) return false;
    for (const KvEvictNext& e : source) order.push_back(&e);
  }
  for (const KvEvictNext* e : order)
    while (need() > have() && (*e)()) {
    }
  return need() <= have();
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
