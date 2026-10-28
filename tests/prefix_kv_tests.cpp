// ling-prefix-kv-tests: the prefix KV store's tree on the host (BACKLOG 35): match, insert, splits at cuts and at a
// match inside a run, the segments a restore copies, states, the budget's leaf-first LRU eviction (never what the
// current prefill stamped) and the release of a buffer only when its last node goes. Buffers are host memory filled
// by `store` with each position's own id, so a restore's segments are checked position by position. Paged mode (P4c):
// the tree's page references against a real KvPool, splits that share the straddling page, eviction, share scopes,
// a failed hold, and random sessions whose every restored position is checked against its own prompt.
#include "core/prefix_kv.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace {

int failures = 0;
#define EXPECT(c)                                                       \
  do {                                                                  \
    if (!(c)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

constexpr size_t kTok = 4;  // bytes per token: one int, the position's tag
constexpr size_t kState = 64;

struct Host {
  std::set<void*> live;
  std::vector<int> session;  // the active session's KV: position p holds session[p]
  ling::PrefixKvTree::Ops ops() {
    ling::PrefixKvTree::Ops o;
    o.alloc = [this](size_t n) {
      void* p = std::malloc(n);
      live.insert(p);
      return p;
    };
    o.release = [this](void* p) {
      EXPECT(live.count(p) == 1);
      live.erase(p);
      std::free(p);
    };
    o.store = [this](void* buf, int buf_start, int buf_len, int from, int to) {
      EXPECT(from >= buf_start && to <= buf_start + buf_len);
      int* b = static_cast<int*>(buf);
      for (int p = from; p < to; ++p) b[p - buf_start] = session[p];
    };
    return o;
  }
};

std::vector<int> prompt(std::vector<int> head, int tail_tag, int n) {
  for (int i = 0; i < n; ++i) head.push_back(tail_tag * 1000 + i);
  return head;
}

// The KV a restore of [0, n) would read, position by position.
std::vector<int> restored(const ling::PrefixKvTree& t, const ling::PrefixKvTree::Match& m, int n) {
  std::vector<int> out(n, -1);
  for (const auto& s : t.segments(m, n))
    for (int p = s.from; p < s.to; ++p) out[p] = static_cast<const int*>(s.buf)[p - s.buf_start];
  return out;
}


// Paged mode (P4c): a real KvPool, and each page's contents on the host: the KV at a position is a hash of the whole
// prompt up to it (the KV depends on every earlier token), so a restore mapping a page another prompt wrote is caught.
constexpr int PT = ling::KvPool::kPageTokens;

std::vector<int> sys_prefix(int n) {
  std::vector<int> v;
  for (int i = 0; i < n; ++i) v.push_back(i);
  return v;
}

uint64_t kv_at(const std::vector<int>& prompt, int pos) {
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i <= pos; ++i) h = (h ^ static_cast<uint64_t>(prompt[i] + 7)) * 1099511628211ull;
  return h;
}

struct Paged {
  ling::KvPool pool{400};
  std::map<int, std::vector<uint64_t>> mem;  // page id -> its PT positions
  std::vector<int> table;                    // the active session's page table
  ling::PrefixKvTree::Ops ops() {
    ling::PrefixKvTree::Ops o;
    o.alloc = [](size_t n) { return std::malloc(n); };
    o.release = [](void* p) { std::free(p); };
    o.hold = [this](int from, int to) {
      std::vector<int> out;
      for (int i = ling::KvPool::index_of(from); i <= ling::KvPool::index_of(to - 1); ++i) {
        const int pg = i < static_cast<int>(table.size()) ? table[i] : 0;
        if (pg) pool.retain(pg);
        out.push_back(pg);
      }
      return out;
    };
    o.retain = [this](int pg) { pool.retain(pg); };
    o.drop = [this](int pg) { pool.release(pg); };
    return o;
  }
  // A session that runs `prompt`: maps the store's pages for [0, n), copies the page n falls in when shared, writes
  // [n, L) into private pages only.
  void run(const std::vector<int>& prompt, const std::vector<int>& from_store, int n) {
    for (int pg : table)
      if (pg) pool.release(pg);
    table.clear();
    for (int i = 0; i < static_cast<int>(from_store.size()); ++i) {
      pool.retain(from_store[i]);
      table.push_back(from_store[i]);
    }
    std::vector<ling::KvPool::Copy> copies;
    const int L = static_cast<int>(prompt.size());
    if (!pool.ensure(table, n, L, &copies)) {
      EXPECT(false);
      return;
    }
    for (const auto& c : copies) mem[c.dst] = mem[c.src];
    for (int pos = n; pos < L; ++pos) {
      const int pg = table[ling::KvPool::index_of(pos)];
      EXPECT(pool.refs(pg) == 1);  // a shared page is never written
      std::vector<uint64_t>& m = mem[pg];
      m.resize(PT, 0);
      m[pos % PT] = kv_at(prompt, pos);
    }
  }
  void end_session() {
    for (int pg : table)
      if (pg) pool.release(pg);
    table.clear();
  }
  int live_pages() const {
    int n = 0;
    for (int pg = 1; pg <= pool.capacity(); ++pg) n += pool.refs(pg) > 0;
    return n;
  }
};

void paged_tests() {
  using ling::KvPool;
  using ling::PrefixKvTree;
  // Insert, then pages() is the session's own table.
  {
    Paged h;
    PrefixKvTree t(2 * kState, 0, kState, h.ops());
    EXPECT(t.paged());
    std::vector<int> a = prompt({}, 1, 150);
    h.run(a, {}, 0);
    const std::vector<int> ta = h.table;
    t.insert(a, 150, {}, 1);
    EXPECT(t.nodes() == 1 && t.pages_held() == 3 && t.bytes() == 0 && t.buffers() == 0);
    const auto m = t.match(a);
    EXPECT(m.len == 150 && t.pages(m, 150) == ta && t.pages(m, 64) == std::vector<int>{ta[0]});
    EXPECT(t.segments(m, 150).empty());
    for (int pg : ta) EXPECT(h.pool.refs(pg) == 2);
    // A split mid-page (a cut at 90): the straddling page 1 is held by both halves.
    t.insert(a, 150, {90}, 2);
    EXPECT(t.nodes() == 2 && h.pool.refs(ta[1]) == 3 && h.pool.refs(ta[0]) == 2 && h.pool.refs(ta[2]) == 2);
    EXPECT(t.pages(t.match(a), 150) == ta);
    // A split on a page boundary (128) shares no page.
    t.insert(a, 150, {128}, 3);
    EXPECT(t.nodes() == 3 && h.pool.refs(ta[1]) == 3 && h.pool.refs(ta[2]) == 2 && t.pages_held() == 4);
    // Evicting leaves gives back exactly their references.
    EXPECT(t.evict_lru_leaf());  // [128, 150)
    EXPECT(t.nodes() == 2 && h.pool.refs(ta[2]) == 1);
    EXPECT(t.evict_lru_leaf() && t.evict_lru_leaf() && !t.evict_lru_leaf());
    for (int pg : ta) EXPECT(h.pool.refs(pg) == 1);
    EXPECT(t.pages_held() == 0 && t.nodes() == 0);
    h.end_session();
    EXPECT(h.live_pages() == 0);
  }
  // Two prompts that diverge mid-page: each restore gets its own copy of the straddling page.
  {
    Paged h;
    PrefixKvTree t(0, 0, kState, h.ops());
    std::vector<int> a = prompt(sys_prefix(100), 1, 60), b = prompt(sys_prefix(100), 2, 60);
    h.run(a, {}, 0);
    t.insert(a, 160, {}, 1);
    const auto mb = t.match(b);
    EXPECT(mb.len == 100);
    h.run(b, t.pages(mb, 100), 100);  // shares page 0, copies page 1
    t.insert(b, 160, {}, 2);
    EXPECT(t.nodes() == 3);
    for (const auto& pr : {a, b}) {
      const auto m = t.match(pr);
      EXPECT(m.len == 160);
      const std::vector<int> pg = t.pages(m, 160);
      for (int pos = 0; pos < 160; ++pos) EXPECT(h.mem[pg[pos / PT]][pos % PT] == kv_at(pr, pos));
    }
    EXPECT(t.pages(t.match(a), 160)[1] != t.pages(t.match(b), 160)[1]);
    EXPECT(t.pages(t.match(a), 160)[0] == t.pages(t.match(b), 160)[0]);
    h.end_session();
    t.clear();
    EXPECT(h.live_pages() == 0 && t.pages_held() == 0);
  }
  // The budget holds states only: a long run costs nothing, the third state does not fit.
  {
    Paged h;
    PrefixKvTree t(2 * kState, 0, kState, h.ops());
    std::vector<int> a = prompt({}, 3, 300);
    h.run(a, {}, 0);
    const auto want = t.insert(a, 300, {100, 200, 300}, 1);
    EXPECT(t.nodes() == 3 && t.bytes() == 0 && want.size() == 3);
    EXPECT(t.add_state(want[0].first, 1) && t.add_state(want[1].first, 1) && !t.add_state(want[2].first, 1));
    EXPECT(t.bytes() == 2 * kState);
    h.end_session();
  }
  // Scopes: the same tokens under two scopes are two runs, never matched across.
  {
    Paged h;
    PrefixKvTree t(0, 0, kState, h.ops());
    std::vector<int> a = prompt({}, 4, 100);
    h.run(a, {}, 0);
    t.insert(a, 100, {}, 1, 1);
    EXPECT(t.match(a, 1).len == 100 && t.match(a, 2).len == 0 && t.match(a).len == 0);
    t.insert(a, 100, {50}, 2, 2);
    EXPECT(t.nodes() == 3 && t.match(a, 2).len == 100 && t.match(a, 1).path.size() == 1);
    EXPECT(t.match(a, 2).path.size() == 2);
    h.end_session();
    t.clear();
    EXPECT(h.live_pages() == 0);
  }
  // A hold that fails (an unmapped page) stores nothing past it and leaks no reference.
  {
    Paged h;
    PrefixKvTree t(0, 0, kState, h.ops());
    std::vector<int> a = prompt({}, 5, 200);
    h.run(a, {}, 0);
    h.pool.release(h.table[2]);
    h.table[2] = 0;
    t.insert(a, 200, {64}, 1);
    EXPECT(t.nodes() == 1 && t.match(a).len == 64 && h.pool.refs(h.table[1]) == 1);
    h.end_session();
    t.clear();
    EXPECT(h.live_pages() == 0);
  }
  // evict_lru_leaf takes the oldest leaf first, whatever its stamp.
  {
    Paged h;
    PrefixKvTree t(0, 0, kState, h.ops());
    std::vector<int> a = prompt({}, 6, 70), b = prompt({}, 7, 70);
    h.run(a, {}, 0);
    t.insert(a, 70, {}, 5);
    h.run(b, {}, 0);
    t.insert(b, 70, {}, 9);
    EXPECT(t.evict_lru_leaf() && t.match(a).len == 0 && t.match(b).len == 70);
    h.end_session();
  }
  // Random sessions over a family of prompts with shared prefixes and two scopes: every restore reads exactly the
  // KV of its own prompt, no shared page is written, and the references all come back.
  {
    Paged h;
    PrefixKvTree t(64 * kState, 0, kState, h.ops());
    std::mt19937 rng(5);
    std::vector<std::vector<int>> family;
    for (int f = 0; f < 6; ++f) family.push_back(prompt(sys_prefix(37 + 41 * f), 10 + f, 400));
    int restored_positions = 0;
    for (uint64_t epoch = 1; epoch <= 600; ++epoch) {
      // The pool's first reclaim: the store's leaves (the engine's kv_make_room), then the step boundary.
      while (h.pool.free_pages() + h.pool.pending_pages() < 24 && t.evict_lru_leaf()) {
      }
      h.pool.retire();
      std::vector<int> p = family[rng() % family.size()];
      p.resize(1 + rng() % p.size());
      if (rng() % 3 == 0)
        for (int i = static_cast<int>(rng() % p.size()); i < static_cast<int>(p.size()); ++i) p[i] = 90000 + static_cast<int>(rng() % 50);
      const uint64_t scope = rng() % 2;
      const auto m = t.match(p, scope);
      int n = std::min<int>(m.len, static_cast<int>(p.size()) - 1);
      if (rng() % 4 == 0) n = n / 2;
      n = std::max(n, 0);
      const std::vector<int> pg = t.pages(m, n);
      for (int pos = 0; pos < n; ++pos) EXPECT(h.mem[pg[pos / PT]][pos % PT] == kv_at(p, pos));
      restored_positions += n;
      h.run(p, pg, n);
      std::vector<int> cuts;
      for (int k = 0; k < 3; ++k) cuts.push_back(1 + static_cast<int>(rng() % p.size()));
      std::sort(cuts.begin(), cuts.end());
      for (const auto& [node, at] : t.insert(p, static_cast<int>(p.size()), cuts, epoch, scope)) t.add_state(node, epoch);
      if (rng() % 5 == 0) t.evict_lru_leaf();
      if (rng() % 7 == 0) h.end_session();
    }
    EXPECT(restored_positions > 20000);
    h.end_session();
    EXPECT(h.live_pages() > 0 && t.pages_held() > 0);
    t.clear();
    EXPECT(h.live_pages() == 0);
  }
}

}  // namespace

int main() {
  std::vector<int> sys;
  for (int i = 0; i < 100; ++i) sys.push_back(i);
  {
    Host h;
    ling::PrefixKvTree t(1 << 20, kTok, kState, h.ops());
    // A: the shared 100 tokens, then its own 50; stored to 120 with a cut at 100.
    const std::vector<int> a = prompt(sys, 1, 50);
    h.session = a;
    auto want = t.insert(a, 120, {100, 120}, 1);
    EXPECT(want.size() == 2 && want[0].second == 100 && want[1].second == 120);
    for (auto [node, pos] : want) EXPECT(t.add_state(node, 1) != nullptr);
    EXPECT(t.nodes() == 2 && t.states() == 2 && t.buffers() == 1);
    EXPECT(t.bytes() == 120 * kTok + 2 * kState);
    // B shares the 100 and diverges: it matches 100, resumes from the state at 100, and its restore reads A's KV.
    const std::vector<int> b = prompt(sys, 2, 60);
    auto m = t.match(b);
    EXPECT(m.len == 100);
    const void* st = nullptr;
    EXPECT(t.best_state(m, 1 << 30, &st) == 100 && st != nullptr);
    EXPECT(t.best_state(m, 99, &st) == 0);  // the limit (the drafter's window) keeps it out
    EXPECT(restored(t, m, 100) == std::vector<int>(sys.begin(), sys.end()));
    // B stored to 130 with a cut at 90 (inside A's run: a split) and 100.
    h.session = b;
    want = t.insert(b, 130, {90, 100}, 2);
    EXPECT(want.size() == 1 && want[0].second == 90);
    EXPECT(t.buffers() == 2 && t.nodes() == 4);  // [0,90) [90,100) [100,120) and B's [100,130)
    m = t.match(b);
    EXPECT(m.len == 130);
    const std::vector<int> rb = restored(t, m, 130);
    EXPECT(std::vector<int>(rb.begin(), rb.end()) == std::vector<int>(b.begin(), b.begin() + 130));
    // A again: still all there, its state at 120 the deepest.
    m = t.match(a);
    EXPECT(m.len == 120 && t.best_state(m, 1 << 30, &st) == 120);
    EXPECT(restored(t, m, 120) == std::vector<int>(a.begin(), a.begin() + 120));
    // A match that ends inside a run: C shares 110 tokens of A.
    std::vector<int> c(a.begin(), a.begin() + 110);
    c.push_back(-7);
    m = t.match(c);
    EXPECT(m.len == 110 && t.best_state(m, 1 << 30, &st) == 100);
    h.session = c;
    want = t.insert(c, 111, {110}, 3);  // the split at 110 and one new token
    EXPECT(want.size() == 1 && want[0].second == 110);
    m = t.match(c);
    EXPECT(m.len == 111 && restored(t, m, 111) == c);
    EXPECT(restored(t, t.match(a), 120) == std::vector<int>(a.begin(), a.begin() + 120));
  }
  {
    // The budget: three branches of 50 tokens each beyond a shared 100, room for the trunk and two branches.
    Host h;
    const size_t budget = (100 + 50 + 50) * kTok + 0;
    ling::PrefixKvTree t(budget, kTok, kState, h.ops());
    std::vector<std::vector<int>> p;
    for (int k = 0; k < 3; ++k) p.push_back(prompt(sys, 10 + k, 60));
    h.session = p[0];
    t.insert(p[0], 150, {100}, 1);
    h.session = p[1];
    t.insert(p[1], 150, {100}, 2);
    EXPECT(t.bytes() == budget);
    EXPECT(t.match(p[0]).len == 150 && t.match(p[1]).len == 150);
    // p[0] used again (a lookup stamps it), then p[2] needs room: p[1]'s leaf goes, the trunk (shared) stays.
    t.touch(t.match(p[0]), 3);
    h.session = p[2];
    t.insert(p[2], 150, {100}, 4);
    EXPECT(t.match(p[2]).len == 150 && t.match(p[0]).len == 150 && t.match(p[1]).len == 100);
    EXPECT(t.bytes() <= budget);
    // Nothing stamped by the current prefill is evicted: a fourth branch stamped with everything else at epoch 9.
    t.touch(t.match(p[0]), 9);
    t.touch(t.match(p[2]), 9);
    auto p3 = prompt(sys, 20, 60);
    h.session = p3;
    t.insert(p3, 150, {100}, 9);
    EXPECT(t.match(p3).len == 100);  // no room: it keeps the shared trunk only
    EXPECT(t.match(p[0]).len == 150 && t.match(p[2]).len == 150);
    // States count in the budget too: none fits now.
    EXPECT(t.add_state(1, 9) == nullptr || t.bytes() <= budget);
    // A buffer goes only with its last node: p[0]'s run is one buffer split by the trunk's cut at 100 (the trunk
    // node holds positions [0,100) of the same buffer), so evicting p[0]'s leaf keeps the buffer.
    const size_t before = h.live.size();
    t.touch(t.match(p[2]), 10);
    h.session = prompt(sys, 30, 60);
    t.insert(h.session, 150, {100}, 11);  // evicts p[0]'s leaf (oldest unstamped), the trunk's buffer stays
    EXPECT(t.match(sys).len == 100);
    EXPECT(h.live.size() <= before + 1);
  }
  {
    // The arena: first fit, alignment, merging on release, refusal when no range is large enough.
    ling::ArenaAlloc a(100, 10);
    size_t x, y, z, w;
    EXPECT(a.alloc(25, &x) && x == 0);  // 30 with the alignment
    EXPECT(a.alloc(30, &y) && y == 30);
    EXPECT(a.alloc(40, &z) && z == 60);
    EXPECT(!a.alloc(10, &w));           // full
    a.release(y);
    EXPECT(!a.alloc(40, &w));           // 30 free in one piece
    a.release(x);                       // merges with y's range: 60 from 0
    EXPECT(a.alloc(60, &w) && w == 0);
    a.release(w);
    a.release(z);
    EXPECT(a.in_use() == 0 && a.alloc(100, &w) && w == 0);
  }
  paged_tests();
  std::printf(failures ? "prefix kv tests: %d FAILED\n" : "all prefix kv tests passed\n", failures);
  return failures ? 1 : 0;
}
