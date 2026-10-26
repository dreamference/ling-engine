// ling-prefix-kv-tests: the prefix KV store's tree on the host (BACKLOG 35): match, insert, splits at cuts and at a
// match inside a run, the segments a restore copies, states, the budget's leaf-first LRU eviction (never what the
// current prefill stamped) and the release of a buffer only when its last node goes. Buffers are host memory filled
// by `store` with each position's own id, so a restore's segments are checked position by position.
#include "core/prefix_kv.hpp"

#include <cstdio>
#include <cstdlib>
#include <map>
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
  std::printf(failures ? "prefix kv tests: %d FAILED\n" : "all prefix kv tests passed\n", failures);
  return failures ? 1 : 0;
}
