// ling-kv-pool-tests: KvPool (core/kv_pool.hpp), P4b's page bookkeeping, on the host. Allocation, copy-on-write,
// sharing, truncation, deferred frees, the all-or-nothing rule, and a random walk of holders against a model that
// tracks, for every position each holder may read, which write produced it: a holder must never read a value another
// holder wrote after sharing it (the property copy-on-write exists for). The memory plan's arithmetic with a page pool
// (serve/admission.hpp, P4b), and the scheduler's policy for a full pool (serve/kv_policy.hpp): admission, reclaim
// order, the last resort, played out against a KvPool. Exits 0 when everything holds.
#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "core/kv_pool.hpp"
#include "serve/admission.hpp"
#include "serve/kv_policy.hpp"

using ling::KvPool;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

constexpr int T = KvPool::kPageTokens;

void basics() {
  KvPool pool(9);  // 8 usable
  check(pool.capacity() == 8 && pool.free_pages() == 8, "capacity");
  std::vector<int> a;
  std::vector<KvPool::Copy> copies;
  check(pool.needed(a, 0, 130) == 3, "needed for 130 positions is 3 pages");
  check(pool.ensure(a, 0, 130, &copies) && a.size() == 3 && copies.empty(), "ensure maps 3 pages");
  for (int p : a) check(p != 0 && pool.refs(p) == 1, "fresh pages have one reference");
  check(pool.free_pages() == 5, "5 left");
  // Appending inside the last page takes nothing.
  check(pool.ensure(a, 130, 190, &copies) && pool.free_pages() == 5, "append inside the last page");
  // All or nothing: 6 more pages do not fit in 5.
  const std::vector<int> before = a;
  check(!pool.ensure(a, 190, 190 + 6 * T, &copies) && a == before && pool.free_pages() == 5, "all or nothing");
  // Truncation releases whole pages past n, but they are free only after retire().
  pool.truncate(a, 65);  // keeps pages 0 and 1 (position 64 is in page 1)
  check(a[2] == 0 && pool.pending_pages() == 1 && pool.free_pages() == 5, "truncate defers the free");
  pool.retire();
  check(pool.free_pages() == 6 && pool.pending_pages() == 0, "retire frees");
  pool.truncate(a, 0);
  pool.retire();
  check(pool.free_pages() == 8, "everything back");
}

void sharing() {
  KvPool pool(17);
  std::vector<int> a, b;
  std::vector<KvPool::Copy> copies;
  pool.ensure(a, 0, 200, &copies);  // 4 pages, the last holds 192..199
  pool.share(b, a, 150);            // b maps pages 0..2 (the third only partly: 128..149)
  check(b.size() == 3 && b[0] == a[0] && b[2] == a[2], "share maps the same pages");
  check(pool.refs(a[0]) == 2 && pool.refs(a[2]) == 2 && pool.refs(a[3]) == 1, "share takes references");
  // b writes from 150: page 2 is shared, so b gets a copy; pages 0 and 1 stay shared.
  copies.clear();
  const int old2 = a[2];
  check(pool.ensure(b, 150, 160, &copies), "ensure for the sharer");
  check(copies.size() == 1 && copies[0].src == old2 && copies[0].dst == b[2] && b[2] != old2, "copy-on-write of page 2");
  check(pool.refs(old2) == 1 && pool.refs(b[2]) == 1 && b[0] == a[0], "after the copy: page 2 private to each");
  // a writes into page 0 (a rewrite after truncation): page 0 is shared with b, so a copies too.
  copies.clear();
  pool.truncate(a, 10);
  check(pool.ensure(a, 10, 20, &copies) && copies.size() == 1 && a[0] != b[0], "the first holder copies as well");
  // Releasing b entirely frees its private pages and drops the shared ones to a.
  pool.truncate(b, 0);
  pool.retire();
  for (int p : a)
    if (p) check(pool.refs(p) == 1, "a's pages back to one reference");
  // share over a holder's own mapping is a no-op on references.
  std::vector<int> c = a;
  for (int p : a)
    if (p) pool.retain(p);
  pool.share(c, a, 10);
  check(pool.refs(a[0]) == 2, "share over the same page takes no extra reference");
}

// A random walk: holders write runs of positions, share prefixes, truncate; every write stamps the positions with a
// fresh version in the model, and every page holds, per position, the version of its last write. Each holder's view
// (through its table) must equal what it wrote or adopted.
void random_walk() {
  std::mt19937 rng(7);
  const int kPages = 24, kHolders = 4;
  KvPool pool(kPages);
  std::vector<std::vector<int>> table(kHolders);
  std::vector<std::vector<int>> want(kHolders);  // per holder: version per position it holds
  std::vector<std::vector<int>> page(kPages, std::vector<int>(T, 0));
  int version = 0, ops = 0, refused = 0;
  auto view = [&](int h, int pos) { return page[table[h][pos / T]][pos % T]; };
  for (int step = 0; step < 20000; ++step) {
    const int h = rng() % kHolders;
    const int len = static_cast<int>(want[h].size());
    switch (rng() % 4) {
      case 0:
      case 1: {  // write [from, to): an append, or a rewrite after a truncation
        const int from = (rng() % 3 == 0 && len > 0) ? static_cast<int>(rng() % len) : len;
        const int to = from + 1 + static_cast<int>(rng() % 150);
        if (from < len) {
          pool.truncate(table[h], from);
          want[h].resize(from);
        }
        std::vector<KvPool::Copy> copies;
        if (!pool.ensure(table[h], from, to, &copies)) {
          ++refused;
          break;
        }
        for (const auto& c : copies) page[c.dst] = page[c.src];
        want[h].resize(to);
        for (int p = from; p < to; ++p) page[table[h][p / T]][p % T] = want[h][p] = ++version;
        ++ops;
        break;
      }
      case 2: {  // adopt another holder's prefix
        const int src = rng() % kHolders;
        if (src == h || want[src].empty()) break;
        const int n = 1 + static_cast<int>(rng() % want[src].size());
        pool.truncate(table[h], 0);
        pool.share(table[h], table[src], n);
        want[h].assign(want[src].begin(), want[src].begin() + n);
        ++ops;
        break;
      }
      case 3:  // truncate (an eviction when n is 0)
        if (len) {
          const int n = static_cast<int>(rng() % len);
          pool.truncate(table[h], n);
          want[h].resize(n);
        }
        break;
    }
    if (rng() % 5 == 0) pool.retire();
    for (int x = 0; x < kHolders; ++x)
      for (int p = 0; p < static_cast<int>(want[x].size()); ++p)
        if (view(x, p) != want[x][p]) {
          std::printf("holder %d position %d: reads %d, wrote %d (step %d)\n", x, p, view(x, p), want[x][p], step);
          check(false, "random walk: a holder reads what it wrote");
          return;
        }
  }
  // Accounting: every page is free, pending or referenced as often as the tables map it.
  std::map<int, int> mapped;
  for (const auto& t : table)
    for (int p : t)
      if (p) ++mapped[p];
  bool counts = true;
  for (int p = 1; p < kPages; ++p) counts &= pool.refs(p) == (mapped.count(p) ? mapped[p] : 0);
  check(counts, "random walk: reference counts equal the tables' mappings");
  pool.retire();
  int used = 0;
  for (int p = 1; p < kPages; ++p) used += pool.refs(p) > 0;
  check(used + pool.free_pages() == pool.capacity(), "random walk: no page lost");
  std::printf("random walk: %d operations, %d refused for want of pages\n", ops, refused);
}

// The plan with a KV page pool (serve/admission.hpp; BATCHING §5.3 "The pool's size", with §3.5's node-b figures).
void plan() {
  using namespace ling::serve;
  ling::ModelConfig c;  // Qwen3.8-27B's shapes (MODEL §5)
  c.layers = 64;
  c.kv_heads = 4;
  c.head_dim = 256;
  c.lin_k_heads = 16;
  c.lin_v_heads = 48;
  c.lin_k_dim = 128;
  c.lin_v_dim = 128;
  c.conv_kernel = 4;
  for (int i = 0; i < 64; ++i) c.full_attention.push_back(i % 4 == 3);
  check(kv_page_bytes(c) == size_t(4) << 20, "a page is 4 MiB");
  check(kKvPageTokens == KvPool::kPageTokens, "the plan's page is the pool's");
  // A session's fixed part: twice the DeltaNet and conv state, the drafter's ring and the history, ~0.5 GB.
  const size_t fixed = session_bytes(c, 262144, true, true);
  check(session_bytes(c, 262144, true) - fixed == size_t(262144) * 65536, "the fixed part leaves out the KV only");
  check(fixed > size_t(0.45e9) && fixed < size_t(0.52e9), "a session's fixed part is ~0.5 GB");
  // node-b (§3.5): 114.7 GB available, weights 23.47 GB, 4 GB of scratch, the reserve 16 GB, the store 1 GB.
  const long avail = static_cast<long>(114.7e9 / (1 << 20)), reserve = 16384;
  const size_t weights = size_t(23.47e9), store = size_t(1024) << 20;
  check(!refuse_start(plan_memory(c, weights, 4, 262144, true), avail, reserve).empty(),
        "flat: four sessions at 262,144 are refused on node-b");
  MemoryPlan p = plan_memory(c, weights, 4, 262144, true, true);
  const long pages = plan_kv_pool(p, c, 262144, avail, reserve, store, -1);
  std::printf("plan on node-b: four sessions at 262,144, pool %s\n", describe_kv_pool(p).c_str());
  check(pages > 15800 && pages < 16000 && p.kv_pool_pages == pages, "node-b's pool is ~15,900 pages");
  check(p.kv_pool == size_t(pages + 1) * kv_page_bytes(c), "the pool's bytes count the reserved page");
  check(refuse_start(p, avail, reserve).empty() && refuse_kv_pool(p, 262144).empty(), "paged: they start");
  check(static_cast<long>((p.total() + store) >> 20) + reserve <= avail, "the pool leaves the reserve and the store");
  // Capped at sessions x max_context: a small context takes no more than it could map.
  MemoryPlan q = plan_memory(c, weights, 4, 65536, true, true);
  check(plan_kv_pool(q, c, 65536, avail, reserve, store, -1) == 4 * 1024, "capped at sessions x max_context");
  // --kv-pool-mb: that many MB of pages, refused at start when the plan with it does not fit.
  MemoryPlan r = plan_memory(c, weights, 4, 262144, true, true);
  check(plan_kv_pool(r, c, 262144, avail, reserve, store, 8192) == 2048 && refuse_start(r, avail, reserve).empty(),
        "--kv-pool-mb 8192: 2,048 pages, fits");
  MemoryPlan big = plan_memory(c, weights, 4, 262144, true, true);
  plan_kv_pool(big, c, 262144, avail, reserve, store, 80000);
  check(!refuse_start(big, avail, reserve).empty(), "--kv-pool-mb 80000 does not fit node-b: refused");
  // Nothing to judge by: every session's max_context.
  MemoryPlan u = plan_memory(c, weights, 2, 65536, true, true);
  check(plan_kv_pool(u, c, 65536, -1, reserve, store, -1) == 2048, "no MemAvailable: sessions x max_context");
  // A host that leaves less than one conversation: refused, and the pool never negative.
  MemoryPlan t = plan_memory(c, weights, 4, 262144, true, true);
  check(plan_kv_pool(t, c, 262144, 40000, reserve, store, -1) == 0 && !refuse_kv_pool(t, 262144).empty(),
        "too little memory: no pool, refused");
  MemoryPlan v = plan_memory(c, weights, 4, 262144, true, true);
  check(plan_kv_pool(v, c, 262144, 50000, reserve, store, -1) > 0 && !refuse_kv_pool(v, 262144).empty(),
        "a pool smaller than one conversation at max_context: refused");
}

// The scheduler's policy for a full pool (serve/kv_policy.hpp), on its own and played out on a KvPool.
void policy() {
  using namespace ling::serve;
  // The verdicts: free now, after the deferred frees, after evicting idle sessions, or not at all.
  const KvPoolView v{10, 5, 20};
  check(kv_verdict(10, v) == KvVerdict::Go && kv_verdict(15, v) == KvVerdict::Retire &&
            kv_verdict(35, v) == KvVerdict::Reclaim && kv_verdict(36, v) == KvVerdict::Wait,
        "verdicts: go, retire, reclaim, wait");
  // Reclaim (kv_reclaim): kept tier in order, stops once need() <= have(); source tier only when need() > reach_kept
  // and (bound <= reach_all or must); nothing evicted when need() > reach_kept, bound > reach_all and !must.
  std::vector<int> ev;
  auto reclaim = [&](int need0, int bound, bool must) {
    ev.clear();
    int cur_need = need0, cur_have = 15, i0 = 0, i1 = 0, is = 0;
    const int g0[] = {4}, g1[] = {0, 9}, gs[] = {4, 2};
    const std::vector<KvEvictNext> kept = {
        [&] {
          if (i0 >= 1) return false;
          cur_have += g0[i0++];
          ev.push_back(0);
          return true;
        },
        [&] {
          if (i1 >= 2) return false;
          cur_have += g1[i1++];
          ev.push_back(1);
          return true;
        },
    };
    const std::vector<KvEvictNext> source = {[&] {
      if (is >= 2) return false;
      cur_have += gs[is++];
      cur_need = std::max(cur_need, bound);
      ev.push_back(2);
      return true;
    }};
    return kv_reclaim([&] { return cur_need; }, [&] { return cur_have; }, kept, need0, 28, source, bound, 34, must);
  };
  check(reclaim(15, 20, false) && ev.empty(), "reclaim: no eviction when have() already covers need()");
  check(reclaim(17, 30, false) && ev == std::vector<int>{0},
        "reclaim: kept tier evicted first and stops as soon as need() <= have()");
  check(reclaim(20, 30, false) && ev == (std::vector<int>{0, 1, 1}),
        "reclaim: kept tier evicted in order, source tier untouched when kept is enough");
  check(reclaim(29, 34, false) && ev == (std::vector<int>{0, 1, 1, 2, 2}),
        "reclaim: source tier used when need() > reach_kept and bound <= reach_all");
  check(!reclaim(29, 35, false) && ev.empty(),
        "reclaim: nothing evicted when need() > reach_kept, bound > reach_all and !must");
  check(reclaim(29, 32, true) && ev == (std::vector<int>{0, 1, 1, 2}),
        "reclaim: with must=true evicts into source and stops as soon as need() <= have()");
  check(!reclaim(29, 35, true) && ev == (std::vector<int>{0, 1, 1, 2, 2}),
        "reclaim: with must=true evicts until nothing is left and returns need() <= have()");
  // The last resort: only when every running request waits and none could get its pages; the last admitted ends.
  const KvPoolView full{0, 0, 0};
  check(kv_last_resort({{1, true, 1}, {3, true, 1}, {2, true, 1}}, full) == 1, "the request admitted last ends");
  check(kv_last_resort({{1, true, 1}, {3, false, 0}}, full) == -1, "not while one still makes progress");
  check(kv_last_resort({{1, true, 1}, {2, true, 1}}, KvPoolView{0, 1, 0}) == -1, "not while pages are pending");
  check(kv_last_resort({{1, true, 1}, {2, true, 1}}, KvPoolView{0, 0, 1}) == -1, "not while an idle session holds one");
  check(kv_last_resort({}, full) == -1, "nothing to end");

  // Played out: three sessions on a pool of 12 pages; a session's prompt is taken whole plus two pages at admission,
  // each decode step needs the pages of its next rows, and an idle session's pages go back when reclaimed.
  KvPool pool(13);
  std::vector<std::vector<int>> t(3);
  std::vector<int> pos(3, 0);
  auto view = [&](int keep, const std::vector<bool>& busy) {
    std::vector<const std::vector<int>*> refs;
    for (int s = 0; s < 3; ++s)  // the LRU order here is the session index
      if (s != keep && !busy[s]) refs.push_back(&t[s]);
    return KvPoolView{pool.free_pages(), pool.pending_pages(), pool.reclaimable(refs)};
  };
  auto admit = [&](int s, int prompt, const std::vector<bool>& busy) {
    auto need = [&] { return pool.needed(t[s], 0, prompt + 2 * T); };
    auto have = [&] { return pool.free_pages() + pool.pending_pages(); };
    const KvPoolView w = view(s, busy);
    const int reach = w.free + w.pending + w.idle;
    int next = 0;
    const std::vector<KvEvictNext> kept = {[&] {
      for (; next < 3; ++next) {
        if (next == s || busy[next]) continue;
        bool held = false;
        for (int p : t[next]) held = held || p != 0;
        if (!held) continue;
        pool.truncate(t[next], 0);
        pos[next] = 0;
        pool.retire();  // nothing runs in this model: the deferred frees retire at once
        ++next;
        return true;
      }
      return false;
    }};
    if (!kv_reclaim(need, have, kept, need(), reach, {}, need(), reach, false)) return false;
    if (!pool.ensure(t[s], 0, prompt + 2 * T, nullptr)) return false;
    pos[s] = prompt;
    return true;
  };
  check(admit(0, 4 * T, {false, false, false}) && pool.free_pages() == 6, "session 0 takes 4 + 2 pages");
  check(admit(1, 3 * T, {true, false, false}) && pool.free_pages() == 1, "session 1 takes 3 + 2 pages");
  // Session 2 wants 2 + 2 pages while 0 and 1 run: it waits (no idle session to reclaim).
  check(!admit(2, 2 * T, {true, true, false}), "session 2 waits while both others run");
  // Session 1's request ends (idle): session 2 is admitted and session 1, the only idle one, is evicted for it.
  check(admit(2, 2 * T, {true, false, false}) && pool.free_pages() == 2,
        "session 2 admitted after session 1 is reclaimed");
  int held1 = 0;
  for (int p : t[1]) held1 += p != 0;
  check(held1 == 0, "the reclaimed session holds nothing");
  // Sessions 0 and 2 decode until the pool is full; then both wait and the last resort ends session 2 (admitted last).
  std::vector<KvRunner> runners = {{1, false, 0}, {3, false, 0}};
  const int sess[2] = {0, 2};
  int ended = -1, steps = 0;
  for (; steps < 1000 && ended < 0; ++steps) {
    const int i = steps % 2, s = sess[i];
    runners[i].need = pool.needed(t[s], pos[s], pos[s] + 16);
    if (kv_verdict(runners[i].need, view(-1, {true, false, true})) == KvVerdict::Go) {
      pool.ensure(t[s], pos[s], pos[s] + 16, nullptr);
      pos[s] += 5;  // a step accepts a few of its 16 rows; the rest are rewritten by the next
      runners[i].waiting = false;
    } else {
      runners[i].waiting = true;
      ended = kv_last_resort(runners, view(-1, {true, false, true}));
    }
  }
  check(ended == 1 && pool.free_pages() < 2, "a full pool ends the request admitted last");
  pool.truncate(t[2], 0);
  pool.retire();
  check(pool.needed(t[0], pos[0], pos[0] + 16) <= pool.free_pages(), "its pages let the other one go on");

  // A new prompt in a session (KvPool::net_need, Engine::kv_admit_need): the missing pages below the prompt's end + 2,
  // less the session's pages past it, which it gives back first.
  {
    KvPool p2(13);
    std::vector<int> a, b;
    p2.ensure(a, 0, 10 * T, nullptr);  // a conversation of 10 pages
    p2.ensure(b, 0, 2 * T, nullptr);   // an idle one of 2
    check(KvPool::net_need(a, 11) == 1 && KvPool::net_need(a, 4) == 0 && KvPool::net_need({}, 3) == 3,
          "net need: grows by the missing pages, a shorter prompt draws nothing");
    check(KvPool::net_need({0, 5, 0, 7}, 2) == 0 && KvPool::net_need({0, 5, 0, 7}, 3) == 1,
          "net need: holes below count, pages past are given back");
    // The conversation's next turn needs all 12 pages: its 10 stay, 2 come from evicting the idle session.
    const KvPoolView w{p2.free_pages(), p2.pending_pages(), p2.reclaimable({&b})};
    check(kv_verdict(KvPool::net_need(a, 12), w) == KvVerdict::Reclaim, "the next turn reclaims the idle session");
    p2.truncate(b, 0);
    p2.retire();
    check(p2.ensure(a, 10 * T, 12 * T, nullptr) && p2.free_pages() == 0, "and the pool is exactly full");
  }

  // P4c played out with shared pages:
  // (a) idle+idle: two idle sessions share most of their pages; each alone reclaims only its private pages, jointly
  // they reclaim all; a need that only the pair covers is met by evicting both in LRU order.
  {
    KvPool p(7);  // 6 usable pages
    std::vector<int> a, b, c;
    p.ensure(a, 0, 5 * T, nullptr);        // a maps 5 pages
    p.share(b, a, 3 * T);                  // b shares the first 3 with a
    p.ensure(b, 3 * T, 4 * T, nullptr);    // and maps 1 private page; pool is full (0 free)
    check(p.reclaimable({&a}) == 2 && p.reclaimable({&b}) == 1 && p.reclaimable({&a, &b}) == 6,
          "idle+idle: each alone reclaims only its private pages, jointly they reclaim all");
    std::vector<int> evicted;
    int next = 0;
    std::vector<int>* idle[2] = {&a, &b};
    const int reach = p.free_pages() + p.reclaimable({&a, &b});
    auto need = [&] { return p.resume_need(c, 0, 5, nullptr); };
    auto have = [&] { return p.free_pages() + p.pending_pages(); };
    const std::vector<KvEvictNext> kept = {[&] {
      if (next >= 2) return false;
      p.truncate(*idle[next], 0);
      p.retire();
      evicted.push_back(next++);
      return true;
    }};
    check(kv_reclaim(need, have, kept, 5, reach, {}, 5, reach, false) && evicted == (std::vector<int>{0, 1}),
          "idle+idle: need covered only by the pair evicts both in LRU order");
    check(p.ensure(c, 0, 5 * T, nullptr) && p.free_pages() == 1, "idle+idle: the admitted unit fits");
  }
  // (b) keep+store: the admitting session `keep` shares pages with the store past its resume point; its need falls
  // once the store is evicted (no unshare copy), fitting although the store's own joint count (excluding keep) was 0.
  {
    KvPool p(6);  // 5 usable pages
    std::vector<int> keep, store;
    p.ensure(keep, 0, 4 * T, nullptr);  // keep holds 4 pages, 1 page free
    p.share(store, keep, 4 * T);        // store shares all 4 with keep
    check(p.reclaimable({&store}) == 0, "keep+store: store's joint count excluding keep is 0");
    auto need = [&] { return p.resume_need(keep, T, 5, nullptr); };
    auto have = [&] { return p.free_pages() + p.pending_pages(); };
    check(need() == 4 && have() == 1, "keep+store: before evicting the store, unsharing 3 shared pages needs 4 pages");
    bool store_evicted = false;
    const int reach = have() + p.reclaimable({&store});
    const std::vector<KvEvictNext> kept = {[&] {
      if (store_evicted) return false;
      store_evicted = true;
      p.truncate(store, 0);
      p.retire();
      return true;
    }};
    // The gate plans the need with the store gone (its shared pages keep's own then): 1, which the free page covers,
    // so the store goes although nothing is reclaimable from it alone and `must` is false (something runs).
    const KvPool::Refs gone = {&store};
    const int need_kept = p.resume_need(keep, T, 5, nullptr, &gone);
    check(need_kept == 1, "keep+store: planned with the store gone, the need is 1");
    check(kv_reclaim(need, have, kept, need_kept, reach, {}, need_kept, reach, false) && store_evicted && need() == 1,
          "keep+store: evicting the store drops resume_need to 1 so it fits");
    p.truncate(keep, 5 * T);
    p.unshare(keep, T);
    p.retire();
    check(p.ensure(keep, T, 5 * T, nullptr) && p.free_pages() == 0, "keep+store: the unit fits without unshare copies");
  }
  // (c) running+store: pages shared by a running session and the store free nothing when the store is evicted; reach
  // is counted without them and kv_reclaim with !must returns false without evicting when nothing else is enough.
  {
    KvPool p(7);  // 6 usable pages
    std::vector<int> running, store, keep;
    p.ensure(running, 0, 4 * T, nullptr);  // running holds 4 pages
    p.share(store, running, 4 * T);        // store shares those 4 pages (1 page private to store, 1 page free)
    p.ensure(store, 4 * T, 5 * T, nullptr);
    check(p.reclaimable({&store}) == 1, "running+store: evicting the store frees none of the running session's pages");
    KvPool play = p;
    std::vector<int> play_store = store;
    play.truncate(play_store, 0);
    play.retire();
    check(play.free_pages() - p.free_pages() == 1, "running+store: truncating the store frees only its private page");
    bool store_evicted = false;
    auto need = [&] { return p.resume_need(keep, 0, 3, nullptr); };
    auto have = [&] { return p.free_pages() + p.pending_pages(); };
    const int reach = have() + p.reclaimable({&store});
    const std::vector<KvEvictNext> kept = {[&] {
      if (store_evicted) return false;
      store_evicted = true;
      p.truncate(store, 0);
      p.retire();
      return true;
    }};
    check(reach == 2 && !kv_reclaim(need, have, kept, 3, reach, {}, 3, reach, false) && !store_evicted,
          "running+store: kv_reclaim with !must refuses without evicting when reach is not enough");
  }
  // (d) donor protection: a plan resuming from a donor session's pages evicts only the other idle session when that is
  // enough (the donor survives); when it is not enough and must=true, the donor goes last, need() is re-planned as a
  // cold prefill and the prompt fits.
  {
    KvPool p(7);  // 6 usable pages
    std::vector<int> donor, other, keep;
    p.ensure(donor, 0, 4 * T, nullptr);  // donor holds 4 pages
    p.ensure(other, 0, 2 * T, nullptr);  // other idle session holds 2 pages; 0 free
    check(p.reclaimable({&other}) == 2 && p.reclaimable({&other, &donor}) == 6,
          "donor protection: kept and all reclaimable counts");
    bool other_evicted = false, donor_evicted = false;
    auto need = [&] {
      return donor_evicted ? p.resume_need(keep, 0, 6, nullptr) : p.resume_need(keep, 4 * T, 6, &donor);
    };
    auto have = [&] { return p.free_pages() + p.pending_pages(); };
    const int reach_kept = have() + p.reclaimable({&other});
    const int reach_all = have() + p.reclaimable({&other, &donor});
    const std::vector<KvEvictNext> kept = {[&] {
      if (other_evicted) return false;
      other_evicted = true;
      p.truncate(other, 0);
      p.retire();
      return true;
    }};
    const std::vector<KvEvictNext> source = {[&] {
      if (donor_evicted) return false;
      donor_evicted = true;
      p.truncate(donor, 0);
      p.retire();
      return true;
    }};
    const KvPool::Refs gone_kept = {&other};
    const int need_kept = p.resume_need(keep, 4 * T, 6, &donor, &gone_kept);
    check(need_kept == 2 && reach_kept == 2, "donor protection: with the other session gone the warm plan needs 2");
    check(kv_reclaim(need, have, kept, need_kept, reach_kept, source, 6, reach_all, false) && other_evicted &&
              !donor_evicted,
          "donor protection: only the other idle session is evicted when it suffices, donor survives");
    p.share(keep, donor, 4 * T);
    p.truncate(keep, 6 * T);
    p.unshare(keep, 4 * T);
    p.retire();
    check(p.ensure(keep, 4 * T, 6 * T, nullptr) && p.free_pages() == 0,
          "donor protection: warm prefill from the surviving donor fits");
  }
  {
    KvPool p(7);  // 6 usable pages
    std::vector<int> donor, other, keep;
    p.ensure(donor, 0, 5 * T, nullptr);  // donor holds 5 pages (keep reuses 3 of them)
    p.ensure(other, 0, 1 * T, nullptr);  // other idle session holds 1 page; 0 free
    check(p.reclaimable({&other}) == 1 && p.reclaimable({&other, &donor}) == 6,
          "donor protection (cold fallback): kept reclaims 1, all reclaims 6");
    std::vector<int> evicted;
    bool donor_gone = false;
    auto need = [&] { return donor_gone ? p.resume_need(keep, 0, 6, nullptr) : p.resume_need(keep, 3 * T, 6, &donor); };
    auto have = [&] { return p.free_pages() + p.pending_pages(); };
    const int reach_kept = have() + p.reclaimable({&other});
    const int reach_all = have() + p.reclaimable({&other, &donor});
    const std::vector<KvEvictNext> kept = {[&] {
      if (!evicted.empty()) return false;
      p.truncate(other, 0);
      p.retire();
      evicted.push_back(0);
      return true;
    }};
    const std::vector<KvEvictNext> source = {[&] {
      if (donor_gone) return false;
      donor_gone = true;
      p.truncate(donor, 0);
      p.retire();
      evicted.push_back(1);
      return true;
    }};
    const KvPool::Refs gone_kept = {&other};
    const int need_kept = p.resume_need(keep, 3 * T, 6, &donor, &gone_kept);
    check(need_kept == 3 && reach_kept == 1, "donor protection (cold fallback): the warm plan needs 3, kept frees 1");
    check(kv_reclaim(need, have, kept, need_kept, reach_kept, source, 6, reach_all, true) &&
              evicted == (std::vector<int>{0, 1}) &&
              need() == 6,
          "donor protection (cold fallback): kept not enough, must: the donor goes last, need re-plans cold");
    check(p.ensure(keep, 0, 6 * T, nullptr) && p.free_pages() == 0,
          "donor protection (cold fallback): cold prefill fits once donor is evicted");
  }
  // (e) copy-on-write of the decode page: a session at position L (L % 64 != 0) whose page index_of(L) the store also
  // references needs 1 page for the copy; after the store releases it, 0.
  {
    KvPool p(3);  // 2 usable pages
    const int L = T + 10;
    std::vector<int> s, store;
    p.ensure(s, 0, L, nullptr);  // 2 pages mapped, 0 free
    p.share(store, s, L);        // store also references page index_of(L)
    check(p.reclaimable({&store}) == 0 && p.needed(s, L, L + 1) == 1,
          "decode cow: shared page at index_of(L) needs 1 page for copy-on-write, store joint count is 0");
    bool store_evicted = false;
    auto need = [&] { return p.needed(s, L, L + 1); };
    auto have = [&] { return p.free_pages() + p.pending_pages(); };
    const int reach = have() + p.reclaimable({&store});
    const std::vector<KvEvictNext> kept = {[&] {
      if (store_evicted) return false;
      store_evicted = true;
      p.truncate(store, 0);
      p.retire();
      return true;
    }};
    const KvPool::Refs gone = {&store};
    const int need_kept = p.needed(s, L, L + 1, &gone);
    check(need_kept == 0, "decode cow: planned with the store gone, the step writes in place");
    check(kv_reclaim(need, have, kept, need_kept, reach, {}, need_kept, reach, false) && store_evicted && need() == 0,
          "decode cow: after the store is evicted, needed(s, L, L + 1) is 0");
    std::vector<KvPool::Copy> copies;
    check(p.ensure(s, L, L + 1, &copies) && copies.empty(), "decode cow: decode step writes in place without a copy");
  }
}

}  // namespace

// P4c: resume_need() against the calls it stands for, played on a copy of the pool: begin_prefill's share (a donor
// session's or the store's pages for [0, reuse)), truncate past the prompt's pages, unshare past reuse, and
// ensure(reuse, end); the free list after a retire must have shrunk by exactly the need (or not at all when the
// releases give back more), and no page the donor holds may have been written (its references stay).
void resume() {
  std::mt19937 rng(11);
  int cases = 0, shared_cases = 0;
  for (int round = 0; round < 3000; ++round) {
    KvPool pool(40);
    std::vector<std::vector<int>> t(3);
    // Random holders: each maps a run of pages, and some share another's prefix (sessions adopting, the store).
    for (int h = 0; h < 3; ++h) {
      const int n = 1 + static_cast<int>(rng() % 9) * T + static_cast<int>(rng() % T);
      if (h > 0 && rng() % 2) {
        const int src = static_cast<int>(rng() % h), k = std::min<int>(n, int(t[src].size()) * T);
        if (k > 0) pool.share(t[h], t[src], k);
      }
      pool.ensure(t[h], std::min<int>(static_cast<int>(t[h].size()) * T, n), n, nullptr);
    }
    pool.retire();
    const int self = static_cast<int>(rng() % 3), other = (self + 1 + static_cast<int>(rng() % 2)) % 3;
    const int prompt = 1 + static_cast<int>(rng() % (12 * T));
    const int pages = KvPool::pages_for(std::min(prompt + 2 * T, 14 * T));
    const bool adopt = rng() % 2 && !t[other].empty();
    int reuse = static_cast<int>(rng() % (prompt));
    if (adopt) reuse = std::min<int>(reuse, int(t[other].size()) * T);
    else reuse = std::min<int>(reuse, int(t[self].size()) * T);
    const std::vector<int> donor = t[other];
    const int need = pool.resume_need(t[self], reuse, pages, adopt ? &donor : nullptr);
    KvPool play = pool;
    std::vector<int> table = t[self];
    const int free0 = play.free_pages();
    bool any_shared = false;
    for (int p : table) any_shared = any_shared || (p && pool.refs(p) > 1);
    if (adopt && reuse > 0) play.share(table, donor, reuse);
    play.truncate(table, pages * T);
    play.unshare(table, reuse);
    play.retire();
    std::vector<KvPool::Copy> copies;
    if (!play.ensure(table, reuse, pages * T, &copies)) {
      check(need > free0, "resume: ensure refused only when the need exceeds the free pages");
      continue;
    }
    play.retire();
    const int used = free0 - play.free_pages();
    if (used != std::max(need, 0) && !(need == 0 && used < 0)) {
      std::printf("  round %d: need %d, used %d (reuse %d, pages %d, adopt %d)\n", round, need, used, reuse, pages, adopt);
      check(false, "resume: resume_need equals what the calls draw");
    }
    for (const KvPool::Copy& c : copies)
      check(KvPool::index_of(reuse) == static_cast<int>(std::find(table.begin(), table.end(), c.dst) - table.begin()),
            "resume: the only copy is the page reuse falls in");
    for (int i = 0; i < static_cast<int>(donor.size()) && adopt; ++i)
      check(play.refs(donor[i]) >= 1, "resume: the donor keeps its pages");
    ++cases;
    shared_cases += any_shared;
  }
  check(cases > 2000 && shared_cases > 500, "resume: enough cases, many with shared pages");

  // reclaimable(): a page counts only when every reference to it is listed.
  KvPool pool(10);
  std::vector<int> a, b, store;
  pool.ensure(a, 0, 3 * T, nullptr);
  pool.share(b, a, 2 * T);  // b maps a's first two pages
  store = {a[0]};
  pool.retain(a[0]);  // the store holds page 0 too
  check(pool.reclaimable({&a}) == 1, "reclaimable: a alone frees only its private page");
  check(pool.reclaimable({&a, &b}) == 2, "reclaimable: a and b free pages 1 and 2, not the store's page 0");
  check(pool.reclaimable({&a, &b, &store}) == 3, "reclaimable: with the store, every page");
  check(pool.reclaimable({&store}) == 0, "reclaimable: the store alone frees nothing the sessions map");
  // unshare(): shared pages at or past pages_for(n) dropped, private ones kept.
  std::vector<int> c;
  pool.share(c, a, 2 * T);  // a third holder of pages 0 and 1
  pool.unshare(c, T);
  check(c[0] == a[0] && c[1] == 0 && pool.refs(a[1]) == 2, "unshare: drops the shared page past n, keeps the one below");
  std::vector<int> d;
  pool.ensure(d, 0, 2 * T, nullptr);
  pool.unshare(d, 0);
  check(d[0] != 0 && d[1] != 0, "unshare: keeps private pages");
}

// P4c regression: admission with shared pages. P4b counted a new prompt's pages as KvPool::net_need (the session's
// unmapped pages below the prompt's end less its pages past it), assuming its pages below the end are its own and
// rewritten in place. Once the prefix KV store shares them, a prompt that diverges early drops the shared ones past the
// resume point (unshare) and maps fresh pages there: more than net_need. Engine::kv_admit_need now plans the resume
// point first and counts KvPool::resume_need, which this checks against the calls begin_prefill makes: a pool with
// exactly that many free pages runs them, one page fewer refuses, and so does P4b's count.
void admission() {
  const int reuse = T, prompt = 5 * T, pages = KvPool::pages_for(prompt + 2 * T);  // 7 page indices
  auto run = [&](int spare, int* need, int* p4b) {
    KvPool pool(1 + 5 + spare);
    std::vector<int> a, store;
    pool.ensure(a, 0, 5 * T, nullptr);  // the session's last prompt: 5 pages, the pool's other `spare` free
    for (int i = 0; i < 4; ++i) {       // the store holds its first four (kv_insert, paged)
      pool.retain(a[i]);
      store.push_back(a[i]);
    }
    *need = pool.resume_need(a, reuse, pages, nullptr);
    *p4b = KvPool::net_need(a, pages);
    // begin_prefill: truncate past the prompt's pages, unshare past reuse, ensure(reuse, end).
    pool.truncate(a, pages * T);
    pool.unshare(a, reuse);
    pool.retire();
    std::vector<KvPool::Copy> copies;
    const bool ok = pool.ensure(a, reuse, pages * T, &copies);
    if (ok)
      for (int i = 1; i < 4; ++i) check(a[i] != store[i] && pool.refs(store[i]) == 1, "admission: the store's pages untouched");
    return ok;
  };
  int need = 0, p4b = 0;
  check(run(64, &need, &p4b), "admission: a large pool runs the prefill");
  check(need == 5 && p4b == 2, "admission: three shared pages past the resume point are mapped afresh (5), P4b counted 2");
  check(run(need, &need, &p4b), "admission: exactly resume_need free pages suffice");
  check(!run(need - 1, &need, &p4b), "admission: one page fewer does not");
  check(!run(p4b, &need, &p4b), "admission: P4b's count would have admitted a prefill that cannot map its pages");
}

namespace {

// The calls a resume plan stands for (Engine::begin_prefill, as resume() plays them), on `pool` and `table`: share the
// donor's pages for [0, reuse), truncate past `pages`, unshare past reuse, retire, ensure(reuse, pages). False when the
// pool cannot map them.
bool play_resume(KvPool& pool, std::vector<int>& table, int reuse, int pages, const std::vector<int>* donor) {
  if (donor && reuse > 0) pool.share(table, *donor, reuse);
  pool.truncate(table, pages * T);
  pool.unshare(table, reuse);
  pool.retire();
  return pool.ensure(table, reuse, pages * T, nullptr);
}

// Releases every reference a holder's table (or the store's list) holds and empties it.
void release_all(KvPool& pool, std::vector<int>& refs) {
  for (int& p : refs)
    if (p) pool.release(p);
  refs.clear();
}

}  // namespace

// P4c follow-up: KvPool::needed and resume_need with `gone`, the references treated as released (Engine::KvGone).
void gone_counts() {
  {
    KvPool pool(10);
    std::vector<int> a, store, b;
    pool.ensure(a, 0, 3 * T, nullptr);  // a: pages 0..2
    store = {a[0], a[1], a[2]};         // the store references all three
    for (int p : store) pool.retain(p);
    check(pool.needed(a, T, 3 * T) == 2, "gone: without it, writing pages 1 and 2 shared with the store copies both");
    const KvPool::Refs g_store = {&store};
    check(pool.needed(a, T, 3 * T, &g_store) == 0, "gone: pages shared only with a gone holder count as private");
    pool.share(b, a, 2 * T);  // b shares pages 0 and 1 as well
    check(pool.needed(a, T, 3 * T, &g_store) == 1, "gone: page 1, also shared with b (not listed), still copies");
    const KvPool::Refs g_b = {&b};
    check(pool.needed(a, T, 3 * T, &g_b) == 2, "gone: b listed alone leaves both pages shared with the store");
    const KvPool::Refs g_both = {&store, &b};
    check(pool.needed(a, T, 3 * T, &g_both) == 0, "gone: page 1 named by two gone holders is subtracted twice");
    // An unmapped index counts whatever is gone.
    check(pool.needed(a, T, 4 * T, &g_both) == 1, "gone: an unmapped page index still takes a page");
    const std::vector<int> nothing;
    const KvPool::Refs g_empty = {&nothing, nullptr};
    check(pool.needed(a, T, 3 * T, &g_empty) == pool.needed(a, T, 3 * T), "gone: an empty list and a null entry change nothing");
  }
  {
    KvPool pool(16);
    std::vector<int> a, store, b;
    pool.ensure(a, 0, 5 * T, nullptr);  // a: 5 pages
    for (int i = 0; i < 4; ++i) {       // the store holds the first four
      pool.retain(a[i]);
      store.push_back(a[i]);
    }
    check(pool.resume_need(a, T, 5, nullptr) == 3, "gone: resume at T unshares the store's pages 1..3 and maps them afresh");
    const KvPool::Refs g_store = {&store};
    check(pool.resume_need(a, T, 5, nullptr, &g_store) == 0, "gone: with the store gone, a rewrites its own pages");
    pool.share(b, a, 3 * T);  // b shares pages 0..2
    check(pool.resume_need(a, T, 5, nullptr, &g_store) == 2, "gone: pages 1, 2 still shared with b are mapped afresh");
    const KvPool::Refs g_both = {&store, &b};
    check(pool.resume_need(a, T, 5, nullptr, &g_both) == 0, "gone: both holders listed, nothing is shared");
    // A donor: c resumes from b's pages; listing b as gone does not stop c from sharing them (the donor is the plan's).
    std::vector<int> c;
    pool.ensure(c, 0, 2 * T, nullptr);
    const int with_donor = pool.resume_need(c, 2 * T, 4, &b);
    check(with_donor == pool.resume_need(c, 2 * T, 4, &b, &g_store), "gone: references the plan does not touch change nothing");
  }
  // Against the calls: random holders sharing prefixes; the gone holders really released on a copy of the pool, then
  // the calls the need stands for played there: it draws exactly needed / resume_need with `gone` from the free list.
  std::mt19937 rng(29);
  int cases = 0, lowered = 0;
  for (int round = 0; round < 3000; ++round) {
    KvPool pool(48);
    std::vector<std::vector<int>> h(4);
    for (int i = 0; i < 4; ++i) {
      const int n = 1 + static_cast<int>(rng() % (8 * T));
      if (i > 0 && rng() % 3) {
        const int src = static_cast<int>(rng() % i), k = std::min<int>(n, int(h[src].size()) * T);
        if (k > 0) pool.share(h[i], h[src], k);
      }
      pool.ensure(h[i], std::min<int>(static_cast<int>(h[i].size()) * T, n), n, nullptr);
    }
    pool.retire();
    const int self = static_cast<int>(rng() % 4);
    KvPool::Refs gone;
    std::vector<int> gone_idx;
    for (int i = 0; i < 4; ++i)
      if (i != self && rng() % 2) {
        gone.push_back(&h[i]);
        gone_idx.push_back(i);
      }
    const bool step = rng() % 2;
    int need = 0, need_plain = 0, from = 0, to = 0, reuse = 0, pages = 0;
    if (step) {
      from = static_cast<int>(rng() % (int(h[self].size()) * T + 1));
      to = from + 1 + static_cast<int>(rng() % 16);
      need = pool.needed(h[self], from, to, &gone);
      need_plain = pool.needed(h[self], from, to);
    } else {
      reuse = static_cast<int>(rng() % (int(h[self].size()) * T + 1));
      pages = KvPool::pages_for(reuse + 1 + static_cast<int>(rng() % (4 * T)));
      need = pool.resume_need(h[self], reuse, pages, nullptr, &gone);
      need_plain = pool.resume_need(h[self], reuse, pages, nullptr);
    }
    check(need <= need_plain, "gone: releasing other holders never raises the need");
    lowered += need < need_plain;
    KvPool play = pool;
    std::vector<std::vector<int>> ph = h;
    for (int i : gone_idx) release_all(play, ph[i]);
    play.retire();
    const int free0 = play.free_pages();
    std::vector<KvPool::Copy> copies;
    int used = 0;
    if (step) {
      check(play.needed(ph[self], from, to) == need, "gone: needed with gone equals needed once they are released");
      if (!play.ensure(ph[self], from, to, &copies)) continue;
      used = free0 - play.free_pages();
    } else {
      if (!play_resume(play, ph[self], reuse, pages, nullptr)) {
        check(need > free0, "gone: the calls fail only when the need exceeds the free pages");
        continue;
      }
      play.retire();
      used = std::max(free0 - play.free_pages(), 0);
    }
    if (used != need) {
      std::printf("  round %d: %s need %d, used %d\n", round, step ? "step" : "resume", need, used);
      check(false, "gone: the need equals what the calls draw once the gone holders are released");
    }
    ++cases;
  }
  check(cases > 2000 && lowered > 300, "gone: enough cases, many where a gone holder lowers the need");
}

// P4c follow-up: kv_reclaim's gates at their edges, with counters for what was evicted.
void reclaim_gates() {
  using namespace ling::serve;
  int cur_need = 0, cur_have = 0, kept_left = 0, source_left = 0, kept_ev = 0, source_ev = 0;
  auto need = [&] { return cur_need; };
  auto have = [&] { return cur_have; };
  const std::vector<KvEvictNext> kept = {[&] {
    if (kept_left == 0) return false;
    --kept_left, ++kept_ev, ++cur_have;
    return true;
  }};
  const std::vector<KvEvictNext> source = {[&] {
    if (source_left == 0) return false;
    --source_left, ++source_ev, cur_have += 2;
    return true;
  }};
  auto reset = [&](int n, int h, int k, int s) {
    cur_need = n, cur_have = h, kept_left = k, source_left = s, kept_ev = source_ev = 0;
  };
  // An empty kept tier and need_kept <= reach_kept: nothing to evict; the answer is need() <= have().
  reset(3, 5, 0, 2);
  check(kv_reclaim(need, have, {}, 3, 5, source, 3, 9, false) && source_ev == 0, "gates: fits already, empty kept: true");
  reset(6, 5, 0, 2);
  check(!kv_reclaim(need, have, {}, 5, 5, source, 6, 9, false) && source_ev == 0,
        "gates: empty kept with need_kept <= reach_kept: false, the source untouched");
  check(!kv_reclaim(need, have, {}, 5, 5, source, 6, 9, true) && source_ev == 0,
        "gates: must does not reach the source while the kept gate holds");
  // need_kept > reach_kept, need_all <= reach_all, an empty source tier: the kept tier alone is evicted.
  reset(8, 5, 2, 0);
  check(!kv_reclaim(need, have, kept, 8, 7, {}, 7, 7, false) && kept_ev == 2 && cur_have == 7,
        "gates: empty source tier: the whole kept tier goes, then false");
  reset(7, 5, 3, 0);
  check(kv_reclaim(need, have, kept, 8, 7, {}, 7, 8, false) && kept_ev == 2,
        "gates: empty source tier: kept evicted until it fits, no further");
  // Both gates fail and !must: nothing is evicted, not even the kept tier.
  reset(9, 5, 2, 1);
  check(!kv_reclaim(need, have, kept, 9, 7, source, 10, 9, false) && kept_ev == 0 && source_ev == 0,
        "gates: both gates fail without must: nothing evicted");
  // must with both tiers empty: no eviction to make, the answer is need() <= have().
  reset(9, 5, 0, 0);
  check(!kv_reclaim(need, have, {}, 9, 5, {}, 9, 5, true), "gates: must with both tiers empty and short: false");
  reset(4, 5, 0, 0);
  check(kv_reclaim(need, have, {}, 9, 5, {}, 9, 5, true), "gates: must with both tiers empty and enough: true");
  // must with both gates failing: everything goes, kept before source, until it fits.
  reset(9, 5, 2, 2);
  check(kv_reclaim(need, have, kept, 9, 7, source, 10, 9, true) && kept_ev == 2 && source_ev == 1,
        "gates: must evicts kept first, then the source until it fits");
}

// P4c follow-up: a random walk of holders (sessions sharing prefixes, prefix KV store leaves holding a prefix's pages),
// with running sessions nothing may evict, and a unit (an admission resuming from a donor session or a store leaf, or
// a decode step) given kv_reclaim's tiers as Scheduler::pick gives them. Whenever kv_reclaim returns true the unit's
// calls succeed; it never returns false when evicting the whole kept tier would have made the unit fit; and with must
// it never returns false when evicting every holder would have (each played out on a copy of the pool).
void reclaim_walk() {
  using namespace ling::serve;
  std::mt19937 rng(41);
  int cases = 0, trues = 0, kept_fits = 0, source_used = 0, refused = 0;
  for (int round = 0; round < 8000; ++round) {
    const int P = 8 + static_cast<int>(rng() % 10);
    KvPool pool(P + 1);
    const int S = 3 + static_cast<int>(rng() % 3);
    std::vector<std::vector<int>> sess(S), leaves;
    std::vector<bool> running(S, false);
    for (int s = 0; s < S; ++s) {
      const int n = 1 + static_cast<int>(rng() % (6 * T));
      if (s > 0 && rng() % 2) {
        const int src = static_cast<int>(rng() % s), k = std::min<int>(n, int(sess[src].size()) * T);
        if (k > 0) pool.share(sess[s], sess[src], k);
      }
      const int from = std::min<int>(static_cast<int>(sess[s].size()) * T, n);
      if (!pool.ensure(sess[s], from, n, nullptr)) {
        pool.truncate(sess[s], from);  // the pool is full: this one keeps what it shares
        sess[s].resize(KvPool::pages_for(from));
      }
      running[s] = rng() % 4 == 0;
    }
    const int nl = static_cast<int>(rng() % 4);
    for (int l = 0; l < nl; ++l) {  // a store leaf: the first pages of a session (kv_insert holds a prefix)
      const int s = static_cast<int>(rng() % S);
      int m = 0;
      while (m < static_cast<int>(sess[s].size()) && sess[s][m]) ++m;
      if (m == 0) continue;
      m = 1 + static_cast<int>(rng() % m);
      leaves.emplace_back(sess[s].begin(), sess[s].begin() + m);
      for (int p : leaves.back()) pool.retain(p);
    }
    pool.retire();
    const int self = static_cast<int>(rng() % S);
    running[self] = false;
    const bool step = rng() % 3 == 0;
    // The plan's source: another session (a donor) or a store leaf, for an admission only.
    int donor_s = -1, donor_l = -1, reuse = 0, pages = 0, from = 0, to = 0;
    if (step) {
      from = static_cast<int>(rng() % (int(sess[self].size()) * T + 1));
      to = from + 1 + static_cast<int>(rng() % 16);
    } else {
      pages = 1 + static_cast<int>(rng() % 12);
      const int pick = static_cast<int>(rng() % 3);
      if (pick == 1) {
        const int s = static_cast<int>(rng() % S);
        if (s != self && !running[s] && !sess[s].empty() && sess[s][0]) donor_s = s;
      } else if (pick == 2 && !leaves.empty()) {
        donor_l = static_cast<int>(rng() % leaves.size());
      }
      const std::vector<int>* d = donor_s >= 0 ? &sess[donor_s] : donor_l >= 0 ? &leaves[donor_l] : nullptr;
      int covered = 0;  // positions the source holds from 0 with no hole
      if (d) {
        while (covered < static_cast<int>(d->size()) && (*d)[covered]) ++covered;
        covered *= T;
      } else {
        covered = static_cast<int>(sess[self].size()) * T;  // its own history
      }
      reuse = std::min<int>(static_cast<int>(rng() % (pages * T)), covered);
      if (d && reuse == 0) donor_s = donor_l = -1;
    }
    auto source_table = [&]() -> std::vector<int>* {
      return donor_s >= 0 ? &sess[donor_s] : donor_l >= 0 ? &leaves[donor_l] : nullptr;
    };
    // The tiers: the kept tier is every store leaf but the source, then every idle session but self and the donor,
    // in a random (LRU) order; the source tier the donor or the source leaf.
    std::vector<std::vector<int>*> kept_h;
    for (int l = 0; l < static_cast<int>(leaves.size()); ++l)
      if (l != donor_l) kept_h.push_back(&leaves[l]);
    for (int s = 0; s < S; ++s)
      if (s != self && s != donor_s && !running[s]) kept_h.push_back(&sess[s]);
    std::shuffle(kept_h.begin(), kept_h.end(), rng);
    std::vector<std::vector<int>*> source_h;
    if (source_table()) source_h.push_back(source_table());
    KvPool::Refs gone_kept(kept_h.begin(), kept_h.end()), gone_all = gone_kept;
    for (auto* t : source_h) gone_all.push_back(t);
    bool source_gone = false;
    auto need = [&] {
      if (step) return pool.needed(sess[self], from, to);
      if (source_gone) return pool.resume_need(sess[self], 0, pages, nullptr);  // planned again: cold
      const std::vector<int>* d = source_table();
      return pool.resume_need(sess[self], reuse, pages, d);
    };
    auto have = [&] { return pool.free_pages() + pool.pending_pages(); };
    const int base = have();
    const int need_kept = step ? pool.needed(sess[self], from, to, &gone_kept)
                               : pool.resume_need(sess[self], reuse, pages, source_table(), &gone_kept);
    const int reach_kept = base + pool.reclaimable(gone_kept);
    const int need_all = step ? pool.needed(sess[self], from, to, &gone_all)
                              : pool.resume_need(sess[self], 0, pages, nullptr, &gone_all);
    const int reach_all = base + pool.reclaimable(gone_all);
    const bool must = rng() % 2;
    // Played out on copies: the unit after the whole kept tier is gone, and after every holder is.
    auto play = [&](bool all) {
      KvPool q = pool;
      std::vector<std::vector<int>> qs = sess, ql = leaves;
      auto local = [&](std::vector<int>* t) -> std::vector<int>* {
        for (int s = 0; s < S; ++s)
          if (t == &sess[s]) return &qs[s];
        for (size_t l = 0; l < leaves.size(); ++l)
          if (t == &leaves[l]) return &ql[l];
        return nullptr;
      };
      for (auto* t : kept_h) release_all(q, *local(t));
      if (all)
        for (auto* t : source_h) release_all(q, *local(t));
      q.retire();
      if (step) return q.ensure(qs[self], from, to, nullptr);
      const std::vector<int>* d = all ? nullptr : (source_table() ? local(source_table()) : nullptr);
      return play_resume(q, qs[self], all ? 0 : reuse, pages, d);
    };
    const bool fits_kept = play(false), fits_all = play(true);
    size_t kept_i = 0;
    const std::vector<KvEvictNext> kept = {[&] {
      if (kept_i >= kept_h.size()) return false;
      release_all(pool, *kept_h[kept_i++]);
      return true;
    }};
    const std::vector<KvEvictNext> source = {[&] {
      if (source_gone || source_h.empty()) return false;
      release_all(pool, *source_h[0]);
      source_gone = true;
      return true;
    }};
    const bool ok = kv_reclaim(need, have, kept, need_kept, reach_kept, source, need_all, reach_all, must);
    ++cases;
    if (fits_kept && !ok) {
      std::printf("  round %d: kept tier fits but kv_reclaim refused (need_kept %d reach_kept %d, %s)\n", round, need_kept,
                  reach_kept, step ? "step" : "admission");
      check(false, "walk: kv_reclaim never refuses a unit the kept tier's eviction makes fit");
    }
    if (must && fits_all && !ok) {
      std::printf("  round %d: everything gone fits but kv_reclaim refused with must\n", round);
      check(false, "walk: with must, kv_reclaim never refuses a unit that fits once everything is gone");
    }
    kept_fits += fits_kept;
    refused += !ok;
    if (!ok) continue;
    ++trues;
    source_used += source_gone;
    // The unit's calls, as the engine makes them after the reclaim (the deferred frees retired first).
    pool.retire();
    bool ran = false;
    if (step) {
      ran = pool.ensure(sess[self], from, to, nullptr);
    } else {
      const std::vector<int>* d = source_gone ? nullptr : source_table();
      ran = play_resume(pool, sess[self], source_gone ? 0 : reuse, pages, d);
    }
    if (!ran) {
      std::printf("  round %d: kv_reclaim true but the %s failed (source gone %d)\n", round, step ? "step" : "prefill",
                  source_gone);
      check(false, "walk: whenever kv_reclaim returns true the unit's calls succeed");
    }
  }
  std::printf("  reclaim walk: %d cases, %d true (%d after evicting the source), %d refused, %d where the kept tier fits\n",
              cases, trues, source_used, refused, kept_fits);
  check(trues > 4000 && refused > 400 && source_used > 40, "walk: enough cases of each outcome");
}

int main() {
  basics();
  sharing();
  resume();
  admission();
  random_walk();
  plan();
  policy();
  gone_counts();
  reclaim_gates();
  reclaim_walk();
  std::printf(failures ? "KV POOL TESTS FAILED (%d)\n" : "all kv pool tests passed\n", failures);
  return failures ? 1 : 0;
}
