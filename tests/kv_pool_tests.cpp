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
  // Victims: least recently used first, only as many as needed, none when all would not do.
  const std::vector<int> lru = {3, 1, 2}, pages = {4, 0, 9};
  check(kv_victims(15, v, lru, pages).empty(), "no victim when the free and pending pages do");
  check(kv_victims(17, v, lru, pages) == std::vector<int>{3}, "the least recently used first");
  check(kv_victims(20, v, lru, pages) == (std::vector<int>{3, 2}), "a session holding nothing is skipped");
  check(kv_victims(29, v, lru, pages).empty(), "nothing evicted when even all would not make enough");
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
  auto view = [&](int keep, const std::vector<bool>& busy, std::vector<int>* order, std::vector<int>* held) {
    KvPoolView w{pool.free_pages(), pool.pending_pages(), 0};
    for (int s = 0; s < 3; ++s) {  // the LRU order here is the session index
      if (s == keep || busy[s]) continue;
      int n = 0;
      for (int p : t[s]) n += p != 0;
      w.idle += n;
      if (order) order->push_back(s);
      if (held) held->push_back(n);
    }
    return w;
  };
  auto admit = [&](int s, int prompt, const std::vector<bool>& busy) {
    const int need = pool.needed(t[s], 0, prompt + 2 * T);
    std::vector<int> order, held;
    const KvPoolView w = view(s, busy, &order, &held);
    const KvVerdict d = kv_verdict(need, w);
    if (d == KvVerdict::Wait) return false;
    for (int x : kv_victims(need, w, order, held)) {
      pool.truncate(t[x], 0);
      pos[x] = 0;
    }
    pool.retire();  // nothing runs in this model: the deferred frees retire at once
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
    if (kv_verdict(runners[i].need, view(-1, {true, false, true}, nullptr, nullptr)) == KvVerdict::Go) {
      pool.ensure(t[s], pos[s], pos[s] + 16, nullptr);
      pos[s] += 5;  // a step accepts a few of its 16 rows; the rest are rewritten by the next
      runners[i].waiting = false;
    } else {
      runners[i].waiting = true;
      ended = kv_last_resort(runners, view(-1, {true, false, true}, nullptr, nullptr));
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
    const KvPoolView w{p2.free_pages(), p2.pending_pages(), 2};
    check(kv_verdict(KvPool::net_need(a, 12), w) == KvVerdict::Reclaim, "the next turn reclaims the idle session");
    p2.truncate(b, 0);
    p2.retire();
    check(p2.ensure(a, 10 * T, 12 * T, nullptr) && p2.free_pages() == 0, "and the pool is exactly full");
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

int main() {
  basics();
  sharing();
  resume();
  admission();
  random_walk();
  plan();
  policy();
  std::printf(failures ? "KV POOL TESTS FAILED (%d)\n" : "all kv pool tests passed\n", failures);
  return failures ? 1 : 0;
}
