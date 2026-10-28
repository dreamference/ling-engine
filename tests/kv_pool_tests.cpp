// ling-kv-pool-tests: KvPool (core/kv_pool.hpp), P4b's page bookkeeping, on the host. Allocation, copy-on-write,
// sharing, truncation, deferred frees, the all-or-nothing rule, and a random walk of holders against a model that
// tracks, for every position each holder may read, which write produced it: a holder must never read a value another
// holder wrote after sharing it (the property copy-on-write exists for). Exits 0 when everything holds.
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "core/kv_pool.hpp"

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

}  // namespace

int main() {
  basics();
  sharing();
  random_walk();
  std::printf(failures ? "KV POOL TESTS FAILED (%d)\n" : "all kv pool tests passed\n", failures);
  return failures ? 1 : 0;
}
