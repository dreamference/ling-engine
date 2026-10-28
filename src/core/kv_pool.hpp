// P4b (specs/DREAMFERENCE_LING_ENGINE_BATCHING.md §5.3): the KV page pool's bookkeeping. The pool is one device
// allocation of fixed-size pages made at engine start; this class only decides which page holds what: a free list, a
// reference count per page, and the page tables of the holders (sessions, the prefix KV store), each a vector of page
// ids indexed by position / kKvPageTokens with 0 for "unmapped" (page 0 is never handed out: an address formed for a
// masked key lands there). The engine turns ids into device offsets and makes the copies this class asks for, so the
// whole policy is tested on the host (tests/kv_pool_tests.cpp).
//
// Rules that keep the KV exact:
//  - a page shared by more than one holder is never written: ensure() gives the writer a private copy first
//    (copy-on-write), and returns the copy for the engine to make before the write is launched;
//  - a released page is not handed out again until retire() (the step boundary, after every stream that could still
//    read it has joined): a reader on another stream never sees it rewritten.
#pragma once

#include <cstddef>
#include <vector>

namespace ling {

class KvPool {
 public:
  static constexpr int kPageTokens = 64;  // kernels::kKvPageTokens
  struct Copy {
    int src, dst;  // a whole page, src into dst, before the write that needed it
  };

  // `pages` ids, 1 .. pages - 1 usable (id 0 reserved).
  explicit KvPool(int pages);

  int capacity() const { return static_cast<int>(refs_.size()) - 1; }
  int free_pages() const { return static_cast<int>(free_.size()); }
  int pending_pages() const { return static_cast<int>(pending_.size()); }
  int refs(int page) const { return refs_.at(page); }

  static int index_of(int pos) { return pos / kPageTokens; }
  static int pages_for(int positions) { return (positions + kPageTokens - 1) / kPageTokens; }

  // The pages ensure(table, from, to) would take from the free list (new pages and copies).
  int needed(const std::vector<int>& table, int from, int to) const;
  // What a holder that keeps its pages below index `pages` (rewriting them), maps the missing ones and gives back the
  // ones at or past it draws from the pool, net: the unmapped indices below `pages` less the mapped ones past it, at
  // least 0 (P4b: a new prompt in a session, Engine::kv_admit_need; no page is shared there).
  static int net_need(const std::vector<int>& table, int pages);
  // Before a pass writes positions [from, to) of the holder of `table` (from <= to): maps a page for every index below
  // pages_for(to) that has none, and gives every index the write touches whose page is shared a private copy (appended
  // to *copies). All or nothing: false, with nothing changed, when the free list is short (see needed()).
  bool ensure(std::vector<int>& table, int from, int to, std::vector<Copy>* copies);
  // `dst` maps the pages that hold src's positions [0, n) (a reference each), releasing what it mapped there before.
  // The last of them may be only partly inside [0, n): a later write by either holder copies it first.
  void share(std::vector<int>& dst, const std::vector<int>& src, int n);
  // Releases the pages wholly at or past position n (the holder keeps positions [0, n)).
  void truncate(std::vector<int>& table, int n);
  void retain(int page);
  void release(int page);
  // The step boundary: pages released since the last call become free.
  void retire();

 private:
  int take();
  std::vector<int> refs_;
  std::vector<int> free_, pending_;
};

}  // namespace ling
