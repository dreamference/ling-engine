// The prefix KV store (BACKLOG 35): the full-attention KV of prompt prefixes kept beyond the resident sessions, in a
// radix tree of token runs, with DeltaNet and conv states at some of its nodes' ends, bounded by a byte budget.
//
// A node holds the tokens [start, end) after its parent's; its KV lives in a buffer, possibly shared with the nodes a
// split made of one run (the buffer's positions [buf_start, buf_start + buf_len)), and may hold a state: the
// recurrent state after exactly the tokens [0, end). A prompt resumes from the tree at a node end with a state, its
// KV for [0, end) copied out of the buffers along the path; a stored KV run also lets a state from the checkpoint
// store at any position up to the matched length be used. Leaves are evicted least recently used first, never one
// stamped by the current prefill. This class holds the tree and its accounting only; the engine supplies the device
// side (allocation, copies) through Ops, so the tree is tested on the host (tests/prefix_kv_tests.cpp).
//
// Paged mode (P4c, specs/DREAMFERENCE_LING_ENGINE_BATCHING.md §5.3, "Sharing and copy-on-write"), chosen by giving
// Ops::hold: a node holds a reference to each KV page that overlaps its run (KvPool's page ids, kPageTokens positions
// each) instead of a copy in a buffer, so the page straddling a boundary between two nodes is held by both, once each.
// The arena and the budget (--prefix-kv-mb) then hold the states only: the KV's bytes are the page pool's and are
// not charged here (bytes_per_token is ignored; the engine passes 0), `store` and segments() are unused, and a restore
// maps pages(). The store's pages are the pool's first reclaimable memory: evict_lru_leaf() gives them back. Without
// `hold` the tree keeps the buffer mode above, unchanged (an engine run without --kv-pages).
//
// Share scopes: a root child carries a scope and every node below it inherits it; a match or insert follows only the
// root children of its own scope, so prefixes are reused across conversations only within one scope (two scopes with
// the same prompt hold separate runs).
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

#include "core/kv_pool.hpp"

namespace ling {

// First-fit offsets in one region allocated once (the prefix KV store's arena: its footprint is fixed when the engine
// starts and never grows while serving), freed ranges merged with their neighbours. Host bookkeeping only.
class ArenaAlloc {
 public:
  ArenaAlloc(size_t size, size_t align);
  bool alloc(size_t bytes, size_t* offset);  // false: no free range is large enough
  void release(size_t offset);
  size_t size() const { return size_; }
  size_t in_use() const { return in_use_; }

 private:
  size_t size_, align_, in_use_ = 0;
  std::map<size_t, size_t> free_;  // offset -> length
  std::map<size_t, size_t> used_;  // offset -> length
};

class PrefixKvTree {
 public:
  struct Ops {
    std::function<void*(size_t bytes)> alloc;  // a device allocation, or nullptr (the insert then stops)
    std::function<void(void* p)> release;
    // Copies the active session's KV for positions [from, to) into `buf` (a buffer of buf_len positions whose first
    // position is buf_start).
    std::function<void(void* buf, int buf_start, int buf_len, int from, int to)> store;
    // Paged mode (set all three, or none). `hold` returns the active session's page ids for the page indices
    // KvPool::index_of(from) .. KvPool::index_of(to - 1), a reference taken on each (from < to); a result of another
    // length, or with an unmapped id (0), is a failure: the tree drops the ids it got and stops the insert there,
    // keeping the nodes already made. `retain` and `drop` take and give back one reference to a page.
    std::function<std::vector<int>(int from, int to)> hold;
    std::function<void(int page)> retain;
    std::function<void(int page)> drop;
  };
  // One piece of a restore: the KV for positions [from, to) is in `buf` (buf_len positions from buf_start).
  struct Segment {
    const void* buf;
    int buf_start, buf_len, from, to;
  };
  struct Match {
    int len = 0;             // the prompt's leading tokens the tree holds the KV of
    std::vector<int> path;   // the nodes from the root's first child to the one holding position len - 1
  };

  PrefixKvTree(size_t budget_bytes, size_t bytes_per_token, size_t state_bytes, Ops ops);
  ~PrefixKvTree();
  PrefixKvTree(const PrefixKvTree&) = delete;
  PrefixKvTree& operator=(const PrefixKvTree&) = delete;

  // The longest run of the prompt's leading tokens the tree holds within share scope `scope`.
  Match match(const std::vector<int>& prompt, uint64_t scope = 0) const;
  // The deepest node end on the match's path that holds a state, at most `limit`: its position (0: none) and state.
  int best_state(const Match& m, int limit, const void** state) const;
  // The buffers holding the KV of [0, n) along the path (n <= m.len). Buffer mode only (empty in paged mode).
  std::vector<Segment> segments(const Match& m, int n) const;
  // Paged mode: the page ids a restore of [0, n) maps (n <= m.len), one per page index below KvPool::pages_for(n),
  // none 0. Index i gets the page of the deepest node on the path whose run overlaps [64 i, min(64 i + 64, n)). That
  // node's page was written by a prompt that matched every shallower node through its end and this node through its
  // own tokens up to n (the tokens from the root to a node's start never change, splits only cut them into more
  // nodes), so it is right for every position below n inside the page; a shallower node's copy of the same index may
  // hold another prompt's tokens past its own end. Positions at or past n are rewritten by the restoring session
  // before it reads them, and the engine copies the partly used last page before that write (copy-on-write).
  std::vector<int> pages(const Match& m, int n) const;
  // Stamps the matched path as used at `epoch` (so the eviction of this prefill's inserts spares it).
  void touch(const Match& m, uint64_t epoch);
  // Stores the KV of prompt[0, q) (the active session holds it), splitting runs so that each position in `cuts`
  // (sorted, in (0, q]) is a node end. Returns the cut nodes still without a state: (node, position) pairs for
  // set_state. Nothing is evicted that was stamped at or after `epoch`; when the budget cannot make room the insert
  // keeps what it has stored.
  // In paged mode each new node [start, e) holds the session's pages for its run (Ops::hold), and the KV needs no
  // room in the budget. `scope` is the share scope the prompt is stored under (see match()).
  std::vector<std::pair<int, int>> insert(const std::vector<int>& prompt, int q, const std::vector<int>& cuts,
                                          uint64_t epoch, uint64_t scope = 0);
  // Gives node `node` a state buffer of state_bytes (allocated here, accounted in the budget); returns it, or nullptr.
  void* add_state(int node, uint64_t epoch);

  // Evicts the least recently used live leaf whatever its stamp (the engine's reclaim of pool pages before idle
  // sessions, §5.3 "A full pool"); false when the tree is empty.
  bool evict_lru_leaf();

  void clear();  // everything released, every page dropped (a cold engine)
  size_t bytes() const { return bytes_; }
  size_t budget() const { return budget_; }
  int nodes() const;     // live nodes, the root excluded
  int states() const;
  int buffers() const;
  bool paged() const { return paged_; }
  int pages_held() const;  // the page references the tree holds, a page counted once per node holding it
  // Paged mode: each live node's page ids (its references; KvPool::reclaimable counts what evicting them frees).
  std::vector<std::vector<int>> page_lists() const;

 private:
  struct Node {
    int parent = -1;
    int start = 0, end = 0;
    std::vector<int> tokens;  // [start, end)
    int buf = -1;             // buffer mode
    std::vector<int> pages;   // paged mode: the page ids for indices index_of(start) .. index_of(end - 1)
    uint64_t scope = 0;       // the share scope (a root child's, inherited below it)
    void* state = nullptr;
    uint64_t used = 0;
    std::vector<int> kids;
    bool live = false;
  };
  struct Buffer {
    void* ptr = nullptr;
    int start = 0, len = 0;
    size_t bytes = 0;
    int refs = 0;
  };
  int split(int node, int at);  // node becomes [start, at); returns the new node [at, end), its child
  bool make_room(size_t need, uint64_t epoch);
  void evict(int leaf);
  int new_node();

  size_t budget_, per_token_, state_bytes_, bytes_ = 0;
  Ops ops_;
  bool paged_;
  std::vector<Node> nodes_;  // nodes_[0] is the root (the empty prefix)
  std::vector<Buffer> bufs_;
  std::vector<int> free_nodes_;
};

}  // namespace ling
