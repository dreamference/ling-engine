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
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <vector>

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

  Match match(const std::vector<int>& prompt) const;
  // The deepest node end on the match's path that holds a state, at most `limit`: its position (0: none) and state.
  int best_state(const Match& m, int limit, const void** state) const;
  // The buffers holding the KV of [0, n) along the path (n <= m.len).
  std::vector<Segment> segments(const Match& m, int n) const;
  // Stamps the matched path as used at `epoch` (so the eviction of this prefill's inserts spares it).
  void touch(const Match& m, uint64_t epoch);
  // Stores the KV of prompt[0, q) (the active session holds it), splitting runs so that each position in `cuts`
  // (sorted, in (0, q]) is a node end. Returns the cut nodes still without a state: (node, position) pairs for
  // set_state. Nothing is evicted that was stamped at or after `epoch`; when the budget cannot make room the insert
  // keeps what it has stored.
  std::vector<std::pair<int, int>> insert(const std::vector<int>& prompt, int q, const std::vector<int>& cuts,
                                          uint64_t epoch);
  // Gives node `node` a state buffer of state_bytes (allocated here, accounted in the budget); returns it, or nullptr.
  void* add_state(int node, uint64_t epoch);

  void clear();  // everything released (a cold engine)
  size_t bytes() const { return bytes_; }
  size_t budget() const { return budget_; }
  int nodes() const;     // live nodes, the root excluded
  int states() const;
  int buffers() const;

 private:
  struct Node {
    int parent = -1;
    int start = 0, end = 0;
    std::vector<int> tokens;  // [start, end)
    int buf = -1;
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
  std::vector<Node> nodes_;  // nodes_[0] is the root (the empty prefix)
  std::vector<Buffer> bufs_;
  std::vector<int> free_nodes_;
};

}  // namespace ling
