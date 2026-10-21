// The prefix KV store's tree (prefix_kv.hpp).
#include "core/prefix_kv.hpp"

#include <algorithm>
#include <iterator>
#include <utility>

namespace ling {

ArenaAlloc::ArenaAlloc(size_t size, size_t align) : size_(size - size % align), align_(align) {
  if (size_ > 0) free_[0] = size_;
}

bool ArenaAlloc::alloc(size_t bytes, size_t* offset) {
  bytes = (bytes + align_ - 1) / align_ * align_;
  for (auto it = free_.begin(); it != free_.end(); ++it)
    if (it->second >= bytes) {
      const size_t at = it->first, len = it->second;
      free_.erase(it);
      if (len > bytes) free_[at + bytes] = len - bytes;
      used_[at] = bytes;
      in_use_ += bytes;
      *offset = at;
      return true;
    }
  return false;
}

void ArenaAlloc::release(size_t offset) {
  auto u = used_.find(offset);
  if (u == used_.end()) return;
  size_t at = u->first, len = u->second;
  used_.erase(u);
  in_use_ -= len;
  auto next = free_.lower_bound(at);
  if (next != free_.end() && at + len == next->first) {
    len += next->second;
    next = free_.erase(next);
  }
  if (next != free_.begin()) {
    auto prev = std::prev(next);
    if (prev->first + prev->second == at) {
      prev->second += len;
      return;
    }
  }
  free_[at] = len;
}

PrefixKvTree::PrefixKvTree(size_t budget_bytes, size_t bytes_per_token, size_t state_bytes, Ops ops)
    : budget_(budget_bytes), per_token_(bytes_per_token), state_bytes_(state_bytes), ops_(std::move(ops)) {
  nodes_.emplace_back();
  nodes_[0].live = true;
}

PrefixKvTree::~PrefixKvTree() { clear(); }

void PrefixKvTree::clear() {
  for (Node& n : nodes_)
    if (n.live && n.state) ops_.release(n.state);
  for (Buffer& b : bufs_)
    if (b.refs > 0 && b.ptr) ops_.release(b.ptr);
  nodes_.assign(1, Node{});
  nodes_[0].live = true;
  bufs_.clear();
  free_nodes_.clear();
  bytes_ = 0;
}

int PrefixKvTree::new_node() {
  if (!free_nodes_.empty()) {
    const int i = free_nodes_.back();
    free_nodes_.pop_back();
    nodes_[i] = Node{};
    nodes_[i].live = true;
    return i;
  }
  nodes_.emplace_back();
  nodes_.back().live = true;
  return static_cast<int>(nodes_.size()) - 1;
}

PrefixKvTree::Match PrefixKvTree::match(const std::vector<int>& prompt) const {
  Match m;
  int n = 0, pos = 0;
  const int L = static_cast<int>(prompt.size());
  while (pos < L) {
    int next = -1;
    for (int k : nodes_[n].kids)
      if (nodes_[k].tokens[0] == prompt[pos]) {
        next = k;
        break;
      }
    if (next < 0) break;
    const Node& c = nodes_[next];
    const int len = c.end - c.start;
    int j = 0;
    while (j < len && pos + j < L && c.tokens[j] == prompt[pos + j]) ++j;
    m.path.push_back(next);
    pos += j;
    if (j < len) break;
    n = next;
  }
  m.len = pos;
  return m;
}

int PrefixKvTree::best_state(const Match& m, int limit, const void** state) const {
  int best = 0;
  *state = nullptr;
  for (int k : m.path) {
    const Node& c = nodes_[k];
    if (c.end > m.len || c.end > limit) break;
    if (c.state) {
      best = c.end;
      *state = c.state;
    }
  }
  return best;
}

std::vector<PrefixKvTree::Segment> PrefixKvTree::segments(const Match& m, int n) const {
  std::vector<Segment> out;
  for (int k : m.path) {
    const Node& c = nodes_[k];
    if (c.start >= n) break;
    const Buffer& b = bufs_[c.buf];
    const int to = std::min(c.end, n);
    if (!out.empty() && out.back().buf == b.ptr && out.back().to == c.start) out.back().to = to;  // one run, split
    else out.push_back({b.ptr, b.start, b.len, c.start, to});
  }
  return out;
}

void PrefixKvTree::touch(const Match& m, uint64_t epoch) {
  for (int k : m.path) nodes_[k].used = std::max(nodes_[k].used, epoch);
}

int PrefixKvTree::split(int node, int at) {
  const int b = new_node();  // may move nodes_: index from here on
  Node& a = nodes_[node];
  Node& nb = nodes_[b];
  nb.parent = node;
  nb.start = at;
  nb.end = a.end;
  nb.tokens.assign(a.tokens.begin() + (at - a.start), a.tokens.end());
  nb.buf = a.buf;
  bufs_[a.buf].refs += 1;
  nb.state = a.state;
  nb.used = a.used;
  nb.kids = std::move(a.kids);
  for (int k : nb.kids) nodes_[k].parent = b;
  a.tokens.resize(at - a.start);
  a.end = at;
  a.state = nullptr;
  a.kids = {b};
  return b;
}

void PrefixKvTree::evict(int leaf) {
  Node& n = nodes_[leaf];
  std::vector<int>& kids = nodes_[n.parent].kids;
  kids.erase(std::find(kids.begin(), kids.end(), leaf));
  if (n.state) {
    ops_.release(n.state);
    bytes_ -= state_bytes_;
  }
  Buffer& b = bufs_[n.buf];
  if (--b.refs == 0) {
    ops_.release(b.ptr);
    b.ptr = nullptr;
    bytes_ -= b.bytes;
  }
  n = Node{};
  free_nodes_.push_back(leaf);
}

bool PrefixKvTree::make_room(size_t need, uint64_t epoch) {
  while (bytes_ + need > budget_) {
    int victim = -1;
    for (size_t i = 1; i < nodes_.size(); ++i) {
      const Node& n = nodes_[i];
      if (!n.live || !n.kids.empty() || n.used >= epoch) continue;
      if (victim < 0 || n.used < nodes_[victim].used) victim = static_cast<int>(i);
    }
    if (victim < 0) return false;
    evict(victim);
  }
  return true;
}

std::vector<std::pair<int, int>> PrefixKvTree::insert(const std::vector<int>& prompt, int q,
                                                      const std::vector<int>& cuts, uint64_t epoch) {
  q = std::min(q, static_cast<int>(prompt.size()));
  if (q <= 0) return {};
  const std::vector<int> head(prompt.begin(), prompt.begin() + q);
  Match m = match(head);
  // A match that ends inside a run, and every cut inside a matched run, become node ends.
  if (!m.path.empty() && nodes_[m.path.back()].end > m.len) split(m.path.back(), m.len);
  for (int c : cuts) {
    if (c <= 0 || c >= m.len) continue;
    for (int k : match(head).path)
      if (nodes_[k].start < c && c < nodes_[k].end) {
        split(k, c);
        break;
      }
  }
  m = match(head);
  touch(m, epoch);
  if (m.len < q && make_room(static_cast<size_t>(q - m.len) * per_token_, epoch)) {
    const size_t bytes = static_cast<size_t>(q - m.len) * per_token_;
    if (void* p = ops_.alloc(bytes)) {
      ops_.store(p, m.len, q - m.len, m.len, q);
      bufs_.push_back({p, m.len, q - m.len, bytes, 0});
      const int buf = static_cast<int>(bufs_.size()) - 1;
      bytes_ += bytes;
      int parent = m.path.empty() ? 0 : m.path.back();
      std::vector<int> ends;
      for (int c : cuts)
        if (c > m.len && c < q) ends.push_back(c);
      ends.push_back(q);
      int start = m.len;
      for (int e : ends) {
        const int k = new_node();
        Node& n = nodes_[k];
        n.parent = parent;
        n.start = start;
        n.end = e;
        n.tokens.assign(head.begin() + start, head.begin() + e);
        n.buf = buf;
        n.used = epoch;
        bufs_[buf].refs += 1;
        nodes_[parent].kids.push_back(k);
        parent = k;
        start = e;
      }
      m = match(head);
    }
  }
  std::vector<std::pair<int, int>> want;
  for (int c : cuts)
    for (int k : m.path)
      if (nodes_[k].end == c && c <= m.len) {
        if (!nodes_[k].state) want.push_back({k, c});
        break;
      }
  return want;
}

void* PrefixKvTree::add_state(int node, uint64_t epoch) {
  if (node <= 0 || node >= static_cast<int>(nodes_.size()) || !nodes_[node].live || nodes_[node].state) return nullptr;
  if (!make_room(state_bytes_, epoch)) return nullptr;
  void* p = ops_.alloc(state_bytes_);
  if (!p) return nullptr;
  nodes_[node].state = p;
  nodes_[node].used = std::max(nodes_[node].used, epoch);
  bytes_ += state_bytes_;
  return p;
}

int PrefixKvTree::nodes() const {
  int n = 0;
  for (size_t i = 1; i < nodes_.size(); ++i) n += nodes_[i].live;
  return n;
}

int PrefixKvTree::states() const {
  int n = 0;
  for (size_t i = 1; i < nodes_.size(); ++i) n += nodes_[i].live && nodes_[i].state;
  return n;
}

int PrefixKvTree::buffers() const {
  int n = 0;
  for (const Buffer& b : bufs_) n += b.refs > 0;
  return n;
}

}  // namespace ling
