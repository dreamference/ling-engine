#include "core/copy_proposer.hpp"

#include <algorithm>
#include <stdexcept>

namespace ling {

CopyIndex::CopyIndex(int ngram) : ngram_(ngram) {
  if (ngram < 1) throw std::invalid_argument("CopyIndex: n-gram length must be at least 1");
}

void CopyIndex::configure(int ngram) {
  if (ngram < 1) throw std::invalid_argument("CopyIndex: n-gram length must be at least 1");
  if (ngram == ngram_) return;
  ngram_ = ngram;
  clear();
}

void CopyIndex::clear() {
  tokens_.clear();
  prev_.clear();
  head_.clear();
}

uint64_t CopyIndex::hash_at(const int* end) const {
  uint64_t h = 0x84222325cbf29ce4ULL;
  for (const int* t = end - ngram_ + 1; t <= end; ++t) {
    h = (h ^ static_cast<uint32_t>(*t)) * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 29;
  }
  return h;
}

void CopyIndex::insert(int end) {
  const uint64_t h = hash_at(tokens_.data() + end);
  auto [it, fresh] = head_.try_emplace(h, end);
  prev_[end] = fresh ? -1 : it->second;
  it->second = end;
}

void CopyIndex::remove(int end) {
  const uint64_t h = hash_at(tokens_.data() + end);
  auto it = head_.find(h);
  // Entries are removed newest first, so `end` is its hash's head.
  if (it == head_.end() || it->second != end) throw std::logic_error("CopyIndex: undo out of order");
  if (prev_[end] < 0) head_.erase(it);
  else it->second = prev_[end];
  prev_[end] = -1;
  ++undone_;
}

void CopyIndex::append(std::span<const int> t) {
  tokens_.reserve(tokens_.size() + t.size());
  prev_.reserve(prev_.size() + t.size());
  for (int x : t) {
    tokens_.push_back(x);
    prev_.push_back(-1);
    // The n-gram ending just before the new token now has a continuation.
    const int e = static_cast<int>(tokens_.size()) - 2;
    if (e >= ngram_ - 1) insert(e);
  }
}

void CopyIndex::truncate(size_t n) {
  const int size = static_cast<int>(tokens_.size());
  if (static_cast<int>(n) >= size) return;
  if (n == 0) {  // the same state as undoing every entry
    clear();
    return;
  }
  // Indexed ends are ngram - 1 .. size - 2; those at n - 1 and after lose their continuation.
  const int lo = std::max(static_cast<int>(n) - 1, ngram_ - 1);
  for (int e = size - 2; e >= lo; --e) remove(e);
  tokens_.resize(n);
  prev_.resize(n);
}

void CopyIndex::sync(std::span<const int> a, std::span<const int> b) {
  const size_t total = a.size() + b.size(), m = std::min(tokens_.size(), total);
  const size_t ma = std::min(m, a.size());
  size_t c = std::mismatch(tokens_.begin(), tokens_.begin() + ma, a.begin()).first - tokens_.begin();
  if (c == ma && ma < m)
    c += std::mismatch(tokens_.begin() + ma, tokens_.begin() + m, b.begin()).first - (tokens_.begin() + ma);
  truncate(c);
  if (c < a.size()) {
    append(a.subspan(c));
    append(b);
  } else {
    append(b.subspan(c - a.size()));
  }
}

CopyProposal CopyIndex::propose(std::span<const int> query, const CopyParams& p) const {
  CopyProposal r;
  const int Q = static_cast<int>(query.size()), n = ngram_, size = static_cast<int>(tokens_.size());
  if (Q < n || size < 2 || p.max_tokens <= 0) return r;
  const int* q = query.data();
  auto it = head_.find(hash_at(q + Q - 1));
  if (it == head_.end()) return r;
  int best = -1, best_m = 0;
  for (int e = it->second; e >= 0 && r.candidates < p.max_candidates; e = prev_[e]) {
    ++r.candidates;
    const int cap = std::min({p.match_cap, e + 1, Q});
    int m = 0;
    while (m < cap && q[Q - 1 - m] == tokens_[e - m]) ++m;
    if (m < std::min(n, cap)) continue;  // a hash collision
    if (m > best_m) {                    // ties: the most recent, seen first
      best_m = m;
      best = e;
    }
  }
  r.match = best_m;
  if (best < 0 || best_m < p.min_match) return r;
  int len = std::min(size - 1 - best, p.max_tokens);
  if (p.alpha > 0) len = std::min(len, std::max(p.min_tokens, static_cast<int>(p.alpha * best_m)));
  if (len <= 0) return r;
  r.source = best + 1;
  r.tokens.assign(tokens_.begin() + best + 1, tokens_.begin() + best + 1 + len);
  return r;
}

int extend_with_copy(std::vector<int>& drafts, const std::vector<int>& copy, int max_tokens) {
  const int E = static_cast<int>(drafts.size());
  if (static_cast<int>(copy.size()) <= E || E >= max_tokens) return 0;
  if (!std::equal(drafts.begin(), drafts.end(), copy.begin())) return 0;
  const int add = std::min(static_cast<int>(copy.size()), max_tokens) - E;
  drafts.insert(drafts.end(), copy.begin() + E, copy.begin() + E + add);
  return add;
}

}  // namespace ling
