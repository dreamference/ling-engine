#include "core/kv_pool.hpp"

#include <algorithm>
#include <stdexcept>

namespace ling {

KvPool::KvPool(int pages) : refs_(std::max(pages, 1), 0) {
  // Handed out from the back: the lowest ids first.
  for (int p = pages - 1; p >= 1; --p) free_.push_back(p);
}

int KvPool::take() {
  if (free_.empty()) throw std::logic_error("KvPool: no free page (needed() was not checked)");
  const int p = free_.back();
  free_.pop_back();
  refs_[p] = 1;
  return p;
}

int KvPool::needed(const std::vector<int>& table, int from, int to) const {
  const int n = pages_for(to), w0 = index_of(from);
  int need = 0;
  for (int i = 0; i < n; ++i) {
    const int p = i < static_cast<int>(table.size()) ? table[i] : 0;
    if (p == 0 || (i >= w0 && from < to && refs_[p] > 1)) ++need;
  }
  return need;
}

bool KvPool::ensure(std::vector<int>& table, int from, int to, std::vector<Copy>* copies) {
  if (from < 0 || to < from) throw std::invalid_argument("KvPool::ensure: bad range");
  if (needed(table, from, to) > free_pages()) return false;
  const int n = pages_for(to), w0 = index_of(from);
  if (static_cast<int>(table.size()) < n) table.resize(n, 0);
  for (int i = 0; i < n; ++i) {
    int& p = table[i];
    if (p == 0) {
      p = take();
    } else if (i >= w0 && from < to && refs_[p] > 1) {
      const int fresh = take();
      if (copies) copies->push_back({p, fresh});
      release(p);
      p = fresh;
    }
  }
  return true;
}

void KvPool::share(std::vector<int>& dst, const std::vector<int>& src, int n) {
  const int np = pages_for(n);
  if (np > static_cast<int>(src.size())) throw std::invalid_argument("KvPool::share: src does not hold [0, n)");
  if (static_cast<int>(dst.size()) < np) dst.resize(np, 0);
  for (int i = 0; i < np; ++i) {
    if (src[i] == 0) throw std::invalid_argument("KvPool::share: src has an unmapped page below n");
    if (dst[i] == src[i]) continue;
    retain(src[i]);
    if (dst[i] != 0) release(dst[i]);
    dst[i] = src[i];
  }
}

void KvPool::truncate(std::vector<int>& table, int n) {
  for (int i = pages_for(std::max(n, 0)); i < static_cast<int>(table.size()); ++i)
    if (table[i] != 0) {
      release(table[i]);
      table[i] = 0;
    }
}

void KvPool::retain(int page) {
  if (page <= 0 || page >= static_cast<int>(refs_.size()) || refs_[page] <= 0)
    throw std::logic_error("KvPool::retain: not a live page");
  ++refs_[page];
}

void KvPool::release(int page) {
  if (page <= 0 || page >= static_cast<int>(refs_.size()) || refs_[page] <= 0)
    throw std::logic_error("KvPool::release: not a live page");
  if (--refs_[page] == 0) pending_.push_back(page);
}

void KvPool::retire() {
  free_.insert(free_.end(), pending_.rbegin(), pending_.rend());
  pending_.clear();
}

}  // namespace ling
