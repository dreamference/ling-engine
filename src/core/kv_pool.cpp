#include "core/kv_pool.hpp"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

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

int KvPool::net_need(const std::vector<int>& table, int pages) {
  int need = 0;
  const int n = std::max(pages, static_cast<int>(table.size()));
  for (int i = 0; i < n; ++i) {
    const bool mapped = i < static_cast<int>(table.size()) && table[i] != 0;
    if (i < pages) need += !mapped;
    else need -= mapped;
  }
  return std::max(need, 0);
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

void KvPool::unshare(std::vector<int>& table, int n) {
  for (int i = pages_for(std::max(n, 0)); i < static_cast<int>(table.size()); ++i)
    if (table[i] != 0 && refs_[table[i]] > 1) {
      release(table[i]);
      table[i] = 0;
    }
}

int KvPool::resume_need(const std::vector<int>& table, int reuse, int pages, const std::vector<int>* donor) const {
  // The sequence of calls replayed on a copy of the table, with the reference changes it makes kept aside.
  std::unordered_map<int, int> delta;
  auto refs = [&](int p) {
    const auto it = delta.find(p);
    return refs_.at(p) + (it == delta.end() ? 0 : it->second);
  };
  int take = 0, freed = 0;
  auto drop = [&](int p) {
    if (refs(p) == 1) ++freed;
    --delta[p];
  };
  std::vector<int> t = table;
  const int pr = pages_for(std::max(reuse, 0));
  if (donor) {  // share()
    if (pr > static_cast<int>(donor->size())) throw std::invalid_argument("KvPool::resume_need: donor does not hold [0, reuse)");
    if (static_cast<int>(t.size()) < pr) t.resize(pr, 0);
    for (int i = 0; i < pr; ++i) {
      const int q = (*donor)[i];
      if (q == 0) throw std::invalid_argument("KvPool::resume_need: donor has an unmapped page below reuse");
      if (t[i] == q) continue;
      ++delta[q];
      if (t[i] != 0) drop(t[i]);
      t[i] = q;
    }
  }
  for (int i = pages; i < static_cast<int>(t.size()); ++i)  // truncate()
    if (t[i] != 0) {
      drop(t[i]);
      t[i] = 0;
    }
  for (int i = pr; i < static_cast<int>(t.size()); ++i)  // unshare()
    if (t[i] != 0 && refs(t[i]) > 1) {
      drop(t[i]);
      t[i] = 0;
    }
  const int w0 = index_of(std::max(reuse, 0));
  for (int i = 0; i < pages; ++i) {  // ensure(reuse, end), end past reuse
    const int p = i < static_cast<int>(t.size()) ? t[i] : 0;
    if (p == 0) {
      ++take;
    } else if (i >= w0 && refs(p) > 1) {
      ++take;
      drop(p);
    }
  }
  return std::max(take - freed, 0);
}

int KvPool::reclaimable(const std::vector<const std::vector<int>*>& refs) const {
  std::unordered_map<int, int> listed;
  for (const std::vector<int>* r : refs)
    if (r)
      for (int p : *r)
        if (p > 0) ++listed[p];
  int n = 0;
  for (const auto& [p, k] : listed) n += k >= refs_.at(p);
  return n;
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
