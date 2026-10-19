// Host-side speculation checks (no GPU):
//   - the rejection-sampling accept rule leaves the output distribution exactly the target's: a toy
//     model whose next-token distribution depends only on the position is decoded speculatively many
//     times, with random draft distributions and with deterministic (lookup) proposals, and the joint
//     distribution of the first tokens is compared with the target's by a chi-square test;
//   - the context lookup's proposals.
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "core/accept.hpp"
#include "core/lookup.hpp"
#include "core/tree.hpp"

namespace {

constexpr int kVocab = 6, kLen = 3, kTrials = 300000;

ling::Dist random_dist(std::mt19937_64& rng, bool sparse) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  ling::Dist d;
  double sum = 0;
  for (int t = 0; t < kVocab; ++t) {
    if (sparse && u(rng) < 0.4) continue;  // some tokens impossible (as after top-k / top-p)
    const double w = u(rng) + 0.05;
    d.push_back({t, w});
    sum += w;
  }
  if (d.empty()) d.push_back({0, sum = 1.0});
  for (auto& [t, w] : d) w /= sum;
  return d;
}

double prob(const ling::Dist& d, int t) {
  for (const auto& [x, p] : d)
    if (x == t) return p;
  return 0;
}

// Decodes kLen tokens speculatively with drafts of length `E`; `lookup`: drafts are fixed tokens (q = 1).
bool check_accept(bool lookup, int E, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<ling::Dist> target(kLen + E + 1), draftq(kLen + E + 1);
  for (auto& d : target) d = random_dist(rng, true);
  for (auto& d : draftq) d = random_dist(rng, false);
  std::map<std::vector<int>, long> counts;
  for (int trial = 0; trial < kTrials; ++trial) {
    std::vector<int> out;
    while (static_cast<int>(out.size()) < kLen) {
      const int pos = static_cast<int>(out.size());
      std::vector<int> drafts(E);
      for (int i = 0; i < E; ++i)
        drafts[i] = lookup ? (pos * 7 + i * 3 + trial % 2) % kVocab : ling::draw(draftq[pos + i], rng);
      std::vector<ling::Dist> P(target.begin() + pos, target.begin() + pos + E + 1);
      auto q = [&](int i, int t) { return lookup ? (t == drafts[i] ? 1.0 : 0.0) : prob(draftq[pos + i], t); };
      const auto [acc, next] = ling::accept_sampled(P, drafts, q, rng);
      out.insert(out.end(), drafts.begin(), drafts.begin() + acc);
      out.push_back(next);
    }
    out.resize(kLen);
    counts[out]++;
  }
  // Chi-square of the joint distribution of the kLen tokens against the product of the target's.
  double chi = 0;
  int cells = 0;
  std::vector<int> seq(kLen, 0);
  for (int idx = 0; idx < std::pow(kVocab, kLen); ++idx) {
    int r = idx;
    double expected = kTrials;
    for (int j = 0; j < kLen; ++j) {
      seq[j] = r % kVocab;
      r /= kVocab;
      expected *= prob(target[j], seq[j]);
    }
    const long seen = counts.count(seq) ? counts[seq] : 0;
    if (expected == 0) {
      if (seen != 0) {
        std::printf("accept (%s, E=%d): impossible sequence emitted %ld times FAIL\n", lookup ? "lookup" : "drafter", E, seen);
        return false;
      }
      continue;
    }
    chi += (seen - expected) * (seen - expected) / expected;
    ++cells;
  }
  // 99.9% quantile of chi-square with k degrees of freedom (Wilson-Hilferty).
  const double k = cells - 1, z = 3.09;
  const double limit = k * std::pow(1 - 2 / (9 * k) + z * std::sqrt(2 / (9 * k)), 3);
  const bool ok = chi < limit;
  std::printf("accept (%s, E=%d): chi-square %.1f over %d cells (99.9%% limit %.1f) %s\n", lookup ? "lookup" : "drafter",
              E, chi, cells, limit, ok ? "ok" : "FAIL");
  return ok;
}

// Tree verification's accept rule (BACKLOG 26, core/tree.hpp): a toy target whose next-token distribution depends on
// the whole prefix, a chain drawn from a draft that depends on the position and the predecessor, extra nodes (siblings
// of chain rows, some with children of their own) chosen at random; the joint distribution of the first kLen tokens
// must be the target's (chi-square), and the greedy walk must emit the target's argmax continuation.
uint64_t hash_prefix(const std::vector<int>& v) {
  uint64_t h = 1469598103934665603ull;
  for (int t : v) h = (h ^ uint64_t(t + 1)) * 1099511628211ull;
  return h;
}

bool check_tree(int E, int extra, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::map<uint64_t, ling::Dist> tcache, qcache;
  auto target = [&](const std::vector<int>& prefix) -> const ling::Dist& {
    const uint64_t h = hash_prefix(prefix);
    auto it = tcache.find(h);
    if (it != tcache.end()) return it->second;
    std::mt19937_64 r(h ^ seed);
    return tcache[h] = random_dist(r, true);
  };
  auto draftq = [&](int pos, int prev) -> const ling::Dist& {
    const uint64_t h = uint64_t(pos) * 131 + prev + 7;
    auto it = qcache.find(h);
    if (it != qcache.end()) return it->second;
    std::mt19937_64 r(h * 977 + seed);
    return qcache[h] = random_dist(r, false);
  };
  std::map<std::vector<int>, long> counts;
  bool greedy_ok = true;
  std::uniform_int_distribution<int> tok(0, kVocab - 1);
  for (int trial = 0; trial < kTrials; ++trial) {
    std::vector<int> out;
    int anchor = 0;
    while (static_cast<int>(out.size()) < kLen) {
      // The tree: the anchor (the last emitted token, or 0), the chain drawn from q, then `extra` random nodes.
      ling::DraftTree t;
      t.E = E;
      auto add = [&](int par, int token) {
        t.parent.push_back(par);
        t.depth.push_back(par < 0 ? 0 : t.depth[par] + 1);
        t.token.push_back(token);
        t.pick.push_back(0);
        t.children.emplace_back();
        if (par >= 0) t.children[par].push_back(t.rows() - 1);
      };
      add(-1, anchor);
      std::vector<int> chain;
      for (int e = 0; e < E; ++e) {
        const int d = ling::draw(draftq(static_cast<int>(out.size()) + e, e ? chain.back() : anchor), rng);
        chain.push_back(d);
        add(e, d);
      }
      for (int x = 0; x < extra; ++x) {
        std::uniform_int_distribution<int> pick(0, t.rows() - 1);
        const int par = pick(rng);
        if (t.depth[par] >= E) continue;
        const int token = tok(rng);
        bool dup = false;
        for (int c : t.children[par]) dup = dup || t.token[c] == token;
        if (!dup) add(par, token);
      }
      // Each row's target distribution: given the emitted prefix and the row's path.
      std::vector<ling::Dist> P(t.rows());
      std::vector<int> argmax(t.rows());
      for (int r = 0; r < t.rows(); ++r) {
        std::vector<int> pre = out;
        std::vector<int> path;
        for (int a = r; a > 0; a = t.parent[a]) path.push_back(t.token[a]);
        pre.insert(pre.end(), path.rbegin(), path.rend());
        P[r] = target(pre);
        argmax[r] = P[r][0].first;
        for (const auto& [x, pr] : P[r])
          if (pr > prob(P[r], argmax[r])) argmax[r] = x;
      }
      auto q = [&](int e, int token) {
        return prob(draftq(static_cast<int>(out.size()) + e, e ? chain[e - 1] : anchor), token);
      };
      if (trial < 2000) {  // greedy: the argmax walk
        int next = 0;
        const std::vector<int> path = ling::accept_tree_greedy(t, argmax.data(), next);
        // Along the path each child's token is its parent row's argmax, and next is the last row's.
        for (size_t j = 1; j < path.size(); ++j) greedy_ok &= t.token[path[j]] == argmax[path[j - 1]];
        greedy_ok &= next == argmax[path.back()];
        for (int c : t.children[path.back()]) greedy_ok &= t.token[c] != next;
      }
      int next = 0;
      const std::vector<int> path = ling::accept_tree_sampled(t, P, q, rng, next);
      for (size_t j = 1; j < path.size(); ++j) out.push_back(t.token[path[j]]);
      out.push_back(next);
      anchor = next;
    }
    out.resize(kLen);
    counts[out]++;
  }
  double chi = 0;
  int cells = 0;
  std::vector<int> seq(kLen, 0);
  for (int idx = 0; idx < std::pow(kVocab, kLen); ++idx) {
    int r = idx;
    for (int j = 0; j < kLen; ++j) {
      seq[j] = r % kVocab;
      r /= kVocab;
    }
    double expected = kTrials;
    std::vector<int> pre;
    for (int j = 0; j < kLen; ++j) {
      expected *= prob(target(pre), seq[j]);
      pre.push_back(seq[j]);
    }
    const long seen = counts.count(seq) ? counts[seq] : 0;
    if (expected == 0) {
      if (seen != 0) {
        std::printf("tree accept (E=%d, +%d): impossible sequence emitted %ld times FAIL\n", E, extra, seen);
        return false;
      }
      continue;
    }
    chi += (seen - expected) * (seen - expected) / expected;
    ++cells;
  }
  const double k = cells - 1, z = 3.09;
  const double limit = k * std::pow(1 - 2 / (9 * k) + z * std::sqrt(2 / (9 * k)), 3);
  const bool ok = chi < limit && greedy_ok;
  std::printf("tree accept (E=%d, +%d nodes): chi-square %.1f over %d cells (99.9%% limit %.1f), greedy walk %s %s\n", E,
              extra, chi, cells, limit, greedy_ok ? "ok" : "WRONG", ok ? "ok" : "FAIL");
  return ok;
}

bool check_lookup() {
  bool ok = true;
  // "a b c d e f a b c" -> the last "a b c" occurred at 0; propose "d e f a" (max 4), match 3.
  std::vector<int> s = {1, 2, 3, 4, 5, 6, 1, 2, 3};
  auto p = ling::lookup_propose(s, 4);
  ok &= p.tokens == std::vector<int>({4, 5, 6, 1}) && p.match == 3;
  // A longer agreement before the n-gram counts in `match`; the most recent occurrence wins.
  s = {9, 1, 2, 3, 7, 0, 9, 1, 2, 3, 8, 9, 1, 2, 3};
  p = ling::lookup_propose(s, 2);
  ok &= p.tokens == std::vector<int>({8, 9}) && p.match == 4;
  // No earlier occurrence: no proposal.
  p = ling::lookup_propose({1, 2, 3, 4, 5}, 4);
  ok &= p.tokens.empty();
  // The proposal stops at the end of the sequence.
  p = ling::lookup_propose({5, 5, 5, 5}, 8);
  ok &= p.tokens == std::vector<int>({5}) && p.match == 3;
  std::printf("lookup proposals %s\n", ok ? "ok" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  bool ok = check_lookup();
  ok &= check_accept(false, 1, 1);
  ok &= check_accept(false, 3, 2);
  ok &= check_accept(true, 2, 3);
  ok &= check_accept(true, 4, 4);
  ok &= check_tree(1, 3, 5);
  ok &= check_tree(3, 6, 6);
  ok &= check_tree(4, 12, 7);
  std::printf(ok ? "all spec tests passed\n" : "SPEC TESTS FAILED\n");
  return ok ? 0 : 1;
}
