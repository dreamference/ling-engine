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
  std::printf(ok ? "all spec tests passed\n" : "SPEC TESTS FAILED\n");
  return ok ? 0 : 1;
}
