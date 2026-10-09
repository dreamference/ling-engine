#include "core/sampling.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <vector>

namespace ling {

bool all_finite(std::span<const float> values) {
  return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

void require_finite(std::span<const float> logits) {
  if (!all_finite(logits)) throw NonFiniteLogits();
}

void require_finite_topk(std::span<const float> vals, std::span<const int> ids, int vocab) {
  for (size_t i = 0; i < vals.size() && i < ids.size(); ++i)
    if (!std::isfinite(vals[i]) || vals[i] == -FLT_MAX || ids[i] < 0 || ids[i] >= vocab) throw NonFiniteLogits();
}

int sample_logits(std::span<const float> logits, const SamplingParams& p, std::span<const int> output,
                  std::mt19937_64& rng) {
  require_finite(logits);
  const int V = static_cast<int>(logits.size());
  std::vector<float> l(logits.begin(), logits.end());
  if (p.presence_penalty != 0.f || p.repetition_penalty != 1.f) {
    std::unordered_set<int> seen(output.begin(), output.end());
    for (int t : seen) {
      if (t < 0 || t >= V) continue;
      if (p.repetition_penalty != 1.f) l[t] = l[t] > 0 ? l[t] / p.repetition_penalty : l[t] * p.repetition_penalty;
      l[t] -= p.presence_penalty;
    }
  }
  if (p.temperature <= 0.f) return static_cast<int>(std::max_element(l.begin(), l.end()) - l.begin());
  const int k = (p.top_k > 0 && p.top_k < V) ? p.top_k : V;
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
  idx.resize(k);
  std::vector<double> probs(k);
  const double mx = l[idx[0]];
  double sum = 0;
  for (int i = 0; i < k; ++i) sum += probs[i] = std::exp((l[idx[i]] - mx) / p.temperature);
  for (double& pr : probs) pr /= sum;
  int keep = k;
  if (p.top_p < 1.f) {
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      cum += probs[i];
      if (cum >= p.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  if (p.min_p > 0.f) {
    const double floor = probs[0] * p.min_p;
    int n = 0;
    while (n < keep && probs[n] >= floor) ++n;
    keep = std::max(n, 1);
  }
  std::discrete_distribution<int> dist(probs.begin(), probs.begin() + keep);
  return idx[dist(rng)];
}

}  // namespace ling
