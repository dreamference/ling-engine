#include "core/accept.hpp"

#include <algorithm>

namespace ling {

int draw(const Dist& d, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const double u = uni(rng);
  double cum = 0;
  for (const auto& [t, pr] : d) {
    cum += pr;
    if (u < cum) return t;
  }
  return d.back().first;
}

std::pair<int, int> accept_sampled(const std::vector<Dist>& P, const std::vector<int>& drafts,
                                   const std::function<double(int, int)>& q, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const int E = static_cast<int>(drafts.size());
  for (int i = 0; i < E; ++i) {
    const int tok = drafts[i];
    double pd = 0;
    for (const auto& [t, pr] : P[i])
      if (t == tok) pd = pr;
    const double qd = q(i, tok);
    if (uni(rng) * qd < pd) continue;  // accepted with probability min(1, p / q)
    Dist R;
    double total = 0;
    for (const auto& [t, pr] : P[i]) {
      const double r = std::max(0.0, pr - q(i, t));
      if (r > 0) {
        R.push_back({t, r});
        total += r;
      }
    }
    if (total <= 0) return {i, draw(P[i], rng)};
    for (auto& [t, r] : R) r /= total;
    return {i, draw(R, rng)};
  }
  return {E, draw(P[E], rng)};
}

}  // namespace ling
