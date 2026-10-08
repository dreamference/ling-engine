// Exact acceptance for speculative sampling (the standard rejection rule), on the host.
#pragma once

#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace ling {

using Dist = std::vector<std::pair<int, double>>;  // (token, probability), probabilities summing to 1

// One verify's chain: P[i] is the target's distribution after row i (i = 0 .. E: row 0 is the anchor's),
// drafts[i] the token drafted for position i + 1, and q(i, t) the draft's probability of token t there.
// Draft i is accepted with probability min(1, P[i](d) / q(i, d)); on the first rejection the next token
// is drawn from max(0, P[i] - q(i, .)) normalized; if every draft is accepted it is drawn from P[E].
// Returns (accepted drafts, next token). The emitted tokens are distributed exactly as sampling from
// P directly would give.
std::pair<int, int> accept_sampled(const std::vector<Dist>& P, const std::vector<int>& drafts,
                                   const std::function<double(int, int)>& q, std::mt19937_64& rng);

// Draws from a distribution (the last token if rounding leaves u above the total).
int draw(const Dist& d, std::mt19937_64& rng);

}  // namespace ling
