// Tree verification of the block drafter's candidates (BACKLOG 26, after CAST, arXiv 2610.00321), the host half:
// the tree a step verifies, its device description, and the accept rules over it.
//
// The tree. Row 0 is the anchor; rows 1 .. E are the drafter's chain (the lattice walk speculate() has always
// verified), so a tree with no extra nodes is today's chain, row for row. The extra nodes follow, best first: every
// node's children at the next depth are the drafter's 16 candidates for that position, scored by the selector's
// lattice given the node's own candidate as predecessor, and a candidate's priority is its path's probability
// (the product of softmax(scores / temperature) along it). Every node's parent has a lower row index.
//
// Exactness. Each row is computed as a chain holding its path would compute it (kernels::TreeDesc), so
//   - greedy: a node survives while its token is its parent row's argmax; the accepted path is the target's own
//     greedy continuation, which is what the chain accepts plus, where the chain fails, a sibling equal to the
//     argmax and its continuation: the output is plain greedy decoding, token for token;
//   - sampled: along the chain, the standard rule (draft accepted with probability min(1, p / q), otherwise a draw
//     from the residual max(0, p - q)), with the same draws as accept_sampled(); the token drawn from the residual,
//     or from p at an off-chain node, is emitted whatever it is, and only if it equals a child's token does the walk
//     continue into that child's row (whose distribution is the target's given the emitted prefix). Every emitted
//     token is therefore drawn from the target's distribution given the tokens before it; the children only decide
//     how far one pass gets (tests/spec_tests.cpp checks the joint distribution).
#pragma once

#include <functional>
#include <random>
#include <vector>

#include "core/accept.hpp"
#include "core/kernels.cuh"

namespace ling {

struct TreeShape {
  int nodes = 0;         // extra nodes beyond the chain (0: the chain alone)
  double temp = 1.0;     // the priority's temperature over the lattice scores
  int width = 16;        // at most this many children per node
  int max_depth = 1000;  // extra nodes only at depths up to this
  int chain = 1000;      // the chain verified: its first this many drafted tokens (the rest of the block's positions
                         // can still hold extra nodes)
};

struct DraftTree {
  int E = 0;                            // the chain's drafted tokens: rows 1 .. E (at most the block's positions)
  std::vector<int> parent, depth, token, pick;  // per row; pick: the node's candidate index at its position
  std::vector<std::vector<int>> children;       // per row, in row order
  int rows() const { return static_cast<int>(parent.size()); }
};

// The chain (chain_tokens[e], chain_picks[e] for e < E), then shape.nodes extra nodes. cand [E][K16] and scores
// [E][K16][K16] as the drafter left them (scores[e][prev][k]: candidate k at position e after predecessor prev; the
// first position's row 0).
DraftTree build_tree(int anchor, const int* cand, const float* scores, int E, int K16, const std::vector<int>& chain_tokens,
                     const std::vector<int>& chain_picks, const TreeShape& shape);

// Fills the device description (kernels::TreeDesc). Returns the number of gdn paths.
int describe_tree(const DraftTree& t, kernels::TreeDesc& d);

// Greedy: argmax[r] is row r's target argmax. Returns the accepted path's rows (row 0 first); next: the token after.
std::vector<int> accept_tree_greedy(const DraftTree& t, const int* argmax, int& next);

// Sampled: P[r] is the target's distribution after row r; q(e, tok) the draft's probability of tok as the chain's
// token at depth e + 1 (the chain's rows were drawn from it). Returns the accepted path's rows; next: the token after.
std::vector<int> accept_tree_sampled(const DraftTree& t, const std::vector<Dist>& P,
                                     const std::function<double(int, int)>& q, std::mt19937_64& rng, int& next);

}  // namespace ling
