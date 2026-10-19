#include "core/tree.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ling {
namespace {

// softmax(row / T) over the K16 candidates.
std::vector<double> soft(const float* row, int K16, double T) {
  std::vector<double> p(K16);
  double mx = row[0];
  for (int k = 1; k < K16; ++k) mx = std::max(mx, double(row[k]));
  double z = 0;
  for (int k = 0; k < K16; ++k) z += p[k] = std::exp((row[k] - mx) / T);
  for (double& v : p) v /= z;
  return p;
}

const float* lattice(const float* scores, int e, int prev_pick, int K16) {
  return scores + (size_t(e) * K16 + (e == 0 ? 0 : prev_pick)) * K16;
}

}  // namespace

DraftTree build_tree(int anchor, const int* cand, const float* scores, int E, int K16, const std::vector<int>& chain_tokens,
                     const std::vector<int>& chain_picks, const TreeShape& shape) {
  DraftTree t;
  const int P = E;  // the block's drafted positions: the tree's deepest level
  E = std::min(E, std::max(1, shape.chain));
  t.E = E;
  auto add = [&](int par, int tok, int pick) {
    t.parent.push_back(par);
    t.depth.push_back(par < 0 ? 0 : t.depth[par] + 1);
    t.token.push_back(tok);
    t.pick.push_back(pick);
    t.children.emplace_back();
    if (par >= 0) t.children[par].push_back(t.rows() - 1);
    return t.rows() - 1;
  };
  add(-1, anchor, 0);
  std::vector<double> prob(1, 1.0);
  for (int e = 0; e < E; ++e) {
    add(e, chain_tokens[e], chain_picks[e]);
    prob.push_back(prob[e] * soft(lattice(scores, e, t.pick[e], K16), K16, shape.temp)[chain_picks[e]]);
  }
  const int limit = std::min(kernels::kMaxStreamRows, E + 1 + shape.nodes);  // E + 1 + nodes rows
  if (t.rows() >= limit) return t;
  // Best first over the children of every node (max-heap on the path probability; ties to the lower row, then
  // the lower candidate).
  using Item = std::tuple<double, int, int>;  // (probability, -parent, -pick)
  std::priority_queue<Item> heap;
  std::set<std::pair<int, int>> have;
  for (int r = 1; r <= E; ++r) have.insert({t.parent[r], t.pick[r]});
  auto push_children = [&](int r) {
    const int e = t.depth[r];  // the children's position in the block
    if (e >= P || e + 1 > shape.max_depth) return;
    const std::vector<double> p = soft(lattice(scores, e, t.pick[r], K16), K16, shape.temp);
    for (int k = 0; k < K16; ++k)
      if (!have.count({r, k})) heap.push({prob[r] * p[k], -r, -k});
  };
  for (int r = 0; r < t.rows(); ++r) push_children(r);
  while (t.rows() < limit && !heap.empty()) {
    const auto [pr, nr, nk] = heap.top();
    heap.pop();
    const int par = -nr, k = -nk;
    if (have.count({par, k}) || static_cast<int>(t.children[par].size()) >= shape.width) continue;
    have.insert({par, k});
    const int r = add(par, cand[size_t(t.depth[par]) * K16 + k], k);
    prob.push_back(pr);
    t.extra_mass += pr;
    push_children(r);
  }
  return t;
}

int describe_tree(const DraftTree& t, kernels::TreeDesc& d) {
  const int n = t.rows();
  if (n > kernels::kMaxStreamRows) throw std::runtime_error("tree: more rows than kMaxStreamRows");
  d = kernels::TreeDesc{};
  d.n = n;
  // Variants: one per leaf, in row order (the chain's end, row E, is the first leaf: variant 0 is the chain).
  std::vector<int> leaf_var(n, -1);
  for (int r = 0; r < n; ++r) {
    d.depth[r] = t.depth[r];
    if (!t.children[r].empty()) continue;
    if (d.nvar >= kernels::kMaxTreeVariants) throw std::runtime_error("tree: too many leaves");
    const int v = d.nvar++;
    leaf_var[r] = v;
    for (int a = r; a >= 0; a = t.parent[a]) d.path[v][t.depth[a]] = a;  // the rest stay 0 (the anchor, masked)
  }
  // A row's variant: the leaf reached through first children (a chain row's first child is the chain's next row).
  for (int r = 0; r < n; ++r) {
    int a = r;
    while (!t.children[a].empty()) a = t.children[a][0];
    d.var[r] = leaf_var[a];
  }
  // gdn paths: the chain first, then a path to each internal off-chain node not yet covered, continued through
  // first internal children. Along a path every node advances the state; the leaf children of a node follow it
  // (computed from its state, not advancing it). Each row is written once.
  std::vector<char> written(n, 0), covered(n, 0);
  auto emit = [&](int p, const std::vector<int>& nodes) {
    int len = 0;
    for (int w : nodes) {
      d.gdn_step[p][len++] = w | (written[w] ? 0 : kernels::TreeDesc::kWrite);
      written[w] = 1;
      covered[w] = 1;
      for (int c : t.children[w])
        if (t.children[c].empty() && !written[c]) {
          d.gdn_step[p][len++] = c | kernels::TreeDesc::kLeaf | kernels::TreeDesc::kWrite;
          written[c] = 1;
          covered[c] = 1;
        }
    }
    d.gdn_len[p] = len;
  };
  std::vector<int> chain;
  for (int r = 0; r <= t.E; ++r) chain.push_back(r);
  emit(d.ngdn++, chain);
  for (int r = 0; r < n; ++r) {
    if (covered[r] || t.children[r].empty()) continue;
    std::vector<int> nodes;
    for (int a = r; a >= 0; a = t.parent[a]) nodes.push_back(a);
    std::reverse(nodes.begin(), nodes.end());
    for (int a = r;;) {  // down through first internal children
      int next = -1;
      for (int c : t.children[a])
        if (!t.children[c].empty()) {
          next = c;
          break;
        }
      if (next < 0) break;
      nodes.push_back(next);
      a = next;
    }
    if (d.ngdn >= kernels::kMaxTreeVariants) throw std::runtime_error("tree: too many gdn paths");
    emit(d.ngdn++, nodes);
  }
  for (int r = 0; r < n; ++r)
    if (!written[r]) throw std::runtime_error("tree: a row no gdn path writes");
  return d.ngdn;
}

std::vector<int> accept_tree_greedy(const DraftTree& t, const int* argmax, int& next) {
  std::vector<int> path{0};
  int u = 0;
  for (;;) {
    const int x = argmax[u];
    int child = -1;
    for (int c : t.children[u])
      if (t.token[c] == x) {
        child = c;
        break;
      }
    if (child < 0) {
      next = x;
      return path;
    }
    path.push_back(child);
    u = child;
  }
}

std::vector<int> accept_tree_sampled(const DraftTree& t, const std::vector<Dist>& P,
                                     const std::function<double(int, int)>& q, std::mt19937_64& rng, int& next) {
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::vector<int> path{0};
  int u = 0;
  for (;;) {
    int x;
    if (u < t.E && t.parent[u + 1] == u) {  // a chain row below its end: the chain's next row is its child u + 1
      const int tok = t.token[u + 1];
      double pd = 0;
      for (const auto& [tk, pr] : P[u])
        if (tk == tok) pd = pr;
      const double qd = q(u, tok);
      if (uni(rng) * qd < pd) {  // accepted with probability min(1, p / q)
        path.push_back(u + 1);
        u = u + 1;
        continue;
      }
      Dist R;
      double total = 0;
      for (const auto& [tk, pr] : P[u]) {
        const double r = std::max(0.0, pr - q(u, tk));
        if (r > 0) {
          R.push_back({tk, r});
          total += r;
        }
      }
      if (total <= 0) {
        x = draw(P[u], rng);
      } else {
        for (auto& [tk, r] : R) r /= total;
        x = draw(R, rng);
      }
    } else {
      x = draw(P[u], rng);
    }
    int child = -1;
    for (int c : t.children[u])
      if (t.token[c] == x && !(u < t.E && c == u + 1)) {
        child = c;
        break;
      }
    if (child < 0) {
      next = x;
      return path;
    }
    path.push_back(child);
    u = child;
  }
}

}  // namespace ling
