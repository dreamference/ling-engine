"""Per-step tree size with a learned acceptance predictor (BACKLOG 26, follow-up 2b), offline on step dumps. For every
step, the nodes of every row bucket (adaptive_sim.py's: the chain alone, and the chain's first 8 drafts plus best-first
nodes to 16, 20, 24, 32 rows) get features from what the engine has before the verify (the drafter's lattice: the node's
own probability at T 1 and 1.5 and its rank among its parent's candidates, the path probability, the parent's and the
grandparent's, the entropy of the node's and its parent's distributions, the chain's probability at the same depth, the
depth, whether it lies on the chain, the context length) and a label (the node lies on the target's continuation, which
for a greedy step is exactly acceptance). A small model (logistic regression, or a 2-layer MLP) predicts each node's
acceptance; a bucket's predicted tokens are 1 + the sum over its nodes, and the step takes the bucket with the most
predicted tokens per ms (the staircase as adaptive_sim.py). Held out by request: each request is predicted by a model
trained on the others (and, with --train-sampled, on a sampled dump's steps too; their labels are one draw of the
continuation). Greedy steps only are scored (exact acceptance).
Usage: learned_sim.py GREEDY_DUMP [...] [--sampled DUMP] [--model logreg|mlp] [--costs ...]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tree_sim import read_steps, with_truth, build, softmax_row, lattice  # noqa: E402
from adaptive_sim import BUCKETS, attn_extra  # noqa: E402


def node_table(s, T=1.5):
    """Union of every bucket's nodes: {path: features}, per bucket the list of paths, and the label per path."""
    K16, E = s["K16"], s["E"]
    truth = s["truth"]
    chain_p = []
    prev = 0
    for e in range(E):
        p1 = softmax_row(lattice(s, e, prev, 1.0), 1.0)
        k = max(range(K16), key=lambda i: (p1[i], -i))
        chain_p.append(p1[k])
        prev = k
    nodes, buckets = {}, {}
    for b, shape in BUCKETS.items():
        t = build(s, shape, T)
        paths = {}
        for nid in sorted(t, key=lambda n: t[n][1]):
            par, e, k = t[nid]
            path = (paths[par] if par != -1 else ()) + (k,)
            paths[nid] = path
            if path not in nodes:
                prevk = path[-2] if len(path) > 1 else 0
                row1 = softmax_row(lattice(s, e, prevk, 1.0), 1.0)
                rowT = softmax_row(lattice(s, e, prevk, T), T)
                rank = sorted(range(K16), key=lambda i: -row1[i]).index(k)
                ent = -sum(x * math.log(x + 1e-30) for x in row1)
                # path probability at T 1
                lp, pp, gp, pent = 0.0, 1.0, 1.0, 0.0
                pk = 0
                for d, kk in enumerate(path):
                    r = softmax_row(lattice(s, d, pk, 1.0), 1.0)
                    lp += math.log(r[kk] + 1e-30)
                    if d == len(path) - 2:
                        pp = r[kk]
                        pent = -sum(x * math.log(x + 1e-30) for x in r)
                    if d == len(path) - 3:
                        gp = r[kk]
                    pk = kk
                on_chain = all(kk == max(range(K16), key=lambda i: (softmax_row(lattice(s, d, path[d - 1] if d else 0, 1.0), 1.0)[i], -i))
                               for d, kk in enumerate(path))
                tok_ok = all(d < len(truth) and s["cand"][d * K16 + kk] == truth[d] for d, kk in enumerate(path))
                nodes[path] = ([math.log(row1[k] + 1e-30), math.log(rowT[k] + 1e-30), lp, math.log(pp + 1e-30),
                                math.log(gp + 1e-30), ent, pent, math.log(chain_p[e] + 1e-30), e, float(on_chain),
                                float(rank == 0), float(rank), math.log(1 + s["pos"]), row1[k], math.exp(lp)],
                               float(tok_ok))
        buckets[b] = list(paths.values())
    return nodes, buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--sampled", default=None)
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--costs", default="12:100.0,16:105.1,20:105.5,24:107.0,32:118.0")
    a = ap.parse_args()
    costs = {int(k): float(v) for k, v in (x.split(":") for x in a.costs.split(","))}
    steps = []
    for p in a.dumps:
        steps += read_steps(p)
    reqs = [[s for s in r if len(s["truth"]) == s["E"] and s["greedy"]] for r in with_truth(steps)]
    reqs = [r for r in reqs if r]
    data = []  # per request: list of (nodes, buckets, step)
    for r in reqs:
        data.append([(node_table(s), s) for s in r])
    extra_X, extra_y = [], []
    if a.sampled:
        for r in with_truth(read_steps(a.sampled)):
            for s in r:
                if len(s["truth"]) != s["E"]:
                    continue
                nodes, _ = node_table(s)
                for f, y in nodes.values():
                    extra_X.append(f)
                    extra_y.append(y)
    print(f"{sum(len(r) for r in data)} greedy steps in {len(data)} requests; {len(extra_X)} sampled nodes for training")
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    tot = {"model": [0, 0.0], "oracle": [0, 0.0]}
    for b in BUCKETS:
        tot[b] = [0, 0.0]
    for hold in range(len(data)):
        X, y = list(extra_X), list(extra_y)
        for i, r in enumerate(data):
            if i == hold:
                continue
            for (nodes, _), _s in r:
                for f, lab in nodes.values():
                    X.append(f)
                    y.append(lab)
        X, y = np.array(X), np.array(y)
        sc = StandardScaler().fit(X)
        if a.model == "mlp":
            m = MLPClassifier(hidden_layer_sizes=(16,), max_iter=400, random_state=0).fit(sc.transform(X), y)
        else:
            m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
        for (nodes, bks), s in data[hold]:
            keys = list(nodes)
            pr = dict(zip(keys, m.predict_proba(sc.transform(np.array([nodes[k][0] for k in keys])))[:, 1]))
            best, bscore, ob, oscore = None, -1, None, -1
            for b, paths in bks.items():
                cost = costs[b] + attn_extra(b, s["pos"])
                acc = 1 + sum(nodes[p][1] for p in paths)
                pred = 1 + sum(pr[p] for p in paths)
                tot[b][0] += acc
                tot[b][1] += cost
                if pred / cost > bscore:
                    best, bscore = b, pred / cost
                if acc / cost > oscore:
                    ob, oscore = b, acc / cost
            for name, b in (("model", best), ("oracle", ob)):
                cost = costs[b] + attn_extra(b, s["pos"])
                tot[name][0] += 1 + sum(nodes[p][1] for p in bks[b])
                tot[name][1] += cost
    n = sum(len(r) for r in data)
    for k, (tk, ms) in tot.items():
        lab = f"fixed {k} rows" if isinstance(k, int) else k
        print(f"{lab:14s} {tk / n:.3f} tokens/step, {ms / n:.1f} ms, {1000 * tk / ms:.2f} tokens/s")


if __name__ == "__main__":
    main()
