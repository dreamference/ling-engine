"""Follow-up 2b, the step-level variant: instead of per-node acceptance, a model per row bucket predicts the step's
accepted tokens directly from step features (the chain's probability at each depth at T 1, their running product, the
entropies of the first four depths, every bucket's summed path probability, the context length), held out by request;
the step takes the bucket with the most predicted tokens per ms. Also the decision alone between two buckets (a
classifier: does the larger bucket give more tokens per ms on this step). Greedy steps scored.
Usage: learned_step_sim.py GREEDY_DUMP [...] [--model gbr|ridge] [--pair 12,24]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tree_sim import read_steps, with_truth, build, accept, softmax_row, lattice  # noqa: E402
from adaptive_sim import BUCKETS, attn_extra, probsum  # noqa: E402


def step_features(s):
    K16, E = s["K16"], s["E"]
    f, prev, run = [], 0, 1.0
    ents = []
    for e in range(E):
        p = softmax_row(lattice(s, e, prev, 1.0), 1.0)
        k = max(range(K16), key=lambda i: (p[i], -i))
        f.append(p[k])
        run *= p[k]
        f.append(run)
        if e < 4:
            ents.append(-sum(x * math.log(x + 1e-30) for x in p))
        prev = k
    return f + ents + [math.log(1 + s["pos"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--model", default="gbr")
    ap.add_argument("--costs", default="12:100.0,16:105.1,20:105.5,24:107.0,32:118.0")
    a = ap.parse_args()
    costs = {int(k): float(v) for k, v in (x.split(":") for x in a.costs.split(","))}
    steps = []
    for p in a.dumps:
        steps += read_steps(p)
    reqs = [[s for s in r if len(s["truth"]) == s["E"] and s["greedy"]] for r in with_truth(steps)]
    reqs = [r for r in reqs if r]
    rows = []  # (req, features, {b: (acc, cost)})
    for i, r in enumerate(reqs):
        for s in r:
            res, ps = {}, []
            for b, shape in BUCKETS.items():
                t = build(s, shape, 1.5)
                res[b] = (accept(s, t) + 1, costs[b] + attn_extra(b, s["pos"]))
                ps.append(probsum(s, t, 1.5))
            rows.append((i, step_features(s) + ps, res))
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    n = len(rows)
    tk = ms = 0
    for hold in range(len(reqs)):
        tr = [r for r in rows if r[0] != hold]
        te = [r for r in rows if r[0] == hold]
        X = np.array([r[1] for r in tr])
        models = {}
        for b in BUCKETS:
            y = np.array([r[2][b][0] for r in tr])
            models[b] = (GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                                                   random_state=0) if a.model == "gbr" else Ridge(alpha=1.0)).fit(X, y)
        for r in te:
            x = np.array([r[1]])
            b = max(BUCKETS, key=lambda b: models[b].predict(x)[0] / r[2][b][1])
            tk += r[2][b][0]
            ms += r[2][b][1]
    print(f"{n} greedy steps, {len(reqs)} requests; per-bucket {a.model}: {tk / n:.3f} tokens/step, {ms / n:.1f} ms, "
          f"{1000 * tk / ms:.2f} tokens/s")
    for b in BUCKETS:
        t = sum(r[2][b][0] for r in rows)
        m = sum(r[2][b][1] for r in rows)
        print(f"fixed {b}: {1000 * t / m:.2f} tokens/s")
    t = sum(max(r[2].values(), key=lambda v: v[0] / v[1])[0] for r in rows)
    m = sum(max(r[2].values(), key=lambda v: v[0] / v[1])[1] for r in rows)
    print(f"oracle: {1000 * t / m:.2f} tokens/s")
    for pair in ((12, 24), (16, 24)):
        t = sum(max((r[2][b] for b in pair), key=lambda v: v[0] / v[1])[0] for r in rows)
        m = sum(max((r[2][b] for b in pair), key=lambda v: v[0] / v[1])[1] for r in rows)
        print(f"oracle over {pair}: {1000 * t / m:.2f} tokens/s")


if __name__ == "__main__":
    main()
