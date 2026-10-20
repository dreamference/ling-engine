"""Follow-up 2b with the drafter's hidden states (LING_DUMP_HIDDEN beside LING_DUMP_STEPS, the same steps in the same
order): learned_sim.py's per-node acceptance model with, added to each node's lattice features, the leading principal
components of the drafter's normed hidden state and of its selector projection at the node's position (fitted on the
training requests only). Held out by request; greedy steps scored; prints the chooser's tokens/s against every fixed
bucket and the oracle.
Usage: learned_hidden_sim.py STEPS HIDDEN [--pcs 16] [--model logreg|mlp] [--costs ...]
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from tree_sim import read_steps, with_truth  # noqa: E402
from adaptive_sim import BUCKETS, attn_extra  # noqa: E402
from learned_sim import node_table  # noqa: E402

MAGIC = 0x48494444


def read_hidden(path):
    data = open(path, "rb").read()
    off, out = 0, []
    while off + 20 <= len(data):
        magic, pos, E, H, R = struct.unpack_from("<5i", data, off)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad record at {off}")
        off += 20
        h = np.frombuffer(data, dtype=np.float32, count=E * H, offset=off).reshape(E, H)
        off += 4 * E * H
        s = np.frombuffer(data, dtype=np.float32, count=E * R, offset=off).reshape(E, R)
        off += 4 * E * R
        out.append({"pos": pos, "h": h, "s": s})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("steps")
    ap.add_argument("hidden")
    ap.add_argument("--pcs", type=int, default=16)
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--costs", default="12:100.0,16:105.1,20:105.5,24:107.0,32:118.0")
    a = ap.parse_args()
    costs = {int(k): float(v) for k, v in (x.split(":") for x in a.costs.split(","))}
    steps = read_steps(a.steps)
    hid = read_hidden(a.hidden)
    if len(hid) != len(steps) or any(x["pos"] != s["pos"] for x, s in zip(hid, steps)):
        sys.exit("the hidden dump does not match the step dump step for step")
    for s, x in zip(steps, hid):
        s["hid"] = x
    reqs = [[s for s in r if len(s["truth"]) == s["E"] and s["greedy"]] for r in with_truth(steps)]
    reqs = [r for r in reqs if r]
    data = [[(node_table(s), s) for s in r] for r in reqs]
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    tot = {"model": [0, 0.0], "lattice only": [0, 0.0], "oracle": [0, 0.0]}
    for b in BUCKETS:
        tot[b] = [0, 0.0]
    for hold in range(len(data)):
        train = [x for i, r in enumerate(data) if i != hold for x in r]
        H = np.concatenate([s["hid"]["h"] for _, s in train])
        S = np.concatenate([s["hid"]["s"] for _, s in train])
        ph = PCA(n_components=a.pcs, random_state=0).fit(H)
        ps = PCA(n_components=a.pcs, random_state=0).fit(S)

        def feats(nodes, s, keys):
            zh, zs = ph.transform(s["hid"]["h"]), ps.transform(s["hid"]["s"])
            return np.array([nodes[k][0] + list(zh[len(k) - 1]) + list(zs[len(k) - 1]) for k in keys])

        X, Xl, y = [], [], []
        for (nodes, _), s in train:
            keys = list(nodes)
            X.append(feats(nodes, s, keys))
            Xl.append(np.array([nodes[k][0] for k in keys]))
            y += [nodes[k][1] for k in keys]
        X, Xl, y = np.concatenate(X), np.concatenate(Xl), np.array(y)
        models = {}
        for name, XX in (("model", X), ("lattice only", Xl)):
            sc = StandardScaler().fit(XX)
            m = (MLPClassifier(hidden_layer_sizes=(32,), max_iter=400, random_state=0) if a.model == "mlp"
                 else LogisticRegression(max_iter=3000, C=0.5)).fit(sc.transform(XX), y)
            models[name] = (sc, m)
        for (nodes, bks), s in data[hold]:
            keys = list(nodes)
            preds = {}
            for name, XX in (("model", feats(nodes, s, keys)), ("lattice only", np.array([nodes[k][0] for k in keys]))):
                sc, m = models[name]
                preds[name] = dict(zip(keys, m.predict_proba(sc.transform(XX))[:, 1]))
            cost = {b: costs[b] + attn_extra(b, s["pos"]) for b in bks}
            acc = {b: 1 + sum(nodes[p][1] for p in bks[b]) for b in bks}
            for b in bks:
                tot[b][0] += acc[b]
                tot[b][1] += cost[b]
            for name in ("model", "lattice only"):
                b = max(bks, key=lambda b: (1 + sum(preds[name][p] for p in bks[b])) / cost[b])
                tot[name][0] += acc[b]
                tot[name][1] += cost[b]
            b = max(bks, key=lambda b: acc[b] / cost[b])
            tot["oracle"][0] += acc[b]
            tot["oracle"][1] += cost[b]
    n = sum(len(r) for r in data)
    print(f"{n} greedy steps in {len(data)} requests, {a.pcs} components of the hidden state and the selector projection")
    for k, (tk, ms) in tot.items():
        lab = f"fixed {k} rows" if isinstance(k, int) else k
        print(f"{lab:14s} {tk / n:.3f} tokens/step, {ms / n:.1f} ms, {1000 * tk / ms:.2f} tokens/s")


if __name__ == "__main__":
    main()
