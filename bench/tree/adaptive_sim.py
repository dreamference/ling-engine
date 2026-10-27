"""Per-step tree size (BACKLOG 26, follow-up 2, after StairCut) simulated on a greedy step dump: each step verifies one of a
few row buckets (the chain alone, or the chain's first 8 drafts plus best-first nodes), chosen from the drafter's own
path probabilities and a verify-cost staircase, before the target is seen. Greedy acceptance is exact (tree_sim.py).

  predicted tokens(b) = 1 + c * sum of the path probabilities (softmax(lattice / T) along each path) of bucket b's nodes
  choice              = argmax_b predicted tokens(b) / cost(b, context)

c calibrates the probabilities (the engine would keep it as a running ratio of accepted to predicted tokens; here the
whole dump's ratio for the 16-row bucket, and the result for a range of c). cost(b, ctx) is ms per step: the table
(--costs, the 11K eager profile of 2026-10-20 by default) plus, above 16 rows, attention's second block of query rows
(bench: about +2.9 ms at 28K, +0.5 at 57K, nothing at 11K, linear between). Prints tokens per step, ms per step and
tokens/s for every fixed bucket, the chooser and an oracle (the bucket with the most tokens/s given the outcome).
Usage: adaptive_sim.py DUMP [DUMP...] [--T 1.5] [--costs 12:100.0,16:105.1,20:105.5,24:107.0,32:118.0]
"""
import argparse
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from tree_sim import read_steps, with_truth, build, accept, softmax_row, lattice  # noqa: E402

BUCKETS = {12: "chain", 16: "c8+15", 20: "c8+19", 24: "c8+23", 32: "c8+31"}


def attn_extra(rows, ctx):
    if rows <= 16:
        return 0.0
    pts = [(11000, 0.0), (28000, 2.9), (57000, 0.5), (65536, 0.5)]
    if ctx <= pts[0][0]:
        return 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if ctx <= x1:
            return y0 + (y1 - y0) * (ctx - x0) / (x1 - x0)
    return pts[-1][1]


def probsum(s, tree, T):
    prob = {-1: 1.0}
    tot = 0.0
    for nid in sorted(tree, key=lambda n: tree[n][1]):
        par, e, k = tree[nid]
        prev = 0 if par == -1 else tree[par][2]
        prob[nid] = prob[par] * softmax_row(lattice(s, e, prev, T), T)[k]
        tot += prob[nid]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--T", type=float, default=1.5)
    ap.add_argument("--costs", default="12:100.0,16:105.1,20:105.5,24:107.0,32:118.0")
    ap.add_argument("--buckets", default="12,16,20,24,32")
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--Tpred", type=float, default=None, help="the temperature of the predicted probabilities (default --T)")
    a = ap.parse_args()
    costs = {int(k): float(v) for k, v in (x.split(":") for x in a.costs.split(","))}
    bks = [int(x) for x in a.buckets.split(",")]
    steps = []
    for p in a.dumps:
        steps += read_steps(p)
    full = [s for r in with_truth(steps) for s in r if len(s["truth"]) == s["E"] and s["greedy"]]
    rec = []
    for s in full:
        row = {}
        for b in bks:
            t = build(s, BUCKETS[b], a.T)
            cost = costs[b] + (0 if a.no_attn else attn_extra(b, s["pos"]))
            row[b] = (accept(s, t) + 1, 1 + probsum(s, t, a.Tpred or a.T), cost)
        rec.append(row)
    n = len(rec)
    print(f"{n} greedy steps; context {min(s['pos'] for s in full)}-{max(s['pos'] for s in full)}")
    for b in bks:
        tok = sum(r[b][0] for r in rec)
        pred = sum(r[b][1] for r in rec)
        ms = sum(r[b][2] for r in rec)
        print(f"fixed {b:2d} rows: {tok / n:.3f} tokens/step (predicted {pred / n:.3f}, ratio {tok / pred:.3f}), "
              f"{ms / n:.1f} ms, {1000 * tok / ms:.2f} tokens/s")
    base = 16 if 16 in bks else bks[0]
    c0 = sum(r[base][0] for r in rec) / sum(r[base][1] for r in rec)
    for c in sorted({round(c0, 3), 0.6, 0.8, 1.0, 1.2}):
        tok = ms = 0
        hist = {b: 0 for b in bks}
        for r in rec:
            b = max(bks, key=lambda b: (1 + c * (r[b][1] - 1)) / r[b][2])
            hist[b] += 1
            tok += r[b][0]
            ms += r[b][2]
        print(f"chooser c={c:.3f}: {tok / n:.3f} tokens/step, {ms / n:.1f} ms, {1000 * tok / ms:.2f} tokens/s; buckets {hist}")
    tok = ms = 0
    for r in rec:
        b = max(bks, key=lambda b: r[b][0] / r[b][2])
        tok += r[b][0]
        ms += r[b][2]
    print(f"oracle: {tok / n:.3f} tokens/step, {ms / n:.1f} ms, {1000 * tok / ms:.2f} tokens/s")


if __name__ == "__main__":
    main()
