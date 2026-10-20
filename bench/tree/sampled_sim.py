"""Offline acceptance of the sampled tree's siblings (BACKLOG 26, follow-up 3) from a sampled step dump (LING_DUMP_STEPS,
chain steps): at every chain node a pass reached, the target's distribution there (the dumped top K under the request's
sampling, as target_dist builds it) and the drafter's q at that node (softmax of its lattice row at the request's
temperature, the distribution the chain token was drawn from) give, per number of siblings S, the probability that the
pass goes on past the node into one of its children:
  chain      the chain token alone: sum min(p, q)
  residual   today's rule: the chain token by the standard rule, else a draw from the residual max(0, p - q) matched against
             S deterministic siblings (the best of q after the chain's token), averaged over the chain's draw
  wor        S siblings drawn without replacement from q after the chain's token, the recursive rule (q renormalised after
             each draw, p replaced by its residual after each rejection); Monte Carlo
  topk       S + 1 deterministic children (q's best), point masses: p of the set
Exact for the first level only (the target is known at chain rows alone); a measure of each rule, not the engine's tokens
per step.
Usage: sampled_sim.py DUMP [--T 1.0 --top-k 20 --top-p 0.95] [--siblings 1,2,3] [--mc 400]
"""
import argparse
import math
import random
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from tree_sim import read_steps  # noqa: E402


def target(vals, ids, T, top_k, top_p):
    k = min(top_k, len(vals)) if top_k > 0 else len(vals)
    pr = [math.exp((vals[i] - vals[0]) / T) for i in range(k)]
    s = sum(pr)
    pr = [x / s for x in pr]
    keep = k
    if top_p < 1:
        c = 0
        for i in range(k):
            c += pr[i]
            if c >= top_p:
                keep = i + 1
                break
    kept = sum(pr[:keep])
    return {ids[i]: pr[i] / kept for i in range(keep)}


def soft(row, T):
    mx = max(row)
    e = [math.exp((x - mx) / T) for x in row]
    s = sum(e)
    return [x / s for x in e]


def node_rules(p, q, chain_tok, S, mc, rng):
    """q: {token: prob} over the 16 candidates; p: the target's {token: prob}."""
    toks = set(p) | set(q)
    mn = {t: min(p.get(t, 0), q.get(t, 0)) for t in toks}
    alpha = sum(mn.values())
    res = {t: max(0.0, p.get(t, 0) - q.get(t, 0)) for t in toks}
    Z = sum(res.values())
    order = sorted(q, key=lambda t: -q[t])
    out = {"chain": alpha}
    # residual rule, averaged over the chain's draw x ~ q
    r = 0.0
    for x, qx in q.items():
        a = min(1.0, p.get(x, 0) / qx) if qx > 0 else 0
        sib = [t for t in order if t != x][:S]
        r += qx * (a + (1 - a) * (sum(res[t] for t in sib) / Z if Z > 0 else 0))
    out["residual"] = r
    # top-k point masses
    out["topk"] = sum(p.get(t, 0) for t in order[:S + 1])
    # without replacement, Monte Carlo over the draws and the accept coins
    hit = 0
    for _ in range(mc):
        qq = dict(q)
        pp = dict(p)
        ok = False
        for i in range(S + 1):
            tot = sum(qq.values())
            if tot <= 0:
                break
            u = rng.random() * tot
            c = 0
            x = None
            for t, w in qq.items():
                c += w
                if u < c:
                    x = t
                    break
            if x is None:
                x = t
            qn = {t: w / tot for t, w in qq.items()}
            if rng.random() * qn[x] < pp.get(x, 0):
                ok = True
                break
            keys = set(pp) | set(qn)
            pr = {t: max(0.0, pp.get(t, 0) - qn.get(t, 0)) for t in keys}
            z = sum(pr.values())
            if z <= 0:
                break
            pp = {t: w / z for t, w in pr.items() if w > 0}
            del qq[x]
        hit += ok
    out["wor"] = hit / mc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--siblings", default="1,2,3")
    ap.add_argument("--mc", type=int, default=400)
    a = ap.parse_args()
    rng = random.Random(1)
    steps = [s for s in read_steps(a.dump) if not s["greedy"]]
    Ss = [int(x) for x in a.siblings.split(",")]
    sums = {S: {} for S in Ss}
    nodes = 0
    for s in steps:
        E, K16, K = s["E"], s["K16"], s["K"]
        cand, sc, rows = s["cand"], s["scores"], s["rows"]
        prev = 0
        for u in range(E):
            if u > s["acc"]:
                break
            row = sc[(u * K16 + (prev if u else 0)) * K16:(u * K16 + (prev if u else 0)) * K16 + K16]
            qv = soft(row, a.T)
            q = {}
            for k in range(K16):
                q[cand[u * K16 + k]] = q.get(cand[u * K16 + k], 0) + qv[k]
            p = target(s["vals"][u * K:(u + 1) * K], s["ids"][u * K:(u + 1) * K], a.T, a.top_k, a.top_p)
            for S in Ss:
                for kk, v in node_rules(p, q, rows[u + 1], S, a.mc, rng).items():
                    sums[S][kk] = sums[S].get(kk, 0) + v
            nodes += 1
            ck = [cand[u * K16 + k] for k in range(K16)]
            prev = ck.index(rows[u + 1]) if rows[u + 1] in ck else 0
    print(f"{len(steps)} sampled steps, {nodes} chain nodes reached")
    for S in Ss:
        print(f"siblings {S}: " + "  ".join(f"{k} {v / nodes:.4f}" for k, v in sums[S].items()))


if __name__ == "__main__":
    main()
