"""Offline: the drafter's sampling temperature against acceptance, from a seeded step dump (LING_DUMP_STEPS).

The verify is lossless for any proposal distribution q, as long as the draft is drawn from the q the accept
rule divides by. The engine (like SGLang) draws the selector's path from softmax(scores / T) at the request's
T. Per drafted position i of each dumped step (seeded run: the verify's top K = top_k per row), with p_i the
target's sampling distribution after the dumped prefix (top-k, then top-p, as target_dist) and the lattice
row of the dumped predecessor:

  a_tau(i)   = sum_x min(p_i(x), q_tau(x)),  q_tau = softmax(scores[i][prev] / tau)   (sampled drafting at tau)
  a_greedy(i) = p_i(argmax q)                                                          (argmax drafting)

the probability position i is accepted given the prefix. Position 1 (row 0) depends on the context only, so
its rates are exact for every policy; deeper positions are conditioned on the dumped (T-sampled) prefix, so
the chain figure sum_k prod_{i<k} a(i) is a comparator between policies, not a prediction.

Usage: draft_temp_sim.py DUMP [DUMP...] [--T 1.0] [--top-k 20] [--top-p 0.95] [--taus 0,0.3,0.5,0.7,0.85,1]
"""
import argparse
import math
import struct

MAGIC = 0x54524545


def read_steps(path):
    data = open(path, "rb").read()
    off, out = 0, []
    while off + 36 <= len(data):
        hdr = struct.unpack_from("<9i", data, off)
        if hdr[0] != MAGIC:
            raise ValueError(f"{path}: bad record at {off}")
        _, pos, anchor, E, K16, K, acc, nxt, greedy = hdr
        off += 36
        cand = struct.unpack_from(f"<{E * K16}i", data, off); off += 4 * E * K16
        scores = struct.unpack_from(f"<{E * K16 * K16}f", data, off); off += 4 * E * K16 * K16
        rows = struct.unpack_from(f"<{E + 1}i", data, off); off += 4 * (E + 1)
        vals = struct.unpack_from(f"<{(E + 1) * K}f", data, off); off += 4 * (E + 1) * K
        ids = struct.unpack_from(f"<{(E + 1) * K}i", data, off); off += 4 * (E + 1) * K
        out.append((pos, E, K16, K, acc, greedy, cand, scores, rows, vals, ids))
    return out


def target_dist(vals, ids, top_k, T, top_p):
    k = min(top_k, len(vals)) if top_k > 0 else len(vals)
    pr = [math.exp((vals[i] - vals[0]) / T) for i in range(k)]
    z = sum(pr)
    pr = [x / z for x in pr]
    keep, cum = k, 0.0
    if top_p < 1:
        for i in range(k):
            cum += pr[i]
            if cum >= top_p:
                keep = i + 1
                break
    kept = sum(pr[:keep])
    d = {}
    for i in range(keep):
        d[ids[i]] = d.get(ids[i], 0.0) + pr[i] / kept
    return d


def softmax(sc, tau):
    m = max(sc)
    e = [math.exp((x - m) / tau) for x in sc]
    z = sum(e)
    return [x / z for x in e]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--taus", default="0,0.3,0.5,0.6,0.7,0.8,0.9,1,1.2")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    n_steps, realized = 0, 0
    pos1 = {t: 0.0 for t in taus}
    chain = {t: 0.0 for t in taus}
    depth_sum = {t: [0.0] * 32 for t in taus}
    depth_n = [0] * 32
    for path in a.dumps:
        for (pos, E, K16, K, acc, greedy, cand, scores, rows, vals, ids) in read_steps(path):
            if greedy or K < 2:
                continue
            n_steps += 1
            realized += acc
            prev = 0
            rates = {t: [] for t in taus}
            for i in range(E):
                p = target_dist(vals[i * K:(i + 1) * K], ids[i * K:(i + 1) * K], a.top_k, a.T, a.top_p)
                c = cand[i * K16:(i + 1) * K16]
                base = (i * K16 + (0 if i == 0 else prev)) * K16
                sc = scores[base:base + K16]
                for t in taus:
                    if t == 0:
                        best = max(range(K16), key=lambda k: (sc[k], -k))
                        r = p.get(c[best], 0.0)
                    else:
                        q = softmax(sc, t)
                        qt = {}
                        for k in range(K16):
                            qt[c[k]] = qt.get(c[k], 0.0) + q[k]
                        r = sum(min(p.get(x, 0.0), v) for x, v in qt.items())
                    rates[t].append(r)
                depth_n[i] += 1
                for t in taus:
                    depth_sum[t][i] += rates[t][i]
                # the dumped path's pick at this depth: the predecessor of the next lattice row
                prev = next((k for k in range(K16) if c[k] == rows[i + 1]), 0)
            for t in taus:
                pos1[t] += rates[t][0]
                e, run = 0.0, 1.0
                for r in rates[t]:
                    run *= r
                    e += run
                chain[t] += e
    print(f"{n_steps} sampled steps; realized accepted drafts per step {realized / max(n_steps, 1):.3f} "
          f"(tokens per step {1 + realized / max(n_steps, 1):.3f})")
    for t in taus:
        name = "argmax" if t == 0 else f"tau={t:g}"
        print(f"{name:>9}: position 1 accepted {pos1[t] / n_steps:.4f}; chain (comparator) {1 + chain[t] / n_steps:.3f} "
              f"tokens/step; by depth " + " ".join(f"{depth_sum[t][i] / depth_n[i]:.3f}" for i in range(12) if depth_n[i]))


if __name__ == "__main__":
    main()
