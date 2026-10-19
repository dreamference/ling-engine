"""Offline acceptance of tree verification (BACKLOG 26, CAST) from a greedy step dump (LING_DUMP_STEPS).

Each dumped step holds the drafter's candidates [E][16] and lattice scores [E][16][16]; the tokens greedy
decoding really produced after each anchor follow from the steps themselves (a greedy run's output is the
target's own greedy continuation, whatever was drafted). A tree accepts its deepest node whose path equals
that continuation, which is exactly what a tree verify with argmax matching would accept, so the numbers
here are exact for greedy (no kernel involved). The chain's figure is checked against the dumped accepted
counts.

Trees are built from the lattice, as the engine's chain is: a node is (depth e, candidate k) under its
parent, its probability softmax(scores[e][parent's k] / T)[k] times its parent's. Shapes:
  chain            the engine's greedy walk (argmax per depth given the predecessor)
  best:N           best-first over path probability, N nodes (the anchor not counted)
  chain+N          the chain's E nodes, then best-first extras up to N nodes in all (the engine's LING_TREE = N - E)
  cK+N             the chain's first K nodes only (LING_TREE_CHAIN=K), then best first up to N nodes in all
  width:W          the chain plus, at every chain depth, the next W - 1 siblings as leaves
  fixed:SPEC       DARTree-like static shape: SPEC like "3,2,2,1,1" gives the children kept per depth along the
                   best path of each subtree (rank-ordered), truncated at E

Usage: tree_sim.py DUMP [DUMP...] [--T 1.0] [--shapes chain,best:16,chain+24,...] [--per-depth]
"""
import argparse
import heapq
import math
import struct
import sys

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
        out.append({"pos": pos, "anchor": anchor, "E": E, "K16": K16, "K": K, "acc": acc, "next": nxt,
                    "greedy": greedy, "cand": cand, "scores": scores, "rows": rows, "vals": vals, "ids": ids})
    return out


def with_truth(steps):
    """Attaches to every step the tokens generated after its anchor (as far as its request went)."""
    reqs, cur = [], []
    for s in steps:
        if cur and s["pos"] != cur[-1]["pos"] + cur[-1]["acc"] + 1:
            reqs.append(cur)
            cur = []
        cur.append(s)
    if cur:
        reqs.append(cur)
    for r in reqs:
        stream = []
        starts = []
        for s in r:
            starts.append(len(stream))
            stream += list(s["rows"][:s["acc"] + 1])
        stream.append(r[-1]["next"])
        for s, o in zip(r, starts):
            s["truth"] = stream[o + 1:o + 1 + s["E"]]
    return reqs


def softmax_row(sc, T):
    m = max(sc)
    e = [math.exp((x - m) / T) for x in sc]
    z = sum(e)
    return [x / z for x in e]


def lattice(s, e, prev, T):
    K16 = s["K16"]
    base = (e * K16 + (0 if e == 0 else prev)) * K16
    return s["scores"][base:base + K16]


def chain_nodes(s):
    """The engine's greedy walk: [(depth, k)]."""
    out, prev = [], 0
    for e in range(s["E"]):
        sc = lattice(s, e, prev, 1.0)
        k = max(range(s["K16"]), key=lambda i: (sc[i], -i))
        out.append(k)
        prev = k
    return out


def accept(s, tree):
    """tree: dict node_id -> (parent_id, depth, k); root id -1. Deepest node matching the truth."""
    truth = s["truth"]
    K16 = s["K16"]
    ok = {-1: True}
    best = 0
    for nid in sorted(tree, key=lambda n: tree[n][1]):
        par, e, k = tree[nid]
        m = ok.get(par, False) and e < len(truth) and s["cand"][e * K16 + k] == truth[e]
        ok[nid] = m
        if m:
            best = max(best, e + 1)
    return best


def build(s, shape, T):
    E, K16 = s["E"], s["K16"]
    tree = {}
    trunc = None
    if shape[0] == "c" and shape[1].isdigit():  # cK+N: the chain's first K nodes, then best first to N in all
        trunc = int(shape[1:].split("+")[0])
        shape = "chain+" + shape.split("+")[1]
    if shape == "chain" or shape.startswith("chain+") or shape.startswith("width:"):
        ch = chain_nodes(s)
        if trunc is not None:
            ch = ch[:trunc]
        par = -1
        for e, k in enumerate(ch):
            tree[len(tree)] = (par, e, k)
            par = len(tree) - 1
    if shape == "chain":
        return tree
    if shape.startswith("width:"):
        W = int(shape.split(":")[1])
        for e in range(E):
            par = e - 1
            prev = tree[par][2] if par >= 0 else 0
            p = softmax_row(lattice(s, e, prev, T), T)
            order = sorted(range(K16), key=lambda i: (-p[i], i))
            for k in order[:W]:
                if k != tree[e][2] and sum(1 for n in tree.values() if n[0] == par and n[1] == e) < W:
                    tree[len(tree)] = (par, e, k)
        return tree
    if shape.startswith("best:") or shape.startswith("chain+"):
        N = int(shape.split(":")[1] if shape.startswith("best:") else shape.split("+")[1])
        have = {(n[0], n[2]) for n in tree.values()}
        heap = []

        def push_children(nid, pp):
            e = 0 if nid == -1 else tree[nid][1] + 1
            if e >= E:
                return
            prev = 0 if nid == -1 else tree[nid][2]
            p = softmax_row(lattice(s, e, prev, T), T)
            for k in range(K16):
                heapq.heappush(heap, (-pp * p[k], e, k, nid))

        # Probabilities of the nodes already in (the chain), so their children can be ranked.
        prob = {-1: 1.0}
        for nid in sorted(tree):
            par, e, k = tree[nid]
            prev = 0 if par == -1 else tree[par][2]
            prob[nid] = prob[par] * softmax_row(lattice(s, e, prev, T), T)[k]
        for nid in [-1] + sorted(tree):
            push_children(nid, prob[nid])
        while len(tree) < N and heap:
            negp, e, k, par = heapq.heappop(heap)
            if (par, k) in have:
                continue
            nid = len(tree)
            tree[nid] = (par, e, k)
            have.add((par, k))
            prob[nid] = -negp
            push_children(nid, -negp)
        return tree
    if shape.startswith("fixed:"):
        spec = [int(x) for x in shape.split(":")[1].split(",")]

        def grow(par, e, prev, pp):
            if e >= E or e >= len(spec):
                return
            p = softmax_row(lattice(s, e, prev, T), T)
            order = sorted(range(K16), key=lambda i: (-p[i], i))
            for rank, k in enumerate(order[:spec[e]]):
                nid = len(tree)
                tree[nid] = (par, e, k)
                # Only the first-ranked child keeps the full width below; the others carry a chain (width 1).
                if rank == 0:
                    grow(nid, e + 1, k, pp * p[k])
                else:
                    sub = e + 1
                    pr, pk = nid, k
                    while sub < E and sub < len(spec):
                        q = softmax_row(lattice(s, sub, pk, T), T)
                        kk = max(range(K16), key=lambda i: (q[i], -i))
                        tree[len(tree)] = (pr, sub, kk)
                        pr, pk = len(tree) - 1, kk
                        sub += 1
                        if len(tree) > 64:
                            return

        grow(-1, 0, 0, 1.0)
        return tree
    raise ValueError(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--shapes", default="chain,chain+16,chain+20,chain+24,chain+31,best:11,best:16,best:24,best:31,"
                                         "width:2,width:3")
    ap.add_argument("--per-depth", action="store_true")
    a = ap.parse_args()
    steps = []
    for p in a.dumps:
        steps += read_steps(p)
    reqs = with_truth(steps)
    full = [s for r in reqs for s in r if len(s["truth"]) == s["E"] and s["greedy"]]
    print(f"{len(steps)} steps in {len(reqs)} requests; {len(full)} greedy steps with a full continuation")
    # The chain reproduces the dumped accepted counts (the simulation's check).
    mism = sum(1 for s in full if accept(s, build(s, "chain", a.T)) != s["acc"])
    print(f"chain vs dumped accepted: {mism} mismatches")
    for shape in a.shapes.split(","):
        tot = rows = 0
        hist = [0] * 33
        for s in full:
            t = build(s, shape, a.T)
            acc = accept(s, t)
            tot += acc + 1
            rows += len(t) + 1
            hist[acc] += 1
        n = len(full)
        print(f"{shape:12s} tokens/step {tot / n:.3f}  rows/step {rows / n:.1f}" +
              (f"  hist {hist[:13]}" if a.per_depth else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
