"""Paired per-turn comparison of agentlong arms against a baseline arm (run.py outputs): tokens per step, ms per
step, draft/commit ms and decode tokens/s, each as the ratio of sums over the turns both arms decoded, with a
bootstrap 95% interval over turns for the tokens-per-step and tokens/s ratios.

Usage: paired.py BASE.jsonl ARM.jsonl [ARM.jsonl ...]
"""
import json
import random
import sys


def load(p):
    recs = [json.loads(l) for l in open(p)]
    return {r["id"]: r for r in recs if not r.get("summary") and not r.get("header") and r.get("srv")}


def sums(rs):
    g = sum(r["srv"]["gen"] - 1 for r in rs)
    v = sum(r["srv"]["verify"] for r in rs)
    d = sum(r["srv"]["decode_s"] for r in rs)
    sp = [r.get("spec") or {} for r in rs]
    st = sum(x.get("steps", 0) for x in sp) or 1
    return g, v, d, 1000 * sum(x.get("draft_seconds", 0) for x in sp) / st, 1000 * sum(
        x.get("commit_seconds", 0) for x in sp) / st


def main():
    base = load(sys.argv[1])
    for p in sys.argv[2:]:
        arm = load(p)
        ids = [i for i in base if i in arm and base[i]["srv"]["gen"] > 1 and arm[i]["srv"]["gen"] > 1]
        b, a = [base[i] for i in ids], [arm[i] for i in ids]
        gb, vb, db, drb, cb = sums(b)
        ga, va, da, dra, ca = sums(a)
        rng = random.Random(1)
        tps, tok = [], []
        for _ in range(2000):
            k = [rng.randrange(len(ids)) for _ in ids]
            B, A = sums([b[j] for j in k]), sums([a[j] for j in k])
            tok.append((A[0] / A[1]) / (B[0] / B[1]))
            tps.append((A[0] / A[2]) / (B[0] / B[2]))
        tok.sort(), tps.sort()
        ci = lambda x: f"{x[50]:.3f}-{x[1949]:.3f}"
        print(f"{p}: {len(ids)} turns | tok/step {ga / va:.3f} vs {gb / vb:.3f} = x{(ga / va) / (gb / vb):.3f} "
              f"[{ci(tok)}] | ms/step {1000 * da / va:.2f} vs {1000 * db / vb:.2f} ({1000 * (da / va - db / vb):+.2f}) | "
              f"draft {dra:.2f} vs {drb:.2f} ({dra - drb:+.2f}) commit {ca:.2f} vs {cb:.2f} ({ca - cb:+.2f}) | "
              f"tok/s {ga / da:.2f} vs {gb / db:.2f} = x{(ga / da) / (gb / db):.3f} [{ci(tps)}]")


if __name__ == "__main__":
    main()
