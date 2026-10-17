"""Paired by request on SGLang runs: accepted tokens per verify step of arm B over arm A (warm requests, same order).
Usage: sg_paired.py A.serverlog A.jsonl B.serverlog B.jsonl [more A/B quadruples...]"""
import json, random, sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "replay"))
from server_stats import load
def per_request(log, jl):
    recs = [json.loads(l) for l in open(jl) if '"rid"' in l and '"label"' in l]
    ms = load(log, {r["rid"] for r in recs})
    out = []
    for r in recs:
        m = ms.get(r["rid"])
        if not m: out.append(None); continue
        warm = m.get("prompt_tokens") and (m.get("cached_tokens") or 0) > 0.5 * m["prompt_tokens"]
        n, s = (m.get("completion_tokens") or 0) - 1, m.get("spec_verify_ct") or 0
        out.append((n, s) if warm and n > 0 and s > 0 else None)
    return out
pairs = []
a = sys.argv[1:]
for k in range(0, len(a), 4):
    A, B = per_request(a[k], a[k + 1]), per_request(a[k + 2], a[k + 3])
    pairs += [(x, y) for x, y in zip(A, B) if x and y]
def ratio(ps):
    ta = sum(x[0] for x, _ in ps) / sum(x[1] for x, _ in ps); tb = sum(y[0] for _, y in ps) / sum(y[1] for _, y in ps)
    return tb / ta, ta, tb
r, ta, tb = ratio(pairs)
rng = random.Random(1)
bs = sorted(ratio([pairs[rng.randrange(len(pairs))] for _ in pairs])[0] for _ in range(4000))
print(f"{len(pairs)} warm pairs: A {ta:.3f} accepted/step, B {tb:.3f}; B/A {r:.3f}, 95% bootstrap {bs[100]:.3f}-{bs[3899]:.3f}")
