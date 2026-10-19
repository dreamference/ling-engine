"""Tokens per step on bench/agentlong runs (run.py output), per run and per arm, with a turn bootstrap.

  al_stats.py A=run1.jsonl,run2.jsonl B=run3.jsonl,...

Per run: decoded tokens (generated - 1) over verify passes, pooled over turns. Per arm: pooled over its runs.
The ratio B/A gets a 95% bootstrap interval that resamples turns (the same turns for both arms), keeping
every run of a turn together, so the spread between runs and between turns is both in it.
"""
import json
import random
import sys


def load(p):
    out = {}
    for ln in open(p):
        r = json.loads(ln)
        if r.get("header") or r.get("summary") or r.get("error") or not r.get("srv"):
            continue
        g, v = r["srv"]["gen"], r["srv"]["verify"]
        if g > 1 and v > 0:
            out[r["id"]] = (g - 1, v, r["srv"]["decode_s"])
    return out


arms = {}
for a in sys.argv[1:]:
    name, files = a.split("=", 1)
    arms[name] = [load(f) for f in files.split(",")]
    for f, runs in zip(files.split(","), arms[name]):
        t = sum(x[0] for x in runs.values()); v = sum(x[1] for x in runs.values()); s = sum(x[2] for x in runs.values())
        print(f"{name} {f}: {len(runs)} turns, {t} tokens, {v} passes, {t / v:.3f} tokens/step, "
              f"{1000 * s / v:.1f} ms/step, {t / s:.1f} tok/s")
turns = sorted(set.intersection(*[set(r) for runs in arms.values() for r in runs]))


def pooled(runs, ts):
    t = sum(r[x][0] for r in runs for x in ts); v = sum(r[x][1] for r in runs for x in ts)
    return t / v


names = list(arms)
for n in names:
    print(f"{n}: {len(arms[n])} runs, {pooled(arms[n], turns):.3f} tokens/step over {len(turns)} common turns")
if len(names) >= 2:
    rng = random.Random(1)
    A, B = names[0], names[1]
    base = pooled(arms[B], turns) / pooled(arms[A], turns)
    bs = sorted(pooled(arms[B], s) / pooled(arms[A], s)
                for s in ([rng.choice(turns) for _ in turns] for _ in range(4000)))
    print(f"{B}/{A} = {base:.3f}, 95% turn bootstrap {bs[100]:.3f}-{bs[3899]:.3f}")
