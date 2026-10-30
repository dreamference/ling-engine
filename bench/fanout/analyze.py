#!/usr/bin/env python3
"""Turn the bench JSONL files into the report's markdown tables."""

import json
import statistics as st
import sys
from pathlib import Path

# per-token KV (fp8 target 11.94 GB + bf16 draft 7.46 GB over 390981 tokens) and per mamba slot
# (conv 0.27 + ssm 6.82 GB over 96 slots), from the startup log
KV_GB = (11.94 + 7.46) / 390981
MAMBA_GB = (0.27 + 6.82) / 96  # conv + ssm state per slot; the speculative buffers are fixed
D = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def load(name):
    p = D / name
    return [json.loads(l) for l in p.open()] if p.exists() else []


def ok(r):
    return [x for x in r["reqs"] if "err" not in x]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def peak(r, k):
    vals = [t[k] for t in r["trace"] if k in t]
    return max(vals) if vals else float("nan")


def steps(r):
    s = [(x["finished"] - x["prefill_done"]) / x["verify_ct"] * 1000 for x in ok(r) if x.get("verify_ct")]
    return st.median(s) if s else float("nan")


def acc(r):
    a = [x["accept_len"] for x in ok(r) if x.get("accept_len")]
    return st.mean(a) if a else float("nan")


tails = {r["ctx"]: r for r in load("tail.jsonl")}


def tail_rate(ctx):
    r = tails.get(ctx)
    if not r:
        return None
    x = ok(r)[0]
    new = x["prompt"] - x["cached"]
    return new / (x["prefill_done"] - x["fwd_entry"])


def fanout():
    rows = load("fanout.jsonl")
    print("| ctx | N | prefill of context s | forks wall s | total s | cached/fork (min) | forks missing prefix | new tok/fork | step ms (p50) | accept | out tok/s | KV pool peak tok | mamba peak | unshared est s | speed-up |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        x = ok(r)
        w = r["warm"]
        rate = (w["prompt"]) / w["wall"]
        n = r["n"]
        new = st.mean(y["prompt"] - y["cached"] for y in x)
        missing = sum(1 for y in x if y["cached"] < r["ctx"] - 128)
        total = w["wall"] + r["wall"]
        # unshared: every fork prefills the whole context itself (compute-bound, serial),
        # decode as measured in the shared case
        uns = n * w["prompt"] / rate + r["wall"]
        kv = peak(r, "num_used_tokens")
        print(
            f"| {r['ctx']//1000}K | {n} | {w['wall']:.1f} | {r['wall']:.1f} | {total:.1f} | {min(y['cached'] for y in x)} | "
            f"{missing} | {new:.0f} | {steps(r):.0f} | {acc(r):.2f} | {n * 100 / r['wall']:.0f} | "
            f"{kv:.0f} ({kv * KV_GB:.1f} GB) | {peak(r, 'mamba_used_tokens'):.0f} ({peak(r, 'mamba_used_tokens') * MAMBA_GB:.1f} GB) | "
            f"{uns:.0f} | {uns / total:.1f}x |"
        )


def score():
    rows = load("score.jsonl")
    print("| ctx | N | forks wall s | new tok total | tail prefill tok/s | ideal s | wall/ideal | p50 e2e s |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        x = ok(r)
        new = sum(y["prompt"] - y["cached"] for y in x)
        rate = tail_rate(r["ctx"])
        ideal = new / rate if rate else float("nan")
        print(
            f"| {r['ctx']//1000}K | {r['n']} | {r['wall']:.2f} | {new} | {rate or float('nan'):.0f} | {ideal:.2f} | "
            f"{r['wall'] / ideal:.1f}x | {pct([y['e2e'] for y in x], 0.5):.2f} |"
        )


def chat():
    rows = load("chat.jsonl")
    print("| N | wall s | combined out tok/s | p50 e2e s | p95 e2e s | p50 TTFT s | p50 queue s | step ms | accept | running peak | KV peak tok |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        x = ok(r)
        e2e = [y["e2e"] for y in x]
        ttft = [y["prefill_done"] - y["received"] for y in x]
        q = [y["queue"] or 0 for y in x]
        print(
            f"| {r['n']} | {r['wall']:.1f} | {sum(y['out'] for y in x) / r['wall']:.0f} | {pct(e2e, .5):.1f} | {pct(e2e, .95):.1f} | "
            f"{pct(ttft, .5):.2f} | {pct(q, .5):.1f} | {steps(r):.0f} | {acc(r):.2f} | {peak(r, 'num_running_reqs'):.0f} | {peak(r, 'num_used_tokens'):.0f} |"
        )


def unshared():
    rows = load("unshared.jsonl")
    print("| ctx | N | wall s | cached/req | step ms | KV peak tok |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        x = ok(r)
        print(f"| {r['ctx']//1000}K | {r['n']} | {r['wall']:.1f} | {st.mean(y['cached'] for y in x):.0f} | {steps(r):.0f} | {peak(r, 'num_used_tokens'):.0f} |")


def tail():
    print("| ctx | new tokens | prefill s | tok/s at the end of the context | full-context prefill tok/s |")
    print("|---|---|---|---|---|")
    for ctx, r in sorted(tails.items()):
        x = ok(r)[0]
        w = r["warm"]
        print(f"| {ctx//1000}K | {x['prompt'] - x['cached']} | {x['prefill_done'] - x['fwd_entry']:.2f} | {tail_rate(ctx):.0f} | {w['prompt'] / w['wall']:.0f} |")


for name, f in [("fanout", fanout), ("tail", tail), ("score", score), ("chat", chat), ("unshared", unshared)]:
    print(f"\n### {name}\n")
    f()
