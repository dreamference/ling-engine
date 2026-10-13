#!/usr/bin/env python3
"""The verdict of moonshot P1's gate replay (bench/copy_gate.sh): tokens per millisecond per arm and per session
against the first arm (DFlash2 alone), the outputs compared request by request, the copy counters by source.

    copy_gate_report.py OUT copy0 copy1 copy2 [...]

Decode tokens per ms counts what the agent waits for: output tokens after the first, over the time after the first
token (time to first token is prefill, which copies do not change). The gate (the owner, 2026-10-13): whole sessions
>= +7% with the output byte-identical; D1: of the two modes, the faster that keeps every output identical.
"""
import json
import statistics as st
import sys
from collections import defaultdict

GATE = 1.07


def load(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if "session" in r and "req" in r]


def rate(rows):
    toks = sum((r.get("out") or 0) - 1 for r in rows if r.get("ttft") and (r.get("out") or 0) > 1)
    secs = sum(r["t_total"] - r["ttft"] for r in rows if r.get("ttft") and (r.get("out") or 0) > 1)
    return toks / (1000 * secs) if secs else 0.0


def main():
    out, arms = sys.argv[1], sys.argv[2:]
    data = {a: load(f"{out}/{a}.jsonl") for a in arms}
    ref = arms[0]
    by_req = {a: {(r["session"], r["req"]): r for r in rows} for a, rows in data.items()}
    base = rate(data[ref])
    print(f"{ref}: {len(data[ref])} requests, {base:.4f} decode tokens/ms")
    verdict = []
    for a in arms[1:]:
        common = sorted(set(by_req[a]) & set(by_req[ref]))
        same = sum(1 for k in common if by_req[a][k].get("out_sha256") == by_req[ref][k].get("out_sha256"))
        errors = sum(1 for r in data[a] if r.get("error"))
        sessions = defaultdict(list)
        for k in common:
            sessions[k[0]].append(k)
        gains = []
        for s, keys in sessions.items():
            ra, rr = rate([by_req[a][k] for k in keys]), rate([by_req[ref][k] for k in keys])
            if ra and rr:
                gains.append(ra / rr)
        whole = rate(data[a]) / base if base else 0.0
        spec = {}
        try:
            spec = json.load(open(f"{out}/{a}-spec.json"))
        except (OSError, ValueError):
            pass
        print(f"{a}: {len(data[a])} requests ({errors} errors), {rate(data[a]):.4f} decode tokens/ms, x{whole:.3f} on the "
              f"whole replay, per session median x{st.median(gains) if gains else 0:.3f} (min {min(gains, default=0):.3f}, "
              f"max {max(gains, default=0):.3f}); outputs identical {same}/{len(common)}; tokens/step "
              f"{spec.get('tokens_per_step', 0):.2f}")
        for key in ("copy_steps", "copy_drafted", "copy_accepted", "copy_steps_by_source", "copy_accepted_by_source"):
            if key in spec:
                print(f"    {key}: {spec[key]}")
        verdict.append((whole, a, same == len(common) and errors == 0 and len(common) == len(by_req[ref])))
    exact = [v for v in verdict if v[2]]
    if not exact:
        print("verdict: no arm kept every output identical: P1 fails its gate (an exactness bug to find first)")
        return
    best = max(exact)
    print(f"verdict (D1): {best[1]} is the faster exact arm, x{best[0]:.3f}; P1's gate (whole sessions >= +7%, "
          f"byte-identical) {'PASSES' if best[0] >= GATE else 'is NOT met'}")


if __name__ == "__main__":
    main()
