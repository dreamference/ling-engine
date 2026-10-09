#!/usr/bin/env python3
"""Checks that ling-serve turns replayed agent requests into exactly the prompt production does.

    render_test.py --bodies B --reference R --ling-template build/ling-template --tokenizer T [--show N]

B is `bench/replay/replay.py --dump-bodies` output (the Responses requests a replay sends); R is
`sglang_render.py`'s output for the same file, run inside production's SGLang image (the token ids
production's scheduler receives). The test converts every body with `ling-template --responses` and
compares token ids, request by request; it passes only if every request is identical. Recorded
sessions are private: B and R stay on the machine that holds them, so the test skips (exit 0) when
they are absent.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--ling-template", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--show", type=int, default=3, help="mismatches to print in detail")
    a = ap.parse_args()
    if not (os.path.exists(a.bodies) and os.path.exists(a.reference)):
        print("skipped: no replay bodies or reference on this machine")
        return 0
    with open(a.bodies) as f:
        out = subprocess.run([a.ling_template, "--responses", a.tokenizer], stdin=f, capture_output=True,
                             text=True, check=True).stdout
    ours = [json.loads(line) for line in out.splitlines()]
    ref = [json.loads(line) for line in open(a.reference)]
    if len(ours) != len(ref):
        print(f"FAIL: {len(ours)} renderings against {len(ref)} references")
        return 1
    bad, deltas = 0, []
    for i, (o, r) in enumerate(zip(ours, ref)):
        if "error" in o:
            bad += 1
            print(f"request {i}: ling-serve refused it: {o['error']}")
            continue
        deltas.append(len(o["ids"]) - len(r["ids"]))
        if o["ids"] == r["ids"]:
            continue
        bad += 1
        if bad <= a.show:
            x, y = r["text"], o["text"]
            k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            print(f"request {i}: {len(o['ids'])} tokens against {len(r['ids'])}; first difference at char {k}:\n"
                  f"  production {x[max(0, k - 80):k + 120]!r}\n  ling-serve {y[max(0, k - 80):k + 120]!r}")
    print(f"{len(ref) - bad}/{len(ref)} requests token-identical; token count difference median "
          f"{statistics.median(deltas) if deltas else 0}, min {min(deltas, default=0)}, max {max(deltas, default=0)}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
