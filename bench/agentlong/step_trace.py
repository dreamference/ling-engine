"""Sums up a ling-serve step trace (LING_STEP_TRACE=FILE, serve/step_trace.hpp): per session, its decode passes,
how many of them were batched, its tokens and tokens per pass, and the decode time it took part in; per kind, the
units and their time. The session with the fewest tokens per second of decode wall time is the slowest.

Usage: step_trace.py TRACE [TRACE...]
"""

import collections
import json
import sys


def summarise(path):
    sessions = collections.defaultdict(lambda: {"passes": 0, "batched": 0, "tokens": 0, "ms": 0.0})
    kinds = collections.defaultdict(lambda: {"units": 0, "ms": 0.0})
    first = last = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            first = s["t"] if first is None else first
            last = s["t"]
            k = kinds[s["kind"]]
            k["units"] += 1
            k["ms"] += s["ms"]
            if s["kind"] != "decode":
                continue
            for sess, tok in zip(s["sessions"], s["tokens"]):
                e = sessions[sess]
                e["passes"] += 1
                e["batched"] += len(s["sessions"]) > 1
                e["tokens"] += tok
                e["ms"] += s["ms"]
    print(f"{path}: {sum(k['units'] for k in kinds.values())} units over {(last or 0) - (first or 0):.1f} s")
    for name, k in sorted(kinds.items()):
        print(f"  {name:12s} {k['units']:7d} units {k['ms'] / 1e3:8.1f} s")
    slowest = None
    for sess, e in sorted(sessions.items()):
        per_pass = e["tokens"] / e["passes"] if e["passes"] else 0
        rate = e["tokens"] / (e["ms"] / 1e3) if e["ms"] else 0
        print(
            f"  session {sess}: {e['passes']} passes ({e['batched']} batched), {e['tokens']} tokens, "
            f"{per_pass:.2f} tok/pass, {e['ms'] / 1e3:.1f} s in its passes ({rate:.1f} tok/s of them)"
        )
        if slowest is None or rate < slowest[1]:
            slowest = (sess, rate)
    if slowest:
        print(f"  slowest: session {slowest[0]} at {slowest[1]:.1f} tok/s of its passes' time")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        summarise(p)
