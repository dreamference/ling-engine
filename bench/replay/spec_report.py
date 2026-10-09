"""Speculation counters over one replay (M1): the difference of ling-serve's /spec_stats before and after.

Prints accepted tokens per step, the acceptance histogram, the share of steps in each bin, where the
step's host-measured time goes (draft, verify + accept, commit), and the context lookup's shadow scores.

Usage: spec_report.py BEFORE.json AFTER.json [REPLAY.jsonl]
"""
import json
import sys


def main():
    a, b = (json.load(open(p)) for p in sys.argv[1:3])
    d = {k: (b[k] - a.get(k, 0)) if isinstance(b[k], (int, float)) else
         [x - y for x, y in zip(b[k], a.get(k, [0] * len(b[k])))] for k in b}
    steps = d["steps"]
    out = {"steps": steps}
    if steps:
        out["accepted_drafts_per_step"] = d["accepted"] / steps
        out["tokens_per_step"] = (d["accepted"] + steps) / steps
        out["draft_ms"] = 1000 * d["draft_seconds"] / steps
        out["verify_accept_ms"] = 1000 * d["verify_seconds"] / steps
        out["commit_ms"] = 1000 * d["commit_seconds"] / steps
        out["step_ms_host"] = out["draft_ms"] + out["verify_accept_ms"] + out["commit_ms"]
        h = d["accept_histogram"]
        last = max(i for i, c in enumerate(h) if c) if any(h) else 0
        out["accept_histogram"] = h[:last + 1]
        out["accept_share"] = [round(c / steps, 3) for c in h[:last + 1]]
    if d.get("lookup_steps"):
        out["lookup_steps"] = d["lookup_steps"]
        out["lookup_tokens_per_step"] = (d["lookup_accepted"] + d["lookup_steps"]) / d["lookup_steps"]
    if d.get("shadow_total_steps"):
        n = d["shadow_total_steps"]
        out["shadow"] = {
            "steps": n,
            "drafter_accepted_per_step": d["shadow_dflash_all"] / n,
            "best_of_both_per_step": d["shadow_best"] / n,
            "by_match": [{"match": name, "steps": s, "lookup": round(l / s, 2) if s else None,
                          "drafter": round(f / s, 2) if s else None}
                         for name, s, l, f in zip(["3", "4-7", "8-15", "16-31", "32+"], d["shadow_steps"],
                                                  d["shadow_lookup"], d["shadow_dflash"])],
        }
    if len(sys.argv) > 3:
        summ = [json.loads(ln) for ln in open(sys.argv[3]) if '"summary"' in ln]
        if summ:
            out["replay"] = {k: summ[-1][k] for k in ("accept_per_step", "server_step_ms", "server_decode_tps",
                                                      "server_mean_e2e_s", "warm_ttft_median", "wall_s")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
