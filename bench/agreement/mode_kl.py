"""One engine mode against the exact engine (specs/DREAMFERENCE_LING_ENGINE_MILESTONES.md §14, the exactness
policy), from two `ling-force score` outputs over the same cases: the exact engine's (the reference) and the mode's.
analyze.py scores builds against production only; this is the engine-against-engine half the policy asks for.

    mode_kl.py want REF.jsonl CASES.jsonl > CASES-mode.jsonl
        The cases with `want` set to the reference's top-K ids at every position, so that `ling-force score` of the
        mode reports its log-probs of exactly those tokens.
    mode_kl.py compare REF.jsonl MODE.jsonl CASES.jsonl [--kl-max 0.0008] [--prod PROD-force.jsonl]
        KL(reference ‖ mode) over the reference's top-K, renormalized (analyze.py's estimator), top-1 flips, top-5
        agreement (the mode's top-1 inside the reference's top-5), the KL by prompt-length bucket, and PASS/FAIL for
        the lines the policy states that these files can judge: mean KL <= --kl-max, top-5 agreement 100%, and the
        longest bucket's mean KL also <= --kl-max (no growth with position). --prod adds both engines' mean KL
        against production's cold reference (the number analyze.py reports), for scale.

What this cannot judge: 64K-token positions (ling_force.cpp allocates a 40,000-token context), tool calls on the
replayed agent prompts (the replay harness scores those) and a SWE-bench night.
"""
import json, math, statistics as st, sys

K_DEFAULT = 20
BUCKETS = [(0, 1024, "<1K"), (1024, 8192, "1-8K"), (8192, 16384, "8-16K"), (16384, 65536, "16K+")]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def kl_top(p_pos, q_lp):
    """KL(P || Q) over P's listed tokens, both renormalized over that set. p_pos: [[id, lp], ...]; q_lp: id -> lp."""
    p = [(c[0], math.exp(c[1])) for c in p_pos]
    q = [(i, math.exp(q_lp[i])) for i, _ in p]
    sp, sq = sum(v for _, v in p), sum(v for _, v in q)
    return sum((pv / sp) * math.log((pv / sp) / max(qv / sq, 1e-30)) for (_, pv), (_, qv) in zip(p, q) if pv > 0)


def lp_map(pos):
    lp = {int(k): v for k, v in pos["lp"].items()}
    for t, v in pos["top"]:
        lp.setdefault(t, v)
    return lp


def want(ref_path, cases_path):
    ref, cases = load(ref_path), load(cases_path)
    if len(ref) != len(cases):
        sys.exit(f"{len(ref)} scored prompts against {len(cases)} cases")
    for r, c in zip(ref, cases):
        old = c.get("want", [])  # production's ids, when the case carries them: kept, so --prod can score the mode too
        c["want"] = [sorted(set(t for t, _ in pos["top"]) | set(old[j] if j < len(old) else []))
                     for j, pos in enumerate(r["pos"])]
        print(json.dumps(c))


def compare(ref_path, mode_path, cases_path, kl_max, prod_path):
    ref, mode, cases = load(ref_path), load(mode_path), load(cases_path)
    if not (len(ref) == len(mode) == len(cases)):
        sys.exit(f"{len(ref)} reference, {len(mode)} mode, {len(cases)} cases: not the same prompts")
    prod = load(prod_path) if prod_path else None
    rows = []  # (kl, prompt index, position, prompt length, flip, top5)
    kl_prod_ref, kl_prod_mode, prod_skipped = [], [], 0
    for i, (r, m, c) in enumerate(zip(ref, mode, cases)):
        plen = len(c["prompt"])
        for j, (pr, pm) in enumerate(zip(r["pos"], m["pos"])):
            lpm = lp_map(pm)
            missing = [t for t, _ in pr["top"] if t not in lpm]
            if missing:
                sys.exit(f"prompt {i} position {j}: the mode did not report {len(missing)} of the reference's top "
                         f"tokens; score the mode with the cases `mode_kl.py want` writes")
            kl = kl_top(pr["top"], lpm)
            ref_top = [t for t, _ in pr["top"]]
            flip = pm["top"][0][0] != ref_top[0]
            rows.append((kl, i, j, plen, flip, pm["top"][0][0] in ref_top[:5]))
            if prod is not None and j < len(prod[i]["top"]):
                pa = prod[i]["top"][j]
                lpr = lp_map(pr)
                if all(t in lpr and t in lpm for t, _ in pa):  # both engines reported production's ids
                    kl_prod_ref.append(kl_top(pa, lpr))
                    kl_prod_mode.append(kl_top(pa, lpm))
                else:
                    prod_skipped += 1
    kls = sorted(k for k, *_ in rows)
    n = len(rows)
    mean, p99, mx = st.mean(kls), kls[min(n - 1, int(math.ceil(0.99 * n)) - 1)], kls[-1]
    flips = sum(1 for *_, f, _ in rows if f)
    top5 = sum(1 for *_, t in rows if t)
    print(f"positions {n} over {len(cases)} prompts")
    print(f"KL(exact || mode) over the exact engine's top-{len(ref[0]['pos'][0]['top'])}: "
          f"mean {mean:.6f}  p99 {p99:.6f}  max {mx:.6f} nats")
    print(f"top-1 flips {flips} ({100.0 * flips / n:.2f}%), top-5 agreement {top5}/{n} ({100.0 * top5 / n:.2f}%)")
    by_bucket = {}
    for lo, hi, name in BUCKETS:
        ks = [k for k, _, _, plen, _, _ in rows if lo <= plen < hi]
        if ks:
            by_bucket[name] = st.mean(ks)
            print(f"  prompt length {name:>6}: {len(ks):5d} positions, mean KL {st.mean(ks):.6f}, max {max(ks):.6f}")
    worst = sorted(rows, key=lambda r: -r[0])[:5]
    print("  worst:", ", ".join(f"prompt {i} pos {j} ({plen} tokens) {k:.5f}" for k, i, j, plen, _, _ in worst))
    if prod is not None and kl_prod_ref:
        print(f"against production's cold reference ({len(kl_prod_ref)} positions, {prod_skipped} without both engines' "
              f"log-probs of its ids): exact engine mean KL {st.mean(kl_prod_ref):.6f}, mode mean KL "
              f"{st.mean(kl_prod_mode):.6f} (analyze.py's number for a build)")
    elif prod is not None:
        print("against production: no position had both engines' log-probs of production's ids (cases without `want`?)")
    ok_mean = mean <= kl_max
    ok_top5 = top5 == n
    longest = [name for name in [b[2] for b in BUCKETS] if name in by_bucket][-1]
    ok_long = by_bucket[longest] <= kl_max
    print(f"{'PASS' if ok_mean else 'FAIL'}: mean KL {mean:.6f} <= {kl_max}")
    print(f"{'PASS' if ok_top5 else 'FAIL'}: top-5 agreement {100.0 * top5 / n:.2f}% == 100%")
    print(f"{'PASS' if ok_long else 'FAIL'}: longest bucket ({longest}) mean KL {by_bucket[longest]:.6f} <= {kl_max} "
          f"(no growth with position, as far as these prompts reach)")
    print("MODE CHECK " + ("PASSED" if ok_mean and ok_top5 and ok_long else "FAILED"))
    return ok_mean and ok_top5 and ok_long


def main(argv):
    if len(argv) >= 4 and argv[1] == "want":
        want(argv[2], argv[3])
        return 0
    if len(argv) >= 5 and argv[1] == "compare":
        kl_max, prod = 0.0008, None
        rest = argv[5:]
        while rest:
            if rest[0] == "--kl-max":
                kl_max = float(rest[1]); rest = rest[2:]
            elif rest[0] == "--prod":
                prod = rest[1]; rest = rest[2:]
            else:
                sys.exit(f"unknown argument {rest[0]}")
        return 0 if compare(argv[2], argv[3], argv[4], kl_max, prod) else 1
    sys.exit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
