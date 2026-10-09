"""Where a speculative step's GPU time goes (M1), from an Nsight Systems trace of ling-run --spec.

Takes the kernels launched after the first "verify" NVTX range began (the decode phase, prefill
excluded), groups them by kernel family, and divides by the number of verify ranges (steps).

Usage: step_breakdown.py TRACE.sqlite      (nsys export --type sqlite TRACE.nsys-rep)
"""
import collections
import sqlite3
import sys

GROUPS = [
    ("gemm nvfp4", lambda n: "stream_gemm_kernel" in n and "(bool)1" in n),
    ("gemm fp8", lambda n: "stream_gemm_kernel" in n and "(bool)0" in n),
    ("split-k reduce", lambda n: "split_reduce" in n),
    ("attention", lambda n: "attention_mma" in n or "attention_combine" in n or "attention_partial" in n),
    ("deltanet recurrence", lambda n: "gdn_recurrent" in n),
    ("deltanet conv/gating/norm", lambda n: "gdn_conv" in n or "gdn_gating" in n or "gated_rmsnorm" in n),
    ("bf16 narrow (beta/alpha)", lambda n: "bf16_rows" in n),
    ("drafter cuBLAS (fc, conv proj, selector)", lambda n: "cutlass" in n or "gemvx" in n or "nvjet" in n or "gemm" in n.lower()),
    ("drafter kernels", lambda n: "draft_" in n or "grouped_conv" in n or "selector" in n),
    ("top-k", lambda n: "topk" in n),
    ("activation conversion (to_half_rows, to_bf16)", lambda n: "to_half_rows" in n or "to_bf16" in n or "copy_rows_bf16" in n),
    ("norms, adds, activations", lambda n: any(k in n for k in ("rmsnorm", "add_kernel", "silu_mul", "sigmoid_mul", "embed", "attn_prepare"))),
]


def phase_split(db, names, t0, steps):
    """Kernel time by (range, family): each kernel belongs to the innermost draft/verify/commit NVTX range
    open on the CPU when it was launched (CUPTI correlation ids join kernels to their launch calls)."""
    try:
        launches = dict(db.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    except sqlite3.Error:
        return None
    ranges = []
    for name in ("draft", "verify", "commit"):
        for s, e in db.execute("SELECT start, end FROM NVTX_EVENTS WHERE (text = ? OR textId IN "
                               "(SELECT id FROM StringIds WHERE value = ?)) AND start >= ?", (name, name, t0 - 10**9)):
            ranges.append((s, e, name))
    ranges.sort()
    starts = [r[0] for r in ranges]
    import bisect
    out = collections.Counter()
    for s, e, nid, corr in db.execute("SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                                      "WHERE start >= ?", (t0,)):
        t = launches.get(corr)
        phase = "other"
        if t is not None:
            i = bisect.bisect_right(starts, t) - 1
            while i >= 0 and i >= len(ranges) - len(ranges):
                rs, re_, nm = ranges[i]
                if rs <= t <= re_:
                    phase = nm
                    break
                if t - rs > 10**9:
                    break
                i -= 1
        n = names.get(nid, str(nid))
        fam = next((g for g, f in GROUPS if f(n)), "other")
        out[(phase, fam)] += e - s
    return out


def main():
    db = sqlite3.connect(sys.argv[1])
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges = db.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE text = 'verify' OR textId IN "
        "(SELECT id FROM StringIds WHERE value = 'verify') ORDER BY start").fetchall()
    if not ranges:
        sys.exit("no verify ranges")
    t0 = ranges[0][0]
    steps = len(ranges)
    kernels = db.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? ORDER BY start",
                         (t0,)).fetchall()
    t_end = kernels[-1][1]
    by_phase = phase_split(db, names, t0, steps)
    by = collections.Counter()
    for s, e, nid in kernels:
        n = names.get(nid, str(nid))
        for g, f in GROUPS:
            if f(n):
                by[g] += e - s
                break
        else:
            by["other: " + n[:60]] += e - s
    total = sum(by.values())
    wall = t_end - t0
    print(f"steps {steps}; wall per step {wall / steps / 1e6:.1f} ms; GPU busy per step {total / steps / 1e6:.1f} ms "
          f"(idle {(wall - total) / steps / 1e6:.1f} ms)")
    for g, v in by.most_common():
        print(f"  {v / steps / 1e6:7.2f} ms  {g}")
    if by_phase:
        print("by the NVTX range that launched the kernel (draft / verify / commit), ms per step:")
        for (phase, g), v in sorted(by_phase.items(), key=lambda kv: (kv[0][0], -kv[1])):
            if v / steps / 1e6 >= 0.05:
                print(f"  {phase:7s} {v / steps / 1e6:7.2f} ms  {g}")


if __name__ == "__main__":
    main()
