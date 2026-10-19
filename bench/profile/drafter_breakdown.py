"""The drafter's GPU time per speculative step, kernel by kernel, from an Nsight Systems trace of ling-run --spec
(eager, or --graphs traced with --cuda-graph-trace=node).

Each kernel belongs to the "draft" or "commit" NVTX range open on the CPU when it (or its graph) was launched.
Every range instance launches the same kernel sequence, so kernels are averaged by position in it and labelled
with the role draft_launch gives that position (per layer: conv projection, q/k/v, o, gate/up, down, ...).
Printed: per role the ms per step (exclusive: under PDL a kernel starts early and waits for its predecessor, so
each is charged from the later of its start and the previous kernel's end), the sum of kernel durations, the range's GPU span (first start to last end) and
the gaps inside it (launch and dependency bubbles).

Usage: drafter_breakdown.py TRACE.sqlite [--positions]     (nsys export --type sqlite TRACE.nsys-rep)
"""
import bisect
import collections
import sqlite3
import sys


def role(name, prev_roles):
    """The role of a draft-range kernel from its name and the roles before it in the same instance."""
    n = name
    gemms = sum(1 for r in prev_roles if r.startswith("nvfp4 ") and "LM head" not in r and "reduce" not in r)
    if "split_reduce" in n:  # a single-matrix stream GEMM's split-K reduction: the GEMM before it
        return (prev_roles[-1] if prev_roles and prev_roles[-1].startswith("nvfp4") else "nvfp4 ?") + " split-K reduce"
    bf16 = sum(1 for r in prev_roles if r.startswith("bf16 "))
    if "stream_gemm" in n:
        # Per layer: qkv (multi), o, gate+up (multi), down; then the LM head.
        order = ["nvfp4 q/k/v", "nvfp4 o", "nvfp4 gate/up", "nvfp4 down"]
        return order[gemms % 4] if gemms < 20 else "nvfp4 LM head"
    if "prefill_gemm" in n:  # LING_DRAFT_ACT=4: one GEMM per matrix
        order = ["nvfp4 q/k/v", "nvfp4 q/k/v", "nvfp4 q/k/v", "nvfp4 o", "nvfp4 gate/up", "nvfp4 gate/up", "nvfp4 down"]
        return (order[gemms % 7] if gemms < 35 else "nvfp4 LM head") + " (W4A4)"
    if "bf16_rows" in n:
        return "bf16 selector" if bf16 >= 10 else "bf16 conv kproj"
    if "topk" in n:
        return "top-k"
    if "selector_lattice" in n or "selector" in n:
        return "selector lattice"
    if "draft_attention" in n:
        return "attention"
    if "grouped_conv" in n:
        return "grouped conv"
    if "quant_nvfp4" in n or "norm_quant" in n:
        return "act quantize (W4A4)"
    if "to_half" in n:
        return "to_half_rows"
    return "small (norm/add/rope/silu/embed)"


def commit_role(name):
    if "gemm_bf16" in name or "cutlass" in name or "nvjet" in name or "gemvx" in name or "bf16_rows" in name:
        return "bf16 fc (materialize)"
    if "stream_gemm" in name or "gemm_nvfp4" in name:
        return "nvfp4 k/v (materialize)"
    if "gdn" in name:
        return "target DeltaNet commit"
    return "small"


def main():
    db = sqlite3.connect(sys.argv[1])
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    launches = dict(db.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    ranges = []
    for nm in ("draft", "commit", "verify"):
        for s, e in db.execute("SELECT start, end FROM NVTX_EVENTS WHERE (text = ? OR textId IN "
                               "(SELECT id FROM StringIds WHERE value = ?)) AND end > start", (nm, nm)):
            ranges.append((s, e, nm))
    ranges.sort()
    starts = [r[0] for r in ranges]
    inst = collections.defaultdict(list)  # range index -> [(start, end, name)]
    for s, e, nid, corr in db.execute("SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        t = launches.get(corr)
        if t is None:
            continue
        i = bisect.bisect_right(starts, t) - 1
        # innermost open range at the launch: walk back over ranges that started before t
        j = i
        while j >= 0 and not (ranges[j][0] <= t <= ranges[j][1]):
            if t - ranges[j][0] > 5 * 10**9:
                j = -1
                break
            j -= 1
        if j >= 0:
            inst[j].append((s, e, names.get(nid, str(nid))))
    # The decode's steps: instances whose kernel count is the modal count for their range name.
    for nm in ("draft", "commit"):
        ids = [j for j in inst if ranges[j][2] == nm]
        if not ids:
            continue
        cnt = collections.Counter(len(inst[j]) for j in ids)
        modal = cnt.most_common(1)[0][0]
        use = [j for j in ids if len(inst[j]) == modal]
        use = use[2:] if len(use) > 6 else use  # the first steps include first-launch effects
        per_role = collections.Counter()
        pos_ms = collections.defaultdict(float)
        pos_name = {}
        span = gaps = busy = 0.0
        for j in use:
            ks = sorted(inst[j])
            roles = []
            prev_end = ks[0][0]
            for p, (s, e, n) in enumerate(ks):
                r = role(n, roles) if nm == "draft" else commit_role(n)
                roles.append(r)
                # Exclusive time: from the later of its start and the previous kernel's end to its end. Under PDL a
                # kernel starts before its predecessor ends and waits in pdl_begin, so its own duration overstates.
                per_role[r] += max(0, e - max(s, prev_end)) / 1e6
                prev_end = max(prev_end, e)
                pos_ms[p] += (e - s) / 1e6
                pos_name[p] = (r, n[:90])
            busy += sum(e - s for s, e, _ in ks) / 1e6
            span += (ks[-1][1] - ks[0][0]) / 1e6
            # gaps: the span not covered by any kernel (kernels may overlap under PDL)
            cov, cur_s, cur_e = 0, ks[0][0], ks[0][1]
            for s, e, _ in ks[1:]:
                if s > cur_e:
                    cov += cur_e - cur_s
                    cur_s, cur_e = s, e
                else:
                    cur_e = max(cur_e, e)
            cov += cur_e - cur_s
            gaps += ((ks[-1][1] - ks[0][0]) - cov) / 1e6
        k = len(use)
        print(f"== {nm}: {k} steps of {modal} kernels (counts seen {dict(cnt)})")
        for r, ms in sorted(per_role.items(), key=lambda x: -x[1]):
            print(f"  {r:40s} {ms / k:7.3f} ms")
        print(f"  {'kernel time (overlapping under PDL)':40s} {busy / k:7.3f} ms")
        print(f"  {'GPU span (first start to last end)':40s} {span / k:7.3f} ms")
        print(f"  {'gaps inside the span':40s} {gaps / k:7.3f} ms")
        if "--positions" in sys.argv:
            for p in sorted(pos_ms):
                print(f"    {p:3d} {pos_ms[p] / k * 1000:8.1f} us  {pos_name[p][0]:28s} {pos_name[p][1]}")
    # The whole step: verify ranges for the step count and the span from one draft's start to the next.
    v = [j for j in inst if ranges[j][2] == "verify"]
    if v:
        vt = sum(sum(e - s for s, e, _ in inst[j]) for j in v) / 1e6 / len(v)
        print(f"== verify kernel time {vt:.3f} ms per step over {len(v)} steps")


if __name__ == "__main__":
    main()
