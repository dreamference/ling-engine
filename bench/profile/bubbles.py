"""B08's trace (BACKLOG §19.24): where a decode step's time goes between and inside kernels.

From an Nsight Systems trace of `ling-run --spec --graphs --pdl` (nsys 2025.3.2 with --cuda-graph-trace=node, so a
graph's kernels appear one by one), the decode phase only (from the first "verify" NVTX range):
  - the GPU's busy time per step as the union of kernel intervals (two streams overlap since the commit runs beside
    the next draft, so a plain sum would count that time twice), and the idle time, the gaps where no kernel runs;
  - the gaps by size (under 2 us: launch-to-launch inside a graph; 2-20 us; 20-200 us; over 200 us: host work between
    graphs) and by the range whose launch ends them (draft, verify, commit, or between ranges);
  - the GEMM kernels' time per step and, given the bytes they stream per step (GEMM_GB, from
    `ling-stream-tests --bench-groups` at the step's rows, plus the drafter's), the rate they reach and how far it
    is from a sustained rate (SUSTAINED_GBS, one kernel streaming at its best): the small-matrix part of Z.

Usage: bubbles.py TRACE.sqlite GEMM_GB SUSTAINED_GBS
"""
import bisect
import collections
import sqlite3
import sys


def main():
    db = sqlite3.connect(sys.argv[1])
    gemm_gb, sustained = float(sys.argv[2]), float(sys.argv[3])
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges = []
    for nm in ("draft", "verify", "commit"):
        ranges += [(s, e, nm) for s, e in db.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = ? OR textId IN (SELECT id FROM StringIds WHERE value = ?)",
            (nm, nm))]
    verify = sorted(r for r in ranges if r[2] == "verify")
    if not verify:
        sys.exit("no verify ranges")
    t0, steps = verify[0][0], len(verify)
    ranges = sorted(r for r in ranges if r[0] >= t0)
    starts = [r[0] for r in ranges]
    ks = db.execute("SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? "
                    "ORDER BY start", (t0,)).fetchall()
    try:  # each kernel's launch call (a graph's kernels: the graph launch), to place it in its NVTX range
        launch_at = dict(db.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    except sqlite3.Error:
        launch_at = {}
    t_end = ks[-1][1]
    wall = t_end - t0
    busy = 0
    gaps = []
    cs, ce = ks[0][0], ks[0][1]
    for s, e, _, c in ks[1:]:
        if s > ce:
            busy += ce - cs
            gaps.append((launch_at.get(c), s - ce))  # placed by the launch of the kernel that ends the gap
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    total_k = sum(e - s for s, e, _, _ in ks)
    gemm = sum(e - s for s, e, n, _ in ks if "stream_gemm" in names.get(n, ""))
    other = collections.Counter()
    for s, e, n, _ in ks:
        nm = names.get(n, str(n))
        if "stream_gemm" not in nm:
            short = nm.removeprefix("void ").replace("ling::kernels::", "").replace("<unnamed>::", "")
            other[short.split("<")[0].split("(")[0][:50]] += e - s

    def where(t):
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            rs, re_, nm = ranges[i]
            if rs <= t <= re_:
                return nm
            if t - rs > 5 * 10**8:
                break
            i -= 1
        return "between"

    bins = [(0, 2000, "< 2 us"), (2000, 20000, "2-20 us"), (20000, 200000, "20-200 us"), (200000, 10**18, "> 200 us")]
    by_bin = collections.Counter()
    n_bin = collections.Counter()
    by_where = collections.Counter()
    for at, g in gaps:
        lab = next(b[2] for b in bins if b[0] <= g < b[1])
        by_bin[lab] += g
        n_bin[lab] += 1
        by_where[where(at) if at is not None else "unknown"] += g
    ms = lambda v: v / steps / 1e6
    print(f"steps {steps}; wall per step {ms(wall):.2f} ms; GPU busy (union) {ms(busy):.2f} ms; idle {ms(wall - busy):.2f} ms; "
          f"kernel time summed {ms(total_k):.2f} ms (overlap {ms(total_k - busy):.2f} ms)")
    print("idle gaps per step by size:")
    for _, _, lab in bins:
        print(f"  {lab:>10s}: {ms(by_bin[lab]):6.2f} ms in {n_bin[lab] / steps:6.1f} gaps")
    print("idle gaps per step by where they fall:")
    for k, v in by_where.most_common():
        print(f"  {k:>10s}: {ms(v):6.2f} ms")
    gemm_phase = collections.Counter()
    for s, e, n, c in ks:
        if "stream_gemm" in names.get(n, "") or "split_reduce" in names.get(n, ""):
            t = launch_at.get(c)
            gemm_phase[where(t) if t is not None else "unknown"] += e - s
    print("GEMM kernel time per step by the range that launched it: " +
          ", ".join(f"{k} {ms(v):.2f} ms" for k, v in gemm_phase.most_common()))
    if gemm_phase.get("verify"):
        gemm = gemm_phase["verify"]  # GEMM_GB is the verify's weight stream (bench-groups' groups)
    rate = gemm_gb / (ms(gemm) / 1e3)
    ideal = gemm_gb / sustained * 1e3
    print(f"verify GEMM kernels {ms(gemm):.2f} ms per step for {gemm_gb:.2f} GB: {rate:.0f} GB/s; at {sustained:.0f} GB/s {ideal:.2f} ms, "
          f"so {ms(gemm) - ideal:.2f} ms is the GEMMs' own shortfall (small matrices, ramps inside a launch)")
    print(f"Z estimate: idle {ms(wall - busy):.2f} ms + GEMM shortfall {ms(gemm) - ideal:.2f} ms = {ms(wall - busy) + ms(gemm) - ideal:.2f} ms")
    print("other kernels, ms per step:")
    for k, v in other.most_common(15):
        print(f"  {ms(v):6.2f}  {k}")


if __name__ == "__main__":
    main()
