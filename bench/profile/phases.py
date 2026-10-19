"""The megakernel probe's step breakdown (BACKLOG 28): one speculative step by phase, kernel time against gaps.

From an Nsight Systems trace of `ling-run --spec --graphs --pdl` (nsys 2025.3.2, --cuda-graph-trace=node), decode
phase only (from the first "verify" range). Every kernel is placed in the top-level NVTX range its launch fell in
(draft, verify, commit; a graph's kernels by the graph launch), and inside a verify in its layer's part (the mixer of
a DeltaNet or attention layer, the FFN, the LM head) by the kernel sequence. Time is attributed by a frontier sweep
over all streams: a kernel owns the time from the later of its start and the previous frontier to its end, so the
owned times add up to the GPU's busy time (PDL's early-launched waiters own only what they extend), and the rest of
the wall time is idle, owned by the phase of the kernel whose launch ends the gap.

Usage: phases.py TRACE.sqlite [--groups GROUPS.txt] [--kernels]
"""
import bisect
import collections
import re
import sqlite3
import sys


def short(nm):
    nm = nm.removeprefix("void ").replace("ling::kernels::", "").replace("(anonymous namespace)::", "")
    nm = nm.replace("<unnamed>::", "")
    base = nm.split("(")[0]
    return base


def main():
    db = sqlite3.connect(sys.argv[1])
    show_kernels = "--kernels" in sys.argv
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges = []
    for nm in ("draft", "verify", "commit", "commit state"):
        ranges += [(s, e, nm) for s, e in db.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text = ? OR textId IN (SELECT id FROM StringIds WHERE value = ?)",
            (nm, nm))]
    verify = sorted(r for r in ranges if r[2] == "verify")
    t0 = verify[0][0]
    steps = len(verify)
    ranges = sorted(r for r in ranges if r[0] >= t0)
    starts = [r[0] for r in ranges]
    cols = [r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    gx = "gridX" if "gridX" in cols else "0"
    ks = db.execute(f"SELECT start, end, demangledName, correlationId, streamId, {gx} FROM CUPTI_ACTIVITY_KIND_KERNEL "
                    "WHERE start >= ? ORDER BY start", (t0,)).fetchall()
    launch_at = dict(db.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    try:
        memcpy = db.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ?",
                            (t0,)).fetchall()
    except sqlite3.Error:
        memcpy = []

    def where(t):
        if t is None:
            return "unknown"
        i = bisect.bisect_right(starts, t) - 1
        best = None
        while i >= 0:
            rs, re_, nm = ranges[i]
            if rs <= t <= re_:
                best = nm if best is None or nm == "commit state" else best
                if nm != "commit state":
                    return "commit" if best == "commit state" else nm
            if t - rs > 5 * 10**8:
                break
            i -= 1
        return "commit" if best == "commit state" else "host/other"

    # events: kernels and memcpys, each with phase and a label
    ev = []
    for s, e, n, c, st, g in ks:
        ev.append([s, e, short(names.get(n, str(n))), where(launch_at.get(c)), st, g])
    for s, e, c in memcpy:
        ev.append([s, e, "memcpy", where(launch_at.get(c)), -1, 0])
    ev.sort()
    # verify sub-parts: walk each verify's kernels in order; a DeltaNet mixer starts at the stream GEMM after a norm and
    # ends at the residual add after the out-projection; an attention mixer likewise; the FFN from the post norm to
    # its residual add. The labels come from the kernels that appear in the stretch.
    part = {}
    cur, seen = None, set()
    vk = [i for i, x in enumerate(ev) if x[3] == "verify"]
    layer_kinds = []
    stretch = []
    for i in vk:
        nm = ev[i][2]
        if nm.startswith("embed"):  # a new verify: what is left of the last one is its LM head and accept
            for j in stretch:
                part[j] = "lm_head/tail"
            stretch, seen = [], set()
        stretch.append(i)
        seen.add(nm)
        if nm.startswith("add_inplace") or nm.startswith("add_kernel") or nm == "add":
            kind = ("gdn" if any("gdn" in s for s in seen) else "attn" if any("attention" in s for s in seen)
                    else "ffn")
            for j in stretch:
                part[j] = kind
            stretch, seen = [], set()
    for j in stretch:
        part[j] = "lm_head/tail"
    # frontier sweep
    owned = collections.Counter()
    owned_k = collections.Counter()
    summed = collections.Counter()
    count = collections.Counter()
    idle = collections.Counter()
    idle_n = collections.Counter()
    front = ev[0][0]
    for i, (s, e, nm, ph, st, g) in enumerate(ev):
        key = ph if ph != "verify" else "verify:" + part.get(i, "?")
        if s > front:
            gap = s - front
            idle[key] += gap
            idle_n[key, "<2us" if gap < 2000 else "2-20us" if gap < 20000 else "20-200us" if gap < 200000 else ">200us"] += gap
        o = max(0, e - max(s, front))
        front = max(front, e)
        owned[key] += o
        kk = (key, nm, g)
        owned_k[kk] += o
        summed[kk] += e - s
        count[kk] += 1
    wall = ev[-1][1] - ev[0][0]
    ms = lambda v: v / steps / 1e6
    busy = sum(owned.values())
    print(f"steps {steps}; wall per step {ms(wall):.2f} ms; GPU busy {ms(busy):.2f}; idle {ms(wall - busy):.2f}")
    print(f"{'phase':22s} {'owned':>8s} {'idle':>7s}  (ms per step)")
    keys = sorted(set(owned) | set(idle), key=lambda k: -owned[k])
    for k in keys:
        print(f"{k:22s} {ms(owned[k]):8.2f} {ms(idle[k]):7.2f}")
    print("idle by size: " + ", ".join(f"{k[0]} {k[1]} {ms(v):.2f}" for k, v in sorted(idle_n.items(), key=lambda x: -x[1])[:12]))
    # by kernel class
    def klass(nm):
        if nm.startswith("stream_gemm") or nm.startswith("split_reduce"):
            return "weight GEMM (stream_gemm)"
        if nm.startswith("bf16_rows") or "narrow" in nm or "fc" in nm.lower():
            return "BF16 GEMM (bf16_rows)"
        if nm.startswith("attention") or "draft_attention" in nm:
            return "attention"
        if "gdn" in nm or "gated_rmsnorm" in nm or "conv" in nm:
            return "DeltaNet chain"
        if "norm" in nm or "half" in nm or "add" in nm or "silu" in nm or "sigmoid" in nm or "embed" in nm or "copy" in nm:
            return "norms/conversions/elementwise"
        if nm == "memcpy":
            return "memcpy"
        return "other (" + nm[:30] + ")"
    byclass = collections.Counter()
    for (key, nm, g), v in owned_k.items():
        byclass[key.split(":")[0], klass(nm)] += v
    print("owned by phase and class:")
    for (ph, kl), v in sorted(byclass.items(), key=lambda x: (x[0][0], -x[1])):
        print(f"  {ph:12s} {kl:34s} {ms(v):7.2f}")
    if show_kernels:
        print(f"{'part':22s} {'kernel':48s} {'grid':>6s} {'n/step':>7s} {'sum ms':>7s} {'owned':>7s} {'us each':>8s}")
        for kk, v in sorted(owned_k.items(), key=lambda x: (x[0][0], -summed[x[0]])):
            key, nm, g = kk
            n = count[kk] / steps
            print(f"{key:22s} {nm[:48]:48s} {g:6d} {n:7.1f} {ms(summed[kk]):7.2f} {ms(v):7.2f} {summed[kk] / count[kk] / 1e3:8.1f}")


if __name__ == "__main__":
    main()
