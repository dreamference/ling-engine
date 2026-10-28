"""Where a C2 fused step's time goes, from an Nsight Systems trace of ling-run --fused-check (eager).

For each outer NVTX range kind (fused, verify, prefill_chunk) the kernels launched inside it (CUPTI correlation ids
join a kernel to its launch call) are grouped per range instance; per instance: the GPU span (first kernel start to
last kernel end), each stream's busy time, the union of busy time (both streams), and the idle gap inside the span.
Kernel time is also split by inner range (attention / deltanet / mlp / lm_head), by stream and by kernel family.

Usage: fused_breakdown.py TRACE.sqlite
"""
import bisect
import collections
import sqlite3
import sys

OUTER = ("fused", "verify", "prefill_chunk", "decode", "draft", "commit")
INNER = ("attention", "deltanet", "mlp", "lm_head")
FAMILIES = [
    ("fused gemm", lambda n: "fused_" in n and ("gemm" in n or "swiglu" in n)),
    ("stream gemm", lambda n: "stream_gemm" in n),
    ("prefill gemm", lambda n: "prefill" in n and "gemm" in n),
    ("attention", lambda n: "attention" in n or "attn_prepare" in n),
    ("deltanet recurrence", lambda n: "gdn_recurrent" in n),
    ("deltanet conv/gating/norm", lambda n: "gdn_conv" in n or "gdn_gating" in n or "gated_rmsnorm" in n),
    ("quant", lambda n: "quant" in n),
    ("bf16 narrow", lambda n: "bf16_rows" in n or "narrow_bf16" in n),
    ("norms/adds/act/copies", lambda n: any(k in n for k in ("rmsnorm", "add_", "silu", "sigmoid", "embed", "to_half",
                                                             "copy_rows", "memcpy"))),
]


def family(n):
    for name, f in FAMILIES:
        if f(n):
            return name
    return "other: " + n.split("(")[0][:60]


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def main(path):
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    launches = dict(db.execute("SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    ranges = []  # (start, end, name)
    for s, e, text, tid in db.execute("SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text if text is not None else strings.get(tid)
        if name in OUTER or name in INNER:
            ranges.append((s, e, name))
    ranges.sort()
    outer = [r for r in ranges if r[2] in OUTER]
    inner = [r for r in ranges if r[2] in INNER]
    o_starts = [r[0] for r in outer]
    i_starts = [r[0] for r in inner]

    def find(rs, starts, t):
        i = bisect.bisect_right(starts, t) - 1
        while i >= 0:
            if rs[i][0] <= t <= rs[i][1]:
                return i
            i -= 1
            if i >= 0 and t - rs[i][0] > 5 * 10**9:
                break
        return None

    per = collections.defaultdict(lambda: {"k": []})  # outer index -> kernels
    for s, e, nid, corr, stream in db.execute(
            "SELECT start, end, demangledName, correlationId, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        t = launches.get(corr)
        if t is None:
            continue
        oi = find(outer, o_starts, t)
        if oi is None:
            continue
        ii = find(inner, i_starts, t)
        sub = inner[ii][2] if ii is not None and outer[oi][0] <= inner[ii][0] <= outer[oi][1] else "-"
        per[oi]["k"].append((s, e, strings.get(nid, str(nid)), stream, sub))

    by_kind = collections.defaultdict(list)
    for oi, d in per.items():
        by_kind[outer[oi][2]].append((oi, d["k"]))
    for kind in ("fused", "verify", "prefill_chunk"):
        items = by_kind.get(kind, [])
        if not items:
            continue
        n = len(items)
        span = cpu = busy = 0
        streams = collections.Counter()
        fam = collections.Counter()
        sub = collections.Counter()
        subspan = collections.Counter()
        nk = 0
        for oi, ks in items:
            s0 = min(k[0] for k in ks)
            e0 = max(k[1] for k in ks)
            span += e0 - s0
            cpu += outer[oi][1] - outer[oi][0]
            busy += union([(k[0], k[1]) for k in ks])
            nk += len(ks)
            for k in ks:
                streams[k[3]] += k[1] - k[0]
                fam[(k[3], family(k[2]))] += k[1] - k[0]
                sub[(k[4], k[3])] += k[1] - k[0]
        ms = lambda x: x / n / 1e6
        print(f"== {kind}: {n} ranges, {nk / n:.0f} kernels each; GPU span {ms(span):.1f} ms, CPU range {ms(cpu):.1f} ms, "
              f"busy (union of streams) {ms(busy):.1f} ms, idle in span {ms(span - busy):.1f} ms")
        for st, t in streams.most_common():
            print(f"   stream {st}: kernel time {ms(t):.1f} ms")
        print("   by inner range and stream (kernel ms per step):")
        for (sb, st), t in sorted(sub.items(), key=lambda x: -x[1]):
            print(f"     {sb:10s} stream {st}: {ms(t):.1f}")
        print("   by family and stream (kernel ms per step):")
        for (st, f), t in sorted(fam.items(), key=lambda x: -x[1])[:24]:
            print(f"     stream {st} {f:40s} {ms(t):.2f}")
        # The critical path inside a fused step, layer by layer: time each stream's kernels sit in a gap where the
        # other stream is idle (exposed) versus overlapped.
        if kind == "fused":
            exposed = collections.Counter()
            for oi, ks in items:
                bys = collections.defaultdict(list)
                for k in ks:
                    bys[k[3]].append((k[0], k[1]))
                ss = list(bys)
                if len(ss) != 2:
                    continue
                a, b = ss
                both = union(bys[a]) + union(bys[b]) - union(bys[a] + bys[b])
                exposed[a] += union(bys[a]) - both
                exposed[b] += union(bys[b]) - both
                exposed["overlap"] += both
            print("   alone vs overlapped (ms per step): " +
                  ", ".join(f"{k if k == 'overlap' else 'stream %s alone' % k} {ms(v):.1f}" for k, v in exposed.items()))


if __name__ == "__main__":
    main(sys.argv[1])
