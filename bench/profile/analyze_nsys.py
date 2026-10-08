"""Split SGLang's speculative decode step into its parts, from an Nsight Systems trace (M0).

Input: the SQLite export of a trace taken with serve.py --nsys --nvtx-patch (`nsys export -t
sqlite`). Each GPU kernel is attributed to the NVTX phase whose host range contains the runtime
call that launched it (matched by correlation id; kernels inside a CUDA graph carry the
correlation id of the graph launch). For every `m0.step` range it then reports:

  - GPU time per phase (prepare, draft, verify, accept, mamba_commit, draft_kv_append): the union
    of that phase's kernel intervals;
  - GPU idle inside the step and between steps (the gaps a fused engine removes);
  - the step period, first kernel of one step to the first kernel of the next;
  - verify time by kernel category (GEMM, attention, DeltaNet, norms, sampling, other).

Usage: analyze_nsys.py TRACE.sqlite [--skip N] [--json OUT]
"""
import argparse
import bisect
import collections
import json
import re
import sqlite3
import statistics as st

PHASES = ["prepare", "draft", "verify", "accept", "mamba_commit", "draft_kv_append"]

CATS = [  # first match wins
    ("deltanet", r"gated_delta|gdn|delta_rule|fused_recurrent|chunk_|causal_conv|conv1d|mamba|recurrent|fla_|selective|ssm"),
    ("attention", r"flashinfer|attention|attn|BatchPrefill|BatchDecode|prefill_kernel|decode_kernel|merge_state|fmha|flash"),
    ("gemm", r"gemm|cutlass|cublas|sm1[02]0|nvjet|matmul|_mm_|fp4|fp8_|scaled_mm|Kernel2|wgmma|mma"),
    ("norm", r"rms|norm"),
    ("sampling", r"softmax|topk|top_k|sort|argmax|sampl|cumsum|multinomial|radix|exponential|uniform|rand"),
    ("quant", r"quant|scale|cvt|fp4_|e2m1"),
    ("rope", r"rope|rotary"),
]


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name, re.I):
            return c
    return "other"


def union_len(iv):
    """Total length covered by a list of (start, end) intervals."""
    if not iv:
        return 0
    iv = sorted(iv)
    tot, cs, ce = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * (len(v) - 1)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--skip", type=int, default=5, help="steps to drop at the start")
    ap.add_argument("--json")
    a = ap.parse_args()
    c = sqlite3.connect(a.db)
    strings = dict(c.execute("select id, value from StringIds"))
    tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}

    # NVTX push/pop ranges named m0.*
    nv = []
    for s, e, text, tid_, tid in c.execute(
            "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"):
        t = text if text is not None else strings.get(tid_, "")
        if t and t.startswith("m0."):
            nv.append((s, e, t[3:], tid))
    by_tid = collections.defaultdict(list)
    for r in nv:
        by_tid[r[3]].append(r)
    for t in by_tid:
        by_tid[t].sort()

    # Runtime API calls: correlation id -> (host start, thread)
    rt = {}
    for s, corr, tid in c.execute("select start, correlationId, globalTid from CUPTI_ACTIVITY_KIND_RUNTIME"):
        rt[corr] = (s, tid)

    starts = {t: [r[0] for r in v] for t, v in by_tid.items()}

    def phase_of(corr):
        if corr not in rt:
            return None, None
        hs, tid = rt[corr]
        rs = by_tid.get(tid)
        if not rs:
            return None, None
        best = None
        step = None
        i = bisect.bisect_right(starts[tid], hs) - 1
        # Ranges nest only one level (step > phase), so the few ranges just before hs suffice.
        for j in range(i, max(-1, i - 12), -1):
            s, e, name, _ = rs[j]
            if s <= hs <= e:
                if name == "step":
                    step = (s, e)
                    break
                if best is None:
                    best = (s, e, name)
        return (best[2] if best else ("step-other" if step else None)), step

    kern = []
    for s, e, corr, nid, gx, gy, gz in c.execute(
            "select start, end, correlationId, demangledName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL"):
        name = strings.get(nid, str(nid))
        # Element types make CUTLASS instances legible (e2m1 = NVFP4, e4m3 = FP8); the grid
        # separates instances of one template that serve different matrices.
        types = sorted(set(re.findall(r"float_e[245]m[0-9]_t|float_ue4m3_t|float_ue8m0_t", name)))
        tag = f" [{','.join(types)}]" if types and "cutlass" in name else ""
        kern.append((s, e, corr, f"{name[:70]}{tag} grid={gx}x{gy}x{gz}"))
    mem = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        for s, e, corr, kind in c.execute(
                "select start, end, correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
            mem.append((s, e, corr, f"memcpy{kind}"))
    ops = sorted(kern + mem)

    steps = collections.OrderedDict()  # step host range -> list of (s, e, phase, name)
    outside = []
    for s, e, corr, name in ops:
        ph, step = phase_of(corr)
        if step is None:
            outside.append((s, e, ph, name))
        else:
            steps.setdefault(step, []).append((s, e, ph, name))
    keys = list(steps)[a.skip:]
    if len(keys) < 3:
        raise SystemExit(f"only {len(keys)} steps found")

    rows = []
    for i, k in enumerate(keys):
        ks = steps[k]
        g0 = min(x[0] for x in ks)
        g1 = max(x[1] for x in ks)
        nxt = min(x[0] for x in steps[keys[i + 1]]) if i + 1 < len(keys) else None
        r = {"span": g1 - g0, "busy": union_len([(x[0], x[1]) for x in ks]),
             "period": (nxt - g0) if nxt else None, "host": k[1] - k[0]}
        for ph in PHASES + ["step-other"]:
            iv = [(x[0], x[1]) for x in ks if x[2] == ph]
            r[ph] = union_len(iv)
            r[ph + "_n"] = len(iv)
        vc = collections.Counter()
        for x in ks:
            if x[2] == "verify":
                vc[cat_of(x[3])] += x[1] - x[0]
        r["verify_cats"] = dict(vc)
        dc = collections.Counter()
        for x in ks:
            if x[2] == "draft":
                dc[cat_of(x[3])] += x[1] - x[0]
        r["draft_cats"] = dict(dc)
        rows.append(r)

    ms = lambda ns: ns / 1e6
    per = [r["period"] for r in rows if r["period"]]
    # Steps can be back to back (two continuous decode steps per scheduler pass) or separated by
    # scheduler work; the period is the honest per-step cost.
    print(f"steps analysed: {len(rows)}  (skipped {a.skip})")
    print(f"step period  median {ms(st.median(per)):.1f} ms  mean {ms(st.mean(per)):.1f}  p10 {ms(q(per,.1)):.1f}  p90 {ms(q(per,.9)):.1f}")
    print(f"GPU busy in step  median {ms(st.median([r['busy'] for r in rows])):.1f} ms; span {ms(st.median([r['span'] for r in rows])):.1f} ms; host range {ms(st.median([r['host'] for r in rows])):.1f} ms")
    print("phase GPU time (median ms, mean ms, kernels):")
    out = {"steps": len(rows), "period_ms_median": ms(st.median(per)), "period_ms_mean": ms(st.mean(per)),
           "busy_ms_median": ms(st.median([r['busy'] for r in rows])), "phases": {}}
    for ph in PHASES + ["step-other"]:
        v = [r[ph] for r in rows]
        n = st.median([r[ph + "_n"] for r in rows])
        print(f"  {ph:16s} {ms(st.median(v)):7.2f} {ms(st.mean(v)):7.2f}  {n:5.0f}")
        out["phases"][ph] = {"median_ms": ms(st.median(v)), "mean_ms": ms(st.mean(v)), "kernels": n}
    idle_in = [r["span"] - r["busy"] for r in rows]
    idle_between = [r["period"] - r["span"] for r in rows if r["period"]]
    print(f"GPU idle inside step: median {ms(st.median(idle_in)):.2f} ms mean {ms(st.mean(idle_in)):.2f}")
    print(f"GPU idle between steps: median {ms(st.median(idle_between)):.2f} ms mean {ms(st.mean(idle_between)):.2f}")
    out["idle_in_ms_mean"] = ms(st.mean(idle_in))
    out["idle_between_ms_mean"] = ms(st.mean(idle_between))
    for key in ("verify_cats", "draft_cats"):
        tot = collections.Counter()
        for r in rows:
            tot.update(r[key])
        print(f"{key} (mean ms per step, kernel time, not union):")
        out[key] = {}
        for k2, v in tot.most_common():
            print(f"  {k2:10s} {ms(v / len(rows)):7.2f}")
            out[key][k2] = ms(v / len(rows))
    # Top kernels in verify and draft by total time.
    for ph in ("verify", "draft", "accept", "mamba_commit", "draft_kv_append", "prepare", "step-other"):
        tk = collections.Counter()
        cnt = collections.Counter()
        for k in keys:
            for x in steps[k]:
                if x[2] == ph:
                    tk[x[3]] += x[1] - x[0]
                    cnt[x[3]] += 1
        print(f"top kernels in {ph} (mean ms/step, launches/step):")
        for name, v in tk.most_common(16):
            print(f"  {ms(v / len(rows)):7.3f} {cnt[name] / len(rows):6.1f}  {name}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
