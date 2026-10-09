"""Agreement study analysis: P1-P5. Reads the files run.sh writes in the study directory; prints the numbers and
writes results.json there.

    analyze.py STUDY_DIR [VALIDATE_DIR]

VALIDATE_DIR (optional) holds validate_v0.py outputs v2-{long,short}-bin-{m1,seq,chunked}.txt for P5's comparison under
validate_v0.py's own protocol. The six-prompt counts below are M1's and M2's on validate_v0.py's prompts (98.6% and
96.2% of 288 tokens).
"""
import json, math, os, re, statistics as st, sys
from collections import Counter, defaultdict

D = sys.argv[1].rstrip("/") + "/"
VDIR = sys.argv[2] if len(sys.argv) > 2 else None
sets = json.load(open(D + "sets.json"))
ids = [json.loads(l)["ids"] for l in open(D + "ids.jsonl")]
gen = [json.loads(l) for l in open(D + "prod-gen48.jsonl")]
ref = [json.loads(l) for l in open(D + "prod-force-cold-c1.jsonl")]
passes = {k: [json.loads(l) for l in open(D + f"prod-force-{k}.jsonl")] for k in ["warm-c1", "cold-c4", "warm-c4"]}
BUILDS = [("m1", "M1"), ("b44", "44c92c5"), ("m2", "M2 final")]
eng = {}
for b, _ in BUILDS:
    try:
        eng[b] = [json.loads(l)["pos"] for l in open(D + f"eng-{b}.jsonl")]
    except FileNotFoundError:
        pass
out = {}


def top1(pos):  # production entry: list of [id, lp] (unsorted?) -> sorted
    s = sorted(pos, key=lambda c: -c[1])
    return s


def kl_top(prod_pos, q_lp):
    """KL(P || Q) over production's top-20 tokens, both renormalized over that set. q_lp: id -> logprob."""
    p = [(c[0], math.exp(c[1])) for c in prod_pos]
    q = [(i, math.exp(q_lp[i])) for i, _ in p]
    sp, sq = sum(v for _, v in p), sum(v for _, v in q)
    return sum((pv / sp) * (math.log((pv / sp) / max(qv / sq, 1e-30))) for (_, pv), (_, qv) in zip(p, q) if pv > 0), sp


def pct(xs, q):
    xs = sorted(xs)
    if not xs: return float("nan")
    k = min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)
    return xs[max(k, 0)]


# ---- P2: production against itself ----
p2 = {}
self_flips_all = []
for k, pr in passes.items():
    flips, n, kls = [], 0, []
    for i, (a, b) in enumerate(zip(ref, pr)):
        for j, (pa, pb) in enumerate(zip(a["top"], b["top"])):
            n += 1
            sa, sb = top1(pa), top1(pb)
            qmap = {c[0]: c[1] for c in pb}
            if all(c[0] in qmap for c in pa):
                kls.append(kl_top(pa, qmap)[0])
            if sa[0][0] != sb[0][0]:
                ma = {c[0]: c[1] for c in pa}
                strict = ma.get(sb[0][0], -99.0) < sa[0][1]  # B's pick is below A's maximum (not a tie in A)
                flips.append({"prompt": i, "set": sets[i], "pos": j, "gap_ref": sa[0][1] - sa[1][1],
                              "gap_pick": sa[0][1] - ma.get(sb[0][0], -99.0), "strict": strict})
    p2[k] = {"positions": n, "flips": len(flips), "strict_flips": sum(f["strict"] for f in flips),
             "flip_gaps": [round(f["gap_ref"], 4) for f in flips],
             "strict_gap_max": max([f["gap_pick"] for f in flips if f["strict"]], default=0.0),
             "kl_mean": st.mean(kls) if kls else None, "kl_max": max(kls) if kls else None,
             "max_abs_lp_diff": None}
    self_flips_all += flips
# decode path (gen) against prefill path (forced) inside production
dflips, dn = [], 0
for i, (g, a) in enumerate(zip(gen, ref)):
    for j, (pg, pa) in enumerate(zip(g["top"], a["top"])):
        dn += 1
        if top1(pg)[0][0] != top1(pa)[0][0]:
            ma = {c[0]: c[1] for c in pa}
            c = top1(pg)[0][0]
            dflips.append({"prompt": i, "set": sets[i], "pos": j, "gap_ref": top1(pa)[0][1] - top1(pa)[1][1],
                           "gap_pick": top1(pa)[0][1] - ma.get(c, -99.0), "strict": ma.get(c, -99.0) < top1(pa)[0][1]})
p2["decode-vs-prefill"] = {"positions": dn, "flips": len(dflips), "strict_flips": sum(f["strict"] for f in dflips),
                           "flip_gaps": [round(f["gap_ref"], 4) for f in dflips],
                           "strict_gap_max": max([f["gap_pick"] for f in dflips if f["strict"]], default=0.0)}
self_flips_all += dflips
noise_band = max([f["gap_pick"] for f in self_flips_all if f["strict"]], default=0.0)
out["P2"] = p2
out["noise_band"] = noise_band

# ---- P1 / P3 per build ----
per_build = {}
for b, name in BUILDS:
    if b not in eng: continue
    flips, kls, n = [], [], 0
    by_set = Counter(); n_set = Counter()
    for i, (a, e) in enumerate(zip(ref, eng[b])):
        for j, (pa, pe) in enumerate(zip(a["top"], e)):
            n += 1; n_set[sets[i]] += 1
            sa = top1(pa)
            ptop = sa[0][0]
            etop = pe["top"][0][0]
            lp = {int(k): v for k, v in pe["lp"].items()}
            for t, v in pe["top"]:
                lp.setdefault(t, v)
            kl, mass = kl_top(pa, lp)
            kls.append({"kl": kl, "prompt": i, "set": sets[i], "pos": j, "mass": mass, "plen": len(ids[i])})
            if etop != ptop:
                ma = {c[0]: c[1] for c in pa}
                strict = ma.get(etop, -99.0) < sa[0][1]
                by_set[sets[i]] += strict
                flips.append({"prompt": i, "set": sets[i], "pos": j, "plen": len(ids[i]), "strict": strict,
                              "prod_gap": sa[0][1] - sa[1][1], "prod_gap_pick": sa[0][1] - ma.get(etop, -99.0),
                              "eng_gap": pe["top"][0][1] - lp.get(ptop, -99.0),
                              "prod_top": ptop, "eng_top": etop,
                              "eng_top_is_prod_2nd": etop == sa[1][0]})
    kv = [k["kl"] for k in kls]
    per_build[b] = {"name": name, "positions": n, "flips": flips, "strict_flips": sum(f["strict"] for f in flips),
                    "bins": {s_: [sum(1 for f in flips if f["strict"] and f["set"] == s_ and lo <= f["pos"] < lo + 16)
                                   for lo in (0, 16, 32)] for s_ in n_set}, "by_set": dict(by_set), "n_set": dict(n_set),
                    "kl_mean": st.mean(kv), "kl_p99": pct(kv, 0.99), "kl_max": max(kv),
                    "kl_by_set": {s: st.mean([k["kl"] for k in kls if k["set"] == s]) for s in n_set},
                    "worst": sorted(kls, key=lambda k: -k["kl"])[:10]}
out["builds"] = per_build


# ---- P5: statistics ----
lf = [0.0]
for i in range(1, 20001): lf.append(lf[-1] + math.log(i))
def lcomb(n, k): return lf[n] - lf[k] - lf[n - k]
def fisher(a, n1, b, n2):
    """Two-sided Fisher exact test for a/n1 vs b/n2 (sum of tables at most as likely as observed)."""
    K, N = a + b, n1 + n2
    def lp(x): return lcomb(n1, x) + lcomb(n2, K - x) - lcomb(N, K)
    obs = lp(a)
    tot = 0.0
    for x in range(max(0, K - n2), min(K, n1) + 1):
        v = lp(x)
        if v <= obs + 1e-9: tot += math.exp(v)
    return min(1.0, tot)
def binom_pmf(n, p):
    return [math.exp(lcomb(n, k) + (k * math.log(p) if k else 0) + ((n - k) * math.log(1 - p) if n - k else 0)) for k in range(n + 1)]
def power(n, p0, p1, alpha=0.05):
    a, b = binom_pmf(n, p0), binom_pmf(n, p1)
    xa = [k for k in range(n + 1) if a[k] > 1e-7]; xb = [k for k in range(n + 1) if b[k] > 1e-7]
    return sum(a[x] * b[y] for x in xa for y in xb if fisher(x, n, y, n) < alpha)
def min_detectable(n, p0):
    lo, hi = p0, min(0.5, p0 + 0.3)
    for _ in range(25):
        mid = (lo + hi) / 2
        if power(n, p0, mid) >= 0.8: hi = mid
        else: lo = mid
    return hi
out["P5"] = {}
out["P5"]["six_prompts"] = {"m1_mismatches": 4, "m2_mismatches": 11, "n": 288, "p": fisher(4, 288, 11, 288),
                            "min_detectable_rate": min_detectable(288, 4 / 288)}
json.dump(out, open(D + "results.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "builds"}, indent=1, default=str))
for b, v in per_build.items():
    print(b, v["name"], "flips", len(v["flips"]), "strict", v["strict_flips"], "bins", v["bins"], v["by_set"], "of", v["n_set"], "KL mean %.4g p99 %.4g max %.4g" % (v["kl_mean"], v["kl_p99"], v["kl_max"]))


# ---- P5 continued: the 48-prompt sets under validate_v0's protocol (each build scored on its own continuation) ----
def v2_counts(path):
    n = mm = 0
    for line in open(path):
        m = re.match(r"\s*(\d+) tokens\s+top1\s+([0-9.]+)%", line)
        if m:
            t = int(m.group(1)); n += t; mm += round(t * (1 - float(m.group(2)) / 100))
    return n, mm
if VDIR:
    v2 = {}
    for b, f in [("m1", "bin-m1"), ("b44", "bin-seq"), ("m2", "bin-chunked")]:
        nl, ml = v2_counts(os.path.join(VDIR, f"v2-long-{f}.txt"))
        ns, ms = v2_counts(os.path.join(VDIR, f"v2-short-{f}.txt"))
        v2[b] = {"n": nl + ns, "mismatches": ml + ms}
    out["P5"]["v2_48"] = {b: dict(v, p_vs_m1=fisher(v2["m1"]["mismatches"], v2["m1"]["n"], v["mismatches"], v["n"]))
                          for b, v in v2.items()}
    n48 = v2["m1"]["n"]
    out["P5"]["v2_48"]["min_detectable_rate"] = min_detectable(n48, v2["m1"]["mismatches"] / n48)
# the teacher-forced protocol: same positions for every build
if "m1" in per_build:
    a = per_build["m1"]["strict_flips"]; N = per_build["m1"]["positions"]
    out["P5"]["forced"] = {b: {"flips": v["strict_flips"], "n": v["positions"], "p_vs_m1": fisher(a, N, v["strict_flips"], v["positions"])}
                           for b, v in per_build.items()}
    out["P5"]["forced"]["min_detectable_rate"] = min_detectable(N, max(a, 1) / N)

# ---- P4: the first response's tool call on the 24 agent prompts ----
def parse_calls(text):
    calls = []
    for blk in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        m = re.search(r"<function=([^>\n]+)>", blk)
        if not m: continue
        params = {k.strip(): v.strip("\n") for k, v in re.findall(r"<parameter=([^>\n]+)>(.*?)</parameter>", blk, re.S)}
        calls.append((m.group(1).strip(), params))
    return calls
p4 = {}
try:
    prod = [json.loads(l)["text"] for l in open(D + "prod-gen400-long.jsonl")]
    for b, name in BUILDS:
        try:
            g = [json.loads(l)["text"] for l in open(D + f"gen400-{b}.jsonl")]
        except FileNotFoundError:
            continue
        rows = []
        for i, (pt, et) in enumerate(zip(prod, g)):
            pc, ec = parse_calls(pt), parse_calls(et)
            if not pc and not ec:
                kind = "text-same" if pt.strip() == et.strip() else "text-differs"
            elif [c[0] for c in pc] != [c[0] for c in ec]:
                kind = "tool-differs"
            elif pc == ec:
                kind = "tool-exact"
            else:
                diffs = []
                for (n1, a1), (n2, a2) in zip(pc, ec):
                    for k in sorted(set(a1) | set(a2)):
                        if a1.get(k) != a2.get(k): diffs.append(k)
                kind = "args-differ:" + ",".join(diffs)
            rows.append({"prompt": i, "kind": kind, "prod": pc, "eng": ec})
        # exact: the same calls with the same arguments, or the same text; same_tool: the same tool names in order
        # (two text answers count as the same).
        p4[b] = {"name": name, "exact": sum(r["kind"] in ("tool-exact", "text-same") for r in rows),
                 "same_tool": sum(r["kind"] in ("tool-exact", "text-same", "text-differs") or r["kind"].startswith("args-differ")
                                  for r in rows),
                 "rows": rows}
except FileNotFoundError:
    pass
# Production against itself (a second concurrency-1 run), and each build against that second run.
def match(A, B):
    exact = same = 0
    for a, b in zip(A, B):
        ca, cb = parse_calls(a), parse_calls(b)
        exact += (not ca and not cb and a.strip() == b.strip()) or bool(ca and ca == cb)
        same += [c[0] for c in ca] == [c[0] for c in cb]
    return {"exact": exact, "same_tool": same, "n": len(A)}
try:
    prod2 = [json.loads(l)["text"] for l in open(D + "prod-gen400-long-run2.jsonl")]
    p4["production_self"] = match(prod, prod2)
    for b, _ in BUILDS:
        if b in p4:
            p4[b]["vs_run2"] = match(prod2, [json.loads(l)["text"] for l in open(D + f"gen400-{b}.jsonl")])
except (FileNotFoundError, NameError):
    pass
out["P4"] = p4
json.dump(out, open(D + "results.json", "w"), indent=1, default=str)
print(json.dumps(out["P5"], indent=1))
for b, v in p4.items():
    if b == "production_self":
        print("P4 production against itself", v)
        continue
    print("P4", v["name"], "exact", v["exact"], "same tool", v["same_tool"], "vs run 2", v.get("vs_run2"),
          Counter(r["kind"].split(":")[0] for r in v["rows"]))
