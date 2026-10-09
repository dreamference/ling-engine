"""Paired bootstrap over prompts for KL and flips (M2 - M1, 44c92c5 - M1), flips against prompt length, and the
contexts of the 10 worst KL positions (written for decoding)."""
import json, math, random, statistics as st, sys
D = sys.argv[1].rstrip("/") + "/"  # the study directory
r = json.load(open(D + "results.json"))
sets = json.load(open(D + "sets.json"))
ids = [json.loads(l)["ids"] for l in open(D + "ids.jsonl")]
gen = [json.loads(l) for l in open(D + "prod-gen48.jsonl")]
ref = [json.loads(l) for l in open(D + "prod-force-cold-c1.jsonl")]


def kl_top(prod_pos, q_lp):
    p = [(c[0], math.exp(c[1])) for c in prod_pos]
    q = [(i, math.exp(q_lp[i])) for i, _ in p]
    sp, sq = sum(v for _, v in p), sum(v for _, v in q)
    return sum((pv / sp) * math.log((pv / sp) / max(qv / sq, 1e-30)) for (_, pv), (_, qv) in zip(p, q) if pv > 0)


per = {}
for b in ["m1", "b44", "m2"]:
    e = [json.loads(l)["pos"] for l in open(D + f"eng-{b}.jsonl")]
    rows = []
    for i, (a, pe_list) in enumerate(zip(ref, e)):
        kls, fl = [], 0
        for j, (pa, pe) in enumerate(zip(a["top"], pe_list)):
            lp = {int(k): v for k, v in pe["lp"].items()}
            for t, v in pe["top"]: lp.setdefault(t, v)
            kls.append(kl_top(pa, lp))
            mx = max(c[1] for c in pa)
            ma = {c[0]: c[1] for c in pa}
            fl += ma.get(pe["top"][0][0], -99.0) < mx
        rows.append((kls, fl))
    per[b] = rows
# production self (warm-c1 and cold-c4 against cold-c1) per prompt KL, for scale
random.seed(1)
def boot(b):
    n = len(ids); diffs_mean, diffs_p99, diffs_fl = [], [], []
    for _ in range(2000):
        s = [random.randrange(n) for _ in range(n)]
        A = [k for i in s for k in per["m1"][i][0]]; B = [k for i in s for k in per[b][i][0]]
        diffs_mean.append(st.mean(B) - st.mean(A))
        A.sort(); B.sort()
        diffs_p99.append(B[int(0.99 * len(B))] - A[int(0.99 * len(A))])
        diffs_fl.append(sum(per[b][i][1] for i in s) - sum(per["m1"][i][1] for i in s))
    q = lambda xs: (sorted(xs)[50], sorted(xs)[1949])
    return {"kl_mean_diff_ci95": q(diffs_mean), "kl_p99_diff_ci95": q(diffs_p99), "flip_diff_ci95": q(diffs_fl)}
out = {b: boot(b) for b in ["b44", "m2"]}
# flips against prompt length (long set): flip rate per prompt vs length
out["long_flips_by_length"] = {b: sorted([(len(ids[i]), per[b][i][1]) for i in range(30, 54)]) for b in per}
json.dump(out, open(D + "extra.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:3000])
# worst positions for decoding
worst = []
for b in ["m1", "b44", "m2"]:
    for w in r["builds"][b]["worst"]:
        i, j = w["prompt"], w["pos"]
        e = json.loads(open(D + f"eng-{b}.jsonl").readlines()[i])["pos"][j]
        pt = sorted(ref[i]["top"][j], key=lambda c: -c[1])[:3]
        worst.append({"build": b, "prompt": i, "set": sets[i], "pos": j, "kl": w["kl"], "plen": len(ids[i]),
                      "context": ids[i][-12:] + gen[i]["cont"][:j], "prod_top3": pt, "eng_top3": e["top"][:3]})
json.dump(worst, open(D + "worst.json", "w"))
