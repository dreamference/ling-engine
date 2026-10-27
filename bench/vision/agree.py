"""The agreement gate on image prompts (VISION §20.5 item 5; MILESTONES §14's lines, recomputed for this set).

  agree.py DIR     DIR holds prod-render.jsonl, prod-gen.jsonl, prod-force.jsonl (prod_vision.py), engine-force.jsonl
                   (ling-run --force-file cases.jsonl) and engine-greedy.jsonl (ling-run --greedy-out)
  agree.py cases DIR   writes DIR/cases.jsonl for ling-run --force-file (production's continuation and its top-20 ids)

A flip is a position where the engine's top token scores strictly below production's greedy token. Production's own
noise is its decode against its prefill on the same positions (prod-gen's choice against prod-force's top).
"""
import json
import math
import sys


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def fisher_upper(a, n1, b, n2):
    """P(X >= a) for the engine's flips a of n1 against production's b of n2 (hypergeometric upper tail)."""
    total, k = n1 + n2, a + b
    def lc(n, r):
        return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)
    den = lc(total, k)
    return sum(math.exp(lc(n1, x) + lc(n2, k - x) - den) for x in range(a, min(n1, k) + 1))


def cases(d):
    gen = rows(f"{d}/prod-gen.jsonl")
    force = rows(f"{d}/prod-force.jsonl")
    with open(f"{d}/cases.jsonl", "w") as f:
        for i, (g, fo) in enumerate(zip(gen, force)):
            f.write(json.dumps({"prompt": i, "cont": g["cont"], "want": [[c[0] for c in pos[:5]] for pos in fo["top"]]}) + "\n")


def main(d):
    render = rows(f"{d}/prod-render.jsonl")
    gen = rows(f"{d}/prod-gen.jsonl")
    pforce = rows(f"{d}/prod-force.jsonl")
    eforce = rows(f"{d}/engine-force.jsonl")
    egreedy = rows(f"{d}/engine-greedy.jsonl")
    n = flips = pflips = top5 = 0
    worst_gap = 0.0
    by_pos = {}
    rendering_ok = 0
    identical = 0
    for i, (r, g, pf, ef, eg) in enumerate(zip(render, gen, pforce, eforce, egreedy)):
        rendering_ok += eg["prompt_ids"] == r["prompt_ids"]
        identical += eg["ids"][: len(g["cont"])] == g["cont"]
        for j, tok in enumerate(g["cont"]):
            n += 1
            ptop = pf["top"][j]  # production's prefill top-20 [(id, logprob)]
            pf_best = ptop[0][0]
            pflips += pf_best != tok
            e = ef["pos"][j]
            e_best_id, e_best_lp = e["top"][0]
            e_lp_tok = e["lp"].get(str(tok), -1e9)
            flip = e_lp_tok < e_best_lp
            if flip:
                flips += 1
                by_pos[j] = by_pos.get(j, 0) + 1
                plp = dict((t, lp) for t, lp in ptop)
                gap = plp.get(tok, ptop[0][1]) - plp.get(e_best_id, ptop[-1][1])
                worst_gap = max(worst_gap, gap)
            top5 += tok in [t for t, _ in e["top"][:5]]
    # Production's own decode against its prefill on the same positions (VISION §20.5 item 5: on image prompts the
    # gap and top-5 lines are judged against these).
    p_worst_gap = 0.0
    p_miss5 = 0
    for g, pf in zip(gen, pforce):
        for j, tok in enumerate(g["cont"]):
            ftop = pf["top"][j]
            if ftop[0][0] != tok:
                flp = dict((t, lp) for t, lp in ftop)
                p_worst_gap = max(p_worst_gap, ftop[0][1] - flp.get(tok, ftop[-1][1]))
            p_miss5 += tok not in [t for t, _ in ftop[:5]]
    p = fisher_upper(flips, n, pflips, n)
    print(f"prompts {len(gen)}: rendering token-identical to production's on {rendering_ok}, free-running greedy identical on {identical}")
    print(f"positions {n}: engine flips {flips} ({100 * flips / max(n, 1):.2f}%), production's own decode-vs-prefill {pflips}"
          f" ({100 * pflips / max(n, 1):.2f}%); one-sided Fisher p = {p:.3f} ({'pass' if p >= 0.05 else 'FAIL'} at 0.05)")
    print(f"worst flip gap {worst_gap:.3f} nats, production's own {p_worst_gap:.3f} "
          f"({'pass' if worst_gap <= max(1.75, p_worst_gap) else 'FAIL'}: image prompts against production's own noise,"
          f" text at 1.75); production's token outside the engine's top-5 at {n - top5} of {n} positions, production's"
          f" own decode token outside its prefill's top-5 at {p_miss5} ({'pass' if n - top5 <= p_miss5 else 'FAIL'})")
    print("flips by position in the continuation:", dict(sorted(by_pos.items())))


if __name__ == "__main__":
    if sys.argv[1] == "cases":
        cases(sys.argv[2])
    else:
        main(sys.argv[1])
