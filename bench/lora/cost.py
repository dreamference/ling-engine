#!/usr/bin/env python3
"""The per-adapter cost on ling-serve (LORA §6): the same greedy request on streams of chosen models at once.

    cost.py URL MAXTOK SCENARIO [SCENARIO ...] [--reps 3]

A SCENARIO is a comma-separated list of models, one stream each (`B` stands for the served base name): `B,B` two
streams on the base, `B+zero,B+zero` both on one adapter, `B+zero,B+zero2` on two. Each scenario runs once to warm
the prefix cache (keyed by adapter, so every stream's prefill is warm after it), then --reps times measured; per run
the wall time, the tokens, the engine steps (/spec_stats) and the milliseconds per joint step, wall * streams / steps.
With zero adapters every stream gives the base's text (checked), so their steps are the base's and the difference in
ms per step is the adapters' cost alone.
"""
import argparse, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("Write a Python module implementing a red-black tree with insert, delete and search, with a docstring on "
          "every method and a short test at the end.")


def call(url, path, body=None):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def one(url, model, maxtok):
    r = call(url, "/v1/completions", {"model": model, "prompt": PROMPT, "max_tokens": maxtok, "temperature": 0})
    return r["choices"][0]["text"], r["usage"]["completion_tokens"]


def run(url, models, maxtok):
    s0 = call(url, "/spec_stats")["steps"]
    t0 = time.monotonic()
    with ThreadPoolExecutor(len(models)) as ex:
        res = list(ex.map(lambda m: one(url, m, maxtok), models))
    wall = time.monotonic() - t0
    steps = call(url, "/spec_stats")["steps"] - s0
    return wall, res, steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url")
    ap.add_argument("maxtok", type=int)
    ap.add_argument("scenarios", nargs="+")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    base = call(a.url, "/v1/models")["data"][0]["id"].split("+")[0]
    texts = {}
    for sc in a.scenarios:
        models = [m.replace("B", base, 1) if m.startswith("B") else m for m in sc.split(",")]
        run(a.url, models, a.maxtok)
        for rep in range(a.reps):
            wall, res, steps = run(a.url, models, a.maxtok)
            toks = sum(n for _, n in res)
            for m, (t, _) in zip(models, res):
                texts.setdefault(m, set()).add(t)
            ms = 1000 * wall * len(models) / steps if steps else 0
            print(json.dumps({"scenario": sc, "rep": rep, "wall_s": round(wall, 3), "tokens": toks, "steps": steps,
                              "tok_s": round(toks / wall, 2), "ms_per_step": round(ms, 3)}), flush=True)
    distinct = {m: len(t) for m, t in texts.items()}
    print(json.dumps({"texts_per_model": distinct, "distinct_texts_overall": len(set().union(*texts.values()))}))


if __name__ == "__main__":
    main()
