"""Acceptance on one text, step for step: SGLang's greedy output replayed through ling-serve with the output forced.

The replay's requests (bench/replay/replay.py --dump-bodies) are rendered to token ids by ling-template --responses
(the prompt both servers prefill, BACKLOG 19 / M4 §6 "the 991 tokens"), then:

  sglang   each prompt to SGLang's /generate at temperature 0; records its output ids and its own counters
           (spec_verify_ct, completion_tokens, spec_accept_histogram) per request;
  engine   each prompt to ling-serve's /v1/completions at temperature 0, with `force_tokens` = SGLang's output
           ids (--force) or free (greedy as it decodes); the passes come from ling-serve's request lines.
  compare  per request and pooled: decoded tokens (the output less its first, prefill-made token) over steps,
           SGLang's steps on its own text against ling-serve's on the same text.

On a forced request the engine's drafter sees exactly the context SGLang's did (the same prompt and output
tokens) and each step accepts the drafts equal to the text, which is SGLang's greedy accept rule on its own
output, so a difference is the drafter's (its inputs, its numerics, its selection), not the text's.

Usage:
  forced.py render  --bodies B.jsonl --ling-template BIN --tokenizer tokenizer.json --out prompts.jsonl
  forced.py sglang  --prompts prompts.jsonl --url http://127.0.0.1:30000 --out sg.jsonl
  forced.py engine  --prompts prompts.jsonl --url http://127.0.0.1:8300 --out en.jsonl [--force sg.jsonl]
  forced.py compare --sglang sg.jsonl --engine-log serve.log --skip N [--engine en.jsonl]
"""
import argparse
import http.client
import json
import random
import re
import subprocess
import sys
import urllib.parse


def post(url, path, body, timeout=3600):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data[:300]!r}")
    return json.loads(data)


def render(a):
    bodies = [json.loads(l) for l in open(a.bodies)]
    p = subprocess.run([a.ling_template, "--responses", a.tokenizer], input="".join(json.dumps(b) + "\n" for b in bodies),
                       capture_output=True, text=True, check=True)
    with open(a.out, "w") as f:
        for b, line in zip(bodies, p.stdout.splitlines()):
            r = json.loads(line)
            if "error" in r:
                raise RuntimeError(r["error"])
            f.write(json.dumps({"session": b["session"], "req": b["req"], "cold": b["cold"],
                                "max_out": b["body"]["max_output_tokens"], "ids": r["ids"]}) + "\n")


def sglang(a):
    out = open(a.out, "w")
    for line in open(a.prompts):
        p = json.loads(line)
        r = post(a.url, "/generate", {"input_ids": p["ids"], "sampling_params": {
            "temperature": 0, "max_new_tokens": p["max_out"], "skip_special_tokens": False}})
        m = r.get("meta_info") or {}
        rec = {k: p[k] for k in ("session", "req", "cold")}
        rec.update({"out_ids": r.get("output_ids"), "completion_tokens": m.get("completion_tokens"),
                    "verify": m.get("spec_verify_ct"), "cached": m.get("cached_tokens"),
                    "hist": m.get("spec_accept_histogram"), "finish": m.get("finish_reason"), "text": r.get("text")})
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{rec['session'][-12:]} #{rec['req']} out {len(rec['out_ids'] or [])} verify {rec['verify']}", flush=True)


def engine(a):
    forced = [json.loads(l) for l in open(a.force)] if a.force else None
    out = open(a.out, "w")
    for i, line in enumerate(open(a.prompts)):
        p = json.loads(line)
        body = {"prompt": p["ids"], "max_tokens": p["max_out"], "temperature": 0, "stream": False}
        if forced:
            f = forced[i]
            assert (f["session"], f["req"]) == (p["session"], p["req"])
            body["force_tokens"] = f["out_ids"]
            body["max_tokens"] = len(f["out_ids"])
        r = post(a.url, "/v1/completions", body)
        rec = {k: p[k] for k in ("session", "req", "cold")}
        rec.update({"usage": r.get("usage"), "text": (r.get("choices") or [{}])[0].get("text"),
                    "finish": (r.get("choices") or [{}])[0].get("finish_reason")})
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{rec['session'][-12:]} #{rec['req']} {rec['usage']}", flush=True)


LINE = re.compile(r"request: session (\d+)( \(new\))?, (\d+) prompt tokens \((\d+) reused\).*?; (\d+) tokens in "
                  r"([\d.]+) s .*?, (\w+); (\d+) passes")


def compare(a):
    sg = [json.loads(l) for l in open(a.sglang)]
    en = [m for m in (LINE.search(l) for l in open(a.engine_log) if l.startswith("request:")) if m][a.skip:]
    assert len(en) == len(sg), (len(en), len(sg))
    pairs = []
    for s, e in zip(sg, en):
        if s["cold"] or not s["verify"]:
            continue
        n = len(s["out_ids"]) - 1  # decoded after the prefill's token (the stop token included, as SGLang counts)
        # ling-serve: `produced` excludes a stop token, the passes include the step that chose it.
        ended_eos = e.group(7) == "stop"
        en_n = int(e.group(5)) - 1 + (1 if ended_eos else 0)
        pairs.append((n, s["verify"], en_n, int(e.group(8))))
    def pooled(ps):
        return sum(p[0] for p in ps) / sum(p[1] for p in ps), sum(p[2] for p in ps) / sum(p[3] for p in ps)
    sg_r, en_r = pooled(pairs)
    rng = random.Random(1)
    bs = sorted((lambda x: x[1] / x[0])(pooled([pairs[rng.randrange(len(pairs))] for _ in pairs])) for _ in range(4000))
    same_n = sum(1 for p in pairs if p[0] == p[2])
    print(f"{len(pairs)} warm requests ({same_n} with the same decoded count): SGLang {sg_r:.3f} tokens per step, "
          f"ling-serve {en_r:.3f}; ling-serve/SGLang {en_r / sg_r:.3f}, 95% bootstrap {bs[100]:.3f}-{bs[3899]:.3f}")
    if a.per_request:
        for s, p in zip([s for s in sg if not s["cold"] and s["verify"]], pairs):
            print(s["session"][-12:], s["req"], *p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("render", "sglang", "engine", "compare"))
    ap.add_argument("--bodies")
    ap.add_argument("--ling-template")
    ap.add_argument("--tokenizer")
    ap.add_argument("--prompts")
    ap.add_argument("--url")
    ap.add_argument("--out")
    ap.add_argument("--force")
    ap.add_argument("--sglang")
    ap.add_argument("--engine-log")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--per-request", action="store_true")
    a = ap.parse_args()
    {"render": render, "sglang": sglang, "engine": engine, "compare": compare}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
