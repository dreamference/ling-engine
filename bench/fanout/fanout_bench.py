#!/usr/bin/env python3
"""Shared-context fan-out, scoring and short-chat bench against a running SGLang server.

Read-only against the server: only /generate, /tokenize and /metrics are used.
Every cell writes one JSON line to the output file: per-request meta_info digests,
client wall times and a /metrics trace sampled every 0.5 s.

Usage:
  fanout_bench.py fanout  --ctx 32000 --n 1,4,8 --out f.jsonl --src ctx.txt
  fanout_bench.py score   --ctx 32000 --n 1,4,8 --out s.jsonl --src ctx.txt
  fanout_bench.py chat    --n 1,4,16 --out c.jsonl --src ctx.txt
  fanout_bench.py unshared --ctx 32000 --n 1,4 --out u.jsonl --src ctx.txt
"""

import argparse
import json
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:8000"
HEADER = "<|im_start|>user\nRead the following source code carefully; questions about it follow.\n\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
METRICS = (
    "num_running_reqs",
    "num_queue_reqs",
    "token_usage",
    "num_used_tokens",
    "mamba_usage",
    "mamba_used_tokens",
    "gen_throughput",
    "cache_hit_rate",
)


def post(path, body, timeout=7200):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def tokenize(text):
    return post("/tokenize", {"prompt": text, "add_special_tokens": False})["tokens"]


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        out = {}
        for line in r.read().decode().splitlines():
            m = re.match(r"^sglang:(\w+)\{[^}]*\} ([0-9.eE+-]+)$", line)
            if m and m.group(1) in METRICS:
                out[m.group(1)] = float(m.group(2))
        return out


def counters():
    """Monotonic counters (summed over label sets)."""
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        out = {}
        for line in r.read().decode().splitlines():
            m = re.match(r"^sglang:(prompt_tokens_total|generation_tokens_total|cached_tokens_total|num_requests_total)\{[^}]*\} ([0-9.eE+-]+)$", line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
        return out


def meminfo():
    with open("/proc/meminfo") as f:
        d = {l.split(":")[0]: int(l.split()[1]) for l in f}
    return round(d["MemAvailable"] / 1048576, 2)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.trace = []

    def run(self):
        t0 = time.time()
        while not self.stop.is_set():
            try:
                m = metrics()
                m["t"] = round(time.time() - t0, 2)
                m["mem_avail_gb"] = meminfo()
                self.trace.append(m)
            except Exception as e:  # noqa: BLE001
                self.trace.append({"err": str(e)[:100]})
            self.stop.wait(0.5)


def digest(resp, t_sub, t_done):
    mi = resp.get("meta_info", {})
    d = {
        "client_s": round(t_done - t_sub, 3),
        "prompt": mi.get("prompt_tokens"),
        "cached": mi.get("cached_tokens"),
        "out": mi.get("completion_tokens"),
        "queue": mi.get("queue_time"),
        "e2e": mi.get("e2e_latency"),
        "verify_ct": mi.get("spec_verify_ct"),
        "accept_len": mi.get("spec_accept_length"),
        "fwd_entry": mi.get("forward_entry_time"),
        "prefill_done": mi.get("prefill_finished_time"),
        "finished": mi.get("request_finished_ts"),
        "received": mi.get("request_received_ts"),
    }
    if "output_token_ids_logprobs" in mi:
        d["id_logprobs"] = mi["output_token_ids_logprobs"]
    return d


def fire(bodies):
    """Submit every body at once; returns (wall seconds, digests)."""
    results = [None] * len(bodies)
    t0 = time.time()

    def one(i):
        ts = time.time()
        try:
            r = post("/generate", bodies[i])
            results[i] = digest(r, ts, time.time())
        except Exception as e:  # noqa: BLE001
            results[i] = {"err": str(e)[:200]}

    with ThreadPoolExecutor(max_workers=min(len(bodies), 1024)) as ex:
        list(ex.map(one, range(len(bodies))))
    return time.time() - t0, results


def build_prefix(src_ids, ctx, tag):
    ids = tokenize(HEADER + f"[session {tag}]\n") + src_ids
    return ids[:ctx]


def questions(names, n, kind):
    qs = []
    for i in range(n):
        a = names[i % len(names)]
        b = names[(i * 7 + 3) % len(names)]
        if kind == "score":
            q = (
                f"\n\nQuestion {i}: in the source code above, does the function `{a}` call, directly or through "
                f"one helper, the function `{b}`? Look at the body of `{a}` and at every function it calls. "
                f"Consider only the code shown, not code that might exist elsewhere, and ignore comments and "
                f"docstrings that merely mention the name. Answer with one word, yes or no."
            )
        else:
            q = (
                f"\n\nQuestion {i}: in the source code above, explain what the function `{a}` does, which "
                f"arguments it takes and what it returns, which error cases it handles, and how it relates to "
                f"`{b}`. Name one caller of `{a}` if there is one. Be concrete and refer to the code, in plain "
                f"prose, without repeating the code itself."
            )
        qs.append(tokenize(q + TAIL))
    return qs


def cell(out, rec, bodies, warm=None):
    before = counters()
    s = Sampler()
    s.start()
    warm_rec = None
    if warm is not None:
        t = time.time()
        r = post("/generate", warm)
        warm_rec = digest(r, t, time.time())
        warm_rec["wall"] = round(time.time() - t, 3)
    wall, res = fire(bodies)
    time.sleep(0.6)
    s.stop.set()
    s.join()
    after = counters()
    rec.update(
        {
            "wall": round(wall, 3),
            "warm": warm_rec,
            "reqs": res,
            "trace": s.trace,
            "counters_delta": {k: after.get(k, 0) - before.get(k, 0) for k in after},
            "ts": time.strftime("%H:%M:%S", time.gmtime()),
        }
    )
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    ok = [r for r in res if "err" not in r]
    print(
        f"{rec['kind']} ctx={rec.get('ctx')} n={rec['n']} wall={wall:.1f}s ok={len(ok)}/{len(res)} "
        f"warm={warm_rec and warm_rec['wall']} cached0={ok and ok[0]['cached']} prompt0={ok and ok[0]['prompt']}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["fanout", "score", "chat", "unshared", "tail"])
    ap.add_argument("--ctx", type=int, default=32000)
    ap.add_argument("--n", default="1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--deadline", default="19:40", help="UTC HH:MM after which no new cell starts")
    a = ap.parse_args()
    text = open(a.src).read()
    names = sorted(set(re.findall(r"def (\w{6,})\(", text)))
    random.seed(1030)
    run = f"{int(time.time())}"
    src_ids = tokenize(text) if a.kind != "chat" else None
    yes_no = [tokenize(" yes")[0], tokenize(" no")[0], tokenize("yes")[0], tokenize("no")[0]]
    hh, mm = map(int, a.deadline.split(":"))
    for n in [int(x) for x in a.n.split(",")]:
        g = time.gmtime()
        if g.tm_hour * 60 + g.tm_min >= hh * 60 + mm:
            print(f"deadline {a.deadline} reached, skipping n={n}", flush=True)
            break
        tag = f"{run}-{a.kind}-{a.ctx}-{n}"
        rec = {"kind": a.kind, "ctx": a.ctx, "n": n}
        if a.kind in ("fanout", "score"):
            prefix = build_prefix(src_ids, a.ctx, tag)
            qs = questions(names, n, a.kind)
            sp = {"max_new_tokens": a.max_new, "temperature": 0, "ignore_eos": True}
            extra = {}
            if a.kind == "score":
                sp = {"max_new_tokens": 1, "temperature": 0}
                extra = {"return_logprob": True, "token_ids_logprob": yes_no, "logprob_start_len": -1}
            bodies = [{"input_ids": prefix + q, "sampling_params": sp, **extra} for q in qs]
            rec["q_tokens"] = [len(q) for q in qs[:4]] + [sum(len(q) for q in qs)]
            warm = {"input_ids": prefix, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
            cell(a.out, rec, bodies, warm)
        elif a.kind == "unshared":
            qs = questions(names, n, "fanout")
            bodies = []
            for i, q in enumerate(qs):
                prefix = build_prefix(src_ids, a.ctx, f"{tag}-{i}")
                bodies.append(
                    {"input_ids": prefix + q, "sampling_params": {"max_new_tokens": a.max_new, "temperature": 0, "ignore_eos": True}}
                )
            cell(a.out, rec, bodies)
        elif a.kind == "tail":
            # prefill rate at the end of a long cached context: warm the prefix, then append n new tokens
            prefix = build_prefix(src_ids, a.ctx, tag)
            extra_ids = src_ids[len(src_ids) // 2 : len(src_ids) // 2 + n]
            bodies = [{"input_ids": prefix + extra_ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}]
            warm = {"input_ids": prefix, "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
            cell(a.out, rec, bodies, warm)
        else:  # chat
            lines = [l for l in text.splitlines() if len(l.strip()) > 20]
            bodies = []
            for i in range(n):
                start = random.randrange(0, len(lines) - 40)
                snippet = "\n".join(lines[start : start + 12])
                head = tokenize(f"<|im_start|>user\n[chat {tag}-{i}] Here is a short piece of Python code:\n\n{snippet}")
                ask = tokenize(f"\n\nIn two or three sentences, what does it do?{TAIL}")
                ids = head[: 200 - len(ask)] + ask
                bodies.append(
                    {"input_ids": ids, "sampling_params": {"max_new_tokens": 200, "temperature": 0, "ignore_eos": True}}
                )
            rec["prompt_tokens_mean"] = sum(len(b["input_ids"]) for b in bodies) / n
            cell(a.out, rec, bodies)


if __name__ == "__main__":
    main()
