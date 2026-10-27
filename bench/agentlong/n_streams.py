"""N agentlong streams at once against one server: what batching several requests gives (BATCHING §2).

The sixteen agentlong turns (select_set.py; 28.5K to 57.3K prompt tokens, recorded agent requests) are dealt
round-robin into N lists (stream j takes turns j, j+N, j+2N, ...), optionally looped R times (--repeat), and
each list is sent one request at a time by its own thread, so the server holds N sequences at once: each turn
prefills beside the others' decodes, then decodes beside them. The total work is the same sixteen turns (times
R) whatever N is, so the wall time compares directly. The same bodies go to ling-serve and SGLang byte for
byte (checked against the manifest's SHA-256).

Measured on the client, per request: the time to the first streamed delta (TTFT, which holds any queueing and
the prefill), the end, the output tokens and prompt tokens from the response's usage, the cached prompt tokens,
and the output's SHA-256 (two greedy runs compare turn for turn with --compare, at any N). Per stream: decode
tokens/s = sum (output - 1) / sum (end - first delta). In all: output tokens over the wall (prefill included),
and the aggregate decode rate while exactly N, and while at least one, stream was decoding (each request's
tokens spread evenly over its decode interval).

On the server, sampled every 0.5 s from /metrics (both servers export SGLang's names): running and queued
requests, generated tokens, verify passes (tokens per step = generated / passes over the run), and the host's
MemAvailable (/proc/meminfo; this machine's GPU memory is the host's). From ling-serve also /spec_stats before
and after (draft, verify, commit seconds, batched steps).

--warm makes every turn a warm follow-up, as an agent's turns are: in rounds, each stream first sends its turn's
prompt without the last tool call and its output (the agent's previous request, max_output_tokens 1, not counted),
all streams wait for each other, then all send their full turns at once (the tool's output appended to a resident
prompt, a few hundred to a few thousand new tokens), and the round ends when all have answered. So N sequences
decode together from a warm start, which is the batching case without the cold 20-35K-token prefills.

Usage:
  n_streams.py --url http://127.0.0.1:8300 --streams N --out RUN.jsonl [--label L] [--mode greedy|seedN]
               [--repeat R] [--bodies ~/bench-sets/agentlong/bodies.jsonl]
  n_streams.py --compare A.jsonl B.jsonl     per-turn output identity (greedy) between two runs
  n_streams.py --summary RUN.jsonl ...       the summary lines
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.parse
import http.client
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as al  # noqa: E402

GAUGES = ("sglang:num_running_reqs", "sglang:num_queue_reqs")
COUNTERS = ("sglang:generation_tokens_total", "sglang:spec_verify_calls_total", "sglang:prompt_tokens_total",
            "sglang:cached_tokens_total")
SPEC_KEYS = ("steps", "batched_steps", "accepted", "draft_seconds", "verify_seconds", "commit_seconds")


def scrape(url):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=5)
    c.request("GET", "/metrics")
    txt = c.getresponse().read().decode()
    out = {}
    for n in GAUGES + COUNTERS:
        vals = re.findall(r"^" + re.escape(n) + r"(?:\{[^}]*\})? ([0-9.e+-]+)$", txt, re.M)
        out[n] = sum(float(v) for v in vals) if vals else None
    return out


def mem_available_mb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) // 1024
    except OSError:
        pass
    return None


def overlap_rates(reqs, n):
    """Aggregate decode tokens/s while exactly n and while at least one request was decoding, each request's
    tokens spread evenly over its decode interval [t_first, t_end]."""
    iv = [(r["t_first"], r["t_end"], r["out"] - 1) for r in reqs if r["out"] > 1 and r["t_end"] > r["t_first"]]
    edges = sorted({t for a, b, _ in iv for t in (a, b)})
    t_n = tok_n = t_any = tok_any = 0.0
    for lo, hi in zip(edges, edges[1:]):
        live = [(a, b, g) for a, b, g in iv if a <= lo and b >= hi]
        if not live:
            continue
        dt = hi - lo
        tok = sum(g * dt / (b - a) for a, b, g in live)
        t_any += dt
        tok_any += tok
        if len(live) == n:
            t_n += dt
            tok_n += tok
    return (round(tok_n / t_n, 2) if t_n else None, round(t_n, 1), round(tok_any / t_any, 2) if t_any else None)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))], 2)


def summarize(reqs, n, wall, timeline, spec0, spec1, label, mode):
    ok = [r for r in reqs if not r.get("error")]
    s = {"summary": True, "label": label, "mode": mode, "streams": n, "requests": len(reqs),
         "errors": len(reqs) - len(ok), "wall_s": round(wall, 1)}
    out = sum(r["out"] for r in ok)
    s["output_tokens"] = out
    s["aggregate_e2e_tps"] = round(out / wall, 2) if wall else None
    per = []
    for j in range(n):
        rs = [r for r in ok if r["stream"] == j and r["out"] > 1]
        g = sum(r["out"] - 1 for r in rs)
        d = sum(r["t_end"] - r["t_first"] for r in rs)
        per.append(round(g / d, 2) if d else None)
    s["stream_decode_tps"] = per
    vals = [p for p in per if p]
    s["stream_decode_tps_mean"] = round(statistics.mean(vals), 2) if vals else None
    g = sum(r["out"] - 1 for r in ok if r["out"] > 1)
    d = sum(r["t_end"] - r["t_first"] for r in ok if r["out"] > 1)
    s["request_decode_tps_pooled"] = round(g / d, 2) if d else None
    s["aggregate_decode_tps_all_n"], s["all_n_decoding_s"], s["aggregate_decode_tps_any"] = overlap_rates(ok, n)
    ttft = [r["ttft"] for r in ok if r.get("ttft") is not None]
    s["ttft_median_s"], s["ttft_p90_s"], s["ttft_max_s"] = pct(ttft, 0.5), pct(ttft, 0.9), pct(ttft, 1.0)
    new = sum(max(0, r["prompt"] - r["cached"]) for r in ok)
    s["prompt_tokens"] = sum(r["prompt"] for r in ok)
    s["prefill_new_tokens"] = new
    s["prompt_tokens_range"] = [min((r["prompt"] for r in ok), default=0), max((r["prompt"] for r in ok), default=0)]
    s["ttft_s_per_1k_new"] = round(1000 * sum(ttft) / new, 3) if new else None
    if len(timeline) >= 2:
        a, b = timeline[0], timeline[-1]
        dg = (b.get("sglang:generation_tokens_total") or 0) - (a.get("sglang:generation_tokens_total") or 0)
        dv = (b.get("sglang:spec_verify_calls_total") or 0) - (a.get("sglang:spec_verify_calls_total") or 0)
        s["server_tokens_per_pass"] = round(dg / dv, 3) if dv else None
        s["server_passes"] = int(dv)
        run = [t["sglang:num_running_reqs"] for t in timeline if t.get("sglang:num_running_reqs") is not None]
        q = [t["sglang:num_queue_reqs"] for t in timeline if t.get("sglang:num_queue_reqs") is not None]
        s["running_mean"] = round(statistics.mean(run), 2) if run else None
        s["queued_mean"] = round(statistics.mean(q), 2) if q else None
        s["queued_max"] = max(q) if q else None
        mem = [t["mem_available_mb"] for t in timeline if t.get("mem_available_mb")]
        s["mem_available_mb_min"] = min(mem) if mem else None
        s["mem_available_mb_start"] = mem[0] if mem else None
    if spec0 and spec1:
        dsp = {k: spec1.get(k, 0) - spec0.get(k, 0) for k in SPEC_KEYS}
        s["spec"] = dsp
        if dsp["steps"]:
            for k in ("draft_seconds", "verify_seconds", "commit_seconds"):
                s[f"{k[:-8]}_ms_per_seq_step"] = round(1000 * dsp[k] / dsp["steps"], 2)
            s["batched_share"] = round(dsp["batched_steps"] / dsp["steps"], 3)
            s["accepted_per_seq_step"] = round(dsp["accepted"] / dsp["steps"], 3)
    return s


def load(path):
    recs = [json.loads(ln) for ln in open(path)]
    return ([r for r in recs if r.get("kind") == "request"], next((r for r in recs if r.get("summary")), {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", default="greedy", help="greedy, or seedN (a per-turn seed from N, as run.py)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warm", action="store_true", help="each turn primed with its previous request first (rounds)")
    ap.add_argument("--cache-key", action="store_true",
                    help="send prompt_cache_key (one per turn, as Codex sends its thread id); ling-serve binds by it")
    ap.add_argument("--bodies", default=os.path.expanduser("~/bench-sets/agentlong/bodies.jsonl"))
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"))
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--summary", nargs="+")
    a = ap.parse_args()

    if a.summary:
        for p in a.summary:
            print(p, json.dumps(load(p)[1]))
        return 0
    if a.compare:
        (ra, sa), (rb, sb) = load(a.compare[0]), load(a.compare[1])
        bb = {(r["id"], r["rep"]): r for r in rb}
        same = n = 0
        for r in sorted(ra, key=lambda r: (r["id"], r["rep"])):
            o = bb.get((r["id"], r["rep"]))
            if o is None:
                continue
            n += 1
            eq = r["text_sha256"] == o["text_sha256"]
            same += eq
            print(f"{r['id']}#{r['rep']} out {r['out']}/{o['out']} {'identical' if eq else 'DIFFERENT'}")
        print(f"{same}/{n} turns identical ({sa.get('label')} at {sa.get('streams')} against {sb.get('label')} at "
              f"{sb.get('streams')})")
        return 0 if same == n else 1

    man = {t["id"]: t for t in json.load(open(a.manifest))["turns"]}
    bodies = [json.loads(ln) for ln in open(a.bodies)]
    for b in bodies:
        if man.get(b["id"], {}).get("body_sha256") != hashlib.sha256(al.canonical(b["body"]).encode()).hexdigest():
            sys.exit(f"{b['id']}: the body's hash does not match the manifest")
    n = a.streams
    greedy = a.mode == "greedy"
    seed0 = None if greedy else int(a.mode[4:].lstrip(":"))
    lists = [[(b, rep) for rep in range(a.repeat) for b in bodies[j::n]] for j in range(n)]
    w = {"model": bodies[0]["body"]["model"], "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "Write a short Python function that reverses a list, then explain it."}]}],
         "stream": True, "store": False, "max_output_tokens": 128, "reasoning": {"effort": "none"}, "temperature": 0.0}
    al.send(a.url, w)

    timeline, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                m = scrape(a.url)
            except (OSError, ValueError):
                m = {}
            m["t"] = round(time.time(), 2)
            m["mem_available_mb"] = mem_available_mb()
            timeline.append(m)
            stop.wait(0.5)

    reqs, lock = [], threading.Lock()

    def body_for(b):
        body = dict(b["body"])
        if a.cache_key:
            body["prompt_cache_key"] = f"{a.label}-{b['id']}"
        if greedy:
            body["temperature"] = 0.0
        else:
            body["seed"] = 1 + zlib.crc32(f"{seed0}:{b['id']}".encode()) % (2**31 - 1)
        return body

    def one(j, b, rep):
        body = body_for(b)
        t0 = time.time()
        r = al.send(a.url, body)
        u = r.get("usage") or {}
        cached = ((u.get("input_tokens_details") or {}).get("cached_tokens")) or 0
        rec = {"kind": "request", "id": b["id"], "rep": rep, "stream": j, "t0": round(t0, 3),
               "t_first": round(t0 + (r.get("ttft") or 0), 3), "t_end": round(t0 + r.get("t_total", 0), 3),
               "ttft": round(r["ttft"], 3) if r.get("ttft") else None, "out": u.get("output_tokens", 0),
               "prompt": u.get("input_tokens", 0), "cached": cached, "status": r.get("status"),
               "text_sha256": r.get("text_sha256"), "error": r.get("error")}
        with lock:
            reqs.append(rec)
        dec = rec["t_end"] - rec["t_first"]
        print(f"stream {j} {b['id']}#{rep} in {rec['prompt']} cached {cached} out {rec['out']} ttft "
              f"{rec['ttft']} s decode {(rec['out'] - 1) / dec if dec > 0 else 0:.1f} tok/s {rec['error'] or ''}",
              flush=True)

    def prime(b):
        body = body_for(b)
        body["input"] = body["input"][:-2]  # the previous request: without the last tool call and its output
        body["max_output_tokens"] = 1
        al.send(a.url, body)

    rounds = max(len(x) for x in lists)
    bar = threading.Barrier(n)

    def worker(j):
        for k in range(rounds):
            item = lists[j][k] if k < len(lists[j]) else None
            if a.warm:
                if item:
                    prime(item[0])
                bar.wait()
            if item:
                one(j, *item)
            if a.warm:
                bar.wait()

    spec0 = al.spec_stats(a.url)
    th = [threading.Thread(target=worker, args=(j,)) for j in range(n)]
    smp = threading.Thread(target=sampler)
    smp.start()
    t_start = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t_start
    time.sleep(1.0)
    stop.set()
    smp.join()
    spec1 = al.spec_stats(a.url)
    s = summarize(reqs, n, wall, timeline, spec0, spec1, a.label, a.mode)
    with open(a.out, "w") as f:
        f.write(json.dumps({"header": True, "label": a.label, "mode": a.mode, "streams": n, "repeat": a.repeat, "warm": a.warm,
                            "url": a.url, "t": t_start}) + "\n")
        for r in sorted(reqs, key=lambda r: (r["id"], r["rep"])):
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"timeline": timeline}) + "\n")
        f.write(json.dumps(s) + "\n")
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
