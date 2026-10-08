"""Replay recorded Mightling agent sessions against a model server (M0, spec section 13).

Each rollout is rebuilt into the exact requests the agent sent: the Responses API body puffin
sends (captured once with capture_proxy.py), with the rollout's own `response_item`s as `input`,
its `base_instructions` as `instructions`, and compaction applied where the session compacted.
SGLang has no forced-decoding mode, so the server generates naturally, capped at the recorded
output length. The prefix cache is warmed as in production by replaying a contiguous window of
each session; the window's first request is reported separately as a cold request.

Usage:
  replay.py --url http://127.0.0.1:8000 --tools tools.json --out run.jsonl \
            [--sessions-file sel.txt | rollouts...] [--window 12] [--concurrency 1]
            [--greedy] [--max-requests N]

Per request it records: input and cached tokens, output tokens, time to first token, total time
and decode tokens/s. Around the run it reads /metrics, so accepted tokens per verify step come
from the server's own counters. Prompt contents never leave this machine.
"""
import argparse
import http.client
import json
import random
import re
import statistics as st
import sys
import threading
import time
import urllib.parse


def parse_rollout(path):
    """Returns (instructions, requests): each request is {input, usage, out_items}."""
    rows = [json.loads(line) for line in open(path)]
    instructions = None
    history = []
    pending_out = []
    reqs = []
    for r in rows:
        ty = r.get("type")
        p = r.get("payload") or {}
        if ty == "session_meta":
            bi = p.get("base_instructions")
            instructions = bi.get("text") if isinstance(bi, dict) else bi
        elif ty == "compacted":
            history = list(p.get("replacement_history") or [])
            pending_out = []
        elif ty == "response_item":
            pt = p.get("type")
            if pt in ("message", "function_call", "custom_tool_call") and (
                    pt != "message" or p.get("role") == "assistant"):
                pending_out.append(p)
            elif pt in ("message", "function_call_output", "custom_tool_call_output"):
                history.append(p)
            else:
                history.append(p)
        elif ty == "token_usage_record":
            reqs.append({"input": list(history), "usage": p.get("usage") or {},
                         "out_items": list(pending_out)})
            history.extend(pending_out)
            pending_out = []
    return instructions, reqs


def clean_item(it):
    """Drops rollout-only fields the wire format does not carry (it keeps `id`, as puffin does)."""
    return {k: v for k, v in it.items() if not k.startswith("internal_")}


def metrics(url):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    c.request("GET", "/metrics")
    txt = c.getresponse().read().decode()
    out = {}
    for name in ("sglang:spec_verify_calls_total", "sglang:generation_tokens_total",
                 "sglang:prompt_tokens_total", "sglang:cached_tokens_total",
                 "sglang:e2e_request_latency_seconds_sum", "sglang:e2e_request_latency_seconds_count",
                 "sglang:time_to_first_token_seconds_sum"):
        out[name] = sum(float(m) for m in re.findall(
            r"^" + re.escape(name) + r"\{[^}]*\} ([0-9.e+]+)$", txt, re.M))
    return out


def send(url, body):
    """Streams one Responses API request; returns timing and usage."""
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    data = json.dumps(body).encode()
    t0 = time.perf_counter()
    c.request("POST", "/v1/responses", body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {r.read()[:300]!r}"}
    t_first = None
    rid = None
    n_deltas = 0
    usage = None
    buf = b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    j = json.loads(payload)
                except ValueError:
                    continue
                t = j.get("type", "")
                if t.endswith(".delta"):
                    n_deltas += 1
                    if t_first is None:
                        t_first = time.perf_counter()
                elif t == "response.created":
                    rid = (j.get("response") or {}).get("id")
                elif t == "response.completed":
                    usage = (j.get("response") or {}).get("usage")
    t1 = time.perf_counter()
    return {"t_total": t1 - t0, "ttft": (t_first - t0) if t_first else None, "deltas": n_deltas,
            "usage": usage, "rid": rid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="RadixArk/Qwen3.8-27B-NVFP4")
    ap.add_argument("--tools", required=True, help="JSON file: the captured tools array")
    ap.add_argument("--tools-index", help="tools array with the code-index tools, for sessions "
                    "whose path does not contain 'index-off'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions-file")
    ap.add_argument("rollouts", nargs="*")
    ap.add_argument("--window", type=int, default=12, help="contiguous requests per session")
    ap.add_argument("--start", default="random", help="'random' (seeded), 'first' or an index")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--effort", default="none", help="reasoning effort; puffin sends none")
    ap.add_argument("--max-out", type=int, default=0, help="override the recorded output cap")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    tools_plain = json.load(open(a.tools))
    tools_index = json.load(open(a.tools_index)) if a.tools_index else tools_plain
    paths = list(a.rollouts)
    if a.sessions_file:
        paths += [ln.strip() for ln in open(a.sessions_file) if ln.strip()]
    rng = random.Random(a.seed)
    jobs = []  # one job per session: (path, instructions, [request indices])
    for p in paths:
        instr, reqs = parse_rollout(p)
        if not reqs:
            continue
        w = min(a.window, len(reqs))
        if a.start == "random":
            s = rng.randrange(0, len(reqs) - w + 1)
        elif a.start == "first":
            s = 0
        else:
            s = min(int(a.start), len(reqs) - w)
        jobs.append((p, instr, reqs, list(range(s, s + w))))

    out = open(a.out, "w")
    lock = threading.Lock()
    queue = list(jobs)
    m0 = metrics(a.url)
    t_run0 = time.time()

    def worker(wid):
        while True:
            with lock:
                if not queue:
                    return
                p, instr, reqs, idxs = queue.pop(0)
            for k, i in enumerate(idxs):
                rq = reqs[i]
                rec_out = int(rq["usage"].get("output_tokens") or 0)
                body = {
                    "model": a.model, "instructions": instr,
                    "input": [clean_item(x) for x in rq["input"]],
                    "tools": tools_plain if "index-off" in p else tools_index, "tool_choice": "auto", "parallel_tool_calls": True,
                    "reasoning": {"effort": a.effort, "summary": "auto"},
                    "store": False, "stream": True,
                    "max_output_tokens": a.max_out or max(rec_out, 1),
                }
                if a.greedy:
                    body["temperature"] = 0.0
                res = send(a.url, body)
                rec = {"label": a.label, "worker": wid, "session": p.split("/")[-1], "req": i,
                       "cold": k == 0, "t_wall": time.time(),
                       "rec_in": rq["usage"].get("input_tokens"),
                       "rec_cached": rq["usage"].get("cached_input_tokens"), "rec_out": rec_out}
                rec.update(res)
                u = res.get("usage") or {}
                rec["in"] = u.get("input_tokens")
                rec["cached"] = (u.get("input_tokens_details") or {}).get("cached_tokens")
                rec["out"] = u.get("output_tokens")
                if rec.get("ttft") and rec.get("out") and rec["out"] > 1:
                    rec["decode_tps"] = (rec["out"] - 1) / max(rec["t_total"] - rec["ttft"], 1e-6)
                with lock:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"[w{wid}] {rec['session'][-20:]} #{i} in {rec.get('in')} (rec {rec['rec_in']}) "
                          f"cached {rec.get('cached')} out {rec.get('out')}/{rec_out} ttft "
                          f"{(rec.get('ttft') or 0):.2f} tot {rec.get('t_total', 0):.2f} "
                          f"tps {rec.get('decode_tps', 0):.1f} {rec.get('error', '')}", flush=True)

    ths = [threading.Thread(target=worker, args=(w,)) for w in range(a.concurrency)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    m1 = metrics(a.url)
    wall = time.time() - t_run0
    recs = [json.loads(ln) for ln in open(a.out)]
    ok = [r for r in recs if not r.get("error")]
    warm = [r for r in ok if not r["cold"]]
    d = {k: m1[k] - m0[k] for k in m0}
    gen = d["sglang:generation_tokens_total"]
    ver = d["sglang:spec_verify_calls_total"]
    nreq = d["sglang:e2e_request_latency_seconds_count"]
    # Server-side decode time: every request's end-to-end time minus its time to first token.
    # Exact for one stream; with several streams it double-counts overlapping requests.
    dec_s = d["sglang:e2e_request_latency_seconds_sum"] - d["sglang:time_to_first_token_seconds_sum"]
    summ = {
        "summary": True, "label": a.label, "requests": len(recs), "errors": len(recs) - len(ok),
        "wall_s": wall, "concurrency": a.concurrency, "greedy": a.greedy,
        "gen_tokens": gen, "verify_calls": ver,
        # Every request's first token comes from its prefill, not a verify step.
        "accept_per_step": (gen - nreq) / ver if ver else None,
        "server_decode_s": dec_s, "server_prefill_s": d["sglang:time_to_first_token_seconds_sum"],
        "server_decode_tps": (gen - nreq) / dec_s if dec_s > 0 else None,
        "server_step_ms": 1000 * dec_s / ver if ver else None,
        "server_mean_e2e_s": d["sglang:e2e_request_latency_seconds_sum"] / nreq if nreq else None,
        "aggregate_out_tps": sum(r["out"] or 0 for r in ok) / wall,
        "warm_decode_tps_median": st.median([r["decode_tps"] for r in warm if r.get("decode_tps")]) if warm else None,
        "warm_decode_tps_mean_weighted": (sum(r["out"] - 1 for r in warm if r.get("decode_tps")) /
                                          sum(r["t_total"] - r["ttft"] for r in warm if r.get("decode_tps")))
        if warm else None,
        "warm_ttft_median": st.median([r["ttft"] for r in warm if r.get("ttft")]) if warm else None,
        "warm_out_mean": st.mean([r["out"] for r in warm]) if warm else None,
        "in_vs_recorded_median_delta": st.median([(r["in"] or 0) - (r["rec_in"] or 0) for r in ok]) if ok else None,
    }
    with open(a.out, "a") as f:
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.exit(main())
