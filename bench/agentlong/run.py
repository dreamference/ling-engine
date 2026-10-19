"""Replays the agentlong set (select_set.py) against ling-serve or SGLang at one stream and reports the decode.

The same sixteen request bodies go to either server, byte for byte (each checked against the
manifest's SHA-256), one at a time, after one short warm-up request that is not counted (it
captures the engine's step graphs). Around every request the server's own counters are read from
/metrics (both servers export SGLang's names), so the numbers are the server's, not the client's:

  decode tokens/s       sum over turns of (generated - 1) / sum of (end-to-end - time to first token)
  ms per step           sum of decode seconds / sum of verify passes
  tokens per step       sum of (generated - 1) / sum of verify passes (accepted drafts + 1 per pass)

The first token of every turn comes from its prefill and is left out of all three. Per turn the
output text (every streamed delta, by event type) and its SHA-256 are recorded, so two greedy runs
can be compared token for token (--compare).

Usage:
  run.py --url http://127.0.0.1:8300 --mode greedy|seed:N --out RUN.jsonl [--label L]
         [--bodies ~/bench-sets/agentlong/bodies.jsonl] [--manifest manifest.json] [--only t00,t03]
  run.py --compare A.jsonl B.jsonl       per-turn output identity and the two summaries
  run.py --summary RUN.jsonl [RUN.jsonl ...]
"""
import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.parse
import zlib

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
COUNTERS = ("sglang:spec_verify_calls_total", "sglang:generation_tokens_total", "sglang:prompt_tokens_total",
            "sglang:cached_tokens_total", "sglang:e2e_request_latency_seconds_sum",
            "sglang:e2e_request_latency_seconds_count", "sglang:time_to_first_token_seconds_sum")


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def metrics(url):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    c.request("GET", "/metrics")
    txt = c.getresponse().read().decode()
    return {n: sum(float(m) for m in re.findall(r"^" + re.escape(n) + r"\{[^}]*\} ([0-9.e+-]+)$", txt, re.M))
            for n in COUNTERS}


def metrics_after(url, before, timeout=10.0):
    """The counters once the request has been counted (SGLang updates them at the request's end)."""
    t_end = time.time() + timeout
    while True:
        m = metrics(url)
        if m["sglang:e2e_request_latency_seconds_count"] > before["sglang:e2e_request_latency_seconds_count"] \
                or time.time() > t_end:
            return m
        time.sleep(0.2)


def send(url, body):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=7200)
    t0 = time.perf_counter()
    c.request("POST", "/v1/responses", body=canonical(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"error": f"HTTP {r.status}: {r.read()[:300]!r}"}
    t_first = None
    parts, usage, buf, status = {}, None, b"", None
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
                    if t_first is None:
                        t_first = time.perf_counter()
                    parts[t] = parts.get(t, "") + str(j.get("delta", ""))
                elif t in ("response.completed", "response.incomplete"):
                    usage = (j.get("response") or {}).get("usage")
                    status = (j.get("response") or {}).get("status")
    t1 = time.perf_counter()
    text = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return {"t_total": t1 - t0, "ttft": (t_first - t0) if t_first else None, "usage": usage, "status": status,
            "text": parts, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def summarize(recs):
    ok = [r for r in recs if not r.get("error") and r.get("srv")]
    gen = sum(r["srv"]["gen"] - 1 for r in ok if r["srv"]["gen"] > 1)
    dec = sum(r["srv"]["decode_s"] for r in ok if r["srv"]["gen"] > 1)
    ver = sum(r["srv"]["verify"] for r in ok if r["srv"]["gen"] > 1)
    return {"summary": True, "turns": len(recs), "errors": len(recs) - len(ok), "decoded_tokens": gen,
            "verify_passes": ver, "decode_s": round(dec, 3),
            "decode_tps": round(gen / dec, 2) if dec else None,
            "ms_per_step": round(1000 * dec / ver, 2) if ver else None,
            "tokens_per_step": round(gen / ver, 3) if ver else None,
            "prefill_s": round(sum(r["srv"]["ttft_s"] for r in ok), 2),
            "prompt_tokens": sum(r["srv"]["prompt"] for r in ok)}


def load(path):
    recs = [json.loads(ln) for ln in open(path)]
    return [r for r in recs if not r.get("summary") and not r.get("header")], next(
        (r for r in recs if r.get("header")), {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--mode", default="greedy", help="greedy, or seed:N (sampled, a per-turn seed from N)")
    ap.add_argument("--out")
    ap.add_argument("--label", default="")
    ap.add_argument("--bodies", default=f"{HOME}/bench-sets/agentlong/bodies.jsonl")
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    ap.add_argument("--only", default="")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--summary", nargs="+")
    a = ap.parse_args()

    if a.summary:
        for p in a.summary:
            recs, hdr = load(p)
            print(p, hdr.get("label", ""), hdr.get("mode", ""), json.dumps(summarize(recs)))
        return 0
    if a.compare:
        (ra, ha), (rb, hb) = load(a.compare[0]), load(a.compare[1])
        bb = {r["id"]: r for r in rb}
        same = 0
        for r in ra:
            o = bb.get(r["id"])
            eq = o is not None and r.get("text_sha256") == o.get("text_sha256")
            same += eq
            print(f"{r['id']} in {r['srv']['prompt']}/{o['srv']['prompt'] if o else '-'} out {r['srv']['gen']}/"
                  f"{o['srv']['gen'] if o else '-'} passes {r['srv']['verify']}/{o['srv']['verify'] if o else '-'} "
                  f"{'identical' if eq else 'DIFFERENT'}")
        print(f"{same}/{len(ra)} turns identical")
        for p, recs in ((a.compare[0], ra), (a.compare[1], rb)):
            print(p, json.dumps(summarize(recs)))
        return 0 if same == len(ra) else 1

    man = {t["id"]: t for t in json.load(open(a.manifest))["turns"]}
    bodies = [json.loads(ln) for ln in open(a.bodies)]
    only = set(a.only.split(",")) if a.only else None
    for b in bodies:
        h = hashlib.sha256(canonical(b["body"]).encode()).hexdigest()
        if man.get(b["id"], {}).get("body_sha256") != h:
            sys.exit(f"{b['id']}: the body's hash does not match the manifest (rebuild with select_set.py)")
    if len(bodies) != len(man):
        sys.exit("the bodies file and the manifest name different turns")
    greedy = a.mode == "greedy"
    seed0 = None if greedy else int(a.mode.split(":", 1)[1])
    out = open(a.out, "w")
    out.write(json.dumps({"header": True, "label": a.label, "mode": a.mode, "url": a.url, "t": time.time(),
                          "manifest_sha256": hashlib.sha256(open(a.manifest, "rb").read()).hexdigest()}) + "\n")
    if not a.no_warmup:
        w = {"model": bodies[0]["body"]["model"], "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "Write a short Python function that reverses a list, then explain it."}]}],
             "stream": True, "store": False, "max_output_tokens": 128, "reasoning": {"effort": "none"}}
        if greedy:
            w["temperature"] = 0.0
        send(a.url, w)
    recs = []
    for b in bodies:
        if only and b["id"] not in only:
            continue
        body = dict(b["body"])
        if greedy:
            body["temperature"] = 0.0
        else:
            body["seed"] = 1 + zlib.crc32(f"{seed0}:{b['id']}".encode()) % (2**31 - 1)
        m0 = metrics(a.url)
        res = send(a.url, body)
        m1 = metrics_after(a.url, m0)
        d = {k: m1[k] - m0[k] for k in COUNTERS}
        e2e, ttft = d["sglang:e2e_request_latency_seconds_sum"], d["sglang:time_to_first_token_seconds_sum"]
        rec = {"id": b["id"], "rec_in": man[b["id"]]["rec_in"], "rec_out": man[b["id"]]["rec_out"], **res,
               "srv": {"prompt": int(d["sglang:prompt_tokens_total"]), "cached": int(d["sglang:cached_tokens_total"]),
                       "gen": int(d["sglang:generation_tokens_total"]), "verify": int(d["sglang:spec_verify_calls_total"]),
                       "ttft_s": ttft, "decode_s": e2e - ttft}}
        g, v, ds = rec["srv"]["gen"], rec["srv"]["verify"], rec["srv"]["decode_s"]
        recs.append(rec)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        print(f"{b['id']} in {rec['srv']['prompt']} (rec {rec['rec_in']}) cached {rec['srv']['cached']} out {g}/"
              f"{rec['rec_out']} passes {v} prefill {ttft:.1f}s decode {(g - 1) / ds if ds > 0 else 0:.1f} tok/s "
              f"{1000 * ds / v if v else 0:.1f} ms/step {(g - 1) / v if v else 0:.2f} tok/step {res.get('error', '')}",
              flush=True)
    s = summarize(recs)
    s["label"], s["mode"] = a.label, a.mode
    out.write(json.dumps(s) + "\n")
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
