#!/usr/bin/env python3
"""Clocks, power and temperature across a long decode (spec 16.1 item 4).

Drives a server with the chat-completions API with back-to-back long completions on N concurrent streams
for DURATION seconds while `nvidia-smi --query-gpu ... -l 1` logs the SM and memory clocks, power,
temperature, utilization and the active throttle reasons once a second. Prints a summary (min /
median / max of each column, the throttle reasons seen, tokens generated and tokens/s per stream)
and leaves the per-second CSV beside it for inspection; the CSV is not meant to be committed.

Usage: decode_clocks.py [--url http://127.0.0.1:8000] [--duration 600] [--streams 2]
                        [--max-tokens 4096] [--csv clocks.csv]
"""
import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.request

PROMPTS = [
    "Write a long, detailed design document for a job scheduler in C++20: requirements, data "
    "structures, the scheduling algorithm, failure handling, testing and a complete code listing "
    "of the core classes. Do not stop early; aim for the maximum length.",
    "Explain, at length and with worked examples, how a modern CPU cache hierarchy, a TLB and DRAM "
    "page policy interact, then write a complete C++ program that measures each level's latency "
    "and explain every line. Do not stop early; aim for the maximum length.",
]

QUERY = ("timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,"
         "clocks_throttle_reasons.active")


def stream(url, model, idx, deadline, max_tokens, out):
    n = 0
    while time.time() < deadline:
        body = json.dumps({
            "model": model, "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
            "messages": [{"role": "user", "content": PROMPTS[(idx + n) % len(PROMPTS)]}],
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=max(60, max_tokens)) as r:
                j = json.load(r)
        except Exception as e:  # noqa: BLE001 - a failed request is recorded and the stream goes on
            out.append((idx, 0, time.time() - t0, str(e)[:80]))
            continue
        out.append((idx, j["usage"]["completion_tokens"], time.time() - t0, ""))
        n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--streams", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--csv", default="clocks.csv")
    a = ap.parse_args()
    with urllib.request.urlopen(a.url + "/v1/models", timeout=10) as r:
        model = json.load(r)["data"][0]["id"]
    csv = open(a.csv, "w")
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=" + QUERY, "--format=csv,noheader,nounits",
                            "-l", "1"], stdout=csv, stderr=subprocess.STDOUT)
    t_start = time.time()
    deadline = t_start + a.duration
    out, threads = [], []
    for i in range(a.streams):
        t = threading.Thread(target=stream, args=(a.url, model, i, deadline, a.max_tokens, out))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t_start
    smi.terminate()
    smi.wait()
    csv.close()

    rows = [line.split(", ") for line in open(a.csv) if line.count(",") >= 6]
    cols = {"sm MHz": [], "mem MHz": [], "W": [], "degC": [], "util %": []}
    reasons = {}
    for r in rows:
        for k, x in zip(cols, r[1:6]):
            try:
                cols[k].append(float(x))  # a column nvidia-smi reports as [N/A] is left out
            except ValueError:
                pass
        key = r[6].strip()
        reasons[key] = reasons.get(key, 0) + 1
    print("model %s; %d streams; %.0f s; %d samples" % (model, a.streams, wall, len(rows)))
    print("| column | min | median | max |")
    print("| --- | --- | --- | --- |")
    for k, v in cols.items():
        if v:
            print("| %s | %.0f | %.0f | %.0f |" % (k, min(v), statistics.median(v), max(v)))
    print("throttle reasons (active mask: seconds):", reasons)
    toks = sum(n for _, n, _, _ in out)
    fails = [e for _, _, _, e in out if e]
    print("requests %d (failed %d), completion tokens %d, %.1f tokens/s total, %.1f per stream"
          % (len(out), len(fails), toks, toks / wall, toks / wall / a.streams))
    per = [n / s for _, n, s, e in out if not e and s > 0]
    if per:
        print("per-request tokens/s: min %.1f median %.1f max %.1f"
              % (min(per), statistics.median(per), max(per)))
    if fails:
        print("first failure:", fails[0])


if __name__ == "__main__":
    main()
