#!/usr/bin/env python3
"""A long prefill beside a running decode (M4 §3, §5 "First requests"; stage 2 of the M4 draft).

Request A streams a long greedy answer; once it has decoded for --lead seconds, request B sends a prompt of
about --prefill-tokens tokens that no earlier request sent (so nothing is reused) and asks for a few tokens.
With ling-serve's interleaved scheduler B's prefill runs in --prefill-chunk-contended chunks between A's
steps. Reported: B's time to its answer (its prefill, shared with A's steps) and A's step period, from the
arrival of A's streamed tokens, before B, during B's prefill and after it. The server's own request log
line ("step period ... beside a prefill ...") gives the same periods measured on the engine thread; the
client's view adds the network and the event loop.

    interleave_bench.py --url http://127.0.0.1:18081 [--prefill-tokens 35000] [--lead 5] [--json OUT]

Standard library only, so it runs on the server machine as is.
"""
import argparse
import json
import random
import statistics as st
import threading
import time
import urllib.request

NO_THINK = {"enable_thinking": False}


def post(url, body, stream):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def code_text(lines, salt):
    return "".join(f"def {salt}_f{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(lines))


def bursts(times, gap=0.004):
    """The start of each burst of arrivals (a speculative step's tokens arrive together)."""
    out = []
    for t in times:
        if not out or t - last > gap:
            out.append(t)
        last = t
    return out


def periods(starts, t0, t1):
    """Gaps between consecutive burst starts that begin inside [t0, t1)."""
    return [b - a for a, b in zip(starts, starts[1:]) if t0 <= a < t1]


def summary(ps):
    if not ps:
        return {"n": 0}
    return {"n": len(ps), "mean_ms": 1e3 * st.mean(ps), "median_ms": 1e3 * st.median(ps), "max_ms": 1e3 * max(ps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18081")
    ap.add_argument("--prefill-tokens", type=int, default=35000)
    ap.add_argument("--lead", type=float, default=5.0, help="seconds A decodes before B is sent")
    ap.add_argument("--a-tokens", type=int, default=4000)
    ap.add_argument("--json")
    a = ap.parse_args()
    salt = f"r{random.randrange(10**9)}"
    arrivals, a_done = [], {}

    def run_a():
        body = {"messages": [{"role": "user", "content": f"({salt}) Write a 5000-word essay on the history of the "
                                                         "printing press, with many dates and names."}],
                "max_tokens": a.a_tokens, "temperature": 0, "stream": True, "chat_template_kwargs": NO_THINK}
        with post(a.url + "/v1/chat/completions", body, True) as r:
            for raw in r:
                line = raw.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    for ch in json.loads(line[6:]).get("choices") or []:
                        if (ch.get("delta") or {}).get("content"):
                            arrivals.append(time.monotonic())
        a_done["t"] = time.monotonic()

    ta = threading.Thread(target=run_a)
    t_start = time.monotonic()
    ta.start()
    while not arrivals and ta.is_alive():
        time.sleep(0.01)
    time.sleep(a.lead)
    # About 31 tokens a definition with the salt in its name.
    prompt = code_text(max(1, a.prefill_tokens // 31), salt) + "\nWhat does the first function return for x = 2? One line."
    t_b0 = time.monotonic()
    with post(a.url + "/v1/completions", {"prompt": prompt, "max_tokens": 4, "temperature": 0}, False) as r:
        b = json.loads(r.read())
    t_b1 = time.monotonic()
    a_alive_after_b = ta.is_alive()
    ta.join()
    starts = bursts(arrivals)
    out = {
        "prefill_tokens": b["usage"]["prompt_tokens"], "b_seconds": t_b1 - t_b0,
        "b_tokens_per_s": b["usage"]["prompt_tokens"] / (t_b1 - t_b0),
        "a_outlived_b": a_alive_after_b, "a_tokens_streamed": len(arrivals),
        "a_period_before": summary(periods(starts, t_start, t_b0)),
        "a_period_during": summary(periods(starts, t_b0, t_b1)),
        "a_period_after": summary(periods(starts, t_b1, a_done.get("t", t_b1))),
    }
    print(json.dumps(out, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)


if __name__ == "__main__":
    main()
