"""The concurrency probe (BATCHING §4 item 2): what each conversation sees while another one prefills.

Three conversations against one server:
  A  decodes a long answer (a short prompt asking for ~1,500 tokens of text, greedy), started first;
  B  sends a cold agent prompt (one of the largest agentlong turns, 50-62K tokens, a different one each repeat)
     once A is decoding;
  C  is a warm follow-up: its conversation (another agentlong turn's previous request) is primed before the probe,
     and five seconds after B arrives (B's prefill under way) it sends that conversation plus a short user question
     answered in text.
Reported: C's time to first token (the client's first streamed delta: C answers in text, so it streams), B's
time to first token and prefill, and from ling-serve's request log A's step period alone and beside a prefill and
its decode tokens/s over the probe. Each conversation has its own prompt_cache_key, so it keeps its session.
Run --repeat R times (new conversations each time); every record and a summary go to --out. Start ling-serve with
--session-protect-seconds 0, or the repeats' new sessions wait for the last ones' protection to expire.

Usage: probe.py --url http://127.0.0.1:8300 --out PROBE.jsonl [--label L] [--serve-log LOG] [--repeat R]
"""
import argparse
import json
import os
import re
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as al  # noqa: E402

A_TEXT = ("Write the numbers from one to four hundred in English words, one number per line, nothing else.")
C_TEXT = "Before you go on: in one sentence, what have you found so far?"


def text_input(text):
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="probe")
    ap.add_argument("--serve-log", default="", help="ling-serve's log, for A's step periods (engine only)")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--decoders", type=int, default=1,
                    help="conversations decoding while B prefills: A, plus A2..AN with the same request in a variant")
    ap.add_argument("--bodies", default=os.path.expanduser("~/bench-sets/agentlong/bodies.jsonl"))
    a = ap.parse_args()
    bodies = [json.loads(ln) for ln in open(a.bodies)]
    size = lambda b: len(json.dumps(b["body"]["input"]))  # noqa: E731
    by_size = sorted(bodies, key=size)
    model = bodies[0]["body"]["model"]
    recs = []

    def body_a(rep):
        return {"model": model, "input": text_input(A_TEXT), "stream": True, "store": False, "max_output_tokens": 1500,
                "reasoning": {"effort": "none"}, "temperature": 0.0, "prompt_cache_key": f"{a.label}-A{rep}"}

    def body_b(rep):
        body = dict(by_size[-1 - rep % 4]["body"])
        body.update(temperature=0.0, prompt_cache_key=f"{a.label}-B{rep}")
        return body

    def body_c(rep, full):
        body = dict(by_size[len(by_size) // 2 + rep % 4]["body"])
        body.update(temperature=0.0, prompt_cache_key=f"{a.label}-C{rep}")
        prev = body["input"][:-2]  # the conversation as its previous request saw it
        if full:
            body["input"] = prev + text_input(C_TEXT)
            body["max_output_tokens"] = 64
            body["reasoning"] = {"effort": "none"}  # a text answer, so its first delta is its first token
        else:
            body["input"] = prev
            body["max_output_tokens"] = 1
        return body

    al.send(a.url, {"model": model, "input": text_input("Say hi."), "stream": True, "store": False,
                    "max_output_tokens": 8, "temperature": 0.0})
    for rep in range(a.repeat):
        al.send(a.url, body_c(rep, False))  # C's conversation resident before the probe
        out = {}

        def go(name, body):
            t0 = time.time()
            r = al.send(a.url, body)
            u = r.get("usage") or {}
            out[name] = {"t0": t0, "ttft": r.get("ttft"), "t_total": r.get("t_total"), "out": u.get("output_tokens", 0),
                         "prompt": u.get("input_tokens", 0),
                         "cached": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                         "error": r.get("error"), "text_sha256": r.get("text_sha256")}

        ta = threading.Thread(target=go, args=("A", body_a(rep)))
        ta.start()
        extra = []
        for i in range(2, a.decoders + 1):  # more decoders, each its own conversation (a different first line)
            bi = body_a(rep)
            bi["prompt_cache_key"] = f"{a.label}-A{i}-{rep}"
            bi["input"] = text_input(f"Variant {i}.\n" + A_TEXT)
            extra.append(threading.Thread(target=go, args=(f"A{i}", bi)))
            extra[-1].start()
        time.sleep(2.0)  # A is decoding
        tb = threading.Thread(target=go, args=("B", body_b(rep)))
        tb.start()
        time.sleep(5.0)
        tc = threading.Thread(target=go, args=("C", body_c(rep, True)))
        tc.start()
        for t in (ta, tb, tc, *extra):
            t.join()
        rec = {"kind": "probe", "rep": rep, **{k: v for k, v in out.items()}}
        recs.append(rec)
        A, B, C = out["A"], out["B"], out["C"]
        print(f"rep {rep}: C ttft {C['ttft'] or 0:.2f} s ({C['prompt'] - C['cached']} new tokens); B ttft "
              f"{B['ttft'] or 0:.2f} s ({B['prompt'] - B['cached']} new); A {A['out']} tokens in "
              f"{A['t_total'] - (A['ttft'] or 0):.1f} s ({(A['out'] - 1) / max(1e-9, A['t_total'] - (A['ttft'] or 0)):.1f} "
              f"tok/s) {A['error'] or ''}{B['error'] or ''}{C['error'] or ''}", flush=True)
    # A's step periods from ling-serve's request lines (the A requests are the ones with A's output count and a
    # prompt of under 200 tokens).
    periods = []
    if a.serve_log and os.path.exists(a.serve_log):
        pat = re.compile(r"^request: .* (\d+) prompt tokens .*?; (\d+) tokens in ([\d.]+) s .*?step period ([\d.]+) ms "
                         r"mean, [\d.]+ max over (\d+); beside a prefill ([\d.]+) ms mean, [\d.]+ max over (\d+)")
        for ln in open(a.serve_log):
            m = pat.match(ln)
            if m and int(m.group(1)) < 200 and int(m.group(2)) > 200:
                periods.append({"alone_ms": float(m.group(4)), "alone_n": int(m.group(5)),
                                "beside_ms": float(m.group(6)), "beside_n": int(m.group(7))})
    med = lambda xs: round(statistics.median(xs), 3) if xs else None  # noqa: E731
    s = {"summary": True, "label": a.label, "repeat": a.repeat,
         "c_ttft_s": med([r["C"]["ttft"] or 0 for r in recs]), "b_ttft_s": med([r["B"]["ttft"] or 0 for r in recs]),
         "a_decode_tps": med([(r["A"]["out"] - 1) / max(1e-9, r["A"]["t_total"] - (r["A"]["ttft"] or 0)) for r in recs]),
         "others_decode_tps": med([(r[k]["out"] - 1) / max(1e-9, r[k]["t_total"] - (r[k]["ttft"] or 0))
                                   for r in recs for k in r if k.startswith("A") and k != "A"]),
         "a_period_alone_ms": med([p["alone_ms"] for p in periods if p["alone_n"]]),
         "a_period_beside_ms": med([p["beside_ms"] for p in periods if p["beside_n"]]),
         "a_steps_beside_share": med([p["beside_n"] / max(1, p["alone_n"] + p["beside_n"]) for p in periods])}
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(s) + "\n")
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
