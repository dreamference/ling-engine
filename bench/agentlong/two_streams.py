"""Two agentlong streams at once against one server (bench only; BACKLOG 26 tree/rows: the batched verify of two
sequences takes 24 rows at block 12). The sixteen turns are split in two lists (even and odd positions in the
manifest), each list sent one request at a time by its own thread, so the server decodes two sequences side by side
(prefills beside the other's decode). Per request the client records the output, its SHA-256, the time to the first
delta and the end; per stream and in all, decode tokens/s = sum (output tokens - 1) / sum (end - first delta), and the
aggregate over the window when both streams were decoding. Greedy by default (outputs comparable across builds).
Usage: two_streams.py --url http://127.0.0.1:8300 --out RUN.jsonl [--label L] [--mode greedy|seedN]
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as al  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", default="greedy")
    ap.add_argument("--bodies", default=os.path.expanduser("~/bench-sets/agentlong/bodies.jsonl"))
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json"))
    a = ap.parse_args()
    man = {t["id"]: t for t in json.load(open(a.manifest))["turns"]}
    bodies = [json.loads(ln) for ln in open(a.bodies)]
    for b in bodies:
        if man.get(b["id"], {}).get("body_sha256") != hashlib.sha256(al.canonical(b["body"]).encode()).hexdigest():
            sys.exit(f"{b['id']}: the body's hash does not match the manifest")
    greedy = a.mode == "greedy"
    seed0 = None if greedy else int(a.mode[4:])
    lists = [bodies[0::2], bodies[1::2]]
    w = {"model": bodies[0]["body"]["model"], "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "Write a short Python function that reverses a list, then explain it."}]}],
         "stream": True, "store": False, "max_output_tokens": 128, "reasoning": {"effort": "none"}, "temperature": 0.0}
    al.send(a.url, w)
    recs = [[], []]

    def worker(i):
        for b in lists[i]:
            body = dict(b["body"])
            if greedy:
                body["temperature"] = 0.0
            else:
                body["seed"] = 1 + zlib.crc32(f"{seed0}:{b['id']}".encode()) % (2**31 - 1)
            t0 = time.time()
            r = al.send(a.url, body)
            out = (r.get("usage") or {}).get("output_tokens", 0)
            recs[i].append({"id": b["id"], "stream": i, "t0": t0, "t_first": t0 + (r.get("ttft") or 0),
                            "t_end": t0 + r.get("t_total", 0), "out": out, "text_sha256": r.get("text_sha256"),
                            "error": r.get("error")})
            print(f"stream {i} {b['id']} out {out} decode {(out - 1) / max(1e-9, r.get('t_total', 0) - (r.get('ttft') or 0)):.1f} tok/s",
                  flush=True)

    th = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    t_start = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t_start
    allr = recs[0] + recs[1]
    summ = {"summary": True, "label": a.label, "mode": a.mode, "wall_s": round(wall, 1)}
    for i in range(2):
        g = sum(r["out"] - 1 for r in recs[i] if r["out"] > 1)
        d = sum(r["t_end"] - r["t_first"] for r in recs[i] if r["out"] > 1)
        summ[f"stream{i}_tps"] = round(g / d, 2) if d else None
    # Both decoding: the overlap of the two streams' decode intervals, and the tokens each made in it (pro rata).
    iv = [[(r["t_first"], r["t_end"], r["out"] - 1) for r in recs[i] if r["out"] > 1] for i in range(2)]
    both_s = tok = 0.0
    for (a0, a1, ga) in iv[0]:
        for (b0, b1, gb) in iv[1]:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                both_s += hi - lo
                tok += ga * (hi - lo) / (a1 - a0) + gb * (hi - lo) / (b1 - b0)
    summ["both_decoding_s"] = round(both_s, 1)
    summ["aggregate_tps_both"] = round(tok / both_s, 2) if both_s else None
    summ["errors"] = sum(1 for r in allr if r["error"])
    with open(a.out, "w") as f:
        for r in sorted(allr, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
