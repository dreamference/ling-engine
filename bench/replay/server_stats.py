"""Per-request server-side numbers from SGLang's request log (M0).

Reads `docker logs` of a server started with --log-requests --log-requests-level 1
--log-requests-format json, keeps the requests whose ids appear in the given replay outputs, and
reports for warm requests (prefix cache hit above half the prompt):
  - decode tokens/s, from the server's own first-token and finish timestamps;
  - step time, decode time over verify steps;
  - accepted tokens per step and its histogram;
  - time to first token as the server saw it (request received to prefill finished).

Usage: server_stats.py SERVER.log REPLAY.jsonl [REPLAY.jsonl...] [--json OUT]
"""
import collections
import json
import statistics as st
import sys


def load(log_path, rids):
    reqs = {}
    for line in open(log_path, errors="replace"):
        if '"request.finished"' not in line:
            continue
        try:
            d = json.loads(line[line.index("{"):])
        except ValueError:
            continue
        m = (d.get("out") or {}).get("meta_info") or {}
        rid = m.get("id") or d.get("rid")
        if rid in rids:
            reqs[rid] = m
    return reqs


def summarize(ms):
    warm = [m for m in ms if m.get("prompt_tokens") and (m.get("cached_tokens") or 0) > 0.5 * m["prompt_tokens"]]
    out = {"requests": len(ms), "warm": len(warm)}
    dec_t = dec_n = steps = 0
    hist = collections.Counter()
    ttft = []
    for m in warm:
        # Decode: from prefill finished (first token) to finish.
        t_dec = m["request_finished_ts"] - (m["prefill_finished_time"])
        n = (m.get("completion_tokens") or 0) - 1
        if n > 0 and t_dec > 0 and m.get("spec_verify_ct"):
            dec_t += t_dec
            dec_n += n
            steps += m["spec_verify_ct"]
        for i, c in enumerate(m.get("spec_accept_histogram") or []):
            hist[i] += c
        ttft.append(m["prefill_finished_time"] - m["request_received_ts"])
    out["warm_decode_tps"] = dec_n / dec_t if dec_t else None
    out["warm_step_ms"] = 1000 * dec_t / steps if steps else None
    out["warm_accept_per_step"] = dec_n / steps if steps else None
    out["warm_ttft_median_s"] = st.median(ttft) if ttft else None
    out["warm_ttft_mean_s"] = st.mean(ttft) if ttft else None
    out["warm_e2e_mean_s"] = st.mean([m["e2e_latency"] for m in warm]) if warm else None
    out["warm_out_mean"] = st.mean([m["completion_tokens"] for m in warm]) if warm else None
    out["warm_prompt_mean"] = st.mean([m["prompt_tokens"] for m in warm]) if warm else None
    out["accept_histogram"] = [hist[i] for i in range(max(hist) + 1)] if hist else []
    # Aggregate decode throughput over all requests (warm and cold): decoded tokens over the
    # union of the intervals in which at least one request was decoding. With several streams
    # this is the "tokens/s for N agents at once" number, free of prefill and idle time.
    iv = sorted((m["prefill_finished_time"], m["request_finished_ts"]) for m in ms
                if m.get("completion_tokens", 0) > 1)
    union = 0.0
    if iv:
        cs, ce = iv[0]
        for a, b in iv[1:]:
            if a > ce:
                union += ce - cs
                cs, ce = a, b
            else:
                ce = max(ce, b)
        union += ce - cs
    out["aggregate_decode_tps"] = (sum(m["completion_tokens"] - 1 for m in ms if m.get("completion_tokens", 0) > 1)
                                   / union) if union else None
    return out


def main():
    args = [a for a in sys.argv[1:]]
    js = None
    if "--json" in args:
        i = args.index("--json")
        js = args[i + 1]
        del args[i:i + 2]
    log, replays = args[0], args[1:]
    rids = set()
    for p in replays:
        for line in open(p):
            r = json.loads(line)
            if r.get("rid"):
                rids.add(r["rid"])
    reqs = load(log, rids)
    s = summarize(list(reqs.values()))
    s["matched"] = f"{len(reqs)}/{len(rids)}"
    print(json.dumps(s, indent=1))
    if js:
        json.dump(s, open(js, "w"), indent=1)


if __name__ == "__main__":
    main()
