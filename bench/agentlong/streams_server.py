"""Server-side decode intervals for an n_streams.py run (BATCHING §2), from the server's own request log.

The client cannot see when a request's decode starts: an answer that is only a tool call streams nothing until the
parser has the whole call (both servers hold tool-call text back), so the first streamed delta can come at the very
end, or never. The servers know: ling-serve logs per request its prefill seconds and its decode ("N tokens in S s",
passes, batched passes, step periods); SGLang started with --log-requests --log-requests-level 1
--log-requests-format json logs per request prefill_finished_time, request_finished_ts and spec_verify_ct. Each
client record is matched to its server line by (prompt tokens, output tokens), and its decode interval becomes
[t_end - decode_s, t_end] on the client's clock. From those, per stream and in all:

  decode tokens/s per request, pooled per stream (sum tokens / sum decode seconds)
  aggregate decode tokens/s while exactly N, and while at least one, request was decoding (tokens spread evenly)
  ms per pass (decode seconds / passes, per request: the request's own step period)
  ling-serve only: the share of passes taken batched, and the step period beside another request's prefill

Usage: streams_server.py RUN.jsonl SERVER.log [--json]
"""
import json
import re
import statistics
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import n_streams as ns  # noqa: E402

ENGINE = re.compile(r"request: session \d+(?: \([^)]*\))?, (\d+) prompt tokens \((\d+) reused[^)]*\), prefill ([\d.]+) s "
                    r"\((\d+) chunks, (\d+) beside a decode.*?\); (\d+) tokens in ([\d.]+) s .*?; (\d+) passes \((\d+) batched\)"
                    r".*?step period ([\d.]+) ms mean, [\d.]+ max over (\d+); beside a prefill ([\d.]+) ms mean, [\d.]+ max "
                    r"over (\d+)")


def engine_lines(path):
    out = []
    for ln in open(path, errors="replace"):
        m = ENGINE.search(ln)
        if m:
            g = m.groups()
            out.append({"prompt": int(g[0]), "reused": int(g[1]), "prefill_s": float(g[2]), "chunks": int(g[3]),
                        "chunks_beside": int(g[4]), "out": int(g[5]), "decode_s": float(g[6]), "passes": int(g[7]),
                        "batched": int(g[8]), "period_alone_ms": float(g[9]), "n_alone": int(g[10]),
                        "period_beside_ms": float(g[11]), "n_beside": int(g[12])})
    return out


def sglang_lines(path):
    out = []
    for ln in open(path, errors="replace"):
        if '"request.finished"' not in ln:
            continue
        try:
            d = json.loads(ln[ln.index("{"):])
        except ValueError:
            continue
        m = (d.get("out") or {}).get("meta_info") or {}
        if not m.get("prompt_tokens"):
            continue
        dec = m["request_finished_ts"] - m["prefill_finished_time"]
        out.append({"prompt": m["prompt_tokens"], "reused": m.get("cached_tokens") or 0,
                    "prefill_s": m["prefill_finished_time"] - m["request_received_ts"],
                    "out": m.get("completion_tokens") or 0, "decode_s": dec, "passes": m.get("spec_verify_ct") or 0})
    return out


def match(reqs, lines):
    pool = list(lines)
    for r in reqs:
        for i, s in enumerate(pool):
            if s["prompt"] == r["prompt"] and abs(s["out"] - r["out"]) <= 1:
                r["srv"] = pool.pop(i)
                break
    return [r for r in reqs if "srv" in r]


def report(path, log):
    reqs, summ = ns.load(path)
    n = summ.get("streams", 1)
    lines = engine_lines(log)
    kind = "engine"
    if not lines:
        lines, kind = sglang_lines(log), "sglang"
    got = match(reqs, lines)
    for r in got:  # the decode interval on the client's clock, from the server's own decode time
        r["t_first"] = r["t_end"] - r["srv"]["decode_s"]
    res = {"label": summ.get("label"), "mode": summ.get("mode"), "streams": n, "kind": kind, "matched": len(got),
           "requests": len(reqs), "wall_s": summ.get("wall_s"), "output_tokens": summ.get("output_tokens"),
           "aggregate_e2e_tps": summ.get("aggregate_e2e_tps")}
    per = []
    for j in range(n):
        rs = [r for r in got if r["stream"] == j and r["out"] > 1]
        g, d = sum(r["out"] - 1 for r in rs), sum(r["srv"]["decode_s"] for r in rs)
        per.append(round(g / d, 2) if d else None)
    res["stream_decode_tps"] = per
    g = sum(r["out"] - 1 for r in got if r["out"] > 1)
    d = sum(r["srv"]["decode_s"] for r in got if r["out"] > 1)
    p = sum(r["srv"]["passes"] for r in got if r["out"] > 1)
    res["decode_tps_pooled"] = round(g / d, 2) if d else None
    res["ms_per_pass"] = round(1000 * d / p, 1) if p else None
    res["tokens_per_pass"] = round(g / p, 3) if p else None
    res["aggregate_decode_tps_all_n"], res["all_n_decoding_s"], res["aggregate_decode_tps_any"] = ns.overlap_rates(got, n)
    res["decode_s_total"] = round(d, 1)
    res["prefill_s_total"] = round(sum(r["srv"]["prefill_s"] for r in got), 1)
    res["prefill_new_tokens"] = sum(r["srv"]["prompt"] - r["srv"]["reused"] for r in got)
    ttft = [r["srv"]["prefill_s"] for r in got]
    res["server_ttft_median_s"] = round(statistics.median(ttft), 2) if ttft else None
    res["server_ttft_max_s"] = round(max(ttft), 2) if ttft else None
    # the client's time to the end of the prefill, queueing included: t_end - decode - t0
    q = [r["t_first"] - r["t0"] for r in got]
    res["client_ttft_median_s"] = round(statistics.median(q), 2) if q else None
    res["client_ttft_max_s"] = round(max(q), 2) if q else None
    # Warm runs (n_streams.py --warm): round k is each stream's k-th request, sent together; a round lasts from its
    # first send to its last answer. Output tokens over the rounds' summed length is the follow-ups' own throughput
    # (their few new tokens of prefill and any waiting included, the priming requests excluded).
    by_stream = {}
    for r in sorted(reqs, key=lambda r: r["t0"]):
        by_stream.setdefault(r["stream"], []).append(r)
    rounds = {}
    for rs in by_stream.values():
        for k, r in enumerate(rs):
            rounds.setdefault(k, []).append(r)
    span = sum(max(r["t_end"] for r in rs) - min(r["t0"] for r in rs) for rs in rounds.values())
    res["rounds_s"] = round(span, 1)
    res["round_tps"] = round(sum(r["out"] for r in reqs) / span, 2) if span else None
    if kind == "engine" and p:
        res["batched_share"] = round(sum(r["srv"]["batched"] for r in got) / p, 3)
        na = sum(r["srv"]["n_alone"] for r in got)
        nb = sum(r["srv"]["n_beside"] for r in got)
        res["period_no_prefill_ms"] = round(sum(r["srv"]["period_alone_ms"] * r["srv"]["n_alone"] for r in got) / na, 1) if na else None
        res["period_beside_prefill_ms"] = round(sum(r["srv"]["period_beside_ms"] * r["srv"]["n_beside"] for r in got) / nb, 1) if nb else None
        res["share_beside_prefill"] = round(nb / (na + nb), 3) if na + nb else None
    return res


def main():
    r = report(sys.argv[1], sys.argv[2])
    print(json.dumps(r))


if __name__ == "__main__":
    main()
