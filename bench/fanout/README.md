# SGLang shared-context fan-out bench (2026-10-30)

Client-only bench against a running SGLang server on `127.0.0.1:8000` (it only calls
`/generate`, `/tokenize` and `/metrics`; it never changes the server).

- `fanout_bench.py`: one mode per measurement: `fanout` (N forks of one cached context,
  100-token answers), `score` (max_tokens 1, yes/no logprobs), `chat` (200 in, 200 out),
  `unshared` (one context per request) and `tail` (prefill rate at the end of a cached
  context). Every cell appends one JSON line with per-request `meta_info` digests and a
  0.5 s `/metrics` trace.
- `run_all.sh`: the sequence that was run. `ctx600k.txt` is any large Python source text
  (concatenated source files, about 600K tokens); it is not checked in.
- `analyze.py <data dir>`: prints the markdown tables of the report.
