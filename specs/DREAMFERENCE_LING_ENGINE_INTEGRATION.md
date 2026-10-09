# ling-engine: Mightling integration

Part of the ling-engine specification; the index is [README.md](./README.md). Section numbers are those of the single-file `SPEC.md` this was split from (Oct 8, 2026 · revised the same day against Mightling's measured workload), so a citation of the form `§n` or `S§n` in the reports and the code still names the same text.

## 12. Mightling integration

`ling-serve` replaces SGLang as the server for this one model and changes nothing else in Mightling.

- **Selected by the registry.** The model's registry entry names the engine (`launch_overrides['engine'] = 'ling'`), the same switch that selects SGLang today; the launch builder emits only what follows the image, and the container comes from the same `docker run` prefix as every engine, so host safety stays engine-independent: the pre-load `check_host_safety()` checks and the PSI watchdog during load (a 20 GB memory-mapped blob is still a load) apply unchanged.
- **Memory.** A configured budget, not "whatever is free": weights and drafter ~19 GB, embedding and vision ~3.5 GB, sessions 30 GB, the shared-prefix store 2 GB, scratch 2 GB, about 57 GB in all, the same footprint as production's `--mem-fraction-static 0.5`, because the rest of the machine runs SWE-bench containers, the code index, the web UI and the desktop app above earlyoom's 5% line.
- **What Mightling calls.** `GET /v1/models` with the served model id and `max_model_len` (the launcher reads both); chat completions with streaming, tools and reasoning output (the `qwen3_coder` tool format and the `qwen3` reasoning split); completions, which `server start`'s NVFP4 canary uses and which caught the FlashInfer sampling bug in production; `GET /health`; `GET /metrics`.
- **The chat template is an input.** Mightling patches the checkpoint's template at launch (`ChatTemplatePatcher`; the original answered HTTP 400 to `high` and `minimal` efforts), so `ling-serve` takes a template file and records its hash.
- **Which model is running is asked of the server.** Mightling's tools ask `/v1/models`, never the config; `ling-serve` must report the same id the registry expects.
- **Idle.** Production uses `--sleep-on-idle`; `ling-serve` idles without spinning, so an idle Spark stays cool.
- **Fallback.** SGLang stays in the registry as the fallback engine for the same checkpoint, one flag away, until M5's release gate.

The programs' command lines are specified in [DREAMFERENCE_LING_ENGINE_CLI.md](./DREAMFERENCE_LING_ENGINE_CLI.md) and `ling-serve`'s HTTP API in [DREAMFERENCE_LING_ENGINE_API.md](./DREAMFERENCE_LING_ENGINE_API.md).
