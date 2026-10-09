# ling-engine specification

> - **Subject:** Spec: Model-Specific Inference for Qwen3.8-27B on DGX Spark
> - **Target hardware:** NVIDIA GB10 (Blackwell SM121, 128 GB unified memory), one node
> - **Served model:** Qwen3.8-27B NVFP4 with the DFlash2 drafter, the model Mightling serves today with SGLang
> - **Original:** Oct 8, 2026 · revised the same day against Mightling's measured workload; split into this directory on 2026-10-09

This directory holds the specification, split into focused documents. This page is the index.

**Where the monolith went.** Until 2026-10-09 the whole specification was one file, `SPEC.md`, at the repository root, with nineteen numbered sections. It was split here by section, and **every document keeps the original section numbers** (`## 16. …`, `### 16.10 …`), because the reports under `reports/` cite them as `S§n` and `SPEC §n`, the backlog's ranking tags are built on them, and four source comments name them. A citation of `§n` resolves through the map below. Only measured results move from the reports into these documents.

## Documents

| Document | What it covers |
|---|---|
| [DREAMFERENCE_LING_ENGINE_OVERVIEW.md](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) | §1 summary, §2 the measured agent workload, §3 goals and non-goals, §4 the GB10, §6 what production does, §7 the performance targets |
| [DREAMFERENCE_LING_ENGINE_MODEL.md](./DREAMFERENCE_LING_ENGINE_MODEL.md) | §5 Qwen3.8-27B's shapes, the real precision map, bytes per pass, the details the kernels must honour |
| [DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) | §8 the offline compiler and the resident runtime, §9 the five shape-specialized kernels per layer |
| [DREAMFERENCE_LING_ENGINE_SPECULATION.md](./DREAMFERENCE_LING_ENGINE_SPECULATION.md) | §10 the DFlash2 drafter, the context-lookup branch, tree verification, exactness |
| [DREAMFERENCE_LING_ENGINE_SESSIONS.md](./DREAMFERENCE_LING_ENGINE_SESSIONS.md) | §11 resident sessions, shared-prefix checkpoints, chunked prefill and batches |
| [DREAMFERENCE_LING_ENGINE_INTEGRATION.md](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md) | §12 how Mightling selects, launches and talks to `ling-serve`; the memory budget; the fallback to SGLang |
| [DREAMFERENCE_LING_ENGINE_CLI.md](./DREAMFERENCE_LING_ENGINE_CLI.md) | *Written from the code on 2026-10-09:* the command lines of `ling-serve`, `ling-run`, `ling-tokenize` and `ling-template`, their defaults, and the `LING_*` environment variables |
| [DREAMFERENCE_LING_ENGINE_API.md](./DREAMFERENCE_LING_ENGINE_API.md) | *Written from the code on 2026-10-09:* `ling-serve`'s HTTP API: the routes, the request fields, the streamed and unstreamed shapes, the error contract, what is ignored |
| [DREAMFERENCE_LING_ENGINE_VALIDATION.md](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) | §13 exactness, quality and the benchmark; §18 known pitfalls, each mapped to a regression test |
| [DREAMFERENCE_LING_ENGINE_MILESTONES.md](./DREAMFERENCE_LING_ENGINE_MILESTONES.md) | §14 the milestones with the standing status, the agreement gate and the exactness policy; §15 risks and open questions |
| [DREAMFERENCE_LING_ENGINE_RESEARCH.md](./DREAMFERENCE_LING_ENGINE_RESEARCH.md) | §16 the low-level speed-ups from the literature, measured or estimated; §17 sources |
| [DREAMFERENCE_LING_ENGINE_BACKLOG.md](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) | §19 what users of SGLang and vLLM asked for and what the four clients send, ranked, plus the maintainer's requests |

## Section map

| Section of the former `SPEC.md` | Now in |
|---|---|
| §1 | [DREAMFERENCE_LING_ENGINE_OVERVIEW.md](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) |
| §2 | [DREAMFERENCE_LING_ENGINE_OVERVIEW.md](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) |
| §3 | [DREAMFERENCE_LING_ENGINE_OVERVIEW.md](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) |
| §4 | [DREAMFERENCE_LING_ENGINE_OVERVIEW.md](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) |
| §5 | [DREAMFERENCE_LING_ENGINE_MODEL.md](./DREAMFERENCE_LING_ENGINE_MODEL.md) |
| §6 | [DREAMFERENCE_LING_ENGINE_OVERVIEW.md](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) |
| §7 | [DREAMFERENCE_LING_ENGINE_OVERVIEW.md](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) |
| §8 | [DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) |
| §9 | [DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) |
| §10 | [DREAMFERENCE_LING_ENGINE_SPECULATION.md](./DREAMFERENCE_LING_ENGINE_SPECULATION.md) |
| §11 | [DREAMFERENCE_LING_ENGINE_SESSIONS.md](./DREAMFERENCE_LING_ENGINE_SESSIONS.md) |
| §12 | [DREAMFERENCE_LING_ENGINE_INTEGRATION.md](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md) |
| §13 | [DREAMFERENCE_LING_ENGINE_VALIDATION.md](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) |
| §14 | [DREAMFERENCE_LING_ENGINE_MILESTONES.md](./DREAMFERENCE_LING_ENGINE_MILESTONES.md) |
| §15 | [DREAMFERENCE_LING_ENGINE_MILESTONES.md](./DREAMFERENCE_LING_ENGINE_MILESTONES.md) |
| §16 | [DREAMFERENCE_LING_ENGINE_RESEARCH.md](./DREAMFERENCE_LING_ENGINE_RESEARCH.md) |
| §17 | [DREAMFERENCE_LING_ENGINE_RESEARCH.md](./DREAMFERENCE_LING_ENGINE_RESEARCH.md) |
| §18 | [DREAMFERENCE_LING_ENGINE_VALIDATION.md](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) |
| §19 | [DREAMFERENCE_LING_ENGINE_BACKLOG.md](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) |

The two documents without a section number, `DREAMFERENCE_LING_ENGINE_CLI.md` and `DREAMFERENCE_LING_ENGINE_API.md`, were written on 2026-10-09 from the code and have no counterpart in the original file.
