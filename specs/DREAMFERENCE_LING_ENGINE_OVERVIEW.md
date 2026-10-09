# ling-engine: Overview: the workload, the hardware and the targets

Part of the ling-engine specification; the index is [README.md](./README.md). Section numbers are those of the single-file `SPEC.md` this was split from (Oct 8, 2026 · revised the same day against Mightling's measured workload), so a citation of the form `§n` or `S§n` in the reports and the code still names the same text.

## 1. Summary

Decoding Qwen3.8-27B on a DGX Spark is bound by its 273 GB/s memory bus, not by compute. Every generated token streams the model's weights through that bus: 17.6 GB per pass for the checkpoint Mightling serves today, which caps plain decoding near 15 tokens/s however good the kernels are. Speculation is the only multiplier, and removing everything that is not weight traffic is the other half. The plan is Husky's: get more tokens out of each weight read, then remove every other cost.

**What we are speeding up is Mightling's agent, not a chat benchmark.** Section 2 measures it from 168 recorded agent sessions (11,431 model requests): 92% of a session's wall time is the model; a request writes 177 tokens on average, 71% of them tool-call arguments and at most a tenth hidden thinking; 46% of what it writes is copied from its own context; and every new session re-reads a ~15k-token prefix it shares with every other session. On that workload today's production server (SGLang with the DFlash2 drafter, section 6) decodes **47 tokens/s** at a **105 ms speculative step** for one agent on an otherwise idle Spark, and about 30 tokens/s at ~160 ms as logged with two sessions at once on a loaded machine (M0, [reports/M0.md](../reports/M0.md)). Its weight pass already streams at 78% of the bus; the gap to a tight engine is ~20–35 ms per step, not the ~65 ms this spec first assumed.

The target is Qwen3.8-27B, the Apache-2.0 dense member of the Qwen3.8 family released in August 2026: 64 layers in a 3:1 mix of Gated DeltaNet (linear attention) and gated full attention, a 248k vocabulary, and a trained DFlash2 block drafter already available. The other two family members do not fit the job: Qwen3.8-Max (2.4T, 95B active) needs four Sparks, and Qwen3.8-Flash-Next (176B MoE) runs at 28.5 tokens/s on one Spark with part of its weights on NVMe. We build a single-model engine for Qwen3.8-27B on the GB10 chip in two parts: an offline compiler, `ling-compile`, that fixes the precision map, folds the norms, lays the bytes out on disk in kernel order, instantiates one fused CUDA kernel per matrix shape and records each decode step as a CUDA graph; and a resident runtime, `ling-serve`, that keeps each conversation's KV cache and DeltaNet state on the GPU and serves Mightling as its model server.

Six levers. The first four are in order of payoff on one agent's session (section 7 has the arithmetic); the fifth multiplies throughput when several agents run; the sixth is a stretch:

1. **Step time at the bandwidth bound.** Shape-specialized fused CUDA kernels over pre-tiled weights, one CUDA graph per step, GPU-side sampling, FP8 KV and a replay scheme for the DeltaNet state: 105 ms per step today for one stream (M0), ≤ 84 ms targeted with the FP8 projections kept and ≤ 71 ms with them in NVFP4, **1.25–1.5×**; more under batching (lever 5).
2. **A hybrid draft.** DFlash2's 16 draft tokens plus a context-lookup branch of up to 32 tokens in the same verify, because agents copy: 4.95 accepted tokens per step today (M0's replay; a chain, not a tree), ≥ 5.8 targeted (6.2 is the simulated upper bound), **×1.2**.
3. **Short requests stay short.** FP4 prefill and no host work between the HTTP request and the first weight pass: a warm request's fixed cost from 0.68 s to ≤ 0.25 s, about 6% of model time. (M0: on an idle machine the server-side time to first token is already 0.31 s median, so this lever is worth ~3–6% and is folded into lever 4's FP4 prefill.)
4. **New sessions start warm.** DeltaNet state checkpoints at shared-prefix boundaries, so the system prompt and tool schema every session shares are computed once: 40% of all prefill work gone, but only ~2% of session time; this is the interactive-latency lever, the first answer of a new session in ≤ 1 s instead of 9–13 s.
5. **Several agents share one weight read.** Night Shift, SWE-bench, refine's two steps and subagents run 2–4 sequences at once; batching them into one pass gives ≥ 2.5× aggregate throughput over a single stream.
6. **An agent-tuned drafter (stretch).** The drafter fine-tuned on the user's own agent traces, on the Spark, overnight, so acceptance rises toward 8 without any data leaving the machine.

Headline targets, single stream on a replay of recorded agent sessions (every one is a gate in [section 14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md); M0's measurements are in [reports/M0.md](../reports/M0.md) and restated below the table):

| | Production today | Target | Stretch |
| --- | --- | --- | --- |
| Decode on agent output | 47 tokens/s idle, one stream (~30 as logged: two streams, loaded host) | **≥ 70 tokens/s (1.5× the idle figure)** | ≥ 100 (2.1×) |
| Mean model time per agent request | ~6.6 s | ≤ 2.8 s | ≤ 2.0 s |
| First answer of a new session | 9–13 s | ≤ 1 s | ≤ 0.5 s |
| Agent session wall time | 1× | **≥ 1.9–2.1× faster** | ≥ 2.4–2.8× faster |
| Aggregate decode, 4 agents at once | ~135 tokens/s while 4 decode (148 ms step), 69 end to end (M0) | ≥ 175 tokens/s | ≥ 220 |

Greedy output stays identical to non-speculative decoding at the same precision. A SWE-bench task that takes 16 minutes with the refine mode today would take about 8.

**Restated by M0 (8 October 2026).** The bus delivers 262 GB/s to a GPU read stream (96% of 273), not the 245 GB/s assumed below. One agent on an idle Spark decodes at 47 tokens/s with production's flags (105 ms step, 4.95 accepted tokens); the ~30 tokens/s and ~160 ms used in sections 2, 6 and 7 are what production logged with two concurrent SWE-bench sessions on a machine also running containers and indexers (two replayed streams: 34 tokens/s each). Production's verify already streams its weights at 204 GB/s (78% of the bus): NVFP4 FFN at 208–225 GB/s, FP8 in-projections at 218, the narrow FP8 out-projections at 140. So lever 1 is worth 1.25× (FP8 projections kept, ~84 ms step) to 1.5× (NVFP4 projections, ~71 ms), not 1.8–2.1×; the ≥ 70 tokens/s target needs the NVFP4 projections or the lookup branch; and batching (lever 5) is where the measured gap is widest, because production's per-sequence DeltaNet and KV traffic grows to 40 ms of a 148 ms step at four streams. Tuned SGLang is production with `--speculative-num-draft-tokens 12` instead of 16. That is +7% decode, measured on the replay and exact; it is recommended for production. FP8 KV, other draft lengths and SGLang 0.5.21 gave nothing, and SGLang's DeltaNet replay does not apply to this model. Against tuned SGLang the engine's measured-basis margin is 1.25–1.47× on the step alone. Against that idle single-stream baseline, a mean request (177 tokens) goes from 0.31 s + 177/47 = 4.1 s to 0.25 s + 177/70 = 2.8 s at the target, 1.46×. The sections below keep their original arithmetic; where it differs, M0's numbers govern.

## 2. The workload: what Mightling's agent asks of the model

Measured on this machine from the SWE-bench rounds of 3–7 October 2026 (`im-*` runs: 168 `puffin exec` sessions, 11,431 model requests, recorded rollouts and token counts per request), tokenized with the checkpoint's own tokenizer. These are the numbers the targets are built on.

| Quantity | Value | What it means for the engine |
| --- | --- | --- |
| Model share of session wall time | 92% (an upper bound: two sessions ran at once, so it includes queueing) | Engine speed turns almost one for one into agent speed |
| Output per request | median 98, mean 177, p90 387 tokens | Many short decodes: fixed per-request costs matter |
| What the output is | Of the visible output, 71% tool-call arguments and 29% assistant text; visible tokens are 90% of the counted output, so hidden thinking and template tokens are at most 10% | Structured, copy-heavy text, not long prose or thinking traces |
| Copyable output | 46% of visible output tokens continue a 3-token match in the session's own context | A context-lookup drafter has real material to work with |
| Context per request | median 25.7k, p90 43k, p99 48k tokens (these runs capped context at 49,152) | Decode always reads a 25k-token KV and a full DeltaNet state |
| Prefix cache hit | 96.9% of input tokens | Prefill is small per request, but not zero |
| Uncached tokens, warm request | median 297, mean 513, p90 917 | One short prefill before each answer |
| Uncached tokens, first request of a session | median 14.7k; 143 such requests carry 40% of all uncached tokens | The shared prefix is recomputed for almost every session ([section 11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)) |
| Latency, warm request | 0.68 s + 33.2 ms per output token (least-squares fit) | 30 tokens/s decode plus a fixed cost per request |
| Latency, first request of a session | median 16.5 s (104 output tokens) | 9–13 s before the first token: ~13 s as recorded, two sessions at a time; ~9 s for 14.7k tokens at the controlled 1,700 tokens/s |

Production's own counters agree: since its last start it generated 3.09M tokens in 615,532 verify steps (5.0 tokens per step), and its log of 2,290 single-stream decode intervals (30 September – 1 October, uncontrolled) shows a median of 30.1 tokens/s at an accept length of 4.62–4.87 out of 16 draft tokens, at a median context of 11.7k tokens.

Two consequences shape everything below. First, the right speed benchmark is a replay of these sessions ([section 13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md)), not a prose or math suite: the greedy microbenchmarks in Mightling's AGENTS.md (prose 25.5, code 50.3, JSON 87.0 tokens/s at short context) overstate agent throughput, and the gains here are larger on the agent workload than on those microbenchmarks, because the agent runs at long context and copies text. Second, concurrency is normal, not exotic: Night Shift, SWE-bench, refine's study and fix steps, subagents and paired nodes' lanes all run more than one agent.

## 3. Goals and non-goals

The engine runs exactly one model, Qwen3.8-27B, and every design choice may assume that.

Goals:

- Make Mightling's agent faster end to end: per-request latency and session wall time on the replay benchmark of [section 13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) are the headline metrics.
- Decode at or above 90% of the measured memory-bandwidth bound for a single sequence, then multiply that by speculation.
- Produce the same output as plain decoding: identical tokens under greedy sampling, the same distribution under temperature sampling, via exact speculative acceptance.
- Start a new session from shared-prefix state, and answer a follow-up in a cached conversation with first-token latency of one weight pass plus its new tokens' prefill.
- Batch 2–4 concurrent sequences into one weight pass, so overnight and multi-agent work gets aggregate throughput nearly for free.
- Live inside Mightling's memory budget ([section 12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)): about 60 GB for weights, sessions and scratch, the same as production's `--mem-fraction-static 0.5`, configurable.
- Serve the standard streaming chat API so existing clients work unchanged, including `reasoning_effort`, `enable_thinking` and `preserve_thinking`.
- Support thinking and non-thinking modes, the Qwen3 tool-call format, and image input through the model's own vision encoder.

Non-goals:

- Running any other model, including Qwen3.8-Flash-Next or Qwen3.6-27B. A new checkpoint with different shapes means recompiling.
- Multi-node inference across two Sparks over ConnectX-7. (A second paired Spark is a second engine and a second lane, as Mightling already does.)
- Training or fine-tuning the target model. Fine-tuning or retraining the DFlash2 drafter is in scope.
- New quantization research. We start from the precision map of the checkpoint Mightling serves and only decide which tensors move from FP8 to NVFP4. Two lower-byte formats were considered and are deliberately left out of the targets ([section 5.3](./DREAMFERENCE_LING_ENGINE_MODEL.md)): 2:4 structured sparsity and ~3-bit weights.
- Video input and contexts beyond the native 262k tokens, in the first release.
- Serving many users at high concurrency. Batching stops at the handful of agents one owner runs.

## 4. Hardware: what the GB10 gives and takes away

The DGX Spark is a compute-rich, bandwidth-poor machine, which is exactly the profile speculative decoding exploits. Its GB10 pairs a Blackwell GPU (compute capability 12.1, 5th-generation tensor cores with native FP4 and FP8) with a 20-core Arm CPU on one coherent 128 GB pool of LPDDR5x behind a 256-bit, 273 GB/s bus ([NVIDIA DGX Spark hardware docs](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)). NVIDIA rates it at 1 PFLOP of FP4 with sparsity, about 500 TFLOPS dense FP4 and 250 TFLOPS dense FP8, with a 24 MB GPU L2 (approximate, from NVIDIA's Hot Chips 2025 talk). The GB10 TDP is 140 W. Two copy engines, a 4 TB NVMe and a 200 Gb/s ConnectX-7 round it out; the ConnectX-7 is out of scope here.

| Quantity | Value | Consequence for this design |
| --- | --- | --- |
| Memory bandwidth | 273 GB/s, shared by CPU and GPU; **262 GB/s measured** for a GPU read stream (M0), and the GPU loses exactly what concurrent CPU streams take | A 17.6 GB weight pass takes 64 ms at peak, 67 ms at the measured 262 GB/s; nothing can beat ~15 tokens/s per weight read |
| Dense FP8 compute | ~250 TFLOPS (derived from 1 PFLOP sparse FP4) | ~900 FLOP per byte read: verifying 32 rows per pass costs compute but no bandwidth |
| Unified memory | 128 GB (121 GB usable per sparkrun) | Mightling gives the model server about half ([section 12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)); there is no separate host RAM to offload to |
| GPU L2 | ~24 MB | The 3.1 MB-per-layer DeltaNet state and all activations stay on chip between kernels |
| Observed generic engines | 38 tokens/s for Llama 3.1 8B q4_K_M in Ollama, i.e. ~190 GB/s effective | Generic runtimes reach 65–75% of the bus; our target is 90% |

The arithmetic that shapes everything else: a single-row decode step does about 2 FLOP per weight, so it runs the tensor cores at well under 1% of peak. A row costs about 49 GFLOP (2 × 24.4B weights), so 64 rows per pass are about 3 TFLOP, 10–15 ms of compute that a 75 ms weight pass largely hides; beyond about 100 rows compute starts to add to the step. Tensor-core tiles below 64 rows also run inefficiently. This design therefore treats 64 rows per pass as the practical ceiling, shared between speculation depth and batch: 16–48 rows for one sequence, 16 per sequence for four.

The bus must be measured, not assumed. Peak is 273 GB/s; sustained GPU streaming on LPDDR5x usually lands lower, and the CPU's own traffic shares the same pins. M0 measures a weight-streaming microbenchmark on the actual box, and every target in this document is restated against that number. The plan below assumes 90% of peak, 245 GB/s, is reachable by a well-written kernel.

One Spark-specific simplification: on this machine "GPU memory" and "host memory" are the same DRAM. There is nothing to offload sessions to, so the engine manages one pool with its own paging and eviction, and never copies state across a PCIe bus that does not exist.

Sources: [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) · [GB10 at Hot Chips 2025](https://www.hc2025.hotchips.org/assets/program/conference/day2/21_nvidia_skende_final.pdf) · [Ollama DGX Spark performance](https://registry.ollama.ai/blog/nvidia-spark-performance) · [llama.cpp on DGX Spark](https://jetsonhacks.com/wp-content/uploads/2025/10/spark-llamacpp-bench.html)

## 6. The baseline: what production does today

Mightling serves the model with SGLang in Docker (`lmsysorg/sglang`, launched by `SGLangLaunchBuilder` from the registry entry `qwen3.8-27b-nvfp4-dflash2`). The flags that matter here: `--speculative-algorithm DFLASH` with the NVFP4 drafter and `--speculative-num-draft-tokens 16`; FlashInfer attention; `--sampling-backend pytorch` (FlashInfer's sampler returned token 0 on untruncated sampling); BF16 KV; BF16 DeltaNet state with the `extra_buffer` radix-cache strategy; chunked prefill of 8,192 tokens without a prefill CUDA graph; decode CUDA graphs up to batch 8 and `torch.compile` up to batch 4; two continuous decode steps per scheduler pass; `--mem-fraction-static 0.5`; at most 8 running requests; 262,144-token context.

| Measurement | Value | Provenance |
| --- | --- | --- |
| Decode, agent workload, single stream | ~30 tokens/s | Server log, 2,290 intervals; least-squares fit over 11,431 requests (33.2 ms per token) |
| Accepted tokens per step | 4.62 median, 4.87 mean (of 16 drafted); 5.0 over all traffic | Server log; Prometheus counters |
| Step time | ~160 ms | Derived: 4.87 tokens per step at 30 tokens/s |
| Fixed cost per warm request | 0.68 s | Least-squares fit over warm requests |
| First request of a session | 9–13 s to the first token, 14.7k uncached tokens | Rollouts: median 16.5 s with 104 output tokens, two sessions at a time; ~9 s at the controlled prefill rate |
| Prefill | ~1,700 tokens/s, ~1,000 at 116k context | Mightling's controlled measurement (AGENTS.md) |
| Greedy microbenchmarks, short context | prose 25.5, code 50.3, JSON 87.0 tokens/s | Same |
| Two streams at once | 27.4 tokens/s each (55 aggregate) | Server log, 14 intervals only |

The public recipe the first draft of this spec compared against ([OpenZeka, 47.9 tokens/s](https://blog.openzeka.com/en/qwen3-8-27b-on-dgx-spark-48-tok-s/), 128-token prompts) is a different point: short prompts, a different workload. The comparison that matters is production on the agent workload, which is lower; part of that gap may be tuning, which is why M0 also measures a **tuned SGLang** (FP8 KV, the drafter's block size, CUDA graph and compile settings, the sampling backend) and every engine gate must beat the tuned number, not the default one.

## 7. Performance targets

Every decode target is tokens accepted per step divided by step time, so the table separates the two. Targets assume 245 GB/s achieved and are restated in M0 against the measured bus and the tuned SGLang baseline. Where a target depends on the NVFP4 projections passing the quality gate, both values are given.

| Metric | Physical ceiling, 1 token per pass | Production today | Target | Stretch |
| --- | --- | --- | --- | --- |
| Step time at 25.7k context, ≤ 48-row verify (ms) | 72 (weights only) | ~160 | ≤ 90 (≤ 80 with NVFP4 projections) | 75 |
| Accepted tokens per step, agent replay | 1 | 4.87 | ≥ 5.8 | ≥ 8 (agent-tuned drafter) |
| Decode on agent output (tokens/s) | 14 | ~30 | ≥ 70 | ≥ 100 |
| Fixed cost per warm request (s) | 0.07 | 0.68 | ≤ 0.25 | 0.15 |
| First token, new session with a shared prefix (s) | 0.07 | 9–13 | ≤ 1 | 0.5 |
| Prefill at 25k context (tokens/s) | ~8,000 (FP4 FFN, FP8 projections, dense compute) | ~1,700 | ≥ 3,000 | 4,000 |
| Mean model time per agent request (s) | — | ~6.6 | ≤ 2.8 | 2.0 |
| Agent session wall time on replay, tool time included | — | 1× | ≥ 1.9–2.1× faster | 2.4–2.8× faster |
| Aggregate decode, 4 agents at once (tokens/s) | — | measured in M0 | ≥ 175 | 220 |
| Greedy microbenchmarks, short context: prose / code / JSON (tokens/s) | — | 25.5 / 50.3 / 87.0 | ≥ 40 / 80 / 140 | — |
| Host overhead per step (ms) | 0 | several (Python scheduler) | ≤ 1 | 0.5 |
| Greedy output vs plain decoding | identical | identical | identical | identical |

How the agent rows follow. At 85 ms per step, 5.8 accepted tokens give 68 tokens/s and 6.2 give 73; at 75 ms they give 77 and 83; an agent-tuned drafter at 8 gives 94–107. A mean request (177 tokens) then takes 0.25 s + 177/70 = 2.8 s instead of 0.68 s + 177/30 = 6.6 s, 2.4× faster; with 92% of session time in the model, the session is 1/(0.08 + 0.92/2.4) = 2.1× faster, and 1.9× if the true model share is 85% (92% is an upper bound, section 2). M0 recomputes this gate from the replay's measured model share. The 4-agent row assumes 16 draft rows per sequence (64 per pass), about 5.3 accepted each and a ~110 ms step: 4 × 5.3 / 0.11 = 190 tokens/s.

The agent rows hold for the agent as Mightling configures it today, which thinks little (section 2). With heavy thinking (`reasoning_effort` xhigh), output becomes prose-like and acceptance falls toward the ~3.3 tokens per step reported for long prose: decode then lands near the prose microbenchmark's target, about 40–45 tokens/s against ~25 today, and the per-request and session rows shrink accordingly. M0's replay set therefore includes sessions run with thinking on, and both numbers are reported.

Why the microbenchmarks gain less than the agent: they run at short context, where production's BF16 KV and per-token DeltaNet states cost little, and their text gives the lookup branch nothing to copy. They are kept so Mightling's existing numbers stay comparable.

Two things are deliberately not promised. Long-context decode slows with the KV read: at 262k tokens an FP8 cache adds 8.6 GB to the pass, so the decode rows scale down by about a third there. And the stretch column needs the agent-tuned drafter and the NVFP4 projections, both of which carry a quality or acceptance risk that M3 and M5 measure before anyone depends on them.
