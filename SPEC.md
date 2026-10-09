# Spec: Model-Specific Inference for Qwen3.8-27B on DGX Spark

Oct 8, 2026 · revised the same day against Mightling's measured workload

## 1. Summary

Decoding Qwen3.8-27B on a DGX Spark is bound by its 273 GB/s memory bus, not by compute. Every generated token streams the model's weights through that bus: 17.6 GB per pass for the checkpoint Mightling serves today, which caps plain decoding near 15 tokens/s however good the kernels are. Speculation is the only multiplier, and removing everything that is not weight traffic is the other half. The plan is Husky's: get more tokens out of each weight read, then remove every other cost.

**What we are speeding up is Mightling's agent, not a chat benchmark.** Section 2 measures it from 168 recorded agent sessions (11,431 model requests): 92% of a session's wall time is the model; a request writes 177 tokens on average, 71% of them tool-call arguments and at most a tenth hidden thinking; 46% of what it writes is copied from its own context; and every new session re-reads a ~15k-token prefix it shares with every other session. On that workload today's production server (SGLang with the DFlash2 drafter, section 6) decodes **47 tokens/s** at a **105 ms speculative step** for one agent on an otherwise idle Spark, and about 30 tokens/s at ~160 ms as logged with two sessions at once on a loaded machine (M0, [reports/M0.md](reports/M0.md)). Its weight pass already streams at 78% of the bus; the gap to a tight engine is ~20–35 ms per step, not the ~65 ms this spec first assumed.

The target is Qwen3.8-27B, the Apache-2.0 dense member of the Qwen3.8 family released in August 2026: 64 layers in a 3:1 mix of Gated DeltaNet (linear attention) and gated full attention, a 248k vocabulary, and a trained DFlash2 block drafter already available. The other two family members do not fit the job: Qwen3.8-Max (2.4T, 95B active) needs four Sparks, and Qwen3.8-Flash-Next (176B MoE) runs at 28.5 tokens/s on one Spark with part of its weights on NVMe. We build a single-model engine for Qwen3.8-27B on the GB10 chip in two parts: an offline compiler, `ling-compile`, that fixes the precision map, folds the norms, lays the bytes out on disk in kernel order, instantiates one fused CUDA kernel per matrix shape and records each decode step as a CUDA graph; and a resident runtime, `ling-serve`, that keeps each conversation's KV cache and DeltaNet state on the GPU and serves Mightling as its model server.

Six levers. The first four are in order of payoff on one agent's session (section 7 has the arithmetic); the fifth multiplies throughput when several agents run; the sixth is a stretch:

1. **Step time at the bandwidth bound.** Shape-specialized fused CUDA kernels over pre-tiled weights, one CUDA graph per step, GPU-side sampling, FP8 KV and a replay scheme for the DeltaNet state: 105 ms per step today for one stream (M0), ≤ 84 ms targeted with the FP8 projections kept and ≤ 71 ms with them in NVFP4, **1.25–1.5×**; more under batching (lever 5).
2. **A hybrid draft.** DFlash2's 16 draft tokens plus a context-lookup branch of up to 32 tokens in the same verify, because agents copy: 4.95 accepted tokens per step today (M0's replay; a chain, not a tree), ≥ 5.8 targeted (6.2 is the simulated upper bound), **×1.2**.
3. **Short requests stay short.** FP4 prefill and no host work between the HTTP request and the first weight pass: a warm request's fixed cost from 0.68 s to ≤ 0.25 s, about 6% of model time. (M0: on an idle machine the server-side time to first token is already 0.31 s median, so this lever is worth ~3–6% and is folded into lever 4's FP4 prefill.)
4. **New sessions start warm.** DeltaNet state checkpoints at shared-prefix boundaries, so the system prompt and tool schema every session shares are computed once: 40% of all prefill work gone, but only ~2% of session time; this is the interactive-latency lever, the first answer of a new session in ≤ 1 s instead of 9–13 s.
5. **Several agents share one weight read.** Night Shift, SWE-bench, refine's two steps and subagents run 2–4 sequences at once; batching them into one pass gives ≥ 2.5× aggregate throughput over a single stream.
6. **An agent-tuned drafter (stretch).** The drafter fine-tuned on the user's own agent traces, on the Spark, overnight, so acceptance rises toward 8 without any data leaving the machine.

Headline targets, single stream on a replay of recorded agent sessions (every one is a gate in section 14; M0's measurements are in [reports/M0.md](reports/M0.md) and restated below the table):

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
| Uncached tokens, first request of a session | median 14.7k; 143 such requests carry 40% of all uncached tokens | The shared prefix is recomputed for almost every session (section 11) |
| Latency, warm request | 0.68 s + 33.2 ms per output token (least-squares fit) | 30 tokens/s decode plus a fixed cost per request |
| Latency, first request of a session | median 16.5 s (104 output tokens) | 9–13 s before the first token: ~13 s as recorded, two sessions at a time; ~9 s for 14.7k tokens at the controlled 1,700 tokens/s |

Production's own counters agree: since its last start it generated 3.09M tokens in 615,532 verify steps (5.0 tokens per step), and its log of 2,290 single-stream decode intervals (30 September – 1 October, uncontrolled) shows a median of 30.1 tokens/s at an accept length of 4.62–4.87 out of 16 draft tokens, at a median context of 11.7k tokens.

Two consequences shape everything below. First, the right speed benchmark is a replay of these sessions (section 13), not a prose or math suite: the greedy microbenchmarks in Mightling's AGENTS.md (prose 25.5, code 50.3, JSON 87.0 tokens/s at short context) overstate agent throughput, and the gains here are larger on the agent workload than on those microbenchmarks, because the agent runs at long context and copies text. Second, concurrency is normal, not exotic: Night Shift, SWE-bench, refine's study and fix steps, subagents and paired nodes' lanes all run more than one agent.

## 3. Goals and non-goals

The engine runs exactly one model, Qwen3.8-27B, and every design choice may assume that.

Goals:

- Make Mightling's agent faster end to end: per-request latency and session wall time on the replay benchmark of section 13 are the headline metrics.
- Decode at or above 90% of the measured memory-bandwidth bound for a single sequence, then multiply that by speculation.
- Produce the same output as plain decoding: identical tokens under greedy sampling, the same distribution under temperature sampling, via exact speculative acceptance.
- Start a new session from shared-prefix state, and answer a follow-up in a cached conversation with first-token latency of one weight pass plus its new tokens' prefill.
- Batch 2–4 concurrent sequences into one weight pass, so overnight and multi-agent work gets aggregate throughput nearly for free.
- Live inside Mightling's memory budget (section 12): about 60 GB for weights, sessions and scratch, the same as production's `--mem-fraction-static 0.5`, configurable.
- Serve the standard streaming chat API so existing clients work unchanged, including `reasoning_effort`, `enable_thinking` and `preserve_thinking`.
- Support thinking and non-thinking modes, the Qwen3 tool-call format, and image input through the model's own vision encoder.

Non-goals:

- Running any other model, including Qwen3.8-Flash-Next or Qwen3.6-27B. A new checkpoint with different shapes means recompiling.
- Multi-node inference across two Sparks over ConnectX-7. (A second paired Spark is a second engine and a second lane, as Mightling already does.)
- Training or fine-tuning the target model. Fine-tuning or retraining the DFlash2 drafter is in scope.
- New quantization research. We start from the precision map of the checkpoint Mightling serves and only decide which tensors move from FP8 to NVFP4. Two lower-byte formats were considered and are deliberately left out of the targets (section 5.3): 2:4 structured sparsity and ~3-bit weights.
- Video input and contexts beyond the native 262k tokens, in the first release.
- Serving many users at high concurrency. Batching stops at the handful of agents one owner runs.

## 4. Hardware: what the GB10 gives and takes away

The DGX Spark is a compute-rich, bandwidth-poor machine, which is exactly the profile speculative decoding exploits. Its GB10 pairs a Blackwell GPU (compute capability 12.1, 5th-generation tensor cores with native FP4 and FP8) with a 20-core Arm CPU on one coherent 128 GB pool of LPDDR5x behind a 256-bit, 273 GB/s bus ([NVIDIA DGX Spark hardware docs](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)). NVIDIA rates it at 1 PFLOP of FP4 with sparsity, about 500 TFLOPS dense FP4 and 250 TFLOPS dense FP8, with a 24 MB GPU L2 (approximate, from NVIDIA's Hot Chips 2025 talk). The GB10 TDP is 140 W. Two copy engines, a 4 TB NVMe and a 200 Gb/s ConnectX-7 round it out; the ConnectX-7 is out of scope here.

| Quantity | Value | Consequence for this design |
| --- | --- | --- |
| Memory bandwidth | 273 GB/s, shared by CPU and GPU; **262 GB/s measured** for a GPU read stream (M0), and the GPU loses exactly what concurrent CPU streams take | A 17.6 GB weight pass takes 64 ms at peak, 67 ms at the measured 262 GB/s; nothing can beat ~15 tokens/s per weight read |
| Dense FP8 compute | ~250 TFLOPS (derived from 1 PFLOP sparse FP4) | ~900 FLOP per byte read: verifying 32 rows per pass costs compute but no bandwidth |
| Unified memory | 128 GB (121 GB usable per sparkrun) | Mightling gives the model server about half (section 12); there is no separate host RAM to offload to |
| GPU L2 | ~24 MB | The 3.1 MB-per-layer DeltaNet state and all activations stay on chip between kernels |
| Observed generic engines | 38 tokens/s for Llama 3.1 8B q4_K_M in Ollama, i.e. ~190 GB/s effective | Generic runtimes reach 65–75% of the bus; our target is 90% |

The arithmetic that shapes everything else: a single-row decode step does about 2 FLOP per weight, so it runs the tensor cores at well under 1% of peak. A row costs about 49 GFLOP (2 × 24.4B weights), so 64 rows per pass are about 3 TFLOP, 10–15 ms of compute that a 75 ms weight pass largely hides; beyond about 100 rows compute starts to add to the step. Tensor-core tiles below 64 rows also run inefficiently. This design therefore treats 64 rows per pass as the practical ceiling, shared between speculation depth and batch: 16–48 rows for one sequence, 16 per sequence for four.

The bus must be measured, not assumed. Peak is 273 GB/s; sustained GPU streaming on LPDDR5x usually lands lower, and the CPU's own traffic shares the same pins. M0 measures a weight-streaming microbenchmark on the actual box, and every target in this document is restated against that number. The plan below assumes 90% of peak, 245 GB/s, is reachable by a well-written kernel.

One Spark-specific simplification: on this machine "GPU memory" and "host memory" are the same DRAM. There is nothing to offload sessions to, so the engine manages one pool with its own paging and eviction, and never copies state across a PCIe bus that does not exist.

Sources: [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) · [GB10 at Hot Chips 2025](https://www.hc2025.hotchips.org/assets/program/conference/day2/21_nvidia_skende_final.pdf) · [Ollama DGX Spark performance](https://registry.ollama.ai/blog/nvidia-spark-performance) · [llama.cpp on DGX Spark](https://jetsonhacks.com/wp-content/uploads/2025/10/spark-llamacpp-bench.html)

## 5. Model: shapes, the real precision map and the per-step byte budget

### 5.1 Shapes

Qwen3.8-27B is a dense hybrid: 64 layers laid out as 16 repeats of three Gated DeltaNet blocks followed by one gated full-attention block, each with its own SwiGLU FFN, hidden size 5120, intermediate size 17,408, a 248,320-token padded vocabulary with untied embedding and output head, a native 262,144-token context, a vision encoder, and a one-layer MTP module ([model card](https://huggingface.co/Qwen/Qwen3.8-27B); confirmed against the served checkpoint's `config.json`). Only the 16 attention layers keep a KV cache; the 48 DeltaNet layers keep a fixed-size recurrent state instead. Every matrix has one size, which is what the compiler specializes on.

| Component | Count | Shape (in × out) | Params |
| --- | --- | --- | --- |
| DeltaNet in-projection (q, k, v fused) and z | 48 | 5120 × 10,240 and 5120 × 6,144 | 83.9M |
| DeltaNet β/α projection, conv1d (k=4), gated norm | 48 | 5120 × 48 each; 10,240 × 4 | 0.5M |
| DeltaNet out-projection | 48 | 6144 × 5120 | 31.5M |
| Attention q-projection (query + output gate) | 16 | 5120 × 12,288 | 62.9M |
| Attention k- and v-projections | 16 | 5120 × 1024 each | 10.5M |
| Attention out-projection | 16 | 6144 × 5120 | 31.5M |
| FFN gate + up (fused) | 64 | 5120 × 34,816 | 178.3M |
| FFN down | 64 | 17,408 × 5120 | 89.1M |
| LM head | 1 | 5120 × 248,320 | 1.27B |
| Token embedding | 1 | 248,320 × 5120 | 1.27B |
| Vision encoder | 1 | | ~1B |

### 5.2 Bytes per pass: what production reads, what the engine reads

The checkpoint Mightling serves (`RadixArk/Qwen3.8-27B-NVFP4`, 21.9 GB on disk) is mixed precision, and its precision map, read from its `quantization_config` and safetensors headers, is not the one the first draft of this spec assumed: the FFN and the LM head are NVFP4 (W4A4, one FP8 scale per 16 values), while **every DeltaNet and attention projection is FP8** (W8A8). The head is 0.72 GB, not a 2.5 GB BF16 head; the FP8 projections are what make the pass 17.6 GB.

| Tensor group | Precision today | GB per pass today | If moved to NVFP4 |
| --- | --- | --- | --- |
| FFN gate, up, down (64 layers) | NVFP4 | 9.63 | 9.63 |
| DeltaNet in-projections (qkv, z) and out-projection (48 layers) | FP8 | 5.54 | ~3.1 |
| Attention q, k, v, o (16 layers) | FP8 | 1.68 | ~0.95 |
| LM head | NVFP4 | 0.72 | 0.72 |
| β/α projections, norms, conv | BF16 | 0.05 | 0.05 |
| **Target weights per pass** | | **17.6** | **~14.4** |
| Token embedding (row lookups only) | BF16 | 10 KB per token | — |
| MTP module (unused while DFlash2 drafts) | BF16 | 0 | — |

Moving the FP8 projections to NVFP4 saves about 3.2 GB per pass (18%), but they were kept at FP8 by the checkpoint's makers, presumably for quality; the quality gate of section 13 decides, group by group, and the targets are stated both ways.

State per sequence is small. A KV cache costs 16 layers × 2 × 4 heads × 256 dims per token: 64 KB in BF16, which production uses (no `--kv-cache-dtype`), 32 KB in FP8. At the workload's median 25.7k tokens that is 1.65 GB or 0.82 GB read per pass. The DeltaNet recurrent state is 48 layers × 48 heads × 128 × 128 values, 151 MB in FP32 (75 MB in BF16, production's `--mamba-ssm-dtype`), independent of context length. The conv1d window adds 3 MB.

Per-step bytes at 25.7k context, single sequence:

| Item | Production (estimate) | Engine, FP8 projections kept | Engine, NVFP4 projections |
| --- | --- | --- | --- |
| Target weights | 17.6 | 17.6 | 14.4 |
| Drafter layers (NVFP4, 1.55 GB) | 1.55 | 1.55 | 1.55 |
| Drafter head | 0.72 (second pass over the target head) | 0.16 (32k-token FP8 head) | 0.16 |
| DeltaNet state traffic | ~1.3 (an intermediate state per verified token, BF16) | 0.30 (one FP32 read and write) | 0.30 |
| KV read | 1.65 (BF16) | 0.82 (FP8) | 0.82 |
| **Total** | **~22.8 GB** | **20.4 GB** | **17.3 GB** |
| Time at 245 GB/s | 93 ms | 83 ms | 71 ms |

Production measures about 160 ms per step on this workload (4.87 tokens per step at 30 tokens/s). Bytes explain about 93 ms of it; roughly 65 ms are something else: kernels below the bandwidth bound at 17 rows, the separate draft pass, sampling in PyTorch over 17 × 248k logits, attention at 25k context, and host work between graphs. M0's first deliverable is a profile that splits those 65 ms; the engine's step target is 76–90 ms, the bytes above plus about 5 ms of compute the pass does not hide.

### 5.3 Fewer bits per weight: considered, not targeted

Fewer bytes per weight is the only lever that raises the one-token-per-read ceiling itself, so two formats were weighed against the "no new quantization research" non-goal.

- **2:4 structured sparsity.** Blackwell's sparse tensor cores take a compressed operand, so a 2:4 FFN would read roughly 40% fewer bytes. But 2:4 pruning of an LLM without retraining loses several benchmark points, and the published recoveries used continued pretraining on billions of tokens, which a Spark cannot do for a 27B model. Rejected for this engine; revisited only if a 2:4 Qwen3.8-27B checkpoint appears that passes the quality gate as shipped.
- **~3-bit weights for the FFN.** About 22% fewer FFN bytes (~2.1 GB, ~12% of the pass), but there is no 3-bit tensor-core path, so the kernels would dequantize to FP8 in registers, and a 27B model loses measurable quality at 3 bits. Deferred to an experiment after M5 under the same quality gate; not in any target.

Moving the FP8 projections to NVFP4 (5.2) saves more bytes than either, with standard tools and the same gate, so it comes first. Speculation acceptance is also a cheaper multiplier: one more accepted token per step is worth more than either format.

### 5.4 Details the kernels must honour

- Gated attention: 24 query heads and 4 KV heads of dimension 256; the q-projection emits a query and an output gate per head; q and k get a per-head RMSNorm before RoPE; RoPE is partial, 64 of 256 dimensions (`partial_rotary_factor` 0.25), in the interleaved M-RoPE form with sections [11, 11, 10] shared with the vision tokens.
- Gated DeltaNet: 16 key/query heads and 48 value heads of dimension 128, a causal conv1d of width 4 with SiLU on q, k and v, per-head decay α and write strength β, and a gated RMSNorm on the output multiplied by SiLU(z).
- Thinking is the model's default (Mightling's agent produces little: at most a tenth of its output, section 2), with `reasoning_effort` xhigh, medium or low, and `preserve_thinking` keeps earlier thinking blocks in the history by default, so the token prefix of a conversation is stable across turns. Recommended sampling: temperature 1.0, top-p 0.95, top-k 20 in thinking mode; temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 in instruct mode.
- The DFlash2 drafter Mightling serves (`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`, an NVFP4 build of [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)) is a 5-layer block-diffusion model with a 2,048-token sliding window, 1.55 GB on disk. It conditions on the target's hidden states at layers 5, 19, 33, 47 and 61, was trained for blocks of 8, and has a top-K candidate selector (`selector_top_k` 16, rank 256); production runs it with 16 draft tokens (`--speculative-num-draft-tokens 16`). Whether SGLang's DFLASH path arranges those 16 as a tree from the selector's alternatives or as a longer chain is read from its code in M0, and the acceptance difference between the two is measured there. It has no output head of its own and reuses the target's LM head ([W8 card](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8)).

Sources: [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) · [Qwen3.8-27B on DGX Spark, OpenZeka](https://blog.openzeka.com/en/qwen3-8-27b-on-dgx-spark-48-tok-s/) · [DFlash2 drafter W8 card](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8) · the served checkpoints' `config.json`, `quantization_config` and safetensors headers on this machine

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

## 8. Architecture: an offline compiler and a resident runtime

Two programs. `ling-compile` runs once per checkpoint and machine and decides everything that can be decided before the first request; `ling-serve` runs for weeks and makes no decisions at all.

![The compiler decides everything before the first request; the runtime only runs](images/compiler-and-runtime.svg)

*compiler and runtime · 7 stages, 6 component groups*

The compiler's seven stages run top to bottom once and hand the runtime a weight blob, compiled kernels, captured graphs and a memory plan; the runtime's components then serve requests without deciding anything.

Compiler stages, in order:

1. Ingest the Hugging Face safetensors (`Qwen3_5ForConditionalGeneration`) and split them into the text stack, embedding, LM head, vision encoder, MTP module and the DFlash2 drafter.
2. Fix the precision map. Start from the served checkpoint's map (NVFP4 FFN and head, FP8 projections; E2M1 values, one E4M3 scale per 16, one FP32 scale per tensor) and move projection groups to NVFP4 only where the quality gate in [section 13](#13-validation-exactness-quality-and-a-benchmark-that-cannot-flatter) passes, using NVIDIA's Model Optimizer with a calibration set that mixes recorded agent sessions, code, prose, math and thinking traces. The map is recorded in the manifest. The drafter keeps its NVFP4 build; drafter error costs speed, never correctness.
3. Fold what folds: each pre-layer RMSNorm's γ is multiplied into the projection that follows it (the DeltaNet in-projection, the attention q/k/v, the FFN gate and up). The per-head q/k norms and the DeltaNet output norm act after their projections and cannot fold; they become kernel epilogues.
4. Tile: permute every matrix into the fragment order its kernel's warps will load (the SM120 block-scaled MMA layout for NVFP4, the FP8 MMA layout for FP8), interleave the scales with the values, and write one 4 KB-aligned blob plus a manifest of offsets and hashes. Load is a memory map; nothing is converted.
5. Instantiate kernels: one template instance per (matrix, precision, row bucket) with all dimensions as compile-time constants, row buckets {1, 8, 16, 32, 48, 64} for decode, verify and batches, and {256, 512, 1024, 2048} for prefill chunks, compiled for sm_121 only. Tile configurations are autotuned once on the Spark and baked in; there is no runtime heuristic.
6. Plan memory: fixed offsets for weights, the paged KV pool (64-token pages), DeltaNet state slots, the shared-prefix checkpoint store, drafter feature cache, activation scratch, logits and candidate buffers, sized from the configured budget (section 12). Nothing is allocated while serving.
7. Capture CUDA graphs: one verify-plus-draft graph per row bucket and batch size, and one prefill graph per chunk size. Graphs read sequence lengths and page tables from device memory, so a single graph serves every context length.

Runtime components:

- One language: C++20 and CUDA for everything that ships, the HTTP layer included (decided 2026-10-08: the maintainer knows C++ far better than Rust, and the kernels it builds on, CUTLASS, CuTe and FlashInfer, are C++). Python appears only in tests and benchmark scripts, and in Triton or CuTe DSL kernel prototypes that are rewritten in CUDA C++ before they ship.
- A C++/CUDA core with no Python on the hot path, behind a thin asynchronous HTTP layer that speaks the chat-completions and completions APIs with server-sent events, `reasoning_effort`, `enable_thinking`, `preserve_thinking`, and the Qwen3 tool-call format (section 12 lists exactly what Mightling calls).
- A session manager keyed by conversation: paged FP8 KV, one DeltaNet snapshot slot, the conv window, the token history, and the target hidden features the drafter conditions on. Eviction is LRU inside the single memory pool.
- A shared-prefix store: DeltaNet state checkpoints (with their KV pages) at message boundaries, keyed by a hash of the token prefix, so a new session that begins with a known prefix starts from its checkpoint (section 11).
- A scheduler with a single-sequence fast path and a batch path for 2–4 sequences. One step is: verify graph (which also commits the previous step's DeltaNet replay) → accept → draft graph → emit tokens. In the batch path the sequences share the step's weight pass and split its row budget.
- Draft sources: the DFlash2 drafter by default, a CPU context-lookup matcher whose proposal becomes a second branch of the same verify, and the model's MTP module as a fallback if a checkpoint ships without a DFlash2 drafter.
- GPU-side sampling inside the graph: temperature, top-k 20, top-p, min-p, presence and repetition penalties (a 31 KB presence bitmap per session over the 248k vocabulary), exact rejection sampling for speculation, and device-resident RNG. One small device-to-host copy per step carries the accepted tokens.
- The Qwen 248k BPE tokenizer in C++ (byte-level BPE with Qwen's pre-tokenizer pattern), tested token-for-token against the reference tokenizer on a large corpus, with incremental detokenization for streaming, and the Qwen3 tool-call and reasoning parsers in the same process.
- The vision encoder, run only at prefill when a request carries images, producing visual tokens with their M-RoPE positions.
- Telemetry per step: pass time, achieved GB/s, rows verified, tokens accepted by source (tree or lookup), drafter time, host time, batch size, exported as Prometheus metrics with the names Mightling already reads where SGLang has an equivalent.

Build and safety rules:

- **Rebuild time is dominated by CUTLASS/CuTe instantiations, so every kernel instance is its own translation unit,** compiled for `sm_121` only. A change to one kernel recompiles that file and relinks.
- **The HTTP layer, the scheduler and the kernels are separate CMake targets,** so a front-end change never recompiles a kernel.
- **Ninja on all cores, with ccache for both the host compiler and nvcc.** Builds, tests and profiling run on the development GB10, never on the production machine.
- **Toolchain:** C++20, GCC 13 (Ubuntu 24.04's), nvcc from the CUDA toolkit on the Spark. The HTTP/2 and HTTP/1.1 server is **proxygen**, with folly, fizz and wangle (decided 2026-10-08). It is a large dependency tree, so it is built once with its own `getdeps` and cached on the development machine. The engine links it as a prebuilt.
- **Debuggable from day one:** a debug build compiles every kernel with `-G` and line info. Every phase of a step (verify, accept, draft, emit, and the HTTP and scheduler work around them) carries an NVTX range, so a single Nsight Systems timeline explains a step. A runtime switch runs a step without its CUDA graph, so one kernel can be stepped through in cuda-gdb and checked with `compute-sanitizer`.
- **Request parsing (HTTP, JSON, tool calls) uses a mature library and is fuzzed** (libFuzzer) in CI. The test suite also runs under AddressSanitizer and UndefinedBehaviorSanitizer. This is the discipline that stands in for Rust's memory safety at the network edge.

## 9. Kernel plan: five kernels per layer, all shape-specialized

Each of the 64 layers runs as five fused kernels, 333 launches per step inside one CUDA graph, with no host work between them. Every kernel reads its weights (NVFP4 or FP8, per the manifest) in their on-disk tile order and dequantizes in registers inside the multiply; no dequantized copy ever exists. Rows (M) are the in-flight tokens of every sequence in the step: 1 for plain decode, 16–48 for a single sequence's verify, up to 64 for a batch.

Gated DeltaNet layer (×48):

1. `gdn_in`: RMSNorm of the input, recomputed per thread block (one 5120-value sum of squares, cheap), then the fused in-projection to q, k, v, z, β and α (N = 16,480). Output M × 16,480 in BF16.
2. `gdn_scan`: causal conv1d of width 4 over the cached 3-token window plus the M rows, SiLU, then the gated delta rule per head over each sequence's chain or tree from the layer's snapshot state; gated RMSNorm of the output times SiLU(z). Reads and writes one 3.1 MB FP32 state per layer per sequence; compute is trivial (M × 48 heads × 128 × 128 multiply-adds).
3. `gdn_out`: out-projection (6144 → 5120) with the residual add in the epilogue.
4. `ffn_up`: RMSNorm recomputed in the prologue, gate and up walked together (N = 34,816), SiLU(gate) · up in the epilogue; writes the M × 17,408 activation once.
5. `ffn_down`: down-projection (17,408 → 5120) with the residual add.

Gated attention layer (×16): the same FFN pair, and in place of the first three:

1. `attn_in`: RMSNorm, fused q(+gate)/k/v projection (N = 14,336), per-head RMSNorm of q and k, partial M-RoPE on 64 of 256 dimensions, FP8 quantization of k and v and their append to the paged cache.
2. `attn`: decode attention over the paged FP8 cache plus the in-flight rows under a chain or tree mask, per sequence. With 6 query heads per KV head and M rows, each KV head sees 6M query rows, enough to run on tensor cores; the FP8 values are dequantized in registers. The epilogue multiplies by sigmoid(gate). At the workload's 25k-token median this kernel reads 0.82 GB per sequence, so it is split across the KV length (flash-decoding) to keep every SM streaming.
3. `attn_out`: out-projection (6144 → 5120) with the residual add.

Head and speculation:

- `lm_head`: final RMSNorm plus the 5120 × 248,320 projection for M rows, NVFP4 weights; logits are M × 248,320 in BF16 (0.5 MB per row).
- `accept`: argmax per row and prefix comparison for greedy; for sampling, the Qwen sampling chain (temperature, presence penalty, top-k 20, top-p, min-p) applied to the target logits, then exact rejection sampling against the draft probabilities. For a tree it walks the branches and keeps the longest accepted path. Emits the accepted count and tokens per sequence.
- `draft_*`: the DFlash2 drafter's five sliding-window layers over the accepted tokens' target features plus the mask tokens, in NVFP4, then its head over a reduced vocabulary ([section 10](#10-speculative-decoding-more-tokens-per-pass)).

Two Qwen3.8-specific decisions:

- DeltaNet state and speculation. A verify runs the recurrence over M draft tokens, but only the first j are accepted, so the committed state is the one after token j. SGLang's verify keeps intermediate states for the verified tokens (by this spec's estimate up to 16 × 75 MB per step in BF16; M0's profile measures it). We instead keep the snapshot S₀ and the M rows' k, v, α, β (130 KB per layer), and the next step's `gdn_scan` replays the j accepted tokens from S₀ before it reads the new draft, writing the result as the new snapshot. Cost: one 151 MB read and one write per step across all layers, in FP32, about 2% of the pass. Tree branches are handled the same way: each branch is a replay from S₀, trivial compute, with S₀ served from L2.
- The 248k-token LM head is 0.72 GB at NVFP4, 4% of the pass, and SGLang's drafter reads it a second time per step. The drafter gets a separate head over the 32k most frequent tokens (0.16 GB in FP8), chosen from the token frequencies of recorded agent sessions, since a draft token outside that set is just a rejected guess.

What fusion buys here is modest on its own and essential in combination. A layer in a generic engine is 10–14 launches with the host encoding between them, 700–900 per step; at a few microseconds each that is several milliseconds, 5–10% of a 75 ms pass. The fused path removes that, keeps the 17,408-wide FFN activation out of memory, and, most important, makes a 32-row verify cost 3–5 ms of visible compute instead of a second pass of memory traffic. That is what turns Spark's compute surplus into accepted tokens.

## 10. Speculative decoding: more tokens per pass

Speculation is the only multiplier on a bandwidth-bound machine, so this section is where the targets are won or lost. The design keeps DFlash2 as the drafter, because a block drafter fits this bus, runs it more cheaply, and adds a second draft source that fits what agents write.

Why DFlash2 and not EAGLE-3. An autoregressive drafter makes one pass per draft token; on a 273 GB/s bus even a 1 GB drafter costs 4 ms per token, 28 ms for seven, a third of a target pass. DFlash2 drafts the whole block in one pass by denoising mask tokens conditioned on the target's hidden states, so drafting costs one small pass whatever the block size. The community measures its acceptance at ~6.0 tokens per step on math and ~3.3 on long prose, and production measures 4.87 on Mightling's agent workload with 16 draft tokens.

Changes to how the drafter runs (production already uses the NVFP4 build, so its weights are not a lever):

- Give it a reduced-vocabulary head over the 32k most frequent tokens, 0.16 GB in FP8, instead of a second pass over the 0.72 GB target head. Candidate selection runs over that subset; its top-K codebooks stay in BF16 as in the W8 recipe. M3 measures the acceptance cost and falls back to a 64k head if it exceeds 0.2 accepted tokens.
- Keep its features resident: the target hidden states it conditions on are written by the target's own kernels at layers 5, 19, 33, 47 and 61, so the drafter never recomputes anything over the context.
- Capture it in the same CUDA graph as the verify, so a step is one graph launch.

**The hybrid tree: DFlash2 plus context lookup.** Agents copy: 46% of the visible tokens Mightling's agent writes continue a match in its own context (file contents into patches, paths and names into commands, earlier tool calls into new ones). A CPU matcher over the session's tokens (context and output so far) looks up the last 3 tokens and proposes the continuation of their most recent occurrence, up to 32 tokens. Its proposal is verified as a second branch beside the DFlash2 draft in the same pass (16 + up to 32 rows), and the accept kernel keeps whichever branch accepts more. On the recorded sessions, a simulation that draws DFlash2's acceptance from production's measured mean gives 4.66 tokens per step for DFlash2 alone and 6.2 with the lookup branch (5.9 with 16 lookup tokens, 6.3 with 64). That simulation treats the two sources as independent, while in reality both succeed on the same copyable spans, so 6.2 is an upper bound; the target is 5.8 and M2 measures the real number on the replay. The matcher costs microseconds and runs during the previous step.

**Verification shape.** Row buckets of 16, 32, 48 and 64 are compiled. The default for a single sequence is DFlash2's 16 draft tokens, arranged as a tree of block 8 with the top-K selector's alternatives at the first uncertain positions (whether production already does this is M0's question) plus the lookup branch when the matcher has a proposal; for a batch, each sequence gets 16 rows and the lookup branch only when it has a match of 8 or more tokens. M2 measures accepted tokens per millisecond for each shape on the replay and picks per request class.

**An agent-tuned drafter (stretch).** DFlash2 was trained on generic data; Mightling's agent writes tool calls, patches and short status text. The drafter is small (5 layers, ~1.9B parameters) and its training signal is the target's own output, which Mightling already produces every night. A fine-tune on the user's own recorded sessions is feasible on the Spark itself: about 11 GFLOP per training token for the drafter, so 50M tokens are roughly 570 PFLOP, a few hours, plus a few hours of target forward passes to produce the hidden features. Nothing leaves the machine, and the result is specific to how this user's agent works. The fine-tune competes with Night Shift for the same memory and hours, so it runs under the same admission as other night work, on nights with time to spare. The target is acceptance 8 on the replay; it is scheduled after M4 and nothing before it depends on it. A block-16 retrain is the same work with a different block size.

**Exactness.** Under greedy decoding a draft token is accepted only if it equals the target's argmax, so the output is identical to plain decoding; under sampling the accept kernel uses the standard rejection rule with the drafter's probabilities (the lookup branch is a deterministic proposal, probability 1), which preserves the target distribution. The Qwen sampling chain (temperature, top-k 20, top-p, min-p, presence penalty 1.5 in instruct mode) is applied to the target logits before acceptance, so penalties are honoured exactly. The MTP module ships with the model and is kept as a fallback drafter; it is autoregressive and about one layer in size, so it drafts 3–4 tokens at modest cost and is only used if no DFlash2 drafter exists for a checkpoint.

Sources: [DFlash paper, arXiv 2602.06036](https://hyper.ai/ja/papers/2602.06036) · [DFlash2 drafter W8 card](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8) · [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) · the simulation in [`bench/lookup_sim.py`](bench/lookup_sim.py), run on the recorded sessions

## 11. Sessions, shared prefixes, prefill and batches

A follow-up message should cost one pass plus its new tokens, and a new session should not recompute what every session shares. Each conversation stays resident as paged FP8 KV (32 KB per token), one DeltaNet snapshot slot (151 MB), the conv window, the token history and the drafter's feature cache. A session at the workload's 25.7k median is about 1 GB, so a 30 GB session pool holds about 30 of them; eviction is LRU and, because there is no separate host memory on this machine, eviction means freeing, not copying.

**Shared prefixes.** On a pure-attention model a prefix cache shares KV pages; on this hybrid it must also have the DeltaNet state at the exact token where the prefix ends, and a state is only available where one was saved. Production's cache mostly did not reuse the prefix every session shares (the agent's instructions, the tool schema and the skills list): the first request of 143 of the 168 recorded sessions reported 0 cached tokens and recomputed a median of 14.7k tokens, 40% of all prefill work, 9–13 s before the first answer. `ling-serve` saves a DeltaNet checkpoint (151 MB FP32, or 75 MB BF16 once the quality gate allows it) at every message boundary of a prefill and keeps the ones that more than one session reaches, keyed by a hash of the token prefix, in a store of about 2 GB. A new session that starts with a stored prefix copies the checkpoint into its slot and prefills only what follows: its task text, typically 1–2k tokens, ≤ 1 s.

**The follow-up path.** The new message's tokens are appended to the resident state in the same work that produces the first output token. On the workload a warm request brings a median 297 and a mean 513 new tokens (tool output, mostly): at ≥ 3,000 tokens/s that is 0.1–0.2 s of prefill plus one decode pass, ≤ 0.25 s to the first token, against 0.68 s of fixed cost today.

**Prefix stability.** Qwen3.8 keeps earlier thinking blocks in the history by default (`preserve_thinking` on), so a conversation's token prefix does not change between turns and the cache extends cleanly. If a client turns `preserve_thinking` off, the template strips earlier thinking and the prefix changes; the runtime then re-prefills the stripped assistant turn in the background as soon as a turn completes, a few hundred tokens of compute-bound work, before the user's next message arrives. The same check guards tool results and system-prompt edits: the runtime compares the new token prefix with the stored one and re-prefills from the first divergence, starting from the nearest saved checkpoint.

**Prefill.** Long prompts are compute-bound: 48.7 GFLOP per token, so a few hundred tokens are still one weight-bound pass (about 75 ms) and 32k tokens are about 1.6 PFLOP, 11 s at 3,000 tokens/s. Prefill runs in chunks of up to 8,192 tokens through the {256–2048}-row kernel instances, the NVFP4 FFN with FP4 activations as the checkpoint is calibrated, the FP8 projections with FP8 activations; DeltaNet layers use the chunked parallel form of the delta rule (64-token chunks); attention layers use a FlashAttention-style kernel that writes FP8 KV directly. The chunk size keeps activation scratch at a few hundred megabytes and lets a long prefill be interrupted by a running decode step.

**Batches.** Two to four agents at once is how Mightling runs overnight: SWE-bench runs two instances, refine runs a study and a fix, Night Shift and subagents add more, and a paired node is one more lane. On a bandwidth-bound machine those sequences can share one weight pass: the step reads the weights once and each sequence adds only its KV, its DeltaNet state and its rows. The scheduler forms a batch whenever more than one sequence is decoding, gives each 16 draft rows (plus a lookup branch on a long match), and keeps a prefill chunk from stalling decodes by interleaving it between steps. Production today runs two streams at 27.4 tokens/s each; the target is ≥ 175 tokens/s aggregate for four (section 7).

**Images.** The vision encoder runs only at prefill for requests that carry images, in BF16, and its visual tokens take M-RoPE positions; everything downstream treats them as ordinary tokens. Video is out of scope for the first release.

**Long context.** The native 262,144 tokens are supported; the KV read then dominates the pass (8.6 GB in FP8), so decode slows to roughly a third less than the table's rows. A 4-bit KV option for sessions above 64k tokens is the mitigation, measured in M5. YaRN to 1M tokens is possible with the model card's `rope_parameters` override, costs 33 GB of KV per session, and is deferred.

## 12. Mightling integration

`ling-serve` replaces SGLang as the server for this one model and changes nothing else in Mightling.

- **Selected by the registry.** The model's registry entry names the engine (`launch_overrides['engine'] = 'ling'`), the same switch that selects SGLang today; the launch builder emits only what follows the image, and the container comes from the same `docker run` prefix as every engine, so host safety stays engine-independent: the pre-load `check_host_safety()` checks and the PSI watchdog during load (a 20 GB memory-mapped blob is still a load) apply unchanged.
- **Memory.** A configured budget, not "whatever is free": weights and drafter ~19 GB, embedding and vision ~3.5 GB, sessions 30 GB, the shared-prefix store 2 GB, scratch 2 GB, about 57 GB in all, the same footprint as production's `--mem-fraction-static 0.5`, because the rest of the machine runs SWE-bench containers, the code index, the web UI and the desktop app above earlyoom's 5% line.
- **What Mightling calls.** `GET /v1/models` with the served model id and `max_model_len` (the launcher reads both); chat completions with streaming, tools and reasoning output (the `qwen3_coder` tool format and the `qwen3` reasoning split); completions, which `server start`'s NVFP4 canary uses and which caught the FlashInfer sampling bug in production; `GET /health`; `GET /metrics`.
- **The chat template is an input.** Mightling patches the checkpoint's template at launch (`ChatTemplatePatcher`; the original answered HTTP 400 to `high` and `minimal` efforts), so `ling-serve` takes a template file and records its hash.
- **Which model is running is asked of the server.** Mightling's tools ask `/v1/models`, never the config; `ling-serve` must report the same id the registry expects.
- **Idle.** Production uses `--sleep-on-idle`; `ling-serve` idles without spinning, so an idle Spark stays cool.
- **Fallback.** SGLang stays in the registry as the fallback engine for the same checkpoint, one flag away, until M5's release gate.

## 13. Validation: exactness, quality and a benchmark that cannot flatter

Three gates, run on the Spark itself, and a release does not pass unless all three do.

**Exactness.** Over a fixed set of 500 prompts (code, prose, math, edits, tool calls, thinking and instruct modes) plus 500 requests sampled from the replay set, greedy output with speculation on must be token-identical to greedy output with speculation off, for every row bucket, every batch size, and with the lookup branch on and off. The accept kernel is unit-tested against a reference implementation of rejection sampling, and a 10,000-sample distribution test on a few short prompts checks that sampled outputs with speculation match the plain distribution within statistical noise. The DeltaNet replay and the shared-prefix checkpoints are tested by comparing the committed state after a partial accept, and after a checkpoint restore, with the state from a plain run over the same tokens, bit for bit in FP32.

**Quality of the quantized model.** Measured against BF16 Qwen3.8-27B served from a cloud GPU on the same prompts: GPQA Diamond, LiveCodeBench v6 and IFBench subsets of 200 items each, in thinking mode at the recommended sampling settings, plus perplexity on a held-out mix, and Mightling's own SWE-bench sample of 24 instances (resolved count against production's, which is the measure that matters to its users). The gate is a drop of at most 1 point on each benchmark and at most 2% in perplexity. Projection groups are moved from FP8 to NVFP4 one group at a time while the gate holds, attention last. The FP8 KV cache is checked separately at 32k and 128k context with a needle-in-a-haystack set.

**Speed: replaying the agent.** The primary benchmark replays recorded Mightling sessions: for each request, the exact token prefix the server received and the exact output the agent produced, decoded with the output forced (so acceptance is measured against what the agent really wrote, and the run is deterministic), with tool time taken from the recording. It reports per-request latency, decode tokens/s, accepted tokens per step by source, first-token latency for warm and new-session requests, and session wall time, single stream and with 2 and 4 sessions interleaved. The replay set is the 168 SWE-bench sessions of section 2, extended by any sessions the user chooses to contribute; it never leaves the machine. SGLang has no forced-decoding mode, so its comparison columns are measured differently: the same request prefixes, generated naturally with greedy decoding, reporting per-request latency, tokens/s, step time and first-token latency. Its outputs and acceptance therefore differ from the forced replay's, and acceptance is compared only through the engine's own natural-generation run on the same prompts, which is also reported. A secondary 16-task suite modelled on Husky's (function edit, add a JSON field, rename a SQL column, write a function, fix typos, invoice to JSON, CSV to table, repeated transcript, tone rewrite, short email, project plan, meeting notes to to-dos, reply to an email thread, call summary, question over a 20k-token document, and a competition math problem in thinking mode) and Mightling's three greedy microbenchmarks keep the chat-style numbers comparable. Every run records the exact prompt set and the acceptance histogram, so a headline number can always be traced to its task.

**Profiling method.** Nsight Systems traces of a 50-step window give the per-kernel timeline and the host gaps; Nsight Compute on the five layer kernels gives `dram__bytes_read` and `dram__bytes_write`, from which achieved GB/s is computed and compared with the 245 GB/s assumption. A step is accepted as bandwidth-bound only when the weight kernels show at least 90% of the measured peak from M0's streaming microbenchmark. The same traces of production in M0 split its ~65 ms of non-byte step time. (M0 found two corrections: GB10 exposes no `dram__*` counters in Nsight Compute, so DRAM traffic is read from L2 lookup misses, `lts__t_sectors_*_lookup_miss` × 32 B; and Nsight Systems 2026.3.2 records no kernels launched through `cuLaunchKernelEx` or CUDA graph nodes under driver 580, so traces use 2025.3.2, the release that ships with CUDA 13.0.)

- [ ] Exactness suite green for every bucket and batch size
- [ ] Quality gate passed with the final precision map recorded in the manifest
- [ ] Replay run on the engine, production SGLang and tuned SGLang on the same Spark, same firmware
- [ ] Nsight traces archived with the release

## 14. Milestones

Six milestones, each gated by a measurement on the Spark; M1 is the long pole, because everything after it multiplies the pass time it establishes. Rough effort for one engineer with review, labelled as an estimate: about four months end to end, M1 taking a third of it, M0 two weeks.

![M1's single-row pass gates every speculation milestone](images/roadmap.svg)

M1 starts as soon as M0's breakdown shows the engine can be meaningfully faster than tuned SGLang, without a separate review (the maintainer's decision, 2026-10-08). All milestone work runs on the development Spark, never on the production machine.

*roadmap · 6 milestones, 6 gates*

Each milestone ends at a measured gate; the highlighted M1 sets the pass time that M2 and M3 multiply.

M0 produces the numbers every later target is restated against (plus the five measurements of section 16.1): the measured streaming bandwidth; the replay harness and the replay of production; a profile of production's step that splits its ~160 ms; a tuned SGLang and its replay; production's aggregate throughput at 2 and 4 streams; and the BF16 quality reference. If tuning alone closes much of the gap, M0 says so and the targets move up with it. M1 is the single-row fused path and must reach 90% of measured bandwidth before any speculation work starts, so that acceptance is measured against a tight step and not against overhead. M2 and M3 add speculation in two stages, exact first and fast second. M4 turns the engine into Mightling's model server: the endpoints, parsers and template of section 12, shared-prefix checkpoints, FP4 prefill, images and batches of 2–4. M5 hardens it and tries the agent-tuned drafter.

**Status (9 October 2026).** M0 is done ([reports/M0.md](reports/M0.md)). M1 is done ([reports/M1.md](reports/M1.md)): exact speculation with the DFlash2 drafter, 45.5 tokens/s on the replay, 97% of production's decode. Prefill came next, ahead of M2's tree speculation, by the maintainer's choice ([reports/M2-prefill.md](reports/M2-prefill.md)): from M4, the prompt template of section 12 (ling-serve's prompts are now token-identical to production's on 396 replayed requests), FP4 prefill (section 11's numerics: NVFP4 activations for the FFN, FP8 for the projections, on block-scaled tensor cores; 2,640 / 2,580 / 2,110 tokens/s at 1K / 8K / 35K tokens against production's 2,140-2,460 / 1,930 / 1,680) and shared-prefix checkpoints (a new session resumes after the shared system message). The DeltaNet prefill runs the chunked form in 32-token chunks rather than 64, the KV cache stays BF16, and prefill attention is the rows path's tensor-core kernel. M3, the decode step, is done ([reports/M3.md](reports/M3.md)): the step's draft, verify and commit as CUDA graphs, programmatic dependent launch, and an L2 prefetch in the verify attention take the step from 115.5 to 110.5 ms on the replay with every output bit unchanged; production's is 105.5, and M3 §7 lists what closes the rest (the weight stream first). Open: M2's tree speculation, the rest of M4 (paged FP8 KV, images, batches of 2-4) and M5. The API and serving fixes from the issue survey are merged (branch `api-fixes`): the four tool-call parser fixes, stop strings held back on UTF-8 boundaries, context overflow in the shapes the clients recognise, the NaN/inf guard, penalties over output tokens only, and chat streams that end properly after an error (section 18 names each guard). Checked on the GPU after merging M3: the kernel, stream and prefill tests, `--spec-check` and `--prefix-check` with and without `--graphs --pdl`, `--graph-check`, and 23 server cases against ling-serve with `--graphs --pdl`.

**Agreement gate (standing, from 9 October 2026).** It replaces the 98.6% top-1 target on validate_v0.py's six prompts, whose 288 tokens cannot detect a change smaller than about 4.5 points ([reports/M2-prefill.md](reports/M2-prefill.md) section 6). Measured with `bench/agreement/` on 54 prompts (validate_v0.py's six, 24 short chat prompts, 24 replayed agent prompts), teacher-forced on production's greedy 48-token continuation, against production at concurrency 1 with a cold cache; a flip is a position where the engine's top token scores strictly below production's top:

- flip rate not significantly above production's own self-noise, its greedy decode against its own prefill on the same positions: a one-sided Fisher exact test (the hypergeometric upper tail, observed table included) at α = 0.05. With production's 65 flips in 2,490 positions the threshold is 85 flips (3.41%): 85 gives p = 0.058, 86 gives p = 0.049;
- every flip's gap (production's top against the engine's choice) ≤ 1.75 nats;
- top-5 agreement 100%;
- no late concentration of flips by position in the continuation or by prompt length;
- tool-name match on the 24 agent prompts' first answers ≥ production's match with itself;
- exact-argument match judged only against production's match with itself (13 of 24 when measured).

M1 (56 flips, 2.25%, p = 0.82), `44c92c5` (68, 2.73%, p = 0.43) and M2 (66, 2.65%, p = 0.50) all pass the flip line; M2 is as far from production as production is from itself. Every other line passes for M1 and M2. `bench/agreement/analyze.py` reports counts, not verdicts; anything that turns them into pass/fail applies this test, with the threshold recomputed from production's own count whenever the prompt set changes.

**Exactness policy.** The engine's numerics are the exact path: decode and verify on the rows path, prefill as M2 built it. Every non-exact mode (a change of numerics, not of speed) sits behind a switch and is off by default. One may become the default only if:

- mean KL(exact engine ‖ mode) ≤ 0.0008 nats (twice production's own warm-against-cold spread at concurrency 1);
- the KL shows no growth with position at 25K and at 64K tokens of context;
- top-5 agreement with the exact engine is 100%;
- tool calls on the 24 replayed agent prompts are no worse;
- `--spec-check` and `--prefix-check` are bit-exact within the mode;
- a SWE-bench night is no worse.

BF16 prefill intermediates may be built behind a switch now. FP8 KV (§16.10) and the all-NVFP4 projections (§16.2) wait until the exact engine has beaten SGLang in a night.

**Backlog (9 October 2026).** The surveys of 9 October are merged and ranked in [reports/backlog-2026-10-09.md](reports/backlog-2026-10-09.md); only measured wins move from it into this spec.

## 15. Risks and open questions

The two risks that can sink the targets are the bus and the toolchain; the third is that the gap closes from the other side.

| Risk | Effect if it lands | Mitigation | Decided by |
| --- | --- | --- | --- |
| Sustained LPDDR5x bandwidth well below 273 GB/s (CPU traffic shares the pins) | Every tokens/s target scales down with it | Measure in M0; restate targets; keep the CPU idle during decode (no tokenization or copies on the hot path) | M0: **did not land** (262 GB/s, 96%); CPU streams take bandwidth byte for byte, so the CPU-idle rule stands |
| Production's 65 ms of non-byte step time is mostly something the engine cannot remove (attention at long context, for example) | Step target slips toward 100 ms; decode toward 60 tokens/s | M0's profile splits it before M1 starts; the flash-decoding attention kernel is built and measured in M1 | M0: **moot** — the single-stream step is 105 ms, not 160, and its non-GEMM part is ~19 ms (DeltaNet 7.5, attention 6.1 at bandwidth, drafter 8.6 incl. GEMMs, commit and sampling 3.5); the GEMMs themselves run at 78% of the bus |
| Tuned SGLang closes much of the gap first | The engine's margin over the tuned baseline shrinks | Every gate is stated against the tuned number; M0's tuned run is the go/no-go for M1; useful tunings ship in production at once | M0 |
| CUTLASS and the CUDA toolchain lack mature block-scaled NVFP4 and FP8 MMA kernels for sm_121 at small M | M1 slips | Prototype the FFN kernel first, both NVFP4 and FP8, before committing the layout; FP8 is the documented fallback precision | M1 |
| The FP8 projections fail the quality gate at NVFP4 | They stay FP8; step at 25k is ~88 ms instead of ~76 | Targets are already stated both ways; move groups one at a time | M1 |
| The lookup branch gains less than simulated (both sources succeed on the same spans) | Acceptance ~5.3 instead of 5.8; decode ~62 tokens/s | Measured on the replay in M2; the 64-token branch and an agent-tuned drafter are the next levers | M2 |
| The drafter's 32k-token head costs acceptance | Decode falls by a few percent | Measure separately; 64k head as fallback | M3 |
| The DeltaNet replay interacts badly with tree speculation and batches (per-branch replays exceed the L2 budget) | Tree capped at 16 nodes; batches at 2 | Measure state traffic in M2; the chain path is unaffected | M2 |
| Shared-prefix checkpoints miss because Codex's prefix varies per session (dates, paths early in the prompt) | New sessions start slower than 1 s | M4 measures the hit rate on replay; the prefix layout is Mightling's to change if needed | M4 |
| Thermal throttling at the 140 W TDP during long decodes | Pass time drifts up after minutes | Record clocks in telemetry; benchmark runs of 10 minutes, not 10 seconds | M0 |
| The Qwen3.8 chat template or tool-call format changes in a model update | Prefix check triggers re-prefill on every turn | Pin the template hash in the manifest; re-prefill path is already required | M4 |
| An agent-tuned drafter overfits one user's sessions | Acceptance falls on new kinds of work | Hold out recent sessions; keep the shipped drafter as the fallback and pick per request class by measured acceptance | M5 |

Open questions:

- Does the SM120 block-scaled MMA path on sm_121 reach 90% of bandwidth at M = 1 and M = 16, or does the 1-row bucket need a separate dequant-and-FMA kernel over the same tile layout? (M0: the instructions exist and work on sm_121a — `mma.sync` `kind::mxf4nvf4.block_scale` and `kind::f8f6f4`, TMA, clusters of up to 8 with distributed shared memory, programmatic dependent launch; at M = 16 production's CUTLASS NVFP4 GEMMs reach 208–225 GB/s and cuBLASLt's FP8 ones 140–218.)
- ~~Where do production's ~65 ms of non-byte step time go?~~ Answered by M0 ([reports/M0.md](reports/M0.md) section 5): one stream's 105 ms step is 86 ms of GEMMs at 204 GB/s plus ~19 ms of DeltaNet, attention, drafter and sampling; the GPU idles ~1 ms. It moved lever 5 (batching) up and lever 3 (short requests) down.
- What is the first-token latency floor once tokenization and HTTP are included: is 0.25 s for a warm agent request comfortable, and how close to one pass can a short follow-up get?
- How much of the copyable output does the lookup branch capture once the two draft sources overlap, and does a cross-session suffix store (proposals from earlier sessions' outputs, not just this one's context) add to it?
- Is a 4-bit KV cache needed for sessions above 64k tokens, or is FP8 enough for the users this engine serves?

## 16. New directions from the literature: low-level speed-ups

A survey of about 60 papers, mostly 2025–2026 arXiv, on what could make this engine faster than sections 7–11 already plan. It concentrates on kernels, the memory system and numerics. Each direction below states:
- what it is;
- what it is worth here, in bytes or milliseconds per step at the workload's 25.7k-token median;
- whether it is exact;
- where it would land.

Two measurements were made for this section on the served checkpoint (`RadixArk/Qwen3.8-27B-NVFP4`, 9 of 64 layers read in full): the entropy of each weight format, and the scales inside each fused group. They are marked *measured*. Everything else is the papers' numbers, on their hardware.

### 16.1 Five measurements M0 should add

1. **Bytes in flight.** Measure GPU DRAM latency with a pointer chase. Bandwidth × latency is the number of bytes every weight-streaming kernel must keep outstanding. At 273 GB/s and ~1 µs, that is ~270 KB across the chip, ~6 KB per SM on 48 SMs. Pipeline depth (TMA or `cp.async` stages) is sized from it, not tuned by trial.
2. **How many SMs saturate the bus.** Stream with 8, 16, 24, 32 and 48 SMs. If about half of them reach the measured peak, then:
   - wave quantization stops mattering for weight streams (16.8);
   - the remaining SMs can run the non-streaming work (the DeltaNet scan, sampling) at the same time (16.6).
3. **The ISA sm_121 exposes:**
   - warp-level block-scaled `mma.sync` for NVFP4 and FP8;
   - TMA;
   - thread-block clusters with distributed shared memory;
   - programmatic dependent launch.

   sm_100's `tcgen05`/TMEM path is not expected. Every kernel in section 9 assumes warp-level MMA, and results that depend on TMEM (MpFA's attention, CuTile's B200 numbers) do not transfer. CuTile's attention reached 53% of FlashAttention-2 on sm_120.
4. **Clocks, power and temperature** across a 10-minute decode and a long prefill. A short paper on the DGX Spark found that alternating compute-heavy and memory-heavy phases at a finer grain avoids throttling, worth up to 2%.
5. **DFlash2's acceptance histogram,** not only its mean. If many steps accept all 16 tokens (the ceiling bin), the drafter's block length is leaving speed unused (16.12).

### 16.2 An all-NVFP4 checkpoint already exists and is validated (−3.2 GB per pass)

Section 5.3 treats moving the FP8 projections to NVFP4 as work for the quality gate. That work has been published for this exact model.
- **The checkpoint:** "Minima" quantizes all 496 linear layers of Qwen3.8-27B to NVFP4 W4A4, the Gated DeltaNet gates included: 17.5 GiB.
- **Quality:** it matches BF16 within seed noise on MMLU-Pro, GSM8K, AIME'25, GPQA-Diamond, LiveCodeBench and RULER to 64K, with a 5-task average −0.52. It is the fastest at prefill of the recipes compared (the paper's +14–19%). Those recipes include the checkpoint Mightling serves today.
- **Why it works:** NVFP4's 16-value blocks localize the residual stream's outliers. The "fragile" gate projections are the least sensitive, because softplus and sigmoid compress their error. The delta rule overwrites state along the current key, so injected noise stays flat over 32K tokens instead of compounding.

Two consequences for this engine:
- **Fewer bytes did not become speed at batch 1 in today's kernels.** On one RTX PRO 6000 at concurrency 1, Minima decoded 47 tokens/s against 51 for the served recipe, despite reading fewer bytes. That gap belongs to the kernels, and closing it is this engine's job.
- **The plan changes.** The quality gate of section 13 is run on this checkpoint first, instead of re-deriving the recipe. Its result decides the "NVFP4 projections" column of 5.2. The DFlash2 drafter conditions on the target's hidden states, so acceptance is remeasured with this target in M2.

### 16.3 Scales inside fused GEMMs (correctness, applies now)

The same paper found that vLLM and ModelOpt serve per-module-calibrated NVFP4 checkpoints with the DeltaNet projections fused into one GEMM, taking the larger of the two global scales without rescaling the block scales.
- **The damage:** the decay and write gates were computed with mis-scaled weights in all 48 layers, with ratios of 1.82× and 2.75×.
- **Why it went unnoticed:** long-context perplexity looked *better* than BF16, because a broken forget gate makes the state remember everything.

Measured on the served checkpoint:
- **FP8 weight scales differ within fused groups:**
  - the per-tensor FP8 weight scales of `in_proj_qkv` and `in_proj_z` differ by 1.83×, 1.26× and 0.81× in layers 0–2;
  - `q_proj`, `k_proj` and `v_proj` differ too.
- **Activation scales match within each group:** qkv/z, q/k/v and gate/up.
- **NVFP4 global scales match** for the gate/up pair.

So:
- **`gdn_in` and `attn_in`:** their epilogues apply one weight scale per output column range. This is free: one multiply chosen by column.
- **The compiler checks every fused group.** A group of NVFP4 matrices with unequal global scales is harmonized offline the way the paper repairs it: rewrite to a shared global scale and fold the ratio into the E4M3 block scales.
- **The exactness test is per module,** never perplexity: each fused GEMM's column range is compared with the unfused reference matrix.

### 16.4 Lossless compression where the bits are compressible (measured)

Measured empirical entropy, over every value of 9 layers' matrices:

| Data | Stored | Entropy | Compressible | Note |
| --- | --- | --- | --- | --- |
| NVFP4 values (FFN, head) | 4 bits | 3.968 bits | ~1% | not worth coding |
| NVFP4 E4M3 block scales | 8 bits (0.5 bit per weight) | 3.58 bits | 55% | 32 distinct codes cover 99.98% of scales |
| FP8 E4M3 projection weights | 8 bits | 6.51 bits | 19% | exponent field 2.58 of 4 bits; 128 byte values cover 98.6% |

- **Scales:** a 5-bit fixed-length code per tile, with a per-tensor 32-entry table and an escape list for the 0.02%.
  - **Savings:** 3 of every 8 scale bits, i.e. 0.19 of 4.5 bits per weight, which is 4.2% of every NVFP4 byte. That is about **−0.43 GB per pass** with today's precision map, and about −0.6 GB if every projection is NVFP4 (16.2).
  - **Cost and exactness:** decoding is one table lookup per 16 values, and the result is exact.
  - **Layout:** the code sits next to its tile's nibbles (16.8).
- **FP8 weights,** only if the FP8 projections stay:
  - **The bound:** the entropy limit is −19% of 7.2 GB, about −1.3 GB per pass. ANS coding aligned to GEMM tiles gets within 0.1 bit of it.
  - **GPU-friendly coding:** fixed-length formats are the practical option, like ZipServ's tensor-core-aware bitmap encoding, which decompresses straight into registers and compresses BF16 by 30%. Here that means a 7-bit code plus an escape list for the 1.4% outliers, about −10% (−0.7 GB).
  - **The risk:** decoding costs ALU work, which this machine has to spare at ≤ 64 rows. It must not reduce bytes in flight.
  - **Moot if 16.2 lands,** because NVFP4 values do not compress.

### 16.5 Never materialize the logits: fused LM head, penalty and top-k

Qwen's recommended sampling always truncates to the top 20 tokens, after the presence penalty. So the target probabilities, the acceptance test and the residual distribution after a rejection all live inside each row's top 20.
- **The fused epilogue:** the LM-head epilogue applies the presence penalty from the session's 31 KB bitmap held in shared memory, and keeps a per-tile top 20.
- **The merge:** a second stage merges the tiles' candidates and runs temperature, top-p, min-p and the acceptance walk over 20 values per row.
- **Precedents:** Qrita (pivot-based, deterministic top-k and top-p, now vLLM's default sampler) and SonicSampler (one CUDA-graph-compatible kernel for penalties, top-k/p/min-p and speculative verification, with a two-stage top-k).
- **What it removes:** the M × 248,320 BF16 logits are never written or read again (0.5 MB per row, ~50 MB of traffic at 48 rows). More important, it removes the separate sampling kernels that 5.2 counts among production's unexplained ~65 ms.
- **Exactness:** exact under the sampling chain whenever top-k is set. A request without top-k takes the full-vocabulary path, and that path must stay. The completions endpoint's defaults (top-p 1, no top-k) are exactly what `server start`'s canary uses.

### 16.6 Tree verification for the DeltaNet layers

Six papers in 2025–2026 converged on the same idea for hybrids. Verify all draft nodes in parallel without materializing a recurrent state per node, then reconstruct only the accepted state.
- **Bole:** a tree-structured closed form; 3.4–7.7× faster linear-attention tree verification and 82–99× less transient state. It is in SGLang and was evaluated on GB10 with Qwen3.5-27B and recorded agent sessions.
- **TreeWY:** the WY transform of the gated delta rule over a tree; one triangular solve, storing only a small pseudo-value matrix.
- **Weaver:** rollback-free tree verification plus a small adapter that builds trees from a block drafter's marginals; +24.7% over DFlash.
- **LumoTree:** path-parallel verification; on a DGX Spark with **Qwen3.8-27B NVFP4** replaying recorded agent requests:
  - plain decoding: 10.5 tokens/s;
  - five-token MTP: 26.1 tokens/s;
  - LumoTree: 28.1 tokens/s;
  - SGLang EAGLE with 8 draft tokens: 29.7 tokens/s.
- **SpecLA, STree and KVBuffer:** the latter buffers recent keys and values and defers state updates, the same idea as this spec's replay.

Two consequences:
- **Calibration point:** 10.5 tokens/s for plain decoding is ~95 ms per single-row pass, ~190–200 GB/s effective with the KV read. That is generic kernels at ~70% of peak, as section 4 assumed.
- **Design change:** section 9's chain replay (keep S₀, replay the j accepted tokens) is the chain case of these methods. For a tree (DFlash2's alternatives plus the lookup branch), `gdn_scan` should use the TreeWY or Bole closed form, so its cost and transient memory do not grow with the branch count. This resolves the "per-branch replays exceed the L2 budget" risk in section 15.

### 16.7 One persistent kernel per step

Section 9 runs 333 fused kernels inside one CUDA graph. Megakernel work goes further and runs the whole step as one persistent kernel that schedules tiles itself:
- **FlashFormer:** a whole-model kernel for low batch.
- **MPK:** up to 1.7× lower latency than kernel-per-operator serving.
- **Event Tensor:** dynamic shapes.
- **Ada-MK:** the schedule is fixed at compile time, which matches `ling-compile`, where every shape is known.
- **MonoMoE:** a weight-major persistent kernel; readiness flags and warp specialization overlap auxiliary work with the weight stream.

Graphs and fusion already remove most of what those papers win against. What is left here:
- **No bubbles between kernels.** SMs issue the next layer's weight loads while the current layer's reduction finishes, so the bus never idles at a layer boundary (cross-operator software pipelining).
- **Overlap of non-streaming work.** The DeltaNet scan, the attention combine and the top-k merge run on a few SMs while the rest stream the next weights. This works only if 16.1(2) shows a subset of SMs saturates the bus.

Estimate: 1–2% from removed launch and ramp gaps, 2–4% from overlap, so 3–6% in all. It lands after M3, once the kernels are right. The graph path stays as the fallback and the debug path, as the build rules require. A cheaper first step inside the graph is programmatic dependent launch: the next kernel's prologue starts prefetching weights before the previous kernel ends. This needs 16.1(3).

### 16.8 Shape mechanics for 16–64 rows on 48 SMs

- **Swap A and B.** Put weight rows on the MMA's M dimension (16) and tokens on N (granularity 8), as Marlin and MonoMoE do. A 17-row verify then pads to 24 rows, not 32.
- **Divide bytes, not tiles.**
  - Three projections have 5,120 outputs (`gdn_out`, `attn_out`, `ffn_down`): only 40 tiles of 128 for 48 SMs.
  - `ffn_up`'s 34,816 outputs are 272 tiles, 5.67 waves.

  `ling-compile` chooses split-K or Stream-K per matrix so that every SM streams the same number of bytes. ClusterFusion keeps the partial sums on chip with cluster-level reductions through distributed shared memory: 1.61× end to end on H100. That beats global atomics or a second pass, if sm_121 supports clusters (16.1(3)).
- **Scales beside their values.** Store each tile's block scales (or their 5-bit codes, 16.4) contiguous with its nibbles, so one TMA request fetches both. This means fewer DRAM page switches, which cost more on LPDDR5x than on HBM.
- **Keep L2 for state.**
  - Weights are loaded with an evict-first or no-allocate cache hint, so streaming 17 GB per step does not flush the 25 MB L2 of the per-layer DeltaNet state (3.1 MB), the activations and the drafter's feature cache.
  - The current layer's state slot is pinned with an access-policy window.

### 16.9 The Spark's memory system

- **Allocation:** weights, the KV pool and the state pool live in `cudaMalloc` memory, never in system-allocated memory reached through the shared CPU–GPU page table. On Grace Hopper, page size and first-touch placement were what decided the speed of GPU access to system-allocated memory. M0's bus benchmark runs both ways to confirm it on GB10.
- **CPU traffic competes for the same 273 GB/s.** On an Apple unified-memory system, memory-bound decode caused enough bus contention that placement aware of the shared memory gave 1.29× on its own.
  - **The engine's own CPU work** must touch little memory during a step: HTTP, the tokenizer and the lookup matcher (a 4-byte-per-token array, 100 KB at 25.7k tokens). No per-step allocations, and threads pinned.
  - **Mightling's admission** for concurrent work such as indexers and Night Shift should count bandwidth, not only memory.
  - **M0 measures** a decode step with a CPU memory stream running beside it.
- **Power:** the 140 W budget is shared by CPU and GPU. Finer alternation of compute and memory phases kept the Spark out of throttling (16.1(4)). Prefill chunk size and verify width are the knobs.

### 16.10 KV cache

- **Calibrated per-layer FP8 KV scales:** free at serving time, and they recovered 83% of FP8 KV's long-context perplexity penalty on this model. The manifest carries them.
- **Load each KV tile once for every node of the tree:** DeFT's KV-guided grouping (73–99% less KV IO) and FastTree. Section 9's `attn` already batches 6M query rows per KV head; the tree mask must not split those loads per branch.
- **A 4-bit KV option for long sessions** (section 11, M5) now has a kernel recipe: BitDecoding decodes NVFP4 KV on Blackwell tensor cores, up to 8.6× faster than FP16 flash-decoding. MpFA (NVFP4 for QK, FP8 for PV) is the prefill counterpart, though its fastest path assumes TMEM.

### 16.11 Recurrent-state precision: memory, not speed

The engine's state traffic is 0.3 GB per step, under 2%.
- **DAMP:** uniform INT8 or FP8 state degrades reasoning, and INT4 collapses it. DAMP keeps the high-risk key channels in FP16 at 9.9 bits on average and stays close to FP32.
- **STEPQuant:** 6 bits on Qwen3.8-27B.
- **LeapQuant:** 8-bit per-window quantization.

These save ≤ 1% per step but shrink the 151 MB session slot and the shared-prefix checkpoint store 3–5×. They are not exact, sit behind the quality gate, and come after M5.

### 16.12 Drafter-side results, outside the low-level brief

- **Trees from block-diffusion marginals:** DDTree, and DominoTree (+10–15% accepted length on Qwen3-8B).
- **Restoring causality inside the block:** xPress (+30% acceptance) and D-Loop.
- **A longer block when the ceiling bin is high:** DBloom grows the block from 16 to 24 with a short post-training, +0.8 to +1.4 committed tokens.
- **For section 10's agent-tuned drafter:** a block-verification-aware loss (BV loss, +13–21% accepted tokens) instead of cross-entropy.
- **DScale:** variable verify length with fixed-address workspaces, so captured CUDA graphs survive changing prefixes. This engine has the same problem.
- **SuffixDecoding:** a suffix tree over all earlier sessions' outputs, with adaptive speculation length, up to 5.3× on SWE-bench agent traces. It answers section 15's open question about a cross-session store: worth building into the lookup matcher.
- **FR-Spec, VocabTrim and SlimSpec:** precedents for section 10's 32k-token drafter head.

### 16.13 Considered and not pursued

- **Activation sparsity** (TEAL-style, and the byte-crossover analysis): it trims projection bytes for one row. But a 16–48-row verify must read the union of every row's active channels, so the saving disappears exactly where this engine runs.
- **Approximate prefix reuse for hybrids** (Tail-Replay, SuffixReplay: rebuild the recurrent state by replaying a recent suffix, 91–100% of full quality): not exact, so opt-in at most. Message-boundary checkpoints stay the default. Marconi's admission policy (FLOPs saved per byte stored) is adopted for the checkpoint store's eviction.
- **Self-speculation from the DeltaNet subgraph:** acceptance 0.038 on sequential hybrids like Qwen3.5.
- **`tcgen05`, TMEM and 2-SM MMA:** sm_100 features, absent on sm_121 (16.1(3)).

### 16.14 What the survey changes in the numbers

| Engine column of 5.2 (25.7k context) | Bytes per step | At 245 GB/s |
| --- | --- | --- |
| NVFP4 projections (section 5.2) | 17.3 GB | 71 ms |
| + all-NVFP4 checkpoint validated (16.2) | 17.3 GB, now without the quality-gate risk | 71 ms |
| + 5-bit scale codes (16.4) | ~16.7 GB | ~68 ms |
| + fused head and top-k (16.5) | ~16.6 GB | ~68 ms, and production's sampling time is gone |
| + one persistent kernel (16.7) | unchanged | 3–6% less time than the graph path |

None of this moves section 7's targets until M0 has measured the bus (16.1). The table only shows that the step-time target (76–90 ms) has room under it, and that the biggest risk in 5.3 has been answered from outside.

Sources for this section, arXiv, read 8 October 2026:
- **Precision and scales:** [Why Gated DeltaNet survives 4-bit quantization (Minima)](https://arxiv.org/abs/2609.04098).
- **DeltaNet tree verification:**
  - [Bole](https://arxiv.org/abs/2608.01651)
  - [TreeWY](https://arxiv.org/abs/2608.20961)
  - [LumoTree](https://arxiv.org/abs/2609.23900)
  - [Trees from Marginals (Weaver)](https://arxiv.org/abs/2607.06763)
  - [SpecLA](https://arxiv.org/abs/2607.16673)
  - [STree](https://arxiv.org/abs/2505.14969)
  - [KVBuffer](https://arxiv.org/abs/2605.19049)
  - [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
  - [Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers](https://arxiv.org/abs/2605.21325)
- **Persistent kernels:**
  - [FlashFormer](https://arxiv.org/abs/2505.22758)
  - [MPK](https://arxiv.org/abs/2512.22219)
  - [Event Tensor](https://arxiv.org/abs/2604.13327)
  - [Ada-MK](https://arxiv.org/abs/2605.11581)
  - [MonoMoE](https://arxiv.org/abs/2609.04244)
- **GEMM and decode kernels:**
  - [ClusterFusion](https://arxiv.org/abs/2508.18850)
  - [MARLIN](https://arxiv.org/abs/2408.11743)
  - [FlashDecoding++](https://arxiv.org/abs/2311.01282)
  - [Evaluating CUDA Tile on Hopper and Blackwell](https://arxiv.org/abs/2604.23466)
  - [Bridging the gap for microscaling FP4](https://arxiv.org/abs/2509.23202)
- **Lossless compression:**
  - [ZipServ](https://arxiv.org/abs/2603.17435)
  - [DFloat11](https://arxiv.org/abs/2504.11651)
  - [Approaching the Shannon bound](https://arxiv.org/abs/2606.15789)
  - [Lossless compression of low-precision formats](https://arxiv.org/abs/2508.19263)
  - [EntroLLM](https://arxiv.org/abs/2505.02380)
- **Sampling:** [Qrita](https://arxiv.org/abs/2602.01518) · [SonicSampler](https://arxiv.org/abs/2607.20475).
- **Attention and KV:**
  - [FlashInfer](https://arxiv.org/abs/2501.01005)
  - [DeFT](https://arxiv.org/abs/2404.00242)
  - [BitDecoding](https://arxiv.org/abs/2503.18773)
  - [MpFA](https://arxiv.org/abs/2609.33135)
- **Recurrent state:**
  - [DAMP](https://arxiv.org/abs/2608.27513)
  - [STEPQuant](https://arxiv.org/abs/2609.38169)
  - [LeapQuant](https://arxiv.org/abs/2609.38166)
- **Memory system and power:**
  - [Grace Hopper unified memory](https://arxiv.org/abs/2407.07850)
  - [EdgeAgent](https://arxiv.org/abs/2610.03394)
  - [One simple trick for energy-limited local inference](https://arxiv.org/abs/2609.11936)
- **Drafters:**
  - [DDTree](https://arxiv.org/abs/2604.12989)
  - [DominoTree](https://arxiv.org/abs/2607.08642)
  - [xPress](https://arxiv.org/abs/2608.02438)
  - [D-Loop](https://arxiv.org/abs/2610.06011)
  - [DBloom (ceiling-clipped histograms)](https://arxiv.org/abs/2608.30427)
  - [BV loss](https://arxiv.org/abs/2609.34832)
  - [DScale](https://arxiv.org/abs/2609.37532)
  - [SuffixDecoding](https://arxiv.org/abs/2411.04975)
  - [FR-Spec](https://arxiv.org/abs/2502.14856)
  - [VocabTrim](https://arxiv.org/abs/2506.22694)
  - [SlimSpec](https://arxiv.org/abs/2605.10453)
  - [Sequoia](https://arxiv.org/abs/2402.12374)
- **Considered and set aside:**
  - [Where activation and KV sparsity cross](https://arxiv.org/abs/2609.33889)
  - [Tail-Replay](https://arxiv.org/abs/2608.30310)
  - [SuffixReplay](https://arxiv.org/abs/2609.33477)
  - [Marconi](https://arxiv.org/abs/2411.19379)
  - [Component-aware self-speculation](https://arxiv.org/abs/2605.01106)

## 17. Sources

Pages opened for the figures in this document, as of 8 October 2026.

- [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B): architecture, shapes, context length, sampling and thinking settings.
- [Qwen3.8-27B on DGX Spark, 48 tok/s (OpenZeka)](https://blog.openzeka.com/en/qwen3-8-27b-on-dgx-spark-48-tok-s/): the public SGLang + DFlash2 recipe, its flags, memory footprint and concurrency sweep.
- [Qwen3.8-Flash-Next on a single DGX Spark (OpenZeka)](https://blog.openzeka.com/en/qwen3-8-flash-next-what-a-single-dgx-spark-or-rtx-pro-6000-can-do/): the 176B MoE alternative at 28.5 tokens/s.
- [DFlash2 drafter, W8 card](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8): drafter architecture, size, acceptance by workload, quantization result.
- [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) and [DFlash paper, arXiv 2602.06036](https://hyper.ai/ja/papers/2602.06036): block-diffusion drafting.
- [NVIDIA DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html): memory, bandwidth, CPU.
- [NVIDIA GB10 at Hot Chips 2025](https://www.hc2025.hotchips.org/assets/program/conference/day2/21_nvidia_skende_final.pdf): L2 size, coherent unified memory.
- [Ollama DGX Spark performance](https://registry.ollama.ai/blog/nvidia-spark-performance) and [llama.cpp on DGX Spark](https://jetsonhacks.com/wp-content/uploads/2025/10/spark-llamacpp-bench.html): what generic engines achieve on this box.
- [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4): the mixed-precision NVFP4 recipe this design starts from.
- Measured on this machine, 8 October 2026: the served checkpoints' configs and safetensors headers; production SGLang's launch flags, Prometheus counters and decode log; and 168 recorded agent sessions from Mightling's SWE-bench rounds of 3–7 October, analysed with the scripts in [`bench/`](bench/) (`workload.py`, `lookup_sim.py`).

## 18. Known pitfalls

Bug classes that SGLang and vLLM shipped between October 2025 and October 2026 and that ling-engine must not repeat. They come from a survey of their issue trackers ([reports/engine-issues-2026-10-09.md](reports/engine-issues-2026-10-09.md), 98 issues read in depth).

Each row names the guard:
- `tests/server/test_regressions.py` (HTTP, against a running ling-serve);
- `tests/api_tests.cpp` (CPU);
- or an existing `ling-run` check.

**Open** means ling-serve has the bug today: an xfail or `KNOWN(...)` test states the right behaviour.

| Pitfall | What went wrong upstream | ling-engine | Guard |
| --- | --- | --- | --- |
| Hybrid state after a prefix hit | Outputs diverge or turn to `!` after a cache hit, mostly with speculation on; a checkpoint attached to the wrong length; NaN when a prompt ends just past a block boundary ([vllm#60174](https://github.com/vllm-project/vllm/issues/60174), [vllm#43559](https://github.com/vllm-project/vllm/issues/43559), [vllm#55766](https://github.com/vllm-project/vllm/issues/55766), [sglang#38815](https://github.com/sgl-project/sglang/issues/38815), [sglang#41351](https://github.com/sgl-project/sglang/issues/41351)) | Answered by design: row invariance, commit by replay, chunking-independent prefill. Now also tested through the server. | `ling-run --prefix-check`; `test_prompt_length_near_chunk_boundary`; `test_growing_conversation_warm_equals_cold` |
| Identical resend loses its hit | Holding back the last token for its logits dropped the hit to zero ([sglang#22935](https://github.com/sgl-project/sglang/issues/22935)) | Tested | `test_prompt_length_near_chunk_boundary` (resend reports cached tokens) |
| Greedy output not reproducible | Top-k tie order, autotuned kernels and verify numerics changed greedy output between identical requests ([sglang#38009](https://github.com/sgl-project/sglang/issues/38009), [vllm#54928](https://github.com/vllm-project/vllm/issues/54928), [vllm#54521](https://github.com/vllm-project/vllm/issues/54521), [sglang#39597](https://github.com/sgl-project/sglang/issues/39597)) | No autotuning at runtime; ties must break to the lower token id everywhere, the selector's top-16 included. Batches (lever 5) must keep a sequence's output independent of its neighbours ([sglang#36548](https://github.com/sgl-project/sglang/issues/36548)). | `ling-run --spec-check`; `test_greedy_is_deterministic_across_repeats` |
| NaN or inf logits become text | A NaN row sampled as token 0 (`!`) until `max_tokens`, or sanitised into a uniform sample and streamed as a healthy answer; a NaN state reused by later requests ([vllm#53305](https://github.com/vllm-project/vllm/issues/53305), [vllm#55291](https://github.com/vllm-project/vllm/issues/55291), [sglang#33187](https://github.com/sgl-project/sglang/issues/33187)) | Fixed. `Engine::sample` and the speculative accept step refuse non-finite logits, and the request ends with an error counted in `ling:nonfinite_logits_total`. The accept step sees each row's top-K, so a row that is entirely NaN or holds an inf is caught; one with only some NaN logits would need a scan on the device. Each prefill pass is checked before its state is kept, so a non-finite state never becomes the end-of-prompt state or a prefix checkpoint; a decode step's state is never kept. | `sampling_tests.cpp` (injected NaN and inf); `test_nonfinite_logits_are_counted`; `degenerate()` checks in four server tests; `test_untruncated_sampling_is_not_degenerate` (production's completions canary) |
| Truncation skipped under rejection sampling | min_p and logit_bias not applied to verified tokens ([vllm#42744](https://github.com/vllm-project/vllm/issues/42744)) | The accept step uses the full chain | `test_truncation_to_one_token_equals_greedy` |
| Penalties over the wrong history | Penalties used a history shifted by one ([sglang#41124](https://github.com/sgl-project/sglang/issues/41124)). Production counts output tokens only. | Fixed: presence and repetition penalties count the request's output tokens only. | `test_penalties_ignore_prompt_tokens`; `sampling_tests.cpp` (penalties over the output only) |
| Seed ignored on one endpoint | [sglang#15481](https://github.com/sgl-project/sglang/issues/15481) | Read on every endpoint | `test_seed_reproduces` |
| Quoted tool-call markup becomes a call | Examples in code fences, the template's placeholder name, or a literal `<tool_call>` in prose executed as calls; text after the marker lost ([vllm#57541](https://github.com/vllm-project/vllm/issues/57541), [vllm#58147](https://github.com/vllm-project/vllm/issues/58147), [vllm#56658](https://github.com/vllm-project/vllm/issues/56658)) | Fixed. A marker opens a call only outside a code fence, when `<function=` follows it, and for a function the request offered; anything else stays text. Markup in the reasoning is safe. | `api_tests.cpp` (fenced example, unoffered name, literal marker before a real call, literal marker released while streaming) |
| Tool parameter boundaries | `strip()` destroyed indentation ([vllm#48753](https://github.com/vllm-project/vllm/issues/48753)); an unclosed parameter dropped or swallowed its neighbour ([vllm#57699](https://github.com/vllm-project/vllm/issues/57699)); arguments were not JSON ([vllm#55495](https://github.com/vllm-project/vllm/issues/55495)) | One newline trimmed each side; an unclosed last parameter is kept; an unclosed middle parameter ends where the next one starts on a line of its own; arguments are always a JSON dump. | `api_tests.cpp` (unclosed middle and last parameters) |
| A tool call inside the reasoning | A complete call before `</think>` is lost, which stops the agent loop ([vllm#39056](https://github.com/vllm-project/vllm/issues/39056)) | Same behaviour. Promoting such calls collides with quoted markup, so measure the rate on the replay before changing it. | `api_tests.cpp` (quoted markup stays reasoning) |
| History the server cannot render | A prior tool call with invalid JSON arguments made every later request a 400 ([vllm#47761](https://github.com/vllm-project/vllm/issues/47761)) | **Open** on chat completions (the Responses path substitutes `{}`) | `api_tests.cpp` (`KNOWN`); `test_history_with_invalid_tool_arguments_is_accepted` (xfail) |
| Template tokens and empty chunks in the stream | `</think>` or an end-of-turn token in content, reasoning streamed as content, an empty-content chunk that ends an AI-SDK client's turn, a stream without a finish_reason ([vllm#51679](https://github.com/vllm-project/vllm/issues/51679), [vllm#49955](https://github.com/vllm-project/vllm/issues/49955), [sglang#29441](https://github.com/sgl-project/sglang/issues/29441), [vllm#27572](https://github.com/vllm-project/vllm/issues/27572)) | Tested. A chat stream that fails after its headers also ends with a finish chunk (`finish_reason: "error"`) and `[DONE]`. | `test_no_template_tokens_leak`; `test_stream_chunk_shape`; `api_tests.cpp` (stream error tail) |
| Streaming and non-streaming disagree | Text before or after a call dropped in one mode; arguments lost when several tokens arrive per step ([vllm#56263](https://github.com/vllm-project/vllm/issues/56263), [sglang#34214](https://github.com/sgl-project/sglang/issues/34214), [vllm#31501](https://github.com/vllm-project/vllm/issues/31501)) | One incremental parser serves both modes; a speculative step emits up to 16 tokens at once | `test_stream_matches_non_stream` |
| Stop strings in the stream | Stop strings matched inside the reasoning end the turn ([sglang#40529](https://github.com/sgl-project/sglang/issues/40529)); partial UTF-8 in streamed text | **Open:** stops match raw text including the reasoning; decide against production first. Fixed: the stop hold-back ends on a UTF-8 character boundary, and every chunk is dumped with invalid bytes replaced, so a dump error cannot end a stream. | `test_streaming_with_stop_and_multibyte_text`; `api_tests.cpp` (stop scanner fed byte by byte, dump of invalid UTF-8) |
| Responses turns rendered apart | One assistant turn replayed as separate blocks, so agents ended turns early ([sglang#42110](https://github.com/sgl-project/sglang/issues/42110), [vllm#37167](https://github.com/vllm-project/vllm/issues/37167)) | Items merged as production merges them | `bench/render/render_test.py`; `api_tests.cpp`; `test_responses_replayed_turn_renders_like_merged_chat` |
| Effort values refused or aborting | An accepted effort aborted the stream; unsupported values refused ([sglang#40789](https://github.com/sgl-project/sglang/issues/40789), [vllm#52738](https://github.com/vllm-project/vllm/issues/52738), [vllm#53284](https://github.com/vllm-project/vllm/issues/53284)) | **Open** for `none` on chat completions (400) | `test_reasoning_efforts_chat`; `test_responses_stream_event_order`; `test_reasoning_effort_none_chat` (xfail) |
| Usage counts | reasoning_tokens 0 or larger than output_tokens under speculation ([vllm#49711](https://github.com/vllm-project/vllm/issues/49711), [sglang#39826](https://github.com/sgl-project/sglang/issues/39826)) | **Open:** reasoning_tokens is always 0 | `test_responses_reasoning_tokens_counted` (xfail); `test_stream_matches_non_stream` |
| Unbounded request values | Huge top_k, logprobs or n killed the server ([sglang#41482](https://github.com/sgl-project/sglang/issues/41482)) | Bounded. A negative max_tokens still means "unlimited" and should be a 400. | `test_absurd_values_leave_the_server_healthy` |
| Abandoned requests keep running | A disconnected client's request decoded to max_tokens ([sglang#36333](https://github.com/sgl-project/sglang/issues/36333)) | Checked per token. A queued or prefilling request still runs its prefill, which blocks the single worker. | `test_disconnect_frees_the_engine` |
| Drafter silently wrong | A quantized or mislaid drafter accepted ~0 tokens with no error; acceptance decayed over uptime ([sglang#39087](https://github.com/sgl-project/sglang/issues/39087), [sglang#40144](https://github.com/sgl-project/sglang/issues/40144), [sglang#37326](https://github.com/sgl-project/sglang/issues/37326)) | The loader should assert every drafter tensor is consumed; acceptance belongs in `/metrics` and a startup probe | `test_many_short_requests_do_not_degrade_later_ones`; `bench/replay/spec_report.py` |
| SM121 is not SM120 | Feature gates written `== 120` sent GB10 down Hopper paths; first-match kernel selection picked a weight-only NVFP4 kernel, −31% prefill ([sglang#36551](https://github.com/sgl-project/sglang/issues/36551), [vllm#55397](https://github.com/vllm-project/vllm/issues/55397)) | One target, `sm_121a`; no runtime selection | Build rule (section 8) |
| Unified memory is not device memory | Memory fractions did not bound startup on GB10; the host froze instead of an OOM kill ([vllm#56824](https://github.com/vllm-project/vllm/issues/56824), [vllm#46307](https://github.com/vllm-project/vllm/issues/46307), [sglang#36941](https://github.com/sgl-project/sglang/issues/36941)) | Fixed budget (section 12). Prefill scratch must stay bounded by the chunk, never by the prompt. | Mightling's host-safety checks; a 100k-token prefill with peak-memory logging (M4) |

## 19. Backlog from user requests

What users of SGLang and vLLM asked for, and what the four clients send that ling-serve does not handle ([reports/api-compat-checklist.md](reports/api-compat-checklist.md)). Ranked for one owner on one GB10 running 1–4 agents.

1. **Context overflow the clients can recognise.**
   - Chat completions and completions: HTTP 400 before any header, code `context_length_exceeded`, with a message that matches LiteLLM's patterns ("maximum context length is …").
   - The Responses API: `response.failed` with that code, the only shape Codex compacts on.

   Today Codex retries the same prompt five times and OpenHands for two minutes; neither compacts.
2. **Images.** Answer with a placeholder instead of a 400 until the vision encoder (M4) lands, and set `input_modalities: ["text"]` in the launcher's catalog meanwhile. Codex treats a 400 as the end of the turn.
3. **Stream fixes:**
   - hold back stop-string output to a character boundary;
   - end every started stream with a finish chunk and `[DONE]`;
   - validate before sending headers.
4. **Cancellation before work.** Check the cancel flag before dequeuing and between prefill chunks. With one worker, an abandoned 30-second prefill delays every agent.
5. **A NaN guard** (section 18) with a counter in `/metrics`.
6. **Parser fixes** from section 18:
   - fences;
   - unoffered names;
   - prose before a call;
   - unclosed middle parameter;
   - unrenderable history arguments rendered instead of refused.
7. **Small request fixes:**
   - presence and repetition penalties over output tokens only;
   - `reasoning_effort: "none"` on chat completions;
   - `reasoning_tokens` counted;
   - `reasoning` accepted as an alias of `reasoning_content` on input;
   - a negative `max_tokens` refused.
8. **Several resident sessions** (section 11). Two interleaved agents evict each other's history today, and each then re-prefills from its last checkpoint. With several sessions resident, `prompt_cache_key` (Codex sends the thread id) is the natural session key. Upstream's equivalent complaint is lost prefix hits per turn ([vllm#53670](https://github.com/vllm-project/vllm/issues/53670), [vllm#53477](https://github.com/vllm-project/vllm/issues/53477)).
9. **Short requests ahead of long prefills.** Run a short request (autocomplete, a title) between the chunks of a long prefill ([sglang#42530](https://github.com/sgl-project/sglang/issues/42530)).
10. **Speculation measured at long context.** Plain against speculative decode at 8k, 32k, 128k and 200k ([vllm#54691](https://github.com/vllm-project/vllm/issues/54691): DFlash fell to 16 tokens/s against 71 at 185k), with an automatic switch-off if it ever loses.
11. **Reject what is not built.** Answer a 400 for `logprobs`, `n > 1`, `response_format` and `tool_choice: "required"` until each exists. When structured output is built, it must still allow tool calls ([vllm#39929](https://github.com/vllm-project/vllm/issues/39929)), bound whitespace ([vllm#38696](https://github.com/vllm-project/vllm/issues/38696)) and mask every verify row ([vllm#60830](https://github.com/vllm-project/vllm/issues/60830)).
12. **A thinking budget** (force `</think>` after N tokens). Users ask for it on both engines ([sglang#25536](https://github.com/sgl-project/sglang/issues/25536)). OpenHands sends effort `high` on every call.
13. **Custom tools for Codex Code Mode**, if the launcher keeps Code Mode. Today they are dropped exactly as SGLang drops them.
14. **`kernels.cu` made readable, with the SASS unchanged** (the user's request, 2026-10-09; no speed gain, so it is not in the ranked backlog). Each step is accepted only when `cuobjdump -sass` of the touched kernels is byte-identical before and after on second-puffin and `ling-kernel-tests` passes:
   - split the file by subsystem: `device_common.cuh` (`warp_sum`, `block_sum`, the FP8 converters, `silu`), then `gemv.cu` (both GEMVs and the two dequants, which share the FP4 lookup table), `attention.cu`, `gdn.cu`, `elementwise.cu`;
   - name the layouts: a small strided view built inside the kernel from the `__restrict__` parameter (never a struct member, where `restrict` is ignored) replaces the repeated `(pos * Hkv + kh) * D + d` arithmetic;
   - type the attention partial record: `struct AttnPartial { float m, l; float acc[256]; }` in place of `part[0]`, `part[1]`, `part[2 + d]` and the `(D + 2)` stride;
   - one online-softmax merge helper for the three places that rescale `(m, l, acc)`: per key, across warps, across splits;
   - a variadic launch helper that does `<<<>>>` and the launch check in one call, and one `dispatch_rows(M, f)` for the FP4 and FP8 row ladders;
   - argument structs for the wide signatures (`attn_prepare` takes seventeen parameters); a struct passed by value lands in the same constant bank;
   - a header comment per kernel: the thread mapping, each buffer's shape, the invariants the host checks.
   Not touched: the fixed-size register arrays with `#pragma unroll`, the issue-all-loads-first pattern, `__ldg` on `uint4`, `__launch_bounds__`, `fmaf` and `__expf`, the padded shared-memory stride. **Not pursued (the user's decision, 2026-10-09):** merging the FP4 and FP8 GEMV kernels into one body over a codec policy; it changes the loop structure and would need the bandwidth bench to accept, so the two kernels stay separate.
