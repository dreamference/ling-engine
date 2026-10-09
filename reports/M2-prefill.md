# M2 (prefill first): prompts, prefill and prefix checkpoints

October 9, 2026 · measured on the development GB10 (`second-puffin`) against production's SGLang on the same machine (M0's reference container, production flags). Recorded sessions are replayed locally and never leave it; only aggregates are reported.

SPEC §14 puts FP4 prefill and shared-prefix checkpoints in M4 and tree speculation in M2. M1 left decode at 97% of production and a whole request at twice production's time because of v0's prefill, so prefill came first; tree speculation (the spec's M2) is still open, and M1 §6 already measured its lookup half at +4% at best.

## Summary

1. **ling-serve now builds exactly production's prompt.** All 396 replayed requests tested (the main set, one session of every run group, and a thinking-on set) render to the same token ids as production; M1's were a median 129 tokens shorter. Three causes, none of them in the chat template (section 1).
2. **Prefill is 3.6-4.8x faster than M1's and 1.1-1.3x production's:**

   | Prompt | M1 | **M2** | Production (SGLang) |
   | --- | --- | --- | --- |
   | 1,061 tokens | 709 tokens/s | **2,637** | 2,142-2,462 |
   | 8,313 tokens | 714 | **2,584** | 1,934 |
   | 34,778 tokens | 439 | **2,106** | 1,683 |

3. **On the replayed agent sessions** (M0's main set, single stream, two runs) a request takes **4.30 / 4.43 s** on average against M1's 8.14 s and production's 4.18 / 4.25 s. The first token of a warm request comes after **0.40 / 0.41 s** median (M1 1.68 s, production 0.49 / 0.50 s), of a session's first request after **5.2 / 4.4 s** (M1 33.4 s, production 8.5 / 10.1 s). What remains of the gap is decode: the step is M1's (115 ms against production's 105.5) and acceptance swings with sampling (4.80-5.20 tokens per step), so decode runs at 41.7-45.0 tokens/s against 45.5 for M1 and 47 for production (section 5).
4. **New sessions start after the shared system message.** Prefix checkpoints keep the DeltaNet and conv state at the end of the first two messages and at chunk multiples; a new session resumes after the 11.4K-token system message that every session shares, as production's radix cache does (section 4).
5. **Exactness holds.** Greedy speculative output is token-identical to plain decoding (`--spec-check`); a prompt resumed from a checkpoint or from the previous prompt's state reaches bit for bit the state a prefill from scratch does (`--prefix-check`); agreement with production, measured against production's own noise on 2,490 positions, is within that noise apart from a 0.001-nat rise in mean KL that the quantized prefill brings (section 6).

## 1. Prompt rendering

M1 measured ling-serve's prompts a median 129 tokens shorter than production's. To find out why, `bench/render/sglang_render.py` runs production's own Responses-API conversion and chat-template code (SGLang 0.5.19, imported inside production's image, no GPU) on the request bodies a replay sends (`replay.py --dump-bodies`), and `bench/render/render_test.py` compares ling-serve's token ids with those request by request. The reference agrees with the replay's recorded `input_tokens` for all 144 main-set requests, and production's decode-and-retokenize round trip (the model is multimodal) changes nothing.

The differences, in the order they were found:

| Cause | Effect | Fix |
| --- | --- | --- |
| Production validates each tool into its `Tool`/`Function` models and dumps them again, so the template sees `{"type", "function": {"description", "name", "parameters", "strict"}, "defer_loading": null}`. v0 put `name` first and had no `defer_loading`. | 129 tokens per request (21 tools) | `canonical_tools()` in `api.cpp`, for chat completions too |
| A system or developer message with several content parts: production joins each part as its own chunk with blank lines; assistant turns whose contents are part lists merge with no separator. | 3 tokens per request | the Responses conversion mirrors production's merge rules |
| Production's tokenizer class (transformers 5.12's `Qwen2Tokenizer`) replaces the checkpoint's pre-tokenizer pattern with the older Qwen2 one, which has no `\p{M}`: combining marks split from the letters before them. | 14 tokens on a request quoting Devanagari; none on text without combining marks | ling-serve tokenizes prompts the same way by default; `--pretokenizer checkpoint` keeps the checkpoint's pattern |

The third is production's behaviour, not the checkpoint's: the model was published with the `\p{M}` pattern, and production tokenizes Indic and other combining-mark text differently from it. Byte-identical prompts mean ling-serve copies that; whether production should change is a separate question, and the flag makes either choice one switch.

Result: **144/144** main-set requests, **168/168** from one session of each of the 22 run groups, **84/84** with thinking on (`reasoning.effort` medium): token-identical. The sessions are private, so the test skips when the bodies are absent.

## 2. The prefill path

v0's prefill dequantized every weight matrix to BF16 for each 2,048-token chunk and ran cuBLAS: 17.6 GB of weights became ~54 GB of BF16 writes and reads per chunk, and a warm request's few hundred tokens paid it in full (M1: 1.7 s to the first token). Its attention ran on CUDA cores.

### Numerics: production's quantization

SPEC §11 prescribes it and the checkpoint carries what it needs: every NVFP4 and FP8 matrix has an `input_scale`. The prefill path now quantizes activations exactly as production's SGLang does (`modelopt_quant.py`):

- **NVFP4 matrices (the FFN):** activations rounded to BF16, then NVFP4 with one E4M3 scale per 16 values, `scale = E4M3(gscale * amax / 6)`, global scale `1 / input_scale`; `alpha = input_scale * weight_scale_2`.
- **FP8 matrices (the projections):** static per-tensor `x / input_scale`, saturated E4M3; `alpha = input_scale * weight_scale`.

Decode and verify keep M1's rows path (FP16 activations, exact weights), so speculative decoding is untouched. The prefill path's quantizers are tested byte for byte against the recipe written out on the host (`ling-prefill-tests`).

### The GEMM: our own, over the decode kernels' tiled weights

`prefill_gemm.cu` multiplies on block-scaled tensor cores: `mma.sync kind::mxf4nvf4` (m16n8k64, E4M3 block scales) for NVFP4 and `kind::f8f6f4` for FP8, FP32 accumulation. Blocks of 128 tokens x 256 weight rows (128 x 128 when that would leave under two waves), 8 warps, three or four `cp.async` stages of 64 bytes of K per row, shared memory XOR-swizzled so every `ldmatrix` phase is conflict-free.

**Why not CUTLASS.** CUTLASS 4.5.1 builds with the repo's toolchain (GCC 13, `-gencode arch=compute_121a,code=sm_121a`, CUDA 13.0 Update 3) and its SM120 block-scaled NVFP4 example runs correctly here; no toolchain change was needed. It is faster on one shape and slower on the other:

| 2,048 tokens | CUTLASS 4.5.1 (example 79a) | ling-engine |
| --- | --- | --- |
| FFN gate (N 17,408, K 5,120) | 325 TFLOPS | 230 TFLOPS (262 with up fused, below) |
| FFN down (N 5,120, K 17,408) | 262 TFLOPS | 287 TFLOPS |

But CUTLASS reads B K-major with its own blocked scale-factor layout, and the weights exist once, in M1's tiled layout (each 16-row x 128-value block contiguous, scales beside the values), which the decode kernels read and which measured +5-6% end to end. Using CUTLASS means a second copy of the FFN weights in its layout: ~10.7 GB on a ~57 GB budget (SPEC §12), for a gain on the gate that the fused SwiGLU GEMM below mostly recovers. Our kernel reads the tiled blocks directly.

**Fusions.** The FFN's gate and up run as one GEMM whose epilogue applies SwiGLU and writes the down projection's NVFP4 input (2.8 ms at 2,048 tokens, against 4.5 ms for two GEMMs writing FP32 and a separate SwiGLU-and-quantize pass; byte-identical output). RMSNorm, the DeltaNet's gated norm and the attention's sigmoid gate write the next GEMM's quantized input directly; the out-projections add into the residual stream in the GEMM epilogue.

**What did not work** (not kept). A warp-specialized variant (one producer warp issuing every copy, mbarriers between it and eight consumer warps) was 8% faster on FP8 and 3-4x slower on NVFP4: one warp cannot issue a stage's ~1,800 copies while the MMAs run. TMA would issue a stage in a few instructions, but the FP8 tiled layout (decode's) has no 64-byte row runs for a swizzled box, so it needs that layout changed first (section 8).

### Attention

The prefill now runs the rows path's tensor-core attention kernel (BF16 `mma.sync`, fixed 4,096-key ranges, combined afterwards) in slices of 256 queries. Against the CUDA-core tiled kernel it replaced: 0.34 s against 2.15 s for an 8K-token prompt, 4.7 s against 41.6 s at 35K. It reaches ~51 TFLOPS of the GPU's 122 (BF16); it is the largest remaining part at long context (section 5).

### DeltaNet

- **The conv** is a 4-tap filter over raw inputs, so it runs in parallel over tokens and channels (out of place), and the window is written afterwards.
- **The recurrence** runs in the gated delta rule's chunked (WY) form on tensor cores: 32-token chunks, one block per head, `A = beta exp(G_i - G_j) K K^T` inverted by forward substitution, `Delta = U - W S`, the output and the next state as FP16 MMAs with FP32 accumulation, the state itself kept in FP32. 0.76 ms per layer at 2,048 tokens against 3.7 ms for M1's sequential kernel. Two faster sequential kernels were tried first (four lanes per value column, inputs staged in shared memory, norms hoisted); both ran at 3.6 ms, bound by the per-token dependency chain of one SM per head.
- **Its error**, measured in the engine on real prompts against the FP32 sequential kernel: 2-4e-4 relative RMS per layer, flat over chunks. Production runs FLA's chunked kernels in BF16.

### One path per prompt token

Every token of a prompt takes the prefill path whatever the chunk's size, and every kernel on it computes a row the same way whatever the other rows are (the GEMMs, the norms, attention over fixed key ranges, the conv) or, for the recurrence, whatever the split as long as it falls on a multiple of 32. So the state a prompt leaves does not depend on how its prefill was chunked or where it resumed. The prefix checkpoints rely on this.

## 3. Production's warm and cold first tokens

Production's prefix cache (SGLang's radix cache with `--mamba-radix-cache-strategy extra_buffer`) decides both. On the main set, a warm request (the next turn of a session) brings a median ~400 uncached tokens; the rest of its ~25K-token prompt is cached. Its "cold" requests, the first of each session's window, are not cold: production finds 8,192-11,648 of their tokens in the cache, because every session begins with the same instructions and tool schemas. So a warm request's first token is one short prefill plus one pass, and a new session's is the prefill of its own ~14K tokens.

M1 resumed only from the end of the previous prompt: right for a session's next turn, nothing for a new session.

## 4. Prefix checkpoints

ling-serve runs one sequence; its KV cache holds the tokens of the last request. Prefill now keeps states besides the end of the last prompt: the DeltaNet and conv state at the end of the first two messages (the system message, then the first user message) and at multiples of 2,048 tokens up to 32K, in 8 slots of ~157 MB, the earliest positions kept when they run out. A new prompt resumes from the furthest kept state inside its common prefix with the cache. Three details:

- **Positions are multiples of 32,** the DeltaNet chunk: the end-of-prompt state is kept at the last multiple of 32, and up to 31 tokens are prefilled again. Resuming there computes exactly what a prefill from scratch does.
- **Memory.** The eight slots and the end-of-prompt state take 1.4 GB; the prefill path's quantized-activation buffers ~60 MB and the attention partials ~0.1 GB at 64K context. That is ~1.5 GB over M1's footprint, inside SPEC §12's 2 GB shared-prefix store.
- **The drafter's context.** It attends to the last 2,048 positions; resuming at a position inside that window needs its context KV from the window's start, which the engine now tracks (`dkv_lo_`), and otherwise resumes further back.

`ling-run --prefix-check` prefills a sequence of replayed prompts (three turns of one session, two of another, a third session, back to the first), each after the previous one and again from scratch, and compares the states: all identical, bit for bit, along with the next token. The cross-session prompts resume at 11,360-11,488 tokens (the system message's end); the next turn of a session resumes at the previous prompt's end.

## 5. Measurements

### Prefill (single prompt, `ling-run`, reference SGLang stopped)

| Prompt | M1 | M2 | Production |
| --- | --- | --- | --- |
| 1,061 tokens | 709 tokens/s (1.50 s) | **2,637** (0.40 s) | 2,142-2,462 (0.45-0.51 s) |
| 8,313 tokens | 714 (11.7 s) | **2,584** (3.22 s) | 1,934 (4.32 s) |
| 34,778 tokens | 439 (79.3 s) | **2,106** (16.5 s) | 1,683 (20.7 s) |

Two runs each, within 1%. Production's numbers are its `/generate` with a unique prefix and one output token, so they include one decode pass (~0.1 s) and the HTTP round trip; ling-run's include neither. A 1,061-token prefill is still mostly one weight pass, so the short end gains least.

Where a 35K-token prefill's 16.5 s go (Nsight Systems, GPU time, 16.6 s in all):

| Part | Seconds | Share |
| --- | --- | --- |
| Attention (tensor cores) and its combine | 4.97 | 30% |
| NVFP4 FFN: fused gate/up 2.81, down 1.68 | 4.49 | 27% |
| FP8 projection GEMMs | 4.08 | 25% |
| DeltaNet recurrence (chunked) | 0.64 | 4% |
| Conv, norms and gates with their quantization, a/b projections, attention prep | 2.35 | 14% |

Before the chunked recurrence, the fused SwiGLU GEMM and the 128 x 256 tiles (the first M2 build, `44c92c5`), the same prefill took 22.7 s: the sequential recurrence 3.2 s, the FFN 6.2 s, the separate SwiGLU-and-quantize pass 1.4 s.

### Replayed agent sessions

M0's harness and main set, unchanged (12 sessions x 12 requests, single stream, production sampling, output capped at the recorded length); server-side counters; production's numbers are M0's on the same machine.

| | Decode tokens/s | Tokens per step | Step (ms) | Mean request (s) | Warm TTFT, median (s) | First request of a window, TTFT median (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Production (M0, two runs) | 47.1 / 46.8 | 4.96 / 4.94 | 105.5 / 105.4 | 4.18 / 4.25 | 0.49 / 0.50 | 8.50 / 10.07 |
| M1 | 45.5 | 5.21 | 114.5 | 8.14 | 1.68 | 33.4 |
| M2, first build (`44c92c5`) | 42.7 | 4.91 | 115.1 | 4.66 | 0.44 | 8.27 |
| M2, chunked DeltaNet and fused SwiGLU (`7c4aa6b`) | 43.5 | 5.00 | 115.0 | 4.53 | 0.55 | 5.37 |
| **M2, final (`472f87f`), two runs** | **45.0 / 41.7** | **5.20 / 4.80** | **115.4 / 115.3** | **4.30 / 4.43** | **0.40 / 0.41** | **5.20 / 4.40** |

- **Prefill time over the whole run** (the server's time-to-first-token sum): 121 / 120 s, against M1's 701 s and production's 119 / 148 s.
- **The first request of each window** is not a cold start for production either: its cache holds 8,192-11,648 of the request's tokens (section 3). ling-serve resumes after the system message as well, and prefills the remaining median 14,348 tokens in about 5 s.
- **Decode.** The step is unchanged from M1 (114.5 ms; 115.3-115.4 here, with prompts 129 tokens longer), so decode follows acceptance, which sampling moves between runs: 4.80-5.20 over the four M2 runs, against M1's single run at 5.21 and its first build's 4.92. Greedy acceptance on 12 replayed prompts, which sampling cannot move, is the same for M1's and M2's builds: 5.85 and 5.89 tokens per step.
- **Time to first token is the client's first streamed delta.** ling-serve's output parser holds a tool call back until `</tool_call>` arrives, so a response that opens with a tool call shows its first delta at the end; production streams a call's argument deltas as they come. This makes ling-serve's numbers look slower, not faster.
- **The intermediate builds** show what moved: the first M2 build already took most of the gap; the chunked recurrence and the fused SwiGLU GEMM cut the first requests' time to first token (8.3 to 5.4 s), but aligning kept states to 32 tokens added a separate pass for each prompt's tail and so 0.1 s to every warm request (`7c4aa6b`); the final build captures that state inside the last pass.

### Agreement with production

Section 6.

## 6. Agreement with production

M2 was first checked with `bench/validate_v0.py` on M1's six short prompts (288 tokens): 96.2% top-1 against M1's 98.6%. To decide whether that drop matters, the comparison was redone so that every build is judged at the same positions, against production's own noise measured on this machine, on 54 prompts. The method and scripts are in `bench/agreement/`; the prompts stay on the machine that holds the recorded sessions.

### Method

- **validate_v0.py, as before:** the engine decodes 48 tokens greedily; production is then given the prompt and the engine's tokens and reports its top 5 at each position. Agreement at a position means the engine's token is production's top choice on the same prefix, so a flip does not cascade inside the metric. But each build is scored on its own continuation, so two builds are compared at different positions.
- **Teacher-forced on production's continuation (new):** production (`m0-prod8000`, concurrency 1, prefix cache emptied first) decodes 48 tokens greedily; that continuation is then fed, token by token, to production (one prefill, top 20 log-probabilities per position) and to each engine build (`bench/agreement/ling_force.cpp`: the prompt prefilled, the continuation stepped through decode, the full distribution kept). Every build is judged at the same 2,490 positions.
- **Prompts:** 54, the same token ids for every system: validate_v0.py's six, 24 short chat prompts, and 24 replayed agent prompts of 15-37K tokens.
- **A flip** is a position where the engine's top token scores strictly below production's top. Production reports log-probabilities in steps of 1/8 nat, so exact ties occur; 17 of M2's 83 raw differences were ties.
- **Builds:** M1, `44c92c5` (this milestone's prefill with the sequential DeltaNet kernel) and M2 final (`472f87f`, the chunked DeltaNet).

### Production against itself (P2)

The same token inputs sent to production several ways, compared with the reference pass (prefill, concurrency 1, cold cache), at the same 2,490 positions:

| Comparison | Top-1 flips | Largest gap of a flip | Mean KL against the reference |
| --- | --- | --- | --- |
| Prefill, concurrency 1, warm cache | 20 (0.8%) | 0.875 nats | 0.0004 |
| Greedy decode, concurrency 1, against its own prefill | 65 (2.6%) | 1.75 | - |
| Prefill, concurrency 4, warm cache | 80 (3.2%) | 1.625 | 0.0167 |
| Prefill, concurrency 4, cache emptied | 89 (3.6%) | 2.125 | 0.0208 |

Greedy outputs, the same 54 prompts, 48 tokens: a second run at concurrency 1 (the cache now warm) differs from the first on 15 prompts, all of them long; runs at concurrency 4 differ from the first on 36 and 37 prompts, including all 30 short ones. The first answers to the 24 agent prompts (below) match production's own second run exactly on 13 of 24. The forced passes are prefills only, without speculation, so their rise at concurrency 4 comes from batched-prefill numerics; the greedy outputs at concurrency 4 also go through decode and speculation. The reference for everything below is production at concurrency 1 with a cold cache.

### Results (P1-P5)

**P1, the margins of the flips.**

| Build | Flips (six / short / long prompts) | Production's gap at a flip: median, largest | Flips with a gap above 0.875 / 1.75 nats | Long prompts: flips at positions 0-15 / 16-31 / 32-47 |
| --- | --- | --- | --- | --- |
| M1 | 56 (6 / 29 / 21), 2.25% | 0.375, 1.125 | 4 / 0 | 8 / 7 / 6 |
| `44c92c5` | 68 (8 / 32 / 28), 2.73% | 0.375, 1.75 | 4 / 1 | 11 / 10 / 7 |
| M2 | 66 (8 / 34 / 24), 2.65% | 0.375, 1.125 | 4 / 0 | 11 / 6 / 7 |

In 57 of M2's 66 flips the engine's choice is production's second. The engine's own margin at a flip is at most 1.18 nats (M1 1.33). Flips do not gather late in the continuation, nor in the longest prompts: M2 has 2 flips on the 6 long prompts under 20K tokens and 11 on the 11 above 25K, M1 4 and 9 (p = 0.82). Top-5 agreement is 100% for all three builds: the engine's choice is always in production's top 5 (at worst its 4th), and production's choice always in the engine's.

**P3, KL(production ‖ engine)** over production's top 20 tokens, both renormalized on that set:

| Build | Mean | p99 | Max |
| --- | --- | --- | --- |
| M1 | 0.01135 | 0.143 | 0.646 |
| `44c92c5` | 0.01252 | 0.161 | 1.124 |
| M2 | 0.01240 | 0.149 | 0.835 |
| Production, concurrency 1, warm against cold | 0.0004 | - | 0.077 |
| Production, concurrency 4, against concurrency 1 | 0.017-0.021 | - | 0.18-0.27 |

The engines are deterministic: repeated runs of M2 (40 prompts) and M1 (30) were byte-identical. Paired bootstrap over prompts, M2 minus M1: mean +0.00014 to +0.0021 (95%), p99 -0.016 to +0.020. `44c92c5` shows the same rise, so it comes from the quantized prefill (production's own recipe), not from the chunked DeltaNet. M2's ten worst positions fall off smoothly (0.835, 0.647, 0.478, 0.376, 0.335, 0.31, 0.303, 0.215, 0.208, 0.207): six are the first token of a short chat answer (`To` against `#`, `###` against `##`), one is in a 24K-token prompt where both pick the same token.

**P4, function:** the first answer to each of the 24 agent prompts, greedy, up to 400 tokens; the same tool calls with the same arguments, or the same text.

| Pair | Exact | Same tool names |
| --- | --- | --- |
| Production against its own second run | 13 | 22 |
| M1 against production (first / second run) | 7 / 10 | 22 / 22 |
| `44c92c5` against production | 6 / 9 | 21 / 21 |
| M2 against production | 6 / 9 | 23 / 23 |
| M1 against M2 | 10 | 23 |

The argument differences are the content of `exec_command`'s command (`sed -n '544,549p'` against `'544,550p'`, a different test command, another line of a script), mostly the same ones for M1 and M2 on the same prompts.

**P5, statistics** (two-sided Fisher exact test; the smallest mismatch rate a set detects with 80% power at α = 0.05):

| Comparison | Mismatches | p | Detectable |
| --- | --- | --- | --- |
| Six prompts, validate_v0.py, M1 against M2 | 4 / 288 against 11 / 288 | 0.114 | 5.9% against 1.4% |
| Six prompts, production's continuation | 6 / 288 against 8 / 288 | 0.79 | - |
| 48 prompts, validate_v0.py | 69 / 2,180 against 72 / 2,177 | 0.80 | 4.9% against 3.2% |
| 54 prompts, production's continuation | 56 / 2,490 against 66 / 2,490 | 0.41 | 3.6% against 2.2% |

### Verdicts, as the criteria were written before the results

| Criterion | Verdict |
| --- | --- |
| P1: every flip's gap within production's noise, no late concentration | Pass: largest gap 1.125 nats against production's 1.75 (decode against prefill, concurrency 1); flips spread over positions and lengths |
| P2: production's own noise floor | Measured (table above) |
| P3: M2's mean and p99 KL no worse than M1's beyond two runs of the same build; no lone spike | **Fails as written** on the mean: M2 0.0124 against M1 0.0114 (+0.001 nats), and builds are deterministic, so "two runs of the same build" spread nothing and any difference fails. The p99 is within the bootstrap interval and the worst positions show no spike. The difference is 2.5x production's warm-against-cold spread and a fortieth of its concurrency-4 spread, and it is the quantized prefill's (`44c92c5` has it too) |
| P4: M2 matches production at least as often as M1 | **Fails as written** on exact arguments by one prompt (6 against 7, 9 against 10); passes on tool names (23 against 22). The criterion had no noise reference: production matches itself exactly on 13 of 24 |
| P5: statistics | No difference on any set; the six-prompt set cannot detect a change smaller than about 4.5 points |

The maintainer accepted M2 on these results and replaced the six-prompt 98.6% target with a standing gate measured this way (SPEC section 14). Against that gate M2 measures 66 flips in 2,490 positions, 2.65%, one flip above production's own 65 (2.61%); M1 measures 56 (2.25%).

## 7. Exactness

- **Greedy speculative output** equals plain decoding token for token, and the state after a speculative run equals a plain run's bit for bit (`ling-run --spec-check`: code, prose, JSON, arithmetic and an 11,316-token replayed prompt). The rows path is unchanged; the prefill feeds plain and speculative runs identically.
- **Checkpoints and resumption:** `--prefix-check`, section 4.
- **Agreement with production:** section 6.
- **Kernels:** `ling-prefill-tests` checks the quantizers byte for byte, both GEMMs against a double-precision product of the same quantized operands at three shapes (with and without accumulation), the fused SwiGLU GEMM byte for byte against the unfused path, the conv and the chunked recurrence against the rows path (including fast-forgetting heads) and against themselves over a split prompt (bit for bit). `ling-api-tests` covers the Responses conversion.
- **Debug rules:** `ling-prefill-tests` passes under `compute-sanitizer` (memcheck, racecheck, synccheck, initcheck: no errors) and in a Debug build (kernels with `-G`); `ling-api-tests` and the rendering test (168 requests) pass under AddressSanitizer and UndefinedBehaviorSanitizer (`-DLING_SANITIZE=ON`).

## 8. What's left

In order of what they would take off a 25K-token prefill:

1. **Attention (~30% at 35K).** The rows kernel holds 255 registers per thread (Q fragments and a 16 x 256 output per warp), so one block of 6 warps runs per SM, at ~51 of 122 BF16 TFLOPS. Larger query tiles would also halve its key and value traffic from L2.
2. **GEMM efficiency.** NVFP4 runs at 47-60% of the MMA peak, FP8 at ~55%. A TMA producer is the known route; it needs the FP8 tiled layout changed (with decode's FP8 kernel) so a stage is a few swizzled boxes.
3. **FP32 intermediates.** The conv, norms and gates move FP32 tensors (~15% at 35K); production keeps them in BF16.
4. **Decode** is M1's and now the larger part of the remaining gap to production: a 115 ms step against 105.5 (M1 §7 lists what is left there: attention, small kernels and launches, the drafter's head pass).
5. **The spec's M2 proper (tree speculation)** and lever 5 (batching) are untouched.

## How to run

```bash
M=~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/<rev>
D=~/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/<rev>
./build/ling-prefill-tests [--bench]                      # prefill kernels: quantizers, GEMMs, SwiGLU, DeltaNet
./build/ling-run --model $M --prompts-file F --prefix-check  # resumed vs from-scratch prefill, bit for bit
./build/ling-serve --model $M --draft $D --port 8300 --max-context 65536   # [--prefix-checkpoints N] [--pretokenizer checkpoint]
# prompt rendering against production (bodies and references stay on the machine):
python3 bench/replay/replay.py --tools T --tools-index TI --sessions-file S --window 12 --out /dev/null --dump-bodies B
docker run --rm --entrypoint python3 -v ... <production image> bench/render/sglang_render.py MODEL TEMPLATE B R
python3 bench/render/render_test.py --bodies B --reference R --ling-template build/ling-template --tokenizer $M/tokenizer.json
python3 bench/validate_v0.py --ling-run build/ling-run --model $M --prompts-file P   # agreement on long prompts
bench/agreement/run.sh DIR $M SHORT LONG m1=... b44=... m2=...   # the agreement study of section 6 (then analyze.py)
```
