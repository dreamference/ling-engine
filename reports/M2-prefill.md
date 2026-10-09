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
5. **Exactness holds.** Greedy speculative output is token-identical to plain decoding (`--spec-check`); a prompt resumed from a checkpoint or from the previous prompt's state reaches bit for bit the state a prefill from scratch does (`--prefix-check`); top-1 agreement with production is unchanged within noise (section 6).

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

**What did not work** (not kept). A warp-specialized variant (one producer warp issuing every copy, mbarriers between it and eight consumer warps) was 8% faster on FP8 and 3-4x slower on NVFP4: one warp cannot issue a stage's ~1,800 copies while the MMAs run. TMA would issue a stage in a few instructions, but the FP8 tiled layout (decode's) has no 64-byte row runs for a swizzled box, so it needs that layout changed first (section 7).

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

ling-serve runs one sequence; its KV cache holds the tokens of the last request. Prefill now keeps states besides the end of the last prompt: the DeltaNet and conv state at the end of the first two messages (the system message, then the first user message) and at multiples of 2,048 tokens up to 32K, in 8 slots of ~157 MB, the earliest positions kept when they run out. A new prompt resumes from the furthest kept state inside its common prefix with the cache. Two details:

- **Positions are multiples of 32,** the DeltaNet chunk: the end-of-prompt state is kept at the last multiple of 32, and up to 31 tokens are prefilled again. Resuming there computes exactly what a prefill from scratch does.
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
- **The intermediate builds** show what moved: the first M2 build already took most of the gap; the chunked recurrence and the fused SwiGLU GEMM cut the first requests' time to first token (8.3 to 5.4 s), but aligning kept states to 32 tokens added a separate pass for each prompt's tail and so 0.1 s to every warm request (`7c4aa6b`); the final build captures that state inside the last pass.

### Agreement with production

`bench/validate_v0.py`: ling-run decodes greedily; production teacher-forces the same tokens and reports its top 5. M1's six short prompts (all 32 tokens or fewer) say little about prefill, so the script now also takes replayed agent prompts:

| Set | M1 | M2, sequential DeltaNet | M2 (chunked) |
| --- | --- | --- | --- |
| M1's 6 prompts, 288 tokens | 98.6% (top-5 100%) | 98.6% | 96.2% |
| 24 short chat prompts, ~1,025 tokens | 96.2% (-0.188) | 96.5% (-0.175) | 96.0% (-0.185) |
| 24 replayed agent prompts, 11-32K tokens, ~1,150 tokens | 97.4% (-0.153) | 97.4% (-0.153) | 97.3% (-0.156) |

Top-1 agreement, mean log-probability under production in parentheses; top-5 agreement is 100% everywhere. On 2,177 tokens the three builds are indistinguishable. The 6-prompt set moves by 7 tokens between M2's two DeltaNet kernels, inside its noise: M1's own build scores 96.2% on the 24 short prompts. The quantized prefill matches production's own quantization, and the chunked recurrence's error (2-4e-4 per layer) is far below the activation quantization's.

## 6. Exactness

- **Greedy speculative output** equals plain decoding token for token, and the state after a speculative run equals a plain run's bit for bit (`ling-run --spec-check`: code, prose, JSON, arithmetic and an 11,316-token replayed prompt). The rows path is unchanged; the prefill feeds plain and speculative runs identically.
- **Checkpoints and resumption:** `--prefix-check`, section 4.
- **Kernels:** `ling-prefill-tests` checks the quantizers byte for byte, both GEMMs against a double-precision product of the same quantized operands at three shapes (with and without accumulation), the fused SwiGLU GEMM byte for byte against the unfused path, the conv and the chunked recurrence against the rows path (including fast-forgetting heads) and against themselves over a split prompt (bit for bit). `ling-api-tests` covers the Responses conversion.
- **Debug rules:** `ling-prefill-tests` passes under `compute-sanitizer` (memcheck, racecheck, synccheck, initcheck: no errors) and in a Debug build (kernels with `-G`); `ling-api-tests` and the rendering test (168 requests) pass under AddressSanitizer and UndefinedBehaviorSanitizer (`-DLING_SANITIZE=ON`).

## 7. What's left

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
```
