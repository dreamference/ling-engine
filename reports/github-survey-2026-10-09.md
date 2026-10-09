# GitHub survey: what other engines, kernel libraries and GB10 owners have that ling-engine can use

October 9, 2026 · a read-only survey of public GitHub work from the last twelve months: inference engines' GB10/SM12x work, kernel libraries, speculative-decoding code, and the DGX Spark community's measurements. arXiv papers are a separate survey and are left out here, except where a repository is the code release of one.

Nothing here was run on a GPU. Every number below is quoted from the linked source, on the hardware that source names; numbers marked *estimate* are this report's own arithmetic against M0, M1 and M2-prefill. Every link was opened for this report.

## Summary

1. **The largest decode lever found is tree verification for DFlash2** (SGLang [#31069](https://github.com/sgl-project/sglang/pull/31069)): +18% to +25% accept length at a 32-token budget on Qwen3-8B, +8% to +23% tokens/s at concurrency 1 on a B200. The GDN half is now in kernel libraries: parent-indexed verify and path commits for ReplaySSM in FlashInfer ([#6155](https://github.com/flashinfer-ai/flashinfer/pull/6155), [#6257](https://github.com/flashinfer-ai/flashinfer/pull/6257)), one of them tested on a GB10 ([#4151](https://github.com/flashinfer-ai/flashinfer/pull/4151)). The one negative GB10 data point, "+7.40 ms per fork" ([DenseSpark](https://github.com/albond/DenseSpark-Qwen3.8-27B)), comes from forking the recurrent state per branch, which ling-engine's replay design does not do. A 32-row verify on the existing code is the first, cheap gate (*estimate*: +8-15% decode if it passes).
2. **The drafter's pass over the 0.72 GB target head has a measured fix on this exact model and box.** vLLM's context-aware sparse head ([#59740](https://github.com/vllm-project/vllm/pull/59740), [#60226](https://github.com/vllm-project/vllm/pull/60226)): a fixed 32k list plus 16k rows picked per token by a rank-256 copy of the head. On Qwen3.8-27B NVFP4 on one GB10 it gives +5.9% decode at c=1 with acceptance unchanged (MTP k=3). It also shows that SPEC §10's *static* 32k head would cost acceptance: −5.7% on average and −33% on multilingual prompts for a static list.
3. **A possible free 2x on the FP8 prefill GEMMs, pending a 10-minute probe.** On GeForce Blackwell, the plain FP8 MMA with FP32 accumulation (`kind::f8f6f4`, which `prefill_gemm.cu` uses) runs at half the rate of the block-scaled form. The block-scaled form with unit scales gives bit-identical results ([triton #11320](https://github.com/triton-lang/triton/issues/11320), [flashinfer #5963](https://github.com/flashinfer-ai/flashinfer/issues/5963), [#3628](https://github.com/flashinfer-ai/flashinfer/issues/3628)). Nobody has published that probe for GB10: nvfp4bench measured the block-scaled form only with FP4 operands. M2's "FP8 at ~55% of peak" may already rule the gain out (section 3, item 6).
4. **The weight stream has ~8% left.** ling-engine streams target weights at 215 GB/s, and its own pure contiguous read reaches 232-235 (M1 §5). Three independent GB10 projects report the same ~235-240 GB/s practical ceiling, and two of them reach it in real kernels ([DenseSpark](https://github.com/albond/DenseSpark-Qwen3.8-27B): 98.3% of 235.5 GB/s; an EXL3 kernel retune went from 150 to 208 GB/s, [MiaAI-Lab PR #6](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/pull/6), AGPL so ideas only). *Estimate*: −5 to −6 ms of the 115 ms step.
5. **Outside the engine: production's own configuration has a reported correctness problem.** On a stock DGX Spark, SGLang with DFlash2 and the packed-FP4-head NVFP4 target gave **111 wrong greedy answers in 304 at concurrency 8**, against 0 in 100 serial runs ([sglang #35860, comment](https://github.com/sgl-project/sglang/issues/35860); also [#36548](https://github.com/sgl-project/sglang/issues/36548), [#35150](https://github.com/sgl-project/sglang/issues/35150), [#38009](https://github.com/sgl-project/sglang/issues/38009)). This is third-party and not reproduced here, but Mightling runs this configuration with up to 8 running requests (section 6).
6. **Community GB10 numbers agree with M0, not against it.** The comparable figure is the per-pass time: "~100 ms per pass, 102 ms ITL" for production-equivalent SGLang + DFlash2 on chat ([pangoleen](https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2)), against M0's 105.5 ms. The 64-78 tok/s headlines are greedy code generation, which accepts 7-9 tokens per pass. Mixed chat gives ~30 tok/s at ~3 accepted. M1's 45.5 tok/s is sampled agent replay. Section 7 has the table.

## 1. Method

**Queries.** 57 searches with `gh search` (28 for PRs, 14 for issues, 15 for repositories), all restricted to items updated since 9 October 2025:

| Scope | Repositories | Search terms (OR-combined) |
| --- | --- | --- |
| Engines | vllm-project/vllm, sgl-project/sglang, NVIDIA/TensorRT-LLM, ggml-org/llama.cpp | `sm120 sm121 SM12x GB10 "DGX Spark"`; `nvfp4 "block scaled"`; `"gated delta" gdn qwen3-next qwen3.5 deltanet qwen3.8`; `dflash eagle3 eagle-3 mtp`; issues `GB10 "DGX Spark" sm121 sm_121` |
| Kernel libraries | flashinfer-ai/flashinfer, Dao-AILab/flash-attention, NVIDIA/cutlass, deepseek-ai/DeepGEMM, HazyResearch/ThunderKittens, fla-org/flash-linear-attention, triton-lang/triton | the SM12x terms above, plus `gdn mamba`, `fp4 "block scale"`, `speculative top-k sampling tree`, `fa4 blackwell 5090`, `dot_scaled mxfp4 nvfp4` |
| Spec-decoding repositories | all of GitHub | `dflash`; `"block diffusion" speculative`; `eagle3 speculative`; `medusa heads`; `tree verification speculative decoding`; `suffix decoding`; `megakernel inference` |
| DGX Spark community | all of GitHub | `"dgx spark"`; `gb10`; `sm121`; `"dgx spark" benchmark`; `nvfp4 gemm`; `b12x`; `qwen3.8-27b`; plus issue/PR searches for MMA rates and PDL on GB10 |

**Counts.** 2,956 distinct PRs and issues and 504 distinct repositories came back; their titles were triaged against ling-engine's measured leftovers (M1 §5 and §7, M2-prefill §5 and §7). **98 PRs and issues were opened and read** (body, files, state, head or merge commit), and the READMEs of **38 repositories** were read along with design or benchmark documents of four of them (NInfer, sm121-kernels, nvfp4bench, DenseSpark). Licences were read from the LICENSE file at the commit cited: the PR's head or merge commit, or the repository's HEAD for community repositories.

**Rules applied.**
- A number is quoted only with the hardware and the conditions it was measured under.
- "Port" means Apache-2.0, MIT or BSD code that could be adapted with its licence notice kept and a NOTICE entry added. "Port by rewriting" means the source is Python (CuTe DSL or Triton), so what transfers is the algorithm and the tests, and the kernel is rewritten in CUDA C++. "Ideas only" means GPL/AGPL or no licence.
- TensorRT-LLM's and FlashInfer's `trtllm-gen` paths are prebuilt cubins and were not considered.
- Work that the decode job (graph per step, PDL, verify attention) already owns is listed in section 4 for coordination, not ranked.

**Limits.**
- GitHub search ranks by relevance and caps at 100 results per query, so the busiest queries (vLLM and SGLang SM12x) were seen only in part.
- Most SM12x results are measured on RTX 5090 or RTX PRO 6000. Those cards have 1.7-1.8 TB/s and 170-188 SMs against GB10's 273 GB/s and 48 SMs. Their compute-bound prefill results may transfer; their bandwidth-bound decode results mostly do not. Each row says which it is.
- Nothing was reproduced.

## 2. Ranked backlog

Ranked by expected gain on Mightling's agent workload (decode first, because M2 left the remaining request-time gap in decode), weighted by how directly the source applies to GB10 and this model, and by cost. Exactness: **exact** = bit-identical to the current path or to plain decoding; **drafter-only** = changes acceptance, never output; **numerics** = changes the target's arithmetic and needs the quality gate.

| # | Item (sources) | What it does | Applies to GB10 + Qwen3.8-27B at batch 1-2? | Expected gain for us (*estimate*) | Cost | Exactness | Licence, reuse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **DFlash2 tree verification**: SGLang [#31069](https://github.com/sgl-project/sglang/pull/31069) (open); FlashInfer [#6155](https://github.com/flashinfer-ai/flashinfer/pull/6155) parent-indexed GDN verify (open), [#6257](https://github.com/flashinfer-ai/flashinfer/pull/6257) path commit (open), [#4151](https://github.com/flashinfer-ai/flashinfer/pull/4151) tree state retrieval, tested on GB10 (open) | Top-k tree from the block drafter's per-position candidates; one verify; longest accepted root-to-leaf path | Yes. Same drafter family. Verify width is nearly free here (M1 §4). Measured on B200 / Qwen3-8B, not GB10 | Accept length +13-22% at a 32-row budget (source: +18% to +25%); step +3-5 ms. Net **+8-15% decode** | High (M2): tree mask in verify attention, tree recurrence from S₀, tree accept | Exact (greedy by construction; sampled needs tree rejection sampling) | Apache-2.0 (SGLang @f52fd3b, FlashInfer @b87bfa9); port by rewriting (Python/Triton/CuTe DSL) |
| 2 | **Context-aware sparse drafter head**: vLLM [#59740](https://github.com/vllm-project/vllm/pull/59740) (open), [#60226](https://github.com/vllm-project/vllm/pull/60226) on quantized heads (open); motivation [#58578](https://github.com/vllm-project/vllm/issues/58578) | Drafter scores a fixed 32k list plus 16k rows chosen per token by a rank-256 SVD copy of the head; exact logits only for those rows | Yes: #60226 is Qwen3.8-27B NVFP4 W4A4 on one GB10 | Head pass 3.3 ms → ~1.3-1.6 ms per step (bytes: 0.72 GB → ~0.27 GB). **−1.7 to −2.0 ms/step, ~+1.5-2%** | Medium | Drafter-only (source: AL −0.2%, multilingual −2.2%) | Apache-2.0 (vLLM @3ed5c09); port by rewriting |
| 3 | **Weight-stream efficiency, 215 → ~235 GB/s**: [DenseSpark](https://github.com/albond/DenseSpark-Qwen3.8-27B) (98.3% of a measured 235.5 GB/s ceiling); [MiaAI-Lab PR #6](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/pull/6) (150 → 208 GB/s on GB10); TRT-LLM [#18739](https://github.com/NVIDIA/TensorRT-LLM/pull/18739) M=1 NVFP4 GEMV on SM121; llama.cpp [#26843](https://github.com/ggml-org/llama.cpp/pull/26843), [#26705](https://github.com/ggml-org/llama.cpp/pull/26705) (merged, DGX Spark tuning); CUTLASS [#2881](https://github.com/NVIDIA/cutlass/pull/2881) TMA prefetch (merged) | More bytes in flight per SM: deeper cp.async rings, two CTAs per SM, bulk/TMA loads, L2 prefetch | Yes: all GB10 except #2881 (SM100 example) | 17.6 GB at 235 instead of 215 GB/s: **−6.9 ms/step (~6%)**, if the 232-235 contiguous read of M1 §5 is reachable in-kernel | Medium (M1 already tried split-K and 2-3 chunk prefetch with no gain) | Exact (same per-warp K order) | MIT (DenseSpark), Apache-2.0 (TRT-LLM), MIT (llama.cpp), BSD-3 (CUTLASS): ideas; **AGPL (MiaAI-Lab): ideas only** |
| 4 | **ReplaySSM fold into the verify**: FlashInfer [#4081](https://github.com/flashinfer-ai/flashinfer/pull/4081) (merged), [#4974](https://github.com/flashinfer-ai/flashinfer/pull/4974), [#4140](https://github.com/flashinfer-ai/flashinfer/pull/4140) (open); TRT-LLM [#16464](https://github.com/NVIDIA/TensorRT-LLM/pull/16464) (merged), [#19586](https://github.com/NVIDIA/TensorRT-LLM/pull/19586) (open); SGLang [#42388](https://github.com/sgl-project/sglang/pull/42388) (open); NInfer [replayssm-gdn.md](https://github.com/Neroued/ninfer) | Keep accepted (k, u, g) or raw inputs in a ring; the next verify folds them into S₀ in the same launch, and the state is written only every few steps | Yes (Qwen3.5-family GDN shapes; #4081's fold runs "roughly once every 3-5 steps") | Commit phase 3.4 ms (M1 §5): **−1 to −2 ms/step**, plus fewer FP32 state writes (151 MB per write) | Medium | Exact only if the fold rounds exactly as the verify does; #42388 pins `mul.rn`/`fma.rn` for that | Apache-2.0 (FlashInfer @d57bfb1, TRT-LLM @ee241d2, NInfer @81c8ce0); port by rewriting (CuTe DSL), NInfer's C++ is a direct reference |
| 5 | **Lossless NVFP4 scale codes decoded in-kernel**: vLLM [#60646](https://github.com/vllm-project/vllm/pull/60646) (open), CSF codec | Per 16-row slab, one base byte per row plus 4-bit codes and an exception list; byte-identical scales | Yes; SPEC §16.4 planned 5-bit codes for the same reason. vLLM decodes at load, so it saves disk, not bandwidth | NVFP4 scales are 0.5 of 4.5 bits per weight; 4-bit codes save ~0.25 bit, ~5% of 10.35 GB: **−0.5 GB, −2 ms/step** | Medium (encoder offline, decode in `stream_gemm`) | Exact (lossless) | Apache-2.0 (vLLM); format idea, port by rewriting |
| 6 | **MMA-rate probe, then unit-scale block-scaled FP8 MMA in prefill**: [triton #11320](https://github.com/triton-lang/triton/issues/11320) (closed), [#11386](https://github.com/triton-lang/triton/pull/11386) (closed, unmerged), [flashinfer #5963](https://github.com/flashinfer-ai/flashinfer/issues/5963), [#3628](https://github.com/flashinfer-ai/flashinfer/issues/3628) (issues); GB10 peaks from [nvfp4bench](https://github.com/secYOUre/nvfp4bench) | Replace `kind::f8f6f4` with `kind::mxf8f6f4.block_scale.scale_vec::1X ... ue8m0` and scales of 1.0 (0x7f) | Unknown: the halving is measured on GeForce only (5060 Ti, 5070 Ti, 5090). GB10's block-scaled `mxf8f6f4` runs at 256 TFLOPS with FP4 operands (nvfp4bench); FP8 operands and the plain form have not been measured | FP8 projections are 4.08 s of a 35K prefill (M2 §5). Somewhere from **0 to −1.5 s** | Low: probe 10 min, one PTX change | Exact (bit-identity is empirical in both sources; verify byte for byte) | Instruction only; nvfp4bench BSD-3 for the probe |
| 7 | **Prefill attention retile**: flash-attention [#2599](https://github.com/Dao-AILab/flash-attention/pull/2599) (8 MMA warps, TMA; open); [sm121-kernels](https://github.com/blake-snc/sm121-kernels) BF16 FA ~75 TFLOPS on GB10; FlashInfer [#4149](https://github.com/flashinfer-ai/flashinfer/pull/4149) MXFP8 prefill attention for SM120/121 (open) | Cut the 255-register Q/O footprint (8 MMA warps halve the accumulator per thread), TMA K/V, larger query tiles | Partly: #2599 is RTX 5090/PRO 5000; sm121-kernels is GB10 | 4.97 s of a 35K prefill at ~51 TFLOPS; at ~70: **−1.4 s (~8%)** | Medium | Same numerics class; must keep the fixed 4,096-key ranges that make rows invariant | BSD-3 (flash-attention @fcf38ee), Apache-2.0 OR MIT (sm121-kernels @e5a28a3): port by rewriting (CuTe DSL) / port (PTX) |
| 8 | **Prefill GEMM producer**: FlashInfer b12x dense block-scaled GEMM [#4253](https://github.com/flashinfer-ai/flashinfer/pull/4253), [#4305](https://github.com/flashinfer-ai/flashinfer/pull/4305) (merged); [NInfer](https://github.com/Neroued/ninfer) `nvfp4_*_a4_tma.cu`; llama.cpp [#28572](https://github.com/ggml-org/llama.cpp/pull/28572) (+14% pp; closed, unmerged); GB10 CUTLASS ceiling from nvfp4bench | TMA-fed, warp-specialized block-scaled GEMMs with scale tiles loaded once per stage | Yes for the kernels' structure; NInfer's 12,819 tok/s at 7,680 tokens is on an RTX 5090 | NVFP4 at 230-287 TFLOPS now; nvfp4bench measured CUTLASS at ~375 on GB10. FFN 4.49 s → ~3.6 s: **−0.9 s at 35K** | High (TMA needs the FP8 tiled layout changed, M2 §7) | Exact per row if the K order is fixed | Apache-2.0 (FlashInfer @71e1745 and @2febce5; NInfer): port (C++) / port by rewriting (CuTe DSL); MIT (llama.cpp) |
| 9 | **Fused producers for the FP32 intermediates**: SGLang [#34934](https://github.com/sgl-project/sglang/pull/34934) (closed, partly landed), [#32443](https://github.com/sgl-project/sglang/pull/32443) (open); FlashInfer [#5210](https://github.com/flashinfer-ai/flashinfer/pull/5210) (open) | RMSNorm → NVFP4 quant in one pass; gated RMSNorm → FP8 quant; SiLU·mul → NVFP4 | Yes (Qwen3.5-family GDN, ModelOpt NVFP4 + FP8 mix) | 2.35 s of a 35K prefill is conv, norms, gates and quantization (M2 §5): **−0.6 to −1 s** | Low-medium | Numerics only where BF16 replaces FP32; re-run M2's agreement check | Apache-2.0: port by rewriting (Triton) |
| 10 | **Batches of 2-4 with draft width per concurrency**: [NInfer](https://github.com/Neroued/ninfer) execution lanes; draft-width data on GB10 from [darkdatter/gb10-repo](https://github.com/darkdatter/gb10-repo) and [0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark); vLLM [#52559](https://github.com/vllm-project/vllm/pull/52559) graph-aware adaptive K for DFlash (open) | 1-8 resident lanes sharing one weight pass; draft width chosen by concurrency | Yes: the GB10 numbers are this model | Lever 5. SGLang + DFlash2 on GB10 reaches 199 tok/s aggregate at 4 streams on code (pangoleen) vs production's 55 at two streams (M0) | High | Exact if every kernel stays row-invariant across batch rows | Apache-2.0 (NInfer, vLLM): port (C++) / ideas |
| 11 | **Rejection-sampling edge cases** (do now): FlashInfer [#3774](https://github.com/flashinfer-ai/flashinfer/pull/3774) (open), SGLang [#42528](https://github.com/sgl-project/sglang/issues/42528) | Ties at the top-k/top-p boundary over-accept; a coin of 1 − 2⁻²⁴ rejects the only token with mass | Yes: `accept.cpp` uses the same rules | None (correctness) | Hours | Restores exactness at events a 300,000-draw chi-square cannot see | Apache-2.0: ideas (tests) |
| 12 | **GDN recurrence micro-tuning**: llama.cpp [#30087](https://github.com/ggml-org/llama.cpp/pull/30087) (merged), [#22587](https://github.com/ggml-org/llama.cpp/pull/22587); vLLM [#54181](https://github.com/vllm-project/vllm/pull/54181) | Two state columns per warp (16-lane butterflies); row-per-warp; BV=16 for batch ≤ 24 | Partly: RTX 4090 / 5090; #54181 reports bit-identical GB10 runs | Verify recurrence 2.1 ms (M1 §5): −0.3 to −0.5 ms | Low | #30087 changes F32 reduction order; #54181 bit-exact | MIT, Apache-2.0: port |
| 13 | **GDN chunked prefill**: [flash-linear-attention #797](https://github.com/fla-org/flash-linear-attention/pull/797) (similarity transform, 1.08-1.40x on H100); vLLM [#38315](https://github.com/vllm-project/vllm/pull/38315) (fuse kkt + solve_tril) | Ungated KKT and solve, gates applied as diagonal scaling (2·BT exps instead of BT²) | Yes (math is shape-independent) | DeltaNet is 0.64 s of a 35K prefill (4%): −0.1 to −0.2 s | Low | Numerics (different rounding of the gates) | MIT, Apache-2.0: port by rewriting (Triton) |
| 14 | **Drafting within the request's top-k/top-p**: vLLM [#56724](https://github.com/vllm-project/vllm/pull/56724) (open) | Apply the target's top-k/top-p to the draft distribution | Partly: DFlash2's q is already over 16 candidates | Small; source +2.7% at top-k 20 (Qwen3.5-9B, c=64) | Low | Exact (distribution preserved) | Apache-2.0: ideas |
| 15 | **Drafter fine-tunes and larger draft heads**: [0xWhiteMage](https://github.com/0xWhiteMage/qwen3.8-27b-kearuga-sglang-dgx-spark-dflash2) (GB10: +7.9% C1 from a fine-tuned DFlash2, +14.5% with a 64K draft head); [z-lab/dflash](https://github.com/z-lab/dflash) | Drafter quality, for SPEC §10's agent-tuned drafter | Yes (this model, GB10) | Stretch lever 6 | High | Drafter-only | MIT: port (drafter code) |
| 16 | **Small-tile block-scaled CUTLASS**: CUTLASS [#3176](https://github.com/NVIDIA/cutlass/pull/3176), [#3292](https://github.com/NVIDIA/cutlass/pull/3292) (merged), [#3441](https://github.com/NVIDIA/cutlass/pull/3441) FP8 (open) | Tile N = 8, 16, 32, 64 for swap-AB decode GEMMs on SM12x | Yes, but `stream_gemm` already beats production's CUTLASS (M1 §5) | Reference only | — | — | BSD-3 |
| 17 | **CUDA-core NVFP4 for M ≤ 16**: TRT-LLM [#16345](https://github.com/NVIDIA/TensorRT-LLM/pull/16345) (TPOT 37.3 → 33.5 ms on Spark, Qwen3.6-35B-A3B) | Skip tensor cores for small M | Partly (MoE model) | Reference for `stream_gemm` at M = 1 | — | Exact | Apache-2.0: ideas |

**Considered and not recommended:**
- **N-gram lookup in front of MTP** (vLLM [#60615](https://github.com/vllm-project/vllm/pull/60615), +28% on copy probes on GB10). It is measured with an MTP drafter. M1 §6 already measured the DFlash2 equivalent at +4% at best: DFlash2 copies well by itself.
- **Screened FP8 LM head for the target** (vLLM [#60566](https://github.com/vllm-project/vllm/pull/60566), exact, +7.7% on GB10 for a BF16 head). An FP8 screen is larger than our NVFP4 head. Its rigorous per-row error bound is worth remembering for item 2.
- **FP16-accumulate MMAs** (FlashInfer [#6179](https://github.com/flashinfer-ai/flashinfer/pull/6179) for attention, [#6227](https://github.com/flashinfer-ai/flashinfer/pull/6227) for GDN prefill, 1.3-1.5x on RTX 5090). They change numerics, and they pay only if the probe of item 6 shows that GB10 halves FP32-accumulate HMMA.
- **NVFP4 KV cache.** −29% single-stream decode against FP8 KV on GB10 (SGLang [#36797](https://github.com/sgl-project/sglang/issues/36797), Flash-Next).
- **Cluster reductions through distributed shared memory** (SPEC §16.8's ClusterFusion idea). DenseSpark measured it "exact, and 2.73x slower" on GB10.

## 3. The top ten in depth

GPU-time figures are *estimates* for the development GB10. Shapes are this model's (SPEC §5.1).

### 1. DFlash2 tree verification (M2)

**What to take.**
- **SGLang #31069** builds a top-k tree per step from the block drafter's candidates. Its fused Triton kernel runs the whole expand-and-top-k recurrence in one launch; top-k is limited to {4, 8, 16}. It verifies the tree under a mask and accepts the best root-to-leaf path, greedy (`verify_tree_greedy_func`) or sampled (`tree_speculative_sampling_target_only`).
- Its measurements (B200, conc 1, thinking off), chain at budget 16 against tree at budget 32:
  - Qwen3-8B: 1061 → 1228 tok/s on gsm8k (accept 6.42 → 7.86) and 546 → 670 on mt-bench (4.04 → 5.06).
  - Gemma-4-31B: 182 → 208 on mt-bench.
  - Budget 128 helps acceptance but loses speed.
- **For the GDN layers**, FlashInfer #6155 adds a `verify_parents [B, T]` array to the verify kernels, and #6257 an `accept_paths` argument to the ReplaySSM commit. Chain parents are bit-identical to the chain path, so a chain pays nothing. #6257's note matters for us: the commit "replays rows 0..n-1 by loop counter, so the rejected row reaches the checkpoint" unless it is given a path.
- **The negative data point.** DenseSpark (vLLM, GB10) measured "+7.40 ms per fork. Every branch must fork the Gated DeltaNet recurrent state."

**How it maps.**
- The DFlash2 selector already scores 16 × 16 transitions between adjacent positions (M1 §2, `drafter.cpp`). The tree comes from its alternatives at the uncertain positions; M0 §7 showed production verifies only one path.
- `speculate.cpp` / `accept.cpp` need the tree accept.
- The verify-attention kernel needs a tree mask (SPEC §9, §16.10: load each KV tile once for all nodes).
- `gdn_scan` / the commit need parent indices. In ling-engine the verify reads S₀ and does not write, so a branch costs recurrence compute, not a forked state (SPEC §16.6). That is the difference from the +7.40 ms measurement.
- The commit replays the accepted path through the same kernels, which keeps it bit-exact.

**Microbenchmark plan.**
1. **Gate (hours).**
   - *Width.* `ling-run --spec-check` already runs blocks of 32: time a 32-row against a 16-row verify on a 24K-token replayed prompt. Success: step growth ≤ 5 ms. M1 §4 found 12 → 16 rows cost 2.8 ms.
   - *Acceptance, offline.* Log the selector's top-16 candidates and transition scores on M0's main set, and compute offline the greedy acceptance of trees with budgets 24, 32 and 48 against the target's recorded argmax tokens. Success: accepted tokens per step ≥ +10% at budget ≤ 32.
   - GPU time: ~1.5 h, mostly replay.
2. **Kernel.** GDN verify with parent indices at 32 nodes, 48 layers, HV = 48, K = V = 128, FP32 state. Success: ≤ +1 ms over the 32-row chain, and the state after commit bit-identical to a plain run (the existing `--spec-check` state compare). GPU time ~1 h.
3. **End to end.** Replay main set, both runs sampled. Success: decode tokens/s ≥ +8% at equal exactness checks (greedy identity, chi-square). GPU time ~2 h.

### 2. Context-aware sparse drafter head (M3, replaces SPEC §10's static 32k head)

**What to take.**
- **The method** (vLLM #59740): a fixed 32k-token list plus, per draft token, the 16k other rows scoring best under a rank-256 SVD copy of the LM head. Exact logits are computed for those rows only. The target verifies with its full head, so only acceptance can move.
- **Measured on Qwen3.8-Flash-Next + MTP, GB10:**
  - the head alone takes "about 7.4 ms per call" for a 248k × 2560 BF16 head;
  - +13.8% decode at c=1 with the BF16 rows, +23.3% with NVFP4 rows;
  - a static 32k list "loses 5.7% of average AL and 33% on multilingual", while the context-picked rows keep AL within 0.1% (multilingual −4.7%).
- **Measured on our model.** #60226 runs it on an already-quantized head, `nvidia/Qwen3.8-27B-NVFP4` (W4A4 head), on one GB10: AL −0.2% average, −2.2% multilingual; decode +5.9% at c=1 (27.2 tok/s stock) and +3.5% at c=4. The extra memory is 772 MiB.
- **Related.**
  - Quantizing the draft head (#59973, +18% at c=1 on GB10) does not apply: ours is already NVFP4.
  - The DFlash2 reduced-vocabulary bug in #60365 is a trap to avoid: candidates must be mapped back to target ids before the selector's codebooks are indexed.

**How it maps.**
- `drafter.cpp` / `draft_kernels.cu`: the drafter's head pass over the 0.72 GB target head is ~3.3 ms per step (M1 §7 item 3).
- Rows and a scorer built from the NVFP4 head at load:
  - 48k rows × 5120 at 0.5625 bytes ≈ 0.14 GB;
  - a rank-256 scorer at 248,320 × 256 ≈ 0.13 GB in BF16, 0.06 GB in FP8.
- The selector's top-16 is then taken over the 48k rows per position.
- SPEC §10 and §15 ("the drafter's 32k-token head costs acceptance") should be revisited with these numbers: the multilingual loss is the risk the static plan did not price.

**Microbenchmark plan.**
1. **Head pass.** Time the current head pass and the 48k-row + scorer pass for 15 positions; target head N = 248,320, K = 5,120, NVFP4. Success: ≤ 1.5 ms (from 3.3). GPU time ~0.5 h.
2. **Acceptance.** Replay main set, greedy and sampled, plus a thinking-on set and a non-English set. Success: accepted tokens per step within 1% on the main set and within 3% on the non-English set. GPU time ~2 h.

### 3. Weight-stream efficiency (decode GEMMs, 215 → ~235 GB/s)

**What to take.**
- **The practical ceiling.** Three independent GB10 measurements of the streaming ceiling put it near 235-240 GB/s:
  - DenseSpark: "235.5 GB/s streaming ceiling", reached at 98.3% by its INT4 body;
  - pangoleen: "~231 GB/s, and 235 GB/s from the per-pass slope";
  - veloGB10: "~238 GB/s measured sustained bandwidth (idle)".
  
  M0 measured 262.8 GB/s with a tuned pure-read kernel; M1 §5 measured 232-235 for a contiguous read and 215 for `stream_gemm`.
- **The recipe that moved a weight-streaming kernel on GB10** (MiaAI-Lab PR #6, AGPL-3.0, so ideas only):
  - the shipped kernel was "barrier-bound at one 512-thread block per SM", with "only ~3 KB of trellis in flight per SM";
  - rebuilt with "6 cp.async stages, 2 register fragment stages and `__launch_bounds__(512, 2)`", it "reaches 208 GB/s" (from 150), with identical output.
- **Other knobs measured on DGX Spark:**
  - llama.cpp #26843: 8 warps per block for bs = 1 halves the K-loop trip count;
  - llama.cpp #26705: an L2 prefetch that helps on Spark and hurts on the 5090;
  - TRT-LLM #18739: one warp per output row, 16-32 warps per CTA autotuned, 1.32x over Marlin on SM121 for one shape.

**How it maps.**
- `stream_gemm.cu` already keeps each warp's 512-byte loads contiguous with the scales beside them (M1 §5, SPEC §16.8). M0 §1 says ~2.2 KB in flight per SM is enough.
- The candidates are the ones M1 has not tried:
  - two CTAs per SM through launch bounds;
  - a deeper ring of 4-6 stages of 16-byte `cp.async` (M1 tried 2-3 chunks of prefetch);
  - 1-D bulk copies (TMA) of whole tiled blocks, which issue a stage in one instruction;
  - the evict-first / no-allocate hint of SPEC §16.8.

**Microbenchmark plan.**
- Shapes:
  - FFN gate+up, N = 34,816, K = 5,120, NVFP4;
  - FFN down, N = 5,120, K = 17,408, NVFP4;
  - DeltaNet in-projection, N = 16,384, K = 5,120, FP8;
  - out-projection, N = 5,120, K = 6,144, FP8;
  - all at M = 1, 16 and 17.
- Metric: effective weight GB/s against a pure-read reference on the same tiled buffer.
- Success: ≥ 230 GB/s on all four at M = 16, with the `ling-stream-tests` row-invariance `memcmp` unchanged.
- GPU time ~2 h.

### 4. ReplaySSM fold into the verify (commit 3.4 ms)

**What to take.**
- **FlashInfer #4081** (merged): "Save only the small per-token ingredients (update vector u, normalized key k, decay g) in a small ring per request. One kernel launch per layer per decode step then does everything:
  - computes the verify output for the draft tokens,
  - appends the new ingredients to the ring,
  - and — only when a request's live window is full, roughly once every 3-5 steps — folds the window into the single checkpoint state."

  #4974 does the same for T = 1. #4140 keeps raw v instead of u, because u in BF16 "costs ~2x steady-state state accuracy".
- **SGLang #42388** makes the replay bit-exact: the verify kernel computes `fma(k, u, round(h * exp(g)))`, and "a naive replay written as one expression lets the compiler contract a different multiply and drifts by 1-2 ULP". Its fix pins `mul.rn.f32` / `fma.rn.f32` in inline PTX.
- **NInfer's design note** (`docs/maintainer/replayssm-gdn.md`, Apache-2.0) covers the same ground in C++ ("verbatim closed-loop replay").

**How it maps.**
- Today the verify reads S₀ and keeps each layer's inputs, and a separate commit replays the accepted rows through the same kernels (M1 §1, §3). That costs 3.4 ms per step together with the drafter's context KV.
- Folding the replay into the next step's `gdn_scan` removes one pass over the 48 states per step, and writing the state only every few steps removes most of the 151 MB FP32 writes.
- The exactness rule of M1 §3 (same kernels, same per-token arithmetic) is what any fold must keep. #42388 is a list of ways to break it.

**Microbenchmark plan.**
- 48 layers, HV = 48, K = V = 128, FP32 state, 16 draft rows, accepted lengths drawn from M1's histogram.
- Compare the current verify + commit against a fused verify-with-ring that flushes at 16 rows.
- Success: per-step time of the GDN part −1 ms or more, and the state after any sequence of steps bit-identical to plain decoding (`--spec-check`).
- GPU time ~1.5 h.

### 5. Lossless scale codes, decoded in `stream_gemm`

**What to take.**
- **vLLM #60646** decodes CSF-compressed NVFP4 block scales at load: "per 16-row slab, one base byte per row plus 4-bit codes, and a sorted list of 24-bit positions with replacement bytes for out-of-window values", "byte-identical to the uncompressed release".
- **For us**, the format is the evidence that a 4-bit-plus-exceptions scale code works on a Qwen3.8 NVFP4 checkpoint. SPEC §16.4 measured a scale entropy of 3.58 bits and planned 5-bit codes.
- vLLM gains nothing in bandwidth because it decodes at load; ling-engine would decode in registers.

**How it maps.** `ling-compile`'s tiling step (SPEC §8 stage 4) writes the codes beside each tile's nibbles; `stream_gemm` expands them with one table lookup per 16 values (SPEC §16.4).

**Microbenchmark plan.**
1. **Offline, no GPU.** Encode every NVFP4 tensor of the served checkpoint with the 4-bit window scheme and with SPEC's 5-bit table, and report the exception rate per tensor.
2. **Kernel, ~1 h.** NVFP4 `stream_gemm` on the FFN shapes with in-register decode. Success: weight time down by ≥ 4% at M = 16, decoded scales `memcmp`-identical.

### 6. MMA-rate probe, then unit-scale block-scaled FP8 MMA (prefill)

**What to take.**
- **The GeForce measurements.** On GeForce Blackwell, FlashInfer #3628 measured (register-resident, SASS checked) on an RTX 5060 Ti:
  - `HMMA.16816.F32` ~51 TFLOP/s against `HMMA.16816.F16` ~103;
  - `QMMA.16832.F32.E4M3.E4M3` ~102 against `QMMA.SF.16832.F32.E4M3.E4M3.E8` ~202.
- **Bit-identical, twice the rate.** Triton #11386 (closed, unmerged) rewrote fp8 `tt.dot` to the block-scaled form with unit scales: "bit-identical results at up to twice the throughput", 1.41-1.46x on an RTX 5070 Ti. FlashInfer #5963 gives the exact instruction and one trap: scales "held in .b32 registers (an immediate there does not assemble)".
- **On GB10,** nvfp4bench measured `mxf4nvf4` at 511 TFLOPS dense and `mxf8f6f4.block_scale` at 256, the latter with byte-padded FP4 operands. It did not measure FP8 operands, plain `f8f6f4` or `HMMA.F32`.
- **The open question.** M2 says FP8 runs at "~55%" of the MMA peak. If that peak is the 250 TFLOPS of SPEC §4 (~137 TFLOPS achieved), plain FP8 on GB10 cannot be half-rate, and the change buys nothing. The probe settles it.

**How it maps.**
- `prefill_gemm.cu` line 84, the FP8 path. The A/B fragments of `m16n8k32` are the same for both instructions; only the scale operands are added.
- The same probe decides whether the prefill attention (BF16 `mma.sync` with FP32 accumulation, ~51 TFLOPS) sits under a hardware cap. It informs item 7.

**Microbenchmark plan.**
1. **Probe, 10 minutes.** Register-resident loops of:
   - `m16n8k16` bf16 → f32 and f16 → f16;
   - `kind::f8f6f4` e4m3 → f32;
   - `kind::mxf8f6f4.block_scale.scale_vec::1X` e4m3 with ue8m0 = 0x7f;
   - `mxf4nvf4` as a control (expect ~511).

   Check the SASS (`QMMA` vs `QMMA.SF`, `HMMA.F32` vs `HMMA.F16`); nvfp4bench's `peak_mma.cu` (BSD-3) is a ready model. **Go if the unit-scale form is ≥ 1.5x the plain form.**
2. **Kernel, ~1 h.** Swap the FP8 instruction. `ling-prefill-tests` must stay byte-identical. Measure the FP8 projections at 2,048 tokens:
   - in-projection, N = 16,384, K = 5,120;
   - attention q, N = 12,288, K = 5,120;
   - out-projections, N = 5,120, K = 6,144.

   Success: ≥ +25% TFLOPS.

### 7. Prefill attention retile

**What to take.**
- **flash-attention #2599** (RTX 5090 / PRO 5000): an SM120 TMA forward with 1 DMA warp plus 8 MMA warps, to remove "~17 MB of local memory spilling"; +3.2% to +11.3% over the cp.async kernel.
- **sm121-kernels, measured on GB10** (D = 128, non-causal): BF16 FA reaches ~75 TFLOPS at B = 2, H = 32, S = 8192 and ~69 at S = 4096; its FP8 FA reaches ~108 at S = 2048.
- **FlashInfer #4149** keeps both attention GEMMs on the block-scaled MXFP8 MMA for SM120/121. It changes numerics (FP8 Q/K/V).

**How it maps.**
- M2 §7 item 1: "255 registers per thread ... one block of 6 warps runs per SM, at ~51 of 122 BF16 TFLOPS". Our head dimension is 256 with GQA 6, so the per-warp output tile is twice the D = 128 sources'.
- Splitting Q across 8 MMA warps, or the output across two passes over D, is the route to two blocks per SM.
- The row-invariance rule (fixed 4,096-key ranges) must be kept.

**Microbenchmark plan.**
- Shapes: 24 q heads, 4 KV heads, D = 256, causal; 256-query slices over 2K, 8K and 35K keys.
- Success: ≥ 70 TFLOPS at 8K and 35K, with outputs bit-identical across chunkings (`--prefix-check`).
- GPU time ~2 h.

### 8. Prefill GEMM producer (NVFP4 and FP8)

**What to take.**
- **GB10 ceilings from nvfp4bench:**
  - CUTLASS reaches "~375 TFLOPS dense (4096×14336×4096)";
  - "large square shapes (e.g. 8192³) run long enough to thermally throttle to <100 TFLOPS".
- **FlashInfer's b12x dense kernel**, which is CuTe DSL (#4253, #4305):
  - "Load a pipeline stage's scale-factor tiles into registers once, when the stage is acquired, instead of once per k block";
  - an epilogue race fix "for short-K multi-wave launches", worth reading before writing our own TMA epilogue;
  - small-M tiles.
- **NInfer** (C++, Apache-2.0) ships TMA variants of its NVFP4 input projections. Its Qwen3.8-27B NVFP4 prefill is 12,819 tok/s at 7,680 tokens on an RTX 5090. That card's compute is several times GB10's, and NInfer's artifact may also quantize the projections to NVFP4, so the number is a design reference, not a target for M2's 2,584.

**How it maps.** `prefill_gemm.cu`. M2 §7 item 2 already names the TMA producer and the FP8 layout change it needs. The warp-specialized variant M2 tried lost on NVFP4 because one producer warp issued ~1,800 `cp.async` per stage; TMA or bulk copies remove that.

**Microbenchmark plan.**
- Shapes, 2,048 tokens: FFN gate+up fused (N = 34,816, K = 5,120, NVFP4), down (N = 5,120, K = 17,408), plus the FP8 shapes of item 6. Reference: CUTLASS example 79a (M2 measured 325 / 262 TFLOPS).
- Success: NVFP4 ≥ 320 TFLOPS on gate+up and ≥ 300 on down, byte-identical outputs against the current kernel's arithmetic where the K order is unchanged.
- Also run one 35K prefill with clocks logged: M0 §1 left the prefill-heavy thermal case open, and nvfp4bench reports throttling on long GEMMs.
- GPU time ~3 h.

### 9. Fused producers for the FP32 intermediates

**What to take.**
- **SGLang #34934** (closed after part of it landed with the model support) for Qwen3.5-family NVFP4 W4A4 + FP8 checkpoints: "every projection input is normalized, written to HBM in bf16, then re-read and quantized by a separate kernel". It fuses SiLU+mul+NVFP4 quant, post-LN+FP4 quant, and gated-norm+FP8 quant.
- **SGLang #32443:** gated RMSNorm + group-128 FP8 quant in one Triton kernel, "preserving the baseline BF16 rounding boundary".
- **FlashInfer #5210:** RMSNorm + NVFP4 that "retains the row through FP32 normalization, per-16 E4M3 scale generation and native E2M1 packing".

**How it maps.** M2 §7 item 3: conv, norms and gates move FP32 tensors, ~15% at 35K. M2 already fuses some of these (the SwiGLU epilogue, the norms writing quantized inputs), so the first step is a byte count per remaining kernel.

**Microbenchmark plan.**
- Nsight Systems on the 35K prefill; list the kernels in the 2.35 s with their bytes. Fuse the top three.
- Success: −0.6 s at 35K, with top-1 agreement on M2's 24 agent prompts unchanged within noise.
- GPU time ~1.5 h.

### 10. Batches of 2-4 (lever 5)

**What to take.**
- **NInfer's lanes.** "One to eight execution lanes fixed at startup", exact-batch CUDA graphs per lane count, and a DFlash2 path with "draft counts 1..15". The design notes (`docs/maintainer/dflash.md`) describe live widths per request and how state is published.
- **What GB10 scaling looks like for this model** (SGLang + DFlash2, greedy code):
  - pangoleen: 81.6 → 119.3 → 199.4 → 276.0 → 359.5 tok/s aggregate at 1, 2, 4, 8 and 16 streams;
  - darkdatter: 16 draft tokens are best for one stream (78.6 tok/s, 9.52 accepted) and 10 for 16 streams (435.1 aggregate);
  - 0xBakeer (4-bit, different engine): "75 tok/s single-stream (k=14), or 246 tok/s aggregate at 8-way concurrency (k=7)".
- **The correctness bar.** Section 6: concurrency is where SGLang's DFlash2 path breaks.

**How it maps.**
- `engine.cpp`'s single-sequence loop and the KV/state pools of SPEC §11.
- `stream_gemm` at M = 32 and 64 (two and four sequences × 16 rows).
- The row-invariance rule must hold across sequences in a batch as well as across rows.

**Microbenchmark plan.**
1. **Step time.** M = 32 and 64 through `stream_gemm` and the verify attention with two and four 24K-token contexts. Success: step ≤ 1.15x the single-sequence step at two sequences and ≤ 1.4x at four. GPU time ~1 h.
2. **Correctness.** Before shipping, a concurrency-8 greedy test in the shape of sglang #35860's comment: 304 runs of one prompt, with a deterministic answer checked. GPU time ~1 h.

## 4. For the decode job (graph per step, PDL, verify attention): coordinate, not ranked

| Source | What it found | Why it matters here |
| --- | --- | --- |
| llama.cpp [#23825](https://github.com/ggml-org/llama.cpp/pull/23825) (merged) | "On DGX Spark ... an internal bug which caused a race condition in a kernel launched with `launch_fattn()`" under PDL; removed from PDL, ~0.2% cost | PDL on GB10 has at least one known race. Run `compute-sanitizer racecheck` and the bit-identity tests with PDL on |
| TRT-LLM [#18714](https://github.com/NVIDIA/TensorRT-LLM/pull/18714) (merged) | Gated RMSNorm "enables PDL when the launch grid underutilizes SMs, improving the kernel by 16–24%"; a fused sampler finish check saves ~45 µs per step | Small-grid kernels are where PDL pays |
| vLLM [#49547](https://github.com/vllm-project/vllm/issues/49547) | On GB10, a full decode graph versus piecewise graphs with spec decode: 55.2 against 47.5 tok/s (+16%, Qwen3.5-122B INT4, MTP) | The size of the prize for one graph per step on this box |
| llama.cpp [#30190](https://github.com/ggml-org/llama.cpp/pull/30190) (open, Vulkan) | Packs verify tokens and GQA heads into one flash-attention row tile: `FLASH_ATTN_EXT` 1072.1 → 562.8 µs at 24 Q / 4 KV heads, head size 256, 52K KV; Qwen3.8-27B +14.6% end to end (W7800); token-exact equivalence "not established" | Same shape as ours (GQA 6, D 256) |
| flash-attention [#2336](https://github.com/Dao-AILab/flash-attention/pull/2336) (open) | SM120 split-KV with FP32 partial outputs; validated on SM121a | Reference for the verify-attention split |
| flash-attention [#2634](https://github.com/Dao-AILab/flash-attention/pull/2634) (open) | FA4 on `sm_120`, incl. "fp8 e4m3/e5m2 decode (≈1.6–1.9× at GQA ratio ≤ 4, half the KV bandwidth)" | If SPEC §16.10's FP8 KV lands |
| FlashInfer [#4481](https://github.com/flashinfer-ai/flashinfer/pull/4481) (merged), [#5012](https://github.com/flashinfer-ai/flashinfer/pull/5012) (open, SM121) | One fused GDN decode step: `in_proj_ba` GEMV, conv1d update, gating, recurrence; "~0.54 ms/step of glue" removed for Qwen3.6-27B on RTX PRO 6000; +2% ITL at c=1, ~8% at c=2-4 | The β/α, norm and conversion kernels are 3.9 ms of our step (M1 §5) |
| SGLang [#43250](https://github.com/sgl-project/sglang/pull/43250) (open), vLLM [#59632](https://github.com/vllm-project/vllm/pull/59632) (merged), FlashInfer [#4250](https://github.com/flashinfer-ai/flashinfer/pull/4250) (closed), SGLang [#37491](https://github.com/sgl-project/sglang/pull/37491) (open) | On SM121, cuBLAS runs small-M BF16 GEMMs through SM80 WMMA kernels at "125–205 GB/s"; a 96 × 5120 GEMV is 3.00x and 2.43x cuBLAS on GB10 at m = 2 and 4; on SM120 a cuBLAS no-split-K cliff at M = 8 costs 32 µs instead of 5.5 | Our narrow β/α kernel is our own, but the shapes are the same |
| vLLM [#58718](https://github.com/vllm-project/vllm/pull/58718) (closed) | A tile table tuned on an RTX PRO 6000 (188 SMs) is 1.13-1.25x slower than the default config at M = 1024 on GB10 (48 SMs); a table re-swept on GB10 was needed | Tune every launch on the GB10 itself |

## 5. What others measured against the spec

| Spec section | Evidence | Effect |
| --- | --- | --- |
| §16.2, all-NVFP4 projections | DenseSpark: "Rolling native NVFP4 out layer by layer: layers 0 and 1 clear the error gate; 2, 3 and the aggregate miss it" (their own mean-NLL gate, their checkpoint) | A caution for the quality gate, not a verdict on the Minima checkpoint |
| §16.3, scales inside fused GEMMs | vLLM [#59845](https://github.com/vllm-project/vllm/pull/59845): fused NVFP4 shards run "under the max of its shards' global weight scales", inflating smaller shards up to 2.5x; [#47396](https://github.com/vllm-project/vllm/pull/47396): per-shard FP8 scales of the fused GDN in-projection crash vLLM's loader | Confirms the problem §16.3 measured on our checkpoint |
| §16.4, scale compression | vLLM [#60646](https://github.com/vllm-project/vllm/pull/60646): a lossless 4-bit-window scale code exists for a Qwen3.8 NVFP4 checkpoint | Supports the plan; item 5 |
| §16.6, tree verification | DenseSpark: "+7.40 ms per fork" when the recurrent state is forked; FlashInfer #6155/#6257 and SGLang #31069 avoid forks | Supports §16.6's no-fork design; item 1 |
| §16.7, PDL | llama.cpp #23825: a PDL race on DGX Spark | Test PDL with racecheck |
| §16.8, cluster reductions | DenseSpark: "Distributed shared memory across the SM cluster: exact, and 2.73x slower" | Weakens the ClusterFusion idea on GB10 |
| §16.8, wave quantization | FlashInfer [#5949](https://github.com/flashinfer-ai/flashinfer/pull/5949) (merged) had to add "SM-count-aware split and head-tile planners ... for GB10 (48 SMs)" | Plans must be made for 48 SMs, not inherited |
| §16.10, KV precision | SGLang [#36797](https://github.com/sgl-project/sglang/issues/36797): NVFP4 KV −29% decode on GB10 against FP8; pangoleen: FP8 KV "+20-29% past 130k, −12-21% prefill"; DenseSpark: attention is "5.7% of this model's decode read" for their engine | FP8 KV pays at long context; NVFP4 KV does not on this box |
| §4 / M0 §1, the bus | Practical ceilings of 231-240 GB/s reported by pangoleen, veloGB10, DenseSpark and the EXL3 kernel author, against M0's 262.8 GB/s pure read | ling-engine's own 232-235 contiguous figure matches them; M0's 262.8 is a tuned pure-read kernel (its simplest kernel reached 251.9) |
| M0 §1, thermals | agjs/gb10-clock-cap: stock clocks "power-limited to ~2455 MHz", 84 °C and 56 W GPU rail at 73.4 tok/s; a 2,000 MHz cap costs 2.7% decode for −49% rail power. nvfp4bench: long square GEMMs throttle "<100 TFLOPS" | M0 saw no decode throttling; the prefill-heavy case is still open (item 8) |

## 6. Findings outside the engine backlog

**Production's DFlash2 path in SGLang is reported to lose exactness, and under concurrency it can mix contexts.** None of this is reproduced here.

- [sglang #35860, comment](https://github.com/sgl-project/sglang/issues/35860), on a stock DGX Spark with the packed-FP4-head NVFP4 target: "this exact configuration gives 36.5% wrong greedy answers (111/304) on a simple ordering prompt at concurrency 8. Serial runs are 0/100 ... The dense BF16-head export brings it down to 1/304, and disabling speculation to 0/304."
- [#36548](https://github.com/sgl-project/sglang/issues/36548): "sometimes the last user message is appended to the wrong context" under concurrent load. A GB10 comment there reproduces it with `RadixArk/Qwen3.8-27B-NVFP4` and up to 8 running requests.
- [#35150](https://github.com/sgl-project/sglang/issues/35150): with every draft force-rejected, TARGET_VERIFY still diverges from plain decode: "accumulated GDN recurrent-state drift". A GB10 comment there: "diverges from ordinary decode on 19/32 frozen cases".
- [#38009](https://github.com/sgl-project/sglang/issues/38009): greedy DFlash2 diverges from target-only at token 30 with thinking on. A GB10 comment there: the output "is not stable across process restarts".

Mightling's production (SPEC §6) is SGLang + DFlash2 with `RadixArk/Qwen3.8-27B-NVFP4` and at most 8 running requests, and SWE-bench runs two sessions at once. Two follow-ups for Mightling, not for the engine:
- run the #35860 test on the production image;
- note that M0's two-stream numbers carry this risk.

The contrast for ling-engine: speculative output is bit-identical to plain decoding and the state is checked bit for bit (M1 §3). That property is now a measured differentiator, and item 10's correctness test should keep it so under batching.

Two smaller SGLang findings:
- [#43207](https://github.com/sgl-project/sglang/pull/43207): under DFLASH, presence, frequency and repetition penalties "are accepted and range-checked, but have no effect on the output".
- [#42209](https://github.com/sgl-project/sglang/pull/42209): the existing replay flag's outputs "are not bitwise equal to the stock verify's ... 79 of 80 requests diverged". Both #42209 and #42388 move production's memory use, not its speed, so the tuned-SGLang baseline (SPEC §15, third risk) would move only if #31069's tree verification ships.

## 7. GB10 measurements from others

All on one GB10 unless stated. They are not directly comparable with M1's 45.5 tok/s, which is **sampled agent replay** (temperature 1, top-p 0.95, top-k 20; 24K median context). Greedy code generation accepts 7-9 tokens per pass with this drafter; chat accepts ~3. The comparable quantity is the time per verify pass.

| Source | Configuration | Measurement (quoted) | Against ours |
| --- | --- | --- | --- |
| [pangoleen/qwen3.8-27b-dgx-spark-dflash2](https://github.com/pangoleen/qwen3.8-27b-dgx-spark-dflash2) | SGLang + `RadixArk/Qwen3.8-27B-NVFP4` + `maurienne-ai/...-DFlash2-NVFP4-RTNcal` (our pair), bf16 KV, greedy | Code: "64-78 tok/s" to 65k context, 6.8-8.3 accepted of 16. Chat: "~30 tok/s ... ~3 tokens per verify pass ... at the same ~100 ms per pass" (102 ms ITL on a second Spark). No drafter: "12.6-12.7 tok/s short". Prefill 1,657 (324 tokens), 2,506 (8,159), 1,811 (65,842). Aggregate 81.6 / 119.3 / 199.4 / 276.0 / 359.5 tok/s at 1-16 streams | ~100 ms per pass vs M0's 105.5 and M1's 115. Prefill: M2 2,637 / 2,584 / 2,106 |
| [darkdatter/gb10-repo](https://github.com/darkdatter/gb10-repo) | SGLang + NVFP4 + DFlash2, draft sweep | "78.6 tok/s" single at 16 draft tokens (accept_len 9.52); 8 tokens: 61.3 (6.63); "480.7 tok/s (16 streams)"; 10 tokens best for concurrency (435.1 at 16) | The draft-width optimum depends on concurrency |
| [hasso5703/dgx-spark-qwen38](https://github.com/hasso5703/dgx-spark-qwen38) | SGLang + NVFP4 + DFlash2, "calibrated NVFP4 head 16 deep", deterministic kernels | "71.4 tok/s greedy median"; frozen battery, thinking on: agentic coding 40 / 34, prose EN/FR/DE 22 / 20 / 18; "135-148" aggregate at 8 streams | Agentic-coding rows are closest to our workload |
| [Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark](https://github.com/Weschera/Qwen3.8-27B-NVFP4-DFlash2-DGX-Spark) | SGLang, synthetic random-token prompts | C1 "42.04 tok/s" at block 10 (3.24 → 4.82 accepted); C8 114.5-120.58; "Two identical launches ... ~42 or ~33 tok/s C1" from autotuner choices at boot | Boot-to-boot spread larger than most tuning gains |
| [0xWhiteMage/...-dflash2](https://github.com/0xWhiteMage/qwen3.8-27b-kearuga-sglang-dgx-spark-dflash2) | SGLang, 50-prompt battery, clock capped at 2,400 MHz | Stock drafter K = 10: 30.85 tok/s C1, 98.31 C4 aggregate; no speculation "10.30 tok/s C1 / 39.56 tok/s C4"; fine-tuned drafter + 64K head: 35.33 / 108.90 | Speculation is ~3x on mixed prompts |
| [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | SGLang, DFlash2 / DSpark / MTP | DFlash2: code "50.9 tok/s", long essay "25.4 tok/s" (n = 5); "two boots of the same image differed 6.5%" | Same 2x code/prose gap |
| [sxuff/qwen38-27b-nvfp4-dflash2-dgx-spark](https://github.com/sxuff/qwen38-27b-nvfp4-dflash2-dgx-spark) | SGLang, 27 frozen requests | "71.50 tok/s median whole-request" vs "57.11" for the original DFlash2 recipe | — |
| vLLM [#60226](https://github.com/vllm-project/vllm/pull/60226) | vLLM + `nvidia/Qwen3.8-27B-NVFP4` + MTP k = 3, SPEED-Bench, greedy | 27.2 tok/s c=1, 25.7 c=4, AL 2.906 | MTP drafting is far behind DFlash2 here |
| [albond/DenseSpark-Qwen3.8-27B](https://github.com/albond/DenseSpark-Qwen3.8-27B) | vLLM 0.27.1, 4-bit body + MTP | "One request decodes at 44.9 tokens per second" (thinking off); 260.2 tok/s at 16 parallel (balanced); "98.3% of the measured 235.5 GB/s streaming ceiling" | Bandwidth ceiling, negative results (section 5) |
| [sf-stav/veloGB10](https://github.com/sf-stav/veloGB10) | Own Rust + CUDA engine, NVFP4 + DFlash2, greedy | "~238 GB/s measured sustained bandwidth (idle)"; single node "> 40 tok/s" average, "~70 tok/s" sustained on code, "~25" on a mixed-content run | Another from-scratch GB10 engine: no faster on this model |
| [secYOUre/nvfp4bench](https://github.com/secYOUre/nvfp4bench) | Microbenchmarks | NVFP4 MMA "511 TFLOPS packed dense", 1022 sparse; `mxf8f6f4` 256; CUTLASS "~375 TFLOPS" on 4096×14336×4096; long square GEMMs throttle "<100 TFLOPS" | M2's NVFP4 GEMMs: 230-287 TFLOPS |
| [blake-snc/sm121-kernels](https://github.com/blake-snc/sm121-kernels) | Hand-written PTX, D = 128 | BF16 FA "~75 TFLOPS" (B = 2, H = 32, S = 8192), FP8 FA "~108" (B = 1, H = 32, S = 2048); BF16 GEMM "~54–56 TFLOPS" at 4096³ | M2's prefill attention: ~51 TFLOPS at D = 256 |
| [agjs/gb10-clock-cap](https://github.com/agjs/gb10-clock-cap) | Clock caps, decode | Stock "~2455 MHz": 73.4 tok/s, 84 °C, 56 W; 2000 MHz: 71.4 tok/s, 72 °C, 28.6 W | M0: 2,405 MHz held, 82 W peak, 79 °C |
| vLLM [#49547](https://github.com/vllm-project/vllm/issues/49547) | Qwen3.5-122B INT4 + MTP | Piecewise graphs 47.5 vs full graphs 55.2 tok/s | Section 4 |
| SGLang [#36797](https://github.com/sgl-project/sglang/issues/36797) | Flash-Next, 2 × GB10, TP = 2 | NVFP4 KV 44.0 vs FP8 KV 56.8-58.6 tok/s | Section 5 |

## 8. Watch list for the weekly digest

**Repositories, with the paths to watch.**
- vllm-project/vllm (`vllm/v1/worker/gpu/spec_decode/`, `vllm/model_executor/layers/draft_vocab.py`)
- sgl-project/sglang (`python/sglang/srt/speculative/dflash*`, `layers/attention/hybrid_linear_attn_backend.py`, `mem_cache/memory_pool.py`)
- NVIDIA/TensorRT-LLM (GDN MTP replay)
- ggml-org/llama.cpp (`ggml-cuda/gated_delta_net.cu`, `fattn*`, `mmvq.cu`)
- flashinfer-ai/flashinfer (`flashinfer/gdn_kernels/`, `flashinfer/experimental/b12x`, `include/flashinfer/attention/sm120/`)
- Dao-AILab/flash-attention (`flash_attn/cute/*sm120*`)
- NVIDIA/cutlass (SM120 block-scaled builders; examples 79, 91)
- triton-lang/triton (sm_120 scaled dot)
- fla-org/flash-linear-attention
- deepseek-ai/DeepGEMM (SM120)
- tile-ai/tilelang
- Neroued/ninfer
- blake-snc/sm121-kernels
- secYOUre/nvfp4bench
- sf-stav/veloGB10
- rdaum/eider
- HawkBearPig/dgpp
- albond/DenseSpark-Qwen3.8-27B
- z-lab/dflash
- kaist-ai-osi-lab/BASTION
- pangoleen/qwen3.8-27b-dgx-spark-dflash2
- hasso5703/dgx-spark-qwen38
- darkdatter/gb10-repo

**Labels.**
- vLLM: `speculative-decoding`, `performance`, `nvidia`, `quantization`
- SGLang: `DGX Spark`, `blackwell`, `linear-attention`, `speculative-decoding`
- FlashInfer: `arch: sm12x`, `op: linear attention`, `op: attention`
- llama.cpp: `Nvidia GPU`, `speculative`
- TensorRT-LLM: `Speculative Decoding`, `CUDA Graph`

**Queries** (weekly, `updated:>=<last run>`):
- `gh search prs --repo <each engine> -- sm121 OR GB10 OR "DGX Spark"`
- `gh search prs -- dflash2 OR "DFlash2"`
- `gh search prs -- "gated delta" verify OR replayssm OR "tree verify"`
- `gh search issues -- GB10 "qwen3.8-27b"`
- `gh search prs -- "block_scale" sm120 OR sm121 fp8`
- `gh search repos -- "qwen3.8-27b" spark`
- `gh search issues --repo sgl-project/sglang -- DFlash2 diverges OR corruption`

**Open items to re-check:**
- SGLang #31069, #42388, #42209
- FlashInfer #6155, #6257, #4140, #5012, #6179, #4149
- vLLM #59740, #60226, #60646
- flash-attention #2599, #2634
- CUTLASS #3441
- the SGLang DFlash2 correctness issues of section 6

**A trap to keep in view when adopting CUTLASS or tilelang code:** tilelang [#3430](https://github.com/tile-ai/tilelang/pull/3430) found that CUTLASS defines `CUTLASS_ARCH_MMA_SM120A_ENABLED` "only when `__CUDA_ARCH__ == 1200`, so on `sm_121a` every block-scaled MMA compiled to `BPT.TRAP`".

## Licences at the cited commits

| Repository | Licence (file, commit) | Reuse |
| --- | --- | --- |
| vllm-project/vllm | Apache-2.0 (`LICENSE` @c04c79e, @3ed5c09) | Port (Python/Triton: rewrite) |
| sgl-project/sglang | Apache-2.0 (`LICENSE` @f52fd3b) | Port (rewrite) |
| flashinfer-ai/flashinfer | Apache-2.0 (`LICENSE` @d57bfb1, @b87bfa9); `trtllm-gen` cubins excluded | Port (C++ headers) / rewrite (CuTe DSL) |
| NVIDIA/TensorRT-LLM | Apache-2.0 (`LICENSE` @ee241d2) | Port / rewrite |
| ggml-org/llama.cpp | MIT (`LICENSE` @c04ccdd) | Port |
| NVIDIA/cutlass | BSD-3-Clause (`LICENSE.txt` @e8ecfad) | Port |
| Dao-AILab/flash-attention | BSD-3-Clause (`LICENSE` @fcf38ee) | Rewrite (CuTe DSL) |
| triton-lang/triton | MIT (`LICENSE` @fcf734b) | Ideas (instruction choice) |
| fla-org/flash-linear-attention | MIT (`LICENSE` @7004d19) | Rewrite (Triton) |
| deepseek-ai/DeepGEMM, HazyResearch/ThunderKittens | MIT (`LICENSE` @main) | Not used |
| Neroued/ninfer | Apache-2.0 (`LICENSE` @81c8ce0) | Port (C++/CUDA) |
| blake-snc/sm121-kernels | Apache-2.0 or MIT (`LICENSE-APACHE` @e5a28a3) | Port (PTX) |
| secYOUre/nvfp4bench | BSD-3-Clause (`LICENSE` @df8e8f4) | Port (probe) |
| albond/DenseSpark-Qwen3.8-27B | MIT (`LICENSE` @9ae122f) | Ideas (measurements) |
| z-lab/dflash, kaist-ai-osi-lab/BASTION | MIT (`LICENSE` @07ebd93, @1577b5d) | Port (drafter code) |
| sf-stav/veloGB10, rdaum/eider, HawkBearPig/dgpp, local-inference-lab/b12x, Luce-Org/lucebox | Apache-2.0 (`LICENSE` @HEAD) | Ideas so far |
| MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks | AGPL-3.0 (`LICENSE` @HEAD) | **Ideas only** |
| Memoriant/dgx-spark-kv-cache-benchmark | No licence file | **Ideas only** |

No code was copied into ling-engine for this report. Any port above needs its licence notice kept and a NOTICE entry; ling-engine's AGPL-3.0 can include Apache-2.0, MIT and BSD code.
