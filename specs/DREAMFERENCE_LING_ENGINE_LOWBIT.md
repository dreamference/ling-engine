# ling-engine: A faster, non-exact target: fewer bits per weight (a study)

Part of the ling-engine specification; the index is [README.md](./README.md). A study written on 2026-10-27 at the owner's decision of that day: look at a faster variant of the target that is not exact, either fewer bits per weight (about 3) or 2:4 structured sparsity, aiming at 1.3–1.6× decode. **It is a study only.** Nothing here changes the shipped model, the served checkpoint or an engine default without the owner's OK. It replaces nothing in [MODEL §5.3](./DREAMFERENCE_LING_ENGINE_MODEL.md) ("Fewer bits per weight: considered, not targeted"). It is the measured, sourced follow-up that §5.3 deferred. Like the CLI and API documents, it has no section number of the former `SPEC.md`.

## 0. Summary

- **The served target is not 4.5 bits a weight.** The FFN and the LM head are NVFP4 (4.375 bits once B26 has coded the scales). Every DeltaNet and attention projection is FP8: 7.22 GB of the 17.33 GB one verify pass reads ([MODEL §5.2](./DREAMFERENCE_LING_ENGINE_MODEL.md), BACKLOG §19 item 27). So the largest byte saving with the least risk is **moving the FP8 projections to NVFP4**: −3.27 GB, about −14 ms a step. That is more than a 3-bit FFN saves (−2.14 GB, −9.4 ms). The move is already queued as a quality-gated lever ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), "Order" item 4). A published all-NVFP4 checkpoint of this model exists (Minima). It is downloaded and registered in Mightling's model matrix, but no results of its SWE-bench night (night 2) exist: MODELS §2.2 says "not yet run" and the SWE-bench directory has no `night2/`.
- **No single candidate reaches 1.3×.** The projections below assume acceptance is unchanged, and they are measured against the base of BACKLOG item 34 (105.86 ms a step, 6.749 tokens a step, 63.75 tokens/s greedy):

  | Candidate | Speed-up |
  |---|---|
  | All-NVFP4 projections | ×1.16 |
  | FP8 KV at agentlong's 41K context | ×1.06 |
  | 3-bit FFN | ×1.10 |
  | All-NVFP4 projections + FP8 KV | ×1.24 |
  | All-NVFP4 projections + 3-bit FFN | ×1.29 |
  | All-NVFP4 projections + 3-bit FFN + FP8 KV | **×1.39** |
  | 3-bit everywhere + FP8 KV | ×1.47–1.50 |

  1.6× would need the step at 66 ms, about 40 ms off, which no weight format above 2 bits gives at one stream. Every one of these candidates can only lower acceptance, the other factor in tokens/s.
- **Measured on GB10 (2026-10-26, third-puffin, dummy data, section 6):**
  - At the verify's 16 rows, a 3-bit format and a 2:4 sparse NVFP4 format both stream at the same 205–220 GB/s as dense NVFP4. A matrix's time therefore falls with its bytes: decoding in registers is free here, unlike item 30's entropy code.
  - sm_121 accepts and runs 2:4 sparse MMAs (`mma.sp`) for FP16, FP8, FP4 and block-scaled NVFP4 and MXFP8.
  - The library path is no use at these shapes. cuSPARSELt 0.8.1 runs 2:4 FP16 2.4–4.6× *slower* than dense cuBLAS at 16 rows and refuses 2:4 FP8.
- **2:4 sparsity is rejected on bytes before quality.** Its metadata is 2 bits per kept value. A 2:4 NVFP4 matrix is therefore 3.375 bits a weight, exactly what a dense 3-bit format costs while keeping every weight. A 2:4 FP8 matrix is 5 bits, more than NVFP4's 4.375. The literature then puts 2:4 well below 3-bit in quality on this model family: Qwen3-32B with only its MLP 2:4-pruned loses 17 points on LiveCodeBench. Getting that back takes about 13 billion tokens of distillation.
- **Recommendation: go, narrowly.** The first candidate to build is **the all-NVFP4 projections (Minima)**, together with the FP8 KV cache, the other queued lever. Expect ×1.24 at agentlong's context if acceptance holds. 3-bit FFN weights come second and only through an offline quality filter (section 7); they are the only way to 1.3× and the riskiest candidate in quality. 2:4 sparsity, MXFP4 and layer skipping are not pursued.

## 1. Where one step's time goes, and which floor

The base: main `532537b` as BACKLOG §19 item 34 measured it (node-b, agentlong at one stream, all sixteen turns, greedy, tree verification, the megakernel stages, the ScaleSearch drafter).

| Part | Time per step |
|---|---|
| Draft | 10.28 ms |
| Verify (16 rows) | 94.84 ms |
| Commit | 0.81 ms |
| **Total** | **105.86 ms** |

The step accepts 6.749 tokens, for 63.75 tokens/s. Main has since gained about 1% from the sparse head (item 19: 65.67 tokens/s greedy), which does not change the arithmetic below.

The bytes one verify pass reads, from item 27's inventory with B26's coded scales:

| Item | Format | GB per verify |
|---|---|---|
| FFN gate, up, down (64 layers) | NVFP4 codes 8.56 + coded scales 0.80 | 9.36 |
| DeltaNet and attention projections | FP8 | 7.22 |
| LM head | NVFP4 | 0.70 |
| β/α projections, norms | BF16 | 0.05 |
| **Target weights** | | **17.33** |
| KV, BF16, at agentlong's mean prompt (41.3K tokens; `bench/agentlong/manifest.json` bins) | 64 KB a token | 2.71 |
| DeltaNet and conv state, read and written | FP32 | 0.31 |
| **Total** | | **20.35** |

The draft reads another 1.6 GB: the drafter's layers, and the target's head again or its sparse rows.

**The floors.** At the in-order read rate item 29 measured (234 GB/s), the verify's 20.35 GB take 87 ms, and the weights alone take 74 ms. The engine's GEMMs stream at 205–235 GB/s at 16 rows, which leaves roughly 8 ms of the 94.84 ms verify that is not bytes: the recurrence's tree paths, attention's compute, the norms and the accept. The "~59 ms weight-read floor" of the owner's brief is about 16 GB at the nominal 273 GB/s. No kernel has reached that rate on this machine: 234 GB/s is 86% of nominal. The time a format saves is therefore its bytes at about 228 GB/s, the middle of the engine's measured GEMM rates (item 34: coded NVFP4 gate|up 232, down 224 GB/s at 16 rows). The microbenchmark of section 6 confirms that rate for the new formats.

**The verify profile.** The verify-profiling work (`specs/DREAMFERENCE_LING_ENGINE_VERIFY_PROFILE.md`, not yet in main on 2026-10-27) will split the 94.84 ms exactly. When it lands, its weight-GEMM total replaces the 228 GB/s assumption in section 2's table, and the non-byte remainder bounds every candidate from below.

## 2. Candidates: bytes, step and tokens per second

Bits per weight below include the scales: one E4M3 scale per 16 values, coded to 6 bits as B26 stores them (0.375 bits a weight). Times are bytes at 228 GB/s (weights) or 226 GB/s (KV, item 29's attention read). The new step is 105.86 ms minus the saving, and tokens/s = 6.749 / step, **at unchanged acceptance**. The last column is how much acceptance the candidate can lose and still reach 1.3×.

| Candidate | Bits/weight (scope) | GB saved per verify | ms saved | Step | Tokens/s | × | Acceptance loss allowed at 1.3× |
|---|---|---|---|---|---|---|---|
| (a1) All-NVFP4 projections | 8 → 4.375 (projections) | 3.27 | 14.3 | 91.5 | 73.8 | 1.157 | none (below 1.3×) |
| (a1′) FP6 projections, e2m3 + E4M3/16 | 8 → 6.375 (projections) | 1.47 | 6.4 | 99.4 | 67.9 | 1.065 | none |
| (a2) 3-bit FFN | 4.375 → 3.375 (FFN) | 2.14 | 9.4 | 96.5 | 70.0 | 1.097 | none |
| (a3) MXFP4 FFN | 4.375 → 4.25 (FFN) | 0.27 | 1.2 | 104.7 | 64.5 | 1.011 | none |
| (b1) 2:4 NVFP4 FFN | 4.375 → 3.375 (FFN) | 2.14 | 9.4 | 96.5 | 70.0 | 1.097 | none |
| (b2) 2:4 FP8 projections | 8 → 5 (projections) | 2.71 | 11.9 | 94.0 | 71.8 | 1.126 | none |
| (c) FP8 KV at 41.3K / 25.7K / 64K / 128K | 16 → 8 (KV) | 1.35 / 0.84 / 2.15 / 4.29 | 6.0 / 3.7 / 9.5 / 19.0 | 99.9 at 41K | 67.6 at 41K | 1.060 at 41K | none |
| (a1)+(c) | | 4.62 | 20.3 | 85.5 | 78.9 | 1.238 | none |
| (a1)+(a2) | | 5.41 | 23.7 | 82.1 | 82.2 | 1.289 | none |
| **(a1)+(a2)+(c)** | | 6.76 | 29.7 | 76.1 | 88.6 | **1.390** | 6.5% |
| (a2) and 3-bit projections | 3.375 everywhere but the head | 6.31 | 27.7 | 78.2 | 86.3 | 1.354 | 4.0% |
| … + (c) | | 7.66 | 33.7 | 72.2 | 93.5 | 1.467 | 11.4% |
| … + 3-bit head (verify and draft) | | 7.98 | 35.1 | 70.8 | 95.4 | 1.496 | 13.1% |

The microbenchmark's own composition of a verify pass agrees with this byte model (section 6, `bench/lowbit/summarize.py`, 16 rows). It gives the projections in NVFP4 −14.6 ms (the model's −14.3), a 3-bit FFN −10.4 (−9.4), both −25.1 (−23.7), and a 2:4 NVFP4 FFN with NVFP4 projections −24.9.

At two streams (32 rows: a batch of two, or wider trees) the savings are about the same in milliseconds (−13 to −28 ms in section 6's 32-row composition) on a step of about 145 ms per pair. Per stream, then, they are worth somewhat less.

**(d) Layer skipping and expert skipping are excluded.** Skipping a layer of the target changes the model on every token, with no exactness and no bound on the damage. The model is dense, so there are no experts to skip. Self-speculation from a sub-network is already beaten by DFlash2: RESEARCH §16.13 records an acceptance of 0.038 for the DeltaNet-subgraph drafter on hybrids like this one. None of it is pursued.

## 3. Expected quality loss: the literature

Numbers for Qwen-family models of 27–32B where they exist; proxies are labelled. Sources checked on 2026-10-26; "not found" is a search result, not proof of absence.

**(a1) All-NVFP4 projections: published for this exact model.** Minima (arXiv 2609.04098, 2026-09-03) quantizes all 496 linear layers of Qwen3.8-27B to NVFP4 W4A4, including the Gated DeltaNet projections. Its "RadixArk" column is the recipe Mightling serves (MLP and head NVFP4, projections FP8).

| Benchmark | BF16 | Minima | Served recipe |
|---|---|---|---|
| MMLU-Pro | 80.4 | 79.7 | 79.1 |
| GSM8K | 95.5 | 95.5 | 95.7 |
| AIME'25 | 86.7 | 86.7 | 84.2 |
| GPQA-Diamond | 86.5 | 85.1 | 85.4 |
| LiveCodeBench v6 | 79.0 | 78.5 | 79.6 |
| Average against BF16 | | −0.52 | −0.82 |

On benchmarks, then, Minima is level with the served recipe. **Perplexity disagrees**: at 4K tokens Minima scores 7.67 against 7.35 (BF16 6.95), and at 32K 10.84 against 9.95 (BF16 10.35). The gap narrows with position. That disagreement is exactly what section 7's paired test on agent traces must settle.

Three caveats:
- The paper ran on an RTX PRO 6000 with vLLM. It decoded *slower* at batch 1 (47 against 51 tokens/s), which it puts down to kernels; our engine reads bytes, not recipes.
- Minima's numbers are W4A4. The engine runs NVFP4 weights against FP16 activations (W4A16, `stream_gemm`), so it skips the activation half of the error. **The paper's numbers are a lower bound for the engine.**
- The fused DeltaNet projections need harmonized global scales. Minima ships them; Mightling's `scripts/check_nvfp4_fused_scales.py` checked 352 of 352 equal (dgxcoder `specs/DREAMFERENCE_MODELS.md` §2.2).

**(a1′) FP6** is the fallback if (a1) fails: it halves the risk with half the saving. ModelOpt 0.48 has an MXFP6 preset. No Qwen result was found.

**(a2) About 3 bits: a measurable loss without healing.**

There is no 3-bit floating format in NVIDIA hardware. PTX ISA 9.4 lists e4m3, e5m2, e3m2, e2m3 and e2m1 as the only floating element types for `mma`, and ModelOpt rejects any other FP format. "NVFP3" does not exist. A 3-bit format is a storage format decoded in registers, which is how `stream_gemm` already treats NVFP4 (section 6).

| Model, method | Result at 3 bits | Source |
|---|---|---|
| QwQ-32B, 5-task average | 78.7 at BF16. w4 AWQ/GPTQ −0.8/−0.9; **w3 AWQ −6.9, GPTQ −4.4** | arXiv 2504.04823 v2 |
| QwQ-32B, LiveCodeBench | 60.2 → 50.3 (AWQ w3), 53.0 (GPTQ w3) | arXiv 2504.04823 v2 |
| DeepSeek-R1-Distill-Qwen-32B | w3 −3.9/−5.1 | arXiv 2504.04823 v2 |
| Qwen3-32B, AWQ w3 per-channel | MMLU 81.2 → 70.0 (the g128 row, 54.9, looks anomalous). GPTQ w4 g128: 80.6 | arXiv 2505.02214 v1 |
| Qwen3.6-27B, mean KL divergence (MLX) | 3-bit 0.073 against 4-bit 0.023: about 3× | Unsloth's Qwen3.6 model documentation |
| Llama-3-70B (proxy), WikiText-2 | QTIP 3-bit 3.18 against BF16 2.59; QuIP# 3.59 | arXiv 2406.11235 |
| Llama-2-70B (proxy), WikiText-2 | QTIP 3.26, AQLM 3.36 against 3.12 | arXiv 2406.11235 |

The paper's verdict on 2504.04823: "4-bit nearly lossless, 3-bit becomes risky", with the largest losses on AIME, LiveCodeBench and GPQA.

Mixed precision is the usual remedy. Minima mentions one variant of this very model with 3-bit MLP codebooks plus a low-rank residual healed by distillation, keeping the DeltaNet and attention projections at 4 bits (12.3 GiB). It reports no accuracy for it. No HumanEval, MBPP or SWE-bench number at 3 bits was found for any Qwen 27–32B model.

**Expectation:** plain post-training 3-bit on the FFN costs 2–7 points on code and reasoning. Better codebooks (QTIP-class) or distillation would be needed to come under the bar of section 7, and distillation is cloud-GPU work.

**(a3) MXFP4 is worse than NVFP4 and saves 3%.** On Qwen3-32B, weight-only recovery is 99.8/99.5% for NVFP4 (RTN/GPTQ) against 91.8/96.7% for MXFP4 (arXiv 2509.23202 v3, Table 10). Not pursued.

**(b) 2:4 sparsity: large losses, expensive recovery.**
- **The only Qwen data at this scale** (arXiv 2606.10445, 2026-06): Qwen3-32B, MLP only, 50% in a 2:4 pattern. SparseGPT takes the 4-task average from 71.68 to 60.27, LiveCodeBench from 60.19 to 43.05 and GPQA from 65.49 to 53.71. Wanda collapses to 18.96.
- **Llama-2-70B (proxy)**, 2:4 WikiText-2: 5.16 (Wanda) and 5.40 (SparseGPT) against 3.12 (arXiv 2306.11695).
- **Learned masks** (MaskLLM, arXiv 2409.17481) close part of the gap on 7–13B at about 2B tokens of training.
- **Full recovery** (Sparse Llama 3.1 8B, 2:4) took 13B tokens of distillation, 26 hours on 32 H100s for an 8B model, to reach 98.4% of the dense model on the Open LLM Leaderboard v1 (MMLU 60.6 against 65.4) (model card, huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-2of4). Scaled by parameters, a 27B model is in the thousands of H100-hours.
- **At matched size, quantization generally beats pruning** (Kuzmin et al., arXiv 2307.02973).
- **Our own drafter measurement** (BACKLOG §19 item 33): magnitude 2:4 cost 4.7% acceptance on the MLP and 14% on every projection. That is a drafter, not the target, but it points the same way.
- **The serving stacks have dropped 2:4.** vLLM removed its 2:4 integration (PR #36799, 2026-03-23), and llm-compressor's main removed `sparsegpt` and `wanda` (#3377, 2026-10-23); release 0.14.0.1 still has them.

**(c) FP8 KV: near-lossless with calibrated scales.**
- **On this model:** Minima measures +0.13 perplexity at 32K on BF16 weights. Static per-layer scales recover 83% of that, and RULER stays at 100 at 32K and 64K (arXiv 2609.04098; RESEARCH §16.10).
- **Qwen3.5-27B with uncalibrated scales:** it matches BF16 on MRCR to 1M tokens, and at most −0.7 points on reasoning (the vLLM blog, 2026-04-22).
- **Known trap:** a Hopper kernel's accumulation bug once took 128K needles from 91% to 13%. The engine's attention must be checked at long context, not assumed.

## 4. The drafter

The target verifies every draft, so a changed target keeps the engine exact *with respect to the new target*: the output is the low-bit model's, token for token. Acceptance moves only speed.

DFlash2 conditions on the target's hidden states at layers 5, 19, 33, 47 and 61 and scores with the target's own LM head ([MODEL §5.4](./DREAMFERENCE_LING_ENGINE_MODEL.md)). Any change to the target shifts both, so acceptance must be measured, never assumed. Mightling's MODELS §2.2 already says this for Minima.

What is known:
- **No measured acceptance was found** for a drafter trained against BF16 and then run against a more heavily quantized target.
- EAGLE-2 reports "nearly no degradation" in acceptance length with W4A16 and W8A8 targets (arXiv 2505.22179, in prose, untabulated).
- A DFlash drafter trained on an INT4 Qwen3.5-4B target accepts 4.97, and 5.03 when trained BF16-then-INT4 (arXiv 2607.04244). The BF16-only drafter on the INT4 target was not reported.
- The served NVFP4 target already differs from the BF16 model the drafter was trained on, and acceptance is fine. For (a1) and (c) a small loss is the expectation; for 3-bit and 2:4 it is open.

**Measuring it is cheap and comes first.** A paired-by-request acceptance A/B on the replay set is the tool item 22 used (`bench/quant/sglang_paired.py` on SGLang, `engine_paired.py` on ling-serve). It takes about one GPU-hour and needs only that one of the two servers loads the candidate. SGLang loads Minima today, and ling-serve after section 5's loader work.

**Retraining** is SpecForge's DFlash2 recipe on the new target's hidden states, the same pipeline BACKLOG §19 item 20 (the agent-tuned drafter) plans. It is needed only if acceptance falls by more than its run-to-run spread (3.0–4.5% between two seeds within one arm, item 22). Its cost is the cloud-GPU question the owner deferred to this study, and no sourced figure for a DFlash2 training run of this size was found. The decision rule: retrain only if the measured acceptance loss takes the candidate's tokens/s gain below half of its byte gain.

## 5. Engineering effort and tooling

| Candidate | Checkpoint and tools | Engine work | Effort |
|---|---|---|---|
| (c) FP8 KV | None (runtime). Calibrated per-layer scales from a calibration pass (RESEARCH §16.10) | The safe build of B41 (`reports/backlog-2026-10-09.md`): a page quantized only when full and accepted, the open tail and verify rows in BF16, two boundary layers BF16, separate K and V scales; the attention kernel's FP8 read. Quality: the 32K and 128K needle checks of [§13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) | 1–2 days plus the quality run |
| (a1) All-NVFP4 projections | Minima, pinned in Mightling's matrix as `qwen3.8-27b-minima-nvfp4-dflash2` (revision `16e768e7…`, 18.8 GB, in the development machine's Hugging Face cache on 2026-10-27), compressed-tensors format | `Model` loads the projections as FP8 only (`src/core/model.cu`: `fp8(ck, …)` for q, k, v, o, `in_proj_qkv`, `in_proj_z`, `out_proj`). It needs: an NVFP4 path for those tensors in compressed-tensors names (`weight_packed`, `weight_scale`, `weight_global_scale`); the layer structs and the megakernel's DeltaNet stage on NVFP4 descriptors (`stream_gemm` already takes NVFP4 at any shape); the prefill GEMM's NVFP4 path for those shapes; the exactness gates within the mode | 3–5 days |
| (a2) 3-bit FFN | A quantizer to an 8-entry per-tensor codebook with E4M3 per-16 scales (section 6's `b3`), GPTQ-style error feedback. GPTQModel 7.7.0 supports `qwen3_5` hybrids but quantizes to uniform INT grids. QTIP/AQLM show no sign of supporting the DeltaNet hybrid. Healing (low-rank residual, distillation) needs cloud GPUs | A `b3` path in `stream_gemm` (the microbenchmark's decode: masks, one `prmt` per four values, `cvt` from E4M3) and in the megakernel's FFN; a prefill path that dequantizes to FP8/BF16 tiles; a new tile layout in the loader | 1–2 weeks after a passing quality filter |
| (b) 2:4 | SparseGPT in llm-compressor 0.14.0.1 or ModelOpt's `sparsegpt`, then recovery distillation (~13B tokens) | `mma.sp` works (section 6), and the kernel is the dense one with metadata. Not the obstacle | Weeks of cloud training: not pursued |

## 6. Measurements: 2026-10-26 on third-puffin (dummy data, no model)

**Windows.** Two GPU windows on third-puffin under the lock convention (`flock ~/.ling-gpu.lock`, logged in `~/.ling-gpu-holds.log` as `lowbit-study-agent`): 23:51–23:52 and 23:54–23:56 UTC. No production model server was running there, so none was stopped. No model was converted or loaded. The scripts are in `bench/lowbit/`: `window.sh`, `sparse_isa_probe.{cu,sh}`, `lowbit_gemm.cu`, `sparse_lib.py`, `summarize.py`. Raw results are in `bench/lowbit/results-2026-10-26/`.

**(1) What sm_121 runs: `sparse_isa_probe`.** Each MMA multiplies an all-ones compressed A by an all-ones B and checks every output against K/2. The build target is `-gencode arch=compute_121a,code=sm_121a`.

| Instruction | Result |
|---|---|
| `mma.sp::ordered_metadata m16n8k32` FP16 → FP32 | accepted, runs, correct |
| `… m16n8k64 kind::f8f6f4` E4M3 × E4M3 | accepted, runs, correct |
| `… m16n8k64 kind::f8f6f4` E2M1 × E2M1 | accepted, runs, correct (FP4 sits in bits 2–5 of each byte) |
| `… m16n8k128 kind::mxf4nvf4 block_scale scale_vec::4X` E2M1, UE4M3 scales (sparse NVFP4) | accepted, runs, correct |
| `… scale_vec::2X` with UE4M3 | rejected by ptxas ("Illegal modifier") |
| `… m16n8k64 kind::mxf8f6f4 block_scale scale_vec::1X` E4M3, UE8M0 (sparse MXFP8) | accepted, runs, correct |

So **SM121's sparse tensor cores take FP4, NVFP4 and FP8 operands.** The engine's design does not need them, though: `stream_gemm` decodes weights to FP16 and multiplies on the FP16 MMA, and the FP16 `mma.sp` is all a 2:4 path would need. At 16 rows the MMA is never the bound.

**(2) Streaming GEMMs at the verify shapes: `lowbit_gemm`.** The kernels have the shape of `stream_gemm`: 8 warps over TILES × 16 weight rows, warp w on 128-value K chunks w, w+8, …, weights decoded to FP16 in registers, `mma.sync m16n8k16` (or `mma.sp m16n8k32` for 2:4), activations FP16 from L2, warp sums added in a fixed order. They are not the engine's tuned kernel. Their dense NVFP4 runs at 205–216 GB/s at 16 rows, against the engine's 224–232. The comparison that matters is therefore format against format inside one harness.

The five formats (bits a weight, scales raw as stored here):

| Format | Bits a weight | Layout |
|---|---|---|
| `fp4` | 4.5 | NVFP4 |
| `fp8` | 8 | FP8 E4M3 |
| `b3` | 3.5 | 3-bit codes into a per-tensor 8-entry E4M3 codebook, E4M3 per-16 scales; a 2-bit plane and a 1-bit plane, decoded with masks and one `prmt` per four values |
| `sp4` | 3.5 | 2:4 NVFP4 with 2-bit indices |
| `sp8` | 5 | 2:4 FP8 |

Every kernel matched a naive reference decoder (relative error ≤ 5e-7 at 1, 16 and 32 rows and every tiling) before it was timed. That check also confirmed the FP16 `mma.sp` metadata layout the kernel assumes. The pure in-order read on this unit was 220–224 GB/s. Two runs give the median of each matrix's fastest tiling.

16 rows, ms (GB/s):

| Matrix (N × K) | fp4 | fp8 | b3 | sp4 | sp8 |
|---|---|---|---|---|---|
| FFN gate+up, 34816 × 5120 | 0.490 (205) | 0.842 (212) | 0.369 (211) | 0.371 (210) | 0.518 (215) |
| FFN down, 5120 × 17408 | 0.232 (216) | 0.420 (212) | 0.189 (206) | 0.190 (205) | 0.259 (215) |
| DeltaNet in, 16384 × 5120 | 0.226 (209) | 0.395 (212) | 0.185 (198) | 0.171 (214) | 0.251 (209) |
| DeltaNet out, 5120 × 6144 | 0.086 (205) | 0.151 (209) | 0.068 (202) | 0.066 (207) | 0.090 (218) |
| Attention q, 12288 × 5120 | 0.170 (208) | 0.306 (206) | 0.132 (209) | 0.131 (211) | 0.197 (200) |
| Attention k+v, 2048 × 5120 | 0.034 (176) | 0.050 (211) | 0.028 (164) | 0.025 (181) | 0.035 (190) |
| Attention o, 5120 × 6144 | 0.086 (206) | 0.148 (213) | 0.070 (196) | 0.070 (198) | 0.092 (214) |
| LM head, 248320 × 5120 | 3.397 (211) | 5.745 (221) | 2.615 (213) | 2.523 (220) | 3.698 (215) |

One verify pass composed from these timings (`summarize.py`, 16 rows; the matrices counted per pass as in MODEL §5.1):

| Map | GB | ms | GB/s | Δ ms |
|---|---|---|---|---|
| Today (FFN fp4, projections fp8) | 17.56 | 83.84 | 209 | |
| All-NVFP4 | 14.40 | 69.22 | 208 | −14.63 |
| 3-bit FFN, projections fp8 | 15.42 | 73.40 | 210 | −10.44 |
| 3-bit FFN, projections fp4 | 12.26 | 58.77 | 209 | −25.07 |
| 3-bit FFN and projections | 11.36 | 54.99 | 207 | −28.85 |
| 2:4 NVFP4 FFN, projections fp4 | 12.26 | 58.92 | 208 | −24.93 |
| FFN fp4, 2:4 FP8 projections | 14.85 | 71.15 | 209 | −12.69 |

At 1 row the rates are the same. At 32 rows every format loses 5–15% of its rate, and the small formats a little more (the activations' L2 traffic is the same while the weight bytes shrink): today's map 88.70 ms, all-NVFP4 −13.32, 3-bit FFN with NVFP4 projections −23.87.

**Reading.**
- Fewer bytes become proportionally less time for both the 3-bit and the 2:4 formats. Their decode (about 2 integer operations a value for `b3`) hides under the memory stream at 1–32 rows.
- This is the opposite of item 30's table-driven entropy decode, which lost about 3 ms a step to its decode. The difference is the absence of a shared-memory table and of variable-length codes.
- 2:4 NVFP4 and dense 3-bit cost the same bytes and the same time. Their only difference is quality (section 3).
- One kernel layout fix mattered. In window 1 the FP8 kernel loaded each lane's 64 bytes with a 64-byte stride and reached 113 GB/s. Storing 16-byte pieces piece-major (one warp load = 512 contiguous bytes) gave 212 GB/s in window 2. The window-1 file is kept as `w1_lowbit_gemm_interleaved_fp8.csv`.

**(3) The library path: cuSPARSELt 0.8.1 through PyTorch 2.14 (CUDA 13.0), `sparse_lib.py`.**
- **2:4 FP16 runs on sm_121, but slowly.** At 16 rows it takes 4.16 ms against dense cuBLAS FP16's 1.77 ms for gate|up (48 against 202 GB/s), and 3.85 against 0.84 ms for down. At 32 rows the result is the same.
- **2:4 FP8 is refused.** `cusparseLtMatmulDescriptorInit` returns "operation not supported" at the layouts PyTorch's wrapper requests. Other layouts were not tried.
- **Dense FP8 is fine.** `torch._scaled_mm` streams at 199–226 GB/s.

cuSPARSELt's release notes list SM 12.1 support from 0.8.0, and its FP4 sparse path uses blocks of 32 (`VEC32_UE4M3`), not NVFP4's 16. Neither matters: a library GEMM tuned for large M is the wrong tool at 16 rows. Any 2:4 path would have to be the engine's own kernel, as (2) shows it can be.

## 7. The quality protocol, the bar, and the order

**Why the bar must change.** Owner's brief: "≤ 2 points on 50 tasks with a paired CI". That is below what 50 tasks can resolve.
- **The evidence:** night 13 ran the *same* configuration three times on the same 50 tasks. It resolved 22, 25 and 26. Between runs 1 and 2, 17 tasks were discordant (10 against 7), giving a paired 95% interval of −10.1 to +22.1 points (`~/.local/share/dreamference/swe-bench/night13/default-2-against-default-1.txt`).
- **What 2 points would take:** with that discordance (34% of tasks), a ±2-point interval needs about 0.34 × (1.96 / 0.02)² ≈ 3,300 paired task-runs. Three paired runs per arm on 50 tasks narrow it only to about ±9 points.
- **Where sensitivity comes from instead:** a SWE-bench night can catch only a large loss. The sensitivity has to come from token-level comparisons on agent traces, which see millions of positions.

**The ladder, cheapest first; each rung must pass before the next is paid for:**

1. **Acceptance (exact, about 1 GPU-hour).** A paired-by-request A/B with the current drafter on the replay set (section 4). This is the speed half: it says whether the candidate's byte gain survives as tokens/s. A candidate whose tokens/s gain is below half its byte gain goes to the drafter-retraining question before anything else.
2. **Teacher-forced divergence on agent traces (a few GPU-hours).** Over at least 1M tokens of recorded agent sessions (the 168-session replay set plus agentlong's turns; positions to 64K), compare against BF16 Qwen3.8-27B:
   - the candidate's KL(BF16 ‖ candidate) and top-1 agreement;
   - the same two for the served NVFP4 checkpoint, which is the scale.
   The tools: `bench/agreement/` (`ling_force.cpp`, `mode_kl.py`, `analyze.py`) for engine-served candidates, and SGLang's log-probs (`prod.py`) for those only SGLang loads yet. A 3-bit candidate needs no engine kernel for this rung: its weights are dequantized to BF16 and run in a PyTorch forward, which fits in the 128 GB.
   - **Pass if:**
     - the candidate's mean KL is ≤ 1.25× the served checkpoint's;
     - its top-1 agreement with BF16 is within 0.5 points of the served checkpoint's;
     - KL does not grow with position (25K and 64K buckets);
     - on the tool-call slice (positions inside tool-call arguments), top-1 agreement is no worse than the served checkpoint's.
   - **Why 1.25×:** Minima's perplexity is 4–9% above the served recipe's with level benchmark scores. A 3-bit model at about 3× the 4-bit KL (the Qwen3.6-27B figure, section 3) fails this rung quickly, which is the point.
   - The exactness policy's near-exact line (KL ≤ 0.0008 against the exact engine, top-5 100%, [MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md)) is for modes that should not change the model. Every format change fails it trivially, so this study's candidates come under the rung above instead. Changing the policy for them is the owner's decision.
3. **Standard benchmarks**, as [VALIDATION §13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) states them: LiveCodeBench v6, GPQA-Diamond, IFBench, a perplexity mix, and the 32K/128K needle checks for (c). Each is paired by item against the served checkpoint, with several samples per item so the paired interval is narrow enough to read. The interval is measured from a same-arm repeat before the bar is applied, as night 13 did for SWE-bench.
4. **SWE-bench, as a guard, not a ruler.** At least three paired nights on night 12's 50 tasks (150 pairs; the engine serving both arms where it can load the candidate, otherwise SGLang serving both). Pass if:
   - the pooled McNemar test shows no significant loss;
   - the point estimate is no worse than −2 points;
   - the lower bound of the 95% interval is above −10 points.
   This replaces "≤ 2 points on 50 tasks" with something the data can decide. The owner may prefer more nights.

**The order to try the candidates** (cheapest to evaluate first):

1. **(a1) All-NVFP4 projections (Minima) and (c) FP8 KV.** Both are the levers MILESTONES §14 already queued. Minima is servable on SGLang with the drafter as it stands, so rungs 1, 2 and 4 can run before any engine change; night 2's script exists (`scripts/swe_bench_night2.sh`). SGLang serves it W4A4, so its quality there understates the engine's W4A16 serving. FP8 KV needs the engine build first (section 5), then rungs 2–3.
2. **(a1′) FP6 projections,** only if (a1) fails rung 2 or 4.
3. **(a2) 3-bit FFN,** fake-quantized in BF16 for rung 2 before any kernel. Two quantizers, plain codebook-GPTQ first and a QTIP-class one second, with the projections at NVFP4 (the stack that reaches ×1.29–1.39). Healing by distillation only after the owner decides on cloud GPUs.
4. **3-bit projections and head:** only after (a2) passes, since the projections are what Minima's authors kept at 4 bits.

Not tried: (b) 2:4, (a3) MXFP4, (d) layer skipping.

## 8. Recommendation

**Go, for one candidate at a time, behind the switch and gate of the exactness policy; no change to the shipped model without the owner's OK.**

- **First to build: (a1), the all-NVFP4 projections from the Minima checkpoint**, with the FP8 KV cache (c) beside it.
  - **Expected:** ×1.16 alone and ×1.24 with FP8 KV at agentlong's 41K context (more at longer contexts), if acceptance holds.
  - **Quality:** published as level with the served recipe on five benchmarks; perplexity is the open question.
  - **Effort:** 3–5 days of loader and kernel work plus 1–2 days for the KV.
  - **Before the engine work:** the cheap rungs (the acceptance A/B on SGLang and the KL on agent traces) run first, and SGLang can serve the checkpoint today.
- **1.3× needs (a2) on top** (×1.29 with (a1), ×1.39 with (a1) and (c)). The kernel side is shown to work: the 3-bit stream runs at the bus rate. The quality side is the open risk: the literature says plain post-training 3-bit loses 4–7 points on code for 32B Qwen models, so it enters only through rung 2.
- **1.6× is out of reach of weight formats at one stream.** It would take about 40 ms off a 106 ms step, more than 3-bit everything plus FP8 KV gives (35 ms). Batches of three and four (BACKLOG §19 item 31) and more tokens per step are the levers for the rest.
- **No-go on 2:4 sparsity:**
  - it costs the same bytes as dense 3-bit;
  - it loses far more on the only Qwen 32B data;
  - recovery is thousands of GPU-hours;
  - cuSPARSELt is slower than dense at our shapes;
  - the serving tools have dropped it.
- **No-go on MXFP4 and on layer skipping.**

## Sources

- Measured here: `bench/lowbit/` and `bench/lowbit/results-2026-10-26/` (third-puffin, 2026-10-26).
- This repository: BACKLOG §19 items 19, 20, 22, 26, 27, 29, 30, 33, 34 and 35; [MODEL §5](./DREAMFERENCE_LING_ENGINE_MODEL.md); [RESEARCH §16.2, §16.10, §16.13](./DREAMFERENCE_LING_ENGINE_RESEARCH.md); [MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md); [VALIDATION §13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md); `src/core/stream_gemm.cu`, `src/core/model.cu`; `bench/agreement/`, `bench/quant/`, `bench/agentlong/manifest.json`.
- Mightling: `specs/DREAMFERENCE_MODELS.md` §2.2 (the Minima candidate, night 2); SWE-bench nights 12 and 13 (`~/.local/share/dreamference/swe-bench/night12/`, `night13/`).
- Minima, "Why Gated DeltaNet Survives 4-Bit Quantization": [arXiv 2609.04098](https://arxiv.org/abs/2609.04098).
- Reasoning models at low bits (QwQ-32B, R1-Distill-Qwen-32B): [arXiv 2504.04823](https://arxiv.org/abs/2504.04823).
- An Empirical Study of Qwen3 Quantization: [arXiv 2505.02214](https://arxiv.org/abs/2505.02214).
- QTIP: [arXiv 2406.11235](https://arxiv.org/abs/2406.11235). SliM-LLM: [arXiv 2405.14917](https://arxiv.org/abs/2405.14917). EfficientQAT: [arXiv 2407.11062](https://arxiv.org/abs/2407.11062).
- NVFP4 against MXFP4 recovery: [arXiv 2509.23202](https://arxiv.org/abs/2509.23202).
- 2:4 on Qwen3-32B: [arXiv 2606.10445](https://arxiv.org/abs/2606.10445). Wanda: [arXiv 2306.11695](https://arxiv.org/abs/2306.11695). SparseGPT: [arXiv 2301.00774](https://arxiv.org/abs/2301.00774). MaskLLM: [arXiv 2409.17481](https://arxiv.org/abs/2409.17481). Pruning against quantization: [arXiv 2307.02973](https://arxiv.org/abs/2307.02973).
- Sparse Llama 3.1 8B 2:4: [model card](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-2of4).
- Speculative decoding with quantized targets: [arXiv 2505.22179](https://arxiv.org/abs/2505.22179); DFlash on an INT4 target: [arXiv 2607.04244](https://arxiv.org/abs/2607.04244); QSpec: [arXiv 2410.11305](https://arxiv.org/abs/2410.11305).
- FP8 KV cache on Qwen3.5-27B: the vLLM project blog, 2026-04-22.
- PTX ISA 9.4 (`mma`, `mma.sp` element types and targets): [NVIDIA docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html). cuSPARSELt release notes: [NVIDIA docs](https://docs.nvidia.com/cuda/cusparselt/release_notes.html).
- Tooling versions as of 2026-10-26: llm-compressor 0.14.0.1 (and main after #3377), ModelOpt 0.48.0rc1, GPTQModel 7.7.0, vLLM PR #36799.
