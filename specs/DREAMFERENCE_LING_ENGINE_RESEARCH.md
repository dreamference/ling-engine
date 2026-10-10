# ling-engine: New directions from the literature, and the sources

Part of the ling-engine specification; the index is [README.md](./README.md). Section numbers are those of the single-file `SPEC.md` this was split from (Oct 8, 2026 · revised the same day against Mightling's measured workload), so a citation of the form `§n` or `S§n` in the reports and the code still names the same text.

## 16. New directions from the literature: low-level speed-ups

A survey of about 60 papers, mostly 2025–2026 arXiv, on what could make this engine faster than sections 7–11 already plan. It concentrates on kernels, the memory system and numerics. Each direction below states:
- what it is;
- what it is worth here, in bytes or milliseconds per step at the workload's 25.7k-token median;
- whether it is exact;
- where it would land.

Two measurements were made for this section on the served checkpoint (`RadixArk/Qwen3.8-27B-NVFP4`, 9 of 64 layers read in full): the entropy of each weight format, and the scales inside each fused group. They are marked *measured*. Everything else is the papers' numbers, on their hardware.

### 16.1 Five measurements M0 should add

1. **Bytes in flight.** Measure GPU DRAM latency with a pointer chase. Bandwidth × latency is the number of bytes every weight-streaming kernel must keep outstanding. At 273 GB/s and ~1 µs, that is ~270 KB across the chip, ~6 KB per SM on 48 SMs. Pipeline depth (TMA or `cp.async` stages) is sized from it, not tuned by trial.

   *Measured 2026-10-10* (M0's pointer chase re-run with the reference server resident but idle; [reports/micro-2026-10-10.md](../reports/micro-2026-10-10.md) §1): 395–402 ns to DRAM, 136 ns in L2, unchanged from M0's idle machine. At the measured 257 GB/s: ~103 KB in flight across the chip, ~2.1 KB per SM on 48 SMs, ~13 KB per SM if 8 SMs stream.
2. **How many SMs saturate the bus.** Stream with 8, 16, 24, 32 and 48 SMs. If about half of them reach the measured peak, then:
   - wave quantization stops mattering for weight streams (16.8);
   - the remaining SMs can run the non-streaming work (the DeltaNet scan, sampling) at the same time (16.6).

   *Measured 2026-10-10* (reference server resident but idle; micro-2026-10-10.md §2): 4 SMs 248 GB/s, 8 SMs 257, 12–24 SMs 257–258, 48 SMs 255–259; the sweep's best read 258.1 GB/s (94.6% of 273) and 257.0 sustained for 15 s, against M0's 262.8 on a bare GPU. Eight SMs reach 99% of the 48-SM figure: both consequences above hold.
3. **The ISA sm_121 exposes:**
   - warp-level block-scaled `mma.sync` for NVFP4 and FP8;
   - TMA;
   - thread-block clusters with distributed shared memory;
   - programmatic dependent launch.

   sm_100's `tcgen05`/TMEM path is not expected. Every kernel in [section 9](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) assumes warp-level MMA, and results that depend on TMEM (MpFA's attention, CuTile's B200 numbers) do not transfer. CuTile's attention reached 53% of FlashAttention-2 on sm_120.

   *Measured 2026-10-10* (CUDA 13.0.88, `bench/membw/isa_probe.cu` and `bench/micro/tcgen05_probe.cu`; micro-2026-10-10.md §3): the four features above all run and check out (clusters up to 8 blocks; PDL's secondary prologue starts while the primary runs). `tcgen05.alloc` is refused by ptxas for `sm_121a` and accepted for `sm_100a`, so TMEM is absent as a compiled fact. `-arch=sm_121a` still emits `compute_121` PTX that rejects the block-scaled MMAs; `-gencode arch=compute_121a,code=sm_121a` is required.
4. **Clocks, power and temperature** across a 10-minute decode and a long prefill. A short paper on the DGX Spark found that alternating compute-heavy and memory-heavy phases at a finer grain avoids throttling, worth up to 2%.

   *Measured 2026-10-10* (ten minutes of two-stream decode on the reference server, `bench/micro/decode_clocks.py`; micro-2026-10-10.md §4): SM clock 2,405–2,476 MHz (median 2,431, never below base), power 44–45 W, temperature 68–69 °C, throttle-reason mask 0x0 in all 709 samples. No throttling at a decode load; the long-prefill case is still unmeasured (M0's replays, with 25K-token prompts, peaked at 82 W and 79 °C, also without throttling).
5. **DFlash2's acceptance histogram,** not only its mean. If many steps accept all 16 tokens (the ceiling bin), the drafter's block length is leaving speed unused (16.12).

   *Measured* in M0 §4 (production, 16 draft tokens, 1,295 steps of the tuning set): ceiling bin 3.6%, 43% of steps accept 0–2 drafted tokens; M1 §4 measured ling-serve's at the same block (ceiling 3.6%). Not re-measured on 2026-10-10: it needs ling-serve's model loaded, which was not done beside the resident reference server (micro-2026-10-10.md §5). Production and ling-serve run 12 draft tokens since 2026-10-09; the 12-draft histogram on the tuning set is still to be taken.

### 16.2 An all-NVFP4 checkpoint already exists and is validated (−3.2 GB per pass)

[Section 5.3](./DREAMFERENCE_LING_ENGINE_MODEL.md) treats moving the FP8 projections to NVFP4 as work for the quality gate. That work has been published for this exact model.
- **The checkpoint:** "Minima" quantizes all 496 linear layers of Qwen3.8-27B to NVFP4 W4A4, the Gated DeltaNet gates included: 17.5 GiB.
- **Quality:** it matches BF16 within seed noise on MMLU-Pro, GSM8K, AIME'25, GPQA-Diamond, LiveCodeBench and RULER to 64K, with a 5-task average −0.52. It is the fastest at prefill of the recipes compared (the paper's +14–19%). Those recipes include the checkpoint Mightling serves today.
- **Why it works:** NVFP4's 16-value blocks localize the residual stream's outliers. The "fragile" gate projections are the least sensitive, because softplus and sigmoid compress their error. The delta rule overwrites state along the current key, so injected noise stays flat over 32K tokens instead of compounding.

Two consequences for this engine:
- **Fewer bytes did not become speed at batch 1 in today's kernels.** On one RTX PRO 6000 at concurrency 1, Minima decoded 47 tokens/s against 51 for the served recipe, despite reading fewer bytes. That gap belongs to the kernels, and closing it is this engine's job.
- **The plan changes.** The quality gate of [section 13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) is run on this checkpoint first, instead of re-deriving the recipe. Its result decides the "NVFP4 projections" column of 5.2. The DFlash2 drafter conditions on the target's hidden states, so acceptance is remeasured with this target in M2.

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
- **Calibration point:** 10.5 tokens/s for plain decoding is ~95 ms per single-row pass, ~190–200 GB/s effective with the KV read. That is generic kernels at ~70% of peak, as [section 4](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) assumed.
- **Design change:** [section 9](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md)'s chain replay (keep S₀, replay the j accepted tokens) is the chain case of these methods. For a tree (DFlash2's alternatives plus the lookup branch), `gdn_scan` should use the TreeWY or Bole closed form, so its cost and transient memory do not grow with the branch count. This resolves the "per-branch replays exceed the L2 budget" risk in [section 15](./DREAMFERENCE_LING_ENGINE_MILESTONES.md).

### 16.7 One persistent kernel per step

[Section 9](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) runs 333 fused kernels inside one CUDA graph. Megakernel work goes further and runs the whole step as one persistent kernel that schedules tiles itself:
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
- **Load each KV tile once for every node of the tree:** DeFT's KV-guided grouping (73–99% less KV IO) and FastTree. [Section 9](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md)'s `attn` already batches 6M query rows per KV head; the tree mask must not split those loads per branch.
- **A 4-bit KV option for long sessions** ([section 11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md), M5) now has a kernel recipe: BitDecoding decodes NVFP4 KV on Blackwell tensor cores, up to 8.6× faster than FP16 flash-decoding. MpFA (NVFP4 for QK, FP8 for PV) is the prefill counterpart, though its fastest path assumes TMEM.

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
- **For [section 10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md)'s agent-tuned drafter:** a block-verification-aware loss (BV loss, +13–21% accepted tokens) instead of cross-entropy.
- **DScale:** variable verify length with fixed-address workspaces, so captured CUDA graphs survive changing prefixes. This engine has the same problem.
- **SuffixDecoding:** a suffix tree over all earlier sessions' outputs, with adaptive speculation length, up to 5.3× on SWE-bench agent traces. It answers [section 15](./DREAMFERENCE_LING_ENGINE_MILESTONES.md)'s open question about a cross-session store: worth building into the lookup matcher.
- **FR-Spec, VocabTrim and SlimSpec:** precedents for [section 10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md)'s 32k-token drafter head.

### 16.13 Considered and not pursued

- **Activation sparsity** (TEAL-style, and the byte-crossover analysis): it trims projection bytes for one row. But a 16–48-row verify must read the union of every row's active channels, so the saving disappears exactly where this engine runs.
- **Approximate prefix reuse for hybrids** (Tail-Replay, SuffixReplay: rebuild the recurrent state by replaying a recent suffix, 91–100% of full quality): not exact, so opt-in at most. Message-boundary checkpoints stay the default. Marconi's admission policy (FLOPs saved per byte stored) is adopted for the checkpoint store's eviction.
- **Self-speculation from the DeltaNet subgraph:** acceptance 0.038 on sequential hybrids like Qwen3.5.
- **`tcgen05`, TMEM and 2-SM MMA:** sm_100 features, absent on sm_121 (16.1(3)).

### 16.14 What the survey changes in the numbers

| Engine column of 5.2 (25.7k context) | Bytes per step | At 245 GB/s |
| --- | --- | --- |
| NVFP4 projections ([section 5.2](./DREAMFERENCE_LING_ENGINE_MODEL.md)) | 17.3 GB | 71 ms |
| + all-NVFP4 checkpoint validated (16.2) | 17.3 GB, now without the quality-gate risk | 71 ms |
| + 5-bit scale codes (16.4) | ~16.7 GB | ~68 ms |
| + fused head and top-k (16.5) | ~16.6 GB | ~68 ms, and production's sampling time is gone |
| + one persistent kernel (16.7) | unchanged | 3–6% less time than the graph path |

None of this moves [section 7](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)'s targets until M0 has measured the bus (16.1). The table only shows that the step-time target (76–90 ms) has room under it, and that the biggest risk in 5.3 has been answered from outside.

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
- Measured on this machine, 8 October 2026: the served checkpoints' configs and safetensors headers; production SGLang's launch flags, Prometheus counters and decode log; and 168 recorded agent sessions from Mightling's SWE-bench rounds of 3–7 October, analysed with the scripts in [`bench/`](../bench/) (`workload.py`, `lookup_sim.py`).
