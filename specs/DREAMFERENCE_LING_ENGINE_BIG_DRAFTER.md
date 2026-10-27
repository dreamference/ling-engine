# ling-engine: A bigger drafter?

Part of the ling-engine specification; the index is [README.md](./README.md). Written on 2026-10-26/27 at the owner's request of 2026-10-26: would a bigger speculative drafter speed up ling-engine? Research and cost measurements only. Nothing was trained; the GPU numbers come from dummy weights. This document has no section number in the former `SPEC.md`. It builds on [SPECULATION §10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md), [BACKLOG §19](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) items 20, 26, 33 and 34, and the memo on training the drafter on agent sessions (`dgxcoder-memos/drafter-agent-sessions.md`, outside this repository).

**The answer: no-go.** A bigger drafter is not worth building now.

- **Cost.** Doubling the drafter in NVFP4 adds 4.0-4.8 ms to the 104 ms step (measured, §3). That is a wash unless acceptance rises by 3.7-4.6% per doubling.
- **Gain.** The literature gives 3-10% per doubling past 5 layers, with the evidence nearest this setup at the low end (§4). The expected result is 65.5 → about 65.7-66.3 tokens/s greedy, with a range of 64.5-69.6 (§5).
- **Four times bigger** loses at the middle of the range.
- **A BF16 drafter** loses 16% even at today's size.
- **Training.** No bigger trained drafter exists for this target (§2). Training one from scratch takes months on one GB10, or $3K-13K of rented GPUs (§6).
- **What to do instead.** The same training effort spent on today's size (the memo's agent-session fine-tune: +5-15% at no step cost) is the better lever.

## 1. Where the drafter stands

- **The drafter.** DFlash2 (`incoai/Qwen3.8-27B-DFlash2`, BF16, Apache-2.0), served as an NVFP4 re-quantization of about 1.5 GB. It has 5 non-causal sliding-window attention layers at the target's hidden size (5,120), an MLP of width 17,408, the grouped convolutions around each sublayer, the fc over the target's layers 5, 19, 33, 47 and 61, and the candidate selector (rank-256 codebooks). It uses the target's embedding and LM head. The layer matrices are 1.60B parameters; the whole file is 1.92B.
- **What it costs.** On agentlong at one stream, greedy:
  - the step is about 104 ms;
  - the draft is about 8.2 ms of it with the sparse head (8.9 at its merge, 9.6-10.3 with the full head);
  - the step accepts about 6.7 tokens (the chain of 12 drafts plus the 16-row tree);
  - the result is about 65.5 tokens/s. agentlong's tokens/s and its per-step figures are sums over different spans of the turns, so 6.7 / 0.104 s is not exactly 65.5; the break-even below uses only ratios.

  Sampled, the step accepts 5.5-6.2 tokens.
- **Where the draft's milliseconds go.** By bytes:
  - the five layers' NVFP4 matrices: 0.94 GB, about 4.6 ms (0.92 ms per layer, measured in §3);
  - the sparse head's gathered rows: about 0.6 ms;
  - everything else (attention over the 2,048-key window, the convolutions, norms, the selector, top-k and the kernel boundaries of ~75 launches): about 3 ms.

  Only the first part and the per-layer share of the third grow with the drafter.
- **Which axes can grow.**
  - The hidden size is pinned at 5,120: the drafter embeds with the target's table, scores with the target's head, and its fc must map 5 × 5,120 features to its own width. A different width would need projections in and out, which changes the architecture.
  - What can grow without new code is depth (`num_hidden_layers`) and the MLP's width (`intermediate_size`). The attention heads could grow as well (the loader requires head dimension 128 and 4 query heads per KV head), but attention is only 16% of a layer's matrices.

## 2. Bigger public drafters (question 1)

**No bigger trained drafter exists for Qwen3.8-27B** (searched on Hugging Face and arXiv on 2026-10-26). Every trained drafter for this target has 5 layers or fewer. The candidates that look bigger are not:

| Repository | Shape | Kind | Reported acceptance (tokens per step) | Licence | Loads here? |
|---|---|---|---|---|---|
| `incoai/Qwen3.8-27B-DFlash2` (ours) | 1.92B, 5 layers, I 17,408 | block diffusion on target features | sampled, 7 drafts, H200: HumanEval 4.39, MBPP 4.79, GSM8K 5.46 | Apache-2.0 | yes (NVFP4 re-quant; BF16 with this document's patch, §3) |
| `RedHatAI/Qwen3.8-27B-speculator.dspark` | 5 layers, 8 tapped target layers, rank-256 Markov head | DSpark | HumanEval 4.42, tool calls 3.76 (temperature not stated) | Apache-2.0 | no: 8 taps and DSpark's head need loader and kernel changes; not bigger anyway |
| `RadixArk/Qwen3.8-27B-DSpark` | 1.86B, 5 layers | DSpark | HumanEval 3.85 (v2) against 3.04 (v1) | "other" | no, and not bigger |
| `tiyuvta/Qwen3.8-27B-DSpark-Agentic` | 1.36B, 5 layers, I 10,240 (*smaller*) | DSpark on agent sessions | real agent turns: 2.88-2.92 at temperature 0.6, 2.41-2.78 greedy | Apache-2.0 | no (DSpark) |
| `EAGLE-Corp/Qwen3.8-27B-EAGLE4` | "3B", but 1 transformer layer (0.44B) plus its own embedding and head | autoregressive | about 4.75 greedy and 3.60 at temperature 1, 4×B300 | source-available, revenue threshold, no competing offerings | no: autoregressive (one pass per draft token, the wrong shape for this bus, [SPECULATION §10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md)), and the licence |
| `RelaxingSnorlax/Qwen3.8-27B-DSpark-BF16-15L`, `-30L` | 5B, 10B | **untrained** dummies made by copying layers | none | — | timing only, as this document's own dummies |
| Qwen3.8-27B's own MTP module | 1 layer | autoregressive MTP | sampled, 7 drafts: mean 4.28 | the model's | the fallback drafter of §10, not built |

Close family members with other targets cannot be used: their features come from another model's layers. These are `z-lab/Qwen3.5-27B-DFlash` (6 layers, 8 taps; greedy HumanEval 6.18 at block 8, 8.90 at block 16), `z-lab/Qwen3.6-27B-DFlash` (5 layers) and `z-lab/Qwen3-Coder-Next-DFlash` (8 layers at hidden 2,048).

**If a bigger DFlash2 for this target appears**, testing it takes about a day:

1. **Check that it loads.** `DraftConfig::from_file` refuses anything but head dimension 128, 2 conv taps, selector rank 256 and top-k 16, 4 query heads per KV head, all-`sliding_attention` layers, a non-causal config and the exact tensor names. Depth and MLP width are free (§3 ran 20 layers and width 79,872). A BF16 checkpoint already loads on branch `bench/big-drafter`. Main needs NVFP4 projections with their `weight_scale_2` and `input_scale` scalars.
2. **Re-quantize to NVFP4.** Round to nearest in ModelOpt's layout, as the served RTNcal file was made (a per-tensor global scale amax / (6 · 448), E4M3 scales per 16 values, activation scales from a calibration over on-policy conversations). Then run `bench/quant/scalesearch.py BF16_DIR NVFP4_DIR OUT_DIR` for the per-block scale search (BACKLOG §19.22).
3. **Run `ling-run --spec-check` and `--graph-check`.** They prove exactness. They pass with any drafter by construction (greedy accepts only the target's argmax), so they say nothing about acceptance.
4. **Run agentlong at one stream.** `bench/agentlong/arms.sh` with the candidate against the served drafter in the same window: greedy plus seeds 1013 and 2026, compared per turn with `bench/drafter/paired.py`. Ship rule as in the memo: at least +5% tokens/s held out, no class 3% worse, exactness gates passing.

## 3. The drafter's cost against its size, measured (question 2)

**Method.**
- **Dummy drafters.** `bench/drafter/big_dummy.py` writes a bigger drafter from the real checkpoint's bytes: layer i is a copy of layer i mod 5, and `gate_proj`/`up_proj` rows and `down_proj` columns are repeated up to the new width. The values stay finite, every kernel runs on real data, and the format is the one served. Acceptance collapses to about one token per step. The draft's cost does not depend on what it proposes: it always runs a block of 12 rows, and attention reaches back at most 2,048 keys.
- **The engine.** `ling-serve` from branch `bench/big-drafter` (main `c5ec4ec` plus a measurement-only patch: BF16 projections on the BF16 rows path, and an FP16 activation buffer wide enough for a down projection wider than the target's) runs with production's flags: 12 drafts, step graphs, PDL, the tree of 16 rows, the sparse head.
- **The measurement.** For each drafter: one warm-up request, then one greedy request of 320 tokens over a ~10K-token prompt. Draft, verify and commit milliseconds per step come from `/spec_stats`, as the difference across the measured request.
- **BF16 arms** also read the fc and conv projections in BF16 (`--draft-bf16`), so the whole drafter is BF16.
- **The window.** `bench/drafter/big_window.sh`, one hold of the GPU lock on node-b, 2026-10-27 01:25-01:44 UTC (19 minutes), with m0-prod8000 stopped (left down for the six jobs waiting); results in `~/bd/w1/`.

| Arm | Precision | Layers | MLP width | Layer matrices (× today) | Drafter in memory | Draft ms/step | Commit ms/step | Verify ms/step | Δ step against today |
|---|---|---|---|---|---|---|---|---|---|
| rtn | NVFP4 | 5 | 17,408 | 1.00 | 1.61 GB | 9.51 | 0.84 | 85.2 | -0.01 ms |
| f4-L5 | NVFP4 | 5 | 17,408 | 1.00 | 1.61 GB | 9.51 | 0.85 | 85.5 | +0.00 ms |
| f4-L8 | NVFP4 | 8 | 17,408 | 1.60 | 2.22 GB | 12.12 | 1.00 | 84.8 | +2.76 ms |
| f4-L10 | NVFP4 | 10 | 17,408 | 2.00 | 2.63 GB | 14.03 | 1.16 | 85.1 | +4.83 ms |
| f4-L20 | NVFP4 | 20 | 17,408 | 4.00 | 4.66 GB | 23.23 | 1.90 | 85.4 | +14.77 ms |
| f4-W1.5 | NVFP4 | 5 | 27,648 | 1.49 | 2.09 GB | 11.39 | 0.85 | 84.5 | +1.88 ms |
| f4-W2 | NVFP4 | 5 | 38,912 | 2.03 | 2.62 GB | 13.48 | 0.84 | 85.0 | +3.97 ms |
| f4-W4 | NVFP4 | 5 | 79,872 | 4.00 | 4.54 GB | 21.07 | 0.79 | 85.1 | +11.50 ms |
| f4-L10W2 | NVFP4 | 10 | 38,912 | 4.07 | 4.64 GB | 21.96 | 1.15 | 84.8 | +12.74 ms |
| bf-L5 | BF16 | 5 | 17,408 | 1.00 | 3.85 GB | 28.33 | 2.21 | 86.7 | +20.18 ms |
| bf-L8 | BF16 | 8 | 17,408 | 1.60 | 5.85 GB | 43.17 | 2.47 | 86.0 | +35.27 ms |
| bf-L10 | BF16 | 10 | 17,408 | 2.00 | 7.18 GB | 52.98 | 2.82 | 86.8 | +45.44 ms |
| bf-L20 | BF16 | 20 | 17,408 | 4.00 | 13.84 GB | 100.92 | 4.47 | 86.1 | +95.03 ms |
| bf-W2 | BF16 | 5 | 38,912 | 2.03 | 7.15 GB | 51.44 | 2.17 | 86.4 | +43.26 ms |
| bf-W4 | BF16 | 5 | 79,872 | 4.00 | 13.44 GB | 133.96 | 2.33 | 85.0 | +125.92 ms |

Reading the table:

- **The dummy reproduces the real drafter.** At today's shape the dummy (`f4-L5`) costs exactly what the served RTNcal file (`rtn`) costs: 9.51 against 9.51 ms of draft. The tiling changes nothing the kernels see. The verify stays at 84.5-86.8 ms in every arm, so only the drafter moved.
- **These arms ran with the full head.** The sparse head's data lives in the served drafter's own directory (`sparse-head/`), and these directories have none. 9.51 ms is therefore the full-head draft, matching agentlong's 9.6-10.3. The sparse head saves the same ~1.3-1.4 ms at every size, so the differences in the last column are what a bigger drafter would add to production's 8.2 ms.
- **Depth costs 0.92 ms of draft per layer** (5 → 8 → 10 → 20 layers: 9.51 → 12.12 → 14.03 → 23.23 ms), plus 0.07 ms of commit (each layer's context K and V). A layer's NVFP4 matrices are 187 MB, which is 0.83 ms at the measured ~225 GB/s, so a layer costs its bytes plus ~0.1 ms of attention, norms and kernel boundaries.
- **Width costs only its bytes.** Doubling the MLP (I 38,912) adds 3.97 ms, about 0.93 GB at 234 GB/s; the commit does not grow. A doubling by width (+4.0 ms) is about 0.8 ms cheaper than a doubling by depth (+4.8).
- **Cost per doubling, NVFP4:** +4.0 to +4.8 ms, or 3.8-4.6% of the step. Four times today's matrices: +11.5 to +14.8 ms (11-14%).
- **BF16 costs three times NVFP4 per byte.** The BF16 rows path reads 666 MB per layer at ~135 GB/s, against the NVFP4 stream's ~225. Today's drafter in BF16 adds 20.2 ms (−16% tokens/s before any acceptance change). The served drafter stays NVFP4 whatever its size.
- **Memory is not the constraint.** A 4× drafter takes 4.5-4.7 GB in NVFP4 (13-14 GB in BF16), against today's 1.6 GB. The ~3 GB difference comes out of the KV and prefix stores' budget ([INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)).
- **Commit and prefill.** The drafter's context K and V are projected for every prompt token in the prefill too: at 20 layers that projection is four times today's, a few percent of a long prefill. Not measured here (decode tokens/s excludes the prefill).

## 4. Acceptance against size, from the literature (question 3)

What one doubling of drafter parameters buys, greedy unless stated:

| Source | Change | Acceptance | Per doubling |
|---|---|---|---|
| DFlash paper, Table 6 ([arXiv 2602.06036](https://arxiv.org/html/2602.06036v2)); block 16, 100K training samples | 3 → 5 → 8 layers | HumanEval 4.61 → 4.94 → 5.29; Math500 5.64 → 5.99 → 6.33; MT-Bench 3.18 → 3.37 → 3.50 | +0.45-0.52 on code and math (+9-10%), +0.2 on chat (+6%). **Speedup peaks at 5 layers** on H200: 8 against 5 layers is 4.64× against 4.71× on Math, 3.96× both on HumanEval, 2.23× against 2.35× on MT-Bench |
| DFlash2 blog ([inco.ai](https://inco.ai/blog/dflash2/)); Qwen3-4B, GSM8K, derived from the per-position recall table (the method reproduces the blog's own 4.27 at 5 layers) | 3 → 5 → 15 layers | 4.08 → 4.27 → 4.49 | 5 → 15 (3×): +0.14 (+3.3%), at +15.2% cycle time on H200. In the same table, the convolutions gave +0.13 for +3% parameters, and the selector +0.34 greedy and +0.47 sampled |
| Scylla ([arXiv 2505.07858](https://arxiv.org/html/2505.07858)); EAGLE-style drafters of 138-413M parameters, fitted law τ = 0.74 · log10(layers) + 4.61 | depth | — | +0.22 (+4.5%) |
| P-EAGLE ([arXiv 2602.01469](https://arxiv.org/html/2602.01469)); parallel drafting, 5 tokens | 1 → 2 → 4 layers | HumanEval 2.69 → 3.58 → 3.92; MT-Bench 2.41 → 2.76 → 3.04 | +33% at the first doubling, +9-10% at the second |
| Hydra++ ([arXiv 2402.05109](https://arxiv.org/html/2402.05109)) | head depth | "5 layers and beyond provides no additional benefit" | 0 beyond 4 |
| Yan et al. ([arXiv 2402.01528](https://arxiv.org/abs/2402.01528)) | drafter quality against throughput | throughput depends mainly on the drafter's latency; its language-modelling quality "does not correlate strongly" | — |

**Estimate for this drafter at 5 layers, on code and agent turns: +3% to +10% tokens per step for each doubling of the layer matrices.** The middle of that range is about +5% (+0.35 tokens on 6.7). Three things push the real number toward the low end:

- The drafter already sits past the steep part of every depth curve. Each study's big gain is from 1-2 layers to 4-5; past 5, the measured gains are +3.3% per doubling (DFlash2 at 15 layers) and +9% (DFlash at 8, trained on only 100K samples, where a small drafter is relatively undertrained).
- Tree verification already recovers part of what a better drafter would add. It verifies the drafter's runner-up candidates at the positions where the chain fails, which is where a stronger drafter would also gain.
- No study measured sampled decoding against depth. Sampled acceptance (5.5-6.2) runs lower, and a sharper drafter's gain under the rejection rule is bounded the same way; the same relative range is assumed, with more doubt.

Width (the MLP) has no published ablation for block drafters. It is assumed to buy no more per parameter than depth.

## 5. The break-even (question 4)

tokens/s = 65.5 × (τ_size / 6.7) × (104 / (104 + Δdraft + Δcommit)), with τ_size = 6.7 × (1 + g)^log2(size), g between 0.03 and 0.10 (middle 0.05), and Δ the measured extra draft and commit milliseconds from §3. Sampled, the same arithmetic starts from about 51 tokens/s at 5.3-5.5 tokens per step and 104 ms (seed 1013 on agentlong, BACKLOG §19 item 34).

| Arm | × matrices | Δ ms | Step | Greedy tok/s at g = 3% / 5% / 10% | Sampled tok/s at 3% / 5% / 10% | Break-even g |
|---|---|---|---|---|---|---|
| f4-L5 | 1.00 | +0.0 | 104.0 | 65.5 / 65.5 / 65.5 | 51.0 / 51.0 / 51.0 | — |
| f4-L8 | 1.60 | +2.8 | 106.8 | 65.1 / 66.0 / 68.1 | 50.7 / 51.4 / 53.0 | 3.9% per doubling |
| f4-W1.5 | 1.49 | +1.9 | 105.9 | 65.4 / 66.2 / 68.0 | 51.0 / 51.5 / 52.9 | 3.2% per doubling |
| f4-L10 | 2.00 | +4.8 | 108.8 | 64.5 / 65.7 / 68.9 | 50.2 / 51.2 / 53.6 | 4.6% per doubling |
| f4-W2 | 2.03 | +4.0 | 108.0 | 65.0 / 66.3 / 69.6 | 50.6 / 51.6 / 54.2 | 3.7% per doubling |
| f4-L20 | 4.00 | +14.8 | 118.8 | 60.8 / 63.2 / 69.4 | 47.4 / 49.2 / 54.0 | 6.9% per doubling |
| f4-W4 | 4.00 | +11.5 | 115.5 | 62.6 / 65.0 / 71.4 | 48.7 / 50.6 / 55.6 | 5.4% per doubling |
| f4-L10W2 | 4.07 | +12.7 | 116.7 | 61.9 / 64.4 / 70.8 | 48.2 / 50.1 / 55.1 | 5.9% per doubling |
| bf-L5 | 1.00 | +20.2 | 124.2 | 54.9 / 54.9 / 54.9 | 42.7 / 42.7 / 42.7 | — |
| bf-L10 | 2.00 | +45.4 | 149.4 | 47.0 / 47.9 / 50.1 | 36.6 / 37.3 / 39.0 | 43.7% per doubling |
| bf-W2 | 2.03 | +43.3 | 147.3 | 47.7 / 48.6 / 51.0 | 37.1 / 37.9 / 39.7 | 40.5% per doubling |
| bf-L20 | 4.00 | +95.0 | 199.0 | 36.3 / 37.7 / 41.4 | 28.3 / 29.4 / 32.2 | 38.3% per doubling |

g is the relative gain in tokens per step for each doubling of the layer matrices. The last column is the g at which the arm only matches today's 65.5 tokens/s; above it the bigger drafter wins.

- **Two times bigger in NVFP4** pays from g = 3.7% (width) or 4.6% (depth). At the literature's middle (5%) it gains +0.3 to +1.2% (65.7-66.3 tokens/s), below what one GPU window can resolve (BACKLOG §19: a gain under ~5% does not show). At the optimistic end (10%) it gains +5-6% (68.9-69.6). At the end measured past 5 layers (3.3%, DFlash2 at 15 layers) it loses up to 1.5%.
- **Four times bigger** needs g of 5.4-6.9% per doubling, sustained over two doublings, and loses at the middle of the range (63.2-65.0 tokens/s).
- **Width is the cheaper axis.** If a bigger drafter is ever built, it should be built wider before deeper: break-even 3.7% against 4.6% per doubling. No study has measured what width buys a block drafter.
- **Sampled** follows the same ratios from 51 tokens/s: 2× in NVFP4 gives 50.2-54.2, 4× gives 47.4-55.6.
- **BF16 at any size loses**, at 2× by 24-28% (break-even ≥ 40% per doubling).

**Where the gain beats the cost:** only in NVFP4, only at 1.5-2× (by width first), and only if a trained drafter reaches at least +4-5% tokens per step per doubling on agent turns with the tree on. That is above what the closest evidence (+3.3% past 5 layers) shows.

## 6. Training a bigger drafter (question 5)

**Measured on node-b** (`bench/drafter/train_step.py` in the SGLang v0.5.21 image's torch 2.13, same window):

- A DFlash2-shaped drafter (fc, the layers, the frozen 248K-row target head with full-vocabulary cross-entropy, AdamW, per-layer checkpointing; 8,192 tokens per step) trains at **3,438 tokens/s at 5 layers** (2.38 s per step, 30 GB peak) and **2,084 tokens/s at 10 layers** (3.93 s, 40 GB).
- That is 16.0 and 26.1 GFLOP per token, **55 TFLOPS achieved** excluding recomputation, against **95.4 TFLOPS** for a plain BF16 8192³ GEMM on the same GPU.
- FP32 master weights would add 8 bytes per parameter (+14 GB at 5 layers, +27 GB at 10). That still fits beside nothing else: the model server must be stopped, as the memo says.

**Data.**

- *From scratch,* at the published recipes' scale: SpecForge's Qwen3.8-27B DFlash2 recipe uses ~1.07M target-regenerated conversations of ~4.5K tokens, about 4.8B tokens per epoch; the DFlash paper used ~800K samples for 6 epochs.
- *Warm-started from today's drafter.* Copy its 5 layers and zero-initialize the new layers' o_proj and down_proj, so the 10-layer drafter starts as today's exactly. For width, split each MLP neuron in two (rows duplicated, down columns halved), which also preserves the function. A fine-tune then needs far less data: an estimate of **30-100M written tokens**, about ten times the few million that same-size adaptations use, since the new parameters start from nothing. *(Estimate: no published number for up-scaled block drafters.)*
- *Fresh agent sessions on public tasks* (SWE-smith, MIT, 50K instances; SWE-rebench, CC-BY-4.0, each repository's licence recorded per instance; no research-only restriction on either):
  - about 12K written tokens per session (night of 10-08: 168 sessions ≈ 2M);
  - so 30-100M tokens is 2,500-8,300 sessions;
  - at ~180K written tokens per stream-hour that is 170-560 stream-hours, **5-15 nights** at four streams on node-b.

**Compute on one GB10** (local only).

- *From scratch:* 4.8B tokens per epoch is 390 h at 5 layers and **640 h (27 days) at 10 layers, per epoch**. The target's features must also be captured: 4.8B tokens of prefill at ~2,400 tokens/s is ~550 h, every epoch, because at 51 KB per token the features cannot be stored. **Months: not feasible.**
- *Warm-started fine-tune of 100M token-epochs at 10 layers:* 13 h of training plus the features. The features come from one prefill of each session's transcript, about 250M context tokens for 4,200 sessions: ~29 h per epoch at ~2,400 tokens/s, done online (the target and the trainer share the GPU, 21 + 40-55 GB) because storing them would take ~13 TB.
- *Total:* **~40-45 GB10-hours per epoch, 4-5 nights per epoch** with the model server stopped. Two epochs plus generating the sessions comes to about three weeks of nights, displacing SWE-bench and Night Shift.

**Compute rented** (estimates from the recipe's reported throughput and October 2026 prices).

- *From scratch:* the SpecForge recipe's 9-23 samples/s on 8×H200 is 105-263 H200-hours per epoch for today's size (capture included). A 2× drafter at ~1.5× that over 6 epochs comes to ~1,000-2,400 H200-hours, **$3.4K-12.7K** at $3.59-5.29 per H200-hour (RunPod community and secure).
- *The warm-started fine-tune:* an H100 at ~40% of its 989 dense BF16 TFLOPS trains ~7× faster than the measured GB10, and prefills the features ~10× faster. That is ~5-10 H100-hours per epoch, **$15-40** at $2.69-3.99 per H100-hour (RunPod, Lambda, Modal).

**Privacy.**

- Renting means sending the training data off the machines: the session transcripts with their full contexts (tool output, file contents, paths).
- For sessions on *public* tasks that is public repositories and the model's own text, an acceptable exposure if the owner agrees.
- The owner's own sessions never leave node-b (standing rule of 10-09, memo §4), and a drafter tuned on them never ships.
- **The local-only path is the GB10 path above:** the same pipeline, slower (4-5 nights per epoch), with nothing leaving the machine. Feature capture could come from ling-engine itself, which already writes the five target layers' features (`cap_`) during every pass, so no second runtime is needed.

## 7. Go/no-go (question 6)

**No-go for a bigger drafter.**

| | Today | 2× NVFP4 (width) | 2× NVFP4 (depth) | 4× NVFP4 | 1× BF16 |
|---|---|---|---|---|---|
| Δ step (measured) | 0 | +4.0 ms | +4.8 ms | +11.5 to +14.8 ms | +20.2 ms |
| Break-even acceptance gain per doubling | — | 3.7% | 4.6% | 5.4-6.9% | impossible |
| Expected greedy tok/s (g 3 / 5 / 10%) | 65.5 | 65.0 / 66.3 / 69.6 | 64.5 / 65.7 / 68.9 | 60.8-62.6 / 63.2-65.0 / 69.4-71.4 | 54.9 |
| Training | — | warm start, 30-100M tokens: 3 weeks of node-b nights, or ~$15-40 per epoch rented | same | worse | — |

Reasons:

- **The gain is within noise at the middle of the range.** At 5% per doubling, +0.3-1.2% tok/s cannot be measured in a window. The best case (+5-6%) needs depth gains that the evidence past 5 layers does not show: +3.3% per doubling for DFlash2 at 15 layers, and the DFlash paper's speedup peaking at 5 layers even on H200, where weights are cheaper to read than here.
- **The tree already takes part of the gain.** Tree verification recovers some of what a stronger drafter would add (§4).
- **The training cost is that of the memo's option (b) or more,** for a smaller expected return. Option (b), today's size fine-tuned on public agent sessions, promises +5-15% at no step cost.

**What would change the answer:**

- a published bigger DFlash2 for Qwen3.8-27B (test it per §2, about a day);
- a block-drafter study showing ≥ +5% per doubling past 5 layers on code;
- option (b)'s pipeline existing. A 2×-width arm then costs only ~1.6× the training compute of the 5-layer run on the same data.

**First concrete step, if the owner decides go anyway:** inside option (b)'s pipeline, once it trains the 5-layer drafter on public agent sessions, add one arm:

- **The arm:** the **2×-width** drafter (I 38,912, the cheaper axis), warm-started from today's weights by neuron splitting (function-preserving, so it starts at today's acceptance), trained on the same data and epochs.
- **Quantization:** NVFP4 per §2 step 2.
- **Measurement:** agentlong against the fine-tuned 5-layer drafter, greedy and two seeds.
- **The bar:** keep it only if tokens/s are ≥ +4% over that 5-layer drafter (≥ +5% tokens per step to clear its measured +4.0 ms).

## 8. Files

- `bench/drafter/big_dummy.py`: a bigger drafter's checkpoint from the real one's bytes (pure Python; node-b's host Python has no numpy).
- `bench/drafter/big_window.sh`: the GPU window, all arms, and the training step.
- `bench/drafter/train_step.py`: a DFlash2-shaped drafter's training step with dummy data, in torch from the SGLang image.
- Branch `bench/big-drafter`: the engine patch the BF16 and wide arms need. It is not merged: main loads only NVFP4 drafter projections, and nothing there needs more.
- Raw results on node-b: `~/bd/w1/` (`results.jsonl`, the serve logs, `train.jsonl`).
