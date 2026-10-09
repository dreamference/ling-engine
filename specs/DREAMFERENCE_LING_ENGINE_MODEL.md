# ling-engine: The model: shapes, precision map and bytes per step

Part of the ling-engine specification; the index is [README.md](./README.md). Section numbers are those of the single-file `SPEC.md` this was split from (Oct 8, 2026 · revised the same day against Mightling's measured workload), so a citation of the form `§n` or `S§n` in the reports and the code still names the same text.

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

Moving the FP8 projections to NVFP4 saves about 3.2 GB per pass (18%), but they were kept at FP8 by the checkpoint's makers, presumably for quality; the quality gate of [section 13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) decides, group by group, and the targets are stated both ways.

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
- Thinking is the model's default (Mightling's agent produces little: at most a tenth of its output, [section 2](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)), with `reasoning_effort` xhigh, medium or low, and `preserve_thinking` keeps earlier thinking blocks in the history by default, so the token prefix of a conversation is stable across turns. Recommended sampling: temperature 1.0, top-p 0.95, top-k 20 in thinking mode; temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 in instruct mode.
- The DFlash2 drafter Mightling serves (`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`, an NVFP4 build of [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)) is a 5-layer block-diffusion model with a 2,048-token sliding window, 1.55 GB on disk. It conditions on the target's hidden states at layers 5, 19, 33, 47 and 61, was trained for blocks of 8, and has a top-K candidate selector (`selector_top_k` 16, rank 256); production runs it with 16 draft tokens (`--speculative-num-draft-tokens 16`). Whether SGLang's DFLASH path arranges those 16 as a tree from the selector's alternatives or as a longer chain is read from its code in M0, and the acceptance difference between the two is measured there. It has no output head of its own and reuses the target's LM head ([W8 card](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8)).

Sources: [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) · [Qwen3.8-27B on DGX Spark, OpenZeka](https://blog.openzeka.com/en/qwen3-8-27b-on-dgx-spark-48-tok-s/) · [DFlash2 drafter W8 card](https://huggingface.co/lued/Qwen3.8-27B-DFlash2-W8) · the served checkpoints' `config.json`, `quantization_config` and safetensors headers on this machine
