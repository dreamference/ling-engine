# ling-engine

**A single-model inference engine for Qwen3.8-27B on the NVIDIA DGX Spark (GB10).** Proposed; nothing is built yet.

Decoding Qwen3.8-27B on a DGX Spark is bound by its 273 GB/s memory bus, not by compute: every generated token streams about 15 GB of 4-bit weights, which caps plain decoding near 17 tokens/s. The best public recipe today (SGLang with the DFlash2 drafter) reaches 47.9 tokens/s. This engine aims to get more tokens out of each weight read, then remove every other cost.

**Spec:** [SPEC.md](SPEC.md) (8 October 2026)

## The plan in brief

Two programs: `ling-compile`, an offline compiler that quantizes to NVFP4, folds the norms, lays the weights out on disk in kernel order, instantiates one fused CUDA kernel per matrix shape and captures each decode step as a CUDA graph; and `ling-serve`, a resident runtime that keeps every conversation's KV cache and DeltaNet state on the GPU.

![Compiler and runtime](images/compiler-and-runtime.svg)

Four levers, in order of expected payoff:

1. **Speculative decoding tuned for this bus:** the DFlash2 drafter quantized to NVFP4 with a reduced-vocabulary head, verification blocks of 8–32 rows, and prompt-lookup drafts spliced in for edit tasks.
2. **Shape-specialized fused CUDA kernels** over pre-tiled NVFP4 weights, so a 16-row verify costs little more than a 1-row step.
3. **FP8 KV cache and a replay scheme for the DeltaNet recurrent state,** so speculation on the hybrid architecture costs one state read and write per step instead of eight.
4. **One CUDA graph per step** with GPU-side sampling and resident sessions: under 1 ms of host overhead, and a follow-up's first token in one weight pass.

**Targets** (estimates until milestone M1 measures the real bus): 60 tokens/s or more on prose and thinking traces, 90 or more on code and math, 150 or more on edits, a follow-up's first token in 80 ms or less, and greedy output identical to non-speculative decoding.

## Milestones

![Roadmap](images/roadmap.svg)

## Part of Mightling

This engine is planned as a model server for [Mightling](https://github.com/dreamference/mightling), Dreamference's local, confidential AI for DGX Spark, where Qwen3.8-27B is the default model.

## License

GNU Affero General Public License v3.0 ([LICENSE](LICENSE)), the same as Mightling.
