# ling-engine: Architecture and the kernel plan

Part of the ling-engine specification; the index is [README.md](./README.md). Section numbers are those of the single-file `SPEC.md` this was split from (Oct 8, 2026 · revised the same day against Mightling's measured workload), so a citation of the form `§n` or `S§n` in the reports and the code still names the same text.

## 8. Architecture: an offline compiler and a resident runtime

Two programs. `ling-compile` runs once per checkpoint and machine and decides everything that can be decided before the first request; `ling-serve` runs for weeks and makes no decisions at all.

![The compiler decides everything before the first request; the runtime only runs](../images/compiler-and-runtime.svg)

*compiler and runtime · 7 stages, 6 component groups*

The compiler's seven stages run top to bottom once and hand the runtime a weight blob, compiled kernels, captured graphs and a memory plan; the runtime's components then serve requests without deciding anything.

Compiler stages, in order:

1. Ingest the Hugging Face safetensors (`Qwen3_5ForConditionalGeneration`) and split them into the text stack, embedding, LM head, vision encoder, MTP module and the DFlash2 drafter.
2. Fix the precision map. Start from the served checkpoint's map (NVFP4 FFN and head, FP8 projections; E2M1 values, one E4M3 scale per 16, one FP32 scale per tensor) and move projection groups to NVFP4 only where the quality gate in [[section 13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md)](#13-validation-exactness-quality-and-a-benchmark-that-cannot-flatter) passes, using NVIDIA's Model Optimizer with a calibration set that mixes recorded agent sessions, code, prose, math and thinking traces. The map is recorded in the manifest. The drafter keeps its NVFP4 build; drafter error costs speed, never correctness.
3. Fold what folds: each pre-layer RMSNorm's γ is multiplied into the projection that follows it (the DeltaNet in-projection, the attention q/k/v, the FFN gate and up). The per-head q/k norms and the DeltaNet output norm act after their projections and cannot fold; they become kernel epilogues.
4. Tile: permute every matrix into the fragment order its kernel's warps will load (the SM120 block-scaled MMA layout for NVFP4, the FP8 MMA layout for FP8), interleave the scales with the values, and write one 4 KB-aligned blob plus a manifest of offsets and hashes. Load is a memory map; nothing is converted.
5. Instantiate kernels: one template instance per (matrix, precision, row bucket) with all dimensions as compile-time constants, row buckets {1, 8, 16, 32, 48, 64} for decode, verify and batches, and {256, 512, 1024, 2048} for prefill chunks, compiled for sm_121 only. Tile configurations are autotuned once on the Spark and baked in; there is no runtime heuristic.
6. Plan memory: fixed offsets for weights, the paged KV pool (64-token pages), DeltaNet state slots, the shared-prefix checkpoint store, drafter feature cache, activation scratch, logits and candidate buffers, sized from the configured budget ([section 12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)). Nothing is allocated while serving.
7. Capture CUDA graphs: one verify-plus-draft graph per row bucket and batch size, and one prefill graph per chunk size. Graphs read sequence lengths and page tables from device memory, so a single graph serves every context length.

Runtime components:

- One language: C++20 and CUDA for everything that ships, the HTTP layer included (decided 2026-10-08: the maintainer knows C++ far better than Rust, and the kernels it builds on, CUTLASS, CuTe and FlashInfer, are C++). Python appears only in tests and benchmark scripts, and in Triton or CuTe DSL kernel prototypes that are rewritten in CUDA C++ before they ship.
- A C++/CUDA core with no Python on the hot path, behind a thin asynchronous HTTP layer that speaks the chat-completions and completions APIs with server-sent events, `reasoning_effort`, `enable_thinking`, `preserve_thinking`, and the Qwen3 tool-call format ([section 12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md) lists exactly what Mightling calls).
- A session manager keyed by conversation: paged FP8 KV, one DeltaNet snapshot slot, the conv window, the token history, and the target hidden features the drafter conditions on. Eviction is LRU inside the single memory pool.
- A shared-prefix store: DeltaNet state checkpoints (with their KV pages) at message boundaries, keyed by a hash of the token prefix, so a new session that begins with a known prefix starts from its checkpoint ([section 11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)).
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
- `draft_*`: the DFlash2 drafter's five sliding-window layers over the accepted tokens' target features plus the mask tokens, in NVFP4, then its head over a reduced vocabulary ([[section 10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md)](#10-speculative-decoding-more-tokens-per-pass)).

Two Qwen3.8-specific decisions:

- DeltaNet state and speculation. A verify runs the recurrence over M draft tokens, but only the first j are accepted, so the committed state is the one after token j. SGLang's verify keeps intermediate states for the verified tokens (by this spec's estimate up to 16 × 75 MB per step in BF16; M0's profile measures it). We instead keep the snapshot S₀ and the M rows' k, v, α, β (130 KB per layer), and the next step's `gdn_scan` replays the j accepted tokens from S₀ before it reads the new draft, writing the result as the new snapshot. Cost: one 151 MB read and one write per step across all layers, in FP32, about 2% of the pass. Tree branches are handled the same way: each branch is a replay from S₀, trivial compute, with S₀ served from L2.
- The 248k-token LM head is 0.72 GB at NVFP4, 4% of the pass, and SGLang's drafter reads it a second time per step. The drafter gets a separate head over the 32k most frequent tokens (0.16 GB in FP8), chosen from the token frequencies of recorded agent sessions, since a draft token outside that set is just a rejected guess.

What fusion buys here is modest on its own and essential in combination. A layer in a generic engine is 10–14 launches with the host encoding between them, 700–900 per step; at a few microseconds each that is several milliseconds, 5–10% of a 75 ms pass. The fused path removes that, keeps the 17,408-wide FFN activation out of memory, and, most important, makes a 32-row verify cost 3–5 ms of visible compute instead of a second pass of memory traffic. That is what turns Spark's compute surplus into accepted tokens.
