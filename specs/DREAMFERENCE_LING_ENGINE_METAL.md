# ling-engine: A native Metal port for Apple silicon (2.0)

Part of the ling-engine specification; the index is [README.md](./README.md). Written on 2026-10-28 from the owner's decisions of that day. **It is a specification only: nothing is built.** The code starts after 1.x is complete (2026-11-18); the first Mac, a Mac mini M5 Pro with 64 GB, arrives about 2026-11-11. Like the CLI, API and BATCHING documents, it has no section number from the former `SPEC.md`.

It builds on [ARCHITECTURE §8–9](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) (the compiler, the runtime, the kernels), [MODEL §5](./DREAMFERENCE_LING_ENGINE_MODEL.md) (shapes, precision map, bytes per pass), [SPECULATION §10](./DREAMFERENCE_LING_ENGINE_SPECULATION.md), [BATCHING §5](./DREAMFERENCE_LING_ENGINE_BATCHING.md) (sessions, paged KV), [LOWBIT](./DREAMFERENCE_LING_ENGINE_LOWBIT.md) (the byte model of a verify pass and the quality protocol), [API](./DREAMFERENCE_LING_ENGINE_API.md), [VALIDATION §13](./DREAMFERENCE_LING_ENGINE_VALIDATION.md) and [MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md) (the agreement gate and the exactness policy). Every claim about the CUDA engine cites the source as `file:line` at `5a88252`; every number about Apple hardware that has not been measured on our own unit is marked **TO VERIFY**, and every projection is marked as an estimate. In this document M0–M9 are the Metal milestones (§7); the CUDA engine's milestones are cited as "CUDA M0" and so on.

## 0. Summary

- **What.** `ling-serve` for Apple silicon: the same engine design (one model, shape-specialized kernels, a rows path that is row-invariant, speculation by DFlash2 with tree verify, resident sessions), with hand-written Metal kernels for Qwen3.8-27B's exact shapes. Not a shared core with backends, not MLX, not llama.cpp. The HTTP layer, the tokenizer, the parsers, and the sampling, accept, tree and lookup code are the same source files compiled unchanged (they contain no CUDA); everything that touches the GPU is rewritten as a mirror of its CUDA counterpart, file for file and kernel for kernel, and the host logic that lives inside the CUDA engine's files is copied into the mirror (§3).
- **Weights.** The checkpoint's NVFP4 and FP8 values and scales are kept bit for bit; only the tile order inside each 16-row block changes, for Apple's 32-lane simdgroups (§3.2). Every weight value the Metal kernels multiply is the value the CUDA kernels multiply, so the Metal build serves the same model; its logits differ from CUDA's only by summation order and transcendental functions.
- **Exactness.** Within a Metal build on one machine, every CUDA rule holds bit for bit: speculative greedy equals plain greedy, a batched sequence equals itself alone, a chunked or resumed prefill equals one from scratch, the pre-encoded step equals the eager one. Across CUDA and Metal, bit identity is not promised; it is replaced by a statistical gate against the CUDA engine (§2).
- **Performance (estimates, §5).** Decode is bound by bytes on both machines and the bytes per pass are identical, so an M5 Pro (307 GB/s published, **TO VERIFY**) should decode at about a GB10's rate, *provided* the 16-row verify's arithmetic hides under the weight stream. That needs the M5's GPU neural accelerators: on simdgroup matrix units alone, the break-even is about 7 verify rows, not 16. Prefill is bound by compute, where the GB10's FP4/FP8 tensor cores have no Apple counterpart: about 2–3× slower on an M5 Max, 4–6× on an M5 Pro by the same method (the owner's 2–4× holds for the Max class; §5.4 says what the Pro would need).
- **Memory.** 36 GB is the floor (one or two sessions at 32K, no vision); 64 GB holds two sessions with a ~235K-token paged pool and vision; 128 GB matches the GB10's four sessions at the full 262,144 context (§4.2).
- **Plan.** Ten milestones, each with an exactness gate and a measured target; about 18 weeks plus nights for one engineer to the first release (M0–M8), an estimate (§7). Fifteen open questions with recommended defaults (§9).

## 1. Decisions and scope

| Decision (the owner, 2026-10-28) | What it means here |
|---|---|
| A native Metal port with the same design | Hand-written Metal Shading Language kernels for this model's shapes; the CUDA engine's structure, names and invariants mirrored. No device abstraction layer, no runtime choice of backend, no third-party inference runtime. |
| The same HTTP API | `/v1/responses` (the Codex Responses API), `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/metrics`, `/spec_stats`, `/model_info`, `/health` exactly as [API §2–§9](./DREAMFERENCE_LING_ENGINE_API.md); the same source (`src/serve/api.cpp`), so the same bytes on the wire. |
| The same exactness rules | §2: within the Metal build, the CUDA engine's rules bit for bit; across builds, a stated statistical gate. |
| The same speculative decoding | DFlash2 (the ScaleSearch drafter), the sparse head, tree verify of 16 rows, exact rejection sampling, commit by replay; the multi-drafter of Mightling 2.0 ([MULTI_DRAFTER](./DREAMFERENCE_LING_ENGINE_MULTI_DRAFTER.md)) as soon as the CUDA engine has it. |
| Targets | Mac mini M5 Pro 64 GB first; then Mac Studio (Max, Ultra) and MacBook Pro (Max); 36 GB the minimum, with short contexts. |
| Timing | 2.0 work: this specification now, code after 1.x is complete (2026-11-18). |

**Out of scope.** Intel Macs; iPhone and iPad; any other model; MLX's or llama.cpp's formats or kernels; a quantization of the target other than the served checkpoint's ([LOWBIT](./DREAMFERENCE_LING_ENGINE_LOWBIT.md)'s candidates stay CUDA-first studies); the Apple Neural Engine (it is not programmable from Metal and has no published low-level interface); two machines serving one model.

## 2. Exactness across CUDA and Metal

The CUDA engine's exactness rests on three properties of its rows path, and the Metal build must reproduce all three:

1. **Row invariance.** A row's result does not depend on how many rows share the pass: every output element is accumulated over the same K chunks by the same warp in the same order, and the warps' partial sums are added in a fixed order (`stream_gemm.cu:10-13`, `stream_gemm.cu:318-348`); a split of K depends on the matrix shape only, never on M (`stream_gemm.cu:200-203`); attention runs over fixed 4,096-key ranges at fixed positions and is combined afterwards (`kernels.cuh:463-468`, `kernels.cu:826-831`). This is what makes a token verified in a block of 16 get exactly the numbers it would get decoded alone (`kernels.cuh:99-102`).
2. **Commit by replay.** A verify reads the DeltaNet and conv state and leaves it unwritten (`engine.cpp:1199-1203`); the commit replays the accepted rows' saved inputs through the same kernels (`speculate.cpp:508-521`), and the recurrence's per-token arithmetic is the same whatever M is (`kernels.cuh:418-423`), so the committed state is plain decoding's.
3. **Chunk invariance of the prefill.** A prompt always takes the prefill path, whatever the chunk's size (`engine.cpp:1016-1018`), and the chunked recurrence gives the same result for any split on a multiple of 32 tokens (`kernels.cuh:429-434`).

**Three levels of "the same".**

| Level | Scope | Rule | Checked by |
|---|---|---|---|
| L1, within a build | One Metal build on one machine | Bit for bit, every rule of the CUDA engine: greedy with speculation (chain, tree, lookup) equals greedy without it, token for token, with the same DeltaNet state hash (`engine.hpp:373-374`); a batched sequence equals itself alone; a resumed or chunked prefill equals one from scratch; the pre-encoded step (§4.3) equals the eager one; seeded sampling is identical whenever the batch shape is, and exact in distribution always | The `ling-run` checks ported unchanged in meaning (`--spec-check`, `--prefix-check`, `--chunk-check`, `--batch-check`, `--interleave-check`, and `--icb-check` for `--graph-check`; `tools/ling_run.cpp:349-365`); the per-kernel memcmp tests of row invariance (§6) |
| L2, across Macs | One Metal build on two GPU families or two macOS versions | Not promised bit for bit: the simdgroup matrix and neural-accelerator units' internal order, and the driver's compiler (the `.metallib` holds an intermediate form that macOS compiles for the GPU at pipeline creation), may differ. Each machine is its own L1 reference. | The L3 gate run once per GPU family and per macOS major version; golden token files carry (GPU family, macOS build) |
| L3, across CUDA and Metal | The Metal build against the CUDA engine on the same requests | The same weights bit for bit (§3.2), the same quantization points (the FP16 activation copy with a power-of-two scale per row, `stream_gemm.cu:3-8` and `stream_gemm.cu:478-500`; NVFP4 and FP8 activations in the prefill, `prefill_gemm.cu:4-7`; BF16 KV; FP32 state), the same algorithms; differences only in summation order, matrix-unit internals and `exp`/`rsqrt` implementations. Greedy outputs may part at near-ties, as SGLang's and the engine's do ([BATCHING §2.3](./DREAMFERENCE_LING_ENGINE_BATCHING.md)) | The statistical gate below |

**The L3 gate** (the reference is the CUDA engine at the release's commit, teacher-forced; the reference logits and continuations are computed once on the GB10 and pinned by SHA-256, so the Mac never talks to the GB10):

- on the agreement study's 54 prompts ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), "Agreement gate"), Metal's top-1 flips against the CUDA engine are not significantly above production's self-noise by the same one-sided Fisher test (production's own 65 flips in 2,490 positions set the threshold at 85 today), every flip's gap ≤ 1.75 nats, top-5 agreement 100%;
- mean KL(CUDA ‖ Metal) ≤ 0.0008 nats, the exactness policy's near-exact line, with no growth with position at 25K and 64K;
- perplexity on VALIDATION §13's held-out mix within a ratio of [0.999, 1.001] of the CUDA engine's. A reordered sum moves perplexity by far less; a format change moves it by percents (Minima's 4–9%, [LOWBIT §3](./DREAMFERENCE_LING_ENGINE_LOWBIT.md)), so this line catches a wrong kernel, not noise;
- acceptance (tokens per step) on agentlong within the run-to-run spread of the CUDA engine's (the drafter and the target are the same models, so acceptance should not move);
- informational, not a gate: the share of agentlong turns whose greedy output is identical to the CUDA engine's.

**Compile rules that make L1 possible.** Metal compiles with fast math unless told otherwise. Every shader is built with fast math off, the precise forms of `exp`, `log` and `rsqrt` (or the engine's own fixed implementations where CUDA uses `__expf`, e.g. `stream_gemm.cu:996`), and every fused multiply-add written as an explicit `fma` with contraction off, so the compiler cannot reassociate or fuse differently between kernels that must agree (the commit and the verify, the FFN's two forms). A build check refuses a shader compiled with other flags (§6.1). The exact MSL pragmas and compiler flags are pinned at M0 (**TO VERIFY** against the Metal compiler shipped with the M5's macOS).

## 3. Component map

### 3.1 Each CUDA component and its Metal counterpart

"Shared" means the same source file compiled into both builds with no change and no abstraction; "mirrored" means a Metal source written to the CUDA one's structure and names, whose header names the CUDA file and lines it mirrors and states each numerical difference; "replaced" means the CUDA mechanism has no Metal equivalent and something else does its job.

| CUDA component | Where | Metal counterpart | Kind |
|---|---|---|---|
| HTTP server, routes, SSE, Responses conversion, tool-call parser | `src/serve/main.cpp` (proxygen and folly, `main.cpp:61-69`), `src/serve/api.cpp` | The same files. `main.cpp` and `api.hpp` include `core/engine.hpp` (`main.cpp:87`, `api.hpp:14`), so the Metal engine declares `class ling::Engine` with the same public members (`engine.hpp:198-378`); the build compiles one engine or the other, never both | Shared |
| Tokenizer, chat template | `src/tokenizer/*` | The same files | Shared |
| Sampling, exact rejection rule, tree build and accept, context lookup | `sampling.cpp`, `accept.cpp`, `tree.cpp`, `lookup.cpp` (host only: `sampling.hpp:1-2`, `accept.hpp:1`, `tree.hpp:1-19`, `lookup.hpp:1-3`) | The same files | Shared |
| Prefix KV store (radix tree, arena) | `prefix_kv.cpp` (host bookkeeping; the device side through `Ops`, `prefix_kv.hpp:1-10`, `prefix_kv.hpp:21-22`) | The same files; `Ops` implemented with Metal blit copies, then page mapping (§3.3) | Shared |
| Safetensors reader | `safetensors.cpp` (a read-only `mmap`, `safetensors.cpp:29`) | The same file, used by the converter | Shared |
| Admission control | `serve/admission.hpp` (reads `/proc/meminfo`, `admission.hpp:72-80`; scratch 4 GiB, `admission.hpp:33`) | The plan's arithmetic kept; the inputs from macOS (§4.8) | Mirrored |
| Image decoding | `serve/image_input.cpp` (libpng, libwebp, libjpeg; nvJPEG opened at run time, `image_input.cpp:127`) | The same decoders, without the nvJPEG branch, so a decoded image is the same pixels as on Linux | Mirrored (one `#if`) |
| Model loader: upload, retile on the GPU | `model.cu:73-96` (the precision map), `model.cu:110-145` (`cudaMalloc`, copy, `retile_nvfp4`/`retile_fp8`) | `ling-convert` writes the Metal tiled blob once; the loader maps it (§4.1) | Replaced |
| Weight formats: tiled NVFP4, B26 coded scales, tiled FP8 | `kernels.cuh:142-164`, `stream_gemm.cu:39-46` | N4M and F8M: the same codes and scales, re-ordered inside each block (§3.2) | Mirrored |
| Stream GEMM, 1–32 rows | `stream_gemm.cu:1-360`, `stream_gemm.cu:592-703` | `stream_gemm.metal` (§3.4) | Mirrored |
| RMSNorm with the FP16 copy; FP16 row copy | `stream_gemm.cu:380-418`, `stream_gemm.cu:478-566` | The same rule for the power-of-two scale (the row's largest magnitude in [2^14, 2^15)) | Mirrored |
| The FFN as one persistent kernel | `stream_gemm.cu:841-1072`; on by default (`engine.hpp:464-467`) | Two dispatches in one encoder; a ticket-queue persistent form behind a switch (§3.5) | Mirrored |
| The DeltaNet mixer as one persistent kernel, chain and tree | `stream_gemm.cu:1074-1457`; on by default (`engine.hpp:469-472`) | Concurrent dispatches; a ticket-queue persistent grid behind a switch (§3.6) | Mirrored |
| DeltaNet recurrence, conv, gating, gated norm (graph path) | `kernels.cu:320-453`, `kernels.cuh:412-423` | `gdn.metal` (§3.6) | Mirrored |
| Attention: prepare, rows attention, combine | `attention_body.cuh:42-140`, `kernels.cu:524-627`, `kernels.cu:826-842` | `attention.metal` (§3.7) | Mirrored |
| The attention layer as one persistent kernel | `stream_gemm.cu:1459-1725`; off by default (`engine.hpp:473`) | Not ported in the first release | — |
| Layer-spanning persistent kernel | `stream_gemm.cu:1727-1788`; off by default (`engine.hpp:476`) | Not ported | — |
| Prefill GEMMs (block-scaled NVFP4 and FP8 tensor cores) | `prefill_gemm.cu:1-21`, `kernels.cuh:170-201` | `prefill_gemm.metal` with the activations quantized as CUDA quantizes them (§3.8) | Mirrored |
| Prefill DeltaNet (parallel conv, chunked WY recurrence in 32-token chunks) | `prefill_gdn.cu:1-21` | `prefill_gdn.metal`, same chunk rule | Mirrored |
| BF16 GEMM for large M (the drafter's fc over a prompt, the vision tower) | `gemm_bf16.cu` (TMA, `kernels.cuh:397-400`) | `gemm_bf16.metal` | Mirrored |
| Top-K per row, ties to the lower id | `topk.cu`, `kernels.cuh:387-390` | `topk.metal`, the same tie rule | Mirrored |
| Drafter: loader, kernels | `drafter.cpp`, `drafter.hpp:1-32`, `draft_kernels.cu`, `kernels.cuh:520-555` | Mirrored (§3.10) | Mirrored |
| Sparse drafter head | `sparse_head.cu`, `kernels.cuh:270-316` | Mirrored | Mirrored |
| Vision tower | `vision.cu`, `vision.hpp:1-8` (BF16 at torch's rounding points) | Mirrored, the same rounding points | Mirrored |
| Engine: forward pass, sessions, checkpoints, prefill cursor | `engine.cpp`, `engine.hpp` | `src/metal/engine_metal.cpp` with the same members and the same host logic (session binding `engine.hpp:350-362`, the checkpoint policy `engine.hpp:438-443`) copied, the CUDA calls replaced | Mirrored |
| Speculative step: plan, draft, verify, accept, commit, batch | `speculate.cpp:131-1228` | `src/metal/speculate_metal.cpp` | Mirrored |
| CUDA graphs per step part, positions read on the device | `speculate.cpp:19-32`, `engine.hpp:538-561`, `kernels.cuh:14-43` | Pre-encoded indirect command buffers and shared events (§4.3) | Replaced |
| Programmatic dependent launch | `launch.cuh:1-19` | None; dependent dispatches in one encoder, independent ones concurrent | Replaced |
| Commit beside the next draft on a second stream | `engine.hpp:113-117`, `engine.hpp:449-458` | A concurrent dispatch in the step's encoder (§4.3) | Replaced |
| Prefill beside a decode on a green context of 40 SMs | `engine.cpp:37-60`, `engine.hpp:118-123` | A second command queue without a core partition; the scheduler's share rule (C1, [BATCHING §2.2](./DREAMFERENCE_LING_ENGINE_BATCHING.md)) does the rest | Replaced |
| Pinned host buffers and memcpy nodes | `engine.hpp:562-578`, `speculate.cpp:931-936`, `engine.cpp:1427-1430` | Shared-storage buffers read and written in place (§4.1) | Replaced |
| NVTX ranges, Nsight, `compute-sanitizer`, `cuda-gdb` | `speculate.cpp:59-62`, [ARCHITECTURE §8](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) | `os_signpost` intervals with the same names, Instruments' Metal System Trace, Metal API and shader validation, Xcode's shader debugger | Replaced |

**Host logic that lives in CUDA files.** Session binding, the checkpoint store's policy, the prefill cursor and the step planner are plain C++ inside `engine.cpp` and `speculate.cpp`. They are copied into the Metal engine, not moved, so the CUDA build does not change for the port. A CI script lists each mirrored function pair and fails when one side changes without the other (§6.2). Whether to extract them later into shared plain files is open question 3.

### 3.2 Weight format: N4M and F8M

**Choice.** Keep NVFP4's numbers and change only the byte order. **N4M** ("NVFP4 on Metal") stores the checkpoint's E2M1 codes, its E4M3 scale per 16 values and its FP32 tensor scale, bit for bit; **F8M** stores the FP8 projections' E4M3 values and per-tensor scale bit for bit. The block geometry stays the CUDA engine's: one contiguous block per (16-row tile, 128-value chunk), values then scales, 1,024 + 128 bytes (`stream_gemm.cu:41-43`), or 1,024 + 96 bytes with B26's 6-bit scale codes and the tensor's 64-entry table (`kernels.cuh:150-159`, `stream_gemm.cu:44-46`). Inside a block, the CUDA order puts each lane's 16 bytes where the `m16n8k16` fragment wants them (`stream_gemm.cu:15-21`); N4M permutes them so each of a simdgroup's 32 threads loads 16 contiguous bytes (one simdgroup load = 512 contiguous bytes, the pattern that took CUDA from 215 to 232 GB/s, `kernels.cuh:142-145`) holding exactly the values its matrix-unit operand needs. The permutation is generated from a probe kernel that records the simdgroup matrix's per-thread element layout on the unit at M0, not assumed. B26's codes are kept: on the GB10 they took 0.43–0.5 GB off each step ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), "Order" item 1, B26), and the prefill reads the codes too, so the CUDA build's second, raw copy of the scales for its prefill (+0.86 GB, same place) does not exist on Metal.

**Decoding.** Apple GPUs have no `cvt` from E2M1 or E4M3 (CUDA uses `cvt.rn.f16x2.e2m1x2` and `cvt.rn.f16x2.e4m3x2`, `stream_gemm.cu:48-60`). The Metal kernels decode in registers with integer operations or a 16-entry (E2M1) or 256-entry (E4M3) table in threadgroup memory; M0 measures both. The result is the same FP16 value either way: an E2M1 value times its E4M3 scale has at most 6 significant bits and an E4M3 value 4, both exact in FP16 (`stream_gemm.cu:6-8`).

**Does it change outputs against CUDA?** The weights do not change: the Metal kernels multiply the very values the CUDA kernels multiply, so this is the same model, not a re-quantization. Outputs differ only as L3 allows (§2): summation order and transcendental functions. Quality is therefore gated as a numerical port, not as a format change: the L3 gate (agreement, KL, perplexity ratio) at M1 and at every release, and one SWE-bench night served by the Mac against the GB10 engine at M7, read by the nights' McNemar test, as a guard against a large loss only ([LOWBIT §7](./DREAMFERENCE_LING_ENGINE_LOWBIT.md) explains why a night cannot resolve a small one).

**Rejected.** MLX's affine 4-bit groups and MXFP4: a re-quantization, so a different model, with MXFP4 measured worse than NVFP4 on this family ([LOWBIT §3](./DREAMFERENCE_LING_ENGINE_LOWBIT.md), (a3)). A block-scaled 4-bit type consumed natively by the M5's neural accelerators (if Metal 4 exposes one, **TO VERIFY**) is adopted only if its decoded values are bit for bit NVFP4's.

### 3.3 KV cache, recurrent state and session memory

- **KV.** BF16, as the CUDA engine keeps it (`kernels.cuh:453-459`; the FP8 cache of [SESSIONS §11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md) is not built). The CUDA engine allocates each session a flat head-major slot `[Hkv][max_context][D]` at start (`engine.cpp:168-181`). The Metal build uses [BATCHING §5.3](./DREAMFERENCE_LING_ENGINE_BATCHING.md)'s P4 layout from its first session: 64-position pages of 4 MiB laid out `[layer][K, V][kv head][64][D]`, one pool, one page table per session, copy-on-write on shared prefixes. Small Macs cannot afford slots sized at `--max-context` whether used or not (§4.2). If CUDA's P4 has not merged when the port starts, Metal still addresses through the table, and the flat layout is the identity table.
- **DeltaNet and conv state.** FP32, 48 layers × 48 heads × 128 × 128 (151 MB) plus the conv window (3 MB) per session (`engine.cpp:182-185`, [MODEL §5.2](./DREAMFERENCE_LING_ENGINE_MODEL.md)), the end-of-prompt copy (`engine.hpp:716-718`), the checkpoint store's slots (157 MB each, eight by default, `engine.hpp:67-72`) and the drafter's context ring (168 MB, `admission.hpp:29`). Unchanged, fixed slots outside the page pool.
- **The prefix KV store** (`engine.hpp:82-88`, `engine.cpp:332-371`): unchanged on the host side; on Metal it holds page references once P4c's sharing is in, so a hit maps pages instead of copying them.

### 3.4 The stream GEMM (decode, verify, drafter)

The contract is `stream_gemm.cu`'s and does not change: `y = gscale · x · Wᵀ` for 1 to 32 rows (`stream_gemm.cu:675-703`, `kernels.cuh:102`), `x` converted once per row to FP16 under a power-of-two scale, FP32 accumulation, the global scale applied in FP32 at the end, a fixed-order reduction, and several matrices sharing one input in one dispatch (`stream_gemm.cu:77-92`). The rules the Metal kernel adds:

- **One arithmetic for every row count.** CUDA runs the same MMA path from 1 to 32 rows and only changes how many 8-row column groups a warp carries (`stream_gemm.cu:697-700`). The Metal kernel must do the same: a dot-product kernel for 1 row and a matrix-unit kernel for 16 would give different bits and break `--spec-check`. Whichever unit M0 picks (simdgroup matrix or the neural accelerators, §4.4) serves every row count from 1 to 32, even where it wastes lanes at one row; a one-row step is bound by bytes anyway.
- **The reduction fits 32 KB.** CUDA's cross-warp reduction buffer is 8 warps × 32 rows × 33 floats = 33.8 KB at 2 tiles and 4 column groups (`stream_gemm.cu:193-196`), above Apple's 32 KB of threadgroup memory (§4.5). The Metal kernel reduces in two fixed halves or with four simdgroups; the order is fixed and independent of M, which is all L1 needs.
- **No L2 prefetch.** CUDA's bulk L2 prefetches (`stream_gemm.cu:822-839`, `attention_body.cuh:120-127`) have no MSL equivalent. The Metal kernel keeps more chunks in flight per thread and more threadgroups per core instead; M0's microbenchmark sizes both.
- **Shapes baked per GPU family.** As on CUDA, tiles per threadgroup and chunks in flight move bytes, never bits (`stream_gemm.cu:616-629`); they are chosen once per GPU family at M0/M9 and compiled in (function constants), with no runtime heuristic ([ARCHITECTURE §8](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md), stage 5).

### 3.5 The FFN

CUDA's stage 1 runs gate|up, SiLU·up with each row's maximum, the FP16 copy and the down projection as one persistent kernel whose phases are separated by grid-wide counters (`stream_gemm.cu:841-851`, `stream_gemm.cu:965-1040`). It was measured bit for bit equal to the four-kernel graph and about as fast (x1.002–1.005, [MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), B32). Metal has no guarantee that all threadgroups of a grid are resident at once (§4.6), and CUDA's version waits for all G blocks at each phase (`stream_gemm.cu:985`), sizing G from an occupancy query (`stream_gemm.cu:1049-1057`) that Metal does not have. The Metal FFN is therefore:

1. one dispatch: gate|up tiles; each tile's epilogue writes SiLU(g)·u (the same expression, `stream_gemm.cu:995-997`) and folds its rows' maxima into a per-row atomic maximum (order-free, as `stream_gemm.cu:999-1002`);
2. one dispatch: the down projection, converting each activation to FP16 under its row's scale as it loads it. Converting on load gives the same FP16 values as a separate pass, since each element is converted on its own (`stream_gemm.cu:512-516` makes the same argument for the wide copy), so the 17,408-wide FP16 activation is never written.

A persistent form behind a switch takes its tiles from an ordered ticket queue, the form CUDA's attention megakernel uses because it completes with any number of resident blocks (`stream_gemm.cu:1459-1478`, `stream_gemm.cu:1234-1241`). It ships only if M3 measures a gain of at least 3% on the step.

### 3.6 The DeltaNet mixer and the recurrence as a persistent grid

CUDA's stage 2 overlaps the mixer's latency-bound chain (a/b projections, gating, conv, recurrence) with the in_qkv and in_z weight streams in one persistent kernel (`stream_gemm.cu:1074-1086`, `stream_gemm.cu:1257-1429`), for chains and trees, on by default (`engine.cpp:1205-1210`). On Metal the same overlap comes from a compute encoder whose dispatches may run concurrently, with explicit barriers only where data flows:

| Stage | Runs beside | Waits for |
|---|---|---|
| in_qkv tiles (F8M stream GEMM) | the a/b projections (BF16, 48 outputs) and the gating | the input norm |
| the conv (tree: each row's taps from its ancestors, `stream_gemm.cu:1298-1313`), then the recurrence | the in_z tiles | in_qkv; the gating |
| the gated RMSNorm of each head row, with the per-row maximum | — | the recurrence; in_z |
| the out-projection, converting to FP16 on load, adding into the residual | — | the norm |

**The recurrence.** CUDA gives each (value head, 32 value columns) item to one warp, 192 items, each lane holding one column's 128 FP32 state values in registers (`kernels.cu:399-413`, `stream_gemm.cu:1185-1232`). On Apple GPUs a 128-register state per thread may spill; M0 measures it. The fallback mapping gives an item to four simdgroups, each holding 32 of the 128 key rows for its 32 columns, and adds the four partial dot products in a fixed order through threadgroup memory (16 KB per item: inside the 32 KB limit). The verify reads the state and leaves it unwritten; the commit replays the accepted rows through the same kernel and writes it, as on CUDA (`speculate.cpp:508-521`). The tree verify walks gdn paths as `TreeDesc` describes them (`kernels.cuh:104-125`).

**As a persistent grid.** Behind a switch, the whole mixer runs as one persistent grid: each threadgroup takes items from ordered queues (in_qkv tiles, then the chain's items, then in_z tiles, then norm rows, then out-projection tiles), and an item waits only on items with smaller tickets. That is the queue form of `stream_gemm.cu:1287-1294` and `stream_gemm.cu:1348-1354` extended to every phase, so it cannot deadlock however many threadgroups are resident. A wait that exceeds its spin budget writes an error record and exits; the host turns the record into a failed request, the counterpart of CUDA's 50 ms deadline and trap (`stream_gemm.cu:865-882`). It ships only if M3 measures it at least 3% faster than the concurrent dispatches.

### 3.7 Attention

The same algorithm as `attention_body.cuh:91-97`: per KV head, its 6 query heads × M rows; fixed key ranges at fixed positions, combined in range order, with empty ranges skipped (`kernels.cu:597-627`); BF16 products, FP32 accumulation and online softmax; the output times sigmoid(gate). What changes:

- **Tiles.** CUDA stages two 32-key tiles of K and V in 99 KB of shared memory (`attention_body.cuh:125`, `kernels.cu:842`). Apple's 32 KB holds one 16-key stage of K and V at D = 256 (16.9 KB); the Metal kernel uses 16-key tiles, or loads its matrix operands straight from device memory, as M2 measures.
- **The range size.** 4,096 keys was measured best on the GB10 (`kernels.cu:826-831`). Any fixed size keeps row invariance, so the Metal size is chosen per GPU family (more ranges keep more of the GPU's cores busy at short contexts) and compiled in.
- **The tree** reads its per-variant tails as the CUDA kernels do (`kernels.cuh:104-112`), and the accepted path's KV and inputs are gathered by the same `GatherBuf` description (`kernels.cuh:132-141`); under P4 the gather computes addresses from the page table ([BATCHING §5.3](./DREAMFERENCE_LING_ENGINE_BATCHING.md), item 3).

### 3.8 Prefill

The CUDA prefill quantizes activations as production does, NVFP4 for the FFN and FP8 for the projections, and multiplies on block-scaled tensor cores (`prefill_gemm.cu:4-7`; the quantizer rounds to BF16 first, `prefill_gemm.cu:405-420`). Apple GPUs have no FP4 or FP8 matrix path (**TO VERIFY** for the M5's neural accelerators). The Metal prefill keeps the same quantization points and emulates the products: activations quantized to NVFP4 or FP8 by the same rule, both operands decoded to FP16 (exactly, §3.2), multiplied with FP32 accumulation, where the product of two FP16 values is exact, then scaled by `alpha` at the end as CUDA does. Each product is the one the block-scaled MMA forms; only the summation order differs. The chunked DeltaNet recurrence keeps 32-token chunks at absolute positions (`kernels.cuh:429-434`), so `--chunk-check` and `--prefix-check` mean what they mean on CUDA. Prefill attention reuses the rows kernel's tiles, as CUDA's does ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), status of 9 October).

### 3.9 Sampling and accept

On the host, as on CUDA: the GPU computes each verified row's top K (`speculate.cpp:707-726`), and the host applies the Qwen chain (`speculate.cpp:72-102`), checks finiteness (`speculate.cpp:835-838`, `sampling.hpp:22-36`), walks the lattice and runs the greedy or rejection rule (`speculate.cpp:794-880`). These are shared files and need no port. With unified memory the top-K values, the candidates and lattice scores and, for `logprobs`, whole logits rows (`speculate.cpp:1012-1018`) are read in place from shared buffers instead of copied.

### 3.10 The drafter

Five sliding-window layers with NVFP4 projections, DFlash2's grouped convolutions, the fc over the target's features at layers 5, 19, 33, 47 and 61, the selector's codebooks (`drafter.hpp:1-32`), its ring of context KV (`engine.hpp:695-699`) and the sparse head (`kernels.cuh:270-316`). Its GEMMs use the same Metal stream GEMM as the target. The drafter only proposes; the verify decides (SPECULATION §10, "Exactness"), so its kernels need determinism (for seeded identity, L1) and acceptance within the CUDA engine's spread (L3), not agreement with CUDA's bits. Mightling 2.0's several drafters ([MULTI_DRAFTER](./DREAMFERENCE_LING_ENGINE_MULTI_DRAFTER.md)) need nothing more from Metal than the drafter ring per session that switching already uses.

### 3.11 Tree verify

The host builds the tree, describes it and accepts over it in shared code (`tree.hpp:1-19`; 16 rows: the chain's first 8 drafts plus 7 nodes, `speculate.cpp:418-423`, `engine.hpp:124-134`). `TreeDesc` is plain integers (`kernels.cuh:115-125`), so one header serves C++ and MSL, with a static assertion on its size. The device sees the description through a shared buffer the host writes before signalling the verify (§4.3) instead of a memcpy node (`speculate.cpp:709-711`). Exactness is CUDA's: each row is computed as a chain holding its path would compute it (`kernels.cuh:104-113`).

## 4. Apple-specific design

### 4.1 Unified memory without copies

The GB10's memory is unified too, but the CUDA engine still copies: weights through staging buffers (`model.cu:110-145`), ids and positions up (`engine.cpp:1039`, `speculate.cpp:931-936`), logits and top-K down (`engine.cpp:1427-1430`, `speculate.cpp:722-725`), through page-locked buffers that graphs' memcpy nodes read (`engine.hpp:562-578`). On Metal every such buffer is one shared-storage allocation that the host and the GPU address directly. Ordering comes from the command buffers' events (§4.3), not from copies.

- **Weights.** `ling-convert` (the compiler's tiling stage, [ARCHITECTURE §8](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md), item 4, for the Metal target) writes one blob with 16 KB-aligned tensors and a manifest of offsets and SHA-256 hashes. `ling-serve` maps the file and wraps it in no-copy Metal buffers, so loading is paging in. **TO VERIFY at M0:** that a file-backed no-copy buffer streams at the same rate as an anonymous allocation and stays resident under memory pressure once added to a residency set. If not, the blob is read once into a shared allocation at load: one copy at start, none while serving.
- **The token embedding** (2.54 GB in BF16, read 10 KB per token, [MODEL §5.2](./DREAMFERENCE_LING_ENGINE_MODEL.md)). On 36 and 48 GB Macs, the host copies each row of the step from the mapped checkpoint straight into the step's shared input buffer (the CPU writes what `embed` reads, `engine.cpp:1040`). Untouched rows then stay in the page cache, which macOS can reclaim, and the 2.54 GB never counts against the GPU's working set. On 64 GB and above the table is GPU-resident, as on CUDA. Both give the same FP32 values.
- **Device-side copies stay copies.** Saving and restoring a state slot, adopting another session's prefix (`engine.cpp:630-660`) and the store's KV copies (`engine.cpp:791-805`) duplicate data on purpose; they become blit commands, and P4's page sharing removes the KV ones.

### 4.2 Memory plan per Mac size

macOS caps what the GPU may keep resident: Metal's `recommendedMaxWorkingSetSize`, which is about 75% of RAM on recent machines (an estimate; **TO VERIFY** on each machine). The engine plans against that cap, never against all of RAM, and leaves the rest to macOS and the user's applications. Raising the cap with a system setting is possible but is the owner's decision (open question 11).

Fixed parts (GB = 10^9 bytes): target weights read per pass 17.33 with B26's coded scales ([LOWBIT §1](./DREAMFERENCE_LING_ENGINE_LOWBIT.md)); drafter 1.55 and its sparse-head file ~0.13; embedding 2.54 (or ~0, §4.1); scratch 4.29 at 2,048-token prefill chunks (`admission.hpp:33`), estimated at 2.1 at 1,024 and 1.1 at 512; checkpoint slots 0.157 each; vision 2.48 (tower 0.88, scratch 1.10, cache 0.5, [INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)). Per session: ~0.48 fixed (state, conv, end-of-prompt copy, drafter ring, [BATCHING §2.5](./DREAMFERENCE_LING_ENGINE_BATCHING.md)) plus 64 KB per token held, in 4 MiB pages.

| RAM | GPU cap (est., TO VERIFY) | OS and applications | Engine fixed | Vision | Prefix KV store | Sessions (fixed part) | KV pool | Default recipe (estimate) |
|---|---|---|---|---|---|---|---|---|
| 36 GB | ~27 GB | ~9 GB | 20.4 (weights 17.33, drafter 1.68, embedding on the host, scratch 1.1, 2 checkpoints) | off | off | 2 × 0.48 | ~5.6 GB ≈ 85K tokens | `--sessions 2 --max-context 32768`, prefill chunk 512 |
| 48 GB | ~36 GB | ~12 GB | 21.7 (embedding on the host, scratch 2.1, 4 checkpoints) | 2.48 or off | off | 2 × 0.48 | ~11 GB ≈ 165K tokens (vision on) | `--sessions 2 --max-context 65536`, vision on |
| 64 GB (Mac mini M5 Pro) | ~48 GB | ~16 GB | 27.1 (embedding resident, scratch 4.29, 8 checkpoints) | 2.48 | 2 | 2 × 0.48 | ~15.4 GB ≈ 235K tokens | `--sessions 2 --max-context 131072`, vision on |
| 128 GB | ~96 GB | ~32 GB | 27.1 | 2.48 | 4 | 4 × 0.48 | ~60 GB ≈ 920K tokens | `--sessions 4 --batch 4 --max-context 262144` (the GB10's P4 recipe) |
| 256 GB and up | ~192 GB | ~64 GB | 27.1 | 2.48 | 4 | 8 × 0.48 | ~154 GB ≈ 2.3M tokens | eight sessions at 262,144, subject to [BATCHING §2.6](./DREAMFERENCE_LING_ENGINE_BATCHING.md)'s eight-session rule and the owner's say |

Paged KV is what makes the small rows work: a session costs what it holds. A default may promise more context than the pool holds at once (at 64 GB, two windows of 131,072 against ~235K positions), and a full pool waits, reclaims idle sessions and, as a last resort, ends the most recently admitted request at `length` ([BATCHING §5.3](./DREAMFERENCE_LING_ENGINE_BATCHING.md), "A full pool"). The FP8 KV cache (P5, non-exact, behind the quality gate) would double the 36 GB row's pool. It stays off by default under the exactness policy (open question 7).

### 4.3 Command-buffer pipelining: the next step encoded while the current one runs

**What CUDA does.** A step is a draft graph, a host synchronization (the lattice walk), a verify graph, a synchronization (the accept), an optional tree gather, and a commit graph with the DeltaNet replay on a second stream, then a synchronization (`speculate.cpp:682-705`, `speculate.cpp:912-1036`; the waits at `speculate.cpp:690`, `:969`, `:1029`). Graphs read positions from device memory (`kernels.cuh:14-43`) and launch the attention ranges of the whole `max_context`, empty ones exiting at once (`speculate.cpp:28-29`), so one graph serves every step of its shape.

**What Metal does.**

- **Encoded once.** Each step part (draft for block B; verify per top-K, chain or tree, batch size; tree gather; commit) is encoded at start into an indirect command buffer with explicit barriers between dependent dispatches. The variable data lives in shared buffers the kernels read: positions, rows, `TreeDesc`, the accepted count `n`. The commit reads `n` from a buffer, so one encoding replaces CUDA's commit graph per count (`engine.hpp:750`). Encoding at start, inside the memory plan, is also BATCHING §5.2's rule that no allocation after start depends on which sessions meet.
- **Committed ahead.** Step n's command buffer is `[wait accept(n−1)] → tree gather(n−1), drafter KV for the accepted rows(n−1) → { DeltaNet replay(n−1) ‖ draft(n) } → signal draft(n) → [wait rows(n)] → verify(n) → signal verify(n)`. The replay runs beside the next draft because the draft reads none of the recurrent state (the reason for `engine.hpp:113-117`). The host commits step n+1's command buffer while step n runs; it then blocks on the GPU's own event wait, so the GPU never waits for encoding or submission, only for the host's two decisions, which take microseconds (`speculate.cpp:19-23`).
- **Waking up.** The host learns that a draft or a verify is done from a shared event and answers by writing the shared buffers and signalling. The latency of a CPU-signalled event releasing a GPU wait is measured at M0 (**TO VERIFY**); the target is a host gap of ≤ 1 ms per step ([OVERVIEW §7](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md), host overhead).
- **Batches** use the same command buffers per batch size, reading each slot's session (page table, state pointers, position) from a descriptor table, which is BATCHING §5.2's design and bounds the set of encodings.

### 4.4 Simdgroup matrices and the M5's neural accelerators

- **Simdgroup matrices** (8 × 8 FP16, BF16 and FP32 operands with FP32 accumulation, on every Apple GPU this port targets) are the baseline matrix unit. A decoded weight tile is the 8 × 8 left operand and the rows of `x` are the right operand, mirroring CUDA's weights-as-A choice (`stream_gemm.cu:311`).
- **The neural accelerators**, one per GPU core on M5-generation chips, reached from Metal 4 through tensor operations inside our own kernels, are what makes a 16-row verify affordable (§5.3). M0 answers four questions before the port depends on them (**all TO VERIFY**): (1) FP16 operands with FP32 accumulation; (2) row invariance, i.e. whether a row's result is bit for bit the same at every M from 1 to 32 (the memcmp test of `ling-stream-tests`, run on the probe); (3) the rate at skinny shapes (N = 8 to 32 rows against K = 5,120 or 17,408) with weights decoded in registers or threadgroup memory; (4) whether any block-scaled 4-bit or 8-bit operand type decodes to exactly NVFP4's or E4M3's values. Rule: the rows path uses the accelerators for every row count if (1) and (2) pass and (3) beats simdgroup matrices; otherwise simdgroup matrices for every row count, and the default verify shrinks (§5.3). The prefill (more than 32 rows) may use the accelerators independently of that choice, since it is a different path with its own chunk invariance, as CUDA's is (`engine.hpp:6-12`).
- **Machines without accelerators** (M3- and M4-generation Mac Studios and MacBook Pros) run the simdgroup path with smaller default verify shapes. Whether they are supported at all is open question 2.

### 4.5 Threadgroup memory and registers

Apple's feature tables give 32 KB of threadgroup memory per threadgroup and 1,024 threads at most, with 32-lane simdgroups (**TO VERIFY** on the M5's family: M0 reads `maxThreadgroupMemoryLength`). The CUDA kernels that exceed 32 KB:

| CUDA kernel | Its shared memory | Metal change |
|---|---|---|
| Stream GEMM reduction at 2 tiles, 4 column groups | 33.8 KB (`stream_gemm.cu:193-196`) | Two-pass or four-simdgroup reduction, fixed order (§3.4) |
| Rows attention, two 32-key K/V stages | 99 KB (`attention_body.cuh:125`) | 16-key stages or direct device loads (§3.7) |
| Attention megakernel | the same ring (`stream_gemm.cu:1696`) | Not ported |
| Prefill GEMM, 3–4 stages of 128 × 256 tiles | above 32 KB (`prefill_gemm.cu:9-13`) | Smaller tiles, two stages; sized at M4 |
| Recurrence, 128 FP32 state values per thread | registers (`kernels.cu:411-413`) | Measured at M0; fallback mapping §3.6 |

Apple GPUs from the M3 generation allocate registers and on-chip memory dynamically, so occupancy depends less on a kernel's peak register count than on CUDA (**TO VERIFY** that the 128-value state does not spill on the M5).

### 4.6 Persistent grids and forward progress

Metal promises no co-residency of a grid's threadgroups, exposes no occupancy query and has no cooperative launch. A kernel whose threadgroups wait on each other through counters may therefore wait for a threadgroup that has not started and never will. Hence: (1) the default path is dispatches in one encoder, with dependent ones separated by barriers and independent ones concurrent (§3.5, §3.6); (2) any persistent kernel uses ordered ticket queues, where an item waits only on items already taken by running threadgroups (`stream_gemm.cu:1459-1478` is the CUDA precedent), never a static partition with grid-wide barriers; (3) each wait has a spin budget and an error record (§3.6). macOS's GPU watchdog, which ends a command buffer that runs too long, is a coarser backstop. The per-dispatch gap between dependent dispatches is measured at M0, since it decides whether a persistent form can gain at all (the GB10 left "no bubble to take" under graphs and PDL, [MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), B32).

### 4.7 Power and thermal behaviour

- **Telemetry.** `/metrics` adds `ling:thermal_state` (macOS's nominal, fair, serious, critical), `ling:on_battery`, `ling:low_power_mode` and `ling:gpu_working_set_bytes`; every bench run records them beside the step time, as MILESTONES §15's thermal risk asks of the GB10.
- **No adaptive numerics.** The engine never changes its row shares, block size or kernels with temperature or power: that would change seeded outputs between runs. It reports; the scheduler may only delay admitting new prefills when the state is `critical`.
- **Sleep.** While a request runs, `ling-serve` holds a power assertion against idle sleep and releases it when idle, so an idle Mac sleeps and stays cool (the counterpart of [INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)'s "idles without spinning").
- **Measured, not assumed:** 10-minute and 60-minute decode runs on the Mac mini (M7), the MacBook Pro on battery and on power (M9), and tokens per joule against the GB10's 140 W budget ([MILESTONES §15](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), thermal risk), read by the owner with a power meter or `powermetrics` in a bench window. No figure is promised before then.

### 4.8 Host memory safety and admission on macOS

The Linux rules carry over in meaning ([BATCHING §5.6](./DREAMFERENCE_LING_ENGINE_BATCHING.md), [VALIDATION §18](./DREAMFERENCE_LING_ENGINE_VALIDATION.md), "Unified memory is not device memory"): plan everything at start, refuse to start (exit 3) when the plan would leave less than the reserve, answer 503 `host_memory_low` while memory is short, and allocate nothing that depends on traffic after start. The inputs differ: the plan is checked against the GPU cap and the device's current allocation; "memory is short" is macOS's memory-pressure level (warning or critical) instead of MemAvailable (`admission.hpp:72-80`). Mightling's PSI watchdog has no macOS equivalent; its launcher subscribes to memory-pressure events instead.

## 5. Expected performance

### 5.1 Method

A speculative step's time is the larger of its bytes over the achieved bandwidth and its arithmetic over the achieved matrix rate, plus what neither hides (latency-bound kernels, dispatch gaps, the host's decisions). Tokens per second are accepted tokens per step divided by step time. Acceptance depends only on the two models' numbers, which L3 keeps equal, so it is taken from the GB10 unchanged: 6.749 tokens per step on agentlong at one stream with tree verify ([LOWBIT §1](./DREAMFERENCE_LING_ENGINE_LOWBIT.md)). The GB10's step is measured, not modelled: 105.86 ms (draft 10.28, verify 94.84, commit 0.81) for 63.75 tokens/s at `532537b`, and 102.64 ms for 65.67 tokens/s after the sparse head ([BACKLOG §19](./DREAMFERENCE_LING_ENGINE_BACKLOG.md), item 19). One verify pass reads 20.35 GB at agentlong's mean 41.3K context: weights 17.33, BF16 KV 2.71, state 0.31 ([LOWBIT §1](./DREAMFERENCE_LING_ENGINE_LOWBIT.md)); the draft reads another 1.6 GB.

### 5.2 Decode, bound by bytes

| Machine | Bandwidth, published | Achieved (est.: 80–90% of published, M0 measures) | Verify bytes at 41.3K, floor | Draft bytes | Notes |
|---|---|---|---|---|---|
| GB10 (measured) | 273 GB/s | 262 stream (CUDA M0); 205–244 in the GEMMs (`stream_gemm.cu:620-625`, LOWBIT §1) | 87 ms at 234 GB/s (LOWBIT §1) | 7 ms | verify measured at 94.84 ms |
| Mac mini, M5 Pro, 20-core GPU | 307 GB/s (**TO VERIFY**) | 246–276 | 74–83 ms | 6–7 ms | needs the accelerators for 16 rows (§5.3) |
| MacBook Pro or Mac Studio, M5 Max, 40-core GPU | 614 GB/s (**TO VERIFY**) | 490–550 | 37–42 ms | 3 ms | fixed costs weigh more (below) |
| Mac Studio, M5-generation Ultra | **TO VERIFY**: availability and bandwidth | — | — | — | two dies: cross-die traffic measured at M9 |
| Mac Studio, M3 Ultra / M4 Max (no accelerators) | 819 / 546 GB/s, published | — | — | — | compute-bound at 16 rows (§5.3) |

**Estimate for the M5 Pro, one stream, agentlong.** If the 16-row verify's arithmetic hides under its bytes: verify 74–83 ms plus ~5 ms not hidden; draft 6–7 ms plus its compute; commit ~1 ms; host ≤ 1 ms. That is a ~90–100 ms step and **~67–75 tokens/s against the GB10's 65.67**: about a GB10. Without the accelerators, at 8 verify rows (§5.3: ~78 ms of arithmetic against 74–83 ms of bytes), the step is about the same, ~90–100 ms, but accepts fewer tokens, an estimated 40–55 tokens/s (the tokens per step at 8 rows are measured at M2, not assumed).

**Fixed costs.** The GB10's step carries ~8 ms that is not bytes (LOWBIT §1), and 333 launches per step ([ARCHITECTURE §9](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md)). At an estimated 2–10 µs per dependent dispatch gap on Apple GPUs (M0 measures it), dispatch gaps alone are 0.7–3.3 ms. On a Max, where the bytes take half as long, such fixed costs are a larger share; the Max's decode is therefore estimated at 1.6–1.9× the GB10's, not the 2× its bandwidth suggests.

### 5.3 The multi-row verify's cost

A row costs 2 × 24.4 G = 49 GFLOP ([OVERVIEW §4](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)), so a verify of M rows costs 49·M GFLOP against 17.33 GB of weights. Bytes and arithmetic take equal time at

  M* = C · W / (2 · P · R) ≈ 0.36 · C / R,

with C the achieved matrix rate, R the achieved bandwidth, W the weight bytes and P the parameters. Below M* rows are free; above it each row adds time. On the GB10 compute starts to add only beyond about 100 rows ([OVERVIEW §4](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)), which is why its verify budget is set by bytes and by the stream GEMM's tiers (32 rows, `kernels.cuh:102`), not by arithmetic.

| M5 Pro, matrix path | Peak FP16 (est., TO VERIFY) | Achieved at 8–32 rows (est.) | M* at R = 260 GB/s | A 16-row verify's arithmetic (0.78 TFLOP) |
|---|---|---|---|---|
| Simdgroup matrices | 8–9 TFLOPS (20 cores × 128 lanes × 2 × ~1.7 GHz) | ~5 TFLOPS | ~7 rows | ~155 ms: compute-bound, twice the bytes |
| Neural accelerators | 30–36 TFLOPS (Apple's "over 4×" AI compute claim for M5 over M4, applied per core) | ~15–18 TFLOPS | ~21–25 rows | ~43–52 ms: hidden under 74–83 ms of bytes |

Two consequences. (1) On an M5 the CUDA engine's verify shapes (16-row tree at one stream; a batch fits `32 / draft_block` sequences, so two of 16 rows, `main.cpp:489`, `engine.hpp:57`) stay bandwidth-bound only on the accelerators; on simdgroup matrices the default becomes an 8-row chain and the tree is off. The shape is chosen per GPU family at M3 by measured tokens per second; greedy output does not depend on it. (2) Cores and bandwidth grow together from Pro to Max to Ultra (20, 40 and 80 cores against 307, 614 and twice that), so M* is a property of the generation, not of the size: a bigger Mac does not buy a bigger verify. The dequantization's integer work (an estimated 2–3 operations per weight, once per pass however many rows) is also counted at M0; it shares the ALUs with simdgroup matrices but not with the accelerators.

### 5.4 Prefill, bound by compute

The GB10's prefill reached 2,640 / 2,580 / 2,110 tokens/s at 1K / 8K / 35K tokens on its FP4 and FP8 tensor cores ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md), status of 9 October), at 48.7 GFLOP per token ([SESSIONS §11](./DREAMFERENCE_LING_ENGINE_SESSIONS.md)): about 103–129 TFLOPS achieved. The Metal prefill multiplies FP16 operands (§3.8).

| Machine | FP16 matrix peak (est., TO VERIFY) | Achieved at 60–75% (est.) | 35K prefill (est.) | Against the GB10's 2,110 |
|---|---|---|---|---|
| M5 Pro, accelerators | 30–36 TFLOPS | 18–27 TFLOPS | 370–550 tokens/s | 3.8–5.7× slower |
| M5 Max, accelerators | 60–72 TFLOPS | 36–54 TFLOPS | 740–1,100 tokens/s | 1.9–2.9× slower |
| M5 Pro, simdgroup matrices only | 8–9 TFLOPS | 5–6 TFLOPS | 100–125 tokens/s | 17–21× slower |

The owner's 2–4× is therefore the Max class's figure. For the Pro to reach the 4× end it needs ≥ 26 TFLOPS sustained on these shapes; for 2×, ≥ 51. M0 measures the accelerators' rate and M4 the prefill itself. A 35K-token cold prompt then takes about 65–95 s on the Pro against ~17 s on the GB10. Resident sessions and the prefix stores, which keep warm turns to a few hundred new tokens ([OVERVIEW §2](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)), matter more on a Mac than on the GB10.

### 5.5 Why decode approaches a GB10 and prefill does not

Decode reads the same 20 GB per step on both machines, because the formats and the model are the same, and accepts the same tokens per step, because the numbers are the same up to summation order. Its speed is therefore the bandwidth ratio times the kernels' efficiency: 307 against 273 GB/s nominal, both unified LPDDR5x pools with the CPU on the same bus. That holds as long as the verify's arithmetic stays below the byte time, which on an M5 is the neural accelerators' job (§5.3). Prefill reads each weight once per chunk of thousands of tokens and is bound by matrix throughput. There the GB10 multiplies FP4 and FP8 on tensor cores rated at about 500 and 250 TFLOPS dense ([OVERVIEW §4](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md)), while an M5 multiplies FP16 at an estimated 30–70 TFLOPS. The gap in matrix throughput, not in bandwidth, sets the prefill ratio.

### 5.6 What is measured first

M0 measures, in this order: the stream bandwidth; the dequantizing GEMM at 1, 8, 16 and 32 rows per format (N4M with raw and coded scales, F8M; LOWBIT §6's format-against-format method), on simdgroup matrices and on the accelerators; the accelerators' row invariance; the dependent-dispatch gap and event wake-up latency; the recurrence's register use; the no-copy weight buffers; and `recommendedMaxWorkingSetSize`. §5.2–§5.4 are restated against these numbers before M1 starts, as CUDA M0 restated the GB10's targets.

## 6. Build, test, CI, packaging and the egress audit

### 6.1 Build

- **Toolchain.** CMake and Ninja; Apple Clang in C++20 for the host; Apple's C++ interface to Metal (metal-cpp), so the host stays C++20 without Objective-C, as ARCHITECTURE §8's one-language rule wants; the Metal compiler from Xcode's command-line tools (**TO VERIFY** whether the M5's Xcode ships it as a separate component). `LANGUAGES CXX CUDA` and `sm_121a` (`CMakeLists.txt:2`, `CMakeLists.txt:13`) become a configure-time target: `-DLING_TARGET=cuda` (today's build, unchanged) or `-DLING_TARGET=metal` (no CUDA language enabled).
- **Shaders.** Compiled offline into one `.metallib` per GPU family. Shape specialization uses function constants and template instances named by the host code, one instance per (matrix, format, row bucket), as [ARCHITECTURE §8](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) stage 5 instantiates CUDA's; each shader source is its own build unit, so one kernel's change recompiles one file. Nothing is compiled at run time.
- **Flags enforced.** Fast math off, precise functions, contraction off (§2); a CMake check fails the build if any shader's command line lacks them. Debug builds add line tables and source for Xcode's shader debugger.
- **Dependencies.** nlohmann_json, PCRE2, ICU (Apple's system ICU has no public headers: Homebrew's or a vendored copy), libpng, libwebp, libjpeg-turbo (not Apple's ImageIO, so decoded pixels match the Linux build's), and proxygen with folly, fizz and wangle built once with their own `getdeps`, as on Linux (`CMakeLists.txt:92-105` finds them under a configured prefix). That proxygen builds on macOS arm64 is checked at M0 (open question 8).
- **Command line.** The same flags ([CLI](./DREAMFERENCE_LING_ENGINE_CLI.md)). `--graphs` selects the pre-encoded step; `--pdl`, `--attn-bulk` and `--attn-prefetch` are accepted and ignored with one startup line, so Mightling's recipe needs no macOS fork; `--side-prefill N` takes only `0` or `-1` (§3.1).

### 6.2 Tests

| CUDA test | Metal |
|---|---|
| `ling-sampling-tests`, `ling-spec-tests`, `ling-prefix-kv-tests`, `ling-api-tests`, `ling-tokenizer-threads-tests`, `ling-vision-input-tests` (`CMakeLists.txt:108-141`) | The same sources, built and run unchanged |
| `ling-stream-tests` (memcmp of each kernel's rows across M, batched against alone) | `ling-metal-stream-tests`: every rows-path kernel at M = 1..32 bit for bit per row; the sequence-batched kernels against each sequence alone; paged against flat KV; the ICB against eager dispatch |
| `ling-kernel-tests`, `ling-tree-tests`, `ling-mk-tests`, `ling-prefill-tests`, `ling-vision-tests` | Ported: each kernel against an FP64 host reference within a stated tolerance, plus the invariance memcmps; the persistent forms (if built) against the dispatch forms bit for bit |
| `ling-run` checks (`tools/ling_run.cpp:349-365`) | The same checks; `--icb-check` for `--graph-check` |
| `tests/server` (pytest against a running server, `tests/server/test_regressions.py` and the rest) | The same suite against the Metal `ling-serve`: the same passes and the same xfails as on Linux |
| — | `ling-l3-check`: the L3 gate (§2) against the pinned CUDA reference; `mirror-check`: the mirrored function pairs (§3.1) changed together |

Debug runs enable Metal's API and shader validation (the counterpart of `compute-sanitizer`); the host suite runs under AddressSanitizer and UndefinedBehaviorSanitizer as on Linux ([ARCHITECTURE §8](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md)).

### 6.3 CI: the Mac mini as a self-hosted runner

Today's CI is the CLA check only (`.github/workflows/cla.yml`); GPU tests run by hand on the development GB10. The Mac mini becomes the first runner with a GPU.

- **Jobs.** On every push to a branch of this repository: build (Release and Debug), the host suites, the Metal kernel suites. Nightly, and on demand by a maintainer: the `ling-run` checks with the real checkpoint, `tests/server`, agentlong greedy (a 4-turn subset nightly, 16 turns weekly), `ling-l3-check`, and the egress audit (§6.5). On tags: packaging (§6.4).
- **Hosted runners.** GitHub's hosted macOS runners are virtual machines without a usable GPU (**TO VERIFY**); they may build and run the host suites only.
- **Safety of a self-hosted runner on a public repository.** Jobs never run on pull requests from forks; a maintainer runs them after review. The runner is a dedicated standard (non-administrator) macOS account with no credentials beyond the runner's own registration, read-only access to the model files, and a workspace wiped after each job. GPU jobs take the same GPU lock as every bench window, so CI never runs inside a measurement.
- **Not a production server.** The Mac mini serves no one until M7's night says it may (open question 13).

### 6.4 Packaging into Mightling releases (Ragon from 1.9.0)

The port ships in 2.0, so its first release is a Ragon release; artifact names stay product-neutral (`ling-serve`, `ling-run`, `ling-convert`) so the rename touches only the release and the launcher.

- **The release asset.** `ling-engine-macos-arm64-<version>.tar.gz`: the three binaries, the `.metallib` files, the licence and the notices. Signed with a Developer ID and the hardened runtime, notarized, and pinned by SHA-256 in the release's signed `SHA256SUMS`, as the drafter assets are ([MULTI_DRAFTER](./DREAMFERENCE_LING_ENGINE_MULTI_DRAFTER.md), release assets). Where the signing keys live is open question 10.
- **The weights.** Not shipped: the converted blob is over 17 GB, far above a release asset's 2 GiB. `ling-convert` runs once on the user's Mac from the Hugging Face checkpoint already in the local cache, reads only local files, and is deterministic, so its output's SHA-256 is pinned in the model registry beside the checkpoint's. A mismatch refuses to start.
- **Launching.** On Linux the registry selects the engine and a container runs it ([INTEGRATION §12](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md)); macOS containers have no GPU, so on a Mac the launcher runs `ling-serve` as a native process with the recipe of §4.2's table, pre-load checks against §4.8's inputs, and the registry's `launch_overrides['engine'] = 'ling'` extended with the platform. SGLang has no macOS fallback; the fallback is the previous Ragon release's engine.

### 6.5 The egress audit

`ling-serve` on macOS opens one listening socket and makes no outbound connection, no DNS lookup and no telemetry call; `ling-convert` and `ling-run` open no socket at all. Mightling's audit rule ("`ling` talks only to its node", [drafts/two-node-tp §8](./drafts/two-node-tp-2026-10-12.md)) is unchanged. The audit job checks it three ways:

1. **Static:** the binaries' undefined symbols include no URL-loading or socket-connect API beyond the server's accept path, and they link no HTTP client library.
2. **Confined run:** `ling-serve` runs `tests/server` and the agentlong subset under a macOS sandbox profile that denies outbound network except loopback replies; any denied attempt fails the job.
3. **Observed run:** the process's sockets are sampled every second during the run, and the only ones allowed are the listening socket and accepted loopback connections.

## 7. Milestones

Each milestone ends at a measured gate on the Mac mini, merges into main only when its exactness gate passes and nothing earlier regresses, and never changes the CUDA build. Efforts are for one engineer with review and are estimates. Code starts 2026-11-18; M0's probes (bench code under `bench/metal/`, no engine code) may run from the Mac's arrival if the owner agrees (open question 12).

| Milestone | What | Exactness gate | Measurable target | Effort (est.) |
|---|---|---|---|---|
| **M0** Platform | §5.6's measurements; the N4M permutation probe; proxygen on macOS; the toolchain and CI runner | Probe kernels: matrix-unit row invariance memcmp at M = 1..32 (decides §4.4) | Bandwidth, GEMM rates, dispatch gap and wake latency recorded in `reports/metal-M0.md`; §5 restated; go if the dequantizing GEMM reaches ≥ 80% of the measured stream bandwidth at 1 and 16 rows | 1 week |
| **M1** Load and plain decode | `ling-convert`; the mapped weights; every rows-path kernel; one session, eager, plain greedy decode and sampling; `ling-run` | Per-kernel references; row invariance memcmp for every kernel; greedy deterministic across repeats; the L3 gate against the CUDA engine (§2), perplexity ratio included | One-row decode at ≥ 85% of M0's stream bandwidth (bytes per token over time) | 3 weeks |
| **M2** Speculation, exact | Drafter, sparse head, chain verify, tree verify, accept, commit by replay, lookup | `--spec-check` (chain, tree, lookup), equal state hashes; the joint-distribution test of `ling-spec-tests` | Tokens per step on agentlong within the CUDA engine's run-to-run spread at the same verify shape; the per-family verify shape chosen (§5.3) | 3 weeks |
| **M3** The step pipeline | ICBs, shared events, steps committed ahead, the replay beside the draft; the concurrent DeltaNet and FFN forms; the persistent forms measured | `--icb-check` bit for bit; `--spec-check` with the pipeline on | Host gap ≤ 1 ms per step; agentlong one-stream decode ≥ 90% of §5's restated byte model (estimate before M0: 67–75 tokens/s on the M5 Pro with accelerators) | 2 weeks |
| **M4** Prefill | Prefill GEMMs with CUDA's activation quantization emulated; chunked DeltaNet; prefill attention; checkpoints at message ends | `--chunk-check`, `--prefix-check`; the L3 gate's prefill positions (the first token after the prompt) | 35K-token prefill rate recorded against M0's restated estimate (370–550 tokens/s before M0) and against the GB10 | 3 weeks |
| **M5** Server, sessions, batches | `ling-serve` on macOS; sessions with paged KV (P4 layout); prefix KV store; macOS admission; batches of two; prefill on a second queue | `tests/server` as on Linux; `--batch-check`, `--interleave-check`; agentlong greedy 16/16 at two streams against one | Memory grows ≤ 1 GB after start over a 1-hour run; two-stream aggregate decode reported against the GB10's | 2.5 weeks |
| **M6** Parity of features | The vision tower; structured output and whatever else the CUDA engine has merged by then | `tests/server/test_vision.py`; prompts token-identical to the Linux build's on image requests; the L3 gate on image prompts | Image prefill time recorded | 1.5 weeks |
| **M7** Quality and endurance | One SWE-bench night served by the Mac against the GB10 engine; an 8-hour soak; 10- and 60-minute thermal runs | The L3 gate on the release candidate; the night with no significant loss (McNemar) | No restart in 8 hours; step-time drift over 60 minutes reported with the thermal state | 1 week plus nights |
| **M8** Release | Signing, notarization, `SHA256SUMS`; the converter's pinned hash; the launcher on macOS; the egress audit; docs | The release checks; the egress audit (§6.5) | A clean install and start on a 64 GB and (or emulated by a GPU cap) a 36 GB Mac within §4.2's plan | 1 week |
| **M9** Other Macs | M5 Max (MacBook Pro, Mac Studio), an Ultra, 36 GB; shapes per GPU family; battery behaviour | L1 checks and the L3 gate per GPU family | Decode and prefill per machine against §5's restated estimates | 1–2 weeks per family, as hardware arrives |

To the first release (M0–M8) that is about 18 weeks plus nights, close to the four months the CUDA engine was estimated at ([MILESTONES §14](./DREAMFERENCE_LING_ENGINE_MILESTONES.md)); from 2026-11-18 it reaches M8 around the end of March 2027 (an estimate). M1 is the long pole, as CUDA M1 was: everything after it multiplies the pass time it establishes.

## 8. Risks

| Risk | Effect if it lands | Mitigation | Decided by |
|---|---|---|---|
| The neural accelerators are not row-invariant, or not reachable with weights decoded in our kernels | Verify on simdgroup matrices: 8-row chain, decode ~40–55 tokens/s on the Pro (est.) instead of ~67–75 | M0 probes before any design depends on them; the simdgroup path is complete on its own | M0 |
| Achieved bandwidth well below 85% of 307 GB/s | Decode scales down with it | Measured first; the tile layout and chunks in flight tuned at M0 | M0 |
| File-backed weight buffers page out under pressure | Steps stall on page faults | Residency set; fallback to one read at load | M0 |
| Dependent-dispatch gaps of tens of µs | Several ms per step | Concurrent dispatches, fewer barriers, persistent ticket forms | M0, M3 |
| A macOS update changes kernel bits (L2) | Golden files differ after an update | L1 checks are self-referential and unaffected; golden files keyed by OS build; the L3 gate re-run | Standing |
| proxygen does not build on macOS arm64 | No server | Built at M0; fallback is open question 8 | M0 |
| 36 GB machines leave too little KV after the OS | Contexts below 32K | Embedding on the host, prefill chunk 512, no vision; FP8 KV only through its quality gate | M5, M9 |
| Thermal throttling on the MacBook Pro | Step time drifts up | Reported, not adapted (§4.7); 60-minute runs | M7, M9 |
| The mirrored host logic drifts from the CUDA engine's | Different behaviour behind the same API | `mirror-check` in CI (§6.2) | Standing |
| A self-hosted runner on a public repository runs untrusted code | Compromise of the Mac | No fork pull requests on the runner; a dedicated unprivileged account; no secrets (§6.3) | M0 |

## 9. Open questions for the owner

Each has a recommended default; the default stands unless the owner rules otherwise.

1. **Cross-platform exactness.** Accept L3 (same weights and quantization points, a statistical gate) as the definition of "the same model" across CUDA and Metal, with no attempt at bit identity? *Default: yes.* Bit identity would need CUDA's exact summation order on Apple's matrix units, which is not possible without giving up those units.
2. **Which Macs.** M5 generation first-class; M3 and M4 generations supported on the simdgroup path with smaller verify shapes; M1 and M2 unsupported? *Default: yes*, with M3/M4 support decided after M9 measures one such machine.
3. **The mirrored host logic.** Copy session binding, the checkpoint policy and the step planner into the Metal engine (no CUDA change), or extract them into shared plain files both builds compile? *Default: copy, guarded by `mirror-check`;* extraction reconsidered after M5, only for functions that are identical in both.
4. **The accelerators' rule.** Use them on the rows path only if they pass the M = 1..32 memcmp, otherwise simdgroup matrices and an 8-row chain? *Default: yes;* never two arithmetics on the rows path.
5. **Paged KV from the first session milestone**, whatever CUDA's P4 status? *Default: yes.*
6. **The token embedding on the host** for 36 and 48 GB Macs, GPU-resident above? *Default: yes.*
7. **FP8 KV on small Macs.** *Default: off* (non-exact); available only after CUDA's P5 passes the quality gate and the owner rules.
8. **The HTTP library on macOS.** proxygen as on Linux, so `main.cpp` is shared? *Default: proxygen;* if `getdeps` fails on macOS at M0, the owner chooses between fixing the build and a second server front end.
9. **Where the code lives.** The same repository, `src/metal/` and `-DLING_TARGET=metal`? *Default: yes:* one specification, shared host files, one CI.
10. **Signing.** Developer ID signing and notarization in a release job on the Mac mini under a maintainer's manual approval, or on the owner's own machine? *Default: on the owner's machine;* the runner holds no signing credentials.
11. **The GPU cap.** Stay within macOS's default `recommendedMaxWorkingSetSize`, or document raising it on dedicated machines? *Default: stay within it;* raising it is the user's documented choice, never the launcher's.
12. **Probes before 2026-11-18.** May M0's bench code run from the Mac's arrival (2026-11-11), while engine code waits for 1.x to finish? *Default: yes.*
13. **The Mac mini's role.** CI runner and bench lane, and Mightling-on-Mac dogfooding only after M7's night? *Default: yes;* never a production server before M7.
14. **The persistent forms (megakernel stages).** Leave them out of the first release unless M3 measures ≥ 3% on the step? *Default: yes*, the same bar CUDA's stage 2 was held to.
15. **Battery.** Serve normally on battery and in Low Power Mode, report it, and hold no assertion while idle? *Default: yes;* refusing to serve on battery is a launcher setting if the owner wants it.

## Sources

- This repository at `5a88252`: `src/core/stream_gemm.cu`, `kernels.cuh`, `kernels.cu`, `attention_body.cuh`, `prefill_gemm.cu`, `prefill_gdn.cu`, `engine.hpp`, `engine.cpp`, `speculate.cpp`, `accept.hpp`, `tree.hpp`, `sampling.hpp`, `lookup.hpp`, `prefix_kv.hpp`, `drafter.hpp`, `vision.hpp`, `model.cu`, `safetensors.cpp`, `launch.cuh`; `src/serve/main.cpp`, `api.hpp`, `admission.hpp`, `image_input.cpp`; `tools/ling_run.cpp`; `CMakeLists.txt`; `.github/workflows/cla.yml`.
- The specifications cited in place: ARCHITECTURE §8–9, MODEL §5, OVERVIEW §1–7, SPECULATION §10, SESSIONS §11, INTEGRATION §12, VALIDATION §13 and §18, MILESTONES §14–15, BACKLOG §19, BATCHING, LOWBIT, MULTI_DRAFTER, API, CLI, drafts/two-node-tp.
- Apple: the Metal feature-set tables (threadgroup memory, threads per threadgroup, simdgroup width); the Metal Shading Language specification (simdgroup matrices, math modes); Metal 4's tensor operations for the M5's neural accelerators; the M5, M5 Pro and M5 Max product pages (bandwidth, GPU cores, the AI-compute claim). Every Apple figure above is **TO VERIFY** on our own unit at M0 or M9.
