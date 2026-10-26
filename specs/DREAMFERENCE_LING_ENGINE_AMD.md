# ling-engine on AMD Strix Halo: a port to the AMD box, researched

Part of the ling-engine specification; the index is [README.md](./README.md). Written on 2026-10-26 in answer to the owner's question of the same day, "can we port ling-engine to Windows so we can run it on the AMD box?". It has no section number in the former `SPEC.md`. **Status: research.** Nothing is ported. The measurements in §1 and §6 were taken on the box that evening over SSH with small probes, kept in `bench/amd/`; nothing was installed on it. §6.5 is the first day of the first milestone, which the owner approved once the study was written.

Every figure is tagged: *measured* (on `amd-box`, 2026-10-26, the settings named), *third-party* (source named in §7), or *estimate* (the arithmetic is shown). Mightling's side of the AMD box (the client test box, the node pieces, the Windows memory model) is in Mightling's `specs/DREAMFERENCE_MIGHTLING_AMD_NODES.md` (cited as AMD_NODES), which this document builds on and does not repeat.

## Short answer

1. **Yes, it can be ported, but Windows is the expensive half, and the port is a rewrite of the kernels, not a recompile.** About 4,600 of ling-engine's 7,050 lines of device code use Blackwell or NVIDIA-only instructions (`mma.sync` fragments, `cp.async`, TMA, `ldmatrix`, block-scaled FP4 MMA) and are rewritten for RDNA 3.5's WMMA. The host side ports mechanically, except the HTTP layer (proxygen/folly) and the loader on Windows. *Estimate:* **10–14 engineer-weeks on Linux, 15–21 for Windows**, which includes the Linux work.
2. **The box's memory bus is not the problem.** *Measured:* a HIP read stream reaches **240 GB/s**, 94% of the 256 GB/s theoretical. That is as fast as the GB10's in-order read on `cudaMalloc` (234 GB/s, BACKLOG 19 item 28). So decode, which is bandwidth-bound, can come close to the GB10's.
3. **Two things are worse than on the GB10: the verify's compute, and prefill.** *Measured:* WMMA sustains **40–43 TFLOPS in FP16, BF16 and INT8, and 70–83 TOPS in INT4**. There is no FP4 or FP8 matrix unit. That is 167 FLOP per byte of bus, against about 900 on the GB10. A 16-row verify still fits under the weight stream (§1.3), but prefill is compute-bound. *Estimate:* its ceiling is about **650–820 tokens/s**, against ling-engine's 2,100–2,600 on the GB10.
4. **What the engine's design is worth here** (*estimate*, §1.2): a ling-engine-grade port with 16-row tree verification reaches about **50–60 tokens/s** on agent output. One agent turn at 32K would take about **8–9 s**, against about 4 s on the GB10, ~15 s for gufo (*third-party*) and ~18 s for llama.cpp (*measured*, §6.3). The engine's *speculation* is not the missing piece: DFlash2 already runs in upstream llama.cpp on this box. What is missing is a **verify that costs no more than a plain decode step**. *Measured* on four agentlong turns: llama.cpp's DFlash2 chain accepts **5.33 tokens per step**, close to ling-engine's 6.02–6.74 on the GB10. But its 8-row step costs **2.6× its plain decode step**, so it decodes at 22 tokens/s instead of about 50.
5. **Recommendation (§5).**
   - **First milestone:** the verify kernel. Prove on the box, in about a week and with no installs, that a 16-row NVFP4/FP8 WMMA streaming GEMM over the 27B's real matrix shapes runs at ≥ 200 GB/s and costs ≤ 1.15× a one-row pass.
   - **Day 1 of that milestone (§6.5):** FP8 passes, at 232 GB/s with 16 rows costing what one costs. NVFP4 reaches 172 GB/s, so its decode is the remaining work.
   - **If it passes:** port on **Linux first** (the dual boot), as a registry engine of the Linux node, and do the Windows host layer after that, only if the owner wants ling-engine on a Windows node.
   - **Until then:** the Windows node serves the 27B with llama.cpp + DFlash2, as AMD_NODES §6.4 plans.

---

## 1. The hardware ceiling

### 1.1 Measured on the box

**Setup.** Windows 11 25H2, AMD Software 26.10.1, VGM 96 GB, power mode Best performance; the firmware's power mode cannot be read from Windows (AMD_NODES §13.2). Device code was compiled on the box with TheRock 10.0's `clang++ -x hip --offload-arch=gfx1151 --cuda-device-only -nogpuinc -nogpulib -O3`, header-free. The host side is C# that PowerShell compiles at run time, calling `amdhip64_7.dll` through P/Invoke (`hipModuleLoad`, `hipModuleLaunchKernel`, events). The box has **no Visual Studio and no Windows SDK**, so no native HIP host program can be built on it as it stands (§3.1). The probes are in §6.

**Memory bandwidth.** The buffers were filled with hashed data (not zeros), 4 GiB and 16 GiB, 320–5,120 blocks of 256 threads, 16-byte non-temporal loads, 5–20 timed iterations.

| Kernel | GB/s |
|---|---|
| read stream, one 16-byte load per thread per iteration | **239–241** |
| read stream, four independent loads per thread | 210–224 |
| write stream | 204–216 |
| copy (read + write counted) | 206–209 |
| `hipMemcpy` device to device (read + write counted) | 209 |

The 16 GiB buffers gave the same figures as the 4 GiB ones, so the 32 MB MALL (Infinity Cache) does not inflate them. Theoretical: LPDDR5X-8000 × 256 bits = 256 GB/s, so the read stream reaches **94%**.

Against the GB10:

- The GB10 measures 262 GB/s in M0 ([MILESTONES](./DREAMFERENCE_LING_ENGINE_MILESTONES.md)), 234.2 GB/s for a pure in-order read on `cudaMalloc`, and 230–245 GB/s for `stream_gemm` per kernel (BACKLOG 19 item 28).
- The box reads at **92% of the GB10's M0 figure and level with its in-order read**. AMD_NODES §1.2's "about 80% of the GB10", from third-party 215–218 GB/s figures, is too pessimistic for a well-shaped read kernel.
- halogen's README also states a 240 GB/s ceiling (*third-party*).

`hipInfo` reports `memoryBusWidth 512`; the bus is 256 bits. It also reports `totalGlobalMem 107.87 GB` and `hipMemGetInfo` 107.9 GiB with a 96 GB carve-out: the count includes memory beyond the carve-out, the double count AMD_NODES §6.2 warns about, now seen here.

**Matrix throughput (WMMA, wave32).** 640 blocks of 8 waves, each wave with 8 independent 16×16×16 accumulators, 2,000 to 100,000 iterations (17 ms to 0.8 s):

| WMMA type | Measured, dense | Peak at 40 CU × 2.9 GHz (512 or 1,024 per CU per clock, GPUOpen) |
|---|---|---|
| FP16 → FP32 | **38.5–41.9 TFLOPS** | 59.4 |
| BF16 → FP32 | 38.5–41.0 TFLOPS | 59.4 |
| INT8 → INT32 | 39.9–43.1 TOPS | 59.4 |
| INT4 → INT32 | 70.0–83.1 TOPS | 118.8 |
| FP8, FP4 | **none** on gfx11 (FP8 WMMA is gfx12 and gfx1170 only, LLVM `BuiltinsAMDGPU.td`) | — |

- The measured figures are about 70% of peak, the same fraction as a third-party hipBLASLt BF16 GEMM (36.9 TFLOPS, llm-tracker). The likely cause is the clock under sustained load; the clock cannot be read from Windows over SSH.
- INT8 runs at the FP16 rate, so W8A8 buys bandwidth, not compute.
- The NVIDIA side, for comparison: the GB10 has about 250 TFLOPS of dense FP8 and 500 TFLOPS of dense FP4 (OVERVIEW §4).

**Other facts from `hipInfo`** (*measured*):

| Fact | Consequence |
|---|---|
| `warpSize 32` | ling-engine's lane arithmetic carries over unchanged |
| `sharedMemPerBlock 64 KB` | Shared memory per block |
| `l2CacheSize 2 MB` | Against about 24 MB on the GB10; ARCHITECTURE §8's "the 3.1 MB-per-layer DeltaNet state stays on chip" does not hold here, beyond what the 32 MB MALL catches |
| `multiProcessorCount 20` | These are WGPs, which is 40 CUs |
| `cooperativeLaunch 0` | Co-residency of persistent blocks is not guaranteed by the runtime on Windows |

### 1.2 Decode: what the bus allows

**Bytes per step.** ling-engine's step on agentlong (BACKLOG 19 item 27b, one sequence, 25.7K context):

- It reads 19.98 GB besides the KV. The coded scales take off about 0.45 GB.
- The BF16 KV adds 1.68 GB.
- That is **about 21.2 GB per step**. Of it, 17.6 GB are the target's weights (NVFP4 FFN and head, FP8 projections, [MODEL §5.2](./DREAMFERENCE_LING_ENGINE_MODEL.md)).

A port that keeps the served checkpoint (§3.1 argues it should) reads the same bytes.

**Plain decode.**

- *Estimate:* 17.6 GB ÷ 240 GB/s = 73 ms, a ceiling of **13.6 tokens/s**.
- *Measured* (§6): llama.cpp decodes UD-Q4_K_XL (about 16.7 GB per pass) at 12.1 tokens/s, so llama.cpp is already at ~84% of the read stream at one row.
- There is nothing left to win at one row, as on the GB10: speculation is the whole story.

**Speculative decode** (*estimate*). tokens/s = accepted tokens per step ÷ step time, at 21.2 GB per step.

| Accepted per step (τ) | Where τ comes from | Ideal (240 GB/s, nothing else) | At the GB10's own step efficiency |
|---|---|---|---|
| 6.74 | ling-engine on GB10, agentlong greedy, tree 7 + chain 8 = 16 rows, ScaleSearch drafter (BACKLOG 19 item 26) | 76 | **61–67** |
| 6.02 | the same, chain only (main before the tree merge) | 68 | 55–60 |
| 5.96 | the served RTN drafter with the tree, seed 1013 | 67 | 54–59 |
| 5.33 | llama.cpp's DFlash2 chain of 7 on four agentlong turns, on this box (*measured*, §6.3) | 60 | 48–53 (llama.cpp measures 22) |
| ~2.7 | gufo's DFlash2 chain on mixed text, Q4, depth 0 (*third-party*) | 31 | 25–27 (gufo measures 26.9) |

The right-hand column applies the GB10's ratio of step-average to read-stream bandwidth:

- On the GB10, 21.2 GB in 102.6 ms is 207 GB/s, which is 79–88% of its read figures.
- At 79–88% of 240 GB/s, a step on the box takes 100–112 ms.

**Risk.** The GB10 hides its verify compute for free and the box does not (§1.3). If none of the verify's compute overlaps the stream, the step grows by about 20–40 ms. At τ = 6.74 that is **48 tokens/s**, the low end. The stated range is therefore **about 50–60 tokens/s** on agentlong greedy, for a port as well made as the GB10 engine.

The acceptance itself should carry over:

- It is a property of the drafter and the target's greedy path, not of the GPU.
- The target's outputs differ slightly between quantizations, and the box would run the same NVFP4 checkpoint under option (a).

*Measured* (§6.3): on four agentlong turns on this box, llama.cpp's DFlash2 chain of 7, with the Q4 drafter on the Q4_K target, accepts τ = 5.33.

### 1.3 The verify's compute: why 16 rows fit and 32 do not, comfortably

| | Box | GB10 |
|---|---|---|
| Machine balance | 40 TFLOPS ÷ 240 GB/s = **167 FLOP per byte** | ~900 FLOP per byte |
| A 4.5-bit weight stream at M rows | 2M FLOP per 0.5625 bytes = 3.6M FLOP per byte | same |

- **Where the box turns compute-bound:** at M ≈ 47 rows at the measured peak. At a realistic 50–60% of WMMA peak inside a kernel that also dequantizes, it is **M ≈ 24–32**.
- **The work at 16 rows:** 2 × 25.6 G weights × 16 rows = 0.82 TFLOP of target work per step. That is 20 ms at the measured peak and about 40 ms at 50%, against a 100 ms stream.
- **One WMMA tile holds 16 rows.** A WMMA instruction is 16×16×16, so for M ≤ 16 every weight tile is used by exactly one instruction. The instruction count of a pass is the same at 1 row and at 16. Verify rows up to 16 cost the dequantization and one WMMA per tile, whatever M is.
- **The fit:** ling-engine's current verify (`--tree 7`, `LING_TREE_CHAIN` 8, `LING_TREE_T` 1.5: 16 rows) fits. Batches of 2–4 sequences at 16 rows each, 32–64 rows, would not fit under the stream as they do on the GB10.
- **Contention:** the dequantization (E2M1 through a 16-entry table, the E4M3 scale through a shift or table) runs on the same SIMDs as the WMMA. That contention is what §5's first milestone measures.

### 1.4 Prefill: what the compute allows

**The work per token.**

- A prompt token costs about 2 × 24.4 G = **49 GFLOP** in the target's matrices; the head runs only for the last token.
- At 32K of depth, the 16 attention layers add about 13 GFLOP per token: 4 × 24 heads × 256 dims × 33K positions × 16 layers.
- The DeltaNet chunked recurrence is small next to both.

**Ceiling** (*estimate*):

| Depth | At the measured 40 TFLOPS | At a realistic 65–80% of it |
|---|---|---|
| 0 | 820 tokens/s | 530–650 |
| 32K | 650 tokens/s | 420–520 |

**Third-party, for comparison:**

- gufo, whose INT8 WMMA prefill is W8A8 with Q8_1 activations: 649 at 0, 413–469 at 32K. That is at this ceiling.
- halogen: 566 at 32K.
- llama.cpp on this box (*measured*, AMD_NODES §13.4): 301 / 226.

**What does not change it:**

- No weight format moves this ceiling, because prefill does not read weights per token.
- The only lever is the matrix rate. INT4 WMMA (70–83 TOPS) would need INT4 activations, a W4A4 integer scheme with per-16 scales applied per instruction in FP32. That is a quality question, not a port question. halogen reports +9% prefill for −0.45 points of top-1 agreement with W4A4 at 64+ rows (*third-party*).
- On the GB10, ling-engine prefills on NVFP4 W4A4 tensor cores at 2,100–2,600 tokens/s ([README](../README.md)). On this box it would be **4–5× slower**, whatever the engine.

### 1.5 What it means for one agent turn

The method of AMD_NODES §4: a 2,000-token tail at 32K of cached context, then a 200-token answer.

| Engine | Prefill at 32K | Decode on agent output | One turn |
|---|---|---|---|
| ling-engine, GB10 | ~2,100+ (README) | ~65 (BACKLOG 19) | **~4 s** |
| ling-engine port, box (*estimate*) | 420–520 | 50–60 | **~8–9 s** |
| gufo, box (*third-party*, Linux) | 413 | 19.8 mixed | ~15 s |
| llama.cpp + DFlash2, box (*measured*, Windows, ROCm) | 226 | 22 (four agentlong turns, §6.3) | **~18 s** |

On this box the port halves the turn against the best open engine. It still leaves the box at about half the GB10's speed, because prefill sets the floor.

---

## 2. What in ling-engine is CUDA- or Blackwell-specific

Read from `main` at `c5ec4ec`. Line counts are from `wc -l`.

| Part | Lines | Platform-specific content | In a port |
|---|---|---|---|
| Tokenizer, parsers, accept, tree, lookup, sampling, prefix store (`src/tokenizer`, `src/core/{accept,lookup,tree,sampling,prefix_kv}`) | ~2,600 | none (host C++) | **reused** as is |
| `src/serve/` (HTTP, API) | 2,560 | proxygen + folly; `SIGINT`/`SIGTERM` | reused on Linux; **replaced on Windows** (§3.1) |
| `safetensors.cpp` loader | small | `open`/`mmap` | `MapViewOfFile` on Windows (a day) |
| `engine.cpp`, `speculate.cpp`, `drafter.cpp` | 3,140 | CUDA runtime, CUDA graphs (`cudaGraphInstantiateFlagUseNodePriority`), green contexts (`cuGreenCtxCreate`), NVTX, programmatic dependent launch | hipify; green contexts dropped; graphs → `hipGraph`; PDL has no HIP equivalent (more gap between kernels) |
| `stream_gemm.cu` (decode, verify, drafter and head GEMMs; the megakernel stages) | 2,208 | `mma.sync` m16n8k16 FP16 fragments with lane maps (lines 15–21), `cvt.rn.f16x2.e2m1x2`/`.e4m3x2` PTX, `cp.async`, `ld.global.nc.L1::no_allocate`, L2 prefetch, persistent blocks with `ld.acquire.gpu`/`red.release.gpu`/`%globaltimer` barriers | **rewritten**: WMMA 16×16×16 fragments, software E2M1/E4M3 decode, buffer loads; barriers on agent-scope atomics and `s_memrealtime` |
| `attention_body.cuh` (split-K decode and verify attention, BF16 KV) | 582 | `mma.sync`, `ldmatrix`, `cp.async`, `cp.async.bulk`, L2 prefetch | **rewritten** on WMMA |
| `prefill_gemm.cu` | 718 | `kind::mxf4nvf4.block_scale` (NVFP4 W4A4) and `kind::f8f6f4` (FP8) MMA | **rewritten** as W4A16/W8A16 FP16 WMMA, or W8A8 INT8 WMMA (gufo's route) |
| `prefill_gdn.cu` (chunked DeltaNet) | 440 | `mma.sync` | rewritten on WMMA |
| `gemm_bf16.cu` (drafter fc, large M) | 229 | TMA (`cp.async.bulk.tensor`, `CUtensorMap`, `mbarrier`) | rewritten (plain loads + WMMA) |
| `sparse_head.cu`, `topk.cu` | 426 | cub, `mma.sync` (sparse head) | hipcub; the MMA part rewritten |
| `kernels.cu` (norms, conv, DeltaNet recurrence, embedding), `draft_kernels.cu`, `model.cu` | 1,732 | `__shfl_*_sync` with full masks, `__nv_fp8` | **hipify** (wave32 matches) |
| Tests (`tests/*.cu`) | 2,786 | as the kernels | follow the kernels |

**Totals.**

- Device code is 7,051 lines. About **4,600 are rewritten** and about 2,400 hipify.
- The CMake target is `CMAKE_CUDA_ARCHITECTURES 121a`.
- No CUTLASS, CuTe, FlashInfer or cuBLAS is linked, so no NVIDIA library needs an AMD counterpart; cub becomes hipcub, which is header-only.
- There is no HIP code, no `#ifdef` for another architecture and no CPU forward pass.

**Two facts make the port smaller than "Blackwell-specific" suggests.**

- **The decode path does not use FP4 tensor cores.** `stream_gemm` decodes NVFP4 and FP8 weights to FP16 in registers and runs FP16 `mma.sync`; only prefill uses block-scaled FP4. So the port can run the **served NVFP4 checkpoint itself**, decoded in software. The loader, the precision map, the drafter's NVFP4 rounding and the same model id all stay; nothing is requantized to GGUF. The FP16 decode maps directly onto FP16 WMMA.
- **The megakernel stages are optional** (`LING_MK_FFN`, `LING_MK_GDN`; together +0.2 ms on the GB10, BACKLOG 19 item 26). A first port can leave them off.

---

## 3. Options

### 3.1 (a) A HIP port of ling-engine

**What is reused.**

- The host engine (scheduler, tree, accept, lookup, sampling, sessions, prefix store, tokenizer, parsers, API semantics), the loader, the tests' references, `ling-run`'s self-checks (`--spec-check`, `--graph-check`, `--prefix-check`) and the agentlong bench.
- The checkpoint: NVFP4 FFN and head and FP8 projections. Both are stored as on the GB10 and decoded to FP16 in registers.

**What is rewritten.** The six MMA-, TMA- and `cp.async`-bound files of §2: the streaming GEMM first, then attention, prefill GEMM, prefill DeltaNet, the BF16 GEMM and the sparse head.

**Weight format.** AMD has no FP4 or FP8 matrix unit, but decode never needed one (§2). Prefill becomes W4A16/W8A16 on FP16 WMMA, or W8A8 on INT8 WMMA (FP16 activations quantized per row). Two formats were weighed:

- **GGUF Q4_K.** It would read about 1 GB less per pass (§6: UD-Q4_K_XL's ~16.7 GB against 17.6 GB). But it changes the model Mightling validated and is not the checkpoint the GB10 serves.
- **A uniform NVFP4 re-quantization of the FP8 projections** (MODEL §5.2: 17.6 → 14.4 GB, −18% per pass). It goes through the same quality gate on both machines, so it is a GB10 decision first.

**Exactness.** Yes, by the engine's own definition: speculative greedy is identical to the same build's non-speculative greedy over the same weights. The port keeps the policy that every row runs through the same kernel. With WMMA at M ≤ 16 a single row and 16 rows execute the same instruction sequence, so batch invariance comes for free. Identity with the GB10's output is not possible, because the arithmetic differs, and it is not the goal.

**Expected speed** (*estimate*, §1.2, §1.4):

| | Linux | Windows |
|---|---|---|
| Decode on agentlong greedy | 50–60 tokens/s | 5–10% less |
| Prefill at 32K | 420–520 tokens/s | 420–520 tokens/s |

The Windows penalty is third-party: gufo's Windows fork measured decode 7.6–9.5% slower without its Windows-only switches, `hipStreamSynchronize` returning about 0.4 ms late, and copy-engine hand-offs of about 150 µs.

**Effort** (*estimate*, one engineer who knows the engine):

| Work | Weeks |
|---|---|
| Bring-up on Linux/ROCm: hipify, portable kernels, the build, kernel tests green | 2 |
| Streaming GEMM on WMMA, NVFP4 and FP8, 1–16 rows, ≥ 85% of the read stream | 3–4 |
| Decode and verify attention, split-K, BF16 KV | 1.5–2 |
| Prefill GEMM and prefill DeltaNet on WMMA | 2–3 |
| `hipGraph` capture of the step; PDL dropped; the megakernel ported or left off | 1–2 |
| Exactness and agentlong validation | 1–2 |
| **Linux total** | **10–14** |
| Windows host: a build with `clang-cl` + MSVC + Windows SDK (gufo's route, `BUILD-WINDOWS.md`) or cross-compiled with the Windows SDK and CRT fetched; vcpkg for ICU and PCRE2; `MapViewOfFile` | 1–1.5 |
| Windows HTTP layer: proxygen/folly replaced by a small server for the routes of [API](./DREAMFERENCE_LING_ENGINE_API.md), or proxygen on Windows (unproven) | 2–3 |
| WDDM specifics: the DXGI budget, TDR, the carve-out, process kill, the sync latency | 1.5–2 |
| Packaging, signing, Smart App Control (AMD_NODES §13.2) | 0.5 |
| **Windows total, Linux included** | **15–21** |

**Risks.**

- The streaming kernel might not reach the bus with WMMA and the decode sharing the SIMDs. §5's first milestone exists for this risk.
- Without PDL, the ~330 launches per step pay a gap each; unmeasured on `hipGraph`.
- The megakernel's spin barriers have no forward-progress guarantee without cooperative launch, which Windows reports as unsupported.
- TheRock on Windows is a moving target: gufo reports that 10.1.0 through the 10.2.0a nightlies compile gfx1151 kernels that stall ("up to hundreds of times slower"), so the toolchain must be pinned to 10.0.0.
- Profilers and debuggers for HIP on Windows are thin.
- It is a second platform to keep exact and fast beside the GB10.

### 3.2 (b) ling-engine's speculation in llama.cpp or in gufo, upstream-style

**What already exists.**

- **llama.cpp** has had DFlash2 since #27342 (merged 2026-08-27, *third-party*). It runs on this box from the Windows release zip with z-lab's Q4_K_M drafter (§6).
- llama.cpp verifies a **chain** of at most 7 drafts: the drafter's block of 8 includes the anchor, and `--spec-draft-n-max` 8, 12 and 16 behaved exactly like 7 (*measured*).
- It has per-token rollback of recurrent state (#25278) and n-gram drafters that combine with DFlash2 in one `--spec-type` list.
- Tree verification is only a draft PR (#30931, opened 2026-10-21).
- **gufo** has the same drafter as a chain of 1–7 with an adaptive width controller, a prompt-lookup draft since 2026-10-21, exact greedy ("matches isolated AR on all 160 measured requests"), and a native Windows build merged upstream (zips since 0.24.0).

**So "bringing the drafter in" is done.** What ling-engine would bring:

1. **Tree verification**: DFlash2's top-K candidates as branches, tree 7 + chain 8. It gave ×1.09 on the GB10 (BACKLOG 19 item 26).
2. **The lookup branch in the same verify**, not as an alternative draft.
3. **Above all, a verify that costs no more than a decode step.** *Measured* (§6.2): llama.cpp's step costs about 12 ms more per extra verify row, so a 7-draft chain costs **2.1× a plain step** (175–178 ms against 83 ms), and its best code result was at 4 drafts, not 7. gufo's cost tables put a 7-draft verify at 1.29× (*third-party*): its decode and verify are FP32 matrix-vector kernels, kept identical to scalar decode for exactness.

**Expected speed.**

| Engine | Expected (*estimate*) |
|---|---|
| llama.cpp with a WMMA verify path and a tree | from today's 26 tokens/s on code, 43 on an edit (§6.2), to perhaps 35–45 on agent output |
| gufo with a tree and a 16-row WMMA verify | near (a)'s 50–60 on Linux, a few percent less on Windows |

**Exactness.**

- llama.cpp: **no**, *measured*. Greedy output changes with the speculation setting on two of the three prompts (§6.2), because the batched verify and the one-row decode run different kernels. Making it exact means batch-invariant kernels, which upstream would take only behind a flag.
- gufo: **yes**, by its own policy.

**Effort** (*estimate*):

| Work | Weeks |
|---|---|
| A tree verify and the lookup branch in gufo | 3–5 |
| A 16-row WMMA verify GEMM for gufo's GGUF block formats (Q4_K, Q5_K, Q6_K, IQ4_XS decoded in kernel) | 3–4 |
| Upstream review (gufo merges outside PRs; no CLA; Conventional Commits titles) | unbounded |
| **Total for gufo** | **6–9** |
| The same in llama.cpp, against a multi-backend code base with a slower review | **8–12** |

**Risks.**

- Other people's roadmaps: gufo is two months old and releases daily.
- The result is not ling-engine. Tree, lookup and kernel work land in an engine Mightling does not control. gufo is MIT, so a fork is always possible.
- The Windows build is gufo's, and gufo itself calls it "a supported native build target for development"; its production packages are Linux-only.

### 3.3 (c) A Vulkan compute backend

**What it is.**

- Every kernel is rewritten in GLSL, HLSL or Slang compute, on `VK_KHR_cooperative_matrix` for the matrix work.
- The host's CUDA runtime calls become Vulkan: pre-recorded command buffers stand in for graphs, plus a memory allocator.
- The engine logic above is reused.

**What it buys.**

- The only stack that **ships with the graphics driver** on Windows (Adrenalin) and on Linux (RADV): no TheRock and no ROCm.
- One binary for AMD, Intel and NVIDIA GPUs.

**What it costs.**

- Vulkan prefill is behind ROCm on this box: *measured*, llama.cpp's Vulkan prefill is 15–25% below its ROCm prefill at 32K (169 against 226 tokens/s), with equal decode.
- Fewer tools; the matrix unit is reached through an extension whose code generation on AMD's Windows driver this study did not measure.

| Expected | Effort (*estimate*) | Exact | Risk |
|---|---|---|---|
| decode as (a); prefill 10–25% below (a) | **20–30 engineer-weeks** | yes, by construction | highest |

Ranked last, unless shipping without ROCm becomes the deciding requirement. The HIP runtime already comes with the driver (§6.4), which removes most of that argument.

### 3.4 (d) The box's Linux side only

**What it is.** (a) on Ubuntu 26.04 with ROCm, plus AMD_NODES §7's minimum Linux node (platform detection, a registry entry, the `/dev/kfd` launch prefix, the GTT pre-flight, the advertisement).

**What it saves.**

- All of (a)'s Windows rows.
- proxygen/folly build on x86-64 Linux as on the GB10.
- The node's host safety (PSI, earlyoom) and the egress audit carry over.
- Profiling uses rocprofv3.

**What it costs.**

- The dual boot is prepared but not installed. The box is then either the Windows test client or a node, never both at once (AMD_NODES §8.3).
- The carve-out conflicts: Windows wants VGM at 96 GB, Linux wants it at its minimum and GTT sized to RAM. Each switch may need a firmware change, which over SSH means the ADLX call and a reboot (AMD_NODES §13.2).

| Expected | Effort (*estimate*) | Exact | Risk |
|---|---|---|---|
| (a)'s Linux numbers: 50–60 tokens/s, 420–520 prefill at 32K | **10–14 weeks** for the engine, plus **2–3** for the node | yes | medium (the kernel risk of (a), without the Windows risks) |

### 3.5 Summary

| | (a) HIP port | (b) into gufo / llama.cpp | (c) Vulkan | (d) Linux only |
|---|---|---|---|---|
| Decode, agent output (*estimate*) | 50–60 Linux; 5–10% less on Windows | gufo: toward 50–60; llama.cpp: 35–45 | as (a) | 50–60 |
| Prefill at 32K | 420–520 | 413–469 (gufo), 226 (llama.cpp, *measured*) | 10–25% below (a) | 420–520 |
| Effort, engineer-weeks | 10–14 Linux; 15–21 with Windows | 6–9 (gufo); 8–12 (llama.cpp) | 20–30 | 12–17 with the node |
| Exact (greedy = own non-speculative) | yes | gufo yes; llama.cpp **no** (*measured*) | yes | yes |
| Same checkpoint as the GB10 | **yes** (NVFP4 decoded in software) | no (GGUF) | yes | **yes** |
| Main risk | the WMMA stream kernel; Windows host and toolchain | others' roadmaps; not our engine | effort | the dual boot and the carve-out switch |

**Best on Windows:**

- **Now:** (b) on an engine that already ships there. That means llama.cpp (release zips, *measured* tonight) or gufo (native Windows build, upstream). This is what AMD_NODES §6.4's minimum Windows node uses.
- **For ling-engine itself:** (a), after (d).

**Best on Linux:** (d), which is (a) built where ROCm is first-class.

---

## 4. How Mightling would use it

### 4.1 The node role

**On Linux (option d).**

- AMD_NODES §7 unchanged.
- The registry entry names `engine: "ling"` with a platform `gfx1151`, the same checkpoint (`RadixArk/Qwen3.8-27B-NVFP4`) and the same DFlash2 drafter. The launch builder gets the AMD device prefix (`--device /dev/kfd --device /dev/dri …`).
- The image is a ROCm base with `ling-serve`, pinned by digest.
- Mightling's tools ask the server which model runs (`/v1/models`), so a client cannot tell the AMD node from a GB10 except by speed.

**On Windows.** There is no Docker. The launcher's engine supervisor (Mightling's WINDOWS_ARM §8.3, designed, not built) runs `ling-serve.exe` as a child in a Job Object.

- **The registry recipe** reaches Rust as a JSON export in the release (AMD_NODES §6.3).
- **The binary** is a release asset, `ling-engine-<version>-windows-gfx1151.zip`: `ling-serve.exe`, its code objects and the vcpkg DLLs, pinned by SHA-256 like llama.cpp's zips in AMD_NODES §13.4.
- **The HIP runtime** needs no redistribution. ling-engine links no BLAS, so it needs only `amdhip64_7.dll`. The driver installs one in `System32` (*measured*: 10.0.3683.0 with AMD Software 26.10.1, newer than TheRock 10.0.0's 10.0.3581.0). A gfx1151 code object compiled with TheRock loads and runs through it (§6.4), so the zip ships no 1.68 GB TheRock; only the build machine needs TheRock 10.0.0.
- **Signing:** an unsigned build meets Smart App Control (off on this box, D2); signing is WINDOWS_ARM §16.3's.

### 4.2 Host safety

**Linux:** as the GB10. That covers `check_host_safety()`, the PSI watchdog and earlyoom, plus AMD_NODES §7's GTT pre-flight, because a killed GPU process can leave tens of GiB of GTT held until a reboot (*third-party*, halogen).

**Windows** (WINDOWS_ARM §9's design):

- **Size the pool from the budget.** The engine sizes its memory pool from the DXGI budget (`QueryVideoMemoryInfo`), never from `hipMemGetInfo`, which over-reports (*measured*, §1.1).
- **Never allocate beyond the carve-out.** There, backing runs at 0.63 GiB/s and decode at −6 to −22% (*third-party*, the gufo fork). An overflow can freeze the whole box under WDDM (AMD_NODES §6.2).
- **Kill by Job Object.** The watchdog is the launcher's, killing through `TerminateJobObject`.
- **TDR.** WDDM resets the GPU when a kernel runs longer than about 2 s, so no kernel may. Prefill chunks are sized for it, and the megakernel's persistent blocks already give up after 50 ms (`stream_gemm.cu`, `__trap` after the deadline).
- **No PSI.** Windows has nothing like it. The memory-resource notification and the DXGI budget are the signals.
- **To measure:** whether a kill with work in flight leaves memory held on Windows, as GTT does on Linux.

### 4.3 The 96 GB carve-out

- **The engine's budget.** INTEGRATION §12 budgets about 57 GB: weights and drafter ~19, embedding ~3.5, sessions 30, prefix store 2–4, scratch 2. That fits inside the 96 GB carve-out with room for a larger session pool, since the box runs nothing else that needs GPU memory.
- **What Windows keeps:** 31.6 GB (AMD_NODES §13.2). That is enough for the launcher, the client and the desktop app.
- **The 27B alone needs far less.** llama.cpp peaked at 19–20 GB (AMD_NODES §13.4), so 64 GB would do. The 96 GB setting is Flash-Next's.
- **On Linux the setting is wrong:** VGM goes to its minimum and the pool comes from GTT (AMD_NODES §7).

---

## 5. Recommendation and staged plan

**Not "port to Windows" first.** The most uncertain part of every ling-engine-grade option (a, b-with-gufo, d) is the same: **can a 16-row verify on gfx1151 cost about what a one-row decode costs?** llama.cpp's verify does not (2.1× on short prompts, 2.6× at 28–32K, *measured*); gufo's costs 1.29× (*third-party*). The engine's 50–60 tokens/s estimate stands or falls on it. It can be measured in a week on the box as it is, with the harness of §6 and no installs.

| Stage | What | Where | Effort | Gate to the next |
|---|---|---|---|---|
| **0 (done, 2026-10-26)** | The bus, WMMA, and llama.cpp's DFlash2 on the box: acceptance, step cost and exactness on three prompts and four agentlong turns (§6.2, §6.3); the driver's HIP runtime (§6.4) | Windows, release zips | — | — |
| **1, the first milestone** | **The verify kernel probe.** A header-free HIP kernel that streams one of the 27B's real matrices in its served format, the FFN gate\|up (5120 × 34,816, NVFP4) and the DeltaNet in-projection (5120 × 16,384, FP8), decoded to FP16 and multiplied on WMMA for M = 1, 4, 8, 16, 32 rows. Measured: GB/s and µs per M. Also: a `hipGraph` of ~330 small kernels (the per-node gap that PDL hides on the GB10) | Windows as it is (§6's harness); **started 2026-10-26, day 1 in §6.5**: FP8 passes, NVFP4 at 172 GB/s | **~1 week** | M = 16 at **≥ 200 GB/s** and **≤ 1.15×** the M = 1 time; graph gap ≤ 10 µs per node |
| 2 | Decode-only port on Linux (dual boot): streaming GEMM, attention, DeltaNet, drafter, tree, graphs; `ling-run --spec-check` and agentlong greedy 16/16 identical to its own non-speculative output | Linux | 7–9 weeks | ≥ 45 tokens/s on agentlong greedy, exact |
| 3 | Prefill on WMMA, `ling-serve` on the Linux node (AMD_NODES §7's minimum node, `engine: "ling"`) | Linux | 4–6 weeks (+2–3 for the node) | a Work session from a client against the box; prefill ≥ 400 tokens/s at 32K |
| 4 (owner's decision) | The Windows host layer of §3.1: build, HTTP, loader, WDDM, packaging; the launcher's supervisor | Windows | 5–7 weeks | the same agentlong numbers within 10% of Linux |

**If stage 1 fails its gate**, a ling-engine port is worth about what gufo already is, and the effort goes into (b): a tree and the lookup branch contributed to gufo, the open, exact, Windows-building engine for this chip.

**Meanwhile the Windows node** serves the 27B with llama.cpp + DFlash2 (zips, no build), as AMD_NODES §6.4 plans. §6.2 sets its flags: `--spec-draft-n-max 4` for code, and only the trained block width of 7 when the output is mostly copied.

---

## 6. What was run on the box (2026-10-26, 22:15–23:35 UTC)

Scratch files are in `C:\llm\amd-port` on the box; the probes are also in this repository under `bench/amd/`. Nothing was installed, no driver or setting was changed, and the downloads came to 1.14 GB (the drafter).

### 6.1 The probes

| File | What |
|---|---|
| `bw.hip`, `bw.ps1` | read, read ×4, write, copy and fill kernels; the host in C# through P/Invoke; §1.1's bandwidth table |
| `wmma.hip`, `wmma.ps1` | FP16, BF16, IU8 and IU4 WMMA loops; §1.1's matrix table |
| `spec.ps1`, `prompts.json` | `llama-server` (b11907, the zips of AMD_NODES §13.4) on three prompts: prose (350 words of explanation), code (a Python function and five tests), edit (return a 70-line file with a rename, type hints and a sorted summary). Temperature 0, thinking off, one warm-up request; the server's own timings and draft counters; SHA-256 of each answer |
| `rt.ps1` | the driver's HIP runtime with TheRock off `PATH` (§6.4) |
| `vg.hip`, `vg.ps1`, `vg-tune.ps1` | the first milestone's verify-GEMM probe (§6.5) |
| `ag.ps1`, `cmp.ps1` (with `ag-t0[0-3].json` and `qwen38-patched.jinja`, kept on the box only) | the first four agentlong turns (25–33K tokens of context, `bench/agentlong/manifest.json`), sent to `llama-server`'s `/v1/responses` with temperature 0 and Mightling's patched chat template |

The drafter is z-lab's `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (1,143,006,816 bytes, SHA-256 `1a25c568…31ebd`, equal to the LFS object). It runs with `--spec-type draft-dflash -ngld 99`. llama.cpp logs `block_size=8, n_extract=5`.

### 6.2 DFlash2 in llama.cpp on three prompts

*Measured.* Vulkan unless marked; 450–477 tokens per answer.

τ is the tokens per step, (generated) ÷ (generated − accepted drafts). The step time is τ ÷ tokens/s.

| Drafting | Prose: tok/s, τ, step | Code: tok/s, τ, step | Edit: tok/s, τ, step |
|---|---|---|---|
| none | 12.2, 1, 82 ms | 12.1, 1, 83 ms | 12.1, 1, 83 ms |
| DFlash2, 2 drafts | **21.7**, 2.30, 106 ms | 25.3, 2.71, 107 ms | 27.5, 2.98, 108 ms |
| DFlash2, 4 drafts | 13.4, 2.73, 204 ms (first run after the download, see below) | **29.4**, 3.91, 133 ms | 35.9, 4.97, 138 ms |
| DFlash2, 7 drafts (8, 12, 16 clamp to 7; two runs) | 17.0–17.1, 3.00, 175 ms | 26.0–26.2, 4.64, 177 ms | **42.9–43.1**, 7.69, 178 ms |
| DFlash2 7 + `ngram-mod` | 17.1 | 26.1 | 33.9 (more drafts rejected: 63% against 97%) |
| MTP, 4 drafts | 19.3, 2.78, 144 ms | 26.6, 3.88, 146 ms | 32.7, 4.87, 149 ms |
| ROCm, none | 11.8 | 11.8 | 11.8 |
| ROCm, DFlash2 4 | 18.3 | 24.7 | 31.7 |
| ROCm, DFlash2 7 (and 12, which clamps to 7) | 13.7–13.8 | 22.0–22.4 | 35.4–35.5 |

**Reading.**

- **Each extra verify row costs about 12 ms in llama.cpp on this chip,** drafter included. The step is 82–83 ms at one row, 106–108 ms at 3 rows, 133–149 ms at 5 and 175–178 ms at 8: the cost is close to linear in the rows. On the GB10, ling-engine's 16-row step is 102.6 ms against a weight pass of about 75 ms (BACKLOG 19).
- **Acceptance is good when the output copies.** τ = 7.7 on the edit, 97% of drafts accepted.
- **The cost of width eats most of the gain.** The best width is 2 drafts on prose, 4 on code and 7 only on the edit.
- **Two runs ran first after the drafter was downloaded:** the first DFlash2-7 run on Vulkan (11.4 / 14.6 / 43.5 tokens/s, identical token counts to the later runs) and DFlash2-4's prose answer. Both are slower than everything around them, so they are read as a cold-file effect: the first DFlash2-7 run is left out, and the prose cell above is marked.
- **ROCm decodes slower than Vulkan with speculation** on this build, by 5–20%.
- **Exactness: llama.cpp's speculative greedy is not identical to its own plain greedy.**
  - The edit's answer is byte-identical in every configuration on both backends (`924EBE51846D`).
  - The prose answer on Vulkan is `2B8AE572D399` without drafting and with MTP, but `A3D0…` with 2 DFlash2 drafts and `064A…` with 7.
  - The code answer is `4244DF038B6D` without drafting, but `D2A7…` with 2 or 4 DFlash2 drafts and with MTP, and `EFF4…` with 7.
  - On ROCm the plain answers (`D737…`, `DD46…`) differ again from both its speculative ones and Vulkan's.
  - The answers part where two logits are near-tied and the batched verify and the one-row decode round differently. ling-engine and gufo avoid this by running every row through one kernel.

### 6.3 Four agentlong turns

*Measured*, ROCm (TheRock 10.0), `-c 40960`, no prompt cache between configurations. The turns are t00–t03 of `bench/agentlong/manifest.json`: 28,307–32,458 tokens of context, the same request bodies `bench/agentlong/run.py` sends to `ling-serve`, at temperature 0, through `/v1/responses` with Mightling's patched chat template. Every answer is a message plus a tool call, 111–522 tokens long.

| Drafting | Decode, pooled over the 4 turns | τ (tokens per step) | Step | Prefill at 28–32K |
|---|---|---|---|---|
| none | **10.7 tokens/s** (1,083 tokens in 101.1 s) | 1 | 93 ms | 228–243 tokens/s |
| DFlash2, 7 drafts | **22.0 tokens/s** (49.3 s) | **5.33** (880 of 1,083 tokens from accepted drafts) | 243 ms | 217–256 |
| DFlash2 7 + `ngram-mod` | 20.7 tokens/s (52.4 s) | 5.28 | 255 ms | 220–238 |

**Reading.**

- **The drafter accepts well on agent output on this box.** Its chain of 7 gives τ = 5.33. ling-engine's chain gives 6.02 on the GB10 and its 16-row tree 6.74 (BACKLOG 19 item 26), on all sixteen turns rather than these four. So the acceptance of §1.2's table carries over to the box's quantization, give or take the workload.
- **llama.cpp's step is the loss.** At 28–32K of context its 8-row step costs **2.6× its one-row step**. That is more than on the short prompts (2.1×), because its verify attention also grows with the rows.
- **The arithmetic:** with τ = 5.33 and a step of 1.15× the plain one (§5's gate, about 107 ms here), the same drafter would give about **50 tokens/s**, against 22 today.
- **Exactness held on these turns.** The four answers' content (message text, tool names and arguments) is byte-identical across the three configurations (`cmp.ps1`), unlike two of the three short prompts of §6.2. So llama.cpp's speculative greedy is often equal to plain greedy, but not guaranteed.
- **One agent turn at 32K** by AMD_NODES §4's method: 2,000 ÷ 226 + 200 ÷ 22 ≈ **18 s** with llama.cpp + DFlash2 on Windows.

### 6.4 The driver's own HIP runtime

*Measured* (`rt.ps1`), with TheRock not on `PATH`:

- The driver's `C:\Windows\System32\amdhip64_7.dll` (10.0.3683.0, AMD Software 26.10.1; `hipRuntimeGetVersion` 71260602) and its `amd_comgr_3.dll` loaded the TheRock-compiled gfx1151 code object (`hipModuleLoad`), found its kernel, launched it and synchronized, all with `hipSuccess`.
- Windows searches `System32` before `PATH`, so the measurements of §1.1 ran on this same driver runtime.

A ling-engine build for Windows therefore needs TheRock only to compile. The release zip ships the program and its code objects, and the driver supplies the runtime (§4.1).

### 6.5 The first milestone, day 1: the verify kernel probe

The owner approved starting §5's first milestone once this study was written. This is its first day (2026-10-26, from 23:24 UTC), with `bench/amd/vg.hip` and `vg.ps1`. Everything is *measured*.

**The setup.**

- **Matrices:** the FFN gate|up (K 5,120 × N 34,816, NVFP4: E2M1 codes, one E4M3 scale per 16, 100.3 MB) and the DeltaNet in-projection (K 5,120 × N 16,384, FP8 E4M3, 83.9 MB).
- **Data:** hashed codes and scales, eight copies of each matrix so the 32 MB MALL cannot hold them.
- **The kernel:** decodes in registers to FP16 and multiplies on WMMA 16×16×16 against M activation rows staged in LDS. M = 1 and 16 use one 16-row tile; M = 32 uses two.
- **Correctness:** checked against an FP32 reference, with a relative error of 1.4–1.6 × 10⁻⁵ for every kernel and M.

**What made the difference**, in order:

1. **Retiling.** In row-major order a wave reads 16 rows 32 bytes at a time: 36 GB/s. With 16 × 64 tiles that one wave loads contiguously (512 bytes for NVFP4), each half-wave decodes only its own K and the halves swap decoded fragments (`v_permlanex16`): 93–108 GB/s.
2. **LDS padding.** Padding the LDS activation rows by 16 halves removes a 16-way bank conflict: 173 GB/s.
3. **K-major tile order.** Ordering the tiles K-major lets neighbouring waves read neighbouring memory at the same moment. The bare loads alone, with decode, WMMA and staging switched off, then reach 227 GB/s, against 198 in N-major order.
4. **Larger blocks for FP8.** 1,024-thread blocks share the staged activations.

ling-engine's loader already retiles every matrix for its kernels on the GB10 (ARCHITECTURE §8, stage 4), so none of this changes the checkpoint.

**Best results:**

| Matrix | M = 1 | M = 16 | M = 32 | Read stream over the same bytes | M = 16 ÷ M = 1 time |
|---|---|---|---|---|---|
| FP8 in-projection, 1,024-thread blocks | 234 GB/s | **232 GB/s** (362 µs) | 151 GB/s (earlier layout) | 242 GB/s | **1.01** |
| NVFP4 gate\|up, 256-thread blocks | 200 GB/s | **172 GB/s** (585 µs) | 112 GB/s (earlier layout) | 241 GB/s | 1.17 |

**Against the gate** (M = 16 at ≥ 200 GB/s and ≤ 1.15× the M = 1 time):

- **FP8 passes.** 96% of the read stream, and 16 rows cost the same as one.
- **NVFP4 does not yet.** It reaches 71% of the stream at 16 rows and 83% at one row.
- **What the NVFP4 gap is not.** The M = 1 and M = 16 runs execute the same instructions (WMMA always computes 16 rows), and stripping the activation staging or the barriers does not close the gap.
- **What it might be.** The cost is in the decode and the WMMA under load, and possibly in a clock that falls with non-zero operands; the firmware's power mode is unknown (AMD_NODES §13.2).
- **32 rows do not fit for NVFP4** (112 GB/s), as §1.3 predicted. FP8 at 32 rows loses 35%.

**What it means for §1.2's estimate.**

- The target's weights would stream in about **92 ms**: 10.35 GB of NVFP4 (FFN and head) at ~172 GB/s and 7.2 GB of FP8 at ~232 GB/s. At 240 GB/s they would take 73 ms.
- With the drafter, KV and state (~3.6 GB at ~200 GB/s), the step is about 110 ms.
- At τ = 6.74 that is about 61 tokens/s; at llama.cpp's measured τ = 5.33, about 48. That sits within §1.2's 50–60.

**Left for the rest of the milestone:**

- The NVFP4 decode: a byte-table decode with `v_perm_b32`, since E2M1's FP16 low bytes are all zero, and the scale folded into the epilogue.
- The person-at-the-box check of the firmware power mode.
- The `hipGraph` gap per node for a ~330-kernel step.

---

## 7. Sources

Read on the web on 2026-10-26 unless dated otherwise. "Third-party" means not measured by us.

- **gufo:** github.com/gufo-org/gufo at `bb5ab8dc` (v0.25.0, 2026-10-26, MIT):
  - `docs/models/qwen3.8-27b/BENCHMARKS.md` (2026-10-19; Q4 AR 12.37 / 11.13 tokens/s, DFlash2 mixed 26.91 / 19.77, prefill 648.89 / 469.28 / 413.41 at 0 / 32K);
  - `src/models/qwen/README.md`, `dflash_policy.hpp`, `EXPERIMENTS.md` (a chain of at most 7 drafts, its cost tables), `QUALITY.md` (greedy DFlash2 equal to AR on 160 requests), `prefill_quant_gemm.hip` (INT8 WMMA), `BUILD-WINDOWS.md` (TheRock 10.0.0, clang-cl, MSVC, vcpkg; 10.1+ stalls);
  - the Windows fork github.com/pixmaate/gufo, `docs/WINDOWS.md` (the DXGI budget, backing beyond the carve-out, the Windows-only switches and their 7.6–9.5%).
- **llama.cpp:**
  - PRs #22105 (DFlash), #27342 (DFlash2), #30663, #30931 (draft: tree speculation), #25278 (recurrent rollback), #26046;
  - `docs/speculative.md`;
  - release b11907, run on the box.
- **The drafter:** huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF (README; a mirror of incoai/Qwen3.8-27B-DFlash2-GGUF).
- **halogen-server:** github.com/peonist-ai/halogen-server (README: 31.71 tokens/s DFlash2 mean, prefill 566 at 32K, the 240 GB/s ceiling, W4A4 prefill at 64+ rows).
- **RDNA 3 WMMA rates:** gpuopen.com/learn/wmma_on_rdna3/; LLVM `BuiltinsAMDGPU.td` (FP8 WMMA gfx12/gfx1170 only) and `GCNFeatures.td`.
- **hipBLASLt on Strix Halo:** llm-tracker.info/_TOORG/Strix-Halo (36.9 TFLOPS BF16).
- **Bandwidth, third-party:** github.com/lhl/strix-halo-testing (memtest_vulkan 214.6–221.0 GB/s).
- **ROCm on Windows:** rocm.docs.amd.com (HIP SDK system requirements list gfx1151).

From this repository: `src/` at `c5ec4ec` (§2's inventory); [BACKLOG](./DREAMFERENCE_LING_ENGINE_BACKLOG.md) §19 items 19, 26, 27 and 28; [MODEL](./DREAMFERENCE_LING_ENGINE_MODEL.md) §5; [OVERVIEW](./DREAMFERENCE_LING_ENGINE_OVERVIEW.md) §4; [ARCHITECTURE](./DREAMFERENCE_LING_ENGINE_ARCHITECTURE.md) §8–9; [INTEGRATION](./DREAMFERENCE_LING_ENGINE_INTEGRATION.md) §12; `bench/agentlong/manifest.json`.

From Mightling: `specs/DREAMFERENCE_MIGHTLING_AMD_NODES.md` §1–§13 (the box, its setup, the Windows memory model, the node pieces, the first llama.cpp measurements); WINDOWS_ARM §8.3, §9 and §16.3 as cited there.
