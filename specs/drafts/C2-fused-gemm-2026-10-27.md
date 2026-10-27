# C2: one weight read for a decode verify and a prefill chunk (feasibility, 2026-10-27)

BATCHING's concurrency plan needs a pass that serves the decoding streams' verify rows (12-32) and a prefill chunk
(128-512 rows) from one read of each weight tile, each path's arithmetic unchanged. This is the prototype: the
kernel (`src/core/fused_gemm.cu`), its byte-for-byte tests and benchmark (`ling-fused-tests`, `--bench`); not the
integration.

## 1. The two reduction orders

Both paths read the same tiled weights (`retile_nvfp4` / `retile_fp8`: per 16-row tile, one block per 128-value
chunk, laid out in stream-lane order).

| | Stream (`stream_gemm`, M <= 32) | Prefill (`prefill_gemm`, `prefill_swiglu`) |
| --- | --- | --- |
| Activations | FP16, a power-of-two scale per row (`xinv`) | NVFP4 (E4M3 per 16, input_scale global) or FP8 (static per-tensor) |
| Weights | dequantized to FP16: NVFP4 `cvt e2m1x2` x E4M3 scale (`mul.rn.f16x2`); FP8 `cvt e4m3x2` | used as stored, block-scaled |
| MMA | m16n8k16 f16 -> f32, weights as A, tokens as B; logical k permuted onto each lane's 32 contiguous values | NVFP4 `kind::mxf4nvf4` m16n8k64 (scale_vec 4X); FP8 `kind::f8f6f4` m16n8k32 |
| K order | warp w of 8 takes chunks w, w + 8, ... in order, 8 MMA steps each; warps summed 0..7 in order from 0.f | one accumulator over 64-byte stages in K order, 2 MMA steps per stage (NVFP4 stage = 1 chunk, FP8 = half) |
| Scale | `(sum * gscale) * xinv[n]` | `alpha * acc` (FP32, `+=`, BF16, or SwiGLU into NVFP4) |
| Invariant to | TILES, NT, M (and ksplit: 1 unless `LING_KSPLIT`) | BN, BM, warp layout, M |

Per projection (bytes per decode step, 17.56 GB in all): FFN gate|up NVFP4 (6.42 GB; prefill as `prefill_swiglu`),
down NVFP4 (3.21 GB; prefill `+=` into the residual), DeltaNet in_qkv|z FP8 (4.03 GB; prefill BF16 out), DeltaNet
out FP8 (1.51 GB, `+=`), attention q|k|v FP8 (1.17 GB) and o (0.50 GB, `+=`), lm head NVFP4 (0.72 GB: decode only,
the prefill needs one row). The a/b BF16 projections are 0.03 GB.

## 2. A common tile exists for every NVFP4 and FP8 projection

Neither order depends on the tiling, so the fused block takes the prefill's (128 tokens x 128 weight rows, 4
cp.async stages, 8 warps) and keeps the stream's order by bookkeeping: stage kt's chunk c goes into decode chain
c % 8, one chain per stream warp, and the epilogue adds chains 0..7 as the stream kernel adds its warps. Warp w
decodes 16-row tile w of the stage from shared memory: the swizzled stage holds exactly the bytes a stream lane
loads (NVFP4 value unit t of rows g and g + 8, and the scale bytes at row * 8 + 2t). A coded blob (B26) is read with
its raw scales, the bytes its codes decode to. FP8: a stage is half a chunk, and each of the chunk's 8 MMA steps
needs all four lanes' 32-value runs, so lanes t < 2 copy their bytes to registers at stage 2c, lanes t >= 2 at
2c + 1, and the MMAs run at 2c + 1. Both paths' code is shared, not copied: `prefill_tile.cuh` (Loader,
compute_stage, epilogue) and `stream_chunk.cuh` (one chunk's dequantization and MMAs).

The cost is eight live accumulators per decode output: 2 x 8-row groups per block = 64 registers (220-230 in all,
no spills), so a block carries at most 16 decode rows; rows 17-32 go to the panel's next block (a second M tile, or
a decode-only block that streams the same panel through L2 beside it).

## 3. Result

Byte for byte: 132 NVFP4 cases (gate|up and down, raw and coded, decode 0-32 rows x chunk 0-512 rows) and 82 FP8
cases (in_qkv BF16 out, out-projection `+=`) identical to `stream_gemm*` and `prefill_*`, with the decode on the
prefill's blocks and on blocks of its own. `ling-stream-tests` and `ling-prefill-tests` pass on the refactored
headers (prefill SASS byte-identical; stream kernels rescheduled, the same floating-point instruction multiset).

Times (ms, L2 flushed before each run; "concurrent" is two plain streams, not production's green context), decode
12-16 rows / 24 rows:

| Chunk 256 | stream | prefill | back to back | concurrent | fused |
| --- | --- | --- | --- | --- | --- |
| gate\|up (100 MB) | 0.48 | 0.56 | 1.00 | 0.98 | 0.66-0.69 / 0.79 |
| down (50 MB) | 0.28 / 0.31 | 0.31 | 0.55 / 0.58 | 0.55 / 0.56 | 0.49-0.51 / 0.45 |
| in_qkv\|z FP8 (84 MB) | 0.40 / 0.41 | 0.86 | 1.23 / 1.25 | 1.21 / 1.22 | 0.87 / 0.87 |
| out FP8 (31 MB) | 0.18 | 0.37 | 0.51 / 0.52 | 0.51 | 0.39 / 0.39 |

At 128 rows beside 12-16: gate|up 0.58 against 0.96 back to back, down 0.31 against 0.52, FP8 0.82 / 0.36 against
1.19 / 0.50. At 512: gate|up 1.02-1.06 against 1.23, down 0.55-0.58 against 0.68. Measured in the final window
(15:03 UTC); three earlier windows agree within ~4%. The FP8 rows are before the L2 prefetch of §6; with it, §4 and §6.

## 4. What a pass would cost

Summed over the model at a 256-token chunk beside 12-24 decode rows, as first measured: fused GEMMs ~160 ms (FFN
74-80, DeltaNet 61-62, attention ~19, lm head ~4) against ~215 ms back to back; with the state kernels (decode ~24 ms,
the chunk's attention and recurrence ~25-30 ms) about 205-215 ms a pass. The pass was set by the prefill GEMMs at 256
rows, and the FP8 one was slow there: 0.86 ms for in_qkv|z, 97 GB/s of weights, nearly constant in M (0.82 / 0.86 /
0.94 ms at 128 / 256 / 512 rows).

With the FP8 weights prefetched into L2 ahead of the MMAs (§6) the FP8 fused launches at 256 rows take 0.55-0.57 ms
for in_qkv|z and 0.31 for out beside 12-16 decode rows, 0.75 and 0.32 beside 24 (FFN unchanged: gate|up 0.67-0.69 /
0.80, down 0.45-0.47 / 0.46, measured in the same window). DeltaNet comes to ~41-42 ms (51 beside 24) and attention to
~13 (~16, its q|k|v scaled from in_qkv|z by bytes), so the fused GEMMs ~129-134 ms beside 12-16 rows and ~151 beside
24, and a pass ~180-188 ms and ~200-205 ms. The decode rows no longer ride free on the FP8 matrices: their
dequantization shows now that the weights arrive faster (in_qkv|z beside 16 rows: 0.58 against 0.47 for the chunk
alone at 128 rows, 0.57 against 0.53 at 256); past 16 rows (a second block per panel) more so. Closing that, and the
gate|up overhead over max(read, compute), would bring a pass toward ~165 ms.

## 5. Risks

- The ~0.1 ms per gate|up the decode adds is its dequantization (`cvt` + `mul.rn.f16x2` per value), which is the
  stream path's arithmetic and cannot change; spreading it over a panel's blocks dequantizes twice (measured worse).
- Wave quantization on the narrow matrices (40 panels): the launch policy (decode on its own blocks under two
  waves) is tuned on four shapes and one chunk size.
- Decode rows past 16 need a second block per panel; past 32 not supported (`kMaxStreamRows`).
- Production details not reproduced yet: PDL, the megakernel (`ffn_persistent`, same bits but its own fusion of
  silu_mul and the residual), split K (refused), multi-matrix launches (one launch per matrix; bits unchanged).
- The pass also needs the two paths' activations ready at once: the engine must run a verify's layer and a chunk's
  layer in lockstep (one schedule for both), which is the integration's design problem, not this kernel's.

## 6. The FP8 prefill loader (2026-10-27, branch `batching/c2-fp8-prefill`)

Why the FP8 prefill GEMM streamed its weights at ~100 GB/s. node-b's GPU counters are for administrators only
(`RmProfilingAdminOnly`), so ncu could not run; instead, variants of the kernel (`prefill_tile.cuh`'s loader and
`compute_stage`, timed with the L2 flushed, in_qkv|z's shape N 16,384 x K 5,120):

| 128 rows, 128 x 128 tile, 4 stages | ms | GB/s of weights |
| --- | --- | --- |
| the kernel | 0.848 | 99 |
| its copies only (no MMAs) | 0.849 | 99 |
| the weight copies only | 0.834 | 101 |
| the MMAs only (no copies) | 0.155 | |
| NVFP4, the same weight bytes, its copies only | 0.546 | 173 |
| one 128-value chunk per pipeline step (half the barriers per byte) | 0.844 | 99 |
| the same, each warp's copies covering whole 128-byte lines (weights, activations) | 0.830 / 0.832 | 101 |
| two blocks per SM (3 stages) / 64-row panels | 0.843 / 0.851 | 99 |
| the weight tiles one chunk further apart | 0.847 | 99 |
| each tile's 2,048-byte block prefetched into L2 one / two / four / eight chunks ahead | 0.839 / 0.482 / 0.490 / 0.513 | 100 / 174 / 171 / 164 |
| one chunk per step, 3 steps (the copies two chunks ahead), with / without the prefetch two ahead | 0.703 / 0.844 | 119 / 99 |
| one chunk per step, 2 steps (the copies one chunk ahead, whole-line copies), with the prefetch two ahead | 0.490 | 171 |

So the kernel's time is its weight copies' time, and neither the barriers, the MMA shape, half-line copies,
occupancy nor the tile stride sets it; nor does the distance the copies run ahead (three one-chunk steps reach two
chunks ahead and gain nothing). What helps is a tile's tiled block requested whole, by one bulk L2 prefetch, before
the stage copies ask for any of it: two chunks ahead of the MMAs, where the copies have not been yet (one chunk ahead
they have, and it adds nothing; where the copies also reach two chunks, it gains less). More bytes in flight from
more blocks (two per SM) do not substitute for it. Why the memory system rewards the whole-block request for FP8 (a
2,048-byte block fetched in one go rather than as 128 sixteen-byte copies interleaved with ~380 other tiles' copies;
the L2's fill) while the NVFP4 copies reach 173 GB/s without it, the counters would tell; the remedy does not depend
on the answer.

The fix (`l2_prefetch_fp8`, `kL2LeadFp8` = 2): every FP8 block, in `prefill_gemm` and in the fused kernel, has
threads 0 .. BN / 16 - 1 issue `cp.async.bulk.prefetch.L2` for chunk c + 2 of their weight tile at chunk c (and
chunks 0 and 1 at the start). It is a hint: the stage copies, the MMAs and their order are untouched, so every output
bit is unchanged. The NVFP4 kernels compile to the same SASS as before (`cuobjdump -sass`, all eight prefill and both
NVFP4 fused instantiations); the prefetch did not help NVFP4 (0.553 against 0.551 ms).

Exactness: `prefill_gemm_fp8` and `prefill_gemm_fp8_bf16out` against the previous build, 336 cases byte for byte
(seven shapes: in_qkv, in_z, in_qkv|z, the DeltaNet out-projection, attention q|k|v, o and a small one; FP32 `=`,
`+=` and BF16 out; 16 row counts from 1 to 2,048; each under LING_GEMM_SHAPE 0, 1 and 2), all identical;
`ling-prefill-tests`, `ling-stream-tests` and `ling-fused-tests` (214 of 214 identical) pass; a cold agentlong greedy
run at 1 stream with ling-serve is 16 of 16 turns identical to the reference (`ref-e1g`).

| ms (GB/s of weights) | 128 rows | 256 rows | 512 rows |
| --- | --- | --- | --- |
| in_qkv\|z, before | 0.817 (103) | 0.857 (98) | 0.934 (90) |
| in_qkv\|z, after | 0.462 (182) | 0.524 (160) | 0.701 (120) |
| out, before | 0.329 (95) | 0.350 (90) | 0.438 (72) |
| out, after | 0.205 (154) | 0.231 (136) | 0.360 (87) |

Single-stream prefill (cold agentlong greedy, 1 stream, 520,529 new prompt tokens in chunks of ~2,048 rows, where the
FP8 GEMMs are compute-bound): 1,812 tok/s before (287.2 s of prefill), 1,831 after (284.3 s), +1%; decode
64.1 tok/s both; both runs 16 of 16 turns identical to the reference and to each other (same window pair, 17:04 and
16:56 UTC).
