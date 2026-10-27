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
(15:03 UTC); three earlier windows agree within ~4%.

## 4. What a pass would cost

Summed over the model at a 256-token chunk beside 12-24 decode rows: fused GEMMs ~160 ms (FFN 74-80, DeltaNet
61-62, attention ~19, lm head ~4) against ~215 ms back to back; with the state kernels (decode ~24 ms, the chunk's
attention and recurrence ~25-30 ms) about 205-215 ms a pass. The decode rows ride almost free (FP8: fused =
prefill alone), so the pass is set by the prefill GEMMs at 256 rows, and the FP8 one is slow there: 0.86 ms for
in_qkv|z is 97 GB/s of weights and ~50 TFLOPS, against ~200 GB/s for the stream kernel, and nearly constant in M
(0.82 / 0.86 / 0.94 ms at 128 / 256 / 512 rows), so not compute-bound. The cause is not profiled; candidates: half a
chunk per stage (twice the barriers of NVFP4 per weight byte) and the FP8 MMA shape. With the FP8 loader at the stream's rate the FP8 parts drop
to ~37 ms and the pass to ~165-170 ms; with the gate|up overhead over max(read, compute) (0.66 against 0.58) closed
as well, ~155 ms.

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
