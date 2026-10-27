// One launch for a projection's decode rows (a verify of up to 32 rows) and a prefill chunk's rows (BATCHING's
// concurrency plan, C2): every block streams its weight panel through shared memory once, and two consumers read
// each stage there, the prefill GEMM's (prefill_tile.cuh: the same Loader, compute_stage and epilogue) and the
// weight-streaming GEMM's (stream_chunk.cuh: the same per-chunk dequantization and MMAs). Each output is computed
// with exactly the instructions, operands and order of its own path, so the decode rows are bit for bit what
// stream_gemm gives them and the chunk rows what prefill_gemm / prefill_swiglu gives them (ling-fused-tests).
//
// The two paths order K differently, and both orders are kept:
//   - prefill: one accumulator per output, over the K stages in order (stage kt = 64 bytes per row: NVFP4, 128
//     values, one tiled block per 16 weight rows; FP8, 64 values, half a block), two MMA K steps per stage. Its bits
//     do not depend on the block tile (BN, BM, warp layout), only on that sequence.
//   - stream: warp w of a block sums chunks w, w + 8, w + 16, ... (128 values each) in order; the eight warps' sums
//     are added in warp order, then scaled: (sum * gscale) * xinv[n]. Its bits do not depend on TILES or NT, only
//     on that chunk-to-chain assignment (and ksplit, which is 1 unless LING_KSPLIT is set; the fused launch
//     refuses a split).
// Here the chunks arrive in K order, and chunk c goes into decode chain c % 8: each chain sees exactly the chunks
// one stream warp sees, in the same order, and the epilogue adds the chains 0..7 as the stream kernel adds its
// warps. The price is eight live accumulators per decode output instead of one: with a 128-row panel, one 16-row
// weight tile per warp and two 8-row column groups per block, 64 registers per thread (221-225 in all, no spills).
// A block therefore holds at most 16 decode rows: past 16 the next blocks of the panel (blockIdx.x) take the next
// groups, and where the chunk has fewer M tiles than that, the launch adds decode-only blocks, which stream the same
// panel through L2 next to the prefill's blocks of that panel (as the prefill's own M tiles do).
//
// A stage's tile in shared memory holds exactly the bytes a stream lane loads. NVFP4: lane (g, t) of the decode warp
// for tile q takes 16 value bytes at row 16q + g + 8h, unit t (swizzled as the prefill stores them) and the two E4M3
// scale bytes at that row's offset 2t, the tiled block's slot (h * 8 + g) * 4 + t. A coded blob (B26) is read with
// its raw scales (wsf), the bytes its codes decode to, so the decode takes the raw path's instructions. FP8: a stage
// is half a chunk (values [0, 64) at even kt, [64, 128) at odd) and each of a chunk's eight MMA steps reads all four
// lanes' 32-value runs, so lanes t < 2 copy their 2 x 32 bytes (units 2t, 2t + 1 of rows g, g + 8) to registers at
// stage 2c, lanes t >= 2 theirs at 2c + 1, and the chunk's MMAs run at 2c + 1 (slot 2c is refilled only after the
// next iteration's barrier).
//
// Experiments (all bit-identical; measured with ling-fused-tests --bench, gate|up at a 256-row chunk beside 16
// decode rows, 2026-10-27): one column group per block, spreading the decode's MMAs over a panel's two prefill
// blocks, 0.773 ms against 0.687 (each block dequantizes the tile again); 64-row panels with two warps per tile on
// the narrow down projection, 0.590 against 0.612 at 256 + 16 but slower at 128 and past 16 rows. Both removed.
// The decode's activations are prefetched into L1 a chunk ahead (LING_FUSED_XPF=0 drops it: gate|up at 128 rows
// beside 12, 0.741 -> 0.588 ms with it; down 0.442 -> 0.312).
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "core/kernels.cuh"
#include "core/prefill_tile.cuh"
#include "core/stream_chunk.cuh"

namespace ling::kernels {
namespace {

using namespace pf;

constexpr int kFusedBN = 128, kFusedStages = 4, kFusedNTC = 2;  // the panel, pipeline depth, decode groups per block

struct FusedDecode {
  const __half* x;    // [M][K] FP16, power-of-two row scales (to_half_rows / rmsnorm_half)
  const float* xinv;  // [M]
  int M;
  float* y[2];        // [M][N]: plain, y[0]; SWIGLU, the gate's and the up projection's
  float gscale[2];    // the weights' global scales (stream_gemm's scale2, or the FP8 per-tensor scale)
  int N;              // output columns of each matrix
};

// Block (r, nb): weight panel nb (BN rows: plain, rows nb * BN ..; SWIGLU, BN / 2 gate rows and the same up rows,
// in the prefill's interleaving); prefill M tile r when r < mtiles; decode column groups [(r - dbase) * NTC, + NTC)
// when r >= dbase (dbase 0: the decode on the prefill's blocks; mtiles: on blocks of its own).
template <bool FP4, int BN, int STAGES, bool SWIGLU, bool OUT_BF16, int NTC>
__global__ void __launch_bounds__(kThreads)
    fused_gemm_kernel(GemmArgs p, FusedDecode d, int mtiles, int dbase, int xpf) {
  using S = Shape<BN, STAGES>;
  constexpr int NF = S::kNF, SPC = FP4 ? 1 : 2;  // stages per 128-value chunk
  static_assert(BN == 16 * 8, "one 16-row weight tile per warp");
  static_assert(FP4 || !SWIGLU, "SwiGLU is the NVFP4 FFN's");
  extern __shared__ __align__(128) uint8_t smem[];
  const int r = blockIdx.x, n0 = blockIdx.y * BN;
  const bool pre = r < mtiles;
  const int m0 = (pre ? r : 0) * kBM;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp & 1, wn = warp >> 1;
  const int g = lane >> 2, t = lane & 3;
  const int KT = p.K / kRowBytes / (FP4 ? 2 : 1);

  float acc[4][NF][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
  // chain[c]: stream_chunk's accumulators [TILES = 1][NT = NTC][4] of the stream warp that takes chunks c, c + 8, ...
  float chain[8][1][NTC][4];
#pragma unroll
  for (int c = 0; c < 8; ++c)
#pragma unroll
    for (int j = 0; j < NTC; ++j) chain[c][0][j][0] = chain[c][0][j][1] = chain[c][0][j][2] = chain[c][0][j][3] = 0.f;
  // This block's decode rows: x from row dr0 on, as a stream launch of M - dr0 rows sees them.
  const int dr0 = 8 * NTC * (r - dbase);
  const bool dec = r >= dbase && dr0 < d.M;
  const __half* xd = d.x + static_cast<size_t>(dec ? dr0 : 0) * p.K;
  const int md = d.M - dr0;
  const uint8_t notab[1] = {0};  // stream_chunk's table argument, unused on the raw path

  // A decode-only block (r >= mtiles) still runs the prefill's loader, its activation copies pointed at the weights
  // (any readable bytes: nothing computes with them), so its weight copies are the prefill blocks' own.
  GemmArgs pl = p;
  if (!pre) {
    pl.xq = p.w;
    pl.xs = p.w;
    pl.M = 1;
  }
  const Loader<FP4, BN, STAGES, SWIGLU> loader(pl, m0, n0);
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) loader.load(smem, s, s);
    cp_commit();
  }
  // The decode tile of this warp: shared-memory rows 16 * warp .. + 15 of each stage's B tile.
  const int row_h0 = 16 * warp + g;
  sg::WeightRegs<1, FP4, false> wr;
  for (int kt0 = 0; kt0 < KT; kt0 += 8 * SPC) {
#pragma unroll
    for (int u = 0; u < 8 * SPC; ++u) {
      const int kt = kt0 + u, c = kt / SPC;
      if (kt < KT) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int nk = kt + STAGES - 1;
        if (nk < KT) loader.load(smem, nk % STAGES, nk);
        cp_commit();
        const uint8_t* st = smem + (kt % STAGES) * S::kStageBytes;
        const uint8_t* sb = st + S::kStageA;
        // xpf: the next chunk's activations (each lane's 64 bytes per row) into L1 ahead of stream_chunk's loads,
        // so their latency hides behind this stage's MMAs; a hint only.
        if (dec && xpf && u % SPC == 0 && c + 1 < KT / SPC) {
#pragma unroll
          for (int j = 0; j < NTC; ++j)
            if (g + 8 * j < md)
              asm volatile("prefetch.global.L1 [%0];" ::"l"(xd + static_cast<size_t>(g + 8 * j) * p.K + (c + 1) * 128 + t * 32));
        }
        if (pre) compute_stage<FP4, BN, STAGES>(st, acc, lane, wm, wn);
        if (dec) {
          if constexpr (FP4) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int row = row_h0 + 8 * h;
              wr.v[0][h][0] = *reinterpret_cast<const uint4*>(sb + swz(row, t));
              wr.sc[0][h] = *reinterpret_cast<const uint16_t*>(sb + S::kStageB + S::kStageSA + row * 8 + 2 * t);
            }
            sg::stream_chunk<NTC, 1, true, false>(xd, md, p.K, c, g, t, wr, notab, chain[u]);
          } else {
            if ((t >> 1) == (u & 1)) {  // this stage holds the lane's 32 values
#pragma unroll
              for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int q = 0; q < 2; ++q)
                  wr.v[0][h][q] = *reinterpret_cast<const uint4*>(sb + swz(row_h0 + 8 * h, 2 * (t & 1) + q));
            }
            if (u & 1) sg::stream_chunk<NTC, 1, false, false>(xd, md, p.K, c, g, t, wr, notab, chain[u >> 1]);
          }
        }
      }
    }
  }
  cp_wait<0>();
  if (pre) epilogue<BN, STAGES, SWIGLU, OUT_BF16>(p, acc, m0, n0, lane, wm, wn);
  if (!dec) return;
  // The stream kernel's epilogue: the chains (its warps) added in order, then (sum * gscale) * xinv.
  int mat = 0, col0;
  if constexpr (SWIGLU) {
    mat = warp & 1;  // the prefill's interleaving: each warp column's 32 rows are 16 gate rows, then 16 up rows
    col0 = n0 / 2 + (warp >> 1) * 16;
  } else {
    col0 = n0 + 16 * warp;
  }
  float* __restrict__ y = d.y[mat];
  const float gscale = d.gscale[mat];
#pragma unroll
  for (int j = 0; j < NTC; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = dr0 + 8 * j + 2 * t + (e & 1), rr = g + 8 * (e >> 1);
      if (n >= d.M) continue;
      float sum = 0.f;
#pragma unroll
      for (int c = 0; c < 8; ++c) sum += chain[c][0][j][e];
      y[static_cast<size_t>(n) * d.N + col0 + rr] = sum * gscale * d.xinv[n];
    }
}

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

int env_int(const char* name, int d) {
  const char* e = std::getenv(name);
  return e ? std::atoi(e) : d;
}

template <bool FP4, bool SWIGLU, bool OUT_BF16>
void launch_fused(const GemmArgs& p, const FusedDecode& d, int n_rows, cudaStream_t s, const char* what) {
  using S = Shape<kFusedBN, kFusedStages>;
  auto kernel = fused_gemm_kernel<FP4, kFusedBN, kFusedStages, SWIGLU, OUT_BF16, kFusedNTC>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSmemBytes);
    configured = true;
  }
  if (d.M < 0 || d.M > kMaxStreamRows) throw std::runtime_error(std::string(what) + ": decode rows must be in [0, 32]");
  if (env_int("LING_KSPLIT", 0) != 0) throw std::runtime_error(std::string(what) + ": split K is not reproduced");
  static const int sep_env = env_int("LING_FUSED_SEP", -1), xpf = env_int("LING_FUSED_XPF", 1);
  const int mtiles = (p.M + kBM - 1) / kBM, groups = (d.M + 7) / 8, panels = n_rows / kFusedBN;
  // The decode on blocks of its own when it would otherwise load one of a panel's several prefill blocks and the
  // launch is under two waves (the narrow projections, 40 panels, at 256 rows: down 0.551 -> 0.473 ms, the FP8
  // out-projection 0.414 -> 0.392 beside 12 rows); on the first prefill block otherwise (down at 128 rows: 0.312
  // against 0.449 on blocks of its own; gate|up at 256: 0.658 against 0.721). LING_FUSED_SEP=0 or 1 forces it.
  const bool sep = sep_env >= 0 ? sep_env != 0 : mtiles >= 2 && groups <= kFusedNTC && panels * mtiles < 96;
  const int dbase = sep ? mtiles : 0, blocks = std::max(mtiles, dbase + (groups + kFusedNTC - 1) / kFusedNTC);
  if (blocks == 0) return;
  kernel<<<dim3(blocks, panels), kThreads, S::kSmemBytes, s>>>(p, d, mtiles, dbase, xpf);
  check_launch(what);
}

}  // namespace

void fused_gemm_nvfp4(const __half* xh, const float* xinv, int Md, float* yd, float scale2, const uint8_t* xq,
                      const uint8_t* xs, int Mp, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                      bool accumulate, cudaStream_t s, const uint8_t* wsf) {
  if (N % kFusedBN != 0 || K % 128 != 0) throw std::runtime_error("fused_gemm_nvfp4: N % 128 and K % 128 must be 0");
  GemmArgs p{xq, xs, w, alpha, y, ldy, Mp, N, K, accumulate};
  p.wsf = wsf;
  launch_fused<true, false, false>(p, FusedDecode{xh, xinv, Md, {yd, nullptr}, {scale2, 0.f}, N}, N, s,
                                   "fused_gemm_nvfp4");
}

void fused_swiglu_nvfp4(const __half* xh, const float* xinv, int Md, float* yd_gate, float* yd_up, float scale2_gate,
                        float scale2_up, const uint8_t* xq, const uint8_t* xs, int Mp, const uint8_t* gate,
                        float alpha_gate, const uint8_t* up, float alpha_up, int N, int K, float out_input_scale,
                        uint8_t* q_out, uint8_t* sf_out, cudaStream_t s, const uint8_t* gate_sf, const uint8_t* up_sf) {
  if (N % 128 != 0 || K % 128 != 0) throw std::runtime_error("fused_swiglu_nvfp4: N % 128 and K % 128 must be 0");
  GemmArgs p{xq, xs, gate, alpha_gate, nullptr, 0, Mp, N, K, false};
  p.w2 = up;
  p.alpha2 = alpha_up;
  p.gscale_out = 1.f / out_input_scale;
  p.q_out = q_out;
  p.sf_out = sf_out;
  p.wsf = gate_sf;
  p.wsf2 = up_sf;
  launch_fused<true, true, false>(p, FusedDecode{xh, xinv, Md, {yd_gate, yd_up}, {scale2_gate, scale2_up}, N}, 2 * N, s,
                                  "fused_swiglu_nvfp4");
}

void fused_gemm_fp8(const __half* xh, const float* xinv, int Md, float* yd, float wscale, const uint8_t* xq, int Mp,
                    const uint8_t* w, float alpha, float* y, int ldy, int N, int K, bool accumulate, cudaStream_t s) {
  if (N % kFusedBN != 0 || K % 128 != 0) throw std::runtime_error("fused_gemm_fp8: N % 128 and K % 128 must be 0");
  launch_fused<false, false, false>(GemmArgs{xq, nullptr, w, alpha, y, ldy, Mp, N, K, accumulate},
                                    FusedDecode{xh, xinv, Md, {yd, nullptr}, {wscale, 0.f}, N}, N, s, "fused_gemm_fp8");
}

void fused_gemm_fp8_bf16out(const __half* xh, const float* xinv, int Md, float* yd, float wscale, const uint8_t* xq,
                            int Mp, const uint8_t* w, float alpha, __nv_bfloat16* y, int ldy, int N, int K,
                            cudaStream_t s) {
  if (N % kFusedBN != 0 || K % 128 != 0) throw std::runtime_error("fused_gemm_fp8_bf16out: N % 128 and K % 128 must be 0");
  GemmArgs p{xq, nullptr, w, alpha, nullptr, ldy, Mp, N, K, false};
  p.y_bf16 = y;
  launch_fused<false, false, true>(p, FusedDecode{xh, xinv, Md, {yd, nullptr}, {wscale, 0.f}, N}, N, s,
                                   "fused_gemm_fp8_bf16out");
}

}  // namespace ling::kernels
