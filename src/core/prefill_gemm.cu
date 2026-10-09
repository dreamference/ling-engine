// Prefill GEMMs (M > 32 rows) on block-scaled tensor cores, reading the weights in the tiled layout the
// decode kernels read (stream_gemm.cu), so no second copy of any weight exists.
//
// Numerics are production's (specs/DREAMFERENCE_LING_ENGINE_SESSIONS.md §11): the NVFP4 FFN multiplies NVFP4 activations (E2M1 values, one
// E4M3 scale per 16, the checkpoint's input_scale as the global scale) with mma.sync kind::mxf4nvf4, and
// the FP8 projections multiply FP8 E4M3 activations (static per-tensor input_scale) with kind::f8f6f4.
// Accumulation is FP32; y = alpha * acc with alpha = input_scale * weight scale, as SGLang computes it.
//
// Tiling: a block computes 128 tokens x 256 weight rows (3 stages) or, when that leaves under two waves,
// 128 x 128 (4 stages), with 8 warps (2 x 4). One pipeline stage holds 64 bytes of K per row (128 NVFP4
// values, one tiled block's width; or 64 FP8 values, half of one): activations and weights in shared
// memory with their 16-byte units XOR-swizzled by row pair, so every ldmatrix phase is free of bank
// conflicts, plus the rows' E4M3 block scales; cp.async fills the stages ahead of the MMAs. The FFN's gate
// and up run as one GEMM whose epilogue applies SwiGLU and writes the down projection's NVFP4 input.
//
// Measured at 2,048 rows: NVFP4 230-290 TFLOPS (CUTLASS 4.5.1's SM120 example: 325 on the gate shape, 262
// on the down shape), FP8 ~133. A warp-specialized variant (one producer warp issuing every cp.async of a
// stage, mbarriers between it and 8 consumer warps) ran FP8 ~8% faster and NVFP4 3-4x slower: one warp
// cannot issue a stage's ~1,800 copies in the time the MMAs take. TMA would issue them in a few
// instructions, but the FP8 tiled layout (decode's) has no 64-byte row runs for a swizzled box.
#include "core/kernels.cuh"

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

constexpr int kBM = 128, kThreads = 256;
constexpr int kRowBytes = 64;  // per row per stage
constexpr int kTileBytesFp4 = 16 * 64 + 16 * 8, kTileBytesFp8 = 16 * 128;

// A block computes kBM tokens x BN weight rows; 8 warps, 2 along M x 4 along N, each 64 x BN / 4.
template <int BN, int STAGES>
struct Shape {
  static constexpr int kStageA = kBM * kRowBytes, kStageB = BN * kRowBytes, kStageSA = kBM * 8, kStageSB = BN * 8;
  static constexpr int kStageBytes = kStageA + kStageB + kStageSA + kStageSB;
  static constexpr int kSmemBytes = STAGES * kStageBytes;
  static constexpr int kNF = BN / 4 / 8;  // n8 fragments per warp
  static_assert(kBM * 4 == 2 * kThreads && (BN * 4) % kThreads == 0 && kBM + BN / 2 <= kThreads, "copy split");
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp8(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

// Byte offset of (row, 16-byte unit) in a stage's value tile.
__device__ __forceinline__ int swz(int r, int u) { return r * kRowBytes + ((u ^ ((r >> 1) & 3)) << 4); }

template <bool FP4>
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                    uint32_t sfb) {
  if constexpr (FP4) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(uint16_t(0)),
          "h"(uint16_t(0)), "r"(sfb), "h"(uint16_t(0)), "h"(uint16_t(0)));
  } else {
    (void)sfa;
    (void)sfb;
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}

struct GemmArgs {
  const uint8_t* xq;  // activations: NVFP4 packed [M][K/2] or FP8 [M][K]
  const uint8_t* xs;  // NVFP4 activation scales [M][K/16] (E4M3)
  const uint8_t* w;   // tiled weights
  float alpha;
  float* y;           // [M][ldy]
  int ldy;
  int M, N, K;        // N: output columns
  bool accumulate;    // y += alpha * acc
  // SwiGLU mode: w is the gate, w2 the up projection (alpha2 its scale), and the output is
  // NVFP4(silu(gate) * up) with global scale gscale_out into q_out [M][N/2] and sf_out [M][N/16].
  const uint8_t* w2 = nullptr;
  float alpha2 = 0.f, gscale_out = 0.f;
  uint8_t* q_out = nullptr;
  uint8_t* sf_out = nullptr;
};

// The weight row behind row r of a block's B tile (block n0 = blockIdx.y * BN). Plain: row n0 + r. SwiGLU:
// each warp's WN rows are WN / 2 gate rows then the up rows of the same outputs, so a thread holds both
// products of its output columns.
template <int BN, bool SWIGLU>
__device__ __forceinline__ const uint8_t* weight_row(const GemmArgs& p, int n0, int r, int& row) {
  if constexpr (!SWIGLU) {
    row = n0 + r;
    return p.w;
  } else {
    constexpr int WN = BN / 4, HALF = WN / 2;
    static_assert(HALF % 16 == 0, "whole 16-row weight tiles");
    const int wn = r / WN, within = r % WN;
    row = n0 / 2 + wn * HALF + within % HALF;
    return within < HALF ? p.w : p.w2;
  }
}

// Each thread's share of a stage's copies, with the global addresses of stage 0 and the shared-memory
// offsets computed once: a stage then costs a few additions per copy. Thread i copies 16-byte pieces
// i and i + 256 of the activations and of the weights, and the block scales of one row (activations,
// threads 0-127) or of two rows (weights, threads 128-191).
template <bool FP4, int BN, int STAGES, bool SWIGLU = false>
struct Loader {
  using S = Shape<BN, STAGES>;
  static constexpr int kBPieces = BN * 4 / kThreads;
  const uint8_t* a[2];
  const uint8_t* b[kBPieces];
  const uint8_t* sf = nullptr;
  uint32_t sa[2], sb[kBPieces], ssf = 0;
  int sf_kind = 0;  // 0 none, 1 activation scales (8 bytes), 2 weight scales (16 bytes)

  __device__ Loader(const GemmArgs& p, int m0, int n0) {
    const int tid = threadIdx.x, chunks = p.K / 128;
    const size_t xrow = FP4 ? p.K / 2 : p.K;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int i = tid + j * kThreads, r = i >> 2, u = i & 3;
      a[j] = p.xq + static_cast<size_t>(min(m0 + r, p.M - 1)) * xrow + u * 16;
      sa[j] = swz(r, u);
    }
#pragma unroll
    for (int j = 0; j < kBPieces; ++j) {
      const int i = tid + j * kThreads, r = i >> 2, u = i & 3;
      int row;
      const uint8_t* w = weight_row<BN, SWIGLU>(p, n0, r, row);
      const int tile = row >> 4, rr = row & 15;
      if constexpr (FP4) {
        b[j] = w + static_cast<size_t>(tile) * chunks * kTileBytesFp4 + (rr * 4 + u) * 16;
      } else {
        // FP8 tiled block: piece ((h * 2 + q) * 8 + g) * 4 + t holds row g + 8h, values [32t + 16q, +16).
        // Stage kt covers half kt & 1 of chunk kt >> 1: units 4 (kt & 1) + u, i.e. t + 2 (kt & 1), 32 bytes on.
        const int t = u >> 1, q = u & 1, g = rr & 7, h = rr >> 3;
        b[j] = w + static_cast<size_t>(tile) * chunks * kTileBytesFp8 + (((h * 2 + q) * 8 + g) * 4 + t) * 16;
      }
      sb[j] = S::kStageA + swz(r, u);
    }
    if constexpr (FP4) {
      if (tid < kBM) {
        sf_kind = 1;
        sf = p.xs + static_cast<size_t>(min(m0 + tid, p.M - 1)) * (p.K / 16);
        ssf = S::kStageA + S::kStageB + tid * 8;
      } else if (tid < kBM + BN / 2) {
        sf_kind = 2;
        const int r = (tid - kBM) * 2;
        int row;
        const uint8_t* w = weight_row<BN, SWIGLU>(p, n0, r, row);
        const int tile = row >> 4, rr = row & 15;
        sf = w + static_cast<size_t>(tile) * chunks * kTileBytesFp4 + 1024 + rr * 8;
        ssf = S::kStageA + S::kStageB + S::kStageSA + r * 8;
      }
    }
  }

  // Issues the copies of K stage `kt` (64 bytes per row) into shared-memory stage `slot`.
  __device__ __forceinline__ void load(uint8_t* smem, int slot, int kt) const {
    const uint32_t st = smem_addr(smem) + slot * S::kStageBytes;
    const size_t boff = FP4 ? static_cast<size_t>(kt) * kTileBytesFp4
                            : static_cast<size_t>(kt >> 1) * kTileBytesFp8 + (kt & 1) * 32;
#pragma unroll
    for (int j = 0; j < 2; ++j) cp16(st + sa[j], a[j] + kt * kRowBytes);
#pragma unroll
    for (int j = 0; j < kBPieces; ++j) cp16(st + sb[j], b[j] + boff);
    if constexpr (FP4) {
      if (sf_kind == 1) cp8(st + ssf, sf + kt * 8);
      else if (sf_kind == 2) cp16(st + ssf, sf + boff);
    }
  }
};

__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
// Two values to one byte: the first in the low nibble (the order the MMA reads and the checkpoint stores).
__device__ __forceinline__ uint32_t e2m1x2(float lo, float hi) {
  uint16_t out;
  asm("{\n .reg .b8 b;\n cvt.rn.satfinite.e2m1x2.f32 b, %1, %2;\n mov.b16 %0, {b, b};\n}" : "=h"(out) : "f"(hi), "f"(lo));
  return out & 0xffu;
}

// One pipeline stage's MMAs: the fragments of both 32-byte K steps, then 2 x 4 x NF MMAs.
template <bool FP4, int BN, int STAGES>
__device__ __forceinline__ void compute_stage(const uint8_t* st, float (&acc)[4][Shape<BN, STAGES>::kNF][4], int lane,
                                              int wm, int wn) {
  using S = Shape<BN, STAGES>;
  constexpr int NF = S::kNF, WN = BN / 4;
  const int lr = lane & 7, mat = lane >> 3;
  const uint32_t sA = smem_addr(st), sB = sA + S::kStageA;
  const uint8_t* sSA = st + S::kStageA + S::kStageB;
  const uint8_t* sSB = sSA + S::kStageSA;
  // Both MMA steps' fragments (32 bytes of K per row each) are loaded before the first MMA, so the
  // shared-memory loads of the second overlap the first's MMAs.
  uint32_t a[2][4][4], b[2][NF][2], sfa[2][4] = {}, sfb[2][NF] = {};
#pragma unroll
  for (int s = 0; s < 2; ++s) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = wm * 64 + 16 * i + lr + 8 * (mat & 1);
      ldsm_x4(a[s][i], sA + swz(r, 2 * s + (mat >> 1)));
      if constexpr (FP4)
        sfa[s][i] = *reinterpret_cast<const uint32_t*>(sSA + (wm * 64 + 16 * i + (lane >> 2) + 8 * (lane & 1)) * 8 + 4 * s);
    }
#pragma unroll
    for (int jp = 0; jp < NF / 2; ++jp) {
      const int r = wn * WN + 16 * jp + 8 * (mat >> 1) + lr;
      uint32_t t[4];
      ldsm_x4(t, sB + swz(r, 2 * s + (mat & 1)));
      b[s][2 * jp][0] = t[0];
      b[s][2 * jp][1] = t[1];
      b[s][2 * jp + 1][0] = t[2];
      b[s][2 * jp + 1][1] = t[3];
    }
    if constexpr (FP4) {
#pragma unroll
      for (int j = 0; j < NF; ++j)
        sfb[s][j] = *reinterpret_cast<const uint32_t*>(sSB + (wn * WN + 8 * j + (lane >> 2)) * 8 + 4 * s);
    }
  }
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < NF; ++j) mma<FP4>(acc[i][j], a[s][i], b[s][j][0], b[s][j][1], sfa[s][i], sfb[s][j]);
}

// The epilogue: alpha * acc into y (or added to it), or SwiGLU straight into NVFP4.
template <int BN, int STAGES, bool SWIGLU>
__device__ __forceinline__ void epilogue(const GemmArgs& p, const float (&acc)[4][Shape<BN, STAGES>::kNF][4], int m0, int n0,
                                         int lane, int wm, int wn) {
  constexpr int NF = Shape<BN, STAGES>::kNF, WN = BN / 4;
  const int g = lane >> 2, t = lane & 3;
  if constexpr (SWIGLU) {
    // Fragments j and j + NF / 2 are the gate and up products of the same columns. Columns 16b .. 16b + 15 of
    // the warp's outputs are fragments 2b and 2b + 1: each lane holds four of them (2t, 2t + 1, 8 + 2t,
    // 9 + 2t), and the lanes t = 0 .. 3 of a row form one NVFP4 block.
    const int oc = n0 / 2 + wn * (WN / 2);
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * 64 + 16 * i + g + 8 * h;
#pragma unroll
        for (int bb = 0; bb < NF / 4; ++bb) {
          float v[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int j = 2 * bb + (e >> 1);
            const float gv = acc[i][j][2 * h + (e & 1)] * p.alpha, uv = acc[i][j + NF / 2][2 * h + (e & 1)] * p.alpha2;
            v[e] = bf16_round(silu(gv) * uv);
          }
          float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
          amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
          const __nv_fp8_e4m3 sc(p.gscale_out * (amax * (1.f / 6.f)));
          const float sv = static_cast<float>(sc);
          const float os = sv != 0.f ? 1.f / (sv * (1.f / p.gscale_out)) : 0.f;
          if (row < p.M) {
            const int col = oc + 16 * bb;  // the block's first column
            uint8_t* qr = p.q_out + static_cast<size_t>(row) * (p.N / 2) + col / 2;
            qr[t] = static_cast<uint8_t>(e2m1x2(v[0] * os, v[1] * os));
            qr[4 + t] = static_cast<uint8_t>(e2m1x2(v[2] * os, v[3] * os));
            if (t == 0) p.sf_out[static_cast<size_t>(row) * (p.N / 16) + col / 16] = sc.__x;
          }
        }
      }
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * 64 + 16 * i + g + 8 * h;
        if (row >= p.M) continue;
        float* yr = p.y + static_cast<size_t>(row) * p.ldy + n0 + wn * WN + 2 * t;
#pragma unroll
        for (int j = 0; j < NF; ++j) {
          float2 v = make_float2(acc[i][j][2 * h] * p.alpha, acc[i][j][2 * h + 1] * p.alpha);
          float2* dst = reinterpret_cast<float2*>(yr + 8 * j);
          if (p.accumulate) {
            const float2 o = *dst;
            v.x += o.x;
            v.y += o.y;
          }
          *dst = v;
        }
      }
  }
}

template <bool FP4, int BN, int STAGES, bool SWIGLU = false>
__global__ void __launch_bounds__(kThreads) prefill_gemm_kernel(GemmArgs p) {
  using S = Shape<BN, STAGES>;
  constexpr int NF = S::kNF;
  extern __shared__ __align__(128) uint8_t smem[];
  const int m0 = blockIdx.x * kBM, n0 = blockIdx.y * BN;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp & 1, wn = warp >> 1;
  const int KT = p.K / kRowBytes / (FP4 ? 2 : 1);  // stages along K

  float acc[4][NF][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const Loader<FP4, BN, STAGES, SWIGLU> loader(p, m0, n0);
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) loader.load(smem, s, s);
    cp_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nk = kt + STAGES - 1;
    if (nk < KT) loader.load(smem, nk % STAGES, nk);
    cp_commit();
    compute_stage<FP4, BN, STAGES>(smem + (kt % STAGES) * S::kStageBytes, acc, lane, wm, wn);
  }
  cp_wait<0>();
  epilogue<BN, STAGES, SWIGLU>(p, acc, m0, n0, lane, wm, wn);
}

// ---- Activation quantization, as production's kernels do it (activations are BF16 there, so each value is
// rounded to BF16 first). ----


// The NVFP4 encoding of four values held by each of four adjacent lanes (one block of 16).
__device__ __forceinline__ void put_nvfp4(const float (&v)[4], float gscale, size_t quad, uint8_t* __restrict__ q,
                                          uint8_t* __restrict__ sf) {
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const __nv_fp8_e4m3 sc(gscale * (amax * (1.f / 6.f)));
  const float sv = static_cast<float>(sc);
  const float os = sv != 0.f ? 1.f / (sv * (1.f / gscale)) : 0.f;
  const uint32_t lo = e2m1x2(v[0] * os, v[1] * os), hi = e2m1x2(v[2] * os, v[3] * os);
  reinterpret_cast<uint16_t*>(q)[quad] = static_cast<uint16_t>(lo | (hi << 8));
  if ((quad & 3) == 0) sf[quad >> 2] = sc.__x;
}

__device__ __forceinline__ uint32_t put_fp8x4(const float (&v)[4], float inv_scale) {
  uint32_t w = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k)
    w |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[k] * inv_scale, __NV_SATFINITE, __NV_E4M3)) << (8 * k);
  return w;
}

// Four lanes per block of 16 values (each loads one float4, coalesced): scale = E4M3(gscale * amax / 6),
// values = E2M1(x * gscale / scale). With `up`, the value quantized is silu(x) * up (the FFN's gate and up
// products, fused so the 17,408-wide activation is never written in FP32).
template <bool SILU>
__global__ void quant_nvfp4_kernel(const float* __restrict__ x, const float* __restrict__ up, size_t quads, float gscale,
                                   uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads; i += stride) {
    float4 f = reinterpret_cast<const float4*>(x)[i];
    if constexpr (SILU) {
      const float4 u = reinterpret_cast<const float4*>(up)[i];
      f = make_float4(silu(f.x) * u.x, silu(f.y) * u.y, silu(f.z) * u.z, silu(f.w) * u.w);
    }
    const float v[4] = {bf16_round(f.x), bf16_round(f.y), bf16_round(f.z), bf16_round(f.w)};
    put_nvfp4(v, gscale, i, q, sf);
  }
}

// RMSNorm of each row (Gemma's x * rstd * (1 + w)) straight into the next GEMM's encoding, and into FP32 too
// when `out` is given. One block per row; each thread takes whole groups of four values, so four adjacent
// lanes hold one NVFP4 block. H % 1024 == 0.
template <bool FP4>
__global__ void __launch_bounds__(256)
    norm_quant_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int H,
                      float eps, float scale, uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  __shared__ float red[8];
  const size_t r = blockIdx.x;
  const float4* row = reinterpret_cast<const float4*>(x + r * H);
  float ss = 0.f;
  for (int i = threadIdx.x; i < H / 4; i += 256) {
    const float4 f = row[i];
    ss = fmaf(f.x, f.x, fmaf(f.y, f.y, fmaf(f.z, f.z, fmaf(f.w, f.w, ss))));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / H + eps);
  for (int i = threadIdx.x; i < H / 4; i += 256) {
    const float4 f = row[i];
    const float xs[4] = {f.x, f.y, f.z, f.w};
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = xs[k] * rstd * (1.f + __bfloat162float(w[4 * i + k]));
    if (out) reinterpret_cast<float4*>(out + r * H)[i] = make_float4(v[0], v[1], v[2], v[3]);
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = bf16_round(v[k]);
    const size_t quad = r * (H / 4) + i;
    if constexpr (FP4) put_nvfp4(v, scale, quad, q, sf);
    else reinterpret_cast<uint32_t*>(q)[quad] = put_fp8x4(v, scale);
  }
}

// The DeltaNet output norm (rmsnorm(x) * w * silu(z) per head of D = 128) straight into FP8 for the
// out-projection: one warp per head row, four values per lane.
__global__ void gated_norm_fp8_kernel(const float* __restrict__ x, const float* __restrict__ z,
                                      const __nv_bfloat16* __restrict__ w, size_t rows, float eps, float inv_scale,
                                      uint8_t* __restrict__ q) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x / 32) + threadIdx.x / 32;
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const float4 f = reinterpret_cast<const float4*>(x + r * 128)[lane];
  const float4 g = reinterpret_cast<const float4*>(z + r * 128)[lane];
  float ss = f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float rstd = rsqrtf(ss / 128 + eps);
  const float xs[4] = {f.x, f.y, f.z, f.w}, zs[4] = {g.x, g.y, g.z, g.w};
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = bf16_round(xs[k] * rstd * __bfloat162float(w[4 * lane + k]) * silu(zs[k]));
  reinterpret_cast<uint32_t*>(q)[r * 32 + lane] = put_fp8x4(v, inv_scale);
}

// x * sigmoid(gate) straight into FP8 (the attention output, for the o-projection).
__global__ void sigmoid_mul_fp8_kernel(const float* __restrict__ x, const float* __restrict__ gate, size_t quads,
                                       float inv_scale, uint8_t* __restrict__ q) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float4 f = reinterpret_cast<const float4*>(x)[i], g = reinterpret_cast<const float4*>(gate)[i];
    const float v[4] = {bf16_round(f.x / (1.f + __expf(-g.x))), bf16_round(f.y / (1.f + __expf(-g.y))),
                        bf16_round(f.z / (1.f + __expf(-g.z))), bf16_round(f.w / (1.f + __expf(-g.w)))};
    reinterpret_cast<uint32_t*>(q)[i] = put_fp8x4(v, inv_scale);
  }
}

// Four values per thread: E4M3(x / input_scale), saturated to +-448.
__global__ void quant_fp8_kernel(const float* __restrict__ x, size_t quads, float inv_scale, uint8_t* __restrict__ q) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float4 f = reinterpret_cast<const float4*>(x)[i];
    const float v[4] = {bf16_round(f.x) * inv_scale, bf16_round(f.y) * inv_scale, bf16_round(f.z) * inv_scale,
                        bf16_round(f.w) * inv_scale};
    uint32_t w = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      w |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[k], __NV_SATFINITE, __NV_E4M3)) << (8 * k);
    reinterpret_cast<uint32_t*>(q)[i] = w;
  }
}

// Narrow BF16 matrices (the DeltaNet a and b projections, 48 rows each) on tensor cores: a one-warp block
// takes 16 rows of x and a quarter of the 2N output columns (blockIdx.y), m16n8k16 MMAs over K in order. x is
// rounded to BF16 (production's activations). A row's result does not depend on M.
__global__ void __launch_bounds__(32)
    narrow_bf16_kernel(const float* __restrict__ x, int M, const __nv_bfloat16* __restrict__ wa,
                       const __nv_bfloat16* __restrict__ wb, float* __restrict__ ya, float* __restrict__ yb, int N, int K) {
  constexpr int NF = 3;  // n8 fragments per block: 4 column groups x 3 x 8 = 96 = 2 x 48 columns
  const int lane = threadIdx.x & 31, warp = blockIdx.y, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const float* xa = x + static_cast<size_t>(min(r0 + g, M - 1)) * K;
  const float* xb = x + static_cast<size_t>(min(r0 + g + 8, M - 1)) * K;
  const __nv_bfloat16* wrow[NF];
#pragma unroll
  for (int j = 0; j < NF; ++j) {
    const int col = (warp * NF + j) * 8 + g;  // this lane's B column
    wrow[j] = col < N ? wa + static_cast<size_t>(col) * K : wb + static_cast<size_t>(col - N) * K;
  }
  float acc[NF][4] = {};
  for (int k = 0; k < K; k += 16) {
    auto pack = [](float2 v) {
      const __nv_bfloat162 h = __floats2bfloat162_rn(v.x, v.y);
      return *reinterpret_cast<const uint32_t*>(&h);
    };
    const uint32_t a0 = pack(*reinterpret_cast<const float2*>(xa + k + 2 * t));
    const uint32_t a1 = pack(*reinterpret_cast<const float2*>(xb + k + 2 * t));
    const uint32_t a2 = pack(*reinterpret_cast<const float2*>(xa + k + 8 + 2 * t));
    const uint32_t a3 = pack(*reinterpret_cast<const float2*>(xb + k + 8 + 2 * t));
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(wrow[j] + k + 2 * t);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(wrow[j] + k + 8 + 2 * t);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
          "{%0, %1, %2, %3};"
          : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
#pragma unroll
  for (int j = 0; j < NF; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = r0 + g + 8 * h;
      if (row >= M) continue;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = (warp * NF + j) * 8 + 2 * t + e;
        if (col < N) ya[static_cast<size_t>(row) * N + col] = acc[j][2 * h + e];
        else yb[static_cast<size_t>(row) * N + col - N] = acc[j][2 * h + e];
      }
    }
}

int grid_for(size_t n) { return static_cast<int>(std::min<size_t>((n + 255) / 256, 48 * 64)); }

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <bool FP4, int BN, int STAGES, bool SWIGLU = false>
void launch_shape(const GemmArgs& p, cudaStream_t s) {
  using S = Shape<BN, STAGES>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(prefill_gemm_kernel<FP4, BN, STAGES, SWIGLU>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         S::kSmemBytes);
    configured = true;
  }
  const dim3 grid((p.M + kBM - 1) / kBM, (SWIGLU ? 2 * p.N : p.N) / BN);
  prefill_gemm_kernel<FP4, BN, STAGES, SWIGLU><<<grid, kThreads, S::kSmemBytes, s>>>(p);
}

// LING_GEMM_SHAPE (experiments): 0 automatic, 1: 128 x 128 x 4 stages, 2: 128 x 256 x 3 stages.
int gemm_shape() {
  static const int v = [] {
    const char* e = std::getenv("LING_GEMM_SHAPE");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}

template <bool FP4>
void launch_gemm(const GemmArgs& p, cudaStream_t s) {
  if (p.N % 128 != 0 || p.K % 128 != 0) throw std::runtime_error("prefill_gemm: N % 128 and K % 128 must be 0");
  if (p.M < 1) return;
  int shape = gemm_shape();
  // 128 x 256 tiles (measured +12-25% on the NVFP4 matrices at 512-2,048 rows) when they still give two waves.
  if (shape == 0) shape = (p.M + kBM - 1) / kBM * (p.N / 256) >= 96 ? 2 : 1;
  if (shape == 2 && p.N % 256 == 0) launch_shape<FP4, 256, 3>(p, s);
  else launch_shape<FP4, 128, 4>(p, s);
  check_launch(FP4 ? "prefill_gemm_nvfp4" : "prefill_gemm_fp8");
}

}  // namespace

void quant_act_nvfp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  quant_nvfp4_kernel<false><<<grid_for(quads), 256, 0, s>>>(x, nullptr, quads, 1.f / input_scale, q, sf);
  check_launch("quant_act_nvfp4");
}

void silu_mul_quant_nvfp4(const float* g, const float* u, int M, int K, float input_scale, uint8_t* q, uint8_t* sf,
                          cudaStream_t s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  quant_nvfp4_kernel<true><<<grid_for(quads), 256, 0, s>>>(g, u, quads, 1.f / input_scale, q, sf);
  check_launch("silu_mul_quant_nvfp4");
}

void quant_act_fp8(const float* x, int M, int K, float input_scale, uint8_t* q, cudaStream_t s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  quant_fp8_kernel<<<grid_for(quads), 256, 0, s>>>(x, quads, 1.f / input_scale, q);
  check_launch("quant_act_fp8");
}

void rmsnorm_quant(const float* x, const __nv_bfloat16* w, float* out, int M, int H, float eps, bool nvfp4,
                   float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s) {
  if (H % 1024 != 0) throw std::runtime_error("rmsnorm_quant: H % 1024 must be 0");
  if (nvfp4) norm_quant_kernel<true><<<M, 256, 0, s>>>(x, w, out, H, eps, 1.f / input_scale, q, sf);
  else norm_quant_kernel<false><<<M, 256, 0, s>>>(x, w, out, H, eps, 1.f / input_scale, q, nullptr);
  check_launch("rmsnorm_quant");
}

void gated_rmsnorm_fp8(const float* x, const float* z, const __nv_bfloat16* w, int rows, int D, float eps,
                       float input_scale, uint8_t* q, cudaStream_t s) {
  if (D != 128) throw std::runtime_error("gated_rmsnorm_fp8: D must be 128");
  gated_norm_fp8_kernel<<<(rows + 7) / 8, 256, 0, s>>>(x, z, w, rows, eps, 1.f / input_scale, q);
  check_launch("gated_rmsnorm_fp8");
}

void sigmoid_mul_fp8(const float* x, const float* gate, size_t n, float input_scale, uint8_t* q, cudaStream_t s) {
  sigmoid_mul_fp8_kernel<<<grid_for(n / 4), 256, 0, s>>>(x, gate, n / 4, 1.f / input_scale, q);
  check_launch("sigmoid_mul_fp8");
}

void prefill_gemm_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, float alpha, float* y, int ldy,
                        int N, int K, bool accumulate, cudaStream_t s) {
  launch_gemm<true>(GemmArgs{xq, xs, w, alpha, y, ldy, M, N, K, accumulate}, s);
}

void narrow_bf16(const float* x, int M, const __nv_bfloat16* wa, const __nv_bfloat16* wb, float* ya, float* yb, int N,
                 int K, cudaStream_t s) {
  if (2 * N != 96 || K % 16 != 0) throw std::runtime_error("narrow_bf16: built for two 48-row matrices, K % 16 == 0");
  narrow_bf16_kernel<<<dim3((M + 15) / 16, 4), 32, 0, s>>>(x, M, wa, wb, ya, yb, N, K);
  check_launch("narrow_bf16");
}

void prefill_swiglu_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* gate, float alpha_gate,
                          const uint8_t* up, float alpha_up, int N, int K, float out_input_scale, uint8_t* q_out,
                          uint8_t* sf_out, cudaStream_t s) {
  if (N % 128 != 0 || K % 128 != 0) throw std::runtime_error("prefill_swiglu_nvfp4: N % 128 and K % 128 must be 0");
  if (M < 1) return;
  GemmArgs p{xq, xs, gate, alpha_gate, nullptr, 0, M, N, K, false};
  p.w2 = up;
  p.alpha2 = alpha_up;
  p.gscale_out = 1.f / out_input_scale;
  p.q_out = q_out;
  p.sf_out = sf_out;
  if ((M + kBM - 1) / kBM * (2 * N / 256) >= 96) launch_shape<true, 256, 3, true>(p, s);
  else launch_shape<true, 128, 4, true>(p, s);
  check_launch("prefill_swiglu_nvfp4");
}

void prefill_gemm_fp8(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                      bool accumulate, cudaStream_t s) {
  launch_gemm<false>(GemmArgs{xq, nullptr, w, alpha, y, ldy, M, N, K, accumulate}, s);
}

}  // namespace ling::kernels
