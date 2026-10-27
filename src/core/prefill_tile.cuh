// The prefill GEMM's block tile (prefill_gemm.cu), as device code the prefill kernels and the fused verify +
// prefill kernel (fused_gemm.cu) share: the shape, the stage loader, one stage's MMAs and the epilogue. The
// numerics and the tiling are described in prefill_gemm.cu; this header only makes them reusable, so a kernel
// that includes it computes the prefill path's bits by construction.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ling::kernels::pf {
namespace {

constexpr int kBM = 128, kThreads = 256;
constexpr int kRowBytes = 64;  // per row per stage
constexpr int kTileBytesFp4 = 16 * 64 + 16 * 8, kTileBytesFp8 = 16 * 128;
constexpr int kValueBytesFp4 = 16 * 64;  // B26: a coded blob's value blocks (its codes follow them all, stream_gemm.cu)

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
  // BF16 output (the OUT_BF16 instantiations, --prefill-bf16): y_bf16 [M][ldy] instead of y, never accumulated.
  // Last, so the fields above keep their offsets (the FP32 kernels' SASS is unchanged).
  __nv_bfloat16* y_bf16 = nullptr;
  // B26: a coded blob's raw scales (128 bytes per (tile, chunk) block), for w and w2; nullptr: the blob carries them.
  const uint8_t* wsf = nullptr;
  const uint8_t* wsf2 = nullptr;
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
  uint32_t bstride[kBPieces];  // NVFP4: each piece's block bytes (a coded blob's or a raw one's, B26)
  uint32_t sfstride = 0;       // the weight scales' bytes per K block: in the blob, or 128 in a separate copy
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
        bstride[j] = (w == p.w ? p.wsf : p.wsf2) ? kValueBytesFp4 : kTileBytesFp4;
        b[j] = w + static_cast<size_t>(tile) * chunks * bstride[j] + (rr * 4 + u) * 16;
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
        const uint8_t* raw = w == p.w ? p.wsf : p.wsf2;
        sfstride = raw ? 128 : kTileBytesFp4;
        sf = raw ? raw + static_cast<size_t>(tile) * chunks * 128 + rr * 8
                 : w + static_cast<size_t>(tile) * chunks * kTileBytesFp4 + 1024 + rr * 8;
        ssf = S::kStageA + S::kStageB + S::kStageSA + r * 8;
      }
    }
  }

  // Issues the copies of K stage `kt` (64 bytes per row) into shared-memory stage `slot`.
  __device__ __forceinline__ void load(uint8_t* smem, int slot, int kt) const {
    const uint32_t st = smem_addr(smem) + slot * S::kStageBytes;
    const size_t boff = FP4 ? 0 : static_cast<size_t>(kt >> 1) * kTileBytesFp8 + (kt & 1) * 32;
#pragma unroll
    for (int j = 0; j < 2; ++j) cp16(st + sa[j], a[j] + kt * kRowBytes);
#pragma unroll
    for (int j = 0; j < kBPieces; ++j) cp16(st + sb[j], b[j] + (FP4 ? static_cast<size_t>(kt) * bstride[j] : boff));
    if constexpr (FP4) {
      if (sf_kind == 1) cp8(st + ssf, sf + kt * 8);
      else if (sf_kind == 2) cp16(st + ssf, sf + static_cast<size_t>(kt) * sfstride);
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

// The epilogue: alpha * acc into y (or added to it), or SwiGLU straight into NVFP4, or (OUT_BF16, the
// --prefill-bf16 instantiations) alpha * acc rounded to BF16 into y_bf16, as production's FP8 GEMMs write their
// BF16 output.
template <int BN, int STAGES, bool SWIGLU, bool OUT_BF16 = false>
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
  } else if constexpr (OUT_BF16) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int row = m0 + wm * 64 + 16 * i + g + 8 * h;
        if (row >= p.M) continue;
        __nv_bfloat16* yr = p.y_bf16 + static_cast<size_t>(row) * p.ldy + n0 + wn * WN + 2 * t;
#pragma unroll
        for (int j = 0; j < NF; ++j)
          *reinterpret_cast<__nv_bfloat162*>(yr + 8 * j) =
              __floats2bfloat162_rn(acc[i][j][2 * h] * p.alpha, acc[i][j][2 * h + 1] * p.alpha);
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

}  // namespace
}  // namespace ling::kernels::pf
