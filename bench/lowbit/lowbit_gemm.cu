// Weight-streaming GEMMs at the verify shapes in five weight formats, on dummy data (the low-bit study,
// specs/DREAMFERENCE_LING_ENGINE_LOWBIT.md section 6). No model is read and no engine code is linked.
//
// The question: once a format stores fewer bytes per weight, does a 16-row verify GEMM still run at the
// bus rate, so that its time falls with its bytes, or does decoding the format in registers (item 30's
// trap, BACKLOG section 19) or the activations' traffic eat the saving? Every kernel has the shape of the
// engine's stream_gemm (src/core/stream_gemm.cu): a block is 8 warps over TILES x 16 weight rows, warp w
// takes 128-value K chunks w, w + 8, ..., a lane holds weight rows g and g + 8, the weights are decoded to
// FP16 in registers and multiplied on the FP16 tensor cores with FP32 accumulation, the activations are
// FP16 and read from L2 per chunk, and the warps' partial sums are added in a fixed order at the end.
//
//   fp4  NVFP4: E2M1 codes, one E4M3 scale per 16 values               4.5 bits a weight (1,152 B a block)
//   fp8  FP8 E4M3, per-tensor scale                                     8.0                (2,048 B)
//   b3   3-bit codes into a per-tensor 8-entry E4M3 codebook, one E4M3 scale per 16 values, stored as a
//        2-bit plane and a 1-bit plane and decoded with masks and one prmt (a byte permute) per 4 values
//                                                                       3.5                (896 B)
//   sp4  2:4 sparse NVFP4: two E2M1 codes kept of every four, 2-bit indices (the sparse MMA's metadata),
//        one E4M3 scale per 16 logical values, mma.sp m16n8k32 FP16     3.5                (896 B)
//   sp8  2:4 sparse FP8 with metadata, mma.sp m16n8k32 FP16             5.0                (1,280 B)
//
// A naive one-thread-per-output reference decodes every format from its documented layout; each tiled
// kernel must match it (relative error below 1e-3, FP32 sums in another order) before it is timed.
//
// Build: nvcc -O3 -std=c++20 -gencode arch=compute_121a,code=sm_121a lowbit_gemm.cu -o lowbit_gemm
// Run:   lowbit_gemm [reps=20]   (prints one CSV line per shape, format, rows and tiles)
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                   \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));        \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

enum Fmt { FP4 = 0, FP8 = 1, B3 = 2, SP4 = 3, SP8 = 4 };
static const char* kFmtName[] = {"fp4", "fp8", "b3", "sp4", "sp8"};
// Bytes of one (16-row tile, 128-value chunk) block.
__host__ __device__ constexpr int blk_bytes(int f) {
  return f == FP4 ? 1152 : f == FP8 ? 2048 : f == B3 ? 896 : f == SP4 ? 896 : 1280;
}
// 32-bit words a lane loads per (tile, chunk).
__host__ __device__ constexpr int lane_words(int f) {
  return f == FP4 ? 9 : f == FP8 ? 16 : f == B3 ? 7 : f == SP4 ? 10 : 10;
}

__constant__ uint32_t c_tab[2];  // the b3 codebook: eight E4M3 bytes

// ---- device helpers ------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t e2m1x2_to_h2(uint32_t byte) {
  uint32_t out;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b0;\n}"
      : "=r"(out)
      : "r"(byte));
  return out;
}
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t pair) {
  uint32_t out;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(out) : "r"(pair));
  return out;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}
// An E4M3 scale byte as f16x2 (both halves).
__device__ __forceinline__ uint32_t scale_h2(uint32_t byte) {
  uint32_t h = e4m3x2_to_h2(byte & 0xffu);
  return (h & 0xffffu) | (h << 16);
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t out;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(sel));
  return out;
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <int SEL>
__device__ __forceinline__ void mma_sp16832(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t e) {
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9, %10, %11}, {%0, %1, %2, %3}, %12, %13;"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(e), "n"(SEL));
}
__device__ __forceinline__ uint4 ldw4(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 ldw2(const uint8_t* p) {
  uint2 v;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t ldw1(const uint8_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

// One lane's words of one (tile, chunk) block, in the order the decode below reads them.
template <int F>
__device__ __forceinline__ void load_block(const uint8_t* blk, int lane, uint32_t (&w)[lane_words(F)]) {
  const int g = lane >> 2;
  // Each 16-byte piece is stored piece-major (piece q of lane L at q * 512 + L * 16), so one warp load
  // covers 512 contiguous bytes.
  if constexpr (F == FP4) {  // values: row g at L*16, row g+8 at 512 + L*16; scales 1024 + L*4
    uint4 a = ldw4(blk + lane * 16), b = ldw4(blk + 512 + lane * 16);
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    w[8] = ldw1(blk + 1024 + lane * 4);
  } else if constexpr (F == FP8) {  // pieces q = 0, 1 row g, 2, 3 row g+8, at q*512 + L*16
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      uint4 a = ldw4(blk + q * 512 + lane * 16);
      w[4 * q] = a.x; w[4 * q + 1] = a.y; w[4 * q + 2] = a.z; w[4 * q + 3] = a.w;
    }
  } else if constexpr (F == B3) {  // 2-bit plane L*16; 1-bit plane 512 + L*8; scales 768 + L*4
    uint4 a = ldw4(blk + lane * 16);
    uint2 h = ldw2(blk + 512 + lane * 8);
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = h.x; w[5] = h.y;
    w[6] = ldw1(blk + 768 + lane * 4);
  } else if constexpr (F == SP4) {  // kept codes L*16; metadata 512 + L*8; scales 768 + row*8 (rows g, g+8)
    uint4 a = ldw4(blk + lane * 16);
    uint2 m = ldw2(blk + 512 + lane * 8);
    uint2 s0 = ldw2(blk + 768 + g * 8), s1 = ldw2(blk + 768 + (g + 8) * 8);
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = m.x; w[5] = m.y;
    w[6] = s0.x; w[7] = s0.y; w[8] = s1.x; w[9] = s1.y;
  } else {  // SP8: kept values L*16 and 512 + L*16; metadata 1024 + L*8
    uint4 a = ldw4(blk + lane * 16), b = ldw4(blk + 512 + lane * 16);
    uint2 m = ldw2(blk + 1024 + lane * 8);
    w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = b.x; w[5] = b.y; w[6] = b.z; w[7] = b.w;
    w[8] = m.x; w[9] = m.y;
  }
}

// Multiply one decoded (tile, chunk) block by the chunk's activations. xw[j][q] is the q-th 16-byte piece
// of token g + 8j's activations in this chunk (dense formats: K 32t + 8q ..; sparse: 32q + 8t ..).
template <int F, int NT>
__device__ __forceinline__ void mac_block(const uint32_t (&w)[lane_words(F)], const uint4 (&xw)[NT][4],
                                          float (&acc)[NT][4], int t) {
  if constexpr (F == FP4 || F == FP8 || F == B3) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {  // k16 steps; the lane's values of step s are K 32t + 4s + {0..3}
      uint32_t a0, a1, a2, a3;
      if constexpr (F == FP4) {
        const uint32_t sg = scale_h2(w[8] >> (8 * (s >> 2))), sh = scale_h2(w[8] >> (16 + 8 * (s >> 2)));
        const uint32_t vg = w[s >> 1] >> (16 * (s & 1)), vh = w[4 + (s >> 1)] >> (16 * (s & 1));
        a0 = hmul2(e2m1x2_to_h2(vg & 0xffu), sg);
        a2 = hmul2(e2m1x2_to_h2((vg >> 8) & 0xffu), sg);
        a1 = hmul2(e2m1x2_to_h2(vh & 0xffu), sh);
        a3 = hmul2(e2m1x2_to_h2((vh >> 8) & 0xffu), sh);
      } else if constexpr (F == FP8) {
        a0 = e4m3x2_to_h2(w[s]);
        a2 = e4m3x2_to_h2(w[s] >> 16);
        a1 = e4m3x2_to_h2(w[8 + s]);
        a3 = e4m3x2_to_h2(w[8 + s] >> 16);
      } else {
        const uint32_t sg = scale_h2(w[6] >> (8 * (s >> 2))), sh = scale_h2(w[6] >> (16 + 8 * (s >> 2)));
        const uint32_t lo = w[s >> 1], hi = w[4 + (s >> 2)];
        uint32_t sel = ((s & 1) ? (lo >> 2) : lo) & 0x33333333u;
        sel |= ((hi >> (s & 3)) & 0x11111111u) << 2;
        const uint32_t pg = prmt(c_tab[0], c_tab[1], sel & 0xffffu), ph = prmt(c_tab[0], c_tab[1], sel >> 16);
        a0 = hmul2(e4m3x2_to_h2(pg), sg);
        a2 = hmul2(e4m3x2_to_h2(pg >> 16), sg);
        a1 = hmul2(e4m3x2_to_h2(ph), sh);
        a3 = hmul2(e4m3x2_to_h2(ph >> 16), sh);
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint4& p = xw[j][s >> 1];
        const uint32_t b0 = (s & 1) ? p.z : p.x, b1 = (s & 1) ? p.w : p.y;
        mma16816(acc[j], a0, a1, a2, a3, b0, b1);
      }
    }
  } else {
#pragma unroll
    for (int s = 0; s < 4; ++s) {  // k32 steps over 32 logical values each
      uint32_t a0, a1, a2, a3;
      if constexpr (F == SP4) {
        const uint32_t v = w[s];
        const uint32_t sg0 = scale_h2(w[6 + (s >> 1)] >> (16 * (s & 1)));
        const uint32_t sg1 = scale_h2(w[6 + (s >> 1)] >> (16 * (s & 1) + 8));
        const uint32_t sh0 = scale_h2(w[8 + (s >> 1)] >> (16 * (s & 1)));
        const uint32_t sh1 = scale_h2(w[8 + (s >> 1)] >> (16 * (s & 1) + 8));
        a0 = hmul2(e2m1x2_to_h2(v & 0xffu), sg0);
        a1 = hmul2(e2m1x2_to_h2((v >> 8) & 0xffu), sh0);
        a2 = hmul2(e2m1x2_to_h2((v >> 16) & 0xffu), sg1);
        a3 = hmul2(e2m1x2_to_h2(v >> 24), sh1);
      } else {
        const uint32_t v0 = w[2 * s], v1 = w[2 * s + 1];
        a0 = e4m3x2_to_h2(v0);
        a1 = e4m3x2_to_h2(v0 >> 16);
        a2 = e4m3x2_to_h2(v1);
        a3 = e4m3x2_to_h2(v1 >> 16);
      }
      const uint32_t e = (F == SP4) ? w[4 + (s >> 1)] : w[8 + (s >> 1)];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint4& p = xw[j][s];
        if (s & 1)
          mma_sp16832<1>(acc[j], a0, a1, a2, a3, p.x, p.y, p.z, p.w, e);
        else
          mma_sp16832<0>(acc[j], a0, a1, a2, a3, p.x, p.y, p.z, p.w, e);
      }
    }
  }
  (void)t;
}

template <int F, int NT>
__device__ __forceinline__ void load_x(const __half* x, int K, int chunk, int g, int t, uint4 (&xw)[NT][4]) {
  constexpr bool sparse = F == SP4 || F == SP8;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const __half* row = x + (size_t)(g + 8 * j) * K + chunk * 128;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int off = sparse ? 32 * q + 8 * t : 32 * t + 8 * q;
      xw[j][q] = __ldg(reinterpret_cast<const uint4*>(row + off));
    }
  }
}

// y[M][N] (FP32) = x[NT*8][K] . W^T, W as [N/16 tiles][K/128 chunks] blocks of format F.
template <int F, int TILES, int NT>
__global__ void __launch_bounds__(256) stream_kernel(const uint8_t* __restrict__ W, const __half* __restrict__ x,
                                                     float* __restrict__ y, int N, int K) {
  extern __shared__ float red[];  // [8 warps][TILES * 16 rows][NT * 8 tokens]
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int nch = K / 128, tile0 = blockIdx.x * TILES;
  float acc[TILES][NT][4];
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  uint32_t cur[TILES][lane_words(F)], nxt[TILES][lane_words(F)];
  uint4 xc[NT][4], xn[NT][4];
  int c = warp;
  if (c < nch) {
#pragma unroll
    for (int i = 0; i < TILES; ++i) load_block<F>(W + ((size_t)(tile0 + i) * nch + c) * blk_bytes(F), lane, cur[i]);
    load_x<F, NT>(x, K, c, g, t, xc);
  }
  for (; c < nch; c += 8) {
    const int cn = c + 8;
    if (cn < nch) {
#pragma unroll
      for (int i = 0; i < TILES; ++i)
        load_block<F>(W + ((size_t)(tile0 + i) * nch + cn) * blk_bytes(F), lane, nxt[i]);
      load_x<F, NT>(x, K, cn, g, t, xn);
    }
#pragma unroll
    for (int i = 0; i < TILES; ++i) mac_block<F, NT>(cur[i], xc, acc[i], t);
#pragma unroll
    for (int i = 0; i < TILES; ++i)
#pragma unroll
      for (int q = 0; q < lane_words(F); ++q) cur[i][q] = nxt[i][q];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) xc[j][q] = xn[j][q];
  }
  // c0, c1: weight row g, tokens 2t, 2t+1; c2, c3: row g + 8.
  constexpr int R = TILES * 16, M = NT * 8;
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      float* base = red + (size_t)warp * R * M;
      base[(i * 16 + g) * M + j * 8 + 2 * t] = acc[i][j][0];
      base[(i * 16 + g) * M + j * 8 + 2 * t + 1] = acc[i][j][1];
      base[(i * 16 + g + 8) * M + j * 8 + 2 * t] = acc[i][j][2];
      base[(i * 16 + g + 8) * M + j * 8 + 2 * t + 1] = acc[i][j][3];
    }
  __syncthreads();
  for (int e = threadIdx.x; e < R * M; e += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[(size_t)w * R * M + e];
    const int r = e / M, m = e % M, n = tile0 * 16 + r;
    if (n < N) y[(size_t)m * N + n] = s;
  }
}

// ---- the reference: one weight decoded from the documented layout -------------------------------------
__device__ float e2m1_val(uint32_t n) {
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (n & 8) ? -mag[n & 7] : mag[n & 7];
}
__device__ float e4m3_val(uint32_t b) {
  __nv_fp8_e4m3 v;
  v.__x = (uint8_t)b;
  return (float)v;
}
__device__ float weight_at(int F, const uint8_t* W, int nch, int n, int k) {
  const int tile = n / 16, r = n % 16, g = r % 8, half = r / 8, chunk = k / 128, kk = k % 128;
  const uint8_t* blk = W + ((size_t)tile * nch + chunk) * blk_bytes(F);
  if (F == FP4 || F == FP8 || F == B3) {
    const int t = kk / 32, within = kk % 32, s = within / 4, e = within % 4, L = g * 4 + t;
    if (F == FP4) {
      const uint32_t byte = blk[half * 512 + L * 16 + within / 2];
      const uint32_t nib = (within & 1) ? byte >> 4 : byte & 15u;
      return e2m1_val(nib) * e4m3_val(blk[1024 + L * 4 + half * 2 + within / 16]);
    }
    if (F == FP8) return e4m3_val(blk[(half * 2 + within / 16) * 512 + L * 16 + within % 16]);
    const int i = half * 4 + e;
    uint32_t lo, hi;
    memcpy(&lo, blk + L * 16 + (s >> 1) * 4, 4);
    memcpy(&hi, blk + 512 + L * 8 + (s >> 2) * 4, 4);
    const uint32_t code = ((lo >> (4 * i + 2 * (s & 1))) & 3u) | (((hi >> (4 * i + (s & 3))) & 1u) << 2);
    const uint32_t entry = (c_tab[code >> 2] >> (8 * (code & 3))) & 0xffu;
    return e4m3_val(entry) * e4m3_val(blk[768 + L * 4 + half * 2 + within / 16]);
  }
  // Sparse: physical p = kk % 32 of step s is held by lane t = p / 8 as logical 2t + 8j + e (p % 8 = 2j + e).
  const int s = kk / 32, p = kk % 32, tB = p / 8, j = (p % 8) / 2, e = p % 2;
  const int logical = 2 * tB + 8 * j + e, group = logical / 4, pos = logical % 4;
  const int tA = group % 4, hiGroup = group / 4;  // compressed columns 2tA, 2tA+1 (a0/a1) or 2tA+8.. (a2/a3)
  // Metadata (hypothesis H1 of the PTX layout): with selector 0 lanes t = 0, 1 supply it, with 1 lanes 2, 3;
  // the first of the pair holds groups 0-3, the second groups 4-7; bits 0-15 row g, 16-31 row g + 8.
  const int tt = (s & 1) * 2 + hiGroup, Lm = g * 4 + tt;
  uint32_t m;
  memcpy(&m, blk + (F == SP4 ? 512 : 1024) + Lm * 8 + (s >> 1) * 4, 4);
  const uint32_t bits = (m >> (half * 16 + 4 * tA)) & 15u, i0 = bits & 3u, i1 = bits >> 2;
  int elem = -1;
  if ((uint32_t)pos == i0) elem = 0;
  else if ((uint32_t)pos == i1) elem = 1;
  if (elem < 0) return 0.f;
  const int L = g * 4 + tA;
  if (F == SP4) {
    const uint32_t byte = blk[L * 16 + 4 * s + hiGroup * 2 + half];
    const uint32_t nib = elem ? byte >> 4 : byte & 15u;
    return e2m1_val(nib) * e4m3_val(blk[768 + r * 8 + 2 * s + hiGroup]);
  }
  const int b = 8 * s + hiGroup * 4 + half * 2 + elem;  // byte of the lane's 32
  return e4m3_val(blk[(b / 16) * 512 + L * 16 + b % 16]);
}
__global__ void reference_kernel(int F, const uint8_t* W, const __half* x, float* y, int N, int K, int M) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x, m = blockIdx.y;
  if (n >= N || m >= M) return;
  float s = 0.f;
  for (int k = 0; k < K; ++k) s += weight_at(F, W, K / 128, n, k) * __half2float(x[(size_t)m * K + k]);
  y[(size_t)m * N + n] = s;
}

__global__ void read_kernel(const uint4* __restrict__ p, size_t n, uint32_t* out) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p + i));
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) *out = acc;
}

// ---- host ---------------------------------------------------------------------------------------------
// Random but valid data: E2M1 / E4M3 codes kept finite and small, sparse metadata from the six legal
// patterns (i0 < i1), scales near 1/8.
static void fill(int F, std::vector<uint8_t>& h, size_t nblocks, std::mt19937& rng) {
  const size_t B = blk_bytes(F);
  h.resize(nblocks * B);
  std::uniform_int_distribution<int> byte(0, 255), pat(0, 5), sc(0x28, 0x34);
  const uint8_t legal[6] = {0x4, 0x8, 0xC, 0x9, 0xD, 0xE};  // (i0, i1) = (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
  auto fp8 = [&]() {  // E4M3 with exponent field kept in 4..10: magnitudes 2^-3 .. 2^3
    int v = byte(rng);
    int ex = 4 + (v % 7);
    return (uint8_t)((v & 0x80) | (ex << 3) | (byte(rng) & 7));
  };
  for (size_t b = 0; b < nblocks; ++b) {
    uint8_t* p = h.data() + b * B;
    for (size_t i = 0; i < B; ++i) p[i] = (uint8_t)byte(rng);
    if (F == FP4) for (int i = 1024; i < 1152; ++i) p[i] = (uint8_t)sc(rng);
    if (F == FP8) for (int i = 0; i < 2048; ++i) p[i] = fp8();
    if (F == B3) for (int i = 768; i < 896; ++i) p[i] = (uint8_t)sc(rng);
    if (F == SP4 || F == SP8) {
      const int m0 = F == SP4 ? 512 : 1024;
      for (int i = m0; i < m0 + 256; ++i) p[i] = (uint8_t)(legal[pat(rng)] | (legal[pat(rng)] << 4));
      if (F == SP4) for (int i = 768; i < 896; ++i) p[i] = (uint8_t)sc(rng);
      if (F == SP8) for (int i = 0; i < 1024; ++i) p[i] = fp8();
    }
  }
}

using KernelFn = void (*)(const uint8_t*, const __half*, float*, int, int);
template <int F, int TILES, int NT>
static KernelFn get_kernel(size_t& smem) {
  smem = (size_t)8 * TILES * 16 * NT * 8 * sizeof(float);
  CK(cudaFuncSetAttribute(stream_kernel<F, TILES, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  return stream_kernel<F, TILES, NT>;
}
template <int F>
static KernelFn pick(int tiles, int nt, size_t& smem) {
#define P(T, N) if (tiles == T && nt == N) return get_kernel<F, T, N>(smem);
  P(1, 1) P(1, 2) P(1, 4) P(2, 1) P(2, 2) P(2, 4) P(4, 1) P(4, 2) P(4, 4)
#undef P
  return nullptr;
}
static KernelFn pick_any(int F, int tiles, int nt, size_t& smem) {
  switch (F) {
    case FP4: return pick<FP4>(tiles, nt, smem);
    case FP8: return pick<FP8>(tiles, nt, smem);
    case B3: return pick<B3>(tiles, nt, smem);
    case SP4: return pick<SP4>(tiles, nt, smem);
    default: return pick<SP8>(tiles, nt, smem);
  }
}

struct Shape { const char* name; int N, K; };

static bool validate(int F, int tiles, int nt, std::mt19937& rng) {
  const int N = 1024, K = 2048, M = nt * 8;
  std::vector<uint8_t> h;
  fill(F, h, (size_t)(N / 16) * (K / 128), rng);
  std::vector<__half> hx((size_t)M * K);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (auto& v : hx) v = __float2half(u(rng));
  uint8_t* dW; __half* dx; float *dy, *dr;
  CK(cudaMalloc(&dW, h.size())); CK(cudaMalloc(&dx, hx.size() * 2));
  CK(cudaMalloc(&dy, (size_t)M * N * 4)); CK(cudaMalloc(&dr, (size_t)M * N * 4));
  CK(cudaMemcpy(dW, h.data(), h.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  size_t smem;
  KernelFn k = pick_any(F, tiles, nt, smem);
  k<<<N / 16 / tiles, 256, smem>>>(dW, dx, dy, N, K);
  reference_kernel<<<dim3(N / 128, M), 128>>>(F, dW, dx, dr, N, K, M);
  CK(cudaDeviceSynchronize());
  std::vector<float> a((size_t)M * N), b((size_t)M * N);
  CK(cudaMemcpy(a.data(), dy, a.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.data(), dr, b.size() * 4, cudaMemcpyDeviceToHost));
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (double)(a[i] - b[i]); den += (double)b[i] * b[i]; }
  const double rel = std::sqrt(num / (den + 1e-30));
  printf("# validate %s tiles %d rows %d: relative error %.2e %s\n", kFmtName[F], tiles, M, rel, rel < 1e-3 ? "ok" : "FAIL");
  cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dr);
  return rel < 1e-3;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 20;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("# %s sm_%d%d, %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  const uint8_t tab[8] = {0xB4, 0xAC, 0xA0, 0x98, 0x18, 0x20, 0x2C, 0x34};  // a symmetric 8-level E4M3 grid
  uint32_t ht[2];
  memcpy(ht, tab, 8);
  CK(cudaMemcpyToSymbol(c_tab, ht, 8));
  std::mt19937 rng(2027);

  bool ok = true;
  for (int F = 0; F < 5; ++F)
    for (int tiles : {1, 2, 4})
      for (int nt : {1, 2, 4}) ok &= validate(F, tiles, nt, rng);
  if (!ok) { printf("# validation failed: no timings\n"); return 1; }

  // The in-order read rate of a 4 GB buffer, the ceiling every kernel is compared with.
  {
    const size_t bytes = (size_t)4 << 30;
    uint4* p; uint32_t* o;
    CK(cudaMalloc(&p, bytes)); CK(cudaMalloc(&o, 4)); CK(cudaMemset(p, 1, bytes));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    read_kernel<<<prop.multiProcessorCount * 8, 512>>>(p, bytes / 16, o);
    cudaEventRecord(a);
    for (int r = 0; r < 5; ++r) read_kernel<<<prop.multiProcessorCount * 8, 512>>>(p, bytes / 16, o);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("# in-order read: %.1f GB/s\n", 5.0 * bytes / (ms * 1e6));
    cudaFree(p); cudaFree(o);
  }

  const Shape shapes[] = {{"ffn_gate_up", 34816, 5120}, {"ffn_down", 5120, 17408}, {"gdn_in", 16384, 5120},
                          {"gdn_out", 5120, 6144},       {"attn_q", 12288, 5120},  {"attn_kv", 2048, 5120},
                          {"attn_o", 5120, 6144},        {"lm_head", 248320, 5120}};
  printf("shape,N,K,fmt,rows,tiles,bytes,ms,gbps\n");
  __half* dx;
  float* dy;
  CK(cudaMalloc(&dx, (size_t)32 * 17408 * 2));
  CK(cudaMemset(dx, 0, (size_t)32 * 17408 * 2));
  CK(cudaMalloc(&dy, (size_t)32 * 248320 * 4));
  for (const Shape& sh : shapes) {
    for (int F = 0; F < 5; ++F) {
      const size_t nblocks = (size_t)(sh.N / 16) * (sh.K / 128), bytes = nblocks * blk_bytes(F);
      // Enough copies to defeat the 24 MB L2: at least 256 MB of weights cycled.
      const int copies = (int)std::max<size_t>(1, ((size_t)256 << 20) / bytes + 1);
      std::vector<uint8_t*> dW(copies);
      std::vector<uint8_t> h;
      fill(F, h, nblocks, rng);
      for (auto& p : dW) { CK(cudaMalloc(&p, bytes)); CK(cudaMemcpy(p, h.data(), bytes, cudaMemcpyHostToDevice)); }
      for (int rows : {1, 16, 32}) {
        const int nt = rows <= 8 ? 1 : rows <= 16 ? 2 : 4;
        for (int tiles : {1, 2, 4}) {
          if ((sh.N / 16) % tiles) continue;
          size_t smem;
          KernelFn k = pick_any(F, tiles, nt, smem);
          const int grid = sh.N / 16 / tiles;
          for (int c = 0; c < copies; ++c) k<<<grid, 256, smem>>>(dW[c], dx, dy, sh.N, sh.K);
          cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
          cudaEventRecord(a);
          for (int r = 0; r < reps; ++r)
            for (int c = 0; c < copies; ++c) k<<<grid, 256, smem>>>(dW[c], dx, dy, sh.N, sh.K);
          cudaEventRecord(b);
          CK(cudaEventSynchronize(b));
          CK(cudaGetLastError());
          float ms; cudaEventElapsedTime(&ms, a, b);
          ms /= (float)(reps * copies);
          printf("%s,%d,%d,%s,%d,%d,%zu,%.4f,%.1f\n", sh.name, sh.N, sh.K, kFmtName[F], rows, tiles, bytes, ms,
                 bytes / (ms * 1e6));
          fflush(stdout);
          cudaEventDestroy(a); cudaEventDestroy(b);
        }
      }
      for (auto p : dW) cudaFree(p);
    }
  }
  return 0;
}
