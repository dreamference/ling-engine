// Weight-streaming GEMM for 1 to 32 rows on tensor cores (decode, verify and the drafter).
//
// y[M][N] = gscale * x[M][K] . W[N][K]^T with W in NVFP4 (packed [N][K/2], E4M3 block scales [N][K/16])
// or FP8 E4M3 ([N][K]), and x converted once to FP16 with a power-of-two scale per row.
//
// Numerics: an NVFP4 value times its E4M3 block scale has at most 6 significant bits and an E4M3 value
// 4, so both are exact in FP16; the only rounding before the FP32 accumulation is x to FP16 (11 bits),
// and the row's power-of-two scale keeps that relative. The global scale is applied in FP32 at the end.
//
// Row invariance: every output element is accumulated over the same K chunks by the same warp in the
// same order, and the warps' partial sums are added in a fixed order, whatever M is. A row's result is
// therefore bit-for-bit the same in a 1-row decode and in a 16- or 32-row verify, which is what makes
// greedy speculative decoding produce exactly the tokens of plain decoding.
//
// Layout: a block is 8 warps over the same TILES x 16 weight rows; warp w takes K chunks w, w + 8, ...
// of 128 values. In a chunk lane (g = lane / 4, t = lane % 4) loads 32 consecutive K values of rows g
// and g + 8 of each tile (16 bytes NVFP4, 32 bytes FP8) and the same 32 K values of x for tokens
// g, g + 8, g + 16, g + 24. The MMA's K index is permuted so that each lane's fragments are those
// contiguous values: logical k {2t, 2t+1, 2t+8, 2t+9} of step s are physical 32t + 4s + {0, 1, 2, 3},
// for the weights (A) and the activations (B) alike.
#include "core/kernels.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

constexpr int kWarps = 8;
constexpr int kChunk = 128;

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

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Per lane and chunk: the weights of rows g and g + 8 of every tile.
template <int TILES, bool FP4>
struct WeightRegs {
  uint4 v[TILES][2][FP4 ? 1 : 2];
  uint32_t sc[TILES][2];  // NVFP4: the two E4M3 block scales of the lane's 32 values (low byte first)
};

template <int TILES, bool FP4>
__device__ __forceinline__ void load_weights(WeightRegs<TILES, FP4>& r, const uint8_t* __restrict__ w,
                                             const uint8_t* __restrict__ ws, int row0, int N, int K, int chunk,
                                             int g, int t) {
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = min(row0 + 16 * i + g + 8 * h, N - 1);
      if constexpr (FP4) {
        const uint8_t* p = w + static_cast<size_t>(n) * (K / 2) + chunk * (kChunk / 2) + t * 16;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.v[i][h][0].x), "=r"(r.v[i][h][0].y), "=r"(r.v[i][h][0].z), "=r"(r.v[i][h][0].w)
                     : "l"(p));
        r.sc[i][h] = *reinterpret_cast<const uint16_t*>(ws + static_cast<size_t>(n) * (K / 16) +
                                                         chunk * (kChunk / 16) + t * 2);
      } else {
        const uint8_t* p = w + static_cast<size_t>(n) * K + chunk * kChunk + t * 32;
#pragma unroll
        for (int q = 0; q < 2; ++q)
          asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(r.v[i][h][q].x), "=r"(r.v[i][h][q].y), "=r"(r.v[i][h][q].z), "=r"(r.v[i][h][q].w)
                       : "l"(p + 16 * q));
      }
    }
}

__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int NT, int TILES, bool FP4>
__global__ void __launch_bounds__(kWarps * 32)
    stream_gemm_kernel(const __half* __restrict__ x, const float* __restrict__ xinv, int M,
                       const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, float gscale,
                       float* __restrict__ y, int N, int K) {
  __shared__ float red[kWarps][TILES * 16][NT * 8 + 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * TILES * 16;
  const int chunks = K / kChunk;

  float acc[TILES][NT][4];
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  WeightRegs<TILES, FP4> cur, nxt;
  if (warp < chunks) load_weights<TILES, FP4>(cur, w, ws, row0, N, K, warp, g, t);
  for (int c = warp; c < chunks; c += kWarps) {
    if (c + kWarps < chunks) load_weights<TILES, FP4>(nxt, w, ws, row0, N, K, c + kWarps, g, t);
    // This chunk's activations: 32 halves (16 words) per token for tokens g + 8j.
    uint32_t xw[NT][16];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int n = g + 8 * j;
      if (n < M) {
        const uint4* p = reinterpret_cast<const uint4*>(x + static_cast<size_t>(n) * K + c * kChunk + t * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 v = __ldg(p + q);
          xw[j][4 * q] = v.x;
          xw[j][4 * q + 1] = v.y;
          xw[j][4 * q + 2] = v.z;
          xw[j][4 * q + 3] = v.w;
        }
      } else {
#pragma unroll
        for (int q = 0; q < 16; ++q) xw[j][q] = 0u;
      }
    }
    uint32_t sc_h2[TILES][2][2];
    if constexpr (FP4) {
#pragma unroll
      for (int i = 0; i < TILES; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t s2 = e4m3x2_to_h2(cur.sc[i][h]);  // low half: first 16 values, high: next 16
          sc_h2[i][h][0] = (s2 & 0xffffu) | (s2 << 16);
          sc_h2[i][h][1] = (s2 >> 16) | (s2 & 0xffff0000u);
        }
    }
#pragma unroll
    for (int s = 0; s < 8; ++s) {
#pragma unroll
      for (int i = 0; i < TILES; ++i) {
        uint32_t a[2][2];  // [row g / g + 8][k pair 0 / 1]
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          if constexpr (FP4) {
            const uint32_t word = word_of(cur.v[i][h][0], s >> 1);
            const uint32_t bytes = word >> (16 * (s & 1));
            const uint32_t scale = sc_h2[i][h][s >> 2];
            a[h][0] = hmul2(e2m1x2_to_h2(bytes & 0xffu), scale);
            a[h][1] = hmul2(e2m1x2_to_h2((bytes >> 8) & 0xffu), scale);
          } else {
            const uint32_t word = word_of(cur.v[i][h][s >> 2], s & 3);
            a[h][0] = e4m3x2_to_h2(word & 0xffffu);
            a[h][1] = e4m3x2_to_h2(word >> 16);
          }
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) mma16816(acc[i][j], a[0][0], a[1][0], a[0][1], a[1][1], xw[j][2 * s], xw[j][2 * s + 1]);
      }
    }
    if (c + kWarps < chunks) cur = nxt;
  }

  // Fixed-order reduction over the warps.
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      red[warp][16 * i + g][8 * j + 2 * t] = acc[i][j][0];
      red[warp][16 * i + g][8 * j + 2 * t + 1] = acc[i][j][1];
      red[warp][16 * i + g + 8][8 * j + 2 * t] = acc[i][j][2];
      red[warp][16 * i + g + 8][8 * j + 2 * t + 1] = acc[i][j][3];
    }
  __syncthreads();
  for (int e = threadIdx.x; e < TILES * 16 * NT * 8; e += blockDim.x) {
    const int r = e / (NT * 8), n = e % (NT * 8);
    if (n >= M || row0 + r >= N) continue;
    float sum = 0.f;
#pragma unroll
    for (int wi = 0; wi < kWarps; ++wi) sum += red[wi][r][n];
    y[static_cast<size_t>(n) * N + row0 + r] = sum * gscale * xinv[n];
  }
}

// One block per row: a power-of-two scale that puts the row's largest magnitude in [2^14, 2^15).
__global__ void to_half_rows_kernel(const float* __restrict__ x, int K, __half* __restrict__ out,
                                    float* __restrict__ xinv) {
  __shared__ float red[32];
  const float* row = x + static_cast<size_t>(blockIdx.x) * K;
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(row[k]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = 0.f;
  for (int i = 0; i < (blockDim.x + 31) / 32; ++i) mx = fmaxf(mx, red[i]);
  int e = 0;
  if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);  // mx < 2^e
  const float s = ldexpf(1.f, 15 - e);
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    out[static_cast<size_t>(blockIdx.x) * K + k] = __float2half_rn(row[k] * s);
  if (threadIdx.x == 0) xinv[blockIdx.x] = ldexpf(1.f, e - 15);
}

template <int NT, int TILES, bool FP4>
void launch(const __half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* ws, float gscale,
            float* y, int N, int K, cudaStream_t s) {
  const int blocks = (N + TILES * 16 - 1) / (TILES * 16);
  stream_gemm_kernel<NT, TILES, FP4><<<blocks, kWarps * 32, 0, s>>>(x, xinv, M, w, ws, gscale, y, N, K);
}

template <bool FP4>
void dispatch(const __half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* ws, float gscale,
              float* y, int N, int K, cudaStream_t s) {
  if (K % kChunk != 0) throw std::runtime_error("stream_gemm: K must be a multiple of 128");
  if (M < 1 || M > kMaxStreamRows) throw std::runtime_error("stream_gemm: M must be in [1, 32]");
  // Two tiles per warp share each activation fragment; narrow matrices keep one so that more blocks
  // stream at once.
  if (N >= 8192) {
    if (M <= 8) launch<1, 2, FP4>(x, xinv, M, w, ws, gscale, y, N, K, s);
    else if (M <= 16) launch<2, 2, FP4>(x, xinv, M, w, ws, gscale, y, N, K, s);
    else launch<4, 2, FP4>(x, xinv, M, w, ws, gscale, y, N, K, s);
  } else {
    if (M <= 8) launch<1, 1, FP4>(x, xinv, M, w, ws, gscale, y, N, K, s);
    else if (M <= 16) launch<2, 1, FP4>(x, xinv, M, w, ws, gscale, y, N, K, s);
    else launch<4, 1, FP4>(x, xinv, M, w, ws, gscale, y, N, K, s);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("stream_gemm: ") + cudaGetErrorString(e));
}

}  // namespace

void to_half_rows(const float* x, int M, int K, __half* out, float* xinv, cudaStream_t s) {
  to_half_rows_kernel<<<M, 256, 0, s>>>(x, K, out, xinv);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("to_half_rows: ") + cudaGetErrorString(e));
}

void stream_gemm_nvfp4(const __half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
                       float scale2, float* y, int N, int K, cudaStream_t s) {
  dispatch<true>(x, xinv, M, w, wscale, scale2, y, N, K, s);
}

void stream_gemm_fp8(const __half* x, const float* xinv, int M, const uint8_t* w, float wscale, float* y, int N,
                     int K, cudaStream_t s) {
  dispatch<false>(x, xinv, M, w, nullptr, wscale, y, N, K, s);
}

}  // namespace ling::kernels
