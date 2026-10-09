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
// and g + 8 of each tile (16 bytes NVFP4, 32 bytes FP8; the weights are stored tiled, in exactly this
// order, see retile_nvfp4) and the same 32 K values of x for tokens
// g, g + 8, g + 16, g + 24. The MMA's K index is permuted so that each lane's fragments are those
// contiguous values: logical k {2t, 2t+1, 2t+8, 2t+9} of step s are physical 32t + 4s + {0, 1, 2, 3},
// for the weights (A) and the activations (B) alike.
#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

constexpr int kWarps = 8;
constexpr int kChunk = 128;
// Bytes of one (16-row tile, 128-value chunk) block: NVFP4 values (16 x 64) then their E4M3 scales
// (16 x 8); FP8 values (16 x 128).
constexpr int kTileBytesFp4 = 16 * 64 + 16 * 8, kTileBytesFp8 = 16 * 128;

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

// Up to four matrices sharing one input, launched together; block0[i] is matrix i's first block.
struct StreamProblems {
  const uint8_t* w[4];
  float gscale[4];
  float* y[4];
  int N[4];
  int block0[4];
  int count;
};

// Per lane and chunk: the weights of rows g and g + 8 of every tile.
template <int TILES, bool FP4>
struct WeightRegs {
  uint4 v[TILES][2][FP4 ? 1 : 2];
  uint32_t sc[TILES][2];  // NVFP4: the two E4M3 block scales of the lane's 32 values (low byte first)
};

template <int TILES, bool FP4>
__device__ __forceinline__ void load_weights(WeightRegs<TILES, FP4>& r, const uint8_t* __restrict__ w, int row0,
                                             int N, int K, int chunk, int g, int t) {
  const int chunks = K / kChunk, last_tile = N / 16 - 1;
#pragma unroll
  for (int i = 0; i < TILES; ++i) {
    // The tiled layout (retile_*): each (16-row tile, 128-value chunk) is one contiguous block, in the
    // order the lanes read it, so every load instruction of the warp covers 512 contiguous bytes.
    const int tile = min((row0 >> 4) + i, last_tile);
    const uint8_t* base = w + (static_cast<size_t>(tile) * chunks + chunk) * (FP4 ? kTileBytesFp4 : kTileBytesFp8);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int lane_slot = (h * 8 + g) * 4 + t;
      if constexpr (FP4) {
        const uint8_t* p = base + lane_slot * 16;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.v[i][h][0].x), "=r"(r.v[i][h][0].y), "=r"(r.v[i][h][0].z), "=r"(r.v[i][h][0].w)
                     : "l"(p));
        r.sc[i][h] = *reinterpret_cast<const uint16_t*>(base + 1024 + lane_slot * 2);
      } else {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const uint8_t* p = base + (((h * 2 + q) * 8 + g) * 4 + t) * 16;
          asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(r.v[i][h][q].x), "=r"(r.v[i][h][q].y), "=r"(r.v[i][h][q].z), "=r"(r.v[i][h][q].w)
                       : "l"(p));
        }
      }
    }
  }
}

__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int NT, int TILES, bool FP4, int PF>  // PF: chunks each warp keeps in flight ahead of the one it computes
__global__ void __launch_bounds__(kWarps * 32)
    stream_gemm_kernel(const __half* __restrict__ x, const float* __restrict__ xinv, int M, StreamProblems pr, int K,
                       int ksplit, float* __restrict__ part) {
  // Several matrices with the same input in one launch (q|k|v, gate|up, ...): this block's matrix.
  int pi = 0;
  while (pi + 1 < pr.count && static_cast<int>(blockIdx.x) >= pr.block0[pi + 1]) ++pi;
  const uint8_t* __restrict__ w = pr.w[pi];
  const float gscale = pr.gscale[pi];
  float* __restrict__ y = pr.y[pi];
  const int N = pr.N[pi];
  __shared__ float red[kWarps][TILES * 16][NT * 8 + 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int row0 = (blockIdx.x - pr.block0[pi]) * TILES * 16;
  // Split K (blockIdx.y of ksplit): this block takes chunks [cbeg, cend). The split depends only on the
  // matrix shape, never on M, so a row's sum is the same in every call.
  const int per_split = (K / kChunk + ksplit - 1) / ksplit;
  const int cbeg = blockIdx.y * per_split, chunks = min(K / kChunk, cbeg + per_split);

  float acc[TILES][NT][4];
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  WeightRegs<TILES, FP4> buf[PF + 1];  // buf[0]: the chunk being computed; buf[1..PF]: in flight
#pragma unroll
  for (int p = 0; p < PF; ++p)
    if (cbeg + warp + p * kWarps < chunks) load_weights<TILES, FP4>(buf[p], w, row0, N, K, cbeg + warp + p * kWarps, g, t);
  // PDL (launch.cuh): the first weight chunks above read only the weights, which never change, so they are
  // in flight while this block waits for the kernel that wrote x and xinv. Nothing else happens before this.
  pdl_begin();
  for (int c = cbeg + warp; c < chunks; c += kWarps) {
    if (c + PF * kWarps < chunks) load_weights<TILES, FP4>(buf[PF], w, row0, N, K, c + PF * kWarps, g, t);
    const WeightRegs<TILES, FP4>& cur = buf[0];
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
#pragma unroll
    for (int p = 0; p < PF; ++p) buf[p] = buf[p + 1];
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
    if (ksplit == 1)
      y[static_cast<size_t>(n) * N + row0 + r] = sum * gscale * xinv[n];
    else
      part[(static_cast<size_t>(blockIdx.y) * M + n) * N + row0 + r] = sum;
  }
}

// The split-K partial sums, added in split order.
__global__ void split_reduce_kernel(const float* __restrict__ part, int ksplit, int M, int N, float gscale,
                                    const float* __restrict__ xinv, float* __restrict__ y) {
  pdl_begin();  // first statement: see launch.cuh
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * N) return;
  float sum = 0.f;
  for (int sp = 0; sp < ksplit; ++sp) sum += part[static_cast<size_t>(sp) * M * N + i];
  y[i] = sum * gscale * xinv[i / N];
}

// RMSNorm of each row, written in FP32 and as the scaled FP16 copy the streaming GEMM reads (the same
// values to_half_rows would produce from the FP32 output).
// 1024 threads: the bound keeps a Debug (-G) build within the register file.
__global__ void __launch_bounds__(1024) rmsnorm_half_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                    float* __restrict__ out, __half* __restrict__ xh, float* __restrict__ xinv, int H,
                                    float eps, bool gemma) {
  pdl_begin();  // first statement: see launch.cuh
  __shared__ float red[32];
  const size_t r = blockIdx.x;
  const float* row = x + r * H;
  float ss = 0.f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) ss += row[h] * row[h];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < (blockDim.x + 31) / 32; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / H + eps);
  float mx = 0.f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    const float wv = __bfloat162float(w[h]);
    const float v = row[h] * rstd * (gemma ? 1.f + wv : wv);
    out[r * H + h] = v;
    mx = fmaxf(mx, fabsf(v));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = 0.f;
  for (int i = 0; i < (blockDim.x + 31) / 32; ++i) mx = fmaxf(mx, red[i]);
  int e = 0;
  if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);
  const float sc = ldexpf(1.f, 15 - e);
  for (int h = threadIdx.x; h < H; h += blockDim.x) xh[r * H + h] = __float2half_rn(out[r * H + h] * sc);
  if (threadIdx.x == 0) xinv[r] = ldexpf(1.f, e - 15);
}

// One block per row: a power-of-two scale that puts the row's largest magnitude in [2^14, 2^15).
__global__ void to_half_rows_kernel(const float* __restrict__ x, int K, __half* __restrict__ out,
                                    float* __restrict__ xinv) {
  pdl_begin();  // first statement: see launch.cuh
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

// Enough blocks for the last wave to be a small part of the launch: blocks x ksplit >= ~1100 (about eight
// waves of 3 blocks per SM on 48 SMs), keeping at least one K chunk per warp in each split.
int choose_ksplit(int blocks, int K) {
  static const int mode = [] {
    const char* e = std::getenv("LING_KSPLIT");  // 0: off (default: no robust gain measured); 1: narrow matrices; 2: all
    return e ? std::atoi(e) : 0;
  }();
  if (mode == 0 || (mode == 1 && blocks > 400)) return 1;
  const int by_waves = (1100 + blocks - 1) / blocks;
  const int by_chunks = std::max(1, (K / kChunk) / kWarps);
  return std::max(1, std::min({by_waves, by_chunks, 8}));
}

float* split_buffer(size_t floats) {
  static float* buf = nullptr;
  static size_t cap = 0;
  if (floats > cap) {
    if (buf) cudaFree(buf);
    cap = std::max(floats, size_t(8) << 20);
    if (cudaMalloc(&buf, cap * sizeof(float)) != cudaSuccess) throw std::runtime_error("stream_gemm: split buffer");
  }
  return buf;
}

template <int NT, int TILES, bool FP4>
void launch(const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K, cudaStream_t s) {
  StreamProblems pr{};
  int blocks = 0;
  for (int i = 0; i < n; ++i) {
    pr.w[i] = t[i].w;
    pr.gscale[i] = t[i].scale;
    pr.y[i] = t[i].y;
    pr.N[i] = t[i].N;
    pr.block0[i] = blocks;
    blocks += (t[i].N + TILES * 16 - 1) / (TILES * 16);
  }
  pr.count = n;
  const int ksplit = n == 1 ? choose_ksplit(blocks, K) : 1;
  float* part = ksplit > 1 ? split_buffer(size_t(ksplit) * M * t[0].N) : nullptr;
  const dim3 grid(blocks, ksplit);
  // One chunk in flight per warp ahead of the one it computes: 2 and 3 were measured no faster.
  launch_kernel("stream_gemm", stream_gemm_kernel<NT, TILES, FP4, 1>, grid, dim3(kWarps * 32), 0, s, x, xinv, M, pr, K,
                ksplit, part);
  if (ksplit > 1)
    launch_kernel("stream_gemm split", split_reduce_kernel, dim3((M * t[0].N + 255) / 256), dim3(256), 0, s, part, ksplit,
                  M, t[0].N, t[0].scale, xinv, t[0].y);
}

template <bool FP4>
void dispatch(const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K, cudaStream_t s) {
  if (K % kChunk != 0) throw std::runtime_error("stream_gemm: K must be a multiple of 128");
  if (M < 1 || M > kMaxStreamRows) throw std::runtime_error("stream_gemm: M must be in [1, 32]");
  if (n < 1 || n > 4) throw std::runtime_error("stream_gemm: 1 to 4 matrices per launch");
  int total = 0;
  for (int i = 0; i < n; ++i) total += t[i].N;
  // Two tiles per warp share each activation fragment; narrow launches keep one so that more blocks
  // stream at once. (The choice does not change any element's arithmetic.)
  if (total < 8192) {
    if (M <= 8) launch<1, 1, FP4>(x, xinv, M, t, n, K, s);
    else if (M <= 16) launch<2, 1, FP4>(x, xinv, M, t, n, K, s);
    else launch<4, 1, FP4>(x, xinv, M, t, n, K, s);
  } else {
    if (M <= 8) launch<1, 2, FP4>(x, xinv, M, t, n, K, s);
    else if (M <= 16) launch<2, 2, FP4>(x, xinv, M, t, n, K, s);
    else launch<4, 2, FP4>(x, xinv, M, t, n, K, s);
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("stream_gemm: ") + cudaGetErrorString(e));
}

// Row-major checkpoint layout -> tiled layout, one thread per 16-byte piece of the output.
__global__ void retile_nvfp4_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ s,
                                    uint8_t* __restrict__ out, int N, int K) {
  const int chunks = K / kChunk;
  const size_t blocks = static_cast<size_t>(N / 16) * chunks;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < blocks * 32 * 2;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t blk = i / 64;
    const int slot = static_cast<int>(i % 64);  // lane slot (h * 8 + g) * 4 + t
    const int tile = static_cast<int>(blk / chunks), chunk = static_cast<int>(blk % chunks);
    const int t = slot % 4, r = slot / 4;  // r = h * 8 + g: row g + 8h of the tile
    const size_t n = static_cast<size_t>(tile) * 16 + (r % 8) + 8 * (r / 8);
    uint8_t* base = out + blk * kTileBytesFp4;
    *reinterpret_cast<uint4*>(base + slot * 16) =
        *reinterpret_cast<const uint4*>(w + n * (K / 2) + chunk * (kChunk / 2) + t * 16);
    *reinterpret_cast<uint16_t*>(base + 1024 + slot * 2) =
        *reinterpret_cast<const uint16_t*>(s + n * (K / 16) + chunk * (kChunk / 16) + t * 2);
  }
}

__global__ void retile_fp8_kernel(const uint8_t* __restrict__ w, uint8_t* __restrict__ out, int N, int K) {
  const int chunks = K / kChunk;
  const size_t pieces = static_cast<size_t>(N / 16) * chunks * 128;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < pieces;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t blk = i / 128;
    const int piece = static_cast<int>(i % 128);  // ((h * 2 + q) * 8 + g) * 4 + t
    const int t = piece % 4, g = (piece / 4) % 8, q = (piece / 32) % 2, h = piece / 64;
    const int tile = static_cast<int>(blk / chunks), chunk = static_cast<int>(blk % chunks);
    const size_t n = static_cast<size_t>(tile) * 16 + g + 8 * h;
    *reinterpret_cast<uint4*>(out + blk * kTileBytesFp8 + piece * 16) =
        *reinterpret_cast<const uint4*>(w + n * K + chunk * kChunk + t * 32 + q * 16);
  }
}

__constant__ float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// Tiled layout -> BF16 row-major (the prefill path's cuBLAS input), one thread per output pair (NVFP4)
// or element (FP8).
__global__ void dequant_tiled_nvfp4_kernel(const uint8_t* __restrict__ tw, float scale2, __nv_bfloat16* __restrict__ out,
                                           int N, int K) {
  const int chunks = K / kChunk;
  const size_t pairs = static_cast<size_t>(N) * K / 2;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < pairs;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int n = static_cast<int>(i / (K / 2)), k = static_cast<int>(i % (K / 2)) * 2;
    const int tile = n / 16, r = n % 16, g = r % 8, h = r / 8, chunk = k / kChunk, kk = k % kChunk, t = kk / 32;
    const int slot = (h * 8 + g) * 4 + t;
    const uint8_t* base = tw + (static_cast<size_t>(tile) * chunks + chunk) * kTileBytesFp4;
    const uint8_t b = base[slot * 16 + (kk % 32) / 2];
    __nv_fp8_e4m3 sc;
    sc.__x = base[1024 + slot * 2 + (kk % 32) / 16];
    const float s = static_cast<float>(sc) * scale2;
    out[static_cast<size_t>(n) * K + k] = __float2bfloat16(kE2M1[b & 0xf] * s);
    out[static_cast<size_t>(n) * K + k + 1] = __float2bfloat16(kE2M1[b >> 4] * s);
  }
}

__global__ void dequant_tiled_fp8_kernel(const uint8_t* __restrict__ tw, float wscale, __nv_bfloat16* __restrict__ out,
                                         int N, int K) {
  const int chunks = K / kChunk;
  const size_t total = static_cast<size_t>(N) * K;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int n = static_cast<int>(i / K), k = static_cast<int>(i % K);
    const int tile = n / 16, r = n % 16, g = r % 8, h = r / 8, chunk = k / kChunk, kk = k % kChunk;
    const int t = kk / 32, q = (kk % 32) / 16;
    const uint8_t* base = tw + (static_cast<size_t>(tile) * chunks + chunk) * kTileBytesFp8;
    __nv_fp8_e4m3 v;
    v.__x = base[(((h * 2 + q) * 8 + g) * 4 + t) * 16 + kk % 16];
    out[i] = __float2bfloat16(static_cast<float>(v) * wscale);
  }
}

int grid_of(size_t n) { return static_cast<int>(std::min<size_t>((n + 255) / 256, 65535 * 4)); }

void check_shape(int N, int K, const char* what) {
  if (N % 16 != 0 || K % kChunk != 0) throw std::runtime_error(std::string(what) + ": N % 16 and K % 128 must be 0");
}

}  // namespace

size_t tiled_bytes_nvfp4(int N, int K) { return static_cast<size_t>(N) * K / 2 + static_cast<size_t>(N) * K / 16; }
size_t tiled_bytes_fp8(int N, int K) { return static_cast<size_t>(N) * K; }

void retile_nvfp4(const uint8_t* w, const uint8_t* wscale, uint8_t* out, int N, int K, cudaStream_t s) {
  check_shape(N, K, "retile_nvfp4");
  retile_nvfp4_kernel<<<grid_of(static_cast<size_t>(N / 16) * (K / kChunk) * 64), 256, 0, s>>>(w, wscale, out, N, K);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("retile_nvfp4: ") + cudaGetErrorString(e));
}

void retile_fp8(const uint8_t* w, uint8_t* out, int N, int K, cudaStream_t s) {
  check_shape(N, K, "retile_fp8");
  retile_fp8_kernel<<<grid_of(static_cast<size_t>(N / 16) * (K / kChunk) * 128), 256, 0, s>>>(w, out, N, K);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("retile_fp8: ") + cudaGetErrorString(e));
}

void dequant_tiled_nvfp4(const uint8_t* tw, float scale2, __nv_bfloat16* out, int N, int K, cudaStream_t s) {
  dequant_tiled_nvfp4_kernel<<<grid_of(static_cast<size_t>(N) * K / 2), 256, 0, s>>>(tw, scale2, out, N, K);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("dequant_tiled_nvfp4: ") + cudaGetErrorString(e));
}

void dequant_tiled_fp8(const uint8_t* tw, float wscale, __nv_bfloat16* out, int N, int K, cudaStream_t s) {
  dequant_tiled_fp8_kernel<<<grid_of(static_cast<size_t>(N) * K), 256, 0, s>>>(tw, wscale, out, N, K);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("dequant_tiled_fp8: ") + cudaGetErrorString(e));
}

void rmsnorm_half(const float* x, const __nv_bfloat16* w, float* out, __half* xh, float* xinv, int rows, int H, float eps,
                  bool gemma, cudaStream_t s) {
  launch_kernel("rmsnorm_half", rmsnorm_half_kernel, dim3(rows), dim3(1024), 0, s, x, w, out, xh, xinv, H, eps, gemma);
}

void to_half_rows(const float* x, int M, int K, __half* out, float* xinv, cudaStream_t s) {
  launch_kernel("to_half_rows", to_half_rows_kernel, dim3(M), dim3(256), 0, s, x, K, out, xinv);
}

void stream_gemm_nvfp4(const __half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
                       float scale2, float* y, int N, int K, cudaStream_t s) {
  (void)wscale;  // the scales live inside the tiled blob
  const StreamTarget t{w, nullptr, scale2, y, N};
  dispatch<true>(x, xinv, M, &t, 1, K, s);
}

void stream_gemm_fp8(const __half* x, const float* xinv, int M, const uint8_t* w, float wscale, float* y, int N,
                     int K, cudaStream_t s) {
  const StreamTarget t{w, nullptr, wscale, y, N};
  dispatch<false>(x, xinv, M, &t, 1, K, s);
}

void stream_gemm_multi(bool nvfp4, const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K,
                       cudaStream_t s) {
  if (nvfp4) dispatch<true>(x, xinv, M, t, n, K, s);
  else dispatch<false>(x, xinv, M, t, n, K, s);
}

}  // namespace ling::kernels
