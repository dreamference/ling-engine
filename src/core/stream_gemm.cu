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
#include <cstdio>
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
// B26: a coded NVFP4 blob is every block's 1,024 value bytes (aligned, as the raw blob's 1,152-byte blocks are), then
// every block's 96 code bytes (kernels.cuh, retile_nvfp4_coded).
constexpr int kValueBytesFp4 = 16 * 64, kCodeBytesFp4 = 128 * 6 / 8;

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
  const uint8_t* table[4];  // CODED: each matrix's scale-code table
};

// Per lane and chunk: the weights of rows g and g + 8 of every tile.
template <int TILES, bool FP4, bool CODED>
struct WeightRegs {
  uint4 v[TILES][2][FP4 ? 1 : 2];
  uint32_t sc[TILES][2];  // NVFP4: the two E4M3 block scales of the lane's 32 values (low byte first)
  uint32_t cw[TILES][CODED ? 2 : 1];  // CODED: the lane's code bytes for both of its slots (low bytes, high nibbles)
};

template <int TILES, bool FP4, bool CODED>
__device__ __forceinline__ void load_weights(WeightRegs<TILES, FP4, CODED>& r, const uint8_t* __restrict__ w, int row0,
                                             int N, int K, int chunk, int g, int t) {
  const int chunks = K / kChunk, last_tile = N / 16 - 1;
#pragma unroll
  for (int i = 0; i < TILES; ++i) {
    // The tiled layout (retile_*): each (16-row tile, 128-value chunk) is one contiguous block, in the
    // order the lanes read it, so every load instruction of the warp covers 512 contiguous bytes.
    const int tile = min((row0 >> 4) + i, last_tile);
    const size_t blk = static_cast<size_t>(tile) * chunks + chunk;
    const uint8_t* base = w + blk * (FP4 ? (CODED ? kValueBytesFp4 : kTileBytesFp4) : kTileBytesFp8);
    const uint8_t* codes = CODED ? w + static_cast<size_t>(N / 16) * chunks * kValueBytesFp4 + blk * kCodeBytesFp4 : nullptr;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int lane_slot = (h * 8 + g) * 4 + t;
      if constexpr (FP4) {
        const uint8_t* p = base + lane_slot * 16;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.v[i][h][0].x), "=r"(r.v[i][h][0].y), "=r"(r.v[i][h][0].z), "=r"(r.v[i][h][0].w)
                     : "l"(p));
        if constexpr (CODED) {
          // Lane L's two slots (L and L + 32) in one 16-bit word of low bytes at 2L and one byte of high nibbles at
          // 64 + L: two loads per tile, as the raw blob's two 16-bit scale loads.
          if (h == 0) {
            const int L = g * 4 + t;
            r.cw[i][0] = *reinterpret_cast<const uint16_t*>(codes + 2 * L);
            r.cw[i][1] = codes[64 + L];
          }
        } else {
          r.sc[i][h] = *reinterpret_cast<const uint16_t*>(base + 1024 + lane_slot * 2);
        }
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

// The kernel's body as a device function of its block coordinates (bx, by) and of what it waits on before it reads
// x (`wait`: pdl_begin() in the kernel; a dependency counter in a persistent kernel, megakernel B32): one body, so
// every launcher computes the same bits. PF: chunks each warp keeps in flight ahead of the one it computes; CODED
// (NVFP4 only): 6-bit scale codes (B26).
template <int NT, int TILES, bool FP4, int PF, bool CODED, typename Wait>
__device__ __forceinline__ void stream_gemm_body(const __half* __restrict__ x, const float* __restrict__ xinv, int M,
                                                 const StreamProblems& pr, int K, int ksplit, float* __restrict__ part,
                                                 int bx, int by, Wait&& wait) {
  // Several matrices with the same input in one launch (q|k|v, gate|up, ...): this block's matrix.
  int pi = 0;
  while (pi + 1 < pr.count && bx >= pr.block0[pi + 1]) ++pi;
  const uint8_t* __restrict__ w = pr.w[pi];
  const float gscale = pr.gscale[pi];
  float* __restrict__ y = pr.y[pi];
  const int N = pr.N[pi];
  __shared__ float red[kWarps][TILES * 16][NT * 8 + 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int row0 = (bx - pr.block0[pi]) * TILES * 16;
  // Split K (blockIdx.y of ksplit): this block takes chunks [cbeg, cend). The split depends only on the
  // matrix shape, never on M, so a row's sum is the same in every call.
  const int per_split = (K / kChunk + ksplit - 1) / ksplit;
  const int cbeg = by * per_split, chunks = min(K / kChunk, cbeg + per_split);

  float acc[TILES][NT][4];
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  WeightRegs<TILES, FP4, CODED> buf[PF + 1];  // buf[0]: the chunk being computed; buf[1..PF]: in flight
#pragma unroll
  for (int p = 0; p < PF; ++p)
    if (cbeg + warp + p * kWarps < chunks)
      load_weights<TILES, FP4, CODED>(buf[p], w, row0, N, K, cbeg + warp + p * kWarps, g, t);
  // CODED: the matrix's table (immutable, like the weights) in shared memory, and each slot's bit offset.
  __shared__ uint8_t tab[CODED ? kScaleCodes : 1];

  if constexpr (CODED) {
    if (threadIdx.x < kScaleCodes) tab[threadIdx.x] = pr.table[pi][threadIdx.x];
    __syncthreads();
  }
  // PDL (launch.cuh): the first weight chunks above read only the weights, which never change, so they are
  // in flight while this block waits for the kernel that wrote x and xinv. Nothing else happens before this.
  wait();
  for (int c = cbeg + warp; c < chunks; c += kWarps) {
    if (c + PF * kWarps < chunks) load_weights<TILES, FP4, CODED>(buf[PF], w, row0, N, K, c + PF * kWarps, g, t);
    const WeightRegs<TILES, FP4, CODED>& cur = buf[0];
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
          uint32_t pair = cur.sc[i][h];
          if constexpr (CODED) {  // the two codes -> the two E4M3 scale bytes, exactly the raw blob's
            const uint32_t f = ((cur.cw[i][0] >> (8 * h)) & 0xffu) | (((cur.cw[i][1] >> (4 * h)) & 0xfu) << 8);
            pair = uint32_t(tab[f & 63u]) | (uint32_t(tab[(f >> 6) & 63u]) << 8);
          }
          const uint32_t s2 = e4m3x2_to_h2(pair);  // low half: first 16 values, high: next 16
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
      part[(static_cast<size_t>(by) * M + n) * N + row0 + r] = sum;
  }
}

struct PdlBegin {
  __device__ __forceinline__ void operator()() const { pdl_begin(); }
};

template <int NT, int TILES, bool FP4, int PF, bool CODED = false>
__global__ void __launch_bounds__(kWarps * 32)
    stream_gemm_kernel(const __half* __restrict__ x, const float* __restrict__ xinv, int M, StreamProblems pr, int K,
                       int ksplit, float* __restrict__ part) {
  stream_gemm_body<NT, TILES, FP4, PF, CODED>(x, xinv, M, pr, K, ksplit, part, blockIdx.x, blockIdx.y, PdlBegin{});
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
// The body for row `r` (one block per row); `wait` as in stream_gemm_body (the kernel's first statement).
template <typename Wait>
__device__ __forceinline__ void to_half_rows_body(const float* __restrict__ x, int K, __half* __restrict__ out,
                                                  float* __restrict__ xinv, int r, Wait&& wait) {
  wait();  // first statement: see launch.cuh
  __shared__ float red[32];
  const float* row = x + static_cast<size_t>(r) * K;
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
    out[static_cast<size_t>(r) * K + k] = __float2half_rn(row[k] * s);
  if (threadIdx.x == 0) xinv[r] = ldexpf(1.f, e - 15);
}

__global__ void to_half_rows_kernel(const float* __restrict__ x, int K, __half* __restrict__ out,
                                    float* __restrict__ xinv) {
  to_half_rows_body(x, K, out, xinv, blockIdx.x, PdlBegin{});
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

template <int NT, int TILES, bool FP4, int PF, bool CODED>
void launch(const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K, cudaStream_t s) {
  StreamProblems pr{};
  int blocks = 0;
  for (int i = 0; i < n; ++i) {
    pr.w[i] = t[i].w;
    pr.gscale[i] = t[i].scale;
    pr.y[i] = t[i].y;
    pr.N[i] = t[i].N;
    pr.table[i] = t[i].wscale;
    pr.block0[i] = blocks;
    blocks += (t[i].N + TILES * 16 - 1) / (TILES * 16);
  }
  pr.count = n;
  const int ksplit = n == 1 ? choose_ksplit(blocks, K) : 1;
  float* part = ksplit > 1 ? split_buffer(size_t(ksplit) * M * t[0].N) : nullptr;
  const dim3 grid(blocks, ksplit);
  launch_kernel("stream_gemm", stream_gemm_kernel<NT, TILES, FP4, PF, CODED>, grid, dim3(kWarps * 32), 0, s, x, xinv, M, pr, K,
                ksplit, part);
  if (ksplit > 1)
    launch_kernel("stream_gemm split", split_reduce_kernel, dim3((M * t[0].N + 255) / 256), dim3(256), 0, s, part, ksplit,
                  M, t[0].N, t[0].scale, xinv, t[0].y);
}

// The launch shape per bucket (B16, measured 2026-10-13 in the step with ling-run --spec and --batch-bench, and per
// launch group with ling-stream-tests --bench-groups). Neither choice changes any element's arithmetic: each output
// is still accumulated by warp w over chunks w, w + 8, ... in the same order, so these move bytes, not bits.
//   - Tiles per block. Every block reads the whole of x for its rows: per chunk a lane loads NT x 64 bytes of
//     activations against 16 or 32 bytes of weights per tile, so with one tile a block's activation reads (from
//     L2) outweigh its weight stream several times over at 17-32 rows, and the narrow matrices (N = 5,120: the FFN
//     down projection, the attention and DeltaNet out-projections) fell to 110-190 GB/s at M = 24 and 32 (the
//     batched verify of two sequences: ffn down 113 GB/s). Two tiles halve the activation reads per weight byte:
//     ffn down 230 GB/s at M = 24, the batched step about 6 ms shorter. At M <= 16 two tiles on the narrow
//     matrices gain too (out-projections 211-228 -> 241-244 GB/s at M = 12).
//   - Chunks in flight per warp (PF): two, now that the narrow launches have half the blocks.
// LING_SG_TN, LING_SG_TW (tiles at M <= 16 for narrow / wide launches), LING_SG_TN4, LING_SG_TW4 (at M > 16) and
// LING_SG_PF override them for experiments (TILES 1, 2 or 4, at most 2 at M > 16; PF 1 to 4).
constexpr int kTilesNarrow = 2, kTilesWide = 2, kTilesNarrow4 = 2, kTilesWide4 = 2, kInFlight = 2;

int sg_env(const char* n, int d) {
  const char* e = std::getenv(n);
  return e ? std::atoi(e) : d;
}

template <int NT, bool FP4, int PF, bool CODED>
void launch_t(int tiles, const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K, cudaStream_t s) {
  if (tiles == 1) launch<NT, 1, FP4, PF, CODED>(x, xinv, M, t, n, K, s);
  else if (tiles == 2 || NT == 4) launch<NT, 2, FP4, PF, CODED>(x, xinv, M, t, n, K, s);  // NT = 4 x 4 tiles: no room
  else launch<NT, (NT == 4 ? 2 : 4), FP4, PF, CODED>(x, xinv, M, t, n, K, s);
}

template <int NT, bool FP4>
void launch_k(int tiles, const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K, cudaStream_t s) {
  static const int pf = sg_env("LING_SG_PF", kInFlight);
  const bool coded = FP4 && t[0].wscale != nullptr;  // dispatch keeps a launch's matrices of one kind
  // PF 3 and 4 (LING_SG_PF): the megakernel probe's deeper run-ahead past a PDL boundary (BACKLOG 28); the same
  // arithmetic in the same order, only more weight chunks in flight per warp.
  if (coded) {
    if constexpr (FP4) {
      if (pf == 4) launch_t<NT, FP4, 4, true>(tiles, x, xinv, M, t, n, K, s);
      else if (pf == 3) launch_t<NT, FP4, 3, true>(tiles, x, xinv, M, t, n, K, s);
      else if (pf == 2) launch_t<NT, FP4, 2, true>(tiles, x, xinv, M, t, n, K, s);
      else launch_t<NT, FP4, 1, true>(tiles, x, xinv, M, t, n, K, s);
    }
  } else if (pf == 4) {
    launch_t<NT, FP4, 4, false>(tiles, x, xinv, M, t, n, K, s);
  } else if (pf == 3) {
    launch_t<NT, FP4, 3, false>(tiles, x, xinv, M, t, n, K, s);
  } else if (pf == 2) {
    launch_t<NT, FP4, 2, false>(tiles, x, xinv, M, t, n, K, s);
  } else {
    launch_t<NT, FP4, 1, false>(tiles, x, xinv, M, t, n, K, s);
  }
}

template <bool FP4>
void dispatch(const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K, cudaStream_t s) {
  if (K % kChunk != 0) throw std::runtime_error("stream_gemm: K must be a multiple of 128");
  if (M < 1 || M > kMaxStreamRows) throw std::runtime_error("stream_gemm: M must be in [1, 32]");
  if (n < 1 || n > 4) throw std::runtime_error("stream_gemm: 1 to 4 matrices per launch");
  // B26: coded and raw NVFP4 blobs in one call go as two launches (a matrix's result does not depend on its
  // launch companions, so nothing changes but the launch count; only a checkpoint with a > 64-scale matrix next to
  // a coded one in one group takes it).
  if (FP4 && n > 1) {
    StreamTarget coded[4], raw[4];
    int nc = 0, nr = 0;
    for (int i = 0; i < n; ++i) (t[i].wscale ? coded[nc++] : raw[nr++]) = t[i];
    if (nc > 0 && nr > 0) {
      dispatch<FP4>(x, xinv, M, coded, nc, K, s);
      dispatch<FP4>(x, xinv, M, raw, nr, K, s);
      return;
    }
  }
  int total = 0;
  for (int i = 0; i < n; ++i) total += t[i].N;
  static const int tn = sg_env("LING_SG_TN", kTilesNarrow), tw = sg_env("LING_SG_TW", kTilesWide),
                   tn4 = sg_env("LING_SG_TN4", kTilesNarrow4), tw4 = sg_env("LING_SG_TW4", kTilesWide4);
  const bool narrow = total < 8192;
  if (M <= 8) launch_k<1, FP4>(narrow ? tn : tw, x, xinv, M, t, n, K, s);
  else if (M <= 16) launch_k<2, FP4>(narrow ? tn : tw, x, xinv, M, t, n, K, s);
  else launch_k<4, FP4>(narrow ? tn4 : tw4, x, xinv, M, t, n, K, s);
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

// B26: the coded blob. One thread per 16-byte piece of values (the raw blob's pieces at the coded stride), and one
// thread per block for its codes (24 words of two 6-bit codes per lane slot) and its raw scales.
__global__ void retile_nvfp4_coded_values_kernel(const uint8_t* __restrict__ w, uint8_t* __restrict__ out, int N, int K) {
  const int chunks = K / kChunk;
  const size_t blocks = static_cast<size_t>(N / 16) * chunks;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < blocks * 64;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t blk = i / 64;
    const int slot = static_cast<int>(i % 64), t = slot % 4, r = slot / 4;
    const int tile = static_cast<int>(blk / chunks), chunk = static_cast<int>(blk % chunks);
    const size_t n = static_cast<size_t>(tile) * 16 + (r % 8) + 8 * (r / 8);
    *reinterpret_cast<uint4*>(out + blk * kValueBytesFp4 + slot * 16) =
        *reinterpret_cast<const uint4*>(w + n * (K / 2) + chunk * (kChunk / 2) + t * 16);
  }
}

__global__ void retile_nvfp4_coded_scales_kernel(const uint8_t* __restrict__ s, const uint8_t* __restrict__ code_of,
                                                 uint8_t* __restrict__ out, uint8_t* __restrict__ raw, int N, int K) {
  const int chunks = K / kChunk;
  const size_t blocks = static_cast<size_t>(N / 16) * chunks;
  for (size_t blk = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; blk < blocks;
       blk += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int tile = static_cast<int>(blk / chunks), chunk = static_cast<int>(blk % chunks);
    uint8_t* dst = out + blocks * kValueBytesFp4 + blk * kCodeBytesFp4;
    uint32_t f[64];
    for (int slot = 0; slot < 64; ++slot) {
      const int t = slot % 4, r = slot / 4;
      const size_t n = static_cast<size_t>(tile) * 16 + (r % 8) + 8 * (r / 8);
      const uint8_t s0 = s[n * (K / 16) + chunk * (kChunk / 16) + t * 2], s1 = s[n * (K / 16) + chunk * (kChunk / 16) + t * 2 + 1];
      raw[blk * 128 + slot * 2] = s0;
      raw[blk * 128 + slot * 2 + 1] = s1;
      f[slot] = uint32_t(code_of[s0]) | (uint32_t(code_of[s1]) << 6);  // 12 bits: the first scale's code low
    }
    // Lane L's slots L and L + 32: their low bytes at 2L, 2L + 1, their high nibbles in byte 64 + L.
    for (int L = 0; L < 32; ++L) {
      dst[2 * L] = static_cast<uint8_t>(f[L] & 0xffu);
      dst[2 * L + 1] = static_cast<uint8_t>(f[L + 32] & 0xffu);
      dst[64 + L] = static_cast<uint8_t>((f[L] >> 8) | ((f[L + 32] >> 8) << 4));
    }
  }
}

__constant__ float kE2M1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// Tiled layout -> BF16 row-major (the prefill path's BF16 GEMM input), one thread per output pair (NVFP4)
// or element (FP8).
__global__ void dequant_tiled_nvfp4_kernel(const uint8_t* __restrict__ tw, float scale2, __nv_bfloat16* __restrict__ out,
                                           int N, int K, const uint8_t* __restrict__ raw) {
  const int chunks = K / kChunk;
  const size_t pairs = static_cast<size_t>(N) * K / 2;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < pairs;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int n = static_cast<int>(i / (K / 2)), k = static_cast<int>(i % (K / 2)) * 2;
    const int tile = n / 16, r = n % 16, g = r % 8, h = r / 8, chunk = k / kChunk, kk = k % kChunk, t = kk / 32;
    const int slot = (h * 8 + g) * 4 + t;
    const size_t blk = static_cast<size_t>(tile) * chunks + chunk;
    const uint8_t* base = tw + blk * (raw ? kValueBytesFp4 : kTileBytesFp4);
    const uint8_t b = base[slot * 16 + (kk % 32) / 2];
    __nv_fp8_e4m3 sc;
    sc.__x = raw ? raw[blk * 128 + slot * 2 + (kk % 32) / 16] : base[1024 + slot * 2 + (kk % 32) / 16];
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

// The megakernel probe's L2 run-ahead (BACKLOG 28): the first `nc` chunks of every 16-row tile of a tiled weight blob
// into L2, one bulk prefetch per tile and region, issued before pdl_begin() since the weights never change. Each block of
// a stream GEMM reads its tiles' chunks in order from chunk 0, so a per-tile prefix is what every block reads first.
// Nothing is written: the GEMM that follows computes the same bits, only some of its loads hit L2.
__global__ void l2_prefetch_tiles_kernel(const uint8_t* __restrict__ w, int tiles, int chunks, int blk_bytes, int nc,
                                         const uint8_t* __restrict__ codes, int code_bytes) {
  const int tile = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile < tiles) {
    const uint8_t* p = w + static_cast<size_t>(tile) * chunks * blk_bytes;
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(nc * blk_bytes) : "memory");
    if (codes) {
      const uint8_t* c = codes + static_cast<size_t>(tile) * chunks * code_bytes;
      const int cb = (nc * code_bytes + 15) & ~15;  // codes of a tile start 16-byte aligned (chunks * 96)
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(c), "r"(cb) : "memory");
    }
  }
  pdl_begin();
}

// ---- The megakernel's stage 1 in the step (B32 measured in place; BACKLOG 28's record) ----
//
// The FFN of the rows path (gate|up, SiLU-multiply, the FP16 copy of the result with its per-row power-of-two scale,
// the down projection) as one persistent kernel: every block takes gate|up blocks b, b + G, ...; then the row step over
// every block (B32's distributed form: silu_mul's expression element by element, each row's maximum of |value| with an
// order-free atomic maximum, then each element to FP16 under to_half_rows' scale); then down blocks b, b + G, ....
// The tiles are the stream GEMM's bodies with the launch shapes dispatch() picks (TILES 2, PF 2, NT by M), so every
// element is the four-kernel path's bit for bit. Dependencies are counters (release/acquire); a wait past its 50 ms
// deadline prints a record and traps (a launch error, never a hang). The last block out resets the counters, so a
// captured graph needs no memset node. The down projection's FP16 input has its own buffer (xh2): the gate|up tiles
// read xh through the non-coherent path earlier in the same kernel, and a stale line must never be returned.
__device__ __forceinline__ unsigned long long mk_now() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned mk_ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void mk_red_release(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
constexpr unsigned long long kMkDeadlineNs = 50'000'000ull;

// Thread 0 waits until *c >= target, then the block goes on.
__device__ __forceinline__ void mk_wait(const unsigned* c, unsigned target) {
  if (threadIdx.x == 0) {
    const unsigned long long t0 = mk_now();
    unsigned v;
    while ((v = mk_ld_acquire(c)) < target) {
      if (mk_now() - t0 > kMkDeadlineNs) {
        printf("megakernel: block %d waited past the deadline (counter at %u of %u)\n", blockIdx.x, v, target);
        __trap();
      }
      __nanosleep(32);
    }
  }
  __syncthreads();
}
// Every thread's writes of the item are made visible, then thread 0 counts it.
__device__ __forceinline__ void mk_done(unsigned* c) {
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) mk_red_release(c, 1u);
}

struct NoWaitMk {
  __device__ __forceinline__ void operator()() const {}
};
struct FirstPdl {  // the block's first tile waits for the predecessor (PDL) after its first weight loads
  bool first;
  __device__ __forceinline__ void operator()() const {
    if (first) pdl_begin();
  }
};
struct CounterWaitMk {
  const unsigned* c;
  unsigned target;
  bool first;
  __device__ __forceinline__ void operator()() const {
    if (first) mk_wait(c, target);
  }
};

struct MkFfnArgs {
  const __half* xh;  // the FFN's input [M][H] in FP16 with its row scales (rmsnorm_half's)
  const float* xinv;
  StreamProblems up;  // gate (matrix 0) and up (matrix 1)
  int n_up;
  float *g, *u;  // their outputs [M][I]; g becomes silu(g) * u
  __half* xh2;   // the down projection's input [M][I] and its row scales
  float* xinv2;
  StreamProblems down;
  int n_down;
  int M, H, I;
  unsigned* c;  // kMkFfnWords: [0] blocks past gate|up, [1] past the SiLU pass, [2] past the FP16 pass, [3] exits,
                // [4..4 + 32) the rows' maxima (as float bits)
};

template <int NT, bool CODED>
// Two blocks per SM at NT <= 2 (the stream GEMM's own occupancy there, 128 registers), one at NT = 4.
__global__ void __launch_bounds__(kWarps * 32, NT == 4 ? 1 : 2) ffn_persistent_kernel(MkFfnArgs a) {
  const int G = gridDim.x;
  unsigned* c = a.c;
  if (static_cast<int>(blockIdx.x) >= a.n_up) pdl_begin();  // never with the grids used, kept for the rule
  for (int i = blockIdx.x; i < a.n_up; i += G) {
    __syncthreads();  // the body's shared memory is reused by the next tile
    stream_gemm_body<NT, 2, true, 2, CODED>(a.xh, a.xinv, a.M, a.up, a.H, 1, nullptr, i, 0,
                                            FirstPdl{i == static_cast<int>(blockIdx.x)});
  }
  mk_done(&c[0]);
  mk_wait(&c[0], G);
  // The row step over every block.
  const size_t n = size_t(a.M) * a.I, stride = size_t(G) * blockDim.x;
  unsigned* rowmax = c + 4;
  __shared__ unsigned smax[kMaxStreamRows];
  __shared__ float sc[kMaxStreamRows];
  if (threadIdx.x < kMaxStreamRows) smax[threadIdx.x] = 0u;
  __syncthreads();
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float x = a.g[i];
    const float v = x / (1.f + __expf(-x)) * a.u[i];  // kernels.cu's silu(g) * u
    a.g[i] = v;
    const float m = fabsf(v);
    if (!isnan(m)) atomicMax(&smax[i / a.I], __float_as_uint(m));  // fmaxf ignores NaN; a maximum has no order
  }
  __syncthreads();
  if (threadIdx.x < a.M && smax[threadIdx.x]) atomicMax(&rowmax[threadIdx.x], smax[threadIdx.x]);
  mk_done(&c[1]);
  mk_wait(&c[1], G);
  if (threadIdx.x < a.M) {
    const float mx = __uint_as_float(mk_ld_acquire(&rowmax[threadIdx.x]));
    int e = 0;
    if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);  // as to_half_rows
    sc[threadIdx.x] = ldexpf(1.f, 15 - e);
    if (blockIdx.x == 0) a.xinv2[threadIdx.x] = ldexpf(1.f, e - 15);
  }
  __syncthreads();
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
    a.xh2[i] = __float2half_rn(a.g[i] * sc[i / a.I]);
  mk_done(&c[2]);
  for (int i = blockIdx.x; i < a.n_down; i += G) {
    __syncthreads();
    stream_gemm_body<NT, 2, true, 2, CODED>(a.xh2, a.xinv2, a.M, a.down, a.I, 1, nullptr, i, 0,
                                            CounterWaitMk{&c[2], static_cast<unsigned>(G), i == static_cast<int>(blockIdx.x)});
  }
  if (static_cast<int>(blockIdx.x) >= a.n_down) mk_wait(&c[2], G);  // (no block idles with the grids used)
  // The last block out resets the counters for the next launch (ordered after this kernel's completion).
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&c[3], 1u) == static_cast<unsigned>(G) - 1u) {
    for (int k = 0; k < kMkFfnWords; ++k) c[k] = 0u;
    __threadfence();
  }
}

template <int NT, bool CODED>
int mk_resident() {
  static const int r = [] {
    int per_sm = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ffn_persistent_kernel<NT, CODED>, kWarps * 32, 0);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    return per_sm * sms;
  }();
  return r;
}

template <int NT, bool CODED>
void mk_ffn_launch(MkFfnArgs& a, cudaStream_t s) {
  static const int blocks_env = sg_env("LING_MK_BLOCKS", 0);
  const int resident = mk_resident<NT, CODED>();
  const int G = blocks_env > 0 ? std::min(blocks_env, resident) : resident;
  if (G < 1) throw std::runtime_error("ffn_persistent: no resident block");
  static bool told = false;
  if (!told) {
    told = true;
    std::fprintf(stderr, "megakernel stage 1: ffn_persistent<%d> on %d blocks (%d resident)\n", NT, G, resident);
  }
  launch_kernel("ffn_persistent", ffn_persistent_kernel<NT, CODED>, dim3(G), dim3(kWarps * 32), 0, s, a);
}

// ---- The megakernel's stage 2 in the step: a DeltaNet layer's mixer as one persistent kernel (LING_MK_GDN) ----
//
// One verify sequence's DeltaNet mixer from the input norm's output to the out-projection: in_qkv|in_z (FP8 stream
// GEMMs), the a/b projections (BF16) with the gating, the causal conv (from the session's window, left unwritten), the
// gated delta rule (from the session's state, left unwritten), the gated RMSNorm, the FP16 copy of its output under each
// row's power-of-two scale, and the out-projection. The point is the overlap the graph path cannot have: blocks
// [0, chain) are the chain's (the a/b projections and the gating at once, then, as soon as the last in_qkv tile is in,
// the conv and the recurrence), while the other blocks stream the in_qkv and then the in_z tiles beside them; the
// chain's blocks join the in_z queue when they are done. Every element is computed by the four-kernel path's arithmetic
// in its order (the stream GEMM's bodies; bf16_rows', gdn_conv's, gdn_gating's, gdn_recurrent_body's and
// gated_rmsnorm's expressions, reductions and orders reproduced per 128-thread group or per warp), so the outputs are
// bit for bit the graph path's; only which block computes an item, and when, changes. One block per SM (the recurrence
// holds 128 state values per thread).
enum MkGdnCounter { kQkvQueue = 0, kQkvDone, kZQueue, kZDone, kAbDone, kConvDone, kRecDone, kNormDone, kHalfDone, kExit };
constexpr int kGdnK = 128;  // the key and value head size (kernels.cu's kGdnDk, kGdnDv)

struct MkGdnArgs {
  const __half* xh;  // the input norm's FP16 copy and row scales (the stream GEMMs' input)
  const float* xinv;
  const float* xn;  // its FP32 output (the a/b projections' input)
  StreamProblems inp;  // in_qkv (matrix 0) and in_z (matrix 1)
  int n_qkv, n_z;      // their tiles
  const __nv_bfloat16 *wa, *wb;
  float *a, *b;
  const __nv_bfloat16 *A_log, *dt_bias;
  float *g, *beta;
  const float* vpre;  // in_qkv's output [M][C] (inp.y[0]; kept for the commit)
  float* mixed;       // the conv's output [M][C]
  const float* conv_state;  // [C][3], read only
  const __nv_bfloat16* conv_w;
  const float* state;  // [HV][128][128], read only
  float* core;         // [M][HV * 128]
  const float* z;      // in_z's output [M][Vs] (inp.y[1])
  const __nv_bfloat16* norm_w;
  float eps;
  float* normed;
  __half* xh2;
  float* xinv2;
  StreamProblems outp;
  int n_out;
  int M, K, C, H, HV, Vs;
  int chain;
  unsigned* c;
};

__device__ __forceinline__ float mk_silu(float x) { return x / (1.f + __expf(-x)); }  // kernels.cu's silu
__device__ __forceinline__ float mk_warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// bf16_rows_kernel's output n for every row, by one group of 128 threads (gt its thread, red its [4][32]).
__device__ __forceinline__ void mk_bf16_row(const float* __restrict__ x, int M, const __nv_bfloat16* __restrict__ w,
                                            float* __restrict__ y, int n, int N, int K, int gt, float (*red)[32]) {
  constexpr int MT = 32, T = 128;
  const int lane = gt & 31, warp = gt >> 5;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;
  const __nv_bfloat16* wr = w + static_cast<size_t>(n) * K;
  for (int k0 = gt * 8; k0 < K; k0 += T * 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(wr + k0);
    const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float wf[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(w2[i]);
      wf[2 * i] = f.x;
      wf[2 * i + 1] = f.y;
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      if (m >= M) break;
      const float4 xa = *reinterpret_cast<const float4*>(x + static_cast<size_t>(m) * K + k0);
      const float4 xb = *reinterpret_cast<const float4*>(x + static_cast<size_t>(m) * K + k0 + 4);
      const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
      float v = acc[m];
#pragma unroll
      for (int i = 0; i < 8; ++i) v = fmaf(wf[i], xv[i], v);
      acc[m] = v;
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m >= M) break;
    const float v = mk_warp_sum(acc[m]);
    if (lane == 0) red[warp][m] = v;
  }
  __syncthreads();  // both groups of the block come here together
  if (gt < M) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < T / 32; ++i) v += red[i][gt];
    y[static_cast<size_t>(gt) * N + n] = v;
  }
  __syncthreads();  // red is reused by the group's next output
}

// gdn_recurrent_body for item bx (head bx / 4, value columns (bx % 4) * 32 ...) by one warp; qs and ks are the warp's.
__device__ __forceinline__ void mk_recurrent(const float* mixed, const float* g, const float* beta,
                                             const float* __restrict__ state_in, float* out, int M, int H, int HV, int bx,
                                             float* qs, float* ks) {
  const int hv = bx / 4, part = bx % 4, lane = threadIdx.x & 31, v = part * 32 + lane;
  const int h = hv / (HV / H);
  const int C = 2 * H * kGdnK + HV * kGdnK;
  const float* st = state_in + static_cast<size_t>(hv) * kGdnK * kGdnK;
  float S[kGdnK];
#pragma unroll
  for (int k = 0; k < kGdnK; ++k) S[k] = st[k * kGdnK + v];
  const float scale = rsqrtf(static_cast<float>(kGdnK));
  for (int t = 0; t < M; ++t) {
    const float* row = mixed + static_cast<size_t>(t) * C;
    const float4 q4 = *reinterpret_cast<const float4*>(row + h * kGdnK + lane * 4);
    const float4 k4 = *reinterpret_cast<const float4*>(row + H * kGdnK + h * kGdnK + lane * 4);
    float qn = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
    float kn = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      qn += __shfl_xor_sync(0xffffffffu, qn, o);
      kn += __shfl_xor_sync(0xffffffffu, kn, o);
    }
    const float qr = rsqrtf(qn + 1e-6f) * scale, kr = rsqrtf(kn + 1e-6f);
    __syncwarp();
    qs[lane * 4] = q4.x * qr, qs[lane * 4 + 1] = q4.y * qr, qs[lane * 4 + 2] = q4.z * qr, qs[lane * 4 + 3] = q4.w * qr;
    ks[lane * 4] = k4.x * kr, ks[lane * 4 + 1] = k4.y * kr, ks[lane * 4 + 2] = k4.z * kr, ks[lane * 4 + 3] = k4.w * kr;
    __syncwarp();
    const float decay = __expf(g[t * HV + hv]);
    const float b = beta[t * HV + hv];
    const float vv = row[2 * H * kGdnK + hv * kGdnK + v];
    float kv = 0.f;
#pragma unroll
    for (int k = 0; k < kGdnK; ++k) {
      S[k] *= decay;
      kv = fmaf(S[k], ks[k], kv);
    }
    const float d = (vv - kv) * b;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < kGdnK; ++k) {
      S[k] = fmaf(d, ks[k], S[k]);
      o = fmaf(S[k], qs[k], o);
    }
    out[static_cast<size_t>(t) * HV * kGdnK + hv * kGdnK + v] = o;
  }
  __syncwarp();  // qs and ks are reused by the warp's next item
}

// A block's next item from a queue (thread 0 takes it, every thread gets it).
__device__ __forceinline__ int mk_take(unsigned* q) {
  __shared__ int item;
  __syncthreads();
  if (threadIdx.x == 0) item = static_cast<int>(atomicAdd(q, 1u));
  __syncthreads();
  return item;
}

struct MaybePdl {  // pdl_begin() once per block, at the first read of the predecessor's output
  bool* waited;
  __device__ __forceinline__ void operator()() const {
    if (!*waited) {
      pdl_begin();
      *waited = true;
    }
  }
};

template <int NT>
__global__ void __launch_bounds__(kWarps * 32, 1) gdn_persistent_kernel(MkGdnArgs a) {
  const int G = gridDim.x, bid = blockIdx.x;
  unsigned* c = a.c;
  const bool chain = bid < a.chain;
  bool waited = false;
  __shared__ float red[2][4][32];
  __shared__ float qk[kWarps][2][kGdnK];
  const int grp = threadIdx.x >> 7, gt = threadIdx.x & 127;
  if (chain) {
    MaybePdl{&waited}();
    // The a/b projections, each output n of both matrices, then the gating of (t, n) for every row t: the chain's
    // 2 x chain groups take n = group, group + 2 x chain, ...
    for (int n = bid * 2 + grp; n < a.HV; n += 2 * a.chain) {
      mk_bf16_row(a.xn, a.M, a.wa, a.a, n, a.HV, a.K, gt, red[grp]);
      mk_bf16_row(a.xn, a.M, a.wb, a.b, n, a.HV, a.K, gt, red[grp]);
      if (gt < a.M) {  // gdn_gating_kernel's element i = gt * HV + n
        const int i = gt * a.HV + n;
        const float x = a.a[i] + __bfloat162float(a.dt_bias[n]);
        const float softplus = x <= 20.f ? log1pf(__expf(x)) : x;
        a.g[i] = -__expf(__bfloat162float(a.A_log[n])) * softplus;
        a.beta[i] = 1.f / (1.f + __expf(-a.b[i]));
      }
    }
    // Every group runs the same number of rounds of the loop above (its barriers are block-wide): HV is a multiple of
    // 2 x chain here (checked on the host).
    mk_done(&c[kAbDone]);
  }
  // The in_qkv tiles, from a queue, on every block.
  for (;;) {
    const int i = mk_take(&c[kQkvQueue]);
    if (i >= a.n_qkv) break;
    stream_gemm_body<NT, 2, false, 2, false>(a.xh, a.xinv, a.M, a.inp, a.K, 1, nullptr, i, 0, MaybePdl{&waited});
    mk_done(&c[kQkvDone]);
  }
  if (chain) {
    mk_wait(&c[kQkvDone], a.n_qkv);
    // gdn_conv_kernel's channel c over the rows, from the session's window, out of place (vpre -> mixed).
    for (int ch = bid * blockDim.x + threadIdx.x; ch < a.C; ch += a.chain * blockDim.x) {
      float s0 = a.conv_state[ch * 3], s1 = a.conv_state[ch * 3 + 1], s2 = a.conv_state[ch * 3 + 2];
      const float w0 = __bfloat162float(a.conv_w[ch * 4]), w1 = __bfloat162float(a.conv_w[ch * 4 + 1]);
      const float w2 = __bfloat162float(a.conv_w[ch * 4 + 2]), w3 = __bfloat162float(a.conv_w[ch * 4 + 3]);
      for (int t = 0; t < a.M; ++t) {
        const float xv = a.vpre[static_cast<size_t>(t) * a.C + ch];
        const float o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv;
        s0 = s1;
        s1 = s2;
        s2 = xv;
        a.mixed[static_cast<size_t>(t) * a.C + ch] = mk_silu(o);
      }
    }
    mk_done(&c[kConvDone]);
    mk_wait(&c[kConvDone], a.chain);
    mk_wait(&c[kAbDone], a.chain);
    // The recurrence: item w of HV x 4 on warp w of the chain's blocks.
    const int warp = threadIdx.x >> 5;
    for (int it = bid * kWarps + warp; it < a.HV * 4; it += a.chain * kWarps)
      mk_recurrent(a.mixed, a.g, a.beta, a.state, a.core, a.M, a.H, a.HV, it, qk[warp][0], qk[warp][1]);
    mk_done(&c[kRecDone]);
  }
  // The in_z tiles, from a queue: the streaming blocks at once, the chain's when the recurrence is done.
  for (;;) {
    const int j = mk_take(&c[kZQueue]);
    if (j >= a.n_z) break;
    stream_gemm_body<NT, 2, false, 2, false>(a.xh, a.xinv, a.M, a.inp, a.K, 1, nullptr, a.n_qkv + j, 0,
                                             MaybePdl{&waited});
    mk_done(&c[kZDone]);
  }
  MaybePdl{&waited}();
  mk_wait(&c[kZDone], a.n_z);
  mk_wait(&c[kRecDone], a.chain);
  // gated_rmsnorm_kernel's row r (D = 128 threads, block_sum over its 4 warps) by group grp, rows r = 2 bid + grp,
  // + 2G, ...; each row's maximum of |output| into the row maxima; then the FP16 copy of the group's own rows.
  unsigned* rowmax = c + 12;
  const int rows = a.M * a.HV;
  for (int r0 = 0; r0 < rows; r0 += 2 * G) {  // every group runs the same rounds (the barriers are block-wide)
    const int r = r0 + 2 * bid + grp;
    const bool live = r < rows;
    const float x = live ? a.core[static_cast<size_t>(r) * kGdnK + gt] : 0.f;
    float ss = x * x;
    // block_sum over the row's 128 threads, exactly: the warp sums, then lanes < 4 of a warp_sum over them.
    ss = mk_warp_sum(ss);
    __syncthreads();
    if ((gt & 31) == 0) red[grp][0][gt >> 5] = ss;
    __syncthreads();
    const float tsum = mk_warp_sum((gt & 31) < 4 ? red[grp][0][gt & 31] : 0.f);
    if (live) {
      const float rstd = rsqrtf(tsum / kGdnK + a.eps);
      const float o = x * rstd * __bfloat162float(a.norm_w[gt]) * mk_silu(a.z[static_cast<size_t>(r) * kGdnK + gt]);
      a.normed[static_cast<size_t>(r) * kGdnK + gt] = o;
      const float m = fabsf(o);
      // the row of the out-projection's input is t = r / HV
      float wm = isnan(m) ? 0.f : m;
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) wm = fmaxf(wm, __shfl_xor_sync(0xffffffffu, wm, s));
      if ((gt & 31) == 0 && wm > 0.f) atomicMax(&rowmax[r / a.HV], __float_as_uint(wm));
    }
  }
  mk_done(&c[kNormDone]);
  mk_wait(&c[kNormDone], G);
  __shared__ float sc[kMaxStreamRows];
  if (threadIdx.x < a.M) {
    const float mx = __uint_as_float(mk_ld_acquire(&rowmax[threadIdx.x]));
    int e = 0;
    if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);  // as to_half_rows
    sc[threadIdx.x] = ldexpf(1.f, 15 - e);
    if (bid == 0) a.xinv2[threadIdx.x] = ldexpf(1.f, e - 15);
  }
  __syncthreads();
  for (int r = 2 * bid + grp; r < rows; r += 2 * G) {
    const size_t i = static_cast<size_t>(r) * kGdnK + gt;  // = t * Vs + hv * 128 + d
    a.xh2[i] = __float2half_rn(a.normed[i] * sc[r / a.HV]);
  }
  mk_done(&c[kHalfDone]);
  // The out-projection, b, b + G, ... after every block's FP16 rows.
  for (int i = bid; i < a.n_out; i += G) {
    __syncthreads();
    stream_gemm_body<NT, 2, false, 2, false>(a.xh2, a.xinv2, a.M, a.outp, a.Vs, 1, nullptr, i, 0,
                                             CounterWaitMk{&c[kHalfDone], static_cast<unsigned>(G), i == bid});
  }
  if (bid >= a.n_out) mk_wait(&c[kHalfDone], G);
  __syncthreads();
  if (threadIdx.x == 0 && atomicAdd(&c[kExit], 1u) == static_cast<unsigned>(G) - 1u) {
    for (int k = 0; k < kMkGdnWords; ++k) c[k] = 0u;
    __threadfence();
  }
}

template <int NT>
int mk_gdn_resident() {
  static const int r = [] {
    int per_sm = 0, sms = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gdn_persistent_kernel<NT>, kWarps * 32, 0);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    return per_sm * sms;
  }();
  return r;
}

template <int NT>
void mk_gdn_launch(MkGdnArgs& a, cudaStream_t s) {
  const int G = mk_gdn_resident<NT>();
  if (G < 2 * a.chain || G < 1) throw std::runtime_error("gdn_persistent: too few resident blocks");
  static bool told = false;
  if (!told) {
    told = true;
    std::fprintf(stderr, "megakernel stage 2: gdn_persistent<%d> on %d blocks, %d of them the chain's\n", NT, G, a.chain);
  }
  launch_kernel("gdn_persistent", gdn_persistent_kernel<NT>, dim3(G), dim3(kWarps * 32), 0, s, a);
}

int grid_of(size_t n) { return static_cast<int>(std::min<size_t>((n + 255) / 256, 65535 * 4)); }

void check_shape(int N, int K, const char* what) {
  if (N % 16 != 0 || K % kChunk != 0) throw std::runtime_error(std::string(what) + ": N % 16 and K % 128 must be 0");
}

}  // namespace

size_t tiled_bytes_nvfp4(int N, int K) { return static_cast<size_t>(N) * K / 2 + static_cast<size_t>(N) * K / 16; }
size_t tiled_bytes_fp8(int N, int K) { return static_cast<size_t>(N) * K; }
size_t tiled_bytes_nvfp4_coded(int N, int K) {
  return static_cast<size_t>(N / 16) * (K / kChunk) * (kValueBytesFp4 + kCodeBytesFp4);
}

bool scale_code_table(const uint8_t* scales, size_t n, uint8_t (&table)[kScaleCodes], uint8_t (&code_of)[256]) {
  size_t count[256] = {};
  for (size_t i = 0; i < n; ++i) ++count[scales[i]];
  int distinct = 0;
  for (int v = 0; v < 256; ++v) distinct += count[v] > 0;
  if (distinct > kScaleCodes) return false;
  for (int i = 0; i < kScaleCodes; ++i) table[i] = 0;
  for (int v = 0; v < 256; ++v) code_of[v] = 0xff;
  int next = 0;
  for (int v = 0; v < 256; ++v)
    if (count[v]) {
      code_of[v] = static_cast<uint8_t>(next);
      table[next++] = static_cast<uint8_t>(v);
    }
  return true;
}

void retile_nvfp4_coded(const uint8_t* w, const uint8_t* wscale, const uint8_t* code_of_dev, uint8_t* out,
                        uint8_t* raw_scales, int N, int K, cudaStream_t s) {
  check_shape(N, K, "retile_nvfp4_coded");
  cudaMemsetAsync(out, 0, tiled_bytes_nvfp4_coded(N, K), s);
  retile_nvfp4_coded_values_kernel<<<grid_of(static_cast<size_t>(N / 16) * (K / kChunk) * 64), 256, 0, s>>>(w, out, N, K);
  retile_nvfp4_coded_scales_kernel<<<grid_of(static_cast<size_t>(N / 16) * (K / kChunk)), 256, 0, s>>>(
      wscale, code_of_dev, out, raw_scales, N, K);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("retile_nvfp4_coded: ") + cudaGetErrorString(e));
}

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

void dequant_tiled_nvfp4(const uint8_t* tw, float scale2, __nv_bfloat16* out, int N, int K, cudaStream_t s,
                         const uint8_t* raw_scales) {
  dequant_tiled_nvfp4_kernel<<<grid_of(static_cast<size_t>(N) * K / 2), 256, 0, s>>>(tw, scale2, out, N, K, raw_scales);
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

void l2_prefetch_weights(const uint8_t* w, bool fp4, bool coded, int N, int K, size_t bytes, cudaStream_t s) {
  if (!w || bytes == 0) return;
  const int tiles = N / 16, chunks = K / kChunk;
  const int blk = fp4 ? (coded ? kValueBytesFp4 : kTileBytesFp4) : kTileBytesFp8;
  const size_t per_chunk = static_cast<size_t>(tiles) * (blk + (coded ? kCodeBytesFp4 : 0));
  const int nc = static_cast<int>(std::min<size_t>(chunks, std::max<size_t>(1, bytes / per_chunk)));
  const uint8_t* codes = coded ? w + static_cast<size_t>(tiles) * chunks * kValueBytesFp4 : nullptr;
  launch_kernel("l2_prefetch", l2_prefetch_tiles_kernel, dim3((tiles + 127) / 128), dim3(128), 0, s, w, tiles, chunks, blk,
                nc, codes, coded ? kCodeBytesFp4 : 0);
}

void to_half_rows(const float* x, int M, int K, __half* out, float* xinv, cudaStream_t s) {
  launch_kernel("to_half_rows", to_half_rows_kernel, dim3(M), dim3(256), 0, s, x, K, out, xinv);
}

void stream_gemm_nvfp4(const __half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
                       float scale2, float* y, int N, int K, cudaStream_t s) {
  const StreamTarget t{w, wscale, scale2, y, N};  // wscale: the code table of a coded blob, or nullptr
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

bool gdn_persistent(const MkGdnParams& p, cudaStream_t s) {
  static const int chain_env = sg_env("LING_MK_CHAIN", 24);
  const int chain = chain_env;
  if (p.M < 1 || p.M > kMaxStreamRows || p.Dk != kGdnK || p.Dv != kGdnK || p.HV % p.H != 0) return false;
  if (chain < 1 || p.HV % (2 * chain) != 0 || p.K % kChunk || p.Vs % kChunk) return false;
  if (p.in_qkv.N != p.C || p.in_z.N != p.Vs || p.out.N != p.K || p.C % 32 || p.Vs % 32 || p.K % 32) return false;
  if (sg_env("LING_SG_PF", kInFlight) != 2 || sg_env("LING_SG_TW", kTilesWide) != 2 || sg_env("LING_SG_TN", kTilesNarrow) != 2 ||
      sg_env("LING_SG_TW4", kTilesWide4) != 2 || sg_env("LING_SG_TN4", kTilesNarrow4) != 2 || sg_env("LING_KSPLIT", 0) != 0)
    return false;
  MkGdnArgs a{};
  a.xh = p.xh;
  a.xinv = p.xinv;
  a.xn = p.xn;
  const StreamTarget* it[2] = {&p.in_qkv, &p.in_z};
  int blocks = 0;
  for (int i = 0; i < 2; ++i) {
    a.inp.w[i] = it[i]->w;
    a.inp.gscale[i] = it[i]->scale;
    a.inp.y[i] = it[i]->y;
    a.inp.N[i] = it[i]->N;
    a.inp.table[i] = nullptr;
    a.inp.block0[i] = blocks;
    blocks += it[i]->N / 32;
  }
  a.inp.count = 2;
  a.n_qkv = p.C / 32;
  a.n_z = p.Vs / 32;
  a.wa = p.wa;
  a.wb = p.wb;
  a.a = p.a;
  a.b = p.b;
  a.A_log = p.A_log;
  a.dt_bias = p.dt_bias;
  a.g = p.g;
  a.beta = p.beta;
  a.vpre = p.in_qkv.y;
  a.mixed = p.mixed;
  a.conv_state = p.conv_state;
  a.conv_w = p.conv_w;
  a.state = p.state;
  a.core = p.core;
  a.z = p.in_z.y;
  a.norm_w = p.norm_w;
  a.eps = p.eps;
  a.normed = p.normed;
  a.xh2 = p.xh2;
  a.xinv2 = p.xinv2;
  a.outp.w[0] = p.out.w;
  a.outp.gscale[0] = p.out.scale;
  a.outp.y[0] = p.out.y;
  a.outp.N[0] = p.out.N;
  a.outp.table[0] = nullptr;
  a.outp.block0[0] = 0;
  a.outp.count = 1;
  a.n_out = p.out.N / 32;
  a.M = p.M;
  a.K = p.K;
  a.C = p.C;
  a.H = p.H;
  a.HV = p.HV;
  a.Vs = p.Vs;
  a.chain = chain;
  a.c = p.counters;
  if (p.M <= 8) mk_gdn_launch<1>(a, s);
  else if (p.M <= 16) mk_gdn_launch<2>(a, s);
  else mk_gdn_launch<4>(a, s);
  return true;
}

bool ffn_persistent(const __half* xh, const float* xinv, int M, const StreamTarget& gate, const StreamTarget& up,
                    const StreamTarget& down, int H, int I, float* g, float* u, __half* xh2, float* xinv2,
                    unsigned* counters, cudaStream_t s) {
  const bool coded = gate.wscale != nullptr;
  // Only the shapes dispatch() gives the four-kernel path: TILES 2 and PF 2 everywhere, one kind of blob.
  if ((up.wscale != nullptr) != coded || (down.wscale != nullptr) != coded) return false;
  if (M < 1 || M > kMaxStreamRows || H % kChunk || I % kChunk || gate.N != I || up.N != I || down.N != H) return false;
  if (sg_env("LING_SG_PF", kInFlight) != 2 || sg_env("LING_SG_TW", kTilesWide) != 2 || sg_env("LING_SG_TN", kTilesNarrow) != 2 ||
      sg_env("LING_SG_TW4", kTilesWide4) != 2 || sg_env("LING_SG_TN4", kTilesNarrow4) != 2 || sg_env("LING_KSPLIT", 0) != 0)
    return false;
  MkFfnArgs a{};
  a.xh = xh;
  a.xinv = xinv;
  const StreamTarget* ut[2] = {&gate, &up};
  int blocks = 0;
  for (int i = 0; i < 2; ++i) {
    a.up.w[i] = ut[i]->w;
    a.up.gscale[i] = ut[i]->scale;
    a.up.y[i] = i == 0 ? g : u;
    a.up.N[i] = ut[i]->N;
    a.up.table[i] = ut[i]->wscale;
    a.up.block0[i] = blocks;
    blocks += ut[i]->N / 32;
  }
  a.up.count = 2;
  a.n_up = blocks;
  a.g = g;
  a.u = u;
  a.xh2 = xh2;
  a.xinv2 = xinv2;
  a.down.w[0] = down.w;
  a.down.gscale[0] = down.scale;
  a.down.y[0] = down.y;
  a.down.N[0] = down.N;
  a.down.table[0] = down.wscale;
  a.down.block0[0] = 0;
  a.down.count = 1;
  a.n_down = down.N / 32;
  a.M = M;
  a.H = H;
  a.I = I;
  a.c = counters;
  if (M <= 8) coded ? mk_ffn_launch<1, true>(a, s) : mk_ffn_launch<1, false>(a, s);
  else if (M <= 16) coded ? mk_ffn_launch<2, true>(a, s) : mk_ffn_launch<2, false>(a, s);
  else coded ? mk_ffn_launch<4, true>(a, s) : mk_ffn_launch<4, false>(a, s);
  return true;
}

}  // namespace ling::kernels
