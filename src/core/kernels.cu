#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

#define LING_LAUNCH_CHECK(name) check(cudaGetLastError(), name)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Sum over the block; every thread gets the result. `red` holds at least 32 floats.
__device__ float block_sum(float v, float* red) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = lane < warps ? red[lane] : 0.f;
  return warp_sum(t);
}

__device__ __forceinline__ float fp8_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}

__device__ __forceinline__ float2 fp8x2_to_float2(uint16_t pair) {
  __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
  return __half22float2(*reinterpret_cast<__half2*>(&h));
}

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

__constant__ float kFp4Values[16] = {0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
                                     -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

constexpr int kGemvWarps = 8;

// ---- NVFP4 weight-streaming GEMV: one warp per output row, tiles of 1024 values of K. ----
constexpr int kFp4Tile = 1024;
constexpr int kFp4Stride = kFp4Tile + kFp4Tile / 32;  // one pad float per 32: no bank conflicts

template <int MT, int R>
__global__ void __launch_bounds__(kGemvWarps * 32)
    gemv_nvfp4_kernel(const float* __restrict__ x, int M, const uint8_t* __restrict__ w,
                      const uint8_t* __restrict__ ws, float scale2, float* __restrict__ y, int N, int K) {
  __shared__ float xs[MT * kFp4Stride];
  __shared__ float lut[16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n0 = (blockIdx.x * kGemvWarps + warp) * R;  // this warp's first row; it owns R rows
  if (threadIdx.x < 16) lut[threadIdx.x] = kFp4Values[threadIdx.x];
  float acc[R][MT];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

  for (int t0 = 0; t0 < K; t0 += kFp4Tile) {
    __syncthreads();
    for (int i = threadIdx.x; i < MT * kFp4Tile; i += blockDim.x) {
      int m = i / kFp4Tile, k = i % kFp4Tile;
      xs[m * kFp4Stride + k + (k >> 5)] = m < M ? x[static_cast<size_t>(m) * K + t0 + k] : 0.f;
    }
    __syncthreads();
    const int k0 = t0 + lane * 32;
    uint4 packed[R];
    uint16_t sc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {  // issue every row's loads before using any of them
      const int n = min(n0 + r, N - 1);
      packed[r] = __ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(n) * (K / 2) + k0 / 2));
      sc[r] = __ldg(reinterpret_cast<const uint16_t*>(ws + static_cast<size_t>(n) * (K / 16) + k0 / 16));
    }
    const int base = lane * 33;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float s0 = fp8_to_float(sc[r] & 0xff), s1 = fp8_to_float(sc[r] >> 8);
      const uint32_t words[4] = {packed[r].x, packed[r].y, packed[r].z, packed[r].w};
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        const float s = wd < 2 ? s0 : s1;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
          const float wv = lut[(words[wd] >> (4 * b)) & 0xf] * s;
          const int kk = base + wd * 8 + b;
#pragma unroll
          for (int m = 0; m < MT; ++m) acc[r][m] = fmaf(wv, xs[m * kFp4Stride + kk], acc[r][m]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float v = warp_sum(acc[r][m]);
      if (lane == 0 && n0 + r < N && m < M) y[static_cast<size_t>(m) * N + n0 + r] = v * scale2;
    }
}

// ---- FP8 weight-streaming GEMV: tiles of 512 values, 16 per lane. ----
constexpr int kFp8Tile = 512;
constexpr int kFp8Stride = kFp8Tile + kFp8Tile / 32;

template <int MT, int R>
__global__ void __launch_bounds__(kGemvWarps * 32)
    gemv_fp8_kernel(const float* __restrict__ x, int M, const uint8_t* __restrict__ w, float wscale,
                    float* __restrict__ y, int N, int K) {
  __shared__ float xs[MT * kFp8Stride];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n0 = (blockIdx.x * kGemvWarps + warp) * R;
  float acc[R][MT];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

  for (int t0 = 0; t0 < K; t0 += kFp8Tile) {
    __syncthreads();
    for (int i = threadIdx.x; i < MT * kFp8Tile; i += blockDim.x) {
      int m = i / kFp8Tile, k = i % kFp8Tile;
      xs[m * kFp8Stride + k + (k >> 5)] = m < M ? x[static_cast<size_t>(m) * K + t0 + k] : 0.f;
    }
    __syncthreads();
    const int k0 = t0 + lane * 16;
    uint4 packed[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int n = min(n0 + r, N - 1);
      packed[r] = __ldg(reinterpret_cast<const uint4*>(w + static_cast<size_t>(n) * K + k0));
    }
    const int base = lane * 16 + (lane >> 1);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t words[4] = {packed[r].x, packed[r].y, packed[r].z, packed[r].w};
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float2 f = fp8x2_to_float2(static_cast<uint16_t>(words[wd] >> (16 * h)));
          const int kk = base + wd * 4 + h * 2;
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            acc[r][m] = fmaf(f.x, xs[m * kFp8Stride + kk], acc[r][m]);
            acc[r][m] = fmaf(f.y, xs[m * kFp8Stride + kk + 1], acc[r][m]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      float v = warp_sum(acc[r][m]);
      if (lane == 0 && n0 + r < N && m < M) y[static_cast<size_t>(m) * N + n0 + r] = v * wscale;
    }
}

__global__ void gemv_bf16_kernel(const float* __restrict__ x, int M, const __nv_bfloat16* __restrict__ w,
                                 float* __restrict__ y, int N, int K) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n = blockIdx.x * kGemvWarps + warp;
  if (n >= N) return;
  for (int m = 0; m < M; ++m) {
    float acc = 0.f;
    for (int k = lane; k < K; k += 32)
      acc = fmaf(__bfloat162float(w[static_cast<size_t>(n) * K + k]), x[static_cast<size_t>(m) * K + k], acc);
    acc = warp_sum(acc);
    if (lane == 0) y[static_cast<size_t>(m) * N + n] = acc;
  }
}

// Narrow BF16 matrices (the DeltaNet beta/alpha projections, N = 48) for up to 32 rows: one block per
// output, each thread a fixed slice of K for every row, then a fixed-order reduction, so the weights are
// read once and a row's result does not depend on M.
constexpr int kBf16RowsThreads = 128;

__device__ __forceinline__ void load8(const float* p, float (&v)[8]) {
  const float4 a = *reinterpret_cast<const float4*>(p), b = *reinterpret_cast<const float4*>(p + 4);
  v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
}
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[8]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

template <typename XT>  // the input: FP32 activations, or BF16 (the drafter's target features)
__global__ void __launch_bounds__(kBf16RowsThreads)
    bf16_rows_kernel(const XT* __restrict__ x, int M, const __nv_bfloat16* __restrict__ w0, float* __restrict__ y0,
                     const __nv_bfloat16* __restrict__ w1, float* __restrict__ y1, int N, int K) {
  constexpr int MT = 32;
  __shared__ float red[kBf16RowsThreads / 32][MT];
  // Blocks [0, N) take the first matrix, [N, 2N) the second (when given).
  const bool second = static_cast<int>(blockIdx.x) >= N;
  const __nv_bfloat16* __restrict__ w = second ? w1 : w0;
  float* __restrict__ y = second ? y1 : y0;
  const int n = blockIdx.x - (second ? N : 0), t = threadIdx.x, lane = t & 31, warp = t >> 5;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;
  const __nv_bfloat16* wr = w + static_cast<size_t>(n) * K;
  // PDL prologue (launch.cuh): the weights never change, so this thread's first 16 bytes of them are
  // requested before waiting for the kernel that writes x.
  if (t * 8 < K) asm volatile("prefetch.global.L2 [%0];" ::"l"(wr + t * 8));
  pdl_begin();
  for (int k0 = t * 8; k0 < K; k0 += kBf16RowsThreads * 8) {
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
      float xv[8];
      load8(x + static_cast<size_t>(m) * K + k0, xv);
      float v = acc[m];
#pragma unroll
      for (int i = 0; i < 8; ++i) v = fmaf(wf[i], xv[i], v);
      acc[m] = v;
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    if (m >= M) break;
    const float v = warp_sum(acc[m]);
    if (lane == 0) red[warp][m] = v;
  }
  __syncthreads();
  if (t < M) {
    float v = 0.f;
#pragma unroll
    for (int i = 0; i < kBf16RowsThreads / 32; ++i) v += red[i][t];
    y[static_cast<size_t>(t) * N + n] = v;
  }
}

__global__ void dequant_nvfp4_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                                     float scale2, __nv_bfloat16* __restrict__ out, size_t total_bytes,
                                     int K) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total_bytes;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t row = i / (K / 2), col = (i % (K / 2)) * 2;
    const float s = fp8_to_float(ws[row * (K / 16) + col / 16]) * scale2;
    const uint8_t b = w[i];
    out[row * K + col] = __float2bfloat16(kFp4Values[b & 0xf] * s);
    out[row * K + col + 1] = __float2bfloat16(kFp4Values[b >> 4] * s);
  }
}

__global__ void dequant_fp8_kernel(const uint8_t* __restrict__ w, float wscale, __nv_bfloat16* __restrict__ out,
                                   size_t total) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<size_t>(gridDim.x) * blockDim.x)
    out[i] = __float2bfloat16(fp8_to_float(w[i]) * wscale);
}

__global__ void to_bf16_kernel(const float* __restrict__ x, __nv_bfloat16* __restrict__ out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = __float2bfloat16(x[i]);
}

__global__ void embed_kernel(const __nv_bfloat16* __restrict__ table, const int* __restrict__ ids, int H,
                             float* __restrict__ out) {
  pdl_begin();  // first statement: see launch.cuh
  const int m = blockIdx.x;
  const __nv_bfloat16* row = table + static_cast<size_t>(ids[m]) * H;
  for (int h = threadIdx.x; h < H; h += blockDim.x) out[static_cast<size_t>(m) * H + h] = __bfloat162float(row[h]);
}

// Up to 1024 threads: the bound keeps a Debug (-G) build within the register file at that size.
__global__ void __launch_bounds__(1024) rmsnorm_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                               float* __restrict__ out, int H, float eps, bool gemma) {
  pdl_begin();  // first statement: see launch.cuh
  __shared__ float red[32];
  const float* row = x + static_cast<size_t>(blockIdx.x) * H;
  float ss = 0.f;
  for (int h = threadIdx.x; h < H; h += blockDim.x) ss += row[h] * row[h];
  const float rstd = rsqrtf(block_sum(ss, red) / H + eps);
  for (int h = threadIdx.x; h < H; h += blockDim.x) {
    float wv = __bfloat162float(w[h]);
    out[static_cast<size_t>(blockIdx.x) * H + h] = row[h] * rstd * (gemma ? 1.f + wv : wv);
  }
}

__global__ void add_kernel(float* h, const float* d, int n) {
  pdl_begin();  // first statement: see launch.cuh
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) h[i] += d[i];
}

__global__ void silu_mul_kernel(const float* g, const float* u, float* out, int n) {
  pdl_begin();  // first statement: see launch.cuh
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = silu(g[i]) * u[i];
}

__global__ void sigmoid_mul_kernel(float* x, const float* gate, int n) {
  pdl_begin();  // first statement: see launch.cuh
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    x[i] *= 1.f / (1.f + __expf(-gate[i]));
}

__global__ void gdn_conv_kernel(float* mixed, float* state, const __nv_bfloat16* __restrict__ w, int M, int C) {
  pdl_begin();  // first statement: see launch.cuh
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s0 = state[c * 3], s1 = state[c * 3 + 1], s2 = state[c * 3 + 2];
  const float w0 = __bfloat162float(w[c * 4]), w1 = __bfloat162float(w[c * 4 + 1]);
  const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
  for (int t = 0; t < M; ++t) {
    const float xv = mixed[static_cast<size_t>(t) * C + c];
    const float o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv;
    s0 = s1;
    s1 = s2;
    s2 = xv;
    mixed[static_cast<size_t>(t) * C + c] = silu(o);
  }
  state[c * 3] = s0;
  state[c * 3 + 1] = s1;
  state[c * 3 + 2] = s2;
}

// The verify's conv over several sequences (M4 stage 4): sequence blockIdx.y's rows [row0, row0 + rows) of mixed,
// starting from its own window, which is read and left unwritten (a verify keeps no state). Each output is the
// same arithmetic as gdn_conv_kernel's over that sequence's rows alone.
__global__ void gdn_conv_seqs_kernel(float* mixed, SeqPtrs<const float> state, const __nv_bfloat16* __restrict__ w,
                                     SeqBatch b, int C) {
  pdl_begin();  // first statement: see launch.cuh
  const int c = blockIdx.x * blockDim.x + threadIdx.x, sq = blockIdx.y;
  if (c >= C) return;
  const float* st = state.p[sq];
  float s0 = st[c * 3], s1 = st[c * 3 + 1], s2 = st[c * 3 + 2];
  const float w0 = __bfloat162float(w[c * 4]), w1 = __bfloat162float(w[c * 4 + 1]);
  const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
  for (int t = b.row0[sq]; t < b.row0[sq + 1]; ++t) {
    const float xv = mixed[static_cast<size_t>(t) * C + c];
    const float o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv;
    s0 = s1;
    s1 = s2;
    s2 = xv;
    mixed[static_cast<size_t>(t) * C + c] = silu(o);
  }
}

// The verify's conv over a tree (BACKLOG 26): row t's taps are the raw inputs of its ancestors at depths d - 3 .. d - 1
// (rows of its variant's path), or the window `state` before depth 0; read from pre, written to post, the window left
// unwritten. The same arithmetic as gdn_conv_kernel's for the row's path as a chain.
__global__ void gdn_conv_tree_kernel(const float* __restrict__ pre, float* __restrict__ post, const float* state,
                                     const __nv_bfloat16* __restrict__ w, int M, int C, const TreeDesc* __restrict__ td) {
  pdl_begin();  // first statement: see launch.cuh
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const float w0 = __bfloat162float(w[c * 4]), w1 = __bfloat162float(w[c * 4 + 1]);
  const float w2 = __bfloat162float(w[c * 4 + 2]), w3 = __bfloat162float(w[c * 4 + 3]);
  for (int t = 0; t < M; ++t) {
    const int d = td->depth[t];
    const int* path = td->path[td->var[t]];
    auto tap = [&](int j) {  // the input at depth j < d
      return j >= 0 ? pre[static_cast<size_t>(path[j]) * C + c] : state[c * 3 + 3 + j];
    };
    const float s0 = tap(d - 3), s1 = tap(d - 2), s2 = tap(d - 1);
    const float xv = pre[static_cast<size_t>(t) * C + c];
    const float o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xv;
    post[static_cast<size_t>(t) * C + c] = silu(o);
  }
}

__global__ void gdn_gating_kernel(const float* a, const float* b, const __nv_bfloat16* A_log,
                                  const __nv_bfloat16* dt_bias, float* g, float* beta, int n, int HV) {
  pdl_begin();  // first statement: see launch.cuh
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int hv = i % HV;
  const float x = a[i] + __bfloat162float(dt_bias[hv]);
  const float softplus = x <= 20.f ? log1pf(__expf(x)) : x;
  g[i] = -__expf(__bfloat162float(A_log[hv])) * softplus;
  beta[i] = 1.f / (1.f + __expf(-b[i]));
}

constexpr int kGdnDk = 128, kGdnDv = 128;

// The gated delta rule, one warp per 32 value columns of a head (4 warps per head, each its own
// block): the q and k norms by warp shuffles, no block-wide barriers, and four times the blocks to load
// the state. Each column's arithmetic per token is the same sequence whatever M is.
__device__ __forceinline__ void gdn_recurrent_body(const float* __restrict__ mixed, const float* __restrict__ g,
                                                   const float* __restrict__ beta, const float* state_in,
                                                   float* state_out, float* __restrict__ out, int M, int H, int HV,
                                                   int bx) {
  __shared__ float qs[kGdnDk], ks[kGdnDk];
  const int hv = bx / 4, part = bx % 4, lane = threadIdx.x, v = part * 32 + lane;
  const int h = hv / (HV / H);
  const int C = 2 * H * kGdnDk + HV * kGdnDv;
  const float* st = state_in + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
  float S[kGdnDk];
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) S[k] = st[k * kGdnDv + v];
  const float scale = rsqrtf(static_cast<float>(kGdnDk));
  for (int t = 0; t < M; ++t) {
    const float* row = mixed + static_cast<size_t>(t) * C;
    const float4 q4 = *reinterpret_cast<const float4*>(row + h * kGdnDk + lane * 4);
    const float4 k4 = *reinterpret_cast<const float4*>(row + H * kGdnDk + h * kGdnDk + lane * 4);
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
    const float vv = row[2 * H * kGdnDk + hv * kGdnDv + v];
    float kv = 0.f;
#pragma unroll
    for (int k = 0; k < kGdnDk; ++k) {
      S[k] *= decay;
      kv = fmaf(S[k], ks[k], kv);
    }
    const float d = (vv - kv) * b;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < kGdnDk; ++k) {
      S[k] = fmaf(d, ks[k], S[k]);
      o = fmaf(S[k], qs[k], o);
    }
    out[static_cast<size_t>(t) * HV * kGdnDv + hv * kGdnDv + v] = o;
  }
  if (state_out == nullptr) return;
  float* so = state_out + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) so[k * kGdnDv + v] = S[k];
}

__global__ void __launch_bounds__(32)
    gdn_recurrent_warp_kernel(const float* __restrict__ mixed, const float* __restrict__ g,
                              const float* __restrict__ beta, const float* state_in, float* state_out,
                              float* __restrict__ out, int M, int H, int HV) {
  pdl_begin();  // first statement: see launch.cuh
  gdn_recurrent_body(mixed, g, beta, state_in, state_out, out, M, H, HV, blockIdx.x);
}

// The verify's recurrence over a tree (BACKLOG 26), blockIdx.y the gdn path: its rows advance the state row by row from
// state_in (unwritten), as gdn_recurrent_body does for a chain; a leaf row's output is computed from the state its
// parent left, which it does not advance (the same products: S[k] * decay, then the fma with d). Only the rows marked
// kWrite are written (each row by one path).
__global__ void __launch_bounds__(32)
    gdn_recurrent_tree_kernel(const float* __restrict__ mixed, const float* __restrict__ g,
                              const float* __restrict__ beta, const float* state_in, float* __restrict__ out, int H,
                              int HV, const TreeDesc* __restrict__ td) {
  pdl_begin();  // first statement: see launch.cuh
  __shared__ float qs[kGdnDk], ks[kGdnDk];
  const int bx = blockIdx.x, pth = blockIdx.y;
  if (pth >= td->ngdn) return;  // the launch covers the configuration's most paths
  const int hv = bx / 4, part = bx % 4, lane = threadIdx.x, v = part * 32 + lane;
  const int h = hv / (HV / H);
  const int C = 2 * H * kGdnDk + HV * kGdnDv;
  const float* st = state_in + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
  float S[kGdnDk];
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) S[k] = st[k * kGdnDv + v];
  const float scale = rsqrtf(static_cast<float>(kGdnDk));
  const int len = td->gdn_len[pth];
  __shared__ int steps[2 * kMaxStreamRows];  // the path's steps, staged once (each row's loads depend on them)
  for (int i = lane; i < len; i += 32) steps[i] = td->gdn_step[pth][i];
  __syncwarp();
  for (int i = 0; i < len; ++i) {
    const int step = steps[i];
    const int t = step & 0xff;
    const bool leaf = (step & TreeDesc::kLeaf) != 0, write = (step & TreeDesc::kWrite) != 0;
    const float* row = mixed + static_cast<size_t>(t) * C;
    const float4 q4 = *reinterpret_cast<const float4*>(row + h * kGdnDk + lane * 4);
    const float4 k4 = *reinterpret_cast<const float4*>(row + H * kGdnDk + h * kGdnDk + lane * 4);
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
    const float vv = row[2 * H * kGdnDk + hv * kGdnDv + v];
    float kv = 0.f, o = 0.f;
    if (!leaf) {
#pragma unroll
      for (int k = 0; k < kGdnDk; ++k) {
        S[k] *= decay;
        kv = fmaf(S[k], ks[k], kv);
      }
      const float d = (vv - kv) * b;
#pragma unroll
      for (int k = 0; k < kGdnDk; ++k) {
        S[k] = fmaf(d, ks[k], S[k]);
        o = fmaf(S[k], qs[k], o);
      }
    } else {
#pragma unroll
      for (int k = 0; k < kGdnDk; ++k) {
        const float sk = S[k] * decay;
        kv = fmaf(sk, ks[k], kv);
      }
      const float d = (vv - kv) * b;
#pragma unroll
      for (int k = 0; k < kGdnDk; ++k) {
        // The same product as the first loop's, recomputed (an opaque multiply: kept products would double the live
        // registers and spill the state).
        float sd;
        asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(sd) : "f"(S[k]), "f"(decay));
        const float sk = fmaf(d, ks[k], sd);
        o = fmaf(sk, qs[k], o);
      }
    }
    if (write) out[static_cast<size_t>(t) * HV * kGdnDv + hv * kGdnDv + v] = o;
  }
}

// tree_gather: block (i, j) moves row rows[j] of buffer i to row j (j ascending is safe in place: rows[j] >= j, and no
// earlier destination is a later source). One block per buffer, the rows in order.
__global__ void tree_gather_kernel(const GatherBuf* __restrict__ bufs, const int* __restrict__ rows, int n, DevPos pos) {
  pdl_begin();  // first statement: see launch.cuh
  const GatherBuf b = bufs[blockIdx.x];
  const size_t off = b.pos_rel ? static_cast<size_t>(pos_value(pos)) : 0;
  for (int j = 1; j < n; ++j) {
    const int r = rows[j];
    if (r == j) continue;
    const int4* src = reinterpret_cast<const int4*>(b.base + (off + r) * b.stride);
    int4* dst = reinterpret_cast<int4*>(b.base + (off + j) * b.stride);
    for (int i = threadIdx.x; i < b.row_bytes / 16; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
  }
}

// The verify's recurrence over several sequences in one launch (M4 stage 4): blockIdx.y is the sequence, its rows
// [row0, row0 + rows) read from its own state, which stays unwritten.
__global__ void __launch_bounds__(32)
    gdn_recurrent_seqs_kernel(const float* __restrict__ mixed, const float* __restrict__ g,
                              const float* __restrict__ beta, SeqPtrs<const float> state, float* __restrict__ out,
                              SeqBatch b, int H, int HV) {
  pdl_begin();  // first statement: see launch.cuh
  const int sq = blockIdx.y;
  const size_t r0 = b.row0[sq];
  const int C = 2 * H * kGdnDk + HV * kGdnDv;
  gdn_recurrent_body(mixed + r0 * C, g + r0 * HV, beta + r0 * HV, state.p[sq], nullptr, out + r0 * HV * kGdnDv,
                     b.row0[sq + 1] - b.row0[sq], H, HV, blockIdx.x);
}

__global__ void gated_rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ z,
                                     const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int D,
                                     float eps) {
  pdl_begin();  // first statement: see launch.cuh
  __shared__ float red[32];
  const size_t r = blockIdx.x;
  float ss = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) ss += x[r * D + d] * x[r * D + d];
  const float rstd = rsqrtf(block_sum(ss, red) / D + eps);
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    out[r * D + d] = x[r * D + d] * rstd * __bfloat162float(w[d]) * silu(z[r * D + d]);
}

// Row m (its position pos) of attn_prepare; the caches are its sequence's.
__device__ __forceinline__ void attn_prepare_body(const float* __restrict__ q_gate, const float* __restrict__ k,
                                                  const float* __restrict__ v, const __nv_bfloat16* __restrict__ q_norm,
                                                  const __nv_bfloat16* __restrict__ k_norm, int m, int pos, int Hq,
                                                  int Hkv, int D, int rot, float theta, float eps, float* __restrict__ q,
                                                  float* __restrict__ gate, __nv_bfloat16* __restrict__ kcache,
                                                  __nv_bfloat16* __restrict__ vcache, size_t hs, size_t ps,
                                                  int cpos) {  // cpos: the cache slot (pos except in a tree)
  extern __shared__ float xn[];
  __shared__ float red[32];
  const int slot = blockIdx.y, d = threadIdx.x;
  if (slot >= Hq + Hkv) {  // v: copy into the cache
    const int kh = slot - Hq - Hkv;
    vcache[kh * hs + cpos * ps + d] = __float2bfloat16(v[(static_cast<size_t>(m) * Hkv + kh) * D + d]);
    return;
  }
  const bool is_q = slot < Hq;
  const int head = is_q ? slot : slot - Hq;
  float x;
  if (is_q) {
    const float* src = q_gate + (static_cast<size_t>(m) * Hq + head) * 2 * D;
    x = src[d];
    gate[(static_cast<size_t>(m) * Hq + head) * D + d] = src[D + d];
  } else {
    x = k[(static_cast<size_t>(m) * Hkv + head) * D + d];
  }
  const float rstd = rsqrtf(block_sum(x * x, red) / D + eps);
  const float wv = __bfloat162float((is_q ? q_norm : k_norm)[d]);
  xn[d] = x * rstd * (1.f + wv);
  __syncthreads();
  float o = xn[d];
  if (d < rot) {
    const int half = rot / 2, i = d % half;
    const float inv_freq = powf(theta, -2.f * i / rot);
    float sn, cs;
    sincosf(static_cast<float>(pos) * inv_freq, &sn, &cs);
    o = d < half ? xn[d] * cs - xn[d + half] * sn : xn[d] * cs + xn[d - half] * sn;
  }
  if (is_q) {
    q[(static_cast<size_t>(m) * Hq + head) * D + d] = o;
  } else {
    kcache[head * hs + cpos * ps + d] = __float2bfloat16(o);
  }
}

__global__ void attn_prepare_kernel(const float* __restrict__ q_gate, const float* __restrict__ k,
                                    const float* __restrict__ v, const __nv_bfloat16* __restrict__ q_norm,
                                    const __nv_bfloat16* __restrict__ k_norm, DevPos pos0, int Hq, int Hkv, int D,
                                    int rot, float theta, float eps, float* __restrict__ q,
                                    float* __restrict__ gate, __nv_bfloat16* __restrict__ kcache,
                                    __nv_bfloat16* __restrict__ vcache, size_t hs, size_t ps,
                                    const TreeDesc* __restrict__ td) {
  pdl_begin();  // first statement: see launch.cuh
  const int m = blockIdx.x, p0 = pos_value(pos0);
  // In a tree (BACKLOG 26) the row's position is its depth's; its key and value still go to slot p0 + m.
  attn_prepare_body(q_gate, k, v, q_norm, k_norm, m, td ? p0 + td->depth[m] : p0 + m, Hq, Hkv, D, rot, theta, eps, q,
                    gate, kcache, vcache, hs, ps, p0 + m);
}

// Several sequences' rows in one launch (M4 stage 4): row m belongs to sequence seq_of(m), at that sequence's
// position, into that sequence's caches.
__global__ void attn_prepare_seqs_kernel(const float* __restrict__ q_gate, const float* __restrict__ k,
                                         const float* __restrict__ v, const __nv_bfloat16* __restrict__ q_norm,
                                         const __nv_bfloat16* __restrict__ k_norm, SeqBatch b, int Hq, int Hkv, int D,
                                         int rot, float theta, float eps, float* __restrict__ q,
                                         float* __restrict__ gate, SeqPtrs<__nv_bfloat16> kc, SeqPtrs<__nv_bfloat16> vc,
                                         size_t hs, size_t ps) {
  pdl_begin();  // first statement: see launch.cuh
  const int m = blockIdx.x, sq = seq_of(b, m);
  const int p = seq_pos(b, sq) + m - b.row0[sq];
  attn_prepare_body(q_gate, k, v, q_norm, k_norm, m, p, Hq, Hkv, D, rot, theta, eps, q, gate, kc.p[sq], vc.p[sq], hs, ps,
                    p);
}

// ---- Rows-path attention on tensor cores (decode and verify, M <= 32). ----
// One block takes all query rows of one KV head (row r = position m * 6 + head g, up to 96 rows: 16
// positions) and one fixed range of attention_chunk() keys, in tiles staged in shared memory.
// S = Q K^T and O += P V run as BF16 m16n8k16 MMAs with FP32 accumulation, one warp per 16 rows, and
// the online softmax lives in the accumulator fragments. Partial (max, sum, O) per row go to `scratch`
// for attention_combine_kernel. A row's arithmetic depends only on its own query and keys: tiles past
// its last key are fully masked and leave its state bit for bit unchanged, so results are row-invariant.
constexpr int kMmaD = 256;
constexpr int kMmaKS = kMmaD + 8;  // padded row stride (bf16) of the K and V tiles: conflict-free fragments
// One K tile and one V tile per stage; `ns` stages in a ring.
constexpr size_t mma_stage(int tk) { return size_t(2) * tk * kMmaKS * sizeof(__nv_bfloat16); }
constexpr size_t mma_smem(int tk, int ns) { return ns * mma_stage(tk); }

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src, bool valid) {
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(gmem_src), "r"(valid ? 16 : 0));
}

// L2 prefetch of the keys [k0, min(k0 + TK, khi)) of one KV head, for the head-major layout, where a tile's K
// (and its V) are one contiguous run: one bulk-prefetch instruction each, by two threads. The ring of shared
// memory holds only two 32-key stages (99 KB per block), so without this a block has one tile in flight while
// it computes the other; the kernel is load-bound (removing its MMAs leaves its time unchanged at 24K) at ~181
// GB/s, where a plain read of the same addresses reaches ~225. Prefetching `pf` tiles further ahead into L2
// costs no shared memory and no registers. Measured at 24K, 16 rows: 0.557 -> 0.492 ms per layer at pf 2;
// 3 and 4 no better. It moves data only: the arithmetic, and so every output bit, is unchanged.
template <int TK>
__device__ __forceinline__ void attn_prefetch_tile(const __nv_bfloat16* __restrict__ kcache,
                                                   const __nv_bfloat16* __restrict__ vcache, int k0, int khi, int kvh,
                                                   size_t hs, size_t ps) {
  const int n = min(TK, khi - k0);
  if (n <= 0) return;
  const size_t off = kvh * hs + static_cast<size_t>(k0) * ps;
  const uint32_t bytes = static_cast<uint32_t>(n) * kMmaD * 2;
  if (threadIdx.x == 0)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(kcache + off), "r"(bytes) : "memory");
  else if (threadIdx.x == 32)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(vcache + off), "r"(bytes) : "memory");
}

// The keys [k0, k0 + TK) of one KV head into a stage (zero-filled past khi).
template <int TK>
__device__ __forceinline__ void attn_load_tile(__nv_bfloat16* stage, const __nv_bfloat16* __restrict__ kcache,
                                               const __nv_bfloat16* __restrict__ vcache, int k0, int khi, int kvh,
                                               size_t hs, size_t ps) {
  constexpr int D = kMmaD, KS = kMmaKS;
  __nv_bfloat16* ks = stage;
  __nv_bfloat16* vs = stage + TK * KS;
  for (int i = threadIdx.x; i < TK * (D / 8); i += blockDim.x) {
    const int j = i / (D / 8), c = i % (D / 8), key = k0 + j;
    const bool valid = key < khi;
    const size_t off = kvh * hs + static_cast<size_t>(valid ? key : 0) * ps + 8 * c;
    cp_async16(ks + j * KS + 8 * c, kcache + off, valid);
    cp_async16(vs + j * KS + 8 * c, vcache + off, valid);
  }
  asm volatile("cp.async.commit_group;");
}

// A tree verify's tile (BACKLOG 26): keys from pos0 on are the variant's path (path[j] is the row at depth j, whose
// key sits at slot pos0 + path[j]), at the positions a chain holding that path would hold them in; the keys below
// pos0 are the context's, as in attn_load_tile.
template <int TK>
__device__ __forceinline__ void attn_load_tile_tree(__nv_bfloat16* stage, const __nv_bfloat16* __restrict__ kcache,
                                                    const __nv_bfloat16* __restrict__ vcache, int k0, int khi, int kvh,
                                                    size_t hs, size_t ps, int pos0, const int* __restrict__ path) {
  constexpr int D = kMmaD, KS = kMmaKS;
  __nv_bfloat16* ks = stage;
  __nv_bfloat16* vs = stage + TK * KS;
  for (int i = threadIdx.x; i < TK * (D / 8); i += blockDim.x) {
    const int j = i / (D / 8), c = i % (D / 8), key = k0 + j;
    const bool valid = key < khi;
    const int src = key < pos0 ? key : pos0 + path[min(key - pos0, kMaxStreamRows - 1)];
    const size_t off = kvh * hs + static_cast<size_t>(valid ? src : 0) * ps + 8 * c;
    cp_async16(ks + j * KS + 8 * c, kcache + off, valid);
    cp_async16(vs + j * KS + 8 * c, vcache + off, valid);
  }
  asm volatile("cp.async.commit_group;");
}

// ---- The same tiles through the bulk-copy (TMA) engine (M3, set_attention_bulk). ----
// One key's 256 values of one KV head are 512 contiguous bytes in both KV layouts, so a tile is TK bulk
// copies for K and TK for V, one per lane of one warp, each into its padded shared-memory row: the layout
// the MMAs read is unchanged. Completion is counted in bytes on one mbarrier per stage instead of
// cp.async groups. The copies run without the SM issuing a 16-byte request per thread, and each is one
// 512-byte request to memory.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_addr(bar)), "r"(count) : "memory");
}

// Waits until phase `parity` of the barrier has completed (every expected byte has landed).
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile(
      "{\n"
      " .reg .pred done;\n"
      "WAIT:\n"
      " mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
      " @!done bra WAIT;\n"
      "}\n" ::"r"(smem_addr(bar)),
      "r"(parity)
      : "memory");
}

// The keys [k0, min(k0 + TK, khi)) of one KV head into a stage; called by every lane of one warp. Rows past
// khi are not copied: they keep finite values from an earlier tile (or the zeros the kernel starts with),
// their scores are masked to -inf and their probabilities are exactly 0, so they add nothing, as the zero
// rows of attn_load_tile add nothing.
template <int TK>
__device__ __forceinline__ void attn_bulk_tile(__nv_bfloat16* stage, const __nv_bfloat16* __restrict__ kcache,
                                               const __nv_bfloat16* __restrict__ vcache, int k0, int khi, int kvh,
                                               size_t hs, size_t ps, uint64_t* bar) {
  static_assert(TK <= 32, "one key per lane");
  constexpr int D = kMmaD, KS = kMmaKS, kRowBytes = D * 2;
  const int lane = threadIdx.x & 31, valid = max(0, min(TK, khi - k0));
  // The barrier's transaction count first (lane 0's arrival), so the phase cannot complete before every
  // copy of this tile has been counted.
  if (lane == 0)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(bar)),
                 "r"(2 * valid * kRowBytes)
                 : "memory");
  __syncwarp();
  if (lane < valid) {
    const size_t off = kvh * hs + static_cast<size_t>(k0 + lane) * ps;
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
            smem_addr(stage + lane * KS)),
        "l"(kcache + off), "r"(kRowBytes), "r"(smem_addr(bar))
        : "memory");
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
            smem_addr(stage + (TK + lane) * KS)),
        "l"(vcache + off), "r"(kRowBytes), "r"(smem_addr(bar))
        : "memory");
  }
}

// query rows per block (16 per warp), keys per tile, pipeline stages; BULK: tiles through the bulk-copy
// engine (attn_bulk_tile) instead of cp.async. The arithmetic is the same either way, bit for bit.
template <int ROWS, int TK, int NS, bool BULK>
__device__ __forceinline__ void attention_mma_body(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kcache,
                                                   const __nv_bfloat16* __restrict__ vcache, int pos0, int M, int Hq,
                                                   int Hkv, int splits, int chunk, float* __restrict__ scratch, size_t hs,
                                                   size_t ps, int pf, int bx, const TreeDesc* __restrict__ td = nullptr) {
  constexpr int D = kMmaD, G = 6, KS = kMmaKS;
  extern __shared__ __align__(16) unsigned char smem[];
  auto stage = [&](int i) { return reinterpret_cast<__nv_bfloat16*>(smem + (i % NS) * mma_stage(TK)); };
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int kvh = blockIdx.y, split = blockIdx.z;
  const int R = M * G, r0 = bx * ROWS, nrows = min(ROWS, R - r0);
  const bool active = warp * 16 < nrows;
  const int last_m = (r0 + nrows - 1) / G;
  const int kend = pos0 + last_m + 1;  // keys [0, kend) are visible to some row of this block
  const int klo = split * chunk, khi = min(kend, klo + chunk);
  if (klo >= kend) {
    // A range no row of this block sees (a graph launches ranges up to a fixed context): an empty partial,
    // max = -inf, which attention_combine_kernel skips without reading its sum or values.
    for (int r = tid; r < nrows; r += blockDim.x) {
      const int row = r0 + r;
      scratch[((static_cast<size_t>(row / G) * Hq + kvh * G + row % G) * splits + split) * (D + 2)] = -INFINITY;
    }
    return;
  }
  // This thread's two rows (g and g + 8 of the warp's 16), their positions, and their queries as MMA
  // A fragments in BF16, scaled by 1 / sqrt(D) (a power of two: exact).
  const int ra = r0 + warp * 16 + g, rb = ra + 8;
  const bool va = ra < r0 + nrows, vb = rb < r0 + nrows;
  // A tree's row sees the keys up to its depth's position (its path's, from the tail tiles on).
  const int pa = pos0 + (td ? (va ? td->depth[ra / G] : 0) : ra / G);
  const int pb = pos0 + (td ? (vb ? td->depth[rb / G] : 0) : rb / G);
  const float* qa = q + (static_cast<size_t>(ra / G) * Hq + kvh * G + ra % G) * D;
  const float* qb = q + (static_cast<size_t>(rb / G) * Hq + kvh * G + rb % G) * D;
  uint32_t qf[D / 16][4];
#pragma unroll
  for (int s = 0; s < D / 16; ++s) {
    const int k = 16 * s + 2 * t;
    const float2 a0 = va ? *reinterpret_cast<const float2*>(qa + k) : make_float2(0.f, 0.f);
    const float2 a1 = vb ? *reinterpret_cast<const float2*>(qb + k) : make_float2(0.f, 0.f);
    const float2 a2 = va ? *reinterpret_cast<const float2*>(qa + k + 8) : make_float2(0.f, 0.f);
    const float2 a3 = vb ? *reinterpret_cast<const float2*>(qb + k + 8) : make_float2(0.f, 0.f);
    qf[s][0] = pack_bf16(a0.x * 0.0625f, a0.y * 0.0625f);
    qf[s][1] = pack_bf16(a1.x * 0.0625f, a1.y * 0.0625f);
    qf[s][2] = pack_bf16(a2.x * 0.0625f, a2.y * 0.0625f);
    qf[s][3] = pack_bf16(a3.x * 0.0625f, a3.y * 0.0625f);
  }
  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f;
  // A ring of NS stages: tiles i + 1 .. i + NS - 1 are in flight while tile i is computed. Every
  // iteration commits one cp.async group (empty past the range), so wait_group NS - 2 always means
  // "tile i has landed". With BULK, stage s's tiles complete phases 0, 1, 0, ... of bar[s] instead.
  // In a tree the pipelined loop stops before the first tile (on the range's own grid of TK keys) holding a key at
  // or past pos0, the tail, which follows per variant below; a chain's loop covers the whole range.
  const int kctx = td ? min(khi, klo + max(0, (pos0 - klo) / TK) * TK) : khi;
  const int ntiles = (kctx - klo + TK - 1) / TK;
  __shared__ __align__(8) uint64_t bar[BULK ? NS : 1];
  if constexpr (BULK) {
    // Zero the stages once (rows past the range are never copied), make those generic-proxy writes
    // visible to the bulk copies about to overwrite them, and set up one barrier per stage (one arrival:
    // warp 0's lane 0, with the tile's byte count).
    for (size_t i = tid; i < mma_smem(TK, NS) / 16; i += blockDim.x) reinterpret_cast<uint4*>(smem)[i] = make_uint4(0, 0, 0, 0);
    if (tid == 0) {
      for (int i = 0; i < NS; ++i) mbar_init(&bar[i], 1);
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
  }
  // One tile's scores, online softmax and P V for this warp's rows; ca / cb: rows a / b take the tile (a chain's
  // rows always do).
  auto tile_compute = [&](const __nv_bfloat16* ks, int k0, bool ca, bool cb) {
    const __nv_bfloat16* vs = ks + TK * KS;
    {
      // S = Q K^T for 16 rows x 32 keys: 4 key tiles of 8.
      float sc[TK / 8][4];
#pragma unroll
      for (int n = 0; n < TK / 8; ++n) sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
#pragma unroll
      for (int s = 0; s < D / 16; ++s) {
#pragma unroll
        for (int n = 0; n < TK / 8; ++n) {
          const __nv_bfloat16* kr = ks + (n * 8 + g) * KS + 16 * s;
          mma_bf16_16816(sc[n], qf[s][0], qf[s][1], qf[s][2], qf[s][3],
                         *reinterpret_cast<const uint32_t*>(kr + 2 * t),
                         *reinterpret_cast<const uint32_t*>(kr + 8 + 2 * t));
        }
      }
      // Mask (causal, and the range end), then the online softmax for rows a and b.
      float ta = -INFINITY, tb = -INFINITY;
#pragma unroll
      for (int n = 0; n < TK / 8; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = k0 + n * 8 + 2 * t + e;
          if (key > pa || key >= khi) sc[n][e] = -INFINITY;
          if (key > pb || key >= khi) sc[n][2 + e] = -INFINITY;
          ta = fmaxf(ta, sc[n][e]);
          tb = fmaxf(tb, sc[n][2 + e]);
        }
      ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, 1));
      ta = fmaxf(ta, __shfl_xor_sync(0xffffffffu, ta, 2));
      tb = fmaxf(tb, __shfl_xor_sync(0xffffffffu, tb, 1));
      tb = fmaxf(tb, __shfl_xor_sync(0xffffffffu, tb, 2));
      const float na = fmaxf(ma, ta), nb = fmaxf(mb, tb);
      const float alpha_a = na == -INFINITY ? 1.f : __expf(ma - na);
      const float alpha_b = nb == -INFINITY ? 1.f : __expf(mb - nb);
      float sa = 0.f, sb = 0.f;
#pragma unroll
      for (int n = 0; n < TK / 8; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          sc[n][e] = sc[n][e] == -INFINITY ? 0.f : __expf(sc[n][e] - na);
          sc[n][2 + e] = sc[n][2 + e] == -INFINITY ? 0.f : __expf(sc[n][2 + e] - nb);
          sa += sc[n][e];
          sb += sc[n][2 + e];
        }
      sa += __shfl_xor_sync(0xffffffffu, sa, 1);
      sa += __shfl_xor_sync(0xffffffffu, sa, 2);
      sb += __shfl_xor_sync(0xffffffffu, sb, 1);
      sb += __shfl_xor_sync(0xffffffffu, sb, 2);
      // A tree's tail tile: a row of another variant keeps its state (alpha 1 and P 0 leave it bit for bit).
      if (ca) {
        la = la * alpha_a + sa;
        ma = na;
      }
      if (cb) {
        lb = lb * alpha_b + sb;
        mb = nb;
      }
      const float fa = ca ? alpha_a : 1.f, fb = cb ? alpha_b : 1.f;
#pragma unroll
      for (int n = 0; n < TK / 8; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          if (!ca) sc[n][e] = 0.f;
          if (!cb) sc[n][2 + e] = 0.f;
        }
#pragma unroll
      for (int i = 0; i < D / 8; ++i) {
        o[i][0] *= fa;
        o[i][1] *= fa;
        o[i][2] *= fb;
        o[i][3] *= fb;
      }
      // O += P V: P from the score fragments, V's B fragments by a transposing ldmatrix (two dim tiles each).
#pragma unroll
      for (int j = 0; j < TK / 16; ++j) {
        const uint32_t a0 = pack_bf16(sc[2 * j][0], sc[2 * j][1]), a1 = pack_bf16(sc[2 * j][2], sc[2 * j][3]);
        const uint32_t a2 = pack_bf16(sc[2 * j + 1][0], sc[2 * j + 1][1]);
        const uint32_t a3 = pack_bf16(sc[2 * j + 1][2], sc[2 * j + 1][3]);
        const int key = j * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int i = 0; i < D / 8; i += 2) {
          const uint32_t addr = static_cast<uint32_t>(
              __cvta_generic_to_shared(vs + key * KS + i * 8 + (lane >> 4) * 8));
          uint32_t b0, b1, b2, b3;
          asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                       : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                       : "r"(addr));
          mma_bf16_16816(o[i], a0, a1, a2, a3, b0, b1);
          mma_bf16_16816(o[i + 1], a0, a1, a2, a3, b2, b3);
        }
      }
    }
  };
#pragma unroll
  for (int p = 0; p < NS - 1; ++p) {
    if constexpr (BULK) {
      if (warp == 0 && p < ntiles) attn_bulk_tile<TK>(stage(p), kcache, vcache, klo + p * TK, khi, kvh, hs, ps, &bar[p]);
    } else {
      if (p < ntiles) attn_load_tile<TK>(stage(p), kcache, vcache, klo + p * TK, khi, kvh, hs, ps);
      else asm volatile("cp.async.commit_group;");
    }
  }
  for (int it = 0; it < ntiles; ++it) {
    const int k0 = klo + it * TK;
    if constexpr (BULK) mbar_wait(&bar[it % NS], (it / NS) & 1);
    else asm volatile("cp.async.wait_group %0;" ::"n"(NS - 2));
    __syncthreads();  // tile it is visible to all, and stage (it - 1) % NS is free again
    if constexpr (BULK) {
      if (warp == 0 && it + NS - 1 < ntiles)
        attn_bulk_tile<TK>(stage(it + NS - 1), kcache, vcache, k0 + (NS - 1) * TK, khi, kvh, hs, ps,
                           &bar[(it + NS - 1) % NS]);
    } else {
      if (it + NS - 1 < ntiles) attn_load_tile<TK>(stage(it + NS - 1), kcache, vcache, k0 + (NS - 1) * TK, khi, kvh, hs, ps);
      else asm volatile("cp.async.commit_group;");
      // Tiles it + NS .. it + NS + pf - 1 on their way into L2 (the first iteration starts the whole window).
      if (pf > 0 && ps == static_cast<size_t>(D)) {
        if (it == 0)
          for (int p = NS; p < NS + pf - 1; ++p)
            if (p < ntiles) attn_prefetch_tile<TK>(kcache, vcache, klo + p * TK, khi, kvh, hs, ps);
        if (it + NS + pf - 1 < ntiles) attn_prefetch_tile<TK>(kcache, vcache, k0 + (NS + pf - 1) * TK, khi, kvh, hs, ps);
      }
    }
    const __nv_bfloat16* ks = stage(it);
    if (active) tile_compute(ks, k0, true, true);
  }
  // The tail of a tree: per variant present in this block, its tile, committed by that variant's rows only. (Trees
  // take the cp.async tiles: the bulk-copy instantiation has no tail, and attention_rows refuses a tree with it.)
  if constexpr (!BULK) if (td != nullptr && kctx < khi) {
    const int first_m = r0 / G;
    for (int k0 = kctx; k0 < khi; k0 += TK) {
      for (int v = 0; v < td->nvar; ++v) {
        bool any = false;
        for (int m = first_m; m <= last_m; ++m) any = any || td->var[m] == v;
        if (!any) continue;
        __syncthreads();  // every warp is done with stage 0
        attn_load_tile_tree<TK>(stage(0), kcache, vcache, k0, khi, kvh, hs, ps, pos0, td->path[v]);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if (active) tile_compute(stage(0), k0, va && td->var[ra / G] == v, vb && td->var[rb / G] == v);
      }
    }
  }
  if (!active) return;
  // Partial state per row, in attention_combine_kernel's layout.
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int row = h ? rb : ra;
    if (!(h ? vb : va)) continue;
    float* part = scratch + ((static_cast<size_t>(row / G) * Hq + kvh * G + row % G) * splits + split) * (D + 2);
    if (t == 0) {
      part[0] = h ? mb : ma;
      part[1] = h ? lb : la;
    }
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      part[2 + i * 8 + 2 * t] = o[i][2 * h];
      part[2 + i * 8 + 2 * t + 1] = o[i][2 * h + 1];
    }
  }
}

template <int ROWS, int TK, int NS, bool BULK>
__global__ void __launch_bounds__(ROWS * 2)
    attention_mma_kernel(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kcache,
                         const __nv_bfloat16* __restrict__ vcache, DevPos pos0_arg, int M, int Hq, int Hkv,
                         int splits, int chunk, float* __restrict__ scratch, size_t hs, size_t ps, int pf,
                         const TreeDesc* __restrict__ td) {
  pdl_begin();  // first statement: see launch.cuh
  attention_mma_body<ROWS, TK, NS, BULK>(q, kcache, vcache, pos_value(pos0_arg), M, Hq, Hkv, splits, chunk, scratch, hs,
                                         ps, pf, blockIdx.x, td);
}

// Several sequences in one launch (M4 stage 4): the x blocks of sequence i are [blk0[i], blk0[i + 1]); each block
// computes what attention_mma_kernel computes for that sequence alone (its queries, its caches, its position), and
// its partials land at the sequence's rows of `scratch`, so one combine over every row follows.
struct AttnBlocks {
  int blk0[kMaxSeqs + 1];
};
template <int ROWS, int TK, int NS, bool BULK>
__global__ void __launch_bounds__(ROWS * 2)
    attention_mma_seqs_kernel(const float* __restrict__ q, SeqPtrs<const __nv_bfloat16> kc,
                              SeqPtrs<const __nv_bfloat16> vc, SeqBatch b, AttnBlocks blocks, int Hq, int Hkv,
                              int splits, int chunk, float* __restrict__ scratch, size_t hs, size_t ps, int pf) {
  pdl_begin();  // first statement: see launch.cuh
  int sq = 0;
  while (sq + 1 < b.n && static_cast<int>(blockIdx.x) >= blocks.blk0[sq + 1]) ++sq;
  constexpr int D = kMmaD;
  const size_t r0 = b.row0[sq];
  attention_mma_body<ROWS, TK, NS, BULK>(q + r0 * Hq * D, kc.p[sq], vc.p[sq], seq_pos(b, sq), b.row0[sq + 1] - b.row0[sq],
                                         Hq, Hkv, splits, chunk, scratch + r0 * Hq * splits * (D + 2), hs, ps, pf,
                                         blockIdx.x - blocks.blk0[sq]);
}

__global__ void attention_combine_kernel(const float* __restrict__ scratch, int splits, int D,
                                         float* __restrict__ out) {
  pdl_begin();  // first statement: see launch.cuh
  const int mh = blockIdx.x, d = threadIdx.x;
  const float* base = scratch + static_cast<size_t>(mh) * splits * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, base[s * (D + 2)]);
  float L = 0.f, A = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float ms = base[s * (D + 2)];
    if (ms == -INFINITY) continue;
    const float c = __expf(ms - M);
    L += base[s * (D + 2) + 1] * c;
    A += base[s * (D + 2) + 2 + d] * c;
  }
  out[static_cast<size_t>(mh) * D + d] = A / L;
}

int grid_for(size_t n, int threads) {
  return static_cast<int>(std::min<size_t>((n + threads - 1) / threads, 65535 * 4));
}

// The KV caches' layout, in elements: head h, position p, dim d at h * hs + p * ps + d. Interleaved
// ([pos][head][dim], v0) unless set_kv_layout says otherwise.
size_t g_kv_hs = 0, g_kv_ps = 0;
size_t kv_hs(int D) { return g_kv_hs ? g_kv_hs : D; }
size_t kv_ps(int Hkv, int D) { return g_kv_ps ? g_kv_ps : static_cast<size_t>(Hkv) * D; }

bool g_pdl = false;
bool g_attn_bulk = false;
int g_attn_prefetch = 0;

}  // namespace

void set_pdl(bool on) { g_pdl = on; }
void set_attention_bulk(bool on) { g_attn_bulk = on; }
void set_attention_prefetch(int tiles) { g_attn_prefetch = tiles < 0 ? 0 : tiles; }
bool pdl_enabled() { return g_pdl; }
bool pdl_attention_enabled() {
  static const bool attn = [] {
    const char* e = std::getenv("LING_PDL_ATTN");
    return e && std::atoi(e) != 0;
  }();
  return g_pdl && attn;
}

void set_kv_layout(size_t head_stride, size_t pos_stride) {
  g_kv_hs = head_stride;
  g_kv_ps = pos_stride;
}

// Rows per warp: with one or two token rows each warp streams 2 weight rows (measured best of 1, 2, 4, 8:
// 235 GB/s NVFP4 and 263 GB/s FP8 at M = 1, against 262 GB/s for a plain read); more rows need the registers.
template <int MT, int R>
void launch_nvfp4(const float* x, int M, const uint8_t* w, const uint8_t* wscale, float scale2, float* y, int N,
                  int K, cudaStream_t s) {
  const int rows_per_block = kGemvWarps * R;
  gemv_nvfp4_kernel<MT, R><<<(N + rows_per_block - 1) / rows_per_block, kGemvWarps * 32, 0, s>>>(x, M, w, wscale,
                                                                                               scale2, y, N, K);
}

template <int MT, int R>
void launch_fp8(const float* x, int M, const uint8_t* w, float wscale, float* y, int N, int K, cudaStream_t s) {
  const int rows_per_block = kGemvWarps * R;
  gemv_fp8_kernel<MT, R><<<(N + rows_per_block - 1) / rows_per_block, kGemvWarps * 32, 0, s>>>(x, M, w, wscale, y,
                                                                                             N, K);
}

void gemv_nvfp4(const float* x, int M, const uint8_t* w, const uint8_t* wscale, float scale2, float* y, int N,
                int K, cudaStream_t s) {
  if (K % kFp4Tile != 0) throw std::runtime_error("gemv_nvfp4: K must be a multiple of 1024");
  if (M <= 1) launch_nvfp4<1, 2>(x, M, w, wscale, scale2, y, N, K, s);
  else if (M <= 2) launch_nvfp4<2, 2>(x, M, w, wscale, scale2, y, N, K, s);
  else if (M <= 4) launch_nvfp4<4, 2>(x, M, w, wscale, scale2, y, N, K, s);
  else if (M <= 8) launch_nvfp4<8, 1>(x, M, w, wscale, scale2, y, N, K, s);
  else throw std::runtime_error("gemv_nvfp4: M > 8");
  LING_LAUNCH_CHECK("gemv_nvfp4");
}

void gemv_fp8(const float* x, int M, const uint8_t* w, float wscale, float* y, int N, int K, cudaStream_t s) {
  if (K % kFp8Tile != 0) throw std::runtime_error("gemv_fp8: K must be a multiple of 512");
  if (M <= 1) launch_fp8<1, 2>(x, M, w, wscale, y, N, K, s);
  else if (M <= 2) launch_fp8<2, 2>(x, M, w, wscale, y, N, K, s);
  else if (M <= 4) launch_fp8<4, 2>(x, M, w, wscale, y, N, K, s);
  else if (M <= 8) launch_fp8<8, 1>(x, M, w, wscale, y, N, K, s);
  else throw std::runtime_error("gemv_fp8: M > 8");
  LING_LAUNCH_CHECK("gemv_fp8");
}

void gemv_bf16(const float* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s) {
  gemv_bf16_kernel<<<(N + kGemvWarps - 1) / kGemvWarps, kGemvWarps * 32, 0, s>>>(x, M, w, y, N, K);
  LING_LAUNCH_CHECK("gemv_bf16");
}

void bf16_rows(const float* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s,
               const __nv_bfloat16* w2, float* y2) {
  if (M < 1 || M > 32 || K % (kBf16RowsThreads * 8) != 0) throw std::runtime_error("bf16_rows: M <= 32, K % 1024 == 0");
  launch_kernel("bf16_rows", bf16_rows_kernel<float>, dim3(w2 ? 2 * N : N), dim3(kBf16RowsThreads), 0, s, x, M, w, y,
                w2, y2, N, K);
}

void bf16_rows_bf16in(const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s) {
  if (M < 1 || M > 32 || K % (kBf16RowsThreads * 8) != 0) throw std::runtime_error("bf16_rows: M <= 32, K % 1024 == 0");
  launch_kernel("bf16_rows_bf16in", bf16_rows_kernel<__nv_bfloat16>, dim3(N), dim3(kBf16RowsThreads), 0, s, x, M, w,
                y, nullptr, nullptr, N, K);
}

void dequant_nvfp4(const uint8_t* w, const uint8_t* wscale, float scale2, __nv_bfloat16* out, int N, int K,
                   cudaStream_t s) {
  const size_t bytes = static_cast<size_t>(N) * K / 2;
  dequant_nvfp4_kernel<<<grid_for(bytes, 256), 256, 0, s>>>(w, wscale, scale2, out, bytes, K);
  LING_LAUNCH_CHECK("dequant_nvfp4");
}

void dequant_fp8(const uint8_t* w, float wscale, __nv_bfloat16* out, int N, int K, cudaStream_t s) {
  const size_t total = static_cast<size_t>(N) * K;
  dequant_fp8_kernel<<<grid_for(total, 256), 256, 0, s>>>(w, wscale, out, total);
  LING_LAUNCH_CHECK("dequant_fp8");
}

void to_bf16(const float* x, __nv_bfloat16* out, int n, cudaStream_t s) {
  to_bf16_kernel<<<grid_for(n, 256), 256, 0, s>>>(x, out, n);
  LING_LAUNCH_CHECK("to_bf16");
}

void embed(const __nv_bfloat16* table, const int* ids, int M, int H, float* out, cudaStream_t s) {
  launch_kernel("embed", embed_kernel, dim3(M), dim3(256), 0, s, table, ids, H, out);
}

void rmsnorm(const float* x, const __nv_bfloat16* w, float* out, int rows, int H, float eps, bool gemma,
             cudaStream_t s) {
  launch_kernel("rmsnorm", rmsnorm_kernel, dim3(rows), dim3(H >= 1024 ? 1024 : H), 0, s, x, w, out, H, eps, gemma);
}

void add_inplace(float* h, const float* d, int n, cudaStream_t s) {
  launch_kernel("add", add_kernel, dim3(grid_for(n, 256)), dim3(256), 0, s, h, d, n);
}

void silu_mul(const float* g, const float* u, float* out, int n, cudaStream_t s) {
  launch_kernel("silu_mul", silu_mul_kernel, dim3(grid_for(n, 256)), dim3(256), 0, s, g, u, out, n);
}

void sigmoid_mul(float* x, const float* gate, int n, cudaStream_t s) {
  launch_kernel("sigmoid_mul", sigmoid_mul_kernel, dim3(grid_for(n, 256)), dim3(256), 0, s, x, gate, n);
}

void gdn_conv(float* mixed, float* conv_state, const __nv_bfloat16* w, int M, int C, cudaStream_t s) {
  launch_kernel("gdn_conv", gdn_conv_kernel, dim3((C + 255) / 256), dim3(256), 0, s, mixed, conv_state, w, M, C);
}

void gdn_gating(const float* a, const float* b, const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
                float* g, float* beta, int M, int HV, cudaStream_t s) {
  const int n = M * HV;
  launch_kernel("gdn_gating", gdn_gating_kernel, dim3((n + 255) / 256), dim3(256), 0, s, a, b, A_log, dt_bias, g, beta,
                n, HV);
}

void gdn_recurrent(const float* mixed, const float* g, const float* beta, const float* state_in, float* state_out,
                   float* out, int M, int H, int HV, cudaStream_t s) {
  launch_kernel("gdn_recurrent", gdn_recurrent_warp_kernel, dim3(HV * 4), dim3(32), 0, s, mixed, g, beta, state_in,
                state_out, out, M, H, HV);
}

void gdn_conv_tree(const float* pre, float* post, const float* state, const __nv_bfloat16* w, int M, int C,
                   const TreeDesc* td, cudaStream_t s) {
  launch_kernel("gdn_conv_tree", gdn_conv_tree_kernel, dim3((C + 255) / 256), dim3(256), 0, s, pre, post, state, w, M, C,
                td);
}

void gdn_recurrent_tree(const float* mixed, const float* g, const float* beta, const float* state_in, float* out,
                        int H, int HV, const TreeDesc* td, int ngdn, cudaStream_t s) {
  launch_kernel("gdn_recurrent_tree", gdn_recurrent_tree_kernel, dim3(HV * 4, ngdn), dim3(32), 0, s, mixed, g, beta,
                state_in, out, H, HV, td);
}

void tree_gather(const GatherBuf* bufs, int nbufs, const int* rows, int n, DevPos pos, cudaStream_t s) {
  if (nbufs <= 0 || n <= 1) return;
  launch_kernel("tree_gather", tree_gather_kernel, dim3(nbufs), dim3(256), 0, s, bufs, rows, n, pos);
}

void gdn_conv_seqs(float* mixed, SeqPtrs<const float> state, const __nv_bfloat16* w, const SeqBatch& b, int C,
                   cudaStream_t s) {
  launch_kernel("gdn_conv_seqs", gdn_conv_seqs_kernel, dim3((C + 255) / 256, b.n), dim3(256), 0, s, mixed, state, w, b, C);
}

void gdn_recurrent_seqs(const float* mixed, const float* g, const float* beta, SeqPtrs<const float> state, float* out,
                        const SeqBatch& b, int H, int HV, cudaStream_t s) {
  launch_kernel("gdn_recurrent_seqs", gdn_recurrent_seqs_kernel, dim3(HV * 4, b.n), dim3(32), 0, s, mixed, g, beta,
                state, out, b, H, HV);
}

void gated_rmsnorm(const float* x, const float* z, const __nv_bfloat16* w, float* out, int rows, int D,
                   float eps, cudaStream_t s) {
  launch_kernel("gated_rmsnorm", gated_rmsnorm_kernel, dim3(rows), dim3(D), 0, s, x, z, w, out, D, eps);
}

void attn_prepare(const float* q_gate, const float* k, const float* v, const __nv_bfloat16* q_norm,
                  const __nv_bfloat16* k_norm, DevPos pos0, int M, int Hq, int Hkv, int D, int rot, float theta,
                  float eps, float* q, float* gate, __nv_bfloat16* kcache, __nv_bfloat16* vcache,
                  cudaStream_t s, const TreeDesc* td) {
  dim3 grid(M, Hq + 2 * Hkv);
  launch_kernel("attn_prepare", attn_prepare_kernel, grid, dim3(D), D * sizeof(float), s, q_gate, k, v, q_norm, k_norm,
                pos0, Hq, Hkv, D, rot, theta, eps, q, gate, kcache, vcache, kv_hs(D), kv_ps(Hkv, D), td);
}

void attn_prepare_seqs(const float* q_gate, const float* k, const float* v, const __nv_bfloat16* q_norm,
                       const __nv_bfloat16* k_norm, const SeqBatch& b, int Hq, int Hkv, int D, int rot, float theta,
                       float eps, float* q, float* gate, SeqPtrs<__nv_bfloat16> kc, SeqPtrs<__nv_bfloat16> vc,
                       cudaStream_t s) {
  dim3 grid(b.row0[b.n], Hq + 2 * Hkv);
  launch_kernel("attn_prepare_seqs", attn_prepare_seqs_kernel, grid, dim3(D), D * sizeof(float), s, q_gate, k, v,
                q_norm, k_norm, b, Hq, Hkv, D, rot, theta, eps, q, gate, kc, vc, kv_hs(D), kv_ps(Hkv, D));
}

// Fixed key ranges at fixed positions (row invariance); LING_ATTN_CHUNK picks the size for experiments.
int attention_chunk() {
  static const int c = [] {
    const char* e = std::getenv("LING_ATTN_CHUNK");
    return e ? std::atoi(e) : 4096;  // measured at 24K, 16 rows, realistic data: 1024 keys 0.63 ms per layer, 2048 0.61, 4096 0.57, 8192 0.66
  }();
  return c;
}

int attention_rows_splits(int ctx) { return (ctx + attention_chunk() - 1) / attention_chunk(); }

size_t attention_rows_scratch_floats(int M, int Hq, int D, int ctx) {
  return static_cast<size_t>(M) * Hq * attention_rows_splits(ctx) * (D + 2);
}

// 32-key tiles, double-buffered. Measured at 24K context, 16 rows, realistic data: 0.57 ms per layer;
// 16-key tiles with 3-5 stages were no faster.
constexpr int kAttnRows = 96, kAttnTk = 32, kAttnNs = 2;

void prepare_kernels() {
  static bool configured = false;
  if (configured) return;
  check(cudaFuncSetAttribute(attention_mma_kernel<kAttnRows, kAttnTk, kAttnNs, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mma_smem(kAttnTk, kAttnNs))),
        "attention_rows smem");
  check(cudaFuncSetAttribute(attention_mma_kernel<kAttnRows, kAttnTk, kAttnNs, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mma_smem(kAttnTk, kAttnNs))),
        "attention_rows smem");
  check(cudaFuncSetAttribute(attention_mma_seqs_kernel<kAttnRows, kAttnTk, kAttnNs, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mma_smem(kAttnTk, kAttnNs))),
        "attention_rows smem");
  check(cudaFuncSetAttribute(attention_mma_seqs_kernel<kAttnRows, kAttnTk, kAttnNs, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(mma_smem(kAttnTk, kAttnNs))),
        "attention_rows smem");
  configured = true;
}

void attention_rows(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, DevPos pos0, int M,
                    int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s, int fixed_ctx,
                    const TreeDesc* td) {
  if (D != 256 || Hq != 6 * Hkv) throw std::runtime_error("attention_rows: built for head_dim 256, 6 query heads per KV head");
  if (pos0.dev && fixed_ctx <= 0) throw std::runtime_error("attention_rows: a device position needs a fixed context");
  if (td && g_attn_bulk) throw std::runtime_error("attention_rows: a tree verify needs the cp.async tiles (no --attn-bulk)");
  // The grid: the ranges covering the rows' keys, or (in a graph) every range up to fixed_ctx.
  const int splits = attention_rows_splits(fixed_ctx > 0 ? fixed_ctx : pos0.host + M);
  constexpr int rows = kAttnRows, tk = kAttnTk, ns = kAttnNs;
  prepare_kernels();
  const int R = M * 6;
  launch_kernel_pdl(pdl_attention_enabled(), "attention_rows",
                    g_attn_bulk ? attention_mma_kernel<rows, tk, ns, true> : attention_mma_kernel<rows, tk, ns, false>,
                    dim3((R + rows - 1) / rows, Hkv, splits), dim3(rows * 2), mma_smem(tk, ns), s, q, kcache, vcache,
                    pos0, M, Hq, Hkv, splits, attention_chunk(), scratch, kv_hs(D), kv_ps(Hkv, D), g_attn_prefetch, td);
  launch_kernel_pdl(pdl_attention_enabled(), "attention_rows_combine", attention_combine_kernel, dim3(M * Hq), dim3(D),
                    0, s, scratch, splits, D, out);
}

void attention_rows_seqs(const float* q, SeqPtrs<const __nv_bfloat16> kc, SeqPtrs<const __nv_bfloat16> vc,
                         const SeqBatch& b, int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s,
                         int fixed_ctx) {
  if (D != 256 || Hq != 6 * Hkv) throw std::runtime_error("attention_rows: built for head_dim 256, 6 query heads per KV head");
  if (b.pos_dev && fixed_ctx <= 0) throw std::runtime_error("attention_rows: a device position needs a fixed context");
  if (b.n < 1 || b.n > kMaxSeqs) throw std::runtime_error("attention_rows_seqs: 1 to kMaxSeqs sequences");
  constexpr int rows = kAttnRows, tk = kAttnTk, ns = kAttnNs;
  // One set of ranges for every sequence: those covering the furthest sequence's keys (or, in a graph, every range
  // up to fixed_ctx). A range past a sequence's last key writes an empty partial that the combine skips, so each
  // sequence's result is bit for bit its own launch's (the argument a captured graph already relies on).
  int ctx = fixed_ctx;
  if (ctx <= 0)
    for (int i = 0; i < b.n; ++i) ctx = std::max(ctx, b.pos[i] + b.row0[i + 1] - b.row0[i]);
  const int splits = attention_rows_splits(ctx);
  AttnBlocks blocks{};
  for (int i = 0; i < b.n; ++i) blocks.blk0[i + 1] = blocks.blk0[i] + ((b.row0[i + 1] - b.row0[i]) * 6 + rows - 1) / rows;
  prepare_kernels();
  launch_kernel_pdl(pdl_attention_enabled(), "attention_rows_seqs",
                    g_attn_bulk ? attention_mma_seqs_kernel<rows, tk, ns, true> : attention_mma_seqs_kernel<rows, tk, ns, false>,
                    dim3(blocks.blk0[b.n], Hkv, splits), dim3(rows * 2), mma_smem(tk, ns), s, q, kc, vc, b, blocks, Hq, Hkv,
                    splits, attention_chunk(), scratch, kv_hs(D), kv_ps(Hkv, D), g_attn_prefetch);
  launch_kernel_pdl(pdl_attention_enabled(), "attention_rows_combine", attention_combine_kernel,
                    dim3(b.row0[b.n] * Hq), dim3(D), 0, s, scratch, splits, D, out);
}

}  // namespace ling::kernels
