#include "core/kernels.cuh"
#include "core/launch.cuh"
#include "core/attention_body.cuh"

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ling::kernels {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

#define LING_LAUNCH_CHECK(name) check(cudaGetLastError(), name)

// warp_sum and block_sum: core/attention_body.cuh.

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

// kGdnDk, kGdnDv: core/attention_body.cuh.

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
  __shared__ int steps[2 * kMaxStreamRows];
  const int bx = blockIdx.x, pth = blockIdx.y;
  if (pth >= td->ngdn) return;  // the launch covers the configuration's most paths
  gdn_recurrent_tree_body(mixed, g, beta, state_in, out, H, HV, td, bx, pth, threadIdx.x, qs, ks, steps);
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
    // A paged KV buffer (P4): each row through its page table; the source and destination may be in different pages.
    auto row = [&](size_t p) {
      return b.pages ? b.base + b.pages[p / kKvPageTokens] * b.page_unit + (p % kKvPageTokens) * b.stride
                     : b.base + p * b.stride;
    };
    const int4* src = reinterpret_cast<const int4*>(row(off + r));
    int4* dst = reinterpret_cast<int4*>(row(off + j));
    for (int i = threadIdx.x; i < b.row_bytes / 16; i += blockDim.x) dst[i] = src[i];
    __syncthreads();
  }
}

// kv_table_set: thread i writes entry i (at most kKvTableSetMax per launch; 3 KB of parameters).
struct KvTableEntries {
  int idx[kKvTableSetMax];
  unsigned long long val[kKvTableSetMax];
};
__global__ void kv_table_set_kernel(uint64_t* __restrict__ table, KvTableEntries e, int n) {
  pdl_begin();  // first statement: see launch.cuh (the previous kernel may still read the old entries)
  const int i = threadIdx.x;
  if (i < n) table[e.idx[i]] = e.val[i];
}

// kv_pages_copy: block (i, lkv) moves position from + i of layer-and-K/V lkv, every KV head, 16 bytes a thread.
__global__ void kv_pages_copy_kernel(__nv_bfloat16* __restrict__ pool, const uint64_t* __restrict__ pages, int Hkv, int D,
                                     __nv_bfloat16* __restrict__ buf, int buf_start, int buf_len, int from, bool to_buf) {
  pdl_begin();  // first statement: see launch.cuh
  const int p = from + blockIdx.x, lkv = blockIdx.y, per_head = D / 8;
  const size_t page = pages[p / kKvPageTokens] + size_t(lkv) * Hkv * kKvPageTokens * D +
                      static_cast<size_t>(p % kKvPageTokens) * D;
  const size_t flat = (size_t(lkv) * Hkv * buf_len + static_cast<size_t>(p - buf_start)) * D;
  for (int i = threadIdx.x; i < Hkv * per_head; i += blockDim.x) {
    const int h = i / per_head, c = i % per_head;
    int4* a = reinterpret_cast<int4*>(pool + page + size_t(h) * kKvPageTokens * D) + c;
    int4* b = reinterpret_cast<int4*>(buf + flat + size_t(h) * buf_len * D) + c;
    if (to_buf) *b = *a;
    else *a = *b;
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

// attn_prepare_body: core/attention_body.cuh.

__global__ void attn_prepare_kernel(const float* __restrict__ q_gate, const float* __restrict__ k,
                                    const float* __restrict__ v, const __nv_bfloat16* __restrict__ q_norm,
                                    const __nv_bfloat16* __restrict__ k_norm, DevPos pos0, int Hq, int Hkv, int D,
                                    int rot, float theta, float eps, float* __restrict__ q,
                                    float* __restrict__ gate, __nv_bfloat16* __restrict__ kcache,
                                    __nv_bfloat16* __restrict__ vcache, size_t hs, size_t ps,
                                    const TreeDesc* __restrict__ td, const int* __restrict__ mrope,
                                    const uint64_t* __restrict__ pages) {
  pdl_begin();  // first statement: see launch.cuh
  const int m = blockIdx.x, p0 = pos_value(pos0);
  // In a tree (BACKLOG 26) the row's position is its depth's; its key and value still go to slot p0 + m. After an
  // image the RoPE position is the slot's plus the sequence's rope delta; a prefill chunk with images reads its rows'
  // (temporal, height, width) from mrope (VISION §20.3 (b)).
  int pt, ph, pw;
  if (mrope) {
    pt = mrope[3 * m];
    ph = mrope[3 * m + 1];
    pw = mrope[3 * m + 2];
  } else {
    pt = ph = pw = (td ? p0 + td->depth[m] : p0 + m) + rope_delta(pos0);
  }
  attn_prepare_body(q_gate, k, v, q_norm, k_norm, m, pt, ph, pw, Hq, Hkv, D, rot, theta, eps, q, gate, kcache, vcache,
                    hs, ps, pages, p0 + m, blockIdx.y);
}

// Several sequences' rows in one launch (M4 stage 4): row m belongs to sequence seq_of(m), at that sequence's
// position, into that sequence's caches.
__global__ void attn_prepare_seqs_kernel(const float* __restrict__ q_gate, const float* __restrict__ k,
                                         const float* __restrict__ v, const __nv_bfloat16* __restrict__ q_norm,
                                         const __nv_bfloat16* __restrict__ k_norm, SeqBatch b, int Hq, int Hkv, int D,
                                         int rot, float theta, float eps, float* __restrict__ q,
                                         float* __restrict__ gate, SeqPtrs<__nv_bfloat16> kc, SeqPtrs<__nv_bfloat16> vc,
                                         size_t hs, size_t ps, SeqPtrs<const uint64_t> pg) {
  pdl_begin();  // first statement: see launch.cuh
  const int m = blockIdx.x, sq = seq_of(b, m);
  const int p = seq_pos(b, sq) + m - b.row0[sq], r = p + seq_rdelta(b, sq);
  attn_prepare_body(q_gate, k, v, q_norm, k_norm, m, r, r, r, Hq, Hkv, D, rot, theta, eps, q, gate, kc.p[sq], vc.p[sq], hs,
                    ps, pg.p[sq], p, blockIdx.y);
}

// The rows path's attention (attention_mma_body and its tiles): core/attention_body.cuh.

template <int ROWS, int TK, int NS, bool BULK>
__global__ void __launch_bounds__(ROWS * 2)
    attention_mma_kernel(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kcache,
                         const __nv_bfloat16* __restrict__ vcache, DevPos pos0_arg, int M, int Hq, int Hkv,
                         int splits, int chunk, float* __restrict__ scratch, size_t hs, size_t ps, int pf,
                         const TreeDesc* __restrict__ td, const uint64_t* __restrict__ pages) {
  pdl_begin();  // first statement: see launch.cuh
  attention_mma_body<ROWS, TK, NS, BULK>(q, kcache, vcache, pos_value(pos0_arg), M, Hq, Hkv, splits, chunk, scratch, hs,
                                         ps, pf, blockIdx.x, blockIdx.y, blockIdx.z, td, pages);
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
                              int splits, int chunk, float* __restrict__ scratch, size_t hs, size_t ps, int pf,
                              SeqPtrs<const uint64_t> pg) {
  pdl_begin();  // first statement: see launch.cuh
  int sq = 0;
  while (sq + 1 < b.n && static_cast<int>(blockIdx.x) >= blocks.blk0[sq + 1]) ++sq;
  constexpr int D = kMmaD;
  const size_t r0 = b.row0[sq];
  attention_mma_body<ROWS, TK, NS, BULK>(q + r0 * Hq * D, kc.p[sq], vc.p[sq], seq_pos(b, sq), b.row0[sq + 1] - b.row0[sq],
                                         Hq, Hkv, splits, chunk, scratch + r0 * Hq * splits * (D + 2), hs, ps, pf,
                                         blockIdx.x - blocks.blk0[sq], blockIdx.y, blockIdx.z, nullptr, pg.p[sq]);
}

__global__ void attention_combine_kernel(const float* __restrict__ scratch, int splits, int D,
                                         float* __restrict__ out) {
  pdl_begin();  // first statement: see launch.cuh
  const int mh = blockIdx.x, d = threadIdx.x;
  out[static_cast<size_t>(mh) * D + d] = attention_combine_body<false>(scratch, splits, D, mh, d);
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
size_t kv_head_stride(int D) { return kv_hs(D); }
size_t kv_pos_stride(int Hkv, int D) { return kv_ps(Hkv, D); }
int attention_prefetch_tiles() { return g_attn_prefetch; }
bool attention_bulk_enabled() { return g_attn_bulk; }
bool pdl_enabled() { return g_pdl; }

__global__ void probe_noop_kernel() { pdl_begin(); }

bool probe_skip(const char* what) {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v;
    const char* e = std::getenv("LING_PROBE_SKIP");
    for (std::string s = e ? e : ""; !s.empty();) {
      const size_t c = s.find(',');
      if (c != 0) v.push_back(s.substr(0, c));
      s = c == std::string::npos ? "" : s.substr(c + 1);
    }
    return v;
  }();
  if (names.empty()) return false;
  const std::string w(what);
  for (const std::string& n : names) {
    if (n.back() == '*' ? w.compare(0, n.size() - 1, n, 0, n.size() - 1) == 0 : w == n) return true;
  }
  return false;
}
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

void kv_pages_copy(__nv_bfloat16* pool, const uint64_t* pages, int layer_kv, int Hkv, int D, __nv_bfloat16* buf,
                   int buf_start, int buf_len, int from, int to, bool to_buf, cudaStream_t s) {
  if (to <= from) return;
  if (D % 8) throw std::runtime_error("kv_pages_copy: head_dim a multiple of 8");
  launch_kernel("kv_pages_copy", kv_pages_copy_kernel, dim3(to - from, layer_kv), dim3(128), 0, s, pool, pages, Hkv, D,
                buf, buf_start, buf_len, from, to_buf);
}

void kv_table_set(uint64_t* table, const int* idx, const uint64_t* val, int n, cudaStream_t s) {
  for (int at = 0; at < n; at += kKvTableSetMax) {
    const int m = std::min(kKvTableSetMax, n - at);
    KvTableEntries e;
    for (int i = 0; i < m; ++i) {
      e.idx[i] = idx[at + i];
      e.val[i] = val[at + i];
    }
    launch_kernel("kv_table_set", kv_table_set_kernel, dim3(1), dim3(kKvTableSetMax), 0, s, table, e, m);
  }
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
                  cudaStream_t s, const TreeDesc* td, const int* mrope, const uint64_t* pages) {
  dim3 grid(M, Hq + 2 * Hkv);
  launch_kernel("attn_prepare", attn_prepare_kernel, grid, dim3(D), D * sizeof(float), s, q_gate, k, v, q_norm, k_norm,
                pos0, Hq, Hkv, D, rot, theta, eps, q, gate, kcache, vcache, kv_hs(D), kv_ps(Hkv, D), td, mrope, pages);
}

void attn_prepare_seqs(const float* q_gate, const float* k, const float* v, const __nv_bfloat16* q_norm,
                       const __nv_bfloat16* k_norm, const SeqBatch& b, int Hq, int Hkv, int D, int rot, float theta,
                       float eps, float* q, float* gate, SeqPtrs<__nv_bfloat16> kc, SeqPtrs<__nv_bfloat16> vc,
                       cudaStream_t s, SeqPtrs<const uint64_t> pages) {
  dim3 grid(b.row0[b.n], Hq + 2 * Hkv);
  launch_kernel("attn_prepare_seqs", attn_prepare_seqs_kernel, grid, dim3(D), D * sizeof(float), s, q_gate, k, v,
                q_norm, k_norm, b, Hq, Hkv, D, rot, theta, eps, q, gate, kc, vc, kv_hs(D), kv_ps(Hkv, D), pages);
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
                    const TreeDesc* td, const uint64_t* pages) {
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
                    pos0, M, Hq, Hkv, splits, attention_chunk(), scratch, kv_hs(D), kv_ps(Hkv, D), g_attn_prefetch, td, pages);
  launch_kernel_pdl(pdl_attention_enabled(), "attention_rows_combine", attention_combine_kernel, dim3(M * Hq), dim3(D),
                    0, s, scratch, splits, D, out);
}

void attention_rows_seqs(const float* q, SeqPtrs<const __nv_bfloat16> kc, SeqPtrs<const __nv_bfloat16> vc,
                         const SeqBatch& b, int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s,
                         int fixed_ctx, SeqPtrs<const uint64_t> pages) {
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
                    splits, attention_chunk(), scratch, kv_hs(D), kv_ps(Hkv, D), g_attn_prefetch, pages);
  launch_kernel_pdl(pdl_attention_enabled(), "attention_rows_combine", attention_combine_kernel,
                    dim3(b.row0[b.n] * Hq), dim3(D), 0, s, scratch, splits, D, out);
}

}  // namespace ling::kernels
