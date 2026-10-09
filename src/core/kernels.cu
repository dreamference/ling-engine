#include "core/kernels.cuh"

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
  const int m = blockIdx.x;
  const __nv_bfloat16* row = table + static_cast<size_t>(ids[m]) * H;
  for (int h = threadIdx.x; h < H; h += blockDim.x) out[static_cast<size_t>(m) * H + h] = __bfloat162float(row[h]);
}

__global__ void rmsnorm_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                               float* __restrict__ out, int H, float eps, bool gemma) {
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
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) h[i] += d[i];
}

__global__ void silu_mul_kernel(const float* g, const float* u, float* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    out[i] = silu(g[i]) * u[i];
}

__global__ void sigmoid_mul_kernel(float* x, const float* gate, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    x[i] *= 1.f / (1.f + __expf(-gate[i]));
}

__global__ void gdn_conv_kernel(float* mixed, float* state, const __nv_bfloat16* __restrict__ w, int M, int C) {
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

__global__ void gdn_gating_kernel(const float* a, const float* b, const __nv_bfloat16* A_log,
                                  const __nv_bfloat16* dt_bias, float* g, float* beta, int n, int HV) {
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
__global__ void __launch_bounds__(32)
    gdn_recurrent_warp_kernel(const float* __restrict__ mixed, const float* __restrict__ g,
                              const float* __restrict__ beta, const float* state_in, float* state_out,
                              float* __restrict__ out, int M, int H, int HV) {
  __shared__ float qs[kGdnDk], ks[kGdnDk];
  const int hv = blockIdx.x / 4, part = blockIdx.x % 4, lane = threadIdx.x, v = part * 32 + lane;
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

__global__ void gated_rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ z,
                                     const __nv_bfloat16* __restrict__ w, float* __restrict__ out, int D,
                                     float eps) {
  __shared__ float red[32];
  const size_t r = blockIdx.x;
  float ss = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) ss += x[r * D + d] * x[r * D + d];
  const float rstd = rsqrtf(block_sum(ss, red) / D + eps);
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    out[r * D + d] = x[r * D + d] * rstd * __bfloat162float(w[d]) * silu(z[r * D + d]);
}

__global__ void attn_prepare_kernel(const float* __restrict__ q_gate, const float* __restrict__ k,
                                    const float* __restrict__ v, const __nv_bfloat16* __restrict__ q_norm,
                                    const __nv_bfloat16* __restrict__ k_norm, int pos0, int Hq, int Hkv, int D,
                                    int rot, float theta, float eps, float* __restrict__ q,
                                    float* __restrict__ gate, __nv_bfloat16* __restrict__ kcache,
                                    __nv_bfloat16* __restrict__ vcache, size_t hs, size_t ps) {
  extern __shared__ float xn[];
  __shared__ float red[32];
  const int m = blockIdx.x, slot = blockIdx.y, d = threadIdx.x;
  const int pos = pos0 + m;
  if (slot >= Hq + Hkv) {  // v: copy into the cache
    const int kh = slot - Hq - Hkv;
    vcache[kh * hs + pos * ps + d] = __float2bfloat16(v[(static_cast<size_t>(m) * Hkv + kh) * D + d]);
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
    kcache[head * hs + pos * ps + d] = __float2bfloat16(o);
  }
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

template <int ROWS, int TK, int NS>  // query rows per block (16 per warp), keys per tile, pipeline stages
__global__ void __launch_bounds__(ROWS * 2)
    attention_mma_kernel(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kcache,
                         const __nv_bfloat16* __restrict__ vcache, int pos0, int M, int Hq, int Hkv, int splits,
                         int chunk, float* __restrict__ scratch, size_t hs, size_t ps) {
  constexpr int D = kMmaD, G = 6, KS = kMmaKS;
  extern __shared__ __align__(16) unsigned char smem[];
  auto stage = [&](int i) { return reinterpret_cast<__nv_bfloat16*>(smem + (i % NS) * mma_stage(TK)); };
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int kvh = blockIdx.y, split = blockIdx.z;
  const int R = M * G, r0 = blockIdx.x * ROWS, nrows = min(ROWS, R - r0);
  const bool active = warp * 16 < nrows;
  const int last_m = (r0 + nrows - 1) / G;
  const int kend = pos0 + last_m + 1;  // keys [0, kend) are visible to some row of this block
  const int klo = split * chunk, khi = min(kend, klo + chunk);
  // This thread's two rows (g and g + 8 of the warp's 16), their positions, and their queries as MMA
  // A fragments in BF16, scaled by 1 / sqrt(D) (a power of two: exact).
  const int ra = r0 + warp * 16 + g, rb = ra + 8;
  const int pa = pos0 + ra / G, pb = pos0 + rb / G;
  const bool va = ra < r0 + nrows, vb = rb < r0 + nrows;
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
  // "tile i has landed".
  const int ntiles = (khi - klo + TK - 1) / TK;
#pragma unroll
  for (int p = 0; p < NS - 1; ++p) {
    if (p < ntiles) attn_load_tile<TK>(stage(p), kcache, vcache, klo + p * TK, khi, kvh, hs, ps);
    else asm volatile("cp.async.commit_group;");
  }
  for (int it = 0; it < ntiles; ++it) {
    const int k0 = klo + it * TK;
    asm volatile("cp.async.wait_group %0;" ::"n"(NS - 2));
    __syncthreads();  // tile it is visible to all, and stage (it - 1) % NS is free again
    if (it + NS - 1 < ntiles) attn_load_tile<TK>(stage(it + NS - 1), kcache, vcache, k0 + (NS - 1) * TK, khi, kvh, hs, ps);
    else asm volatile("cp.async.commit_group;");
    const __nv_bfloat16* ks = stage(it);
    const __nv_bfloat16* vs = ks + TK * KS;
    if (active) {
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
      la = la * alpha_a + sa;
      lb = lb * alpha_b + sb;
      ma = na;
      mb = nb;
#pragma unroll
      for (int i = 0; i < D / 8; ++i) {
        o[i][0] *= alpha_a;
        o[i][1] *= alpha_a;
        o[i][2] *= alpha_b;
        o[i][3] *= alpha_b;
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

__global__ void attention_combine_kernel(const float* __restrict__ scratch, int splits, int D,
                                         float* __restrict__ out) {
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

}  // namespace

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
  bf16_rows_kernel<float><<<w2 ? 2 * N : N, kBf16RowsThreads, 0, s>>>(x, M, w, y, w2, y2, N, K);
  LING_LAUNCH_CHECK("bf16_rows");
}

void bf16_rows_bf16in(const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s) {
  if (M < 1 || M > 32 || K % (kBf16RowsThreads * 8) != 0) throw std::runtime_error("bf16_rows: M <= 32, K % 1024 == 0");
  bf16_rows_kernel<__nv_bfloat16><<<N, kBf16RowsThreads, 0, s>>>(x, M, w, y, nullptr, nullptr, N, K);
  LING_LAUNCH_CHECK("bf16_rows_bf16in");
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

void gemm_bf16_cublas(cublasHandle_t h, const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N,
                      int K) {
  const float alpha = 1.f, beta = 0.f;
  cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16BF, K, x,
                                   CUDA_R_16BF, K, &beta, y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT);
  if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasGemmEx failed: " + std::to_string(st));
}

void embed(const __nv_bfloat16* table, const int* ids, int M, int H, float* out, cudaStream_t s) {
  embed_kernel<<<M, 256, 0, s>>>(table, ids, H, out);
  LING_LAUNCH_CHECK("embed");
}

void rmsnorm(const float* x, const __nv_bfloat16* w, float* out, int rows, int H, float eps, bool gemma,
             cudaStream_t s) {
  rmsnorm_kernel<<<rows, H >= 1024 ? 1024 : H, 0, s>>>(x, w, out, H, eps, gemma);
  LING_LAUNCH_CHECK("rmsnorm");
}

void add_inplace(float* h, const float* d, int n, cudaStream_t s) {
  add_kernel<<<grid_for(n, 256), 256, 0, s>>>(h, d, n);
  LING_LAUNCH_CHECK("add");
}

void silu_mul(const float* g, const float* u, float* out, int n, cudaStream_t s) {
  silu_mul_kernel<<<grid_for(n, 256), 256, 0, s>>>(g, u, out, n);
  LING_LAUNCH_CHECK("silu_mul");
}

void sigmoid_mul(float* x, const float* gate, int n, cudaStream_t s) {
  sigmoid_mul_kernel<<<grid_for(n, 256), 256, 0, s>>>(x, gate, n);
  LING_LAUNCH_CHECK("sigmoid_mul");
}

void gdn_conv(float* mixed, float* conv_state, const __nv_bfloat16* w, int M, int C, cudaStream_t s) {
  gdn_conv_kernel<<<(C + 255) / 256, 256, 0, s>>>(mixed, conv_state, w, M, C);
  LING_LAUNCH_CHECK("gdn_conv");
}

void gdn_gating(const float* a, const float* b, const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
                float* g, float* beta, int M, int HV, cudaStream_t s) {
  const int n = M * HV;
  gdn_gating_kernel<<<(n + 255) / 256, 256, 0, s>>>(a, b, A_log, dt_bias, g, beta, n, HV);
  LING_LAUNCH_CHECK("gdn_gating");
}

void gdn_recurrent(const float* mixed, const float* g, const float* beta, const float* state_in, float* state_out,
                   float* out, int M, int H, int HV, cudaStream_t s) {
  gdn_recurrent_warp_kernel<<<HV * 4, 32, 0, s>>>(mixed, g, beta, state_in, state_out, out, M, H, HV);
  LING_LAUNCH_CHECK("gdn_recurrent");
}

void gated_rmsnorm(const float* x, const float* z, const __nv_bfloat16* w, float* out, int rows, int D,
                   float eps, cudaStream_t s) {
  gated_rmsnorm_kernel<<<rows, D, 0, s>>>(x, z, w, out, D, eps);
  LING_LAUNCH_CHECK("gated_rmsnorm");
}

void attn_prepare(const float* q_gate, const float* k, const float* v, const __nv_bfloat16* q_norm,
                  const __nv_bfloat16* k_norm, int pos0, int M, int Hq, int Hkv, int D, int rot, float theta,
                  float eps, float* q, float* gate, __nv_bfloat16* kcache, __nv_bfloat16* vcache,
                  cudaStream_t s) {
  dim3 grid(M, Hq + 2 * Hkv);
  attn_prepare_kernel<<<grid, D, D * sizeof(float), s>>>(q_gate, k, v, q_norm, k_norm, pos0, Hq, Hkv, D, rot,
                                                         theta, eps, q, gate, kcache, vcache, kv_hs(D), kv_ps(Hkv, D));
  LING_LAUNCH_CHECK("attn_prepare");
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

void attention_rows(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, int pos0, int M, int Hq,
                    int Hkv, int D, float* scratch, float* out, cudaStream_t s) {
  if (D != 256 || Hq != 6 * Hkv) throw std::runtime_error("attention_rows: built for head_dim 256, 6 query heads per KV head");
  const int splits = attention_rows_splits(pos0 + M);
  // 32-key tiles, double-buffered. Measured at 24K context, 16 rows, realistic data: 0.57 ms per layer;
  // 16-key tiles with 3-5 stages were no faster.
  constexpr int rows = 96, tk = 32, ns = 2;
  static bool configured = false;
  if (!configured) {
    check(cudaFuncSetAttribute(attention_mma_kernel<rows, tk, ns>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(mma_smem(tk, ns))),
          "attention_rows smem");
    configured = true;
  }
  const int R = M * 6;
  attention_mma_kernel<rows, tk, ns><<<dim3((R + rows - 1) / rows, Hkv, splits), rows * 2, mma_smem(tk, ns), s>>>(
      q, kcache, vcache, pos0, M, Hq, Hkv, splits, attention_chunk(), scratch, kv_hs(D), kv_ps(Hkv, D));
  LING_LAUNCH_CHECK("attention_rows");
  attention_combine_kernel<<<M * Hq, D, 0, s>>>(scratch, splits, D, out);
  LING_LAUNCH_CHECK("attention_rows_combine");
}

}  // namespace ling::kernels
