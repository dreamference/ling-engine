#include "core/kernels.cuh"

#include <cuda_fp8.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
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

template <int MT>
__global__ void __launch_bounds__(kGemvWarps * 32)
    gemv_nvfp4_kernel(const float* __restrict__ x, int M, const uint8_t* __restrict__ w,
                      const uint8_t* __restrict__ ws, float scale2, float* __restrict__ y, int N, int K) {
  __shared__ float xs[MT * kFp4Stride];
  __shared__ float lut[16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n = blockIdx.x * kGemvWarps + warp;
  if (threadIdx.x < 16) lut[threadIdx.x] = kFp4Values[threadIdx.x];
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;
  const uint8_t* wrow = w + static_cast<size_t>(n) * (K / 2);
  const uint8_t* srow = ws + static_cast<size_t>(n) * (K / 16);

  for (int t0 = 0; t0 < K; t0 += kFp4Tile) {
    __syncthreads();
    for (int i = threadIdx.x; i < MT * kFp4Tile; i += blockDim.x) {
      int m = i / kFp4Tile, k = i % kFp4Tile;
      xs[m * kFp4Stride + k + (k >> 5)] = m < M ? x[static_cast<size_t>(m) * K + t0 + k] : 0.f;
    }
    __syncthreads();
    if (n < N) {
      const int k0 = t0 + lane * 32;
      uint4 packed = __ldg(reinterpret_cast<const uint4*>(wrow + k0 / 2));
      uint16_t sc = __ldg(reinterpret_cast<const uint16_t*>(srow + k0 / 16));
      const float s0 = fp8_to_float(sc & 0xff), s1 = fp8_to_float(sc >> 8);
      const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
      const int base = lane * 33;
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        const float s = wd < 2 ? s0 : s1;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
          const float wv = lut[(words[wd] >> (4 * b)) & 0xf] * s;
          const int kk = base + wd * 8 + b;
#pragma unroll
          for (int m = 0; m < MT; ++m) acc[m] = fmaf(wv, xs[m * kFp4Stride + kk], acc[m]);
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = warp_sum(acc[m]);
    if (lane == 0 && n < N && m < M) y[static_cast<size_t>(m) * N + n] = v * scale2;
  }
}

// ---- FP8 weight-streaming GEMV: tiles of 512 values, 16 per lane. ----
constexpr int kFp8Tile = 512;
constexpr int kFp8Stride = kFp8Tile + kFp8Tile / 32;

template <int MT>
__global__ void __launch_bounds__(kGemvWarps * 32)
    gemv_fp8_kernel(const float* __restrict__ x, int M, const uint8_t* __restrict__ w, float wscale,
                    float* __restrict__ y, int N, int K) {
  __shared__ float xs[MT * kFp8Stride];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n = blockIdx.x * kGemvWarps + warp;
  float acc[MT];
#pragma unroll
  for (int m = 0; m < MT; ++m) acc[m] = 0.f;
  const uint8_t* wrow = w + static_cast<size_t>(n) * K;

  for (int t0 = 0; t0 < K; t0 += kFp8Tile) {
    __syncthreads();
    for (int i = threadIdx.x; i < MT * kFp8Tile; i += blockDim.x) {
      int m = i / kFp8Tile, k = i % kFp8Tile;
      xs[m * kFp8Stride + k + (k >> 5)] = m < M ? x[static_cast<size_t>(m) * K + t0 + k] : 0.f;
    }
    __syncthreads();
    if (n < N) {
      const int k0 = t0 + lane * 16;
      uint4 packed = __ldg(reinterpret_cast<const uint4*>(wrow + k0));
      const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
      const int base = lane * 16 + (lane >> 1);
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float2 f = fp8x2_to_float2(static_cast<uint16_t>(words[wd] >> (16 * h)));
          const int kk = base + wd * 4 + h * 2;
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            acc[m] = fmaf(f.x, xs[m * kFp8Stride + kk], acc[m]);
            acc[m] = fmaf(f.y, xs[m * kFp8Stride + kk + 1], acc[m]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float v = warp_sum(acc[m]);
    if (lane == 0 && n < N && m < M) y[static_cast<size_t>(m) * N + n] = v * wscale;
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

__global__ void __launch_bounds__(kGdnDv)
    gdn_recurrent_kernel(const float* __restrict__ mixed, const float* __restrict__ g,
                         const float* __restrict__ beta, float* __restrict__ state, float* __restrict__ out, int M,
                         int H, int HV) {
  __shared__ float qs[kGdnDk], ks[kGdnDk], red[32];
  const int hv = blockIdx.x, v = threadIdx.x, h = hv / (HV / H);
  const int C = 2 * H * kGdnDk + HV * kGdnDv;
  float* st = state + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
  float S[kGdnDk];
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) S[k] = st[k * kGdnDv + v];
  const float scale = rsqrtf(static_cast<float>(kGdnDk));
  for (int t = 0; t < M; ++t) {
    const float* row = mixed + static_cast<size_t>(t) * C;
    const float qv = row[h * kGdnDk + v], kv_in = row[H * kGdnDk + h * kGdnDk + v];
    const float qn = block_sum(qv * qv, red);
    const float kn = block_sum(kv_in * kv_in, red);
    __syncthreads();
    qs[v] = qv * rsqrtf(qn + 1e-6f) * scale;
    ks[v] = kv_in * rsqrtf(kn + 1e-6f);
    __syncthreads();
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
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) st[k * kGdnDv + v] = S[k];
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
                                    __nv_bfloat16* __restrict__ vcache) {
  extern __shared__ float xn[];
  __shared__ float red[32];
  const int m = blockIdx.x, slot = blockIdx.y, d = threadIdx.x;
  const int pos = pos0 + m;
  if (slot >= Hq + Hkv) {  // v: copy into the cache
    const int kh = slot - Hq - Hkv;
    vcache[(static_cast<size_t>(pos) * Hkv + kh) * D + d] =
        __float2bfloat16(v[(static_cast<size_t>(m) * Hkv + kh) * D + d]);
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
    kcache[(static_cast<size_t>(pos) * Hkv + head) * D + d] = __float2bfloat16(o);
  }
}

constexpr int kAttnSplitKeys = 1024;
constexpr int kAttnMaxSplits = 64;

// Enough key-range splits to fill the GPU when there are few (query, head) pairs, as in decode; none
// when a prefill chunk already supplies thousands of blocks.
int attention_splits(int rows, int ctx) {
  const int by_ctx = (ctx + kAttnSplitKeys - 1) / kAttnSplitKeys;
  const int by_rows = (1024 + rows - 1) / rows;
  return std::clamp(std::min(by_ctx, by_rows), 1, kAttnMaxSplits);
}

// D = 256: each lane owns 8 dims; each of the 8 warps walks its own keys with an online softmax. One block
// serves the G query heads that share a KV head, so each key and value is read once for all of them.
template <int G>
__global__ void __launch_bounds__(256)
    attention_partial_kernel(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kcache,
                             const __nv_bfloat16* __restrict__ vcache, int pos0, int Hq, int Hkv, int splits,
                             float* __restrict__ scratch) {
  constexpr int D = 256;
  __shared__ float wm[8][G], wl[8][G];
  __shared__ float wacc[8][D];
  const int mk = blockIdx.x, split = blockIdx.y;
  const int m = mk / Hkv, kvh = mk % Hkv;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int ctx = pos0 + m + 1;
  const int per = (ctx + splits - 1) / splits;
  const int start = split * per, end = min(ctx, start + per);
  const float scale = rsqrtf(static_cast<float>(D));
  float qv[G][8], acc[G][8], mx[G], l[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float* qrow = q + (static_cast<size_t>(m) * Hq + kvh * G + g) * D + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      qv[g][i] = qrow[i] * scale;
      acc[g][i] = 0.f;
    }
    mx[g] = -INFINITY;
    l[g] = 0.f;
  }
  for (int j = start + warp; j < end; j += 8) {
    const size_t off = (static_cast<size_t>(j) * Hkv + kvh) * D + lane * 8;
    uint4 kr = *reinterpret_cast<const uint4*>(kcache + off);
    uint4 vr = *reinterpret_cast<const uint4*>(vcache + off);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(&kr);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&vr);
    float kf[8], vf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      kf[i] = __bfloat162float(kb[i]);
      vf[i] = __bfloat162float(vb[i]);
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float dot = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) dot = fmaf(qv[g][i], kf[i], dot);
      const float sc = warp_sum(dot);
      const float nm = fmaxf(mx[g], sc);
      const float corr = __expf(mx[g] - nm), p = __expf(sc - nm);
      l[g] = l[g] * corr + p;
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[g][i] = acc[g][i] * corr + p * vf[i];
      mx[g] = nm;
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int g = 0; g < G; ++g) {
      wm[warp][g] = mx[g];
      wl[warp][g] = l[g];
    }
  }
  const int d = threadIdx.x;
#pragma unroll
  for (int g = 0; g < G; ++g) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i) wacc[warp][lane * 8 + i] = acc[g][i];
    __syncthreads();
    float M = -INFINITY;
    for (int w = 0; w < 8; ++w) M = fmaxf(M, wm[w][g]);
    float L = 0.f, A = 0.f;
    if (M != -INFINITY) {
      for (int w = 0; w < 8; ++w) {
        if (wm[w][g] == -INFINITY) continue;
        const float c = __expf(wm[w][g] - M);
        L += wl[w][g] * c;
        A += wacc[w][d] * c;
      }
    }
    float* part = scratch + ((static_cast<size_t>(m) * Hq + kvh * G + g) * splits + split) * (D + 2);
    if (d == 0) {
      part[0] = M;
      part[1] = L;
    }
    part[2 + d] = A;
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

}  // namespace

void gemv_nvfp4(const float* x, int M, const uint8_t* w, const uint8_t* wscale, float scale2, float* y, int N,
                int K, cudaStream_t s) {
  if (K % kFp4Tile != 0) throw std::runtime_error("gemv_nvfp4: K must be a multiple of 1024");
  dim3 grid((N + kGemvWarps - 1) / kGemvWarps), block(kGemvWarps * 32);
  if (M <= 1) gemv_nvfp4_kernel<1><<<grid, block, 0, s>>>(x, M, w, wscale, scale2, y, N, K);
  else if (M <= 2) gemv_nvfp4_kernel<2><<<grid, block, 0, s>>>(x, M, w, wscale, scale2, y, N, K);
  else if (M <= 4) gemv_nvfp4_kernel<4><<<grid, block, 0, s>>>(x, M, w, wscale, scale2, y, N, K);
  else if (M <= 8) gemv_nvfp4_kernel<8><<<grid, block, 0, s>>>(x, M, w, wscale, scale2, y, N, K);
  else throw std::runtime_error("gemv_nvfp4: M > 8");
  LING_LAUNCH_CHECK("gemv_nvfp4");
}

void gemv_fp8(const float* x, int M, const uint8_t* w, float wscale, float* y, int N, int K, cudaStream_t s) {
  if (K % kFp8Tile != 0) throw std::runtime_error("gemv_fp8: K must be a multiple of 512");
  dim3 grid((N + kGemvWarps - 1) / kGemvWarps), block(kGemvWarps * 32);
  if (M <= 1) gemv_fp8_kernel<1><<<grid, block, 0, s>>>(x, M, w, wscale, y, N, K);
  else if (M <= 2) gemv_fp8_kernel<2><<<grid, block, 0, s>>>(x, M, w, wscale, y, N, K);
  else if (M <= 4) gemv_fp8_kernel<4><<<grid, block, 0, s>>>(x, M, w, wscale, y, N, K);
  else if (M <= 8) gemv_fp8_kernel<8><<<grid, block, 0, s>>>(x, M, w, wscale, y, N, K);
  else throw std::runtime_error("gemv_fp8: M > 8");
  LING_LAUNCH_CHECK("gemv_fp8");
}

void gemv_bf16(const float* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s) {
  gemv_bf16_kernel<<<(N + kGemvWarps - 1) / kGemvWarps, kGemvWarps * 32, 0, s>>>(x, M, w, y, N, K);
  LING_LAUNCH_CHECK("gemv_bf16");
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

void gdn_recurrent(const float* mixed, const float* g, const float* beta, float* state, float* out, int M,
                   int H, int HV, cudaStream_t s) {
  gdn_recurrent_kernel<<<HV, kGdnDv, 0, s>>>(mixed, g, beta, state, out, M, H, HV);
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
                                                         theta, eps, q, gate, kcache, vcache);
  LING_LAUNCH_CHECK("attn_prepare");
}

size_t attention_scratch_floats(int M, int Hq, int Hkv, int D, int ctx) {
  return static_cast<size_t>(M) * Hq * attention_splits(M * Hkv, ctx) * (D + 2);
}

void attention(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, int pos0, int M,
               int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s) {
  if (D != 256) throw std::runtime_error("attention: v0 supports head_dim 256 only");
  if (Hq / Hkv != 6 || Hq % Hkv != 0) throw std::runtime_error("attention: v0 is built for 6 query heads per KV head");
  const int splits = attention_splits(M * Hkv, pos0 + M);
  attention_partial_kernel<6><<<dim3(M * Hkv, splits), 256, 0, s>>>(q, kcache, vcache, pos0, Hq, Hkv, splits,
                                                                   scratch);
  LING_LAUNCH_CHECK("attention_partial");
  attention_combine_kernel<<<M * Hq, D, 0, s>>>(scratch, splits, D, out);
  LING_LAUNCH_CHECK("attention_combine");
}

}  // namespace ling::kernels
