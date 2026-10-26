// The vision tower (core/vision.hpp; specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.3 (a)).
#include "core/vision.hpp"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/kernels.cuh"
#include "core/safetensors.hpp"

namespace ling {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("vision: ") + what + ": " + cudaGetErrorString(e));
}

constexpr int kChunk = 8192;  // rows per GEMM call into the FP32 scratch

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ __nv_bfloat16 rb(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float gelu_tanh(float x) {
  // torch's gelu(approximate="tanh"): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  const float inner = k0 * (x + k1 * x * x * x);
  return 0.5f * x * (1.f + tanhf(inner));
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.7071067811865476f)); }

// ---- LayerNorm: one block per row, FP32 statistics (two passes over the row in registers/shared memory). ----
__global__ void layernorm_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                 const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int cols,
                                 float eps) {
  const size_t r = blockIdx.x;
  const __nv_bfloat16* xr = x + r * cols;
  __shared__ float red[32];
  __shared__ float stat;
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += bf(xr[c]);
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  if (lane == 0) red[warp] = s;
  __syncthreads();
  if (warp == 0) {
    float t = lane < nw ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) stat = t / cols;
  }
  __syncthreads();
  const float mean = stat;
  float v = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float d = bf(xr[c]) - mean;
    v += d * d;
  }
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = lane < nw ? red[lane] : 0.f;
    for (int o = 16; o; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
    if (lane == 0) stat = rsqrtf(t / cols + eps);
  }
  __syncthreads();
  const float rstd = stat;
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    out[r * cols + c] = rb((bf(xr[c]) - mean) * rstd * bf(w[c]) + bf(b[c]));
}

// ---- Epilogue: bias, activation, BF16 rounding, optional residual add. ----
__global__ void bias_act_kernel(const float* __restrict__ y, const __nv_bfloat16* __restrict__ b, int M, int N,
                                __nv_bfloat16* __restrict__ out, int ld_out, int act, bool residual) {
  const size_t total = size_t(M) * ld_out;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const int r = static_cast<int>(i / ld_out), c = static_cast<int>(i % ld_out);
    if (c >= N) {
      if (!residual) out[i] = rb(0.f);
      continue;
    }
    float v = bf(rb(y[size_t(r) * N + c] + bf(b[c])));  // the linear layer's BF16 output
    if (act == 1) v = bf(rb(gelu_tanh(v)));
    else if (act == 2) v = bf(rb(gelu_erf(v)));
    out[i] = residual ? rb(bf(out[i]) + v) : rb(v);
  }
}

// ---- The interpolated position embedding, added to the patch embeddings. ----
__global__ void add_pos_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ table,
                               const int* __restrict__ idx, const __nv_bfloat16* __restrict__ wt, int n, int hidden) {
  const int p = blockIdx.x;
  for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const size_t k = size_t(i) * n + p;
      s += bf(rb(bf(table[size_t(idx[k]) * hidden + c]) * bf(wt[k])));  // embeds *= weights (BF16)
    }
    const size_t o = size_t(p) * hidden + c;
    x[o] = rb(bf(x[o]) + bf(rb(s)));  // .sum(dim=0) rounds once; x += pos_embeds
  }
}

// ---- Rotary on q and k in place. ----
__global__ void rotary_kernel(__nv_bfloat16* __restrict__ qkv, const int* __restrict__ pos, int heads) {
  const int n = blockIdx.x;
  const int hidden = heads * 72;
  const float row = static_cast<float>(pos[2 * n]), col = static_cast<float>(pos[2 * n + 1]);
  // threads: (q or k) x heads x 36 pairs
  for (int t = threadIdx.x; t < 2 * heads * 36; t += blockDim.x) {
    const int part = t / (heads * 36), rem = t % (heads * 36), h = rem / 36, d = rem % 36;
    const int j = d < 18 ? d : d - 18;
    const float inv = 1.f / powf(10000.f, static_cast<float>(2 * j) / 36.f);
    const float ang = (d < 18 ? row : col) * inv;
    const float c = cosf(ang), s = sinf(ang);
    __nv_bfloat16* base = qkv + size_t(n) * 3 * hidden + size_t(part) * hidden + size_t(h) * 72;
    const float x1 = bf(base[d]), x2 = bf(base[d + 36]);
    base[d] = rb(x1 * c + (-x2) * s);
    base[d + 36] = rb(x2 * c + x1 * s);
  }
}

// ---- Attention (FlashAttention-2 form) for head dimension 72, bidirectional. ----
constexpr int kAD = 72;     // head dimension
constexpr int kAStr = 88;   // shared-memory row stride (BF16): 80 used (72 + zero pad to a 16-multiple) + 8 for banks
constexpr int kABQ = 64;    // queries per block (4 warps x 16)
constexpr int kABK = 64;    // keys per tile
constexpr size_t kATile = size_t(kABK) * kAStr * sizeof(__nv_bfloat16);
constexpr size_t kASmem = kATile * 5;  // Q, K x 2, V x 2

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x2_trans(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(smem_u32(p)));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// Loads rows [row0, row0 + 64) of one head's q, k or v (part 0, 1, 2) into a tile; rows past n are zeros.
__device__ __forceinline__ void load_tile(__nv_bfloat16* tile, const __nv_bfloat16* qkv, int part, int head, int heads,
                                          int row0, int n) {
  const int hidden = heads * kAD;
  for (int i = threadIdx.x; i < kABK * 9; i += blockDim.x) {
    const int r = i / 9, c = i % 9;
    const int row = row0 + r;
    const bool ok = row < n;
    const __nv_bfloat16* src = qkv + size_t(ok ? row : 0) * 3 * hidden + size_t(part) * hidden + head * kAD + c * 8;
    cp_async16(tile + r * kAStr + c * 8, src, ok);
  }
}

__global__ void __launch_bounds__(128) attention_kernel(const __nv_bfloat16* __restrict__ qkv,
                                                        __nv_bfloat16* __restrict__ out, int n, int heads,
                                                        float scale_log2) {
  extern __shared__ __align__(16) uint8_t smem[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Ks[2] = {Qs + kABK * kAStr, Qs + 2 * kABK * kAStr};
  __nv_bfloat16* Vs[2] = {Qs + 3 * kABK * kAStr, Qs + 4 * kABK * kAStr};
  const int head = blockIdx.y, q0 = blockIdx.x * kABQ;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  // Zero the pad columns 72 .. 79 of Q and both K stages (cp.async never writes them; Q x K over them must be 0).
  if (tid < 64) {
    *reinterpret_cast<uint4*>(Qs + tid * kAStr + kAD) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(Ks[0] + tid * kAStr + kAD) = make_uint4(0, 0, 0, 0);
  } else {
    *reinterpret_cast<uint4*>(Ks[1] + (tid - 64) * kAStr + kAD) = make_uint4(0, 0, 0, 0);
  }
  load_tile(Qs, qkv, 0, head, heads, q0, n);
  load_tile(Ks[0], qkv, 1, head, heads, 0, n);
  load_tile(Vs[0], qkv, 2, head, heads, 0, n);
  cp_commit();
  const int tiles = (n + kABK - 1) / kABK;
  uint32_t qa[5][4];
  float o[9][4];
#pragma unroll
  for (int i = 0; i < 9; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  for (int j = 0; j < tiles; ++j) {
    const int st = j & 1;
    if (j + 1 < tiles) {
      load_tile(Ks[st ^ 1], qkv, 1, head, heads, (j + 1) * kABK, n);
      load_tile(Vs[st ^ 1], qkv, 2, head, heads, (j + 1) * kABK, n);
      cp_commit();
      cp_wait<1>();
    } else {
      cp_wait<0>();
    }
    __syncthreads();
    if (j == 0) {
      const __nv_bfloat16* qr = Qs + (warp * 16 + g) * kAStr;
#pragma unroll
      for (int kk = 0; kk < 5; ++kk) {
        qa[kk][0] = *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 2 * t);
        qa[kk][1] = *reinterpret_cast<const uint32_t*>(qr + 8 * kAStr + kk * 16 + 2 * t);
        qa[kk][2] = *reinterpret_cast<const uint32_t*>(qr + kk * 16 + 2 * t + 8);
        qa[kk][3] = *reinterpret_cast<const uint32_t*>(qr + 8 * kAStr + kk * 16 + 2 * t + 8);
      }
    }
    // S = Q K^T for this warp's 16 rows and the tile's 64 keys.
    float s[8][4];
    const __nv_bfloat16* K = Ks[st];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
      const __nv_bfloat16* kr = K + (nt * 8 + g) * kAStr;
#pragma unroll
      for (int kk = 0; kk < 5; ++kk) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr + kk * 16 + 2 * t);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + kk * 16 + 2 * t + 8);
        mma16816(s[nt], qa[kk], b0, b1);
      }
    }
    // Scale (log2 domain), mask keys past n, the rows' running max.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const int key = j * kABK + nt * 8 + 2 * t;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const bool valid = key + (e & 1) < n;
        s[nt][e] = valid ? s[nt][e] * scale_log2 : -INFINITY;
      }
      mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
      mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    const float nm0 = fmaxf(m0, mx0), nm1 = fmaxf(m1, mx1);
    const float base0 = nm0 == -INFINITY ? 0.f : nm0, base1 = nm1 == -INFINITY ? 0.f : nm1;
    const float c0 = exp2f(m0 - base0), c1 = exp2f(m1 - base1);
    m0 = nm0;
    m1 = nm1;
    l0 *= c0;
    l1 *= c1;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      o[i][0] *= c0;
      o[i][1] *= c0;
      o[i][2] *= c1;
      o[i][3] *= c1;
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      s[nt][0] = exp2f(s[nt][0] - base0);
      s[nt][1] = exp2f(s[nt][1] - base0);
      s[nt][2] = exp2f(s[nt][2] - base1);
      s[nt][3] = exp2f(s[nt][3] - base1);
      l0 += s[nt][0] + s[nt][1];
      l1 += s[nt][2] + s[nt][3];
    }
    // O += P V.
    const __nv_bfloat16* V = Vs[st];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      uint32_t pa[4];
      pa[0] = pack_bf16(s[2 * kk][0], s[2 * kk][1]);
      pa[1] = pack_bf16(s[2 * kk][2], s[2 * kk][3]);
      pa[2] = pack_bf16(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      pa[3] = pack_bf16(s[2 * kk + 1][2], s[2 * kk + 1][3]);
      const __nv_bfloat16* vr = V + (kk * 16 + (lane & 15)) * kAStr;
#pragma unroll
      for (int dn = 0; dn < 9; ++dn) {
        uint32_t b0, b1;
        ldsm_x2_trans(b0, b1, vr + dn * 8);
        mma16816(o[dn], pa, b0, b1);
      }
    }
    __syncthreads();  // the stage just read is the next prefetch's target
  }
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  const int r0 = q0 + warp * 16 + g, r1 = r0 + 8;
  const int hidden = heads * kAD;
#pragma unroll
  for (int dn = 0; dn < 9; ++dn) {
    const int col = head * kAD + dn * 8 + 2 * t;
    if (r0 < n)
      *reinterpret_cast<__nv_bfloat162*>(out + size_t(r0) * hidden + col) =
          __floats2bfloat162_rn(o[dn][0] * i0, o[dn][1] * i0);
    if (r1 < n)
      *reinterpret_cast<__nv_bfloat162*>(out + size_t(r1) * hidden + col) =
          __floats2bfloat162_rn(o[dn][2] * i1, o[dn][3] * i1);
  }
}

int grid_for(size_t n, int block = 256) { return static_cast<int>(std::min<size_t>((n + block - 1) / block, 65535 * 8)); }

}  // namespace

namespace vision_kernels {

void layernorm(const __nv_bfloat16* x, const __nv_bfloat16* w, const __nv_bfloat16* b, __nv_bfloat16* out, int rows,
               int cols, float eps, cudaStream_t s) {
  if (rows < 1) return;
  layernorm_kernel<<<rows, 256, 0, s>>>(x, w, b, out, cols, eps);
  check(cudaGetLastError(), "layernorm");
}

void rotary(__nv_bfloat16* qkv, const int* pos, int n, int heads, cudaStream_t s) {
  if (n < 1) return;
  rotary_kernel<<<n, 256, 0, s>>>(qkv, pos, heads);
  check(cudaGetLastError(), "rotary");
}

void attention(const __nv_bfloat16* qkv, __nv_bfloat16* out, int n, int heads, cudaStream_t s) {
  if (n < 1) return;
  static bool attr = false;
  if (!attr) {
    check(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kASmem)),
          "attention smem");
    attr = true;
  }
  const float scale_log2 = (1.f / std::sqrt(float(kAD))) * 1.4426950408889634f;
  attention_kernel<<<dim3((n + kABQ - 1) / kABQ, heads), 128, kASmem, s>>>(qkv, out, n, heads, scale_log2);
  check(cudaGetLastError(), "attention");
}

void bias_act(const float* y, const __nv_bfloat16* b, int M, int N, __nv_bfloat16* out, int ld_out, int act,
              bool residual, cudaStream_t s) {
  if (M < 1) return;
  bias_act_kernel<<<grid_for(size_t(M) * ld_out), 256, 0, s>>>(y, b, M, N, out, ld_out, act, residual);
  check(cudaGetLastError(), "bias_act");
}

void add_pos(__nv_bfloat16* x, const __nv_bfloat16* table, const int* idx, const __nv_bfloat16* wt, int n, int hidden,
             cudaStream_t s) {
  if (n < 1) return;
  add_pos_kernel<<<n, 256, 0, s>>>(x, table, idx, wt, n, hidden);
  check(cudaGetLastError(), "add_pos");
}

}  // namespace vision_kernels

std::vector<float> torch_linspace(float start, float end, int steps) {
  std::vector<float> v(std::max(steps, 0));
  if (steps == 1) v[0] = start;
  if (steps <= 1) return v;
  const float step = (end - start) / static_cast<float>(steps - 1);
  const int halfway = steps / 2;
  for (int i = 0; i < steps; ++i)
    v[i] = i < halfway ? start + step * static_cast<float>(i) : end - step * static_cast<float>(steps - i - 1);
  return v;
}

void vision_positions(int gh, int gw, int merge, int side, std::vector<int>& idx4, std::vector<float>& wt4,
                      std::vector<int>& rowcol) {
  const int n = gh * gw;
  idx4.assign(size_t(4) * n, 0);
  wt4.assign(size_t(4) * n, 0.f);
  rowcol.assign(size_t(2) * n, 0);
  const std::vector<float> hs = torch_linspace(0.f, float(side - 1), gh), ws = torch_linspace(0.f, float(side - 1), gw);
  int p = 0;
  for (int bh = 0; bh < gh / merge; ++bh)
    for (int bw = 0; bw < gw / merge; ++bw)
      for (int mh = 0; mh < merge; ++mh)
        for (int mw = 0; mw < merge; ++mw, ++p) {
          const int r = bh * merge + mh, c = bw * merge + mw;
          const int hf = static_cast<int>(hs[r]), wf = static_cast<int>(ws[c]);
          const int hc = std::min(hf + 1, side - 1), wc = std::min(wf + 1, side - 1);
          const float dh = hs[r] - static_cast<float>(hf), dw = ws[c] - static_cast<float>(wf);
          // production's form: w11 = dh dw, w10 = dh - w11, w01 = dw - w11, w00 = 1 - dh - w01
          const float w11 = dh * dw, w10 = dh - w11, w01 = dw - w11, w00 = 1.f - dh - w01;
          const int idx[4] = {hf * side + wf, hf * side + wc, hc * side + wf, hc * side + wc};
          const float wt[4] = {w00, w01, w10, w11};
          for (int i = 0; i < 4; ++i) {
            idx4[size_t(i) * n + p] = idx[i];
            wt4[size_t(i) * n + p] = wt[i];
          }
          rowcol[2 * size_t(p)] = r;
          rowcol[2 * size_t(p) + 1] = c;
        }
}

const __nv_bfloat16* VisionTower::upload(const Checkpoint& ck, const std::string& name, std::vector<int64_t> shape) {
  const TensorView& t = ck.get(name);
  if (t.dtype != "BF16") throw std::runtime_error("vision: " + name + " is " + t.dtype + ", expected BF16");
  int64_t numel = 1;
  for (int64_t d : shape) numel *= d;
  if (t.numel() != numel) throw std::runtime_error("vision: " + name + " has an unexpected shape");
  void* p = nullptr;
  check(cudaMalloc(&p, t.bytes), "weights");
  check(cudaMemcpy(p, t.data, t.bytes, cudaMemcpyHostToDevice), "weights");
  owned_.push_back(p);
  weight_bytes_ += t.bytes;
  return static_cast<const __nv_bfloat16*>(p);
}

VisionTower::VisionTower(const std::string& model_dir, int max_patches) : max_patches_(max_patches) {
  if (max_patches < 4 || max_patches % 4 != 0) throw std::runtime_error("vision: max_patches must be a multiple of 4");
  Checkpoint ck(model_dir);
  const VisionConfig& c = cfg_;
  const std::string v = "model.visual.";
  if (!ck.has(v + "patch_embed.proj.weight")) throw std::runtime_error("vision: the checkpoint has no vision tower");
  patch_w_ = upload(ck, v + "patch_embed.proj.weight", {c.hidden, c.patch_dim()});
  patch_b_ = upload(ck, v + "patch_embed.proj.bias", {c.hidden});
  pos_ = upload(ck, v + "pos_embed.weight", {c.pos_side * c.pos_side, c.hidden});
  for (int i = 0; i < c.depth; ++i) {
    const std::string b = v + "blocks." + std::to_string(i) + ".";
    Block k{};
    k.n1w = upload(ck, b + "norm1.weight", {c.hidden});
    k.n1b = upload(ck, b + "norm1.bias", {c.hidden});
    k.n2w = upload(ck, b + "norm2.weight", {c.hidden});
    k.n2b = upload(ck, b + "norm2.bias", {c.hidden});
    k.qkv_w = upload(ck, b + "attn.qkv.weight", {3 * c.hidden, c.hidden});
    k.qkv_b = upload(ck, b + "attn.qkv.bias", {3 * c.hidden});
    k.proj_w = upload(ck, b + "attn.proj.weight", {c.hidden, c.hidden});
    k.proj_b = upload(ck, b + "attn.proj.bias", {c.hidden});
    k.fc1_w = upload(ck, b + "mlp.linear_fc1.weight", {c.inter, c.hidden});
    k.fc1_b = upload(ck, b + "mlp.linear_fc1.bias", {c.inter});
    k.fc2_b = upload(ck, b + "mlp.linear_fc2.bias", {c.hidden});
    {  // fc2 [hidden][inter] padded to [hidden][inter_pad] with zero columns
      const TensorView& t = ck.get(b + "mlp.linear_fc2.weight");
      if (t.dtype != "BF16" || t.numel() != int64_t(c.hidden) * c.inter) throw std::runtime_error("vision: fc2 shape");
      std::vector<uint16_t> pad(size_t(c.hidden) * c.inter_pad, 0);
      for (int r = 0; r < c.hidden; ++r)
        std::memcpy(pad.data() + size_t(r) * c.inter_pad, t.data + size_t(r) * c.inter * 2, size_t(c.inter) * 2);
      void* p = nullptr;
      check(cudaMalloc(&p, pad.size() * 2), "fc2");
      check(cudaMemcpy(p, pad.data(), pad.size() * 2, cudaMemcpyHostToDevice), "fc2");
      owned_.push_back(p);
      weight_bytes_ += pad.size() * 2;
      k.fc2_w = static_cast<const __nv_bfloat16*>(p);
    }
    blocks_.push_back(k);
  }
  mn_w_ = upload(ck, v + "merger.norm.weight", {c.hidden});
  mn_b_ = upload(ck, v + "merger.norm.bias", {c.hidden});
  m1_w_ = upload(ck, v + "merger.linear_fc1.weight", {c.merged(), c.merged()});
  m1_b_ = upload(ck, v + "merger.linear_fc1.bias", {c.merged()});
  m2_w_ = upload(ck, v + "merger.linear_fc2.weight", {c.out, c.merged()});
  m2_b_ = upload(ck, v + "merger.linear_fc2.bias", {c.out});

  auto alloc = [&](size_t bytes) {
    void* p = nullptr;
    check(cudaMalloc(&p, bytes), "scratch");
    owned_.push_back(p);
    scratch_bytes_ += bytes;
    return p;
  };
  const size_t P = max_patches;
  x_ = static_cast<__nv_bfloat16*>(alloc(P * c.hidden * 2));
  xn_ = static_cast<__nv_bfloat16*>(alloc(P * c.hidden * 2));
  qkv_ = static_cast<__nv_bfloat16*>(alloc(P * std::max(3 * c.hidden, c.patch_dim()) * 2));
  attn_ = static_cast<__nv_bfloat16*>(alloc(P * c.hidden * 2));
  const size_t chunk = std::min<size_t>(kChunk, P);
  act_ = static_cast<__nv_bfloat16*>(alloc(chunk * std::max(c.inter_pad, c.merged()) * 2));
  y_ = static_cast<float*>(alloc(chunk * std::max({3 * c.hidden, c.inter_pad, c.merged(), c.out}) * 4));
  pos_idx_ = static_cast<int*>(alloc(4 * P * sizeof(int)));
  pos_wt_ = static_cast<__nv_bfloat16*>(alloc(4 * P * 2));
  rot_pos_ = static_cast<int*>(alloc(2 * P * sizeof(int)));
}

VisionTower::~VisionTower() {
  for (void* p : owned_) cudaFree(p);
}

void VisionTower::linear(const __nv_bfloat16* x, int M, int K, const __nv_bfloat16* w, const __nv_bfloat16* b, int N,
                         __nv_bfloat16* out, int ld_out, int act, bool residual, cudaStream_t s) {
  for (int r0 = 0; r0 < M; r0 += kChunk) {
    const int m = std::min(kChunk, M - r0);
    kernels::gemm_bf16(x + size_t(r0) * K, m, w, y_, N, K, s);
    vision_kernels::bias_act(y_, b, m, N, out + size_t(r0) * ld_out, ld_out, act, residual, s);
  }
}

void VisionTower::forward(const uint16_t* patches, int gh, int gw, __nv_bfloat16* out, cudaStream_t s, int tap,
                          std::vector<uint16_t>* tap_out) {
  const VisionConfig& c = cfg_;
  if (gh < 2 || gw < 2 || gh % c.merge || gw % c.merge) throw std::runtime_error("vision: the grid must be even");
  const int n = gh * gw;
  if (n > max_patches_)
    throw std::runtime_error("vision: " + std::to_string(n) + " patches, more than the " + std::to_string(max_patches_) +
                             " the tower was sized for");
  // Patches in: the BF16 bit patterns straight into the qkv scratch.
  check(cudaMemcpyAsync(qkv_, patches, size_t(n) * c.patch_dim() * 2, cudaMemcpyHostToDevice, s), "patches");
  // Positions.
  std::vector<int> idx4, rowcol;
  std::vector<float> wt4;
  vision_positions(gh, gw, c.merge, c.pos_side, idx4, wt4, rowcol);
  std::vector<__nv_bfloat16> wtb(wt4.size());
  for (size_t i = 0; i < wt4.size(); ++i) wtb[i] = __float2bfloat16_rn(wt4[i]);
  check(cudaMemcpyAsync(pos_idx_, idx4.data(), idx4.size() * sizeof(int), cudaMemcpyHostToDevice, s), "pos idx");
  check(cudaMemcpyAsync(pos_wt_, wtb.data(), wtb.size() * 2, cudaMemcpyHostToDevice, s), "pos wt");
  check(cudaMemcpyAsync(rot_pos_, rowcol.data(), rowcol.size() * sizeof(int), cudaMemcpyHostToDevice, s), "rot pos");

  auto tap_copy = [&](int at) {
    if (tap != at || !tap_out) return;
    tap_out->resize(size_t(n) * c.hidden);
    check(cudaMemcpyAsync(tap_out->data(), x_, tap_out->size() * 2, cudaMemcpyDeviceToHost, s), "tap");
  };
  // Patch embedding, then the interpolated position embedding.
  linear(qkv_, n, c.patch_dim(), patch_w_, patch_b_, c.hidden, x_, c.hidden, 0, false, s);
  vision_kernels::add_pos(x_, pos_, pos_idx_, pos_wt_, n, c.hidden, s);
  tap_copy(-1);
  for (int i = 0; i < c.depth; ++i) {
    const Block& k = blocks_[i];
    vision_kernels::layernorm(x_, k.n1w, k.n1b, xn_, n, c.hidden, 1e-6f, s);
    linear(xn_, n, c.hidden, k.qkv_w, k.qkv_b, 3 * c.hidden, qkv_, 3 * c.hidden, 0, false, s);
    vision_kernels::rotary(qkv_, rot_pos_, n, c.heads, s);
    vision_kernels::attention(qkv_, attn_, n, c.heads, s);
    linear(attn_, n, c.hidden, k.proj_w, k.proj_b, c.hidden, x_, c.hidden, 0, true, s);
    vision_kernels::layernorm(x_, k.n2w, k.n2b, xn_, n, c.hidden, 1e-6f, s);
    // The MLP in row chunks: fc1 + tanh GELU into act_, fc2 added into the residual stream.
    for (int r0 = 0; r0 < n; r0 += kChunk) {
      const int m = std::min(kChunk, n - r0);
      linear(xn_ + size_t(r0) * c.hidden, m, c.hidden, k.fc1_w, k.fc1_b, c.inter, act_, c.inter_pad, 1, false, s);
      linear(act_, m, c.inter_pad, k.fc2_w, k.fc2_b, c.hidden, x_ + size_t(r0) * c.hidden, c.hidden, 0, true, s);
    }
    tap_copy(i);
  }
  // The merger: LayerNorm per patch, four consecutive patches as one 4,608-wide row, fc1 + erf GELU, fc2.
  vision_kernels::layernorm(x_, mn_w_, mn_b_, xn_, n, c.hidden, 1e-6f, s);
  const int rows = n / 4;
  for (int r0 = 0; r0 < rows; r0 += kChunk) {
    const int m = std::min(kChunk, rows - r0);
    linear(xn_ + size_t(r0) * c.merged(), m, c.merged(), m1_w_, m1_b_, c.merged(), act_, c.merged(), 2, false, s);
    linear(act_, m, c.merged(), m2_w_, m2_b_, c.out, out + size_t(r0) * c.out, c.out, 0, false, s);
  }
  if (tap_out && tap >= -1) check(cudaStreamSynchronize(s), "tap");
}

}  // namespace ling
