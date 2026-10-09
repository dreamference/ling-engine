// Kernels of the DFlash2 drafter (z-lab's block-diffusion drafter with DFlash2's grouped convolutions and
// candidate selector), written against SGLang's reference implementation (models/dflash.py).
#include "core/kernels.cuh"

#include <cmath>
#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

constexpr int kDD = 128;  // drafter head dim

// Per-head RMSNorm (plain weight) and NeoX RoPE over the whole head, in place. One block per (row, head),
// one thread per dim.
__global__ void draft_qk_rope_kernel(float* __restrict__ x, int row_stride, const __nv_bfloat16* __restrict__ w,
                                     DevPos pos0, float theta, float eps) {
  __shared__ float red[4], xs[kDD];
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  float* p = x + static_cast<size_t>(r) * row_stride + h * kDD;
  const float v = p[d];
  float ss = warp_sum(v * v);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  ss = red[0] + red[1] + red[2] + red[3];
  xs[d] = v * rsqrtf(ss / kDD + eps) * __bfloat162float(w[d]);
  __syncthreads();
  const int half = kDD / 2, i = d % half;
  const float inv_freq = powf(theta, -2.f * i / kDD);
  float sn, cs;
  sincosf(static_cast<float>(pos_value(pos0) + r) * inv_freq, &sn, &cs);
  p[d] = d < half ? xs[d] * cs - xs[d + half] * sn : xs[d] * cs + xs[d - half] * sn;
}

__global__ void draft_store_kv_kernel(const float* __restrict__ k, const float* __restrict__ v, int kv_size,
                                      DevPos pos0_arg, __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc) {
  const int r = blockIdx.x, pos0 = pos_value(pos0_arg);
  for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
    const size_t dst = static_cast<size_t>(pos0 + r) * kv_size + i;
    kc[dst] = __float2bfloat16(k[static_cast<size_t>(r) * kv_size + i]);
    vc[dst] = __float2bfloat16(v[static_cast<size_t>(r) * kv_size + i]);
  }
}

// Non-causal attention of the block's B queries over the context cache (a sliding window ending before the
// block) and over the block itself. One block per (query row, KV head); 8 warps walk the keys with an online
// softmax and are merged at the end. Lane l owns dims 4l .. 4l + 3.
template <int G>
__global__ void __launch_bounds__(256)
    draft_attention_kernel(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                           const __nv_bfloat16* __restrict__ vc, const float* __restrict__ kb,
                           const float* __restrict__ vb, DevPos L_arg, int window_left, int Hkv, float* __restrict__ out) {
  __shared__ float wm[8][G], wl[8][G], wacc[8][kDD];
  const int j = blockIdx.x, kvh = blockIdx.y, B = gridDim.x, L = pos_value(L_arg);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int Hq = Hkv * G;
  const int lo = max(0, L + j - window_left);
  const int nctx = L - lo, nk = nctx + B;
  const float scale = rsqrtf(static_cast<float>(kDD));
  float qv[G][4], acc[G][4], mx[G], l[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    const float4 v4 = *reinterpret_cast<const float4*>(q + (static_cast<size_t>(j) * Hq + kvh * G + g) * kDD + lane * 4);
    qv[g][0] = v4.x * scale;
    qv[g][1] = v4.y * scale;
    qv[g][2] = v4.z * scale;
    qv[g][3] = v4.w * scale;
    acc[g][0] = acc[g][1] = acc[g][2] = acc[g][3] = 0.f;
    mx[g] = -INFINITY;
    l[g] = 0.f;
  }
  for (int kk = warp; kk < nk; kk += 8) {
    float kf[4], vf[4];
    if (kk < nctx) {
      const size_t off = (static_cast<size_t>(lo + kk) * Hkv + kvh) * kDD + lane * 4;
      const uint2 kr = *reinterpret_cast<const uint2*>(kc + off);
      const uint2 vr = *reinterpret_cast<const uint2*>(vc + off);
      const __nv_bfloat16* k16 = reinterpret_cast<const __nv_bfloat16*>(&kr);
      const __nv_bfloat16* v16 = reinterpret_cast<const __nv_bfloat16*>(&vr);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        kf[i] = __bfloat162float(k16[i]);
        vf[i] = __bfloat162float(v16[i]);
      }
    } else {
      const size_t off = (static_cast<size_t>(kk - nctx) * Hkv + kvh) * kDD + lane * 4;
      const float4 k4 = *reinterpret_cast<const float4*>(kb + off);
      const float4 v4 = *reinterpret_cast<const float4*>(vb + off);
      kf[0] = k4.x, kf[1] = k4.y, kf[2] = k4.z, kf[3] = k4.w;
      vf[0] = v4.x, vf[1] = v4.y, vf[2] = v4.z, vf[3] = v4.w;
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float dot = qv[g][0] * kf[0];
      dot = fmaf(qv[g][1], kf[1], dot);
      dot = fmaf(qv[g][2], kf[2], dot);
      dot = fmaf(qv[g][3], kf[3], dot);
      const float sc = warp_sum(dot);
      const float nm = fmaxf(mx[g], sc);
      const float corr = __expf(mx[g] - nm), p = __expf(sc - nm);
      l[g] = l[g] * corr + p;
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[g][i] = acc[g][i] * corr + p * vf[i];
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
#pragma unroll
  for (int g = 0; g < G; ++g) {
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) wacc[warp][lane * 4 + i] = acc[g][i];
    __syncthreads();
    if (threadIdx.x < kDD) {
      const int d = threadIdx.x;
      float M = -INFINITY;
      for (int w = 0; w < 8; ++w) M = fmaxf(M, wm[w][g]);
      float Ls = 0.f, A = 0.f;
      for (int w = 0; w < 8; ++w) {
        if (wm[w][g] == -INFINITY) continue;
        const float c = __expf(wm[w][g] - M);
        Ls += wl[w][g] * c;
        A += wacc[w][d] * c;
      }
      out[(static_cast<size_t>(j) * Hq + kvh * G + g) * kDD + d] = A / Ls;
    }
  }
}

// DFlash2's grouped dynamic convolution over the block (SGLang's _grouped_conv): per channel c of group
// c / group_size, out[n] = (base[side][0][c] + coef[n][side][0][grp]) * h[n]
//                       + (base[side][1][c] + coef[n][side][1][grp]) * h[n - 1]   (only if n % block >= 1).
__global__ void grouped_conv_kernel(const float* __restrict__ h, const float* __restrict__ coef,
                                    const __nv_bfloat16* __restrict__ base, int side, float* __restrict__ out, int C,
                                    int groups, int block) {
  const int n = blockIdx.x, gs = C / groups;
  const float* cf = coef + static_cast<size_t>(n) * 4 * groups + side * 2 * groups;
  const bool prev = n % block >= 1;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const int grp = c / gs;
    const float c0 = __bfloat162float(base[(side * 2) * C + c]) + cf[grp];
    float v = c0 * h[static_cast<size_t>(n) * C + c];
    if (prev) {
      const float c1 = __bfloat162float(base[(side * 2 + 1) * C + c]) + cf[groups + grp];
      v = fmaf(c1, h[static_cast<size_t>(n - 1) * C + c], v);
    }
    out[static_cast<size_t>(n) * C + c] = v;
  }
}

// The selector's lattice: score[e][p][c] = unary[e][c] + sum_r P[pred(e, p)][r] * hp[e][r] * S[cand[e][c]][r],
// pred(0, p) = the anchor and pred(e, p) = cand[e - 1][p]. One block per position e, one thread per (p, c).
constexpr int kSelK = 16, kSelR = 256;
__global__ void __launch_bounds__(256)
    selector_lattice_kernel(const float* __restrict__ hp, const int* __restrict__ cand,
                            const float* __restrict__ unary, const __nv_bfloat16* __restrict__ P,
                            const __nv_bfloat16* __restrict__ S, int anchor_arg, const int* __restrict__ anchor_dev,
                            float* __restrict__ scores) {
  __shared__ float Ps[kSelK][kSelR + 1], Ss[kSelK][kSelR + 1], hs[kSelR];
  const int e = blockIdx.x, t = threadIdx.x;
  const int anchor = anchor_dev ? *anchor_dev : anchor_arg;
  hs[t] = hp[static_cast<size_t>(e) * kSelR + t];
  for (int i = 0; i < kSelK; ++i) {
    const int pid = e == 0 ? anchor : cand[(e - 1) * kSelK + i];
    const int cid = cand[e * kSelK + i];
    Ps[i][t] = __bfloat162float(P[static_cast<size_t>(pid) * kSelR + t]);
    Ss[i][t] = __bfloat162float(S[static_cast<size_t>(cid) * kSelR + t]);
  }
  __syncthreads();
  const int p = t / kSelK, c = t % kSelK;
  float sum = 0.f;
  for (int r = 0; r < kSelR; ++r) sum = fmaf(Ps[p][r] * hs[r], Ss[c][r], sum);
  scores[(static_cast<size_t>(e) * kSelK + p) * kSelK + c] = unary[e * kSelK + c] + sum;
}

__global__ void copy_rows_bf16_kernel(const float* __restrict__ src, int H, __nv_bfloat16* __restrict__ dst,
                                      int dst_stride) {
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    dst[static_cast<size_t>(r) * dst_stride + i] = __float2bfloat16(src[static_cast<size_t>(r) * H + i]);
}

}  // namespace

void copy_rows_bf16(const float* src, int rows, int H, __nv_bfloat16* dst, int dst_stride, cudaStream_t s) {
  if (rows <= 0) return;
  copy_rows_bf16_kernel<<<rows, 256, 0, s>>>(src, H, dst, dst_stride);
  check_launch("copy_rows_bf16");
}

void draft_qk_rope(float* x, int rows, int heads, int row_stride, const __nv_bfloat16* norm, DevPos pos0, float theta,
                   float eps, cudaStream_t s) {
  draft_qk_rope_kernel<<<dim3(rows, heads), kDD, 0, s>>>(x, row_stride, norm, pos0, theta, eps);
  check_launch("draft_qk_rope");
}

void draft_store_kv(const float* k, const float* v, int rows, int kv_size, DevPos pos0, __nv_bfloat16* kc,
                    __nv_bfloat16* vc, cudaStream_t s) {
  draft_store_kv_kernel<<<rows, 256, 0, s>>>(k, v, kv_size, pos0, kc, vc);
  check_launch("draft_store_kv");
}

void draft_attention(const float* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const float* kb,
                     const float* vb, DevPos L, int B, int window_left, int Hq, int Hkv, float* out, cudaStream_t s) {
  if (Hq != 4 * Hkv) throw std::runtime_error("draft_attention: built for 4 query heads per KV head");
  draft_attention_kernel<4><<<dim3(B, Hkv), 256, 0, s>>>(q, kc, vc, kb, vb, L, window_left, Hkv, out);
  check_launch("draft_attention");
}

void grouped_conv(const float* h, const float* coef, const __nv_bfloat16* base, int side, float* out, int rows,
                  int C, int groups, int block, cudaStream_t s) {
  grouped_conv_kernel<<<rows, 256, 0, s>>>(h, coef, base, side, out, C, groups, block);
  check_launch("grouped_conv");
}

void selector_lattice(const float* hp, const int* cand, const float* unary, const __nv_bfloat16* P,
                      const __nv_bfloat16* S, int anchor, int E, float* scores, cudaStream_t s, const int* anchor_dev) {
  selector_lattice_kernel<<<E, 256, 0, s>>>(hp, cand, unary, P, S, anchor, anchor_dev, scores);
  check_launch("selector_lattice");
}

}  // namespace ling::kernels
