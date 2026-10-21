// Kernels of the DFlash2 drafter (z-lab's block-diffusion drafter with DFlash2's grouped convolutions and
// candidate selector), written against SGLang's reference implementation (models/dflash.py).
#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ling::kernels {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

constexpr int kDD = 128;  // drafter head dim
// draft_attention_body's U for the warp-exchanged batches of eight keys (BACKLOG 28).
constexpr int kExchange = 0;

// Per-head RMSNorm (plain weight) and NeoX RoPE over the whole head, in place. One block per (row, head),
// one thread per dim.
__device__ __forceinline__ void draft_qk_rope_body(float* __restrict__ x, int row_stride,
                                                   const __nv_bfloat16* __restrict__ w, int r, int pos, float theta,
                                                   float eps) {
  __shared__ float red[4], xs[kDD];
  const int h = blockIdx.y, d = threadIdx.x;
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
  sincosf(static_cast<float>(pos) * inv_freq, &sn, &cs);
  p[d] = d < half ? xs[d] * cs - xs[d + half] * sn : xs[d] * cs + xs[d - half] * sn;
}

__global__ void draft_qk_rope_kernel(float* __restrict__ x, int row_stride, const __nv_bfloat16* __restrict__ w,
                                     DevPos pos0, float theta, float eps) {
  pdl_begin();  // first statement: see launch.cuh
  const int r = blockIdx.x;
  draft_qk_rope_body(x, row_stride, w, r, pos_value(pos0) + r, theta, eps);
}

__global__ void draft_qk_rope_seqs_kernel(float* __restrict__ x, int row_stride, const __nv_bfloat16* __restrict__ w,
                                          SeqBatch b, float theta, float eps) {
  pdl_begin();  // first statement: see launch.cuh
  const int r = blockIdx.x, sq = seq_of(b, r);
  draft_qk_rope_body(x, row_stride, w, r, seq_pos(b, sq) + r - b.row0[sq], theta, eps);
}

__global__ void draft_store_kv_kernel(const float* __restrict__ k, const float* __restrict__ v, int kv_size,
                                      DevPos pos0_arg, __nv_bfloat16* __restrict__ kc, __nv_bfloat16* __restrict__ vc,
                                      int ring) {
  pdl_begin();  // first statement: see launch.cuh
  const int r = blockIdx.x, pos0 = pos_value(pos0_arg);
  for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
    const size_t dst = static_cast<size_t>((pos0 + r) % ring) * kv_size + i;
    kc[dst] = __float2bfloat16(k[static_cast<size_t>(r) * kv_size + i]);
    vc[dst] = __float2bfloat16(v[static_cast<size_t>(r) * kv_size + i]);
  }
}

// Non-causal attention of the block's B queries over the context cache (a sliding window ending before the
// block) and over the block itself. One block per (query row, KV head); 8 warps walk the keys with an online
// softmax and are merged at the end. Lane l owns dims 4l .. 4l + 3.
// Query row j of a block of B rows at positions L .. L + B - 1 (q, kb, vb and out start at the block's first row).
template <int G, int U>
__device__ __forceinline__ void draft_attention_body(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                                                     const __nv_bfloat16* __restrict__ vc, int ring,
                                                     const float* __restrict__ kb, const float* __restrict__ vb, int L,
                                                     int window_left, int Hkv, float* __restrict__ out, int j, int B) {
  __shared__ float wm[8][G], wl[8][G], wacc[8][kDD];
  const int kvh = blockIdx.y;
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
  int kk = warp;
  // BACKLOG 28's lead (1.46 ms a step for ~42 MB): one key at a time, a warp waited for each key's load, then for
  // its G dependent warp sums (5 shuffles each), and every lane computed the same two exponentials per key and head.
  // U > 1: the context keys go U at a time, the U loads first, all in flight together, then the G x U dot products,
  // then the online updates in key order. U = kExchange: eight keys, and the 32 (key, head) partial dot products are
  // reduce-scattered across the warp, so lane j ends with pair j's sum after 31 shuffles instead of 160, and lane j
  // alone computes pair j's two exponentials (the running maximum is a scan over the pairs of its head: a maximum is
  // exact in any order); the scores, factors and maxima go to every lane through shared memory. Either way a warp
  // takes the same keys in the same order, each sum adds the same operands in the same tree as warp_sum (the
  // butterfly's stages in its order: lane j's sum is warp_sum's) and each update is the same expression, so the
  // result is the old loop's bit for bit (U = 1). The tail of the context and the block's own keys go one at a time.
  if constexpr (U > 1 || U == kExchange) {
    constexpr bool XP = U == kExchange;
    constexpr int NU = XP ? 8 : U;
    static_assert(!XP || NU * G == 32, "the exchange takes 32 (key, head) pairs");
    __shared__ float xs[XP ? 8 : 1][XP ? 4 : 1][32];  // per warp: scores, factors, weights, running maxima
    for (; kk + 8 * (NU - 1) < nctx; kk += 8 * NU) {
      const int p0 = (lo + kk) % ring;
      uint2 kr[NU], vr[NU];
#pragma unroll
      for (int u = 0; u < NU; ++u) {
        int p = p0 + 8 * u;
        if (p >= ring) p -= ring;  // ring > 8 (NU - 1): the host checks
        const size_t off = (static_cast<size_t>(p) * Hkv + kvh) * kDD + lane * 4;
        kr[u] = *reinterpret_cast<const uint2*>(kc + off);
        vr[u] = *reinterpret_cast<const uint2*>(vc + off);
      }
      float sc[NU][G];
#pragma unroll
      for (int u = 0; u < NU; ++u) {
        const __nv_bfloat16* k16 = reinterpret_cast<const __nv_bfloat16*>(&kr[u]);
        float kf[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kf[i] = __bfloat162float(k16[i]);
#pragma unroll
        for (int g = 0; g < G; ++g) {
          float dot = qv[g][0] * kf[0];
          dot = fmaf(qv[g][1], kf[1], dot);
          dot = fmaf(qv[g][2], kf[2], dot);
          dot = fmaf(qv[g][3], kf[3], dot);
          if constexpr (XP) sc[u][g] = dot;  // the lane's partial, reduced below
          else sc[u][g] = warp_sum(dot);
        }
      }
      if constexpr (XP) {
        // Reduce-scatter: at offset o a lane keeps the half of its pairs whose index bit o matches its own and adds
        // the partner's value of each (v_i + v_(i^o), warp_sum's operands in its order).
        float* part = &sc[0][0];  // pair j = u * G + g
#pragma unroll
        for (int o = 16, n = 32; o > 0; o >>= 1, n >>= 1) {
          const bool hi = lane & o;
#pragma unroll
          for (int k = 0; k < n / 2; ++k) {
            const float keep = hi ? part[k + n / 2] : part[k], send = hi ? part[k] : part[k + n / 2];
            part[k] = keep + __shfl_xor_sync(0xffffffffu, send, o);
          }
        }
        const int g = lane % G;
        const float s = part[0];
        float m = s;  // the running maximum of head g over this batch's keys up to lane's: a scan over lanes g + 4u
#pragma unroll
        for (int o = G; o < 32; o <<= 1) {
          const float up = __shfl_up_sync(0xffffffffu, m, o);
          if (lane >= o) m = fmaxf(m, up);
        }
        float before = mx[0];
#pragma unroll
        for (int gg = 1; gg < G; ++gg) before = g == gg ? mx[gg] : before;
        const float nm = fmaxf(before, m);
        const float prev_in = __shfl_up_sync(0xffffffffu, nm, G), prev = lane >= G ? prev_in : before;
        xs[warp][0][lane] = __expf(prev - nm);  // corr
        xs[warp][1][lane] = __expf(s - nm);     // p
        xs[warp][2][lane] = nm;
        __syncwarp();
#pragma unroll
        for (int u = 0; u < NU; ++u) {
          const __nv_bfloat16* v16 = reinterpret_cast<const __nv_bfloat16*>(&vr[u]);
          float vf[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) vf[i] = __bfloat162float(v16[i]);
          const float4 c4 = *reinterpret_cast<const float4*>(&xs[warp][0][u * G]);
          const float4 p4 = *reinterpret_cast<const float4*>(&xs[warp][1][u * G]);
          const float cs[4] = {c4.x, c4.y, c4.z, c4.w}, ps[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
          for (int gg = 0; gg < G; ++gg) {
            const float corr = cs[gg], p = ps[gg];
            l[gg] = l[gg] * corr + p;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[gg][i] = acc[gg][i] * corr + p * vf[i];
          }
        }
        const float4 m4 = *reinterpret_cast<const float4*>(&xs[warp][2][(NU - 1) * G]);
        mx[0] = m4.x, mx[1] = m4.y, mx[2] = m4.z, mx[3] = m4.w;
        __syncwarp();  // before the next batch writes xs
      } else {
#pragma unroll
        for (int u = 0; u < NU; ++u) {
          const __nv_bfloat16* v16 = reinterpret_cast<const __nv_bfloat16*>(&vr[u]);
          float vf[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) vf[i] = __bfloat162float(v16[i]);
#pragma unroll
          for (int g = 0; g < G; ++g) {
            const float nm = fmaxf(mx[g], sc[u][g]);
            const float corr = __expf(mx[g] - nm), p = __expf(sc[u][g] - nm);
            l[g] = l[g] * corr + p;
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[g][i] = acc[g][i] * corr + p * vf[i];
            mx[g] = nm;
          }
        }
      }
    }
  }
  for (; kk < nk; kk += 8) {
    float kf[4], vf[4];
    if (kk < nctx) {
      const size_t off = (static_cast<size_t>((lo + kk) % ring) * Hkv + kvh) * kDD + lane * 4;
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

template <int G, int U>
__global__ void __launch_bounds__(256)
    draft_attention_kernel(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kc,
                           const __nv_bfloat16* __restrict__ vc, int ring, const float* __restrict__ kb,
                           const float* __restrict__ vb, DevPos L_arg, int window_left, int Hkv, float* __restrict__ out) {
  pdl_begin();  // first statement: see launch.cuh
  draft_attention_body<G, U>(q, kc, vc, ring, kb, vb, pos_value(L_arg), window_left, Hkv, out, blockIdx.x, gridDim.x);
}

// Several sequences' blocks in one launch (M4 stage 4): row jg belongs to sequence seq_of(jg), whose block is its
// rows of b, over its own ring at its own position.
template <int G, int U>
__global__ void __launch_bounds__(256)
    draft_attention_seqs_kernel(const float* __restrict__ q, SeqPtrs<const __nv_bfloat16> kc,
                                SeqPtrs<const __nv_bfloat16> vc, int ring, const float* __restrict__ kb,
                                const float* __restrict__ vb, SeqBatch b, int window_left, int Hkv,
                                float* __restrict__ out) {
  pdl_begin();  // first statement: see launch.cuh
  const int jg = blockIdx.x, sq = seq_of(b, jg);
  const size_t r0 = b.row0[sq], Hq = static_cast<size_t>(Hkv) * G;
  draft_attention_body<G, U>(q + r0 * Hq * kDD, kc.p[sq], vc.p[sq], ring, kb + r0 * Hkv * kDD, vb + r0 * Hkv * kDD,
                          seq_pos(b, sq), window_left, Hkv, out + r0 * Hq * kDD, jg - b.row0[sq],
                          b.row0[sq + 1] - b.row0[sq]);
}

// DFlash2's grouped dynamic convolution over the block (SGLang's _grouped_conv): per channel c of group
// c / group_size, out[n] = (base[side][0][c] + coef[n][side][0][grp]) * h[n]
//                       + (base[side][1][c] + coef[n][side][1][grp]) * h[n - 1]   (only if n % block >= 1).
__global__ void grouped_conv_kernel(const float* __restrict__ h, const float* __restrict__ coef,
                                    const __nv_bfloat16* __restrict__ base, int side, float* __restrict__ out, int C,
                                    int groups, int block) {
  pdl_begin();  // first statement: see launch.cuh
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
  pdl_begin();  // first statement: see launch.cuh
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
  pdl_begin();  // first statement: see launch.cuh
  const int r = blockIdx.x;
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    dst[static_cast<size_t>(r) * dst_stride + i] = __float2bfloat16(src[static_cast<size_t>(r) * H + i]);
}

// to_half_rows (stream_gemm.cu) from BF16 rows: one block per row, the power-of-two scale that puts the row's largest
// magnitude in [2^14, 2^15), the same arithmetic on the same values.
__global__ void bf16_to_half_rows_kernel(const __nv_bfloat16* __restrict__ x, int K, __half* __restrict__ out,
                                         float* __restrict__ xinv) {
  pdl_begin();  // first statement: see launch.cuh
  __shared__ float red[32];
  const __nv_bfloat16* row = x + static_cast<size_t>(blockIdx.x) * K;
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(__bfloat162float(row[k])));
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
    out[static_cast<size_t>(blockIdx.x) * K + k] = __float2half_rn(__bfloat162float(row[k]) * s);
  if (threadIdx.x == 0) xinv[blockIdx.x] = ldexpf(1.f, e - 15);
}

// The wide copy (BACKLOG 28's lead 1, as to_half_rows_wide_kernel in stream_gemm.cu): row blockIdx.x over gridDim.y
// blocks, each finding the whole row's largest magnitude (exact in any order) and converting its slice of K, so the
// values are the one-block kernel's bit for bit. VEC: x and out aligned to 16 bytes, K and slice multiples of 8.
__device__ __forceinline__ float abs_max8(float m, const uint4& v) {
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
  for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(__bfloat162float(b[i])));
  return m;
}

template <bool VEC>
__global__ void __launch_bounds__(256) bf16_to_half_rows_wide_kernel(const __nv_bfloat16* __restrict__ x, int K,
                                                                     int slice, __half* __restrict__ out,
                                                                     float* __restrict__ xinv) {
  pdl_begin();  // first statement: see launch.cuh
  __shared__ float red[8];
  const size_t r = blockIdx.x;
  const __nv_bfloat16* row = x + r * K;
  const int tid = threadIdx.x;
  float m0 = 0.f, m1 = 0.f;
  if constexpr (VEC) {
    const uint4* row8 = reinterpret_cast<const uint4*>(row);
    const int K8 = K / 8;
    int k = tid;
    for (; k + 3 * 256 < K8; k += 4 * 256) {
      uint4 v[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = row8[k + i * 256];
      m0 = abs_max8(m0, v[0]), m1 = abs_max8(m1, v[1]), m0 = abs_max8(m0, v[2]), m1 = abs_max8(m1, v[3]);
    }
    for (; k < K8; k += 256) m0 = abs_max8(m0, row8[k]);
  } else {
    for (int k = tid; k < K; k += 256) m0 = fmaxf(m0, fabsf(__bfloat162float(row[k])));
  }
  float mx = fmaxf(m0, m1);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((tid & 31) == 0) red[tid >> 5] = mx;
  __syncthreads();
  mx = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) mx = fmaxf(mx, red[i]);
  int e = 0;
  if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);  // mx < 2^e
  const float s = ldexpf(1.f, 15 - e);
  const int k0 = blockIdx.y * slice, k1 = min(K, k0 + slice);
  if constexpr (VEC) {
    const uint4* row8 = reinterpret_cast<const uint4*>(row);
    uint4* out8 = reinterpret_cast<uint4*>(out + r * K);
    for (int k = k0 / 8 + tid; k < k1 / 8; k += 256) {
      const uint4 v = row8[k];
      const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&v);
      uint4 o;
      __half* h = reinterpret_cast<__half*>(&o);
#pragma unroll
      for (int i = 0; i < 8; ++i) h[i] = __float2half_rn(__bfloat162float(b[i]) * s);
      out8[k] = o;
    }
  } else {
    for (int k = k0 + tid; k < k1; k += 256) out[r * K + k] = __float2half_rn(__bfloat162float(row[k]) * s);
  }
  if (blockIdx.y == 0 && tid == 0) xinv[r] = ldexpf(1.f, e - 15);
}

// draft_attention's batches (U above): LING_DRAFT_ATTN_KEYS or set_draft_attention_keys; 1, 2, 4 or 8 keys a load,
// 0 (the default) eight keys exchanged across the warp.
int draft_keys = [] {
  const char* e = std::getenv("LING_DRAFT_ATTN_KEYS");
  return e ? std::atoi(e) : 0;
}();

template <typename Launch>
void with_keys(int ring, Launch&& launch) {
  const int u = draft_keys == 0 ? (8 * 7 < ring ? 0 : 1) : draft_keys == 1 || 8 * (draft_keys - 1) >= ring ? 1 : draft_keys;
  if (u == 0) launch(std::integral_constant<int, kExchange>());
  else if (u == 2) launch(std::integral_constant<int, 2>());
  else if (u == 4) launch(std::integral_constant<int, 4>());
  else if (u == 8) launch(std::integral_constant<int, 8>());
  else launch(std::integral_constant<int, 1>());
}

}  // namespace

// About LING_HALF_BLOCKS (default 96) blocks in all, as to_half_rows; LING_HALF_BLOCKS=0: one block per row.
void bf16_to_half_rows(const __nv_bfloat16* x, int M, int K, __half* out, float* xinv, cudaStream_t s) {
  if (M <= 0) return;
  static const int target = [] {
    const char* e = std::getenv("LING_HALF_BLOCKS");
    return e ? std::atoi(e) : 96;
  }();
  if (target <= 0) {
    launch_kernel("bf16_to_half_rows", bf16_to_half_rows_kernel, dim3(M), dim3(256), 0, s, x, K, out, xinv);
    return;
  }
  const bool vec = K % 8 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 && reinterpret_cast<uintptr_t>(out) % 16 == 0;
  const int split = std::max(1, std::min((target + M - 1) / M, std::max(1, K / 1024)));
  int slice = (K + split - 1) / split;
  if (vec) slice = (slice + 7) / 8 * 8;
  const dim3 grid(M, (K + slice - 1) / slice);
  if (vec)
    launch_kernel("bf16_to_half_rows", bf16_to_half_rows_wide_kernel<true>, grid, dim3(256), 0, s, x, K, slice, out, xinv);
  else
    launch_kernel("bf16_to_half_rows", bf16_to_half_rows_wide_kernel<false>, grid, dim3(256), 0, s, x, K, slice, out, xinv);
}

void set_draft_attention_keys(int u) { draft_keys = u; }

void copy_rows_bf16(const float* src, int rows, int H, __nv_bfloat16* dst, int dst_stride, cudaStream_t s) {
  if (rows <= 0) return;
  launch_kernel("copy_rows_bf16", copy_rows_bf16_kernel, dim3(rows), dim3(256), 0, s, src, H, dst, dst_stride);
}

void draft_qk_rope(float* x, int rows, int heads, int row_stride, const __nv_bfloat16* norm, DevPos pos0, float theta,
                   float eps, cudaStream_t s) {
  launch_kernel("draft_qk_rope", draft_qk_rope_kernel, dim3(rows, heads), dim3(kDD), 0, s, x, row_stride, norm, pos0,
                theta, eps);
}

void draft_store_kv(const float* k, const float* v, int rows, int kv_size, DevPos pos0, __nv_bfloat16* kc,
                    __nv_bfloat16* vc, int ring, cudaStream_t s) {
  if (rows > ring) throw std::runtime_error("draft_store_kv: more rows than the ring holds");
  launch_kernel("draft_store_kv", draft_store_kv_kernel, dim3(rows), dim3(256), 0, s, k, v, kv_size, pos0, kc, vc,
                ring);
}

void draft_attention(const float* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, int ring, const float* kb,
                     const float* vb, DevPos L, int B, int window_left, int Hq, int Hkv, float* out, cudaStream_t s) {
  if (Hq != 4 * Hkv) throw std::runtime_error("draft_attention: built for 4 query heads per KV head");
  if (ring <= window_left) throw std::runtime_error("draft_attention: the ring is shorter than the window");
  with_keys(ring, [&](auto u) {
    launch_kernel("draft_attention", draft_attention_kernel<4, decltype(u)::value>, dim3(B, Hkv), dim3(256), 0, s, q, kc,
                  vc, ring, kb, vb, L, window_left, Hkv, out);
  });
}

void draft_qk_rope_seqs(float* x, int heads, int row_stride, const __nv_bfloat16* norm, const SeqBatch& b, float theta,
                        float eps, cudaStream_t s) {
  launch_kernel("draft_qk_rope_seqs", draft_qk_rope_seqs_kernel, dim3(b.row0[b.n], heads), dim3(kDD), 0, s, x,
                row_stride, norm, b, theta, eps);
}

void draft_attention_seqs(const float* q, SeqPtrs<const __nv_bfloat16> kc, SeqPtrs<const __nv_bfloat16> vc, int ring,
                          const float* kb, const float* vb, const SeqBatch& b, int window_left, int Hq, int Hkv,
                          float* out, cudaStream_t s) {
  if (Hq != 4 * Hkv) throw std::runtime_error("draft_attention: built for 4 query heads per KV head");
  if (ring <= window_left) throw std::runtime_error("draft_attention: the ring is shorter than the window");
  with_keys(ring, [&](auto u) {
    launch_kernel("draft_attention_seqs", draft_attention_seqs_kernel<4, decltype(u)::value>, dim3(b.row0[b.n], Hkv),
                  dim3(256), 0, s, q, kc, vc, ring, kb, vb, b, window_left, Hkv, out);
  });
}

void grouped_conv(const float* h, const float* coef, const __nv_bfloat16* base, int side, float* out, int rows,
                  int C, int groups, int block, cudaStream_t s) {
  launch_kernel("grouped_conv", grouped_conv_kernel, dim3(rows), dim3(256), 0, s, h, coef, base, side, out, C, groups,
                block);
}

void selector_lattice(const float* hp, const int* cand, const float* unary, const __nv_bfloat16* P,
                      const __nv_bfloat16* S, int anchor, int E, float* scores, cudaStream_t s, const int* anchor_dev) {
  launch_kernel("selector_lattice", selector_lattice_kernel, dim3(E), dim3(256), 0, s, hp, cand, unary, P, S, anchor,
                anchor_dev, scores);
}

}  // namespace ling::kernels
