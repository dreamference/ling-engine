// The Gated DeltaNet's causal conv and recurrence for the prefill path (any number of tokens).
//
// The rows path's kernels (kernels.cu) walk the tokens in order: the conv one thread per channel in place,
// the recurrence one warp per 32 value columns of a head. Right for 1-16 tokens; for 2,048 they take 0.8 ms
// and 3.7 ms per layer. Here:
//   - the conv is a 4-tap filter over raw inputs, so it runs in parallel over tokens and channels, out of
//     place, and the new window (the last three raw inputs) is written afterwards;
//   - the recurrence runs in its chunked (WY) form on tensor cores, 0.76 ms at 2,048 tokens. Two faster
//     sequential versions were tried first (four lanes per value column, inputs staged in shared memory,
//     the norms hoisted) and ran as slowly as the rows kernel, 3.6 ms: one SM per head walking 2,048 tokens
//     is bound by the per-token dependency chain, not by arithmetic.
// Each token's conv is computed the same way whatever M is; the recurrence is the same for any split that
// falls on a multiple of 32 tokens.
#include "core/kernels.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// out[t][c] = silu(w0 x[t-3] + w1 x[t-2] + w2 x[t-1] + w3 x[t]), x[-3..-1] from the state window.
__global__ void conv_prefill_kernel(const float* __restrict__ in, float* __restrict__ out, const float* __restrict__ state,
                                    const __nv_bfloat16* __restrict__ w, int M, int C) {
  const size_t n = static_cast<size_t>(M) * C;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int t = static_cast<int>(i / C), c = static_cast<int>(i % C);
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int tt = t - 3 + j;
      x[j] = tt >= 0 ? in[static_cast<size_t>(tt) * C + c] : state[c * 3 + 3 + tt];
    }
    const float o = __bfloat162float(w[c * 4]) * x[0] + __bfloat162float(w[c * 4 + 1]) * x[1] +
                    __bfloat162float(w[c * 4 + 2]) * x[2] + __bfloat162float(w[c * 4 + 3]) * x[3];
    out[i] = silu(o);
  }
}

// The window after the first M tokens: the three most recent raw inputs (into `out`, which may be `state`).
__global__ void conv_window_kernel(const float* __restrict__ in, const float* state, float* out, int M, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int tt = M - 3 + j;
    s[j] = tt >= 0 ? in[static_cast<size_t>(tt) * C + c] : state[c * 3 + 3 + tt];
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) out[c * 3 + j] = s[j];
}

// ---- The chunked (WY) form of the gated delta rule on tensor cores ----
//
// Per chunk of T = 32 tokens with start state S0 (Dk x Dv), cumulative log-decay G_i, gamma_i = exp(G_i):
//   A_ij   = beta_i exp(G_i - G_j) k_i . k_j  (j < i)          T^-1 = (I + A)^-1, by forward substitution
//   U      = T^-1 (beta v),  W = T^-1 (beta gamma k)            Delta = U - W S0
//   O      = diag(gamma) Q S0 + [exp(G_i - G_j) q_i . k_j]_{j <= i} Delta
//   S_next = gamma_T S0 + K'^T Delta,  k'_i = (gamma_T / gamma_i) k_i
// which is the token recurrence S_i = a_i S_{i-1} + k_i beta_i (v_i - a_i S_{i-1}^T k_i)^T, o_i = S_i^T q_i
// unrolled over the chunk. The matrix products run as FP16 m16n8k16 MMAs with FP32 accumulation (q, k
// normalized; the state kept in FP32 and multiplied in FP16), the substitution in FP32. Chunks sit at
// absolute positions (multiples of 32), so a prompt's result does not depend on where a prefill call
// started as long as it starts at a multiple of 32 (the engine resumes only there).
constexpr int kDk = 128, kDv = 128, kT = 32, kChunkThreads = 512;
constexpr int kPad = 136;   // FP16 row stride of [*][128] tiles: rows 272 bytes apart, conflict-free ldmatrix
constexpr int kTPad = 40;   // FP16 row stride of [32][32] tiles

struct ChunkSmem {
  __half q[kT * kPad];       // q / |q| / sqrt(Dk)
  __half kw[kT * kPad];      // k / |k|; after the products, W
  __half bv[kT * kPad];      // beta v; after U and W, Delta
  __half bgk[kT * kPad];     // beta gamma k
  __half kp[kT * kPad];      // (gamma_T / gamma) k
  __half s16[kDk * kPad];    // the chunk's start state in FP16 [dk][dv]
  float A[kT][kT + 1];
  float P[kT][kT + 1];
  __half tinv[kT * kTPad];
  __half p16[kT * kTPad];
  float G[kT], gam[kT], beta[kT];
};

__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr(p)));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr(p)));
}
__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(saddr(p)));
}
__device__ __forceinline__ void mma_h(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// A fragment (16 x 16) of a row-major [m][k] tile at (m0, k0), row stride ld halves.
__device__ __forceinline__ void frag_a(uint32_t (&a)[4], const __half* base, int ld, int m0, int k0, int lane) {
  ldsm4(a, base + (m0 + (lane & 7) + 8 * ((lane >> 3) & 1)) * ld + k0 + 8 * (lane >> 4));
}
// A fragment of the transpose of a row-major [k][m] tile (A = X^T).
__device__ __forceinline__ void frag_at(uint32_t (&a)[4], const __half* base, int ld, int m0, int k0, int lane) {
  ldsm4t(a, base + (k0 + (lane & 7) + 8 * (lane >> 4)) * ld + m0 + 8 * ((lane >> 3) & 1));
}
// B fragments (16 x 8) for n-frags n0 and n0 + 8 of a row-major [k][n] tile: b[0], b[1] for n0; b[2], b[3] for n0 + 8.
__device__ __forceinline__ void frag_b_kn(uint32_t (&b)[4], const __half* base, int ld, int k0, int n0, int lane) {
  ldsm4t(b, base + (k0 + (lane & 7) + 8 * ((lane >> 3) & 1)) * ld + n0 + 8 * (lane >> 4));
}
// B fragment (16 x 8) of a row-major [n][k] tile (B = X^T, the k . k products).
__device__ __forceinline__ void frag_b_nk(uint32_t (&b)[2], const __half* base, int ld, int n0, int k0, int lane) {
  ldsm2(b, base + (n0 + (lane & 7)) * ld + k0 + 8 * ((lane >> 3) & 1));
}

__global__ void __launch_bounds__(kChunkThreads)
    chunk_gdn_kernel(const float* __restrict__ mixed, const float* __restrict__ g, const float* __restrict__ beta,
                     float* __restrict__ state, float* __restrict__ out, int M, int H, int HV, int phase,
                     float* __restrict__ state_at, int at) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  ChunkSmem& sm = *reinterpret_cast<ChunkSmem*>(smem_raw);
  const int hv = blockIdx.x, h = hv / (HV / H), tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int gq = lane >> 2, tq = lane & 3;
  const int C = 2 * H * kDk + HV * kDv;
  float* st = state + static_cast<size_t>(hv) * kDk * kDv;
  // This warp's tiles of the state: dk rows [16 (w % 8), +16), dv columns [64 (w / 8), +64), 8 n-fragments.
  const int sm0 = 16 * (warp & 7), sn0 = 64 * (warp >> 3);
  float S[8][4];
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int r = sm0 + gq + 8 * (e >> 1), c = sn0 + 8 * j + 2 * tq + (e & 1);
      S[j][e] = st[r * kDv + c];
      sm.s16[r * kPad + c] = __float2half_rn(S[j][e]);
    }
  const float scale = rsqrtf(static_cast<float>(kDk));
  const int nchunks = (phase + M + kT - 1) / kT;
  for (int ch = 0; ch < nchunks; ++ch) {
    const int t0 = max(0, ch * kT - phase), t1 = min(M, (ch + 1) * kT - phase), n = t1 - t0;
    // 1. Load: warp w takes tokens 2w and 2w + 1 of the chunk (rows past n are zero, with no decay).
    float4 kreg[2], vreg[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = 2 * warp + u;
      float4 q4 = make_float4(0.f, 0.f, 0.f, 0.f), k4 = q4, v4 = q4;
      float gv = 0.f, bv = 0.f;
      if (i < n) {
        const float* row = mixed + static_cast<size_t>(t0 + i) * C;
        q4 = reinterpret_cast<const float4*>(row + h * kDk)[lane];
        k4 = reinterpret_cast<const float4*>(row + H * kDk + h * kDk)[lane];
        v4 = reinterpret_cast<const float4*>(row + 2 * H * kDk + hv * kDv)[lane];
        gv = g[static_cast<size_t>(t0 + i) * HV + hv];
        bv = beta[static_cast<size_t>(t0 + i) * HV + hv];
      }
      float qn = q4.x * q4.x + q4.y * q4.y + q4.z * q4.z + q4.w * q4.w;
      float kn = k4.x * k4.x + k4.y * k4.y + k4.z * k4.z + k4.w * k4.w;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        qn += __shfl_xor_sync(0xffffffffu, qn, o);
        kn += __shfl_xor_sync(0xffffffffu, kn, o);
      }
      const float qr = rsqrtf(qn + 1e-6f) * scale, kr = rsqrtf(kn + 1e-6f);
      k4 = make_float4(k4.x * kr, k4.y * kr, k4.z * kr, k4.w * kr);
      __half2* qd = reinterpret_cast<__half2*>(sm.q + i * kPad + 4 * lane);
      __half2* kd = reinterpret_cast<__half2*>(sm.kw + i * kPad + 4 * lane);
      qd[0] = __floats2half2_rn(q4.x * qr, q4.y * qr);
      qd[1] = __floats2half2_rn(q4.z * qr, q4.w * qr);
      kd[0] = __floats2half2_rn(k4.x, k4.y);
      kd[1] = __floats2half2_rn(k4.z, k4.w);
      kreg[u] = k4;
      vreg[u] = v4;
      if (lane == 0) {
        sm.G[i] = gv;
        sm.beta[i] = bv;
      }
    }
    __syncthreads();
    if (warp == 0) {  // cumulative log-decay
      float x = sm.G[lane];
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const float y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
      }
      sm.G[lane] = x;
      sm.gam[lane] = __expf(x);
    }
    __syncthreads();
    const float gamT = sm.gam[kT - 1];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int i = 2 * warp + u;
      // exp(G_T - G_i) in log space: a fast-forgetting head's gamma underflows to 0 within a chunk.
      const float b = sm.beta[i], gm = sm.gam[i], kpf = __expf(sm.G[kT - 1] - sm.G[i]), bg = b * gm;
      const float4 k4 = kreg[u], v4 = vreg[u];
      __half2* d0 = reinterpret_cast<__half2*>(sm.bv + i * kPad + 4 * lane);
      __half2* d1 = reinterpret_cast<__half2*>(sm.bgk + i * kPad + 4 * lane);
      __half2* d2 = reinterpret_cast<__half2*>(sm.kp + i * kPad + 4 * lane);
      d0[0] = __floats2half2_rn(b * v4.x, b * v4.y);
      d0[1] = __floats2half2_rn(b * v4.z, b * v4.w);
      d1[0] = __floats2half2_rn(bg * k4.x, bg * k4.y);
      d1[1] = __floats2half2_rn(bg * k4.z, bg * k4.w);
      d2[0] = __floats2half2_rn(kpf * k4.x, kpf * k4.y);
      d2[1] = __floats2half2_rn(kpf * k4.z, kpf * k4.w);
    }
    // 2. K K^T and Q K^T (32 x 32 each): warp w takes matrix w / 8, m-fragment (w / 4) % 2, n-fragment w % 4.
    {
      const int which = warp >> 3, mf = (warp >> 2) & 1, nf = warp & 3;
      const __half* xa = which ? sm.q : sm.kw;
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k0 = 0; k0 < kDk; k0 += 16) {
        uint32_t a[4], b[2];
        frag_a(a, xa, kPad, 16 * mf, k0, lane);
        frag_b_nk(b, sm.kw, kPad, 8 * nf, k0, lane);
        mma_h(c, a, b[0], b[1]);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int i = 16 * mf + gq + 8 * (e >> 1), j = 8 * nf + 2 * tq + (e & 1);
        const float dec = __expf(sm.G[i] - sm.G[j]);
        if (which == 0) sm.A[i][j] = j < i ? sm.beta[i] * dec * c[e] : 0.f;
        else {
          const float pv = j <= i ? dec * c[e] : 0.f;
          sm.P[i][j] = pv;
          sm.p16[i * kTPad + j] = __float2half_rn(pv);
        }
      }
    }
    __syncthreads();
    // 3. T^-1 = (I + A)^-1 by forward substitution: lane c of warp 0 computes column c.
    if (warp == 0) {
      float x[kT];
#pragma unroll
      for (int i = 0; i < kT; ++i) {
        float v = i == lane ? 1.f : 0.f;
#pragma unroll
        for (int j = 0; j < i; ++j) v = fmaf(-sm.A[i][j], x[j], v);
        x[i] = v;
        sm.tinv[i * kTPad + lane] = __float2half_rn(v);
      }
    }
    __syncthreads();
    // 4. U = T^-1 (beta v) and W = T^-1 (beta gamma k): warp w takes m-fragment w % 2 and n-fragments
    // 2 (w / 2), 2 (w / 2) + 1 of both (the same tiles as Delta and O below).
    const int mf = warp & 1, nf0 = 16 * (warp >> 1);
    float U[2][4] = {}, Wt[2][4] = {};
#pragma unroll
    for (int k0 = 0; k0 < kT; k0 += 16) {
      uint32_t a[4], bu[4], bw[4];
      frag_a(a, sm.tinv, kTPad, 16 * mf, k0, lane);
      frag_b_kn(bu, sm.bv, kPad, k0, nf0, lane);
      frag_b_kn(bw, sm.bgk, kPad, k0, nf0, lane);
      mma_h(U[0], a, bu[0], bu[1]);
      mma_h(U[1], a, bu[2], bu[3]);
      mma_h(Wt[0], a, bw[0], bw[1]);
      mma_h(Wt[1], a, bw[2], bw[3]);
    }
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int i = 16 * mf + gq + 8 * hh, c = nf0 + 8 * f + 2 * tq;
        *reinterpret_cast<__half2*>(sm.kw + i * kPad + c) = __floats2half2_rn(Wt[f][2 * hh], Wt[f][2 * hh + 1]);
      }
    __syncthreads();
    // 5. Delta = U - W S0 and the inter-chunk output Q S0 (scaled by gamma below).
    float D[2][4], O[2][4] = {};
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int e = 0; e < 4; ++e) D[f][e] = 0.f;
#pragma unroll
    for (int k0 = 0; k0 < kDk; k0 += 16) {
      uint32_t aw[4], aq[4], b[4];
      frag_a(aw, sm.kw, kPad, 16 * mf, k0, lane);
      frag_a(aq, sm.q, kPad, 16 * mf, k0, lane);
      frag_b_kn(b, sm.s16, kPad, k0, nf0, lane);
      mma_h(D[0], aw, b[0], b[1]);
      mma_h(D[1], aw, b[2], b[3]);
      mma_h(O[0], aq, b[0], b[1]);
      mma_h(O[1], aq, b[2], b[3]);
    }
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int i = 16 * mf + gq + 8 * hh, c = nf0 + 8 * f + 2 * tq;
        const float d0 = U[f][2 * hh] - D[f][2 * hh], d1 = U[f][2 * hh + 1] - D[f][2 * hh + 1];
        *reinterpret_cast<__half2*>(sm.bv + i * kPad + c) = __floats2half2_rn(d0, d1);
        O[f][2 * hh] *= sm.gam[i];
        O[f][2 * hh + 1] *= sm.gam[i];
      }
    __syncthreads();
    // 6. O += P Delta, written out; S = gamma_T S + K'^T Delta (this warp's state tiles), and its FP16 copy.
#pragma unroll
    for (int k0 = 0; k0 < kT; k0 += 16) {
      uint32_t a[4], b[4];
      frag_a(a, sm.p16, kTPad, 16 * mf, k0, lane);
      frag_b_kn(b, sm.bv, kPad, k0, nf0, lane);
      mma_h(O[0], a, b[0], b[1]);
      mma_h(O[1], a, b[2], b[3]);
    }
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int i = 16 * mf + gq + 8 * hh, c = nf0 + 8 * f + 2 * tq;
        if (i < n)
          *reinterpret_cast<float2*>(out + static_cast<size_t>(t0 + i) * HV * kDv + hv * kDv + c) =
              make_float2(O[f][2 * hh], O[f][2 * hh + 1]);
      }
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) S[j][e] *= gamT;
#pragma unroll
    for (int k0 = 0; k0 < kT; k0 += 16) {
      uint32_t a[4];
      frag_at(a, sm.kp, kPad, sm0, k0, lane);
#pragma unroll
      for (int jp = 0; jp < 4; ++jp) {
        uint32_t b[4];
        frag_b_kn(b, sm.bv, kPad, k0, sn0 + 16 * jp, lane);
        mma_h(S[2 * jp], a, b[0], b[1]);
        mma_h(S[2 * jp + 1], a, b[2], b[3]);
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int r = sm0 + gq + 8 * hh, c = sn0 + 8 * j + 2 * tq;
        *reinterpret_cast<__half2*>(sm.s16 + r * kPad + c) = __floats2half2_rn(S[j][2 * hh], S[j][2 * hh + 1]);
      }
    if (t1 == at) {  // the state after token `at` (a chunk end), kept for a later prompt
      float* sa = state_at + static_cast<size_t>(hv) * kDk * kDv;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
          sa[(sm0 + gq + 8 * (e >> 1)) * kDv + sn0 + 8 * j + 2 * tq + (e & 1)] = S[j][e];
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int r = sm0 + gq + 8 * (e >> 1), c = sn0 + 8 * j + 2 * tq + (e & 1);
      st[r * kDv + c] = S[j][e];
    }
}

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void gdn_conv_prefill(const float* in, float* out, float* conv_state, const __nv_bfloat16* w, int M, int C,
                      cudaStream_t s, float* window_at, int at) {
  const size_t n = static_cast<size_t>(M) * C;
  conv_prefill_kernel<<<static_cast<int>(std::min<size_t>((n + 255) / 256, 48 * 64)), 256, 0, s>>>(in, out, conv_state,
                                                                                                     w, M, C);
  check_launch("gdn_conv_prefill");
  if (window_at && at > 0 && at <= M) {
    conv_window_kernel<<<(C + 255) / 256, 256, 0, s>>>(in, conv_state, window_at, at, C);
    check_launch("gdn_conv_window_at");
  }
  conv_window_kernel<<<(C + 255) / 256, 256, 0, s>>>(in, conv_state, conv_state, M, C);
  check_launch("gdn_conv_window");
}

void gdn_recurrent_prefill(const float* mixed, const float* g, const float* beta, float* state, float* out, int M, int H,
                           int HV, int pos0, cudaStream_t s, float* state_at, int at) {
  if (M < 1) return;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(chunk_gdn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(ChunkSmem)));
    configured = true;
  }
  if ((pos0 + at) % kT != 0 && state_at) throw std::runtime_error("gdn_recurrent_prefill: state_at must be at a chunk end");
  chunk_gdn_kernel<<<HV, kChunkThreads, sizeof(ChunkSmem), s>>>(mixed, g, beta, state, out, M, H, HV, pos0 % kT,
                                                                 state_at, state_at ? at : -1);
  check_launch("gdn_recurrent_prefill");
}

}  // namespace ling::kernels
