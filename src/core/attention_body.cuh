// The rows path's attention (decode and verify) and its prepare step as device bodies of their block coordinates,
// shared by their kernels (kernels.cu) and the megakernel's attention layer (stream_gemm.cu, LING_MK_ATTN, BACKLOG 28):
// one source, so every launcher computes the same bits. Also warp_sum and block_sum, and the tree verify's DeltaNet
// recurrence (gdn_recurrent_tree_body, for the megakernel's DeltaNet mixer, LING_MK_GDN). CUDA source files only.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "core/kernels.cuh"

namespace ling::kernels {
namespace {

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

// The interleaved M-RoPE sections of Qwen3.8 ([11, 11, 10], mrope_interleaved; VISION §20.1): frequency i of the 32
// takes the height position when i % 3 == 1 and i < 3 * 11, the width position when i % 3 == 2 and i < 3 * 10, the
// temporal one otherwise. A text token's three positions are equal, so it is plain RoPE at that position.
constexpr int kMropeH = 11, kMropeW = 10;
__device__ __forceinline__ int mrope_pos(int i, int pt, int ph, int pw) {
  return (i % 3 == 1 && i < 3 * kMropeH) ? ph : (i % 3 == 2 && i < 3 * kMropeW) ? pw : pt;
}

// Row m (its RoPE positions pos, ph, pw: (temporal, height, width), all equal for text) of attn_prepare; the caches
// are its sequence's.
__device__ __forceinline__ void attn_prepare_body(const float* __restrict__ q_gate, const float* __restrict__ k,
                                                  const float* __restrict__ v, const __nv_bfloat16* __restrict__ q_norm,
                                                  const __nv_bfloat16* __restrict__ k_norm, int m, int pos, int ph,
                                                  int pw, int Hq,
                                                  int Hkv, int D, int rot, float theta, float eps, float* __restrict__ q,
                                                  float* __restrict__ gate, __nv_bfloat16* __restrict__ kcache,
                                                  __nv_bfloat16* __restrict__ vcache, size_t hs, size_t ps,
                                                  int cpos,  // cpos: the cache slot (pos except in a tree)
                                                  int slot) {  // q heads, then k heads, then v heads (the launch's blockIdx.y)
  extern __shared__ float xn[];
  __shared__ float red[32];
  const int d = threadIdx.x;
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
    sincosf(static_cast<float>(mrope_pos(i, pos, ph, pw)) * inv_freq, &sn, &cs);
    o = d < half ? xn[d] * cs - xn[d + half] * sn : xn[d] * cs + xn[d - half] * sn;
  }
  if (is_q) {
    q[(static_cast<size_t>(m) * Hq + head) * D + d] = o;
  } else {
    kcache[head * hs + cpos * ps + d] = __float2bfloat16(o);
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
// engine (attn_bulk_tile) instead of cp.async. The arithmetic is the same either way, bit for bit. (bx, kvh, split):
// the block's row block, KV head and key range (the launch's blockIdx). CQ: the queries are read through the coherent
// path (ld.global.cg), for a launch that writes them itself (the megakernel's attention layer); the same values.
template <int ROWS, int TK, int NS, bool BULK, bool CQ = false>
__device__ __forceinline__ void attention_mma_body(const float* __restrict__ q, const __nv_bfloat16* __restrict__ kcache,
                                                   const __nv_bfloat16* __restrict__ vcache, int pos0, int M, int Hq,
                                                   int Hkv, int splits, int chunk, float* __restrict__ scratch, size_t hs,
                                                   size_t ps, int pf, int bx, int kvh, int split,
                                                   const TreeDesc* __restrict__ td = nullptr) {
  constexpr int D = kMmaD, G = 6, KS = kMmaKS;
  extern __shared__ __align__(16) unsigned char smem[];
  auto stage = [&](int i) { return reinterpret_cast<__nv_bfloat16*>(smem + (i % NS) * mma_stage(TK)); };
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
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
    auto ldq = [](const float* p) {
      if constexpr (CQ) return __ldcg(reinterpret_cast<const float2*>(p));
      else return *reinterpret_cast<const float2*>(p);
    };
    const float2 a0 = va ? ldq(qa + k) : make_float2(0.f, 0.f);
    const float2 a1 = vb ? ldq(qb + k) : make_float2(0.f, 0.f);
    const float2 a2 = va ? ldq(qa + k + 8) : make_float2(0.f, 0.f);
    const float2 a3 = vb ? ldq(qb + k + 8) : make_float2(0.f, 0.f);
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

// attention_combine_kernel's output (row-head mh, dimension d): the partials of `splits` key ranges added in range
// order, skipping the empty ones (max -inf). CG: the partials are read through the coherent path (the megakernel's
// attention layer writes them in the same launch); the same values.
template <bool CG>
__device__ __forceinline__ float attention_combine_body(const float* __restrict__ scratch, int splits, int D, int mh,
                                                        int d) {
  auto ld = [](const float* p) {
    if constexpr (CG) return __ldcg(p);
    else return *p;
  };
  const float* base = scratch + static_cast<size_t>(mh) * splits * (D + 2);
  float M = -INFINITY;
  for (int s = 0; s < splits; ++s) M = fmaxf(M, ld(base + s * (D + 2)));
  float L = 0.f, A = 0.f;
  for (int s = 0; s < splits; ++s) {
    const float ms = ld(base + s * (D + 2));
    if (ms == -INFINITY) continue;
    const float c = __expf(ms - M);
    L += ld(base + s * (D + 2) + 1) * c;
    A += ld(base + s * (D + 2) + 2 + d) * c;
  }
  return A / L;
}

// The gated delta rule's key and value head sizes (kernels.cu's recurrences).
constexpr int kGdnDk = 128, kGdnDv = 128;

// The verify's recurrence over a tree (BACKLOG 26), path pth (its launch's blockIdx.y): its rows advance the state row by
// row from state_in (unwritten), as gdn_recurrent_body does for a chain; a leaf row's output is computed from the state
// its parent left, which it does not advance (the same products: S[k] * decay, then the fma with d). Only the rows
// marked kWrite are written (each row by one path).
// The body for (bx, pth) by one warp (lane its lane); qs, ks and steps are the warp's own shared memory.
__device__ __forceinline__ void gdn_recurrent_tree_body(const float* __restrict__ mixed, const float* __restrict__ g,
                                                        const float* __restrict__ beta, const float* state_in,
                                                        float* __restrict__ out, int H, int HV,
                                                        const TreeDesc* __restrict__ td, int bx, int pth, int lane,
                                                        float* qs, float* ks, int* steps) {
  const int hv = bx / 4, part = bx % 4, v = part * 32 + lane;
  const int h = hv / (HV / H);
  const int C = 2 * H * kGdnDk + HV * kGdnDv;
  const float* st = state_in + static_cast<size_t>(hv) * kGdnDk * kGdnDv;
  float S[kGdnDk];
#pragma unroll
  for (int k = 0; k < kGdnDk; ++k) S[k] = st[k * kGdnDv + v];
  const float scale = rsqrtf(static_cast<float>(kGdnDk));
  const int len = td->gdn_len[pth];
  // The path's steps, staged once (each row's loads depend on them).
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

}  // namespace
}  // namespace ling::kernels
