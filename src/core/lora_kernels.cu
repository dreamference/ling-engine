// The LoRA kernels (lora_kernels.cuh). Both are bandwidth-bound on the adapter's weights: a decode step reads each
// slot's A and B once per site (rank 16: about 96 MB for every site of the 64 layers), the rows' inputs from L2.
#include "core/launch.cuh"
#include "core/lora_kernels.cuh"

namespace ling::kernels {
namespace {

constexpr int kThreads = 256;
constexpr int kTileRows = 8;  // rows per block: the shrink's accumulators, the expand's t tile
constexpr int kMaxRank = kLoraRankRows / 4;

static_assert(kTileRows <= 32, "a tile's rows are one mask");

__device__ __forceinline__ int row_slot(const LoraRows& rows, int r) {
  int i = 0;
  while (i + 1 < rows.n && r >= rows.row0[i + 1]) ++i;
  return rows.slot_dev ? rows.slot_dev[i] : rows.slot[i];
}

// The tile's rows on slot s, as a mask (the same for every thread of the block).
__device__ __forceinline__ unsigned tile_mask(const LoraRows& rows, int r0, int M, int s) {
  unsigned mask = 0;
  for (int i = 0; i < kTileRows && r0 + i < M; ++i)
    if (row_slot(rows, r0 + i) == s) mask |= 1u << i;
  return mask;
}

// One A row j of slot s's site against the tile's rows on s. Each thread runs over K in a fixed stride, then the
// warps' butterflies and the warps in order: a row's sum never depends on the tile it shares.
__global__ void __launch_bounds__(kThreads) lora_shrink_kernel(const float* __restrict__ x, int ldx, int M, LoraRows rows,
                                                                const LoraSiteDev* __restrict__ site, int slot_stride,
                                                                float* __restrict__ t) {
  const int s = blockIdx.z, j = blockIdx.x, r0 = blockIdx.y * kTileRows;
  const unsigned mask = tile_mask(rows, r0, M, s);
  if (!mask) return;
  const LoraSiteDev& d = site[size_t(s) * slot_stride];
  if (!d.a || j >= d.blocks * d.rank) return;
  const __nv_bfloat16* a = d.a + size_t(j) * d.K;
  float acc[kTileRows];
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) acc[i] = 0.f;
  for (int k = threadIdx.x * 4; k < d.K; k += kThreads * 4) {
    const float2 a01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a + k));
    const float2 a23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(a + k + 2));
#pragma unroll
    for (int i = 0; i < kTileRows; ++i) {
      if (!(mask >> i & 1u)) continue;
      const float4 v = *reinterpret_cast<const float4*>(x + size_t(r0 + i) * ldx + k);
      acc[i] = fmaf(v.x, a01.x, acc[i]);
      acc[i] = fmaf(v.y, a01.y, acc[i]);
      acc[i] = fmaf(v.z, a23.x, acc[i]);
      acc[i] = fmaf(v.w, a23.y, acc[i]);
    }
  }
  __shared__ float part[kThreads / 32][kTileRows];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) {
    float v = acc[i];
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) part[warp][i] = v;
  }
  __syncthreads();
  const int i = threadIdx.x;
  if (i < kTileRows && (mask >> i & 1u)) {
    float v = 0.f;
    for (int w = 0; w < kThreads / 32; ++w) v += part[w][i];
    t[size_t(r0 + i) * kLoraRankRows + j] = v;
  }
}

// One output row n of one segment of slot s's site, for the tile's rows on s: y += scale * (B[n] . t[block]).
__global__ void __launch_bounds__(kThreads) lora_expand_kernel(const float* __restrict__ t, int M, LoraRows rows,
                                                                const LoraSiteDev* __restrict__ site, int slot_stride,
                                                                LoraOuts outs, int tiles) {
  const int s = blockIdx.z / tiles, r0 = (blockIdx.z % tiles) * kTileRows, sg = blockIdx.y;
  const unsigned mask = tile_mask(rows, r0, M, s);
  if (!mask) return;
  const LoraSiteDev& d = site[size_t(s) * slot_stride];
  if (!d.a || sg >= d.nsegs) return;
  const LoraSegDev seg = d.segs[sg];
  const int R = d.rank;
  __shared__ float ts[kTileRows][kMaxRank];
  for (int e = threadIdx.x; e < kTileRows * kMaxRank; e += kThreads) {
    const int i = e / kMaxRank, jj = e % kMaxRank;
    ts[i][jj] = (mask >> i & 1u) && jj < R ? t[size_t(r0 + i) * kLoraRankRows + seg.block * R + jj] : 0.f;
  }
  __syncthreads();
  const int n = seg.n0 + blockIdx.x * kThreads + threadIdx.x;
  if (n >= seg.n1) return;
  const __nv_bfloat16* b = d.b + seg.b_off + size_t(n - seg.n0) * R;
  float bv[kMaxRank];
#pragma unroll
  for (int jj = 0; jj < kMaxRank; ++jj) bv[jj] = jj < R ? __bfloat162float(b[jj]) : 0.f;
  float* y = outs.y[seg.out];
  const int ld = outs.ld[seg.out];
#pragma unroll
  for (int i = 0; i < kTileRows; ++i) {
    if (!(mask >> i & 1u)) continue;
    float acc = 0.f;
#pragma unroll
    for (int jj = 0; jj < kMaxRank; ++jj)
      if (jj < R) acc = fmaf(bv[jj], ts[i][jj], acc);
    float& out = y[size_t(r0 + i) * ld + n];
    out = out + d.scale * acc;
  }
}

}  // namespace

void lora_shrink(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                 int slots, float* t, cudaStream_t stream) {
  if (M <= 0 || slots <= 0) return;
  const dim3 grid(kLoraRankRows, (M + kTileRows - 1) / kTileRows, slots);
  launch_kernel("lora_shrink", lora_shrink_kernel, grid, dim3(kThreads), 0, stream, x, ldx, M, rows, site,
                slot_stride, t);
}

void lora_expand(const float* t, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride, int slots,
                 const LoraOuts& outs, int max_n, cudaStream_t stream) {
  if (M <= 0 || slots <= 0 || max_n <= 0) return;
  const int tiles = (M + kTileRows - 1) / kTileRows;
  const dim3 grid((max_n + kThreads - 1) / kThreads, kLoraSegs, slots * tiles);
  launch_kernel("lora_expand", lora_expand_kernel, grid, dim3(kThreads), 0, stream, t, M, rows, site, slot_stride,
                outs, tiles);
}

}  // namespace ling::kernels
