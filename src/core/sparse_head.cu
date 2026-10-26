// The drafter's context-aware sparse head (BACKLOG 19): instead of all 248,320 rows of the target's LM head, the
// drafter scores a fixed list of frequent rows, the session's context vocabulary (every token id of its prompt and
// output) and the rows a rank-256 copy of the head rates best for this step's drafted positions, and takes its
// top-16 candidates among those. The selected rows' logits are the full head's,
// bit for bit (stream_gemm_gather_nvfp4), so a position whose full-head top 16 lie inside the selection proposes
// exactly what it did; the target verifies every token either way, so the output is unchanged by construction.
//
// Per step: p = h . V^T (bf16_rows, M x 256), then here: approx[n] = max over the M rows of p[m] . US[n] (US the
// head's rank-256 left factor scaled by its singular values, BF16), a radix select of the `dynamic` best rows no mask
// marks, and their ids with the marked ones in id order. The masks: each session's fixed list joined with its context
// vocabulary (one per sequence of a batched draft, joined here).
#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ling::kernels {
namespace {

constexpr int kRank = 256;

// An order-preserving map of floats to unsigned keys (larger float, larger key); marked rows get key 0.
__device__ __forceinline__ uint32_t float_key(float f) {
  const uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ bool marked(const SelMasks& ms, int n) {
  bool on = false;
#pragma unroll
  for (int i = 0; i < kMaxSeqs; ++i)
    if (i < ms.n) on |= ms.m[i][n] != 0;
  return on;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// approx = P . US^T on tensor cores, a warp per 16 rows of US at a time: lane (g, t) loads 64 contiguous values of
// rows g and g + 8 (the MMA's K index permuted so a lane's fragments are contiguous, the same for P: the sum over
// the rank is the same products in another order, an approximation either way), P (BF16) in shared memory, and the
// maximum over P's NT * 8 rows (those past M masked) is the row's score.
template <int NT>
__global__ void __launch_bounds__(256) approx_kernel(const float* __restrict__ p, int M, const __nv_bfloat16* __restrict__ us,
                                                     int N, SelMasks fixed, uint32_t* __restrict__ keys) {
  pdl_begin();  // first statement: see launch.cuh
  constexpr int PS = kRank + 8;  // padded row stride (BF16)
  __shared__ __align__(16) __nv_bfloat16 ps[NT * 8 * PS];
  for (int i = threadIdx.x; i < NT * 8 * kRank; i += blockDim.x) {
    const int m = i / kRank, c = i % kRank;
    ps[m * PS + c] = __float2bfloat16(m < M ? p[m * kRank + c] : 0.f);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int warps = gridDim.x * (blockDim.x / 32), warp = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  for (int n0 = warp * 16; n0 < N; n0 += warps * 16) {
    uint4 a[2][8];  // rows g and g + 8: values t * 64 .. t * 64 + 63
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = min(n0 + g + 8 * h, N - 1);
      const uint4* src = reinterpret_cast<const uint4*>(us + static_cast<size_t>(n) * kRank + t * 64);
#pragma unroll
      for (int q = 0; q < 8; ++q) a[h][q] = __ldg(src + q);
    }
    float c[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
#pragma unroll
    for (int st = 0; st < 16; ++st) {  // physical values t * 64 + 4 st .. + 3
      const uint4& ra = a[0][st >> 1];
      const uint4& rb = a[1][st >> 1];
      const uint32_t a0 = (st & 1) ? ra.z : ra.x, a2 = (st & 1) ? ra.w : ra.y;
      const uint32_t a1 = (st & 1) ? rb.z : rb.x, a3 = (st & 1) ? rb.w : rb.y;
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint2 bv = *reinterpret_cast<const uint2*>(ps + (j * 8 + g) * PS + t * 64 + 4 * st);
        mma_bf16(c[j], a0, a1, a2, a3, bv.x, bv.y);
      }
    }
    // c[j]: rows g (0, 1) and g + 8 (2, 3), columns j * 8 + 2t, + 1.
    float best[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e)
        if (j * 8 + 2 * t + e < M) {
          best[0] = fmaxf(best[0], c[j][e]);
          best[1] = fmaxf(best[1], c[j][2 + e]);
        }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      best[h] = fmaxf(best[h], __shfl_xor_sync(0xffffffffu, best[h], 1));
      best[h] = fmaxf(best[h], __shfl_xor_sync(0xffffffffu, best[h], 2));
      const int n = n0 + g + 8 * h;
      if (t == 0 && n < N) keys[n] = marked(fixed, n) ? 0u : float_key(best[h]);
    }
  }
}

// The key of the `dynamic`-th best, 12 + 12 + 8 bits at a time: each level a histogram of the keys that share the
// prefix found so far, per block in shared memory and then summed into global memory, and one warp that finds the bin
// the threshold falls in. state: [0] the prefix, [1] how many keys are still wanted; hist: 4096 bins per level.

__global__ void __launch_bounds__(512) hist_kernel(const uint32_t* __restrict__ keys, int N, uint32_t* __restrict__ hist,
                                                   const uint32_t* __restrict__ state, int level) {
  pdl_begin();  // first statement: see launch.cuh
  __shared__ uint32_t h[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) h[i] = 0;
  __syncthreads();
  const int shift = level == 0 ? 20 : level == 1 ? 8 : 0;
  const uint32_t pmask = level == 0 ? 0u : level == 1 ? 0xfff00000u : 0xffffff00u;
  const uint32_t prefix = level ? state[0] : 0u, bins = level == 2 ? 255u : 4095u;
  for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += gridDim.x * blockDim.x) {
    const uint32_t k = keys[n];
    if (k != 0 && (k & pmask) == prefix) atomicAdd(&h[(k >> shift) & bins], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i <= static_cast<int>(bins); i += blockDim.x)
    if (h[i]) atomicAdd(&hist[level * 4096 + i], h[i]);
}

// One warp: lane l sums bins [128 l, 128 l + 128); lane 0 walks the lanes' sums from the top, then its lane's bins.
__global__ void find_kernel(const uint32_t* __restrict__ hist, uint32_t* __restrict__ state, int dynamic, int level,
                            uint32_t* __restrict__ thr) {
  pdl_begin();  // first statement: see launch.cuh
  const uint32_t* h = hist + level * 4096;
  const int lane = threadIdx.x, per = level == 2 ? 8 : 128;
  uint32_t sum = 0;
  for (int b = lane * per; b < lane * per + per; ++b) sum += h[b];
  __shared__ uint32_t lanes[32];
  lanes[lane] = sum;
  __syncwarp();
  if (lane != 0) return;
  const uint32_t want = level ? state[1] : static_cast<uint32_t>(dynamic);
  uint32_t above = 0;
  int c = 31;
  for (; c > 0 && above + lanes[c] < want; --c) above += lanes[c];
  int b = c * per + per - 1;
  for (; b > c * per && above + h[b] < want; --b) above += h[b];
  const int shift = level == 0 ? 20 : level == 1 ? 8 : 0;
  const uint32_t prefix = (level ? state[0] : 0u) | (static_cast<uint32_t>(b) << shift);
  state[0] = prefix;
  state[1] = want > above ? want - above : 0;
  if (level == 2) thr[0] = prefix;
}

// The selection's flags: the marked rows, and every key at or above the threshold (keys equal to it are rare; every
// tie is taken, and the capacity bounds the total).
__global__ void flags_kernel(const uint32_t* __restrict__ keys, SelMasks fixed, int N,
                             const uint32_t* __restrict__ thr, uint8_t* __restrict__ flags) {
  pdl_begin();  // first statement: see launch.cuh
  const uint32_t t = thr[0];
  for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += gridDim.x * blockDim.x)
    flags[n] = marked(fixed, n) || (keys[n] != 0 && keys[n] >= t && t != 0);
}

__global__ void clamp_kernel(int* nrows, int cap) {
  pdl_begin();  // first statement: see launch.cuh
  if (threadIdx.x == 0) *nrows = min(*nrows, cap);
}

__global__ void mark_kernel(uint8_t* __restrict__ mask, const int* __restrict__ ids, int n) {
  pdl_begin();  // first statement: see launch.cuh
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) mask[ids[i]] = 1;
}

__global__ void ids_kernel(const int* __restrict__ rows, int* __restrict__ ids, int n) {
  pdl_begin();  // first statement: see launch.cuh
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) ids[i] = rows[ids[i]];
}

size_t cub_bytes(int N) {
  size_t bytes = 0;
  cub::DeviceSelect::Flagged(nullptr, bytes, thrust::counting_iterator<int>(0), static_cast<const uint8_t*>(nullptr),
                             static_cast<int*>(nullptr), static_cast<int*>(nullptr), N);
  return bytes;
}

}  // namespace

// keys [N] u32, flags [N] u8, the threshold and state (64 bytes), the histograms (3 x 4096 bins), then CUB's storage.
size_t sparse_head_scratch_bytes(int N) { return size_t(N) * 5 + 256 + 3 * 4096 * 4 + cub_bytes(N); }

void sparse_head_select(const float* p, int M, const __nv_bfloat16* us, int N, int R, SelMasks fixed_mask, int dynamic,
                        int cap, void* scratch, int* rows, int* nrows, cudaStream_t s) {
  if (R != kRank) throw std::runtime_error("sparse_head_select: built for rank 256");
  if (fixed_mask.n < 1 || fixed_mask.n > kMaxSeqs) throw std::runtime_error("sparse_head_select: 1 to kMaxSeqs masks");
  if (M < 1 || M > kMaxStreamRows) throw std::runtime_error("sparse_head_select: M must be in [1, 32]");
  uint8_t* base = static_cast<uint8_t*>(scratch);
  uint32_t* keys = reinterpret_cast<uint32_t*>(base);
  uint8_t* flags = base + size_t(N) * 4;
  uint32_t* thr = reinterpret_cast<uint32_t*>(base + ((size_t(N) * 5 + 15) & ~size_t(15)));
  uint32_t* state = thr + 4;
  uint32_t* hist = thr + 16;
  void* temp = reinterpret_cast<uint8_t*>(hist) + 3 * 4096 * 4;
  size_t temp_bytes = cub_bytes(N);
  // The approximate scores loop over the rows grid-stride: exactly as many blocks as fit at once, so none waits for
  // a first wave to finish.
  auto grid = [](auto kernel) {
    static int sms = 0;
    if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    int per = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, kernel, 256, 0);
    return dim3(sms * std::max(per, 1));
  };
  if (M <= 8) launch_kernel("sparse_head approx", approx_kernel<1>, grid(approx_kernel<1>), dim3(256), 0, s, p, M, us, N, fixed_mask, keys);
  else if (M <= 16) launch_kernel("sparse_head approx", approx_kernel<2>, grid(approx_kernel<2>), dim3(256), 0, s, p, M, us, N, fixed_mask, keys);
  else launch_kernel("sparse_head approx", approx_kernel<4>, grid(approx_kernel<4>), dim3(256), 0, s, p, M, us, N, fixed_mask, keys);
  if (cudaMemsetAsync(hist, 0, 3 * 4096 * 4, s) != cudaSuccess) throw std::runtime_error("sparse_head_select: memset");
  for (int level = 0; level < 3; ++level) {
    launch_kernel("sparse_head hist", hist_kernel, dim3(96), dim3(512), 0, s, static_cast<const uint32_t*>(keys), N, hist,
                  static_cast<const uint32_t*>(state), level);
    launch_kernel("sparse_head find", find_kernel, dim3(1), dim3(32), 0, s, static_cast<const uint32_t*>(hist), state,
                  dynamic, level, thr);
  }
  launch_kernel("sparse_head flags", flags_kernel, dim3((N + 255) / 256), dim3(256), 0, s,
                static_cast<const uint32_t*>(keys), fixed_mask, N, static_cast<const uint32_t*>(thr), flags);
  if (cub::DeviceSelect::Flagged(temp, temp_bytes, thrust::counting_iterator<int>(0), flags, rows, nrows, N, s) !=
      cudaSuccess)
    throw std::runtime_error("sparse_head_select: compaction");
  launch_kernel("sparse_head clamp", clamp_kernel, dim3(1), dim3(32), 0, s, nrows, cap);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("sparse_head_select: ") + cudaGetErrorString(e));
}

void sparse_head_ids(const int* rows, int* ids, int n, cudaStream_t s) {
  launch_kernel("sparse_head ids", ids_kernel, dim3((n + 255) / 256), dim3(256), 0, s, rows, ids, n);
}

void draft_head_rows(const SparseHeadBufs& b, const float* x, int n, int E, const uint8_t* const* masks, int K16,
                     float* vals, int* ids, cudaStream_t s) {
  if (n < 1 || E < 1 || E > kMaxStreamRows) throw std::runtime_error("draft_head_rows: 1 to 32 rows per sequence");
  auto check = [](cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("draft_head_rows: ") + what + ": " + cudaGetErrorString(e));
  };
  // The full head's sequences, as many at a time as fit 32 rows (contiguous in x: in place; else copied together).
  const int per = kMaxStreamRows / E;
  std::vector<int> full;
  for (int i = 0; i < n; ++i)
    if (!masks[i]) full.push_back(i);
  for (size_t c0 = 0; c0 < full.size(); c0 += per) {
    const int cn = static_cast<int>(std::min(full.size() - c0, size_t(per)));
    bool contiguous = true;
    for (int j = 1; j < cn; ++j) contiguous &= full[c0 + j] == full[c0] + j;
    const float* in = x + size_t(full[c0]) * E * b.K;
    if (!contiguous) {
      for (int j = 0; j < cn; ++j)
        check(cudaMemcpyAsync(b.xtmp + size_t(j) * E * b.K, x + size_t(full[c0 + j]) * E * b.K,
                              size_t(E) * b.K * sizeof(float), cudaMemcpyDeviceToDevice, s),
              "copy");
      in = b.xtmp;
    }
    to_half_rows(in, cn * E, b.K, b.xh, b.xinv, s);
    stream_gemm_nvfp4(b.xh, b.xinv, cn * E, b.w, b.table, b.scale2, b.logits, b.N, b.K, s);
    for (int j = 0; j < cn; ++j) {
      const size_t o = size_t(full[c0 + j]) * E * K16;
      topk_rows(b.logits + size_t(j) * E * b.N, E, b.N, K16, b.topk_scratch, vals + o, ids + o, s);
    }
  }
  // Each sparse sequence alone: the rank-256 scores choose the rows beyond its mask's; their exact logits (the full
  // head's bits, read from the same tiled blob) give its top K16, whose columns map back to token ids.
  for (int i = 0; i < n; ++i) {
    if (!masks[i]) continue;
    const float* xi = x + size_t(i) * E * b.K;
    const size_t o = size_t(i) * E * K16;
    bf16_rows(xi, E, b.vt, b.p, b.R, b.K, s);
    sparse_head_select(b.p, E, b.us, b.N, b.R, SelMasks{{masks[i]}, 1}, b.dynamic, b.cap, b.scratch, b.rows, b.nrows, s);
    to_half_rows(xi, E, b.K, b.xh, b.xinv, s);
    stream_gemm_gather_nvfp4(b.xh, b.xinv, E, b.w, b.table ? b.raw : nullptr, b.scale2, b.rows, b.nrows, b.cap, b.logits,
                             b.K, s);
    topk_rows(b.logits, E, b.cap, K16, b.topk_scratch, vals + o, ids + o, s);
    sparse_head_ids(b.rows, ids + o, E * K16, s);
  }
}

void sparse_head_mark(uint8_t* mask, const int* ids, int n, cudaStream_t s) {
  if (n <= 0) return;
  launch_kernel("sparse_head mark", mark_kernel, dim3((n + 255) / 256), dim3(256), 0, s, mask, ids, n);
}

}  // namespace ling::kernels
