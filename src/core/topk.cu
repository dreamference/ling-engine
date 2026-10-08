// Top-K over long rows (the 248k-token vocabulary), K <= 64, for every row of a verify or a draft.
//
// Order: larger value first, and among equal values the lower index first, so the first entry is
// exactly what std::max_element returns on the host. Stage one gives each warp a segment of a row and
// keeps a sorted list of its best K; stage two merges the segments' lists with the same comparison, so
// the result does not depend on how the row was split.
#include "core/kernels.cuh"

#include <cfloat>
#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

constexpr int kSegments = 32;

__device__ __forceinline__ bool better(float v, int i, float v2, int i2) { return v > v2 || (v == v2 && i < i2); }

// A warp's sorted list in shared memory; lane 0 inserts, the warp reads the threshold.
struct WarpList {
  float* v;
  int* i;
};

__device__ void insert(WarpList L, int& count, int K, float v, int idx) {
  int p = min(count, K - 1);
  if (count == K && !better(v, idx, L.v[K - 1], L.i[K - 1])) return;
  while (p > 0 && better(v, idx, L.v[p - 1], L.i[p - 1])) {
    L.v[p] = L.v[p - 1];
    L.i[p] = L.i[p - 1];
    --p;
  }
  L.v[p] = v;
  L.i[p] = idx;
  if (count < K) ++count;
}

// Streams `n` candidates (value, index) given by `get` through the warp's list.
template <typename Get>
__device__ void warp_topk(Get get, int n, int K, WarpList L, int& count_out) {
  const int lane = threadIdx.x & 31;
  __shared__ int count_s;
  __shared__ float thr_v;
  __shared__ int thr_i;
  if (lane == 0) {
    count_s = 0;
    thr_v = -FLT_MAX;
    thr_i = 0x7fffffff;
  }
  __syncwarp();
  for (int base = 0; base < n; base += 32) {
    float v = -FLT_MAX;
    int idx = 0x7fffffff;
    if (base + lane < n) get(base + lane, v, idx);
    const bool full = count_s == K;
    const bool cand = (base + lane < n) && v == v && (!full || better(v, idx, thr_v, thr_i));
    unsigned mask = __ballot_sync(0xffffffffu, cand);
    while (mask) {
      const int src = __ffs(mask) - 1;
      mask &= mask - 1;
      const float sv = __shfl_sync(0xffffffffu, v, src);
      const int si = __shfl_sync(0xffffffffu, idx, src);
      if (lane == 0) {
        int c = count_s;
        insert(L, c, K, sv, si);
        count_s = c;
        if (c == K) {
          thr_v = L.v[K - 1];
          thr_i = L.i[K - 1];
        }
      }
      __syncwarp();
    }
  }
  count_out = count_s;
}

__global__ void topk_stage1(const float* __restrict__ x, int V, int K, float* __restrict__ pv, int* __restrict__ pi) {
  __shared__ float lv[64];
  __shared__ int li[64];
  const int row = blockIdx.y, seg = blockIdx.x;
  const int per = (V + kSegments - 1) / kSegments;
  const int start = seg * per, n = max(0, min(V, start + per) - start);
  const float* r = x + static_cast<size_t>(row) * V + start;
  int count = 0;
  warp_topk([&](int e, float& v, int& i) { v = r[e]; i = start + e; }, n, K, WarpList{lv, li}, count);
  __syncwarp();
  const int lane = threadIdx.x;
  for (int e = lane; e < K; e += 32) {
    const size_t o = (static_cast<size_t>(row) * kSegments + seg) * K + e;
    pv[o] = e < count ? lv[e] : -FLT_MAX;
    pi[o] = e < count ? li[e] : 0x7fffffff;
  }
}

__global__ void topk_stage2(const float* __restrict__ pv, const int* __restrict__ pi, int K, float* __restrict__ vals,
                            int* __restrict__ ids) {
  __shared__ float lv[64];
  __shared__ int li[64];
  const int row = blockIdx.x;
  const float* rv = pv + static_cast<size_t>(row) * kSegments * K;
  const int* ri = pi + static_cast<size_t>(row) * kSegments * K;
  int count = 0;
  warp_topk([&](int e, float& v, int& i) { v = rv[e]; i = ri[e]; }, kSegments * K, K, WarpList{lv, li}, count);
  __syncwarp();
  for (int e = threadIdx.x; e < K; e += 32) {
    vals[static_cast<size_t>(row) * K + e] = lv[e];
    ids[static_cast<size_t>(row) * K + e] = li[e];
  }
}

}  // namespace

size_t topk_scratch_floats(int rows, int K) { return static_cast<size_t>(rows) * kSegments * K * 2; }

void topk_rows(const float* x, int rows, int V, int K, float* scratch, float* vals, int* ids, cudaStream_t s) {
  if (K < 1 || K > 64) throw std::runtime_error("topk_rows: K must be in [1, 64]");
  float* pv = scratch;
  int* pi = reinterpret_cast<int*>(scratch + static_cast<size_t>(rows) * kSegments * K);
  topk_stage1<<<dim3(kSegments, rows), 32, 0, s>>>(x, V, K, pv, pi);
  topk_stage2<<<rows, 32, 0, s>>>(pv, pi, K, vals, ids);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("topk_rows: ") + cudaGetErrorString(e));
}

}  // namespace ling::kernels
