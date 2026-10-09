// The Gated DeltaNet's causal conv for the prefill path (any number of tokens).
//
// The rows path's conv (kernels.cu) walks the tokens in order, one thread per channel, in place: right for
// 1-16 tokens, latency-bound for 2,048. The conv is a 4-tap filter over raw inputs, so here it runs in
// parallel over tokens and channels, out of place, and the new window (the last three raw inputs) is
// written afterwards. Each token is computed the same way whatever M is.
//
// The recurrence stays the rows path's kernel. Measured at 2,048 tokens it takes 3.7 ms per layer, and two
// rewrites took as long: four lanes per value column with the inputs prefetched or staged in shared
// memory (3.7-4.3 ms). The sequential form is bound by instruction issue: one SM per head does ~5 FP32
// operations per state element per token, 128 x 128 elements, plus each thread's own q and k norms. The
// chunked (WY) form on tensor cores is the way past it.
#include "core/kernels.cuh"

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

// The window after the last token: the three most recent raw inputs.
__global__ void conv_window_kernel(const float* __restrict__ in, float* __restrict__ state, int M, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int tt = M - 3 + j;
    s[j] = tt >= 0 ? in[static_cast<size_t>(tt) * C + c] : state[c * 3 + 3 + tt];
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) state[c * 3 + j] = s[j];
}

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

void gdn_conv_prefill(const float* in, float* out, float* conv_state, const __nv_bfloat16* w, int M, int C,
                      cudaStream_t s) {
  const size_t n = static_cast<size_t>(M) * C;
  conv_prefill_kernel<<<static_cast<int>(std::min<size_t>((n + 255) / 256, 48 * 64)), 256, 0, s>>>(in, out, conv_state,
                                                                                                     w, M, C);
  check_launch("gdn_conv_prefill");
  conv_window_kernel<<<(C + 255) / 256, 256, 0, s>>>(in, conv_state, M, C);
  check_launch("gdn_conv_window");
}

}  // namespace ling::kernels
