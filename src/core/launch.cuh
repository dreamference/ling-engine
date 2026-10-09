// Kernel launches with programmatic dependent launch (PDL; SPEC.md §16.7, M3). CUDA source files only.
//
// Normally a kernel starts only after the previous kernel in the stream has finished and its writes have
// been flushed: every launch pays the gap between the last block of one kernel and the first block of the
// next, and the next kernel's first weight loads start from an idle bus. With PDL (sm_90 and later; M0 §2
// measured it on this GB10), a kernel launched with the programmatic-serialization attribute may start while
// its predecessor is still running, once every block of the predecessor has executed
// griddepcontrol.launch_dependents (or exited). Its blocks then wait in griddepcontrol.wait until the
// predecessor has completed and its memory is visible.
//
// The rule every kernel launched through launch_kernel() follows, and the only thing that keeps PDL exact:
//   - pdl_begin() (wait, then trigger) is its first statement, before any read of memory a previous kernel
//     writes and before any write at all. A kernel whose prologue only reads immutable data (the weights) may
//     issue those reads before pdl_begin(), and nothing else: stream_gemm_kernel and bf16_rows_kernel do.
//   - Every block reaches pdl_begin() (no early return before it), so a kernel never completes before its
//     predecessor: the chain of waits makes completion transitive, and a kernel three launches later that
//     reads this one's predecessor's output is still ordered after it.
// With PDL off (the default) the attribute is not set, griddepcontrol.wait returns at once and the launch is
// an ordinary one, so the kernels compute exactly the same thing either way.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

#include "core/kernels.cuh"

namespace ling::kernels {

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Lets the next kernel in the stream launch as soon as every block of this one has started (its blocks
// then wait in pdl_wait() for this kernel to finish).
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The first statement of every kernel launched through launch_kernel().
__device__ __forceinline__ void pdl_begin() {
  pdl_wait();
  pdl_trigger();
}

// kernel<<<grid, block, smem, s>>>(args...), with the PDL attribute when `pdl`.
template <typename... Params, typename... Args>
void launch_kernel_pdl(bool pdl, const char* what, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem,
                       cudaStream_t s, Args&&... args) {
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// The same with the attribute when set_pdl(true).
template <typename... Params, typename... Args>
void launch_kernel(const char* what, void (*kernel)(Params...), dim3 grid, dim3 block, size_t smem, cudaStream_t s,
                   Args&&... args) {
  launch_kernel_pdl(pdl_enabled(), what, kernel, grid, block, smem, s, std::forward<Args>(args)...);
}

// The attention kernels are left out of PDL unless LING_PDL_ATTN=1: llama.cpp found a race on DGX Spark in
// its flash-attention kernels launched with PDL (ggml-org/llama.cpp#23825, attributed to an internal bug, fixed
// there by launching those kernels without it, ~0.2% slower). Until a stress run here shows ours are safe,
// they launch as ordinary kernels; they still begin with pdl_begin(), which is then a no-op.
bool pdl_attention_enabled();

}  // namespace ling::kernels
