// Compile-only probe: does sm_121a accept the sm_100 tensor-memory path (spec 16.1 item 3)?
//
// tcgen05 (TMEM allocation, the 5th-generation tensor-core MMA, 2-SM MMA) is what MpFA and
// CuTile build on for B200. It is expected to be absent on sm_121. This file is meant to fail:
//   nvcc -std=c++20 -gencode arch=compute_121a,code=sm_121a -c tcgen05_probe.cu
// A ptxas error of the form "not supported on .target sm_121a" is the measurement; the same
// file compiles for -gencode arch=compute_100a,code=sm_100a.
#include <cuda_runtime.h>
#include <cstdint>
__global__ void tmem_alloc(uint32_t* out) {
  __shared__ uint32_t addr;
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
               :: "r"((uint32_t)__cvta_generic_to_shared(&addr)));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  __syncthreads();
  out[threadIdx.x] = addr;
}
