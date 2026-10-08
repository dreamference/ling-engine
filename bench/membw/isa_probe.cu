// What sm_121 exposes to the kernels of spec section 9 (M0, spec 16.1 item 3).
//
// Each probe runs on the device and checks its own result:
//   1. warp-level block-scaled NVFP4 MMA (mma.sync kind::mxf4nvf4, scale_vec::4X, UE4M3 scales);
//   2. warp-level FP8 MMA (mma.sync kind::f8f6f4, E4M3 x E4M3);
//   3. TMA: a 2-D tile copied by cp.async.bulk.tensor with an mbarrier;
//   4. thread-block clusters with distributed shared memory (one block reads its neighbour's smem),
//      and the largest cluster the device will schedule;
//   5. programmatic dependent launch: the secondary kernel starts before the primary ends.
//
// Build: nvcc -O3 -std=c++20 -gencode arch=compute_121a,code=sm_121a isa_probe.cu -o isa_probe -lcuda
// (`-arch=sm_121a` alone emitted compute_121 PTX here, which ptxas rejects for the block-scaled
// and kind::f8f6f4 MMAs: name the arch-specific target explicitly.)
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static int failures = 0;
static void report(const char* what, bool ok, const char* detail) {
  printf("%-44s %s  %s\n", what, ok ? "OK  " : "FAIL", detail);
  if (!ok) ++failures;
}

// ---- 1 and 2: MMA -------------------------------------------------------------------------
// All operands are 1.0 and all scales 1.0, so every output element equals K.
__global__ void mma_nvfp4(float* out) {
  uint32_t a = 0x22222222u;   // eight E2M1 values of 1.0
  uint32_t b = 0x22222222u;
  uint32_t sf = 0x38383838u;  // four UE4M3 scales of 1.0
  uint16_t z = 0;
  float d0, d1, d2, d3, c = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15,%16}, {%17}, {%18,%19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b), "f"(c), "f"(c), "f"(c), "f"(c),
        "r"(sf), "h"(z), "h"(z), "r"(sf), "h"(z), "h"(z));
  out[threadIdx.x * 4 + 0] = d0; out[threadIdx.x * 4 + 1] = d1;
  out[threadIdx.x * 4 + 2] = d2; out[threadIdx.x * 4 + 3] = d3;
}

__global__ void mma_fp8(float* out) {
  uint32_t a = 0x38383838u;  // four E4M3 values of 1.0
  float d0, d1, d2, d3, c = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "f"(c), "f"(c), "f"(c), "f"(c));
  out[threadIdx.x * 4 + 0] = d0; out[threadIdx.x * 4 + 1] = d1;
  out[threadIdx.x * 4 + 2] = d2; out[threadIdx.x * 4 + 3] = d3;
}

// ---- 3: TMA ---------------------------------------------------------------------------------
constexpr int TW = 64, TH = 32;  // tile: 64 x 32 uint32
__global__ void tma_tile(const __grid_constant__ CUtensorMap map, uint32_t* out, int x0, int y0) {
  __shared__ alignas(128) uint32_t tile[TH][TW];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
  __syncthreads();
  cuda::barrier<cuda::thread_scope_block>::arrival_token tok;
  if (threadIdx.x == 0) {
    int32_t coords[2] = {x0, y0};
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared, cuda::ptx::space_global, &tile, &map, coords,
                                    cuda::device::barrier_native_handle(bar));
    tok = cuda::device::barrier_arrive_tx(bar, 1, sizeof(tile));
  } else {
    tok = bar.arrive();
  }
  bar.wait(std::move(tok));
  for (int i = threadIdx.x; i < TW * TH; i += blockDim.x) out[i] = tile[i / TW][i % TW];
}

// ---- 4: clusters and distributed shared memory ----------------------------------------------
__global__ void __cluster_dims__(2, 1, 1) dsmem(int* out) {
  __shared__ int v;
  cg::cluster_group cl = cg::this_cluster();
  if (threadIdx.x == 0) v = 1000 + (int)cl.block_rank();
  cl.sync();
  if (threadIdx.x == 0) {
    int* other = cl.map_shared_rank(&v, (cl.block_rank() + 1) % cl.num_blocks());
    out[blockIdx.x] = *other;
  }
  cl.sync();
}

__global__ void cluster_any(int* out) {
  if (threadIdx.x == 0) out[blockIdx.x] = (int)cg::this_cluster().num_blocks();
}

// ---- 5: programmatic dependent launch ------------------------------------------------------
__device__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__global__ void primary(unsigned long long* t, volatile int* flag) {
  // Let the secondary launch early, then keep running for ~200 us.
  cudaTriggerProgrammaticLaunchCompletion();
  unsigned long long s = now_ns();
  while (now_ns() - s < 200000) {}
  if (threadIdx.x == 0 && blockIdx.x == 0) { t[1] = now_ns(); *flag = 1; }
}
__global__ void secondary(unsigned long long* t, const volatile int* flag, int* seen) {
  if (threadIdx.x == 0 && blockIdx.x == 0) t[2] = now_ns();  // prologue, before the dependency
  cudaGridDependencySynchronize();
  if (threadIdx.x == 0 && blockIdx.x == 0) { t[3] = now_ns(); *seen = *flag; }
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("%s sm_%d%d, %d SMs, smem/block opt-in %zu KB, smem/SM %zu KB, regs/SM %d, L2 %.1f MB\n", p.name,
         p.major, p.minor, p.multiProcessorCount, p.sharedMemPerBlockOptin / 1024, p.sharedMemPerMultiprocessor / 1024,
         p.regsPerMultiprocessor, p.l2CacheSize / 1e6);
  char buf[256];

  float* f;
  CK(cudaMalloc(&f, 32 * 4 * sizeof(float)));
  std::vector<float> h(128);
  mma_nvfp4<<<1, 32>>>(f);
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess) CK(cudaMemcpy(h.data(), f, 128 * 4, cudaMemcpyDeviceToHost));
  bool ok = e == cudaSuccess;
  for (float x : h) ok = ok && x == 64.f;
  snprintf(buf, sizeof buf, "%s, d[0]=%g (expect 64)", cudaGetErrorString(e), h[0]);
  report("NVFP4 block-scaled mma.sync m16n8k64", ok, buf);

  mma_fp8<<<1, 32>>>(f);
  e = cudaDeviceSynchronize();
  if (e == cudaSuccess) CK(cudaMemcpy(h.data(), f, 128 * 4, cudaMemcpyDeviceToHost));
  ok = e == cudaSuccess;
  for (float x : h) ok = ok && x == 32.f;
  snprintf(buf, sizeof buf, "%s, d[0]=%g (expect 32)", cudaGetErrorString(e), h[0]);
  report("FP8 E4M3 mma.sync m16n8k32 (kind::f8f6f4)", ok, buf);

  {  // TMA
    const int W = 1024, H = 256;
    std::vector<uint32_t> src(W * H);
    for (int i = 0; i < W * H; ++i) src[i] = i;
    uint32_t *g, *o;
    CK(cudaMalloc(&g, src.size() * 4));
    CK(cudaMalloc(&o, TW * TH * 4));
    CK(cudaMemcpy(g, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
    CUtensorMap map;
    cuuint64_t dims[2] = {W, H};
    cuuint64_t strides[1] = {W * 4};
    cuuint32_t box[2] = {TW, TH};
    cuuint32_t es[2] = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, g, dims, strides, box, es,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    int x0 = 128, y0 = 64;
    tma_tile<<<1, 128>>>(map, o, x0, y0);
    e = cudaDeviceSynchronize();
    std::vector<uint32_t> t(TW * TH, 0);
    if (e == cudaSuccess) CK(cudaMemcpy(t.data(), o, t.size() * 4, cudaMemcpyDeviceToHost));
    ok = r == CUDA_SUCCESS && e == cudaSuccess;
    for (int i = 0; i < TW * TH && ok; ++i) ok = t[i] == (uint32_t)((y0 + i / TW) * W + x0 + i % TW);
    snprintf(buf, sizeof buf, "encode %d, %s", (int)r, cudaGetErrorString(e));
    report("TMA 2-D tile (cp.async.bulk.tensor + mbarrier)", ok, buf);
  }

  {  // clusters
    int* o;
    CK(cudaMalloc(&o, 64 * sizeof(int)));
    dsmem<<<4, 32>>>(o);
    e = cudaDeviceSynchronize();
    int r[4] = {0};
    if (e == cudaSuccess) CK(cudaMemcpy(r, o, sizeof r, cudaMemcpyDeviceToHost));
    ok = e == cudaSuccess && r[0] == 1001 && r[1] == 1000 && r[2] == 1001 && r[3] == 1000;
    snprintf(buf, sizeof buf, "%s, reads %d %d %d %d", cudaGetErrorString(e), r[0], r[1], r[2], r[3]);
    report("cluster of 2 + distributed shared memory", ok, buf);
    // Largest cluster the device schedules (8 is portable; 16 needs the non-portable attribute).
    CK(cudaFuncSetAttribute(cluster_any, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    int best = 0;
    for (int cs : {1, 2, 4, 8, 16}) {
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(cs * 2);
      cfg.blockDim = dim3(128);
      cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = cs;
      at[0].val.clusterDim.y = 1;
      at[0].val.clusterDim.z = 1;
      cfg.attrs = at;
      cfg.numAttrs = 1;
      int nclusters = 0;
      if (cudaOccupancyMaxActiveClusters(&nclusters, cluster_any, &cfg) != cudaSuccess) { cudaGetLastError(); break; }
      if (cudaLaunchKernelEx(&cfg, cluster_any, o) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        break;
      }
      best = cs;
      printf("    cluster size %2d: up to %d active clusters\n", cs, nclusters);
    }
    snprintf(buf, sizeof buf, "largest launched cluster: %d blocks", best);
    report("cluster sizes", best >= 2, buf);
  }

  {  // PDL
    unsigned long long* t;
    int *flag, *seen;
    CK(cudaMalloc(&t, 4 * 8));
    CK(cudaMalloc(&flag, 4));
    CK(cudaMalloc(&seen, 4));
    CK(cudaMemset(flag, 0, 4));
    CK(cudaMemset(seen, 0, 4));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    // The first launch of a kernel pair is cold (module load) and never overlaps; time the second.
    for (int rep = 0; rep < 2; ++rep) {
      primary<<<p.multiProcessorCount / 2, 128, 0, s>>>(t, flag);
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(p.multiProcessorCount / 2);
      cfg.blockDim = dim3(128);
      cfg.stream = s;
      cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      at[0].val.programmaticStreamSerializationAllowed = 1;
      cfg.attrs = at;
      cfg.numAttrs = 1;
      CK(cudaLaunchKernelEx(&cfg, secondary, t, (const volatile int*)flag, seen));
      e = cudaStreamSynchronize(s);
    }
    unsigned long long ht[4] = {0};
    int hs = 0;
    if (e == cudaSuccess) {
      CK(cudaMemcpy(ht, t, sizeof ht, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&hs, seen, 4, cudaMemcpyDeviceToHost));
    }
    long long early = (long long)ht[1] - (long long)ht[2];  // >0: secondary started before primary ended
    // The primary triggers at its start, so the dependency is released early by design; what
    // matters is that the secondary's prologue overlapped the primary's body.
    ok = e == cudaSuccess && early > 0;
    snprintf(buf, sizeof buf, "secondary prologue started %.1f us before the primary ended (flag seen: %d)",
             early / 1e3, hs);
    report("programmatic dependent launch", ok, buf);
  }
  printf("%d failure(s)\n", failures);
  return failures;
}
