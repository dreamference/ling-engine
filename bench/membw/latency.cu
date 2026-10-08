// DRAM latency and how many SMs saturate the bus (M0, spec 16.1 items 1 and 2).
//
// 1. Pointer chase: one thread follows a random cyclic permutation of 128-byte lines; the mean
//    time per dependent load over buffers from 1 MB (L2) to 1 GiB (DRAM) gives the load-to-use
//    latency. Two layouts: lines spread over the whole buffer, and lines confined to 64 KB
//    windows walked in order (fewer TLB and DRAM page misses), to bound what latency a streaming
//    kernel sees.
// 2. SM scaling: the read-only streaming kernel of membw.cu launched with one or two blocks of
//    1024 threads per SM on 4..48 SMs. The bandwidth curve says how many SMs a weight stream
//    needs, and so how many are free for other work (spec 16.6/16.7).
//
// Build: nvcc -O3 -std=c++20 -gencode arch=compute_121a,code=sm_121a latency.cu -o latency
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void chase(const uint32_t* __restrict__ next, uint32_t start, long steps, uint32_t* out, long long* cycles) {
  uint32_t p = start;
  // Warm the path once so the timed loop measures steady state, not cold TLB fills only.
  for (long i = 0; i < steps / 4; ++i) p = __ldcg(next + p);
  long long t0 = clock64();
  for (long i = 0; i < steps; ++i) p = __ldcg(next + p);
  long long t1 = clock64();
  *out = p;
  *cycles = t1 - t0;
}

__global__ void stream_read(const uint4* __restrict__ p, size_t n, unsigned* out) {
  unsigned acc = 0;
  size_t stride = (size_t)gridDim.x * blockDim.x * 4;
  for (size_t base = (size_t)blockIdx.x * blockDim.x * 4 + threadIdx.x; base < n; base += stride) {
    uint4 v[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      size_t i = base + (size_t)u * blockDim.x;
      v[u] = i < n ? __ldcs(p + i) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) *out = acc;
}

// Reports which SM each block ran on, to confirm a grid of k blocks spreads over k SMs.
__global__ void which_sm(int* sm) {
  unsigned id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  if (threadIdx.x == 0) sm[blockIdx.x] = (int)id;
}

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  int clk_khz = 0;
  CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
  printf("device %s, %d SMs, reported max SM clock %.0f MHz\n", prop.name, prop.multiProcessorCount, clk_khz / 1e3);

  const size_t max_bytes = 1ull << 30;
  uint32_t* d_next;
  CK(cudaMalloc(&d_next, max_bytes));
  uint32_t* d_out;
  long long* d_cyc;
  CK(cudaMalloc(&d_out, 4));
  CK(cudaMalloc(&d_cyc, 8));
  std::mt19937_64 rng(42);
  const int line = 128 / 4;  // uint32 per line

  printf("\n== pointer chase (one thread, dependent loads, ld.cg) ==\n");
  printf("%10s %14s %14s\n", "buffer", "random ns", "64KB-window ns");
  for (size_t bytes : {1ull << 20, 8ull << 20, 16ull << 20, 64ull << 20, 256ull << 20, 1ull << 30}) {
    size_t lines = bytes / 128;
    double res[2];
    for (int mode = 0; mode < 2; ++mode) {
      std::vector<uint32_t> order(lines);
      std::iota(order.begin(), order.end(), 0);
      if (mode == 0) {
        std::shuffle(order.begin(), order.end(), rng);
      } else {  // shuffle inside 64 KB windows (512 lines), windows in order
        for (size_t w = 0; w < lines; w += 512) std::shuffle(order.begin() + w, order.begin() + std::min(lines, w + 512), rng);
      }
      std::vector<uint32_t> next(bytes / 4, 0);
      for (size_t i = 0; i < lines; ++i) next[(size_t)order[i] * line] = order[(i + 1) % lines] * line;
      CK(cudaMemcpy(d_next, next.data(), bytes, cudaMemcpyHostToDevice));
      long steps = 1 << 20;
      chase<<<1, 1>>>(d_next, order[0] * line, steps, d_out, d_cyc);
      CK(cudaDeviceSynchronize());
      long long cyc;
      CK(cudaMemcpy(&cyc, d_cyc, 8, cudaMemcpyDeviceToHost));
      // clock64 counts SM cycles; convert with the clock measured below via events.
      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      chase<<<1, 1>>>(d_next, order[0] * line, steps, d_out, d_cyc);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms;
      CK(cudaEventElapsedTime(&ms, a, b));
      res[mode] = ms * 1e6 / (steps * 1.25);  // warm-up quarter included in the event time
    }
    printf("%8zu MB %14.1f %14.1f\n", bytes >> 20, res[0], res[1]);
  }

  printf("\n== bandwidth by number of SMs (8 GiB read-only stream, best of 3) ==\n");
  size_t sbytes = 8ull << 30;
  uint4* buf;
  CK(cudaMalloc(&buf, sbytes));
  CK(cudaMemset(buf, 1, sbytes));
  size_t n = sbytes / 16;
  int* d_sm;
  CK(cudaMalloc(&d_sm, 4096 * sizeof(int)));
  for (int per_sm : {1, 2}) {
    for (int k : {4, 8, 12, 16, 20, 24, 32, 40, 48}) {
      int blocks = k * per_sm;
      which_sm<<<blocks, 1024>>>(d_sm);
      std::vector<int> sm(blocks);
      CK(cudaMemcpy(sm.data(), d_sm, blocks * sizeof(int), cudaMemcpyDeviceToHost));
      std::sort(sm.begin(), sm.end());
      int distinct = (int)(std::unique(sm.begin(), sm.end()) - sm.begin());
      double best = 0;
      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      for (int r = 0; r < 3; ++r) {
        CK(cudaEventRecord(a));
        stream_read<<<blocks, 1024>>>(buf, n, d_out);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms;
        CK(cudaEventElapsedTime(&ms, a, b));
        best = std::max(best, sbytes / (ms * 1e-3) / 1e9);
      }
      printf("SMs %2d (blocks %3d x 1024 thr, %2d distinct SMs): %6.1f GB/s\n", k, blocks, distinct, best);
    }
  }
  return 0;
}
