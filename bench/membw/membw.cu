// Weight-streaming bandwidth on GB10 (M0, deliverable 1).
//
// Measures what a decode step can get from the LPDDR5x bus: a read-only stream over a buffer far
// larger than L2, the way a weight pass reads each byte once. Also write-only and copy for
// reference, cudaMalloc against cudaMallocManaged and plain malloc (system-allocated, reached
// through the coherent fabric), and a sustained run with clocks logged to catch throttling.
//
// Build: nvcc -O3 -std=c++20 -arch=sm_121 membw.cu -o membw
// Usage: membw [GiB=8] [seconds=12] [sustained]
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

// Read-only stream: each thread reads UNROLL 16-byte vectors per iteration through the
// non-coherent read-only path (ld.global.nc), with a streaming (evict-first) hint so the
// buffer does not thrash L2. The XOR keeps the loads alive.
template <int UNROLL, bool STREAM_HINT>
__global__ void read_kernel(const uint4* __restrict__ p, size_t n, unsigned* out) {
  unsigned acc = 0;
  size_t stride = (size_t)gridDim.x * blockDim.x * UNROLL;
  for (size_t base = ((size_t)blockIdx.x * blockDim.x * UNROLL) + threadIdx.x; base < n; base += stride) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      size_t i = base + (size_t)u * blockDim.x;
      if (i < n) {
        if constexpr (STREAM_HINT) v[u] = __ldcs(p + i); else v[u] = __ldg(p + i);
      } else v[u] = make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) out[0] = acc;  // practically never taken; defeats dead-code elimination
}

__global__ void write_kernel(uint4* __restrict__ p, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    __stcs(p + i, make_uint4((unsigned)i, 1, 2, 3));
}

__global__ void copy_kernel(const uint4* __restrict__ s, uint4* __restrict__ d, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    __stcs(d + i, __ldcs(s + i));
}

__global__ void init_kernel(uint4* p, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    p[i] = make_uint4((unsigned)i * 2654435761u, (unsigned)(i >> 7), 7u, (unsigned)i);
}

static int g_sms = 0;
static unsigned* g_out = nullptr;

using Launch = void (*)(const uint4*, size_t, int, int);

template <int U, bool H>
static void launch_read(const uint4* p, size_t n, int blocks, int threads) {
  read_kernel<U, H><<<blocks, threads>>>(p, n, g_out);
}

// Best-of-reps bandwidth in GB/s (1e9 bytes) for one configuration.
static double time_read(Launch f, const uint4* p, size_t n, int blocks, int threads, int reps) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  f(p, n, blocks, threads); CK(cudaDeviceSynchronize());  // warm
  double best = 0;
  for (int r = 0; r < reps; ++r) {
    CK(cudaEventRecord(a)); f(p, n, blocks, threads); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    best = std::max(best, n * 16.0 / (ms * 1e-3) / 1e9);
  }
  CK(cudaGetLastError());
  return best;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  double gib = argc > 1 ? atof(argv[1]) : 8.0;
  double seconds = argc > 2 ? atof(argv[2]) : 12.0;
  size_t bytes = (size_t)(gib * (1ull << 30));
  size_t n = bytes / 16;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  g_sms = prop.multiProcessorCount;
  int l2 = prop.l2CacheSize;
  printf("device %s sm_%d%d SMs %d L2 %.1f MB buffer %.2f GiB\n", prop.name, prop.major, prop.minor,
         g_sms, l2 / 1e6, gib);
  CK(cudaMalloc(&g_out, 4));

  uint4* d; CK(cudaMalloc(&d, bytes));
  init_kernel<<<g_sms * 8, 512>>>(d, n); CK(cudaDeviceSynchronize());

  // 1. Sweep launch shapes for the read kernel.
  struct Cfg { const char* name; Launch f; };
  Cfg cfgs[] = {
    {"ldg  unroll1", launch_read<1, false>}, {"ldg  unroll4", launch_read<4, false>},
    {"ldg  unroll8", launch_read<8, false>}, {"ldcs unroll4", launch_read<4, true>},
    {"ldcs unroll8", launch_read<8, true>},
  };
  double best_all = 0; const char* best_name = ""; int best_b = 0, best_t = 0; Launch best_f = nullptr;
  bool sustained_only = argc > 3 && std::string(argv[3]) == "sustained";
  if (sustained_only) { best_f = launch_read<4, true>; best_b = g_sms * 32; best_t = 1024; }
  else printf("\n== read-only sweep (best of 5, GB/s) ==\n");
  for (auto& c : cfgs) {
    if (sustained_only) break;
    for (int t : {256, 512, 1024}) {
      for (int bm : {1, 2, 4, 8, 16, 32}) {
        int blocks = g_sms * bm;
        double gbs = time_read(c.f, d, n, blocks, t, 5);
        printf("%s threads %4d blocks/SM %2d : %7.1f\n", c.name, t, bm, gbs);
        if (gbs > best_all) { best_all = gbs; best_name = c.name; best_b = blocks; best_t = t; best_f = c.f; }
      }
    }
  }
  printf("BEST read: %.1f GB/s (%s, %d blocks x %d threads) = %.1f%% of 273\n", best_all, best_name, best_b,
         best_t, best_all / 273.0 * 100);

  if (sustained_only) goto sustained;
  // 2. Sizes: does the figure hold from one layer's weights (~0.3 GB) to a whole pass (17.6 GB)?
  printf("\n== read-only by size (best config, median of 9) ==\n");
  for (double s : {0.064, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}) {
    size_t nn = std::min(n, (size_t)(s * 1e9) / 16);
    std::vector<double> v;
    for (int r = 0; r < 9; ++r) v.push_back(time_read(best_f, d, nn, best_b, best_t, 1));
    printf("size %6.3f GB : %7.1f GB/s\n", nn * 16 / 1e9, median(v));
  }

  // 3. Write and copy for reference (copy counts read + write bytes).
  {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double bw = 0, bc = 0;
    for (int r = 0; r < 5; ++r) {
      CK(cudaEventRecord(a)); write_kernel<<<g_sms * 8, 512>>>(d, n); CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b));
      bw = std::max(bw, bytes / (ms * 1e-3) / 1e9);
      size_t h = n / 2;
      CK(cudaEventRecord(a)); copy_kernel<<<g_sms * 8, 512>>>(d, d + h, h); CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
      bc = std::max(bc, 2.0 * h * 16 / (ms * 1e-3) / 1e9);
    }
    printf("\n== write-only %.1f GB/s, copy (read+write) %.1f GB/s ==\n", bw, bc);
    float ms; CK(cudaEventRecord(a)); CK(cudaMemcpy(d + n / 2, d, (n / 2) * 16, cudaMemcpyDeviceToDevice));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms, a, b));
    printf("cudaMemcpy D2D (read+write) %.1f GB/s\n", 2.0 * (n / 2) * 16 / (ms * 1e-3) / 1e9);
  }

  // 4. Other allocators.
  {
    uint4* m; CK(cudaMallocManaged(&m, bytes));
    CK(cudaMemAdvise(m, bytes, cudaMemAdviseSetPreferredLocation, cudaMemLocation{cudaMemLocationTypeDevice, 0}));
    init_kernel<<<g_sms * 8, 512>>>(m, n); CK(cudaDeviceSynchronize());
    printf("\n== cudaMallocManaged read: %.1f GB/s ==\n", time_read(best_f, m, n, best_b, best_t, 5));
    CK(cudaFree(m));
    uint4* h = (uint4*)aligned_alloc(4096, bytes);
    memset(h, 1, bytes);  // first touch on the CPU: pages live wherever the kernel put them
    int pageable = 0; CK(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
    if (pageable) printf("== malloc (pageable, CPU first-touch) read: %.1f GB/s ==\n",
                         time_read(best_f, h, n, best_b, best_t, 5));
    free(h);
  }

sustained:
  // 5. Sustained: back-to-back passes for `seconds`, per-second GB/s (run nvidia-smi beside it).
  printf("\n== sustained read, %.0f s ==\n", seconds);
  {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto t0 = std::chrono::steady_clock::now();
    double sec_bytes = 0, sec_ms = 0; int sec = 0; std::vector<double> per_sec;
    while (true) {
      CK(cudaEventRecord(a)); best_f(d, n, best_b, best_t); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms, a, b)); sec_bytes += bytes; sec_ms += ms;
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (el >= sec + 1) {
        per_sec.push_back(sec_bytes / (sec_ms * 1e-3) / 1e9);
        printf("t=%2d s %7.1f GB/s\n", sec, per_sec.back()); fflush(stdout);
        sec_bytes = sec_ms = 0; ++sec;
      }
      if (el >= seconds) break;
    }
    auto mm = std::minmax_element(per_sec.begin(), per_sec.end());
    printf("SUSTAINED median %.1f min %.1f max %.1f GB/s\n", median(per_sec), *mm.first, *mm.second);
  }
  return 0;
}
