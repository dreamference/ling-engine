// BACKLOG §19 item 29: memory layout and address translation on GB10, a measurement (no engine change).
//
//   1  what the driver reports: allocation granularity, pageable access through the host page tables
//   2  address translation, by timing (Nsight Compute exposes no TLB counters here): a dependent pointer chase at a
//      page-sized stride over growing footprints (where the latency steps is the TLB's reach, which gives the page
//      size), and the same bytes streamed in an order that touches a new page on every access, against in order
//   3  a pure streaming read of the same size on each kind of allocation: cudaMalloc, cudaMallocManaged,
//      cudaHostAlloc, and malloc'd pageable memory on 4 KB pages and on 2 MB transparent huge pages
//   4  NVFP4 weights with each block's scales interleaved after its codes (today's blob) against the codes and the
//      scales in two separate streams, the same bytes, in a copy of stream_gemm's NVFP4 path, at 1, 12 and 24 rows
//   5  attention's KV read as the engine lays it out (head-major, [Hkv][max_context][D] per layer), at 25,700 and
//      65,536 tokens
//
// Parts 2 carries two controls for power-of-two aliasing: the scattered read at each stride plus 512 B, and the chase
// through L2 only (ld.global.cg) on the page's first line and on a pseudo-random line of each page.
//
// Usage: layout_bench [--gb N] [--parts 12345]; LAYOUT_KIND=0-4 runs parts 2 and 3 on one kind of allocation only
// (one process per kind, so a fault on one cannot end the others)
#include "core/kernels.cuh"
#include "core/launch.cuh"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace k = ling::kernels;

#define CK(x)                                                                                          \
  do {                                                                                                 \
    cudaError_t e_ = (x);                                                                              \
    if (e_ != cudaSuccess) {                                                                           \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
      std::exit(2);                                                                                    \
    }                                                                                                  \
  } while (0)

namespace {

__device__ unsigned long long g_sink;

// In-order stream: each warp reads 4 KB runs, eight 512-byte warp loads in flight per lane (as stream_gemm keeps a
// few chunks in flight); the runs are dealt to the warps in order, so the GPU sweeps the buffer front to back.
__device__ __forceinline__ uint4 ld16(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__global__ void read_stream(const uint8_t* __restrict__ p, size_t bytes) {
  const int lane = threadIdx.x & 31;
  const size_t warps = size_t(gridDim.x) * blockDim.x / 32, runs = bytes / 4096;
  uint32_t acc = 0;
  for (size_t r = (blockIdx.x * size_t(blockDim.x) + threadIdx.x) / 32; r < runs; r += warps) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = ld16(p + r * 4096 + j * 512 + lane * 16);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x12345679u) g_sink = acc;
}

// The same bytes and the same eight loads in flight, but each of a warp's loads and consecutive warps' runs land
// `stride` bytes apart, so every load in flight is on a different page: load a reads page (a % pages), 512-byte
// piece (a / pages) of it.
__global__ void read_scattered(const uint8_t* __restrict__ p, size_t bytes, size_t stride) {
  // stride need not be a power of two (the controls add 512 B): pages * (stride / 512) pieces, all inside `bytes`.
  const size_t pages = bytes / stride, accesses = pages * (stride / 512);
  const int lane = threadIdx.x & 31;
  uint32_t acc = 0;
  const size_t warps = size_t(gridDim.x) * blockDim.x / 32;
  for (size_t r = (blockIdx.x * size_t(blockDim.x) + threadIdx.x) / 32; r * 8 < accesses; r += warps) {
    uint4 v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const size_t a = min(r * 8 + j, accesses - 1);
      v[j] = ld16(p + (a % pages) * stride + (a / pages) * 512 + lane * 16);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
  }
  if (acc == 0x12345679u) g_sink = acc;
}

// One thread, dependent loads: next = p[cur]; the chain visits `hops` offsets `stride` apart in a shuffled order.
__global__ void chase(const uint64_t* __restrict__ p, size_t start, int warm, int hops, uint64_t* out, long long* cycles) {
  size_t cur = start;
  for (int i = 0; i < warm; ++i) cur = __ldcg(p + cur);  // one lap first: every line of the chain in L2
  const long long t0 = clock64();
  for (int i = 0; i < hops; ++i) cur = __ldcg(p + cur);  // L2 only: an L1 hit would hide the steps
  const long long t1 = clock64();
  *out = cur;
  *cycles = t1 - t0;
}

__global__ void fill_kernel(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
    p[i] = h & 0x77777777u;
  }
}

int g_sms = 0;

double time_ms(const std::function<void()>& f, int reps = 5) {
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  f();
  CK(cudaDeviceSynchronize());
  std::vector<float> v;
  for (int r = 0; r < reps; ++r) {
    cudaEventRecord(a);
    f();
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  cudaEventDestroy(a), cudaEventDestroy(b);
  return v[v.size() / 2];
}

double stream_gbs(const uint8_t* p, size_t bytes) {
  double best = 1e30;
  for (int per_sm : {2, 4, 8}) {  // blocks of 256 per SM: the best of three occupancies
    best = std::min(best, time_ms([&] { read_stream<<<g_sms * per_sm, 256>>>(p, bytes); }));
    CK(cudaGetLastError());
  }
  return bytes / (best * 1e6);
}

double scattered_gbs(const uint8_t* p, size_t bytes, size_t stride) {
  double best = 1e30;
  for (int per_sm : {2, 4, 8}) {
    best = std::min(best, time_ms([&] { read_scattered<<<g_sms * per_sm, 256>>>(p, bytes, stride); }));
    CK(cudaGetLastError());
  }
  return double(bytes / stride * (stride / 512) * 512) / (best * 1e6);
}

// ---- part 1 ----
void part1() {
  int dev = 0;
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  int pma = 0, pmht = 0, cma = 0, dmma = 0, l2 = 0;
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, dev);
  cudaDeviceGetAttribute(&pmht, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev);
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, dev);
  cudaDeviceGetAttribute(&dmma, cudaDevAttrDirectManagedMemAccessFromHost, dev);
  cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev);
  std::printf("[1] %s, %d SMs, L2 %d MB, integrated %d; pageable access %d (through the host page tables %d), "
              "concurrent managed %d, host direct managed access %d\n",
              prop.name, prop.multiProcessorCount, l2 >> 20, prop.integrated, pma, pmht, cma, dmma);
  CUmemAllocationProp ap = {};
  ap.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  ap.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  ap.location.id = 0;
  size_t gmin = 0, grec = 0;
  cuMemGetAllocationGranularity(&gmin, &ap, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  cuMemGetAllocationGranularity(&grec, &ap, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  std::printf("[1] cuMemCreate granularity on the device: minimum %zu KB, recommended %zu KB\n", gmin >> 10, grec >> 10);
  // cudaMalloc's alignment: the low bits of a run of small allocations' addresses.
  std::vector<void*> ps;
  for (int i = 0; i < 8; ++i) {
    void* p;
    CK(cudaMalloc(&p, size_t(3) << 20));  // 3 MB each
    ps.push_back(p);
  }
  size_t align = size_t(1) << 40;
  for (void* p : ps) align = std::min(align, reinterpret_cast<size_t>(p) & (~reinterpret_cast<size_t>(p) + 1));
  std::printf("[1] eight 3 MB cudaMallocs: addresses aligned to at least %zu KB, first %p, step %zd KB\n", align >> 10,
              ps[0], (reinterpret_cast<char*>(ps[1]) - reinterpret_cast<char*>(ps[0])) >> 10);
  for (void* p : ps) cudaFree(p);
}

// ---- allocations for parts 2 and 3 ----
struct Alloc {
  std::string name;
  uint8_t* p = nullptr;
  int kind = 0;  // 0 cudaMalloc, 1 managed, 2 cudaHostAlloc, 3 malloc 4 KB, 4 malloc THP
};

bool make(Alloc& a, size_t bytes) {
  switch (a.kind) {
    case 0:
      if (cudaMalloc(&a.p, bytes) != cudaSuccess) return false;
      break;
    case 1:
      if (cudaMallocManaged(&a.p, bytes) != cudaSuccess) return false;
      cudaMemAdvise(a.p, bytes, cudaMemAdviseSetPreferredLocation, cudaMemLocation{cudaMemLocationTypeDevice, 0});
      cudaMemPrefetchAsync(a.p, bytes, cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0, nullptr);
      break;
    case 2:
      if (cudaHostAlloc(reinterpret_cast<void**>(&a.p), bytes, cudaHostAllocDefault) != cudaSuccess) return false;
      break;
    case 3:
    case 4: {
      void* p = nullptr;
      if (posix_memalign(&p, size_t(2) << 20, bytes) != 0) return false;
      madvise(p, bytes, a.kind == 4 ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
      a.p = static_cast<uint8_t*>(p);
      break;
    }
  }
  if (a.kind >= 2) {  // first touch on the CPU, then the pattern from the GPU
    std::memset(a.p, 1, bytes);
  }
  fill_kernel<<<1024, 256>>>(reinterpret_cast<uint32_t*>(a.p), bytes / 4, 7);
  return cudaDeviceSynchronize() == cudaSuccess;
}

void release(Alloc& a) {
  if (!a.p) return;
  if (a.kind <= 1) cudaFree(a.p);
  else if (a.kind == 2) cudaFreeHost(a.p);
  else std::free(a.p);
  a.p = nullptr;
}

size_t huge_kb(const void* p) {  // AnonHugePages of the mapping holding p, from /proc/self/smaps
  FILE* f = std::fopen("/proc/self/smaps", "r");
  if (!f) return 0;
  char line[512];
  bool in = false;
  size_t kb = 0;
  const uintptr_t a = reinterpret_cast<uintptr_t>(p);
  while (std::fgets(line, sizeof line, f)) {
    uintptr_t lo, hi;
    if (std::sscanf(line, "%lx-%lx", &lo, &hi) == 2 && std::strchr(line, '-') < std::strchr(line, ' ')) {
      in = a >= lo && a < hi;
      continue;
    }
    if (in && std::strncmp(line, "AnonHugePages:", 14) == 0) kb = std::strtoull(line + 14, nullptr, 10);
  }
  std::fclose(f);
  return kb;
}

void chase_curve(const Alloc& a, size_t bytes, size_t stride, bool jitter) {
  // Footprints from 16 MB to the whole buffer; the chain visits every stride-th 8-byte slot once, shuffled.
  uint64_t* out;
  long long* cyc;
  CK(cudaMalloc(&out, 8));
  CK(cudaMalloc(&cyc, 8));
  std::printf("[2] %-22s chase, stride %5zu KB%s:", a.name.c_str(), stride >> 10,
              jitter ? ", a random 128-byte line of each page" : ", the page's first line");
  std::mt19937_64 rng(5);
  for (size_t fp = size_t(16) << 20; fp <= bytes; fp *= 2) {
    const size_t n = fp / stride;
    std::vector<uint64_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::shuffle(order.begin() + 1, order.end(), rng);
    std::vector<uint64_t> next(n);
    // Slot i: the page's first 8 bytes, or (jitter) a pseudo-random 128-byte line of the page, so the chain's lines
    // do not all fall in the same cache sets and DRAM banks (a power-of-two stride aliases; a page does not).
    auto slot = [&](size_t i) -> size_t {
      size_t line = 0;
      if (jitter) {
        uint64_t h = (i + 1) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
        line = h % (stride / 128);
      }
      return i * (stride / 8) + line * 16;
    };
    for (size_t i = 0; i < n; ++i) next[order[i]] = slot(order[(i + 1) % n]);
    for (size_t i = 0; i < n; ++i)
      CK(cudaMemcpy(reinterpret_cast<uint64_t*>(a.p) + slot(i), &next[i], 8, cudaMemcpyDefault));
    const int hops = static_cast<int>(std::max<size_t>(n, 4096));
    chase<<<1, 1>>>(reinterpret_cast<const uint64_t*>(a.p), slot(0), static_cast<int>(n), hops, out, cyc);
    CK(cudaDeviceSynchronize());
    long long c = 0;
    CK(cudaMemcpy(&c, cyc, 8, cudaMemcpyDeviceToHost));
    std::printf(" %zuM:%lld", fp >> 20, c / hops);
    if (n > (size_t(1) << 16)) break;  // the chain writes are host copies; enough points
  }
  std::printf("  (cycles per load, the chain's lines in L2: the steps are address translation)\n");
  cudaFree(out), cudaFree(cyc);
}

void part23(size_t bytes, bool do2, bool do3) {
  const char* names[] = {"cudaMalloc", "cudaMallocManaged", "cudaHostAlloc", "malloc, 4 KB pages", "malloc, 2 MB THP"};
  const char* only = std::getenv("LAYOUT_KIND");  // one kind per process: a fault on one cannot end the others
  for (int kind = 0; kind < 5; ++kind) {
    if (only && std::atoi(only) != kind) continue;
    Alloc a{names[kind], nullptr, kind};
    if (!make(a, bytes)) {
      std::printf("[3] %-22s %.1f GB: allocation or first access failed (%s)\n", a.name.c_str(), bytes / 1e9,
                  cudaGetErrorString(cudaGetLastError()));
      release(a);
      continue;
    }
    if (kind >= 3) std::printf("[1] %-22s AnonHugePages %zu MB of %zu MB\n", a.name.c_str(), huge_kb(a.p) >> 10, bytes >> 20);
    if (do3) {
      const double g = stream_gbs(a.p, bytes);
      std::printf("[3] %-22s %.1f GB in-order stream %6.1f GB/s\n", a.name.c_str(), bytes / 1e9, g);
    }
    if (do2) {
      std::printf("[2] %-22s same bytes, a new page per warp access:", a.name.c_str());
      for (size_t stride : {size_t(4) << 10, size_t(64) << 10, size_t(2) << 20})
        std::printf("  %zu KB apart %6.1f GB/s", stride >> 10, scattered_gbs(a.p, bytes, stride));
      std::printf("\n[2] %-22s control, the strides plus 512 B (no power-of-two aliasing):", a.name.c_str());
      for (size_t stride : {size_t(4) << 10, size_t(64) << 10, size_t(2) << 20})
        std::printf("  %zu KB + 512 apart %6.1f GB/s", stride >> 10, scattered_gbs(a.p, bytes, stride + 512));
      std::printf("\n");
      std::fflush(stdout);
      if (kind == 0 || kind == 3 || kind == 4) {
        for (bool jitter : {false, true}) {
          chase_curve(a, std::min(bytes, size_t(8) << 30), size_t(2) << 20, jitter);
          chase_curve(a, std::min(bytes, size_t(1) << 30), size_t(64) << 10, jitter);
          chase_curve(a, std::min(bytes, size_t(256) << 20), size_t(4) << 10, jitter);
        }
      }
      std::fflush(stdout);
    }
    release(a);
  }
}

// ---- part 4: NVFP4, scales interleaved per block against a separate scale stream (a copy of stream_gemm's path) ----
constexpr int kWarps = 8, kChunk = 128;

__device__ __forceinline__ uint32_t e2m1x2_to_h2(uint32_t byte) {
  uint32_t out;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b0;\n}" : "=r"(out) : "r"(byte));
  return out;
}
__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t pair) {
  uint32_t out;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(out) : "r"(pair));
  return out;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

template <int TILES>
struct Fp4Regs {
  uint4 v[TILES][2];
  uint32_t sc[TILES][2];
};

template <int NT, int TILES, int PF, bool SPLIT>
__global__ void __launch_bounds__(kWarps * 32)
    fp4_kernel(const __half* __restrict__ x, int M, const uint8_t* __restrict__ w, float* __restrict__ y, int N, int K) {
  __shared__ float red[kWarps][TILES * 16][NT * 8 + 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * TILES * 16, chunks = K / kChunk, last_tile = N / 16 - 1;
  const size_t nblk = size_t(N / 16) * chunks;
  auto load = [&](Fp4Regs<TILES>& r, int chunk) {
#pragma unroll
    for (int i = 0; i < TILES; ++i) {
      const int tile = min((row0 >> 4) + i, last_tile);
      const size_t blk = size_t(tile) * chunks + chunk;
      const uint8_t* base = w + blk * (SPLIT ? 1024 : 1152);
      const uint8_t* sbase = SPLIT ? w + nblk * 1024 + blk * 128 : base + 1024;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int slot = (h * 8 + g) * 4 + t;
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(r.v[i][h].x), "=r"(r.v[i][h].y), "=r"(r.v[i][h].z), "=r"(r.v[i][h].w)
                     : "l"(base + slot * 16));
        r.sc[i][h] = *reinterpret_cast<const uint16_t*>(sbase + slot * 2);
      }
    }
  };
  float acc[TILES][NT][4] = {};
  Fp4Regs<TILES> buf[PF + 1];
#pragma unroll
  for (int p = 0; p < PF; ++p)
    if (warp + p * kWarps < chunks) load(buf[p], warp + p * kWarps);
  for (int c = warp; c < chunks; c += kWarps) {
    if (c + PF * kWarps < chunks) load(buf[PF], c + PF * kWarps);
    const Fp4Regs<TILES>& cur = buf[0];
    uint32_t xw[NT][16];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int n = g + 8 * j;
      if (n < M) {
        const uint4* p = reinterpret_cast<const uint4*>(x + size_t(n) * K + c * kChunk + t * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 v = __ldg(p + q);
          xw[j][4 * q] = v.x, xw[j][4 * q + 1] = v.y, xw[j][4 * q + 2] = v.z, xw[j][4 * q + 3] = v.w;
        }
      } else {
#pragma unroll
        for (int q = 0; q < 16; ++q) xw[j][q] = 0u;
      }
    }
    uint32_t sc_h2[TILES][2][2];
#pragma unroll
    for (int i = 0; i < TILES; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t s2 = e4m3x2_to_h2(cur.sc[i][h]);
        sc_h2[i][h][0] = (s2 & 0xffffu) | (s2 << 16);
        sc_h2[i][h][1] = (s2 >> 16) | (s2 & 0xffff0000u);
      }
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int i = 0; i < TILES; ++i) {
        uint32_t a[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t bytes = word_of(cur.v[i][h], s >> 1) >> (16 * (s & 1));
          const uint32_t scale = sc_h2[i][h][s >> 2];
          a[h][0] = hmul2(e2m1x2_to_h2(bytes & 0xffu), scale);
          a[h][1] = hmul2(e2m1x2_to_h2((bytes >> 8) & 0xffu), scale);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) mma16816(acc[i][j], a[0][0], a[1][0], a[0][1], a[1][1], xw[j][2 * s], xw[j][2 * s + 1]);
      }
#pragma unroll
    for (int p = 0; p < PF; ++p) buf[p] = buf[p + 1];
  }
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      red[warp][16 * i + g][8 * j + 2 * t] = acc[i][j][0];
      red[warp][16 * i + g][8 * j + 2 * t + 1] = acc[i][j][1];
      red[warp][16 * i + g + 8][8 * j + 2 * t] = acc[i][j][2];
      red[warp][16 * i + g + 8][8 * j + 2 * t + 1] = acc[i][j][3];
    }
  __syncthreads();
  for (int e = threadIdx.x; e < TILES * 16 * NT * 8; e += blockDim.x) {
    const int r = e / (NT * 8), n = e % (NT * 8);
    if (n >= M || row0 + r >= N) continue;
    float sum = 0.f;
#pragma unroll
    for (int wi = 0; wi < kWarps; ++wi) sum += red[wi][r][n];
    y[size_t(n) * N + row0 + r] = sum;
  }
}

template <int NT, bool SPLIT>
void fp4_launch(const __half* x, int M, const uint8_t* w, float* y, int N, int K) {
  fp4_kernel<NT, 2, 2, SPLIT><<<(N + 31) / 32, kWarps * 32>>>(x, M, w, y, N, K);
}

void fp4_call(bool split, const __half* x, int M, const uint8_t* w, float* y, int N, int K) {
  if (M <= 8) split ? fp4_launch<1, true>(x, M, w, y, N, K) : fp4_launch<1, false>(x, M, w, y, N, K);
  else if (M <= 16) split ? fp4_launch<2, true>(x, M, w, y, N, K) : fp4_launch<2, false>(x, M, w, y, N, K);
  else split ? fp4_launch<4, true>(x, M, w, y, N, K) : fp4_launch<4, false>(x, M, w, y, N, K);
}

// Interleaved blob -> split blob (values of every block, then every block's 128 scale bytes).
__global__ void split_kernel(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, size_t nblk) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < nblk * 1152 / 16; i += size_t(gridDim.x) * blockDim.x) {
    const size_t blk = i / 72, piece = i % 72;
    const uint4 v = reinterpret_cast<const uint4*>(in)[i];
    if (piece < 64) reinterpret_cast<uint4*>(out + blk * 1024)[piece] = v;
    else reinterpret_cast<uint4*>(out + nblk * 1024 + blk * 128)[piece - 64] = v;
  }
}

void part4() {
  struct G {
    const char* name;
    int N, K, count;
  };
  // The NVFP4 matrices of a step: the FFN (64 layers) and the LM head (twice: the target and the drafter's scoring).
  const G gs[] = {{"ffn gate|up", 2 * 17408, 5120, 64}, {"ffn down", 5120, 17408, 64}, {"lm head", 248320, 5120, 2}};
  __half* x;
  float *y1, *y2;
  CK(cudaMalloc(&x, size_t(32) * 17408 * 2));
  CK(cudaMalloc(&y1, size_t(32) * 248320 * 4));
  CK(cudaMalloc(&y2, size_t(32) * 248320 * 4));
  fill_kernel<<<256, 256>>>(reinterpret_cast<uint32_t*>(x), size_t(32) * 17408 / 2, 3);
  // Keep x finite: halves of the pattern & 0x7777 are finite.
  double step[2][3] = {};
  for (const G& g : gs) {
    const size_t nblk = size_t(g.N / 16) * (g.K / kChunk), bytes = nblk * 1152;
    const int copies = int(std::max<size_t>(2, (size_t(1) << 30) / bytes));
    uint8_t *wi, *ws;
    CK(cudaMalloc(&wi, bytes * copies));
    CK(cudaMalloc(&ws, bytes * copies));
    fill_kernel<<<1024, 256>>>(reinterpret_cast<uint32_t*>(wi), bytes * copies / 4, 9);
    for (int c = 0; c < copies; ++c) split_kernel<<<1024, 256>>>(wi + bytes * c, ws + bytes * c, nblk);
    CK(cudaDeviceSynchronize());
    std::printf("[4] %-12s %6.1f MB:", g.name, bytes / 1e6);
    const int Ms[] = {1, 12, 24};
    for (int mi = 0; mi < 3; ++mi) {
      const int M = Ms[mi];
      fp4_call(false, x, M, wi, y1, g.N, g.K);
      fp4_call(true, x, M, ws, y2, g.N, g.K);
      CK(cudaDeviceSynchronize());
      std::vector<float> a(size_t(M) * g.N), b(size_t(M) * g.N);
      CK(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost));
      const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
      double ms[2];
      for (int sp = 0; sp < 2; ++sp) {
        const int iters = std::max(20, copies * 4);
        ms[sp] = time_ms([&] {
                   for (int it = 0; it < iters; ++it)
                     fp4_call(sp == 1, x, M, (sp ? ws : wi) + bytes * (it % copies), sp ? y2 : y1, g.N, g.K);
                 }, 3) / iters;
        step[sp][mi] += ms[sp] * g.count;
      }
      std::printf("  M=%-2d interleaved %5.1f split %5.1f GB/s%s", M, bytes / (ms[0] * 1e6), bytes / (ms[1] * 1e6),
                  same ? "" : " (outputs DIFFER)");
    }
    std::printf("\n");
    cudaFree(wi), cudaFree(ws);
  }
  std::printf("[4] the step's NVFP4 GEMMs (ffn and the head twice): M=1 %.2f vs %.2f ms, M=12 %.2f vs %.2f ms, M=24 %.2f vs %.2f ms "
              "(interleaved vs split)\n",
              step[0][0], step[1][0], step[0][1], step[1][1], step[0][2], step[1][2]);
  cudaFree(x), cudaFree(y1), cudaFree(y2);
}

// ---- part 5: attention's KV read in the engine's layout ----
__global__ void fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    p[i] = __float2bfloat16((static_cast<float>(h & 0xffff) / 65536.f - 0.5f) * 2.f);
  }
}

void part5() {
  const int Hq = 24, Hkv = 4, D = 256, layers = 16, max_context = 65536 + 64;
  __nv_bfloat16 *kc, *vc;
  const size_t kv = size_t(Hkv) * max_context * D;
  CK(cudaMalloc(&kc, kv * 2));
  CK(cudaMalloc(&vc, kv * 2));
  fill_bf16<<<1024, 256>>>(kc, kv, 1);
  fill_bf16<<<1024, 256>>>(vc, kv, 2);
  float *q, *out, *scratch;
  CK(cudaMalloc(&q, size_t(32) * Hq * D * 4));
  CK(cudaMalloc(&out, size_t(32) * Hq * D * 4));
  CK(cudaMemset(q, 0, size_t(32) * Hq * D * 4));
  CK(cudaMalloc(&scratch, k::attention_rows_scratch_floats(32, Hq, D, max_context) * 4));
  k::set_kv_layout(size_t(max_context) * D, D);
  k::set_attention_prefetch(2);
  for (int ctx : {25700, 65536}) {
    for (int M : {1, 12}) {
      const double ms = time_ms([&] {
        for (int i = 0; i < 10; ++i) k::attention_rows(q, kc, vc, ctx - M, M, Hq, Hkv, D, scratch, out, nullptr);
      }) / 10;
      const double gb = 2.0 * ctx * Hkv * D * 2 / 1e9;
      std::printf("[5] attention, %5d tokens, M=%-2d: %.3f ms a layer, %.2f ms a step (16 layers, %.2f GB of KV), %.0f GB/s\n",
                  ctx, M, ms, ms * layers, gb * layers, gb / (ms * 1e-3));
    }
  }
  k::set_kv_layout(0, 0);
  k::set_attention_prefetch(0);
  cudaFree(kc), cudaFree(vc), cudaFree(q), cudaFree(out), cudaFree(scratch);
}

}  // namespace

int main(int argc, char** argv) {
  double gb = 16;
  std::string parts = "12345";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--gb")) gb = std::atof(argv[i + 1]);
    if (!std::strcmp(argv[i], "--parts")) parts = argv[i + 1];
  }
  cuInit(0);
  CK(cudaFree(nullptr));
  cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
  const size_t bytes = static_cast<size_t>(gb * (1 << 30)) & ~((size_t(2) << 20) - 1);
  if (parts.find('1') != std::string::npos) part1();
  if (parts.find('2') != std::string::npos || parts.find('3') != std::string::npos)
    part23(bytes, parts.find('2') != std::string::npos, parts.find('3') != std::string::npos);
  if (parts.find('4') != std::string::npos) part4();
  if (parts.find('5') != std::string::npos) part5();
  return 0;
}
