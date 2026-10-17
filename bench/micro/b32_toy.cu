// B32 (megakernel stage 1, specs/research/megakernel-debuggable-2026-10-14.md §6): the FFN of one layer at M = 16 rows
// (gate|up NVFP4 GEMMs, SiLU-multiply, the FP16 copy of the result, the down GEMM) as one persistent kernel with
// dependency counters, against the same four kernels as a CUDA graph with programmatic dependent launch. Every
// instruction is the engine's own kernel body (stream_gemm_body, to_half_rows_body; the SiLU-multiply's expression
// is kernels.cu's), with the same tile decomposition, so the two must agree bit for bit.
//
// From day one, as the note asks: a host validator of the schedule (every wait names a counter whose producers
// exist and reach the target exactly), a deadline on every wait (a stuck counter writes a diagnostic record and the
// kernel exits; it never hangs the GPU of a unified-memory host), and a timings store (per item: SM, start and end
// in globaltimer nanoseconds) for the per-SM timeline.
//
//   nvcc -std=c++20 -O3 -gencode arch=compute_121a,code=sm_121a -Isrc bench/micro/b32_toy.cu build/libling_core.a \
//        -o build/b32-toy && build/b32-toy [--timings out.csv]
//
// Coherence: the cross-block inputs (g and u for the row step, the FP16 copy for the down tiles) are written inside
// the kernel and read by each consumer exactly once, after its counter, with no earlier read of those addresses on
// that SM, so the read-only path the bodies use has no stale line to return.
#include "core/stream_gemm.cu"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ling::kernels {
namespace {

constexpr int kM = 16, kH = 5120, kI = 17408, kTiles = 2, kNT = 2, kPF = 2;

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release_add(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

struct Diag {
  int failed;  // 1: a wait passed its deadline
  int block, item, counter;
  unsigned observed, expected;
};

struct ToyArgs {
  const __half* xh;  // the FFN's input, [M][H] FP16 with per-row scales xinv
  const float* xinv;
  StreamProblems up;  // gate (matrix 0) and up (matrix 1), N = I each
  int n_up;           // the gate|up launch's blocks
  float *g, *u;       // their outputs [M][I]; g becomes silu(g) * u
  __half* xh2;        // the down projection's input, [M][I], and its row scales
  float* xinv2;
  StreamProblems down;
  int n_down;
  unsigned* counters;  // [0]: gate|up blocks done; [1]: rows done, or (distributed) blocks past the SiLU pass;
                       // [2]: (distributed) blocks past the FP16 pass; then kM row maxima (all zeroed before each launch)
  int distributed;     // the row step over every block (SiLU pass, row maxima, FP16 pass) instead of one block per row
  long long deadline_ns;
  Diag* diag;
  unsigned long long* timings;  // per item: smid, start, end (3 values), or nullptr
};

// Thread 0 waits until *c >= target (acquire), with the deadline; then the block goes on. False: the deadline passed
// (or another block gave up), the record is written, the caller returns.
__device__ bool wait_counter(const ToyArgs& a, int which, unsigned target, int item) {
  __shared__ int ok;
  if (threadIdx.x == 0) {
    const unsigned long long t0 = now_ns();
    unsigned v;
    ok = 1;
    while ((v = ld_acquire(&a.counters[which])) < target) {
      if (*reinterpret_cast<volatile int*>(&a.diag->failed)) {
        ok = 0;
        break;
      }
      if (now_ns() - t0 > static_cast<unsigned long long>(a.deadline_ns)) {
        if (atomicCAS(&a.diag->failed, 0, 1) == 0) {
          a.diag->block = blockIdx.x;
          a.diag->item = item;
          a.diag->counter = which;
          a.diag->observed = v;
          a.diag->expected = target;
        }
        ok = 0;
        break;
      }
      __nanosleep(64);
    }
  }
  __syncthreads();
  return ok;
}

__device__ __forceinline__ void done(const ToyArgs& a, int which) {
  __syncthreads();  // every thread's writes of this item are issued
  if (threadIdx.x == 0) red_release_add(&a.counters[which], 1u);
}

__device__ __forceinline__ void stamp(const ToyArgs& a, int item, unsigned long long t0) {
  if (a.timings && threadIdx.x == 0) {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    a.timings[3 * item] = s;
    a.timings[3 * item + 1] = t0;
    a.timings[3 * item + 2] = now_ns();
  }
}

struct NoWait {
  __device__ __forceinline__ void operator()() const {}
};
struct CounterWait {  // the down tiles: their weight prefetch is issued before this, as under PDL
  const ToyArgs* a;
  int which;
  unsigned target;
  int item;
  bool* ok;
  __device__ __forceinline__ void operator()() const { *ok = wait_counter(*a, which, target, item); }
};

// The schedule: block b takes gate|up blocks b, b + G, ...; then rows b, b + G, ... (after every gate|up block); then
// down blocks b, b + G, ... (after every row). Items are numbered up, rows, down for the timings store.
__global__ void __launch_bounds__(kWarps * 32) ffn_toy_kernel(ToyArgs a) {
  const int G = gridDim.x;
  for (int i = blockIdx.x; i < a.n_up; i += G) {
    __syncthreads();
    const unsigned long long t0 = now_ns();
    stream_gemm_body<kNT, kTiles, true, kPF, false>(a.xh, a.xinv, kM, a.up, kH, 1, nullptr, i, 0, NoWait{});
    done(a, 0);
    stamp(a, i, t0);
  }
  if (a.distributed) {
    // The row step over every block, the same arithmetic: the SiLU-multiply element by element (silu_mul's
    // expression), each row's maximum of |value| (to_half_rows' fmaxf, NaN ignored as fmaxf does; a maximum does not
    // depend on its order), then each element to FP16 under the row's power-of-two scale. Each thread converts the
    // elements it wrote itself, so nothing crosses blocks but the 16 maxima.
    const size_t n = size_t(kM) * kI, stride = size_t(G) * blockDim.x;
    unsigned* rowmax = a.counters + 3;
    if (!wait_counter(a, 0, a.n_up, a.n_up + blockIdx.x)) return;
    unsigned long long t0 = now_ns();
    __shared__ unsigned smax[kM];
    if (threadIdx.x < kM) smax[threadIdx.x] = 0u;
    __syncthreads();
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
      const float x = a.g[i];
      const float v = x / (1.f + __expf(-x)) * a.u[i];
      a.g[i] = v;
      const float m = fabsf(v);
      if (!isnan(m)) atomicMax(&smax[i / kI], __float_as_uint(m));
    }
    __syncthreads();
    if (threadIdx.x < kM && smax[threadIdx.x]) atomicMax(&rowmax[threadIdx.x], smax[threadIdx.x]);
    done(a, 1);
    stamp(a, a.n_up + blockIdx.x, t0);
    if (!wait_counter(a, 1, G, a.n_up + G + blockIdx.x)) return;
    t0 = now_ns();
    __shared__ float sc[kM];
    if (threadIdx.x < kM) {
      const float mx = __uint_as_float(ld_acquire(&rowmax[threadIdx.x]));
      int e = 0;
      if (mx > 0.f && isfinite(mx)) frexpf(mx, &e);  // as to_half_rows
      sc[threadIdx.x] = ldexpf(1.f, 15 - e);
      if (blockIdx.x == 0) a.xinv2[threadIdx.x] = ldexpf(1.f, e - 15);
    }
    __syncthreads();
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
      a.xh2[i] = __float2half_rn(a.g[i] * sc[i / kI]);
    done(a, 2);
    stamp(a, a.n_up + G + blockIdx.x, t0);
  }
  for (int r = blockIdx.x; r < kM && !a.distributed; r += G) {
    const int item = a.n_up + r;
    if (!wait_counter(a, 0, a.n_up, item)) return;
    const unsigned long long t0 = now_ns();
    // kernels.cu's silu_mul over the row (the same expression: silu(x) = x / (1 + __expf(-x))).
    float* gr = a.g + static_cast<size_t>(r) * kI;
    const float* ur = a.u + static_cast<size_t>(r) * kI;
    for (int k = threadIdx.x; k < kI; k += blockDim.x) {
      const float x = gr[k];
      gr[k] = x / (1.f + __expf(-x)) * ur[k];
    }
    __syncthreads();
    to_half_rows_body(a.g, kI, a.xh2, a.xinv2, r, NoWait{});
    done(a, 1);
    stamp(a, item, t0);
  }
  for (int i = blockIdx.x; i < a.n_down; i += G) {
    const int item = a.n_up + (a.distributed ? 2 * G : kM) + i;
    __syncthreads();
    bool ok = true;
    const unsigned long long t0 = now_ns();
    const CounterWait w = a.distributed ? CounterWait{&a, 2, static_cast<unsigned>(G), item, &ok}
                                        : CounterWait{&a, 1, static_cast<unsigned>(kM), item, &ok};
    stream_gemm_body<kNT, kTiles, true, kPF, false>(a.xh2, a.xinv2, kM, a.down, kI, 1, nullptr, i, 0, w);
    // A failed wait returns before the body's reads only in the record: the body ran on, so its output is garbage
    // and the host reports the record instead of the result.
    if (!ok) return;
    stamp(a, item, t0);
  }
}

// The host validator: the counters' targets equal their producers' counts, every wait's counter has producers that
// run before any consumer can be blocked behind them (phase order within a block, all blocks resident), and the
// schedule covers every item once.
std::string validate(int G, int n_up, int n_down, int resident) {
  if (G > resident) return "more blocks than can be resident at once: a waiting block could hold an SM a producer needs";
  std::vector<int> seen(n_up + kM + n_down, 0);
  int producers0 = 0, producers1 = 0;
  for (int b = 0; b < G; ++b) {
    for (int i = b; i < n_up; i += G) ++seen[i], ++producers0;
    for (int r = b; r < kM; r += G) ++seen[n_up + r], ++producers1;
    for (int i = b; i < n_down; i += G) ++seen[n_up + kM + i];
  }
  for (size_t i = 0; i < seen.size(); ++i)
    if (seen[i] != 1) return "item " + std::to_string(i) + " scheduled " + std::to_string(seen[i]) + " times";
  if (producers0 != n_up) return "counter 0: " + std::to_string(producers0) + " producers for target " + std::to_string(n_up);
  if (producers1 != kM) return "counter 1: " + std::to_string(producers1) + " producers for target " + std::to_string(kM);
  return "";
}

}  // namespace
}  // namespace ling::kernels

using namespace ling::kernels;

template <typename F>
double avg_ms(F&& f, int iters, cudaStream_t s) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 5; ++i) f();
  cudaStreamSynchronize(s);
  cudaEventRecord(a, s);
  for (int i = 0; i < iters; ++i) f();
  cudaEventRecord(b, s);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  return ms / iters;
}

int main(int argc, char** argv) {
  const char* timings_out = nullptr;
  int distributed = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--timings") == 0 && i + 1 < argc) timings_out = argv[++i];
    if (std::strcmp(argv[i], "--distributed-rows") == 0) distributed = 1;
  }
  std::mt19937 rng(5);
  std::uniform_int_distribution<int> byte(0, 255), sbyte(0x30, 0x40);
  std::normal_distribution<float> normal(0.f, 1.f);
  // Weights in the tiled NVFP4 layout (random values, modest positive E4M3 scales).
  auto weights = [&](int N, int K) {
    std::vector<uint8_t> h(size_t(N / 16) * (K / kChunk) * kTileBytesFp4);
    for (size_t blk = 0; blk < h.size() / kTileBytesFp4; ++blk) {
      uint8_t* p = h.data() + blk * kTileBytesFp4;
      for (int i = 0; i < 1024; ++i) p[i] = byte(rng);
      for (int i = 1024; i < kTileBytesFp4; ++i) p[i] = sbyte(rng);
    }
    uint8_t* d;
    cudaMalloc(&d, h.size());
    cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
    return d;
  };
  uint8_t *wg = weights(kI, kH), *wu = weights(kI, kH), *wd = weights(kH, kI);
  const float sg = 0.0021f, su = 0.0017f, sd = 0.0009f;
  std::vector<__half> hx(size_t(kM) * kH);
  for (auto& v : hx) v = __float2half(normal(rng) * 8192.f);
  std::vector<float> hinv(kM, 1.f / 8192.f);
  __half *xh, *xh2;
  float *xinv, *xinv2, *g, *u, *y;
  cudaMalloc(&xh, hx.size() * 2);
  cudaMalloc(&xh2, size_t(kM) * kI * 2);
  cudaMalloc(&xinv, kM * 4);
  cudaMalloc(&xinv2, kM * 4);
  cudaMalloc(&g, size_t(kM) * kI * 4);
  cudaMalloc(&u, size_t(kM) * kI * 4);
  cudaMalloc(&y, size_t(kM) * kH * 4);
  cudaMemcpy(xh, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(xinv, hinv.data(), kM * 4, cudaMemcpyHostToDevice);

  cudaStream_t s;
  cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  set_pdl(true);
  // The reference: the four kernels, as the engine's rows path runs the FFN, captured as a graph with PDL.
  const StreamTarget up_t[2] = {{wg, nullptr, sg, g, kI}, {wu, nullptr, su, u, kI}};
  auto four = [&] {
    stream_gemm_multi(true, xh, xinv, kM, up_t, 2, kH, s);
    silu_mul(g, u, g, kM * kI, s);
    to_half_rows(g, kM, kI, xh2, xinv2, s);
    stream_gemm_nvfp4(xh2, xinv2, kM, wd, nullptr, sd, y, kH, kI, s);
  };
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  four();
  cudaStreamEndCapture(s, &graph);
  if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
    std::printf("graph instantiate failed\n");
    return 1;
  }
  cudaGraphLaunch(exec, s);
  cudaStreamSynchronize(s);
  std::vector<float> y_ref(size_t(kM) * kH), g_ref(size_t(kM) * kI);
  cudaMemcpy(y_ref.data(), y, y_ref.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(g_ref.data(), g, g_ref.size() * 4, cudaMemcpyDeviceToHost);

  // The persistent kernel.
  ToyArgs a{};
  a.xh = xh;
  a.xinv = xinv;
  a.up.w[0] = wg;
  a.up.w[1] = wu;
  a.up.gscale[0] = sg;
  a.up.gscale[1] = su;
  a.up.y[0] = g;
  a.up.y[1] = u;
  a.up.N[0] = a.up.N[1] = kI;
  a.up.block0[0] = 0;
  a.up.block0[1] = kI / (kTiles * 16);
  a.up.count = 2;
  a.n_up = 2 * kI / (kTiles * 16);
  a.g = g;
  a.u = u;
  a.xh2 = xh2;
  a.xinv2 = xinv2;
  a.down.w[0] = wd;
  a.down.gscale[0] = sd;
  a.down.y[0] = y;
  a.down.N[0] = kH;
  a.down.block0[0] = 0;
  a.down.count = 1;
  a.n_down = kH / (kTiles * 16);
  cudaMalloc(&a.counters, (3 + kM) * sizeof(unsigned));
  a.distributed = distributed;
  cudaMalloc(&a.diag, sizeof(Diag));
  a.deadline_ns = 50'000'000;  // 50 ms
  int per_sm = 0, sms = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ffn_toy_kernel, kWarps * 32, 0);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  const int resident = per_sm * sms;
  std::printf("persistent kernel: %d blocks per SM x %d SMs = %d resident; gate|up %d blocks, rows %d, down %d\n", per_sm,
              sms, resident, a.n_up, kM, a.n_down);
  bool all_same = true;
  for (int G : {sms, 2 * sms, resident}) {
    if (G > resident) continue;
    const std::string bad = distributed ? (G > resident ? "too many blocks" : "") : validate(G, a.n_up, a.n_down, resident);
    if (!bad.empty()) {
      std::printf("G=%d: schedule REJECTED: %s\n", G, bad.c_str());
      continue;
    }
    auto toy = [&] {
      cudaMemsetAsync(a.counters, 0, (3 + kM) * sizeof(unsigned), s);
      ffn_toy_kernel<<<G, kWarps * 32, 0, s>>>(a);
    };
    cudaMemset(a.diag, 0, sizeof(Diag));
    cudaMemset(y, 0xff, y_ref.size() * 4);
    toy();
    const cudaError_t e = cudaStreamSynchronize(s);
    Diag d;
    cudaMemcpy(&d, a.diag, sizeof d, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess || d.failed) {
      std::printf("G=%d: %s; diag failed=%d block %d item %d counter %d observed %u expected %u\n", G,
                  cudaGetErrorString(e), d.failed, d.block, d.item, d.counter, d.observed, d.expected);
      all_same = false;
      continue;
    }
    std::vector<float> y_mk(y_ref.size()), g_mk(g_ref.size());
    cudaMemcpy(y_mk.data(), y, y_mk.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_mk.data(), g, g_mk.size() * 4, cudaMemcpyDeviceToHost);
    const bool same = std::memcmp(y_mk.data(), y_ref.data(), y_ref.size() * 4) == 0 &&
                      std::memcmp(g_mk.data(), g_ref.data(), g_ref.size() * 4) == 0;
    all_same &= same;
    // Timing: the graph and the persistent kernel (with its counter reset) alternated.
    double tg = 0, tm = 0;
    for (int rep = 0; rep < 3; ++rep) {
      tg += avg_ms([&] { cudaGraphLaunch(exec, s); }, 200, s);
      tm += avg_ms(toy, 200, s);
    }
    std::printf("G=%-4d bit-identical %s; graph of four kernels (PDL) %.3f ms, persistent kernel %.3f ms (x%.3f)\n", G,
                same ? "yes" : "NO", tg / 3, tm / 3, tm / tg);
    if (timings_out && G == resident) {
      const int rows_items = distributed ? 2 * G : kM;
      const int items = a.n_up + rows_items + a.n_down;
      unsigned long long* tdev;
      cudaMalloc(&tdev, size_t(items) * 3 * 8);
      a.timings = tdev;
      toy();
      cudaStreamSynchronize(s);
      std::vector<unsigned long long> th(size_t(items) * 3);
      cudaMemcpy(th.data(), tdev, th.size() * 8, cudaMemcpyDeviceToHost);
      a.timings = nullptr;
      unsigned long long t0 = ~0ull, up_end = 0, rows_beg = ~0ull, rows_end = 0, down_beg = ~0ull, down_end = 0;
      for (int i = 0; i < items; ++i) {
        t0 = std::min(t0, th[3 * i + 1]);
        if (i < a.n_up) up_end = std::max(up_end, th[3 * i + 2]);
        else if (i < a.n_up + rows_items) rows_beg = std::min(rows_beg, th[3 * i + 1]), rows_end = std::max(rows_end, th[3 * i + 2]);
        else down_beg = std::min(down_beg, th[3 * i + 1]), down_end = std::max(down_end, th[3 * i + 2]);
      }
      std::printf("timeline (us from the first item): gate|up done %.1f; rows %.1f-%.1f; down %.1f-%.1f\n",
                  (up_end - t0) / 1e3, (rows_beg - t0) / 1e3, (rows_end - t0) / 1e3, (down_beg - t0) / 1e3,
                  (down_end - t0) / 1e3);
      FILE* f = std::fopen(timings_out, "w");
      if (f) {
        std::fprintf(f, "item,phase,sm,start_ns,end_ns\n");
        for (int i = 0; i < items; ++i)
          std::fprintf(f, "%d,%s,%llu,%llu,%llu\n", i, i < a.n_up ? "up" : i < a.n_up + rows_items ? "row" : "down", th[3 * i],
                       th[3 * i + 1] - t0, th[3 * i + 2] - t0);
        std::fclose(f);
      }
    }
  }
  std::printf(all_same ? "B32 TOY: bit-identical to the graph path\n" : "B32 TOY: DIFFERS or FAILED\n");
  return all_same ? 0 : 1;
}
