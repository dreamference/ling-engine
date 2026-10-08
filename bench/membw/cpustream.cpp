// CPU memory traffic beside a GPU stream (M0): N threads read (or copy) a large buffer for T
// seconds and report GB/s, so the GPU's achieved bandwidth can be measured while the CPU shares
// the LPDDR5x pins, as the scheduler, tokenizer and HTTP layer do in production.
//
// Build: g++ -O3 -std=c++20 -march=native -pthread cpustream.cpp -o cpustream
// Usage: cpustream threads seconds [GiB=4] [read|copy]
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 4;
  double seconds = argc > 2 ? atof(argv[2]) : 10;
  double gib = argc > 3 ? atof(argv[3]) : 4;
  std::string mode = argc > 4 ? argv[4] : "read";
  size_t bytes = (size_t)(gib * (1ull << 30));
  size_t per = bytes / threads / 64 * 64;
  std::vector<uint64_t*> bufs, dst;
  for (int t = 0; t < threads; ++t) {
    auto* b = (uint64_t*)aligned_alloc(64, per); memset(b, t + 1, per); bufs.push_back(b);
    if (mode == "copy") { auto* c = (uint64_t*)aligned_alloc(64, per); memset(c, 0, per); dst.push_back(c); }
  }
  std::atomic<double> total{0};
  std::atomic<uint64_t> sink{0};
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) pool.emplace_back([&, t] {
    double moved = 0; uint64_t acc = 0; size_t n = per / 8;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
      if (mode == "copy") { memcpy(dst[t], bufs[t], per); moved += 2.0 * per; }
      else {
        const uint64_t* p = bufs[t];
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (size_t i = 0; i < n; i += 4) { a0 ^= p[i]; a1 ^= p[i + 1]; a2 ^= p[i + 2]; a3 ^= p[i + 3]; }
        acc ^= a0 ^ a1 ^ a2 ^ a3; moved += per;
      }
    }
    sink ^= acc;
    double cur = total.load(); while (!total.compare_exchange_weak(cur, cur + moved)) {}
  });
  for (auto& th : pool) th.join();
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("cpu %s threads %d: %.1f GB/s (sink %llu)\n", mode.c_str(), threads, total / el / 1e9,
         (unsigned long long)(sink.load() & 1));
  return 0;
}
