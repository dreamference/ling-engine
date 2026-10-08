// The tensor-core streaming GEMM (1-32 rows) and the row top-K: accuracy against a host reference,
// bit-for-bit row invariance across M (what makes greedy speculation reproduce plain decoding), and
// bandwidth on the model's own matrix shapes.
//
//   ling-stream-tests            correctness only
//   ling-stream-tests --bench    plus timings
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.cuh"

namespace {

const float kFp4[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};

float fp8(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}

template <typename T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

// y = x . W^T through the FP16 conversion and the streaming kernel.
void run(bool fp4, const float* dx, int M, int K, __half* xh, float* xinv, const uint8_t* w, const uint8_t* s,
         float gs, float* dy, int N) {
  ling::kernels::to_half_rows(dx, M, K, xh, xinv, nullptr);
  if (fp4) ling::kernels::stream_gemm_nvfp4(xh, xinv, M, w, s, gs, dy, N, K, nullptr);
  else ling::kernels::stream_gemm_fp8(xh, xinv, M, w, gs, dy, N, K, nullptr);
}

bool test_gemm() {
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> byte(0, 255), scale_byte(0x28, 0x48);
  std::normal_distribution<float> normal(0.f, 1.f);
  const int N = 520, K = 2048, MMAX = ling::kernels::kMaxStreamRows;
  std::vector<uint8_t> w4(size_t(N) * K / 2), s4(size_t(N) * K / 16), w8(size_t(N) * K);
  for (auto& b : w4) b = byte(rng);
  for (auto& b : s4) b = scale_byte(rng);
  for (auto& b : w8) {
    do b = byte(rng);
    while ((b & 0x7f) == 0x7f);
  }
  const float gs4 = 0.37f, gs8 = 0.021f;
  std::vector<float> x(size_t(MMAX) * K);
  for (size_t i = 0; i < x.size(); ++i) x[i] = normal(rng) * (i % 97 == 0 ? 30.f : 1.f);  // a few outliers
  uint8_t *dw4 = upload(w4), *ds4 = upload(s4), *dw8 = upload(w8);
  float* dx = upload(x);
  __half* xh;
  float *xinv, *dy;
  cudaMalloc(&xh, size_t(MMAX) * K * sizeof(__half));
  cudaMalloc(&xinv, MMAX * sizeof(float));
  cudaMalloc(&dy, size_t(MMAX) * N * sizeof(float));
  bool ok = true;
  for (bool fp4 : {true, false}) {
    std::vector<float> ref(size_t(MMAX) * N);
    for (int m = 0; m < MMAX; ++m)
      for (int n = 0; n < N; ++n) {
        double a = 0;
        for (int k = 0; k < K; ++k) {
          float wv;
          if (fp4) {
            const uint8_t b = w4[(size_t(n) * K + k) / 2];
            wv = kFp4[(k & 1) ? b >> 4 : b & 0xf] * fp8(s4[size_t(n) * (K / 16) + k / 16]);
          } else {
            wv = fp8(w8[size_t(n) * K + k]);
          }
          a += double(wv) * x[size_t(m) * K + k];
        }
        ref[size_t(m) * N + n] = float(a * (fp4 ? gs4 : gs8));
      }
    run(fp4, dx, MMAX, K, xh, xinv, fp4 ? dw4 : dw8, ds4, fp4 ? gs4 : gs8, dy, N);
    const std::vector<float> full = download(dy, size_t(MMAX) * N);
    for (int M : {1, 2, 7, 8, 9, 15, 16, 17, 24, 32}) {
      run(fp4, dx, M, K, xh, xinv, fp4 ? dw4 : dw8, ds4, fp4 ? gs4 : gs8, dy, N);
      const std::vector<float> got = download(dy, size_t(M) * N);
      double worst = 0;
      for (int m = 0; m < M; ++m) {
        double rms = 0;
        for (int n = 0; n < N; ++n) rms += double(ref[size_t(m) * N + n]) * ref[size_t(m) * N + n];
        rms = std::sqrt(rms / N);
        for (int n = 0; n < N; ++n)
          worst = std::max(worst, std::abs(got[size_t(m) * N + n] - ref[size_t(m) * N + n]) / rms);
      }
      const bool same = std::memcmp(got.data(), full.data(), got.size() * sizeof(float)) == 0;
      const bool good = worst < 2e-3 && same;
      ok &= good;
      std::printf("stream_gemm %-5s M=%-2d err/rms %.2e, rows identical to M=32: %s %s\n", fp4 ? "nvfp4" : "fp8", M,
                  worst, same ? "yes" : "NO", good ? "ok" : "FAIL");
    }
  }
  cudaFree(dw4), cudaFree(ds4), cudaFree(dw8), cudaFree(dx), cudaFree(xh), cudaFree(xinv), cudaFree(dy);
  return ok;
}

bool test_topk() {
  std::mt19937 rng(5);
  std::normal_distribution<float> normal(0.f, 3.f);
  const int rows = 5, V = 248320;
  std::vector<float> x(size_t(rows) * V);
  for (auto& v : x) v = std::round(normal(rng) * 64.f) / 64.f;  // coarse values: many ties
  float* dx = upload(x);
  bool ok = true;
  for (int K : {1, 16, 20, 64}) {
    float *scratch, *vals;
    int* ids;
    cudaMalloc(&scratch, ling::kernels::topk_scratch_floats(rows, K) * sizeof(float));
    cudaMalloc(&vals, size_t(rows) * K * sizeof(float));
    cudaMalloc(&ids, size_t(rows) * K * sizeof(int));
    ling::kernels::topk_rows(dx, rows, V, K, scratch, vals, ids, nullptr);
    const auto gv = download(vals, size_t(rows) * K);
    const auto gi = download(ids, size_t(rows) * K);
    bool good = true;
    for (int r = 0; r < rows; ++r) {
      std::vector<int> idx(V);
      std::iota(idx.begin(), idx.end(), 0);
      const float* row = x.data() + size_t(r) * V;
      std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                        [&](int a, int b) { return row[a] > row[b] || (row[a] == row[b] && a < b); });
      for (int k = 0; k < K; ++k)
        good &= gi[size_t(r) * K + k] == idx[k] && gv[size_t(r) * K + k] == row[idx[k]];
      good &= gi[size_t(r) * K] == int(std::max_element(row, row + V) - row);
    }
    ok &= good;
    std::printf("topk_rows K=%-2d %s\n", K, good ? "ok" : "FAIL");
    cudaFree(scratch), cudaFree(vals), cudaFree(ids);
  }
  cudaFree(dx);
  return ok;
}

void bench() {
  struct Shape {
    const char* name;
    bool fp4;
    int N, K;
  };
  const Shape shapes[] = {
      {"ffn gate|up (nvfp4)", true, 17408, 5120}, {"ffn down (nvfp4)", true, 5120, 17408},
      {"lm head (nvfp4)", true, 248320, 5120},    {"gdn in_qkv (fp8)", false, 10240, 5120},
      {"gdn in_z (fp8)", false, 6144, 5120},      {"out proj (fp8)", false, 5120, 6144},
      {"attn q (fp8)", false, 12288, 5120},       {"attn k|v (fp8)", false, 1024, 5120},
  };
  std::mt19937 rng(3);
  for (const Shape& sh : shapes) {
    const size_t wbytes = sh.fp4 ? size_t(sh.N) * sh.K / 2 : size_t(sh.N) * sh.K;
    const size_t sbytes = sh.fp4 ? size_t(sh.N) * sh.K / 16 : 0;
    // Enough copies to read 512 MB per sweep, so the 25 MB L2 never holds the next matrix.
    const int copies = int(std::max<size_t>(1, (512u << 20) / (wbytes + sbytes)));
    uint8_t *w, *s = nullptr;
    cudaMalloc(&w, wbytes * copies);
    cudaMemset(w, 0x22, wbytes * copies);
    if (sbytes) {
      cudaMalloc(&s, sbytes * copies);
      cudaMemset(s, 0x38, sbytes * copies);
    }
    float *x, *y, *xinv;
    __half* xh;
    cudaMalloc(&x, size_t(32) * sh.K * sizeof(float));
    cudaMemset(x, 0, size_t(32) * sh.K * sizeof(float));
    cudaMalloc(&xh, size_t(32) * sh.K * sizeof(__half));
    cudaMalloc(&xinv, 32 * sizeof(float));
    cudaMalloc(&y, size_t(32) * sh.N * sizeof(float));
    ling::kernels::to_half_rows(x, 32, sh.K, xh, xinv, nullptr);
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    std::printf("bench %-22s", sh.name);
    for (int M : {1, 8, 16, 32}) {
      const int iters = std::max(20, copies * 4);
      for (int pass = 0; pass < 2; ++pass) {
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i) {
          const int c = i % copies;
          if (sh.fp4) ling::kernels::stream_gemm_nvfp4(xh, xinv, M, w + wbytes * c, s + sbytes * c, 1.f, y, sh.N, sh.K, nullptr);
          else ling::kernels::stream_gemm_fp8(xh, xinv, M, w + wbytes * c, 1.f, y, sh.N, sh.K, nullptr);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        if (pass == 1) {
          float ms = 0;
          cudaEventElapsedTime(&ms, a, b);
          std::printf("  M=%-2d %6.3f ms %4.0f GB/s", M, ms / iters, (wbytes + sbytes) / (ms / iters * 1e-3) / 1e9);
        }
      }
    }
    // The v0 GEMV at one row, for comparison.
    {
      const int iters = std::max(20, copies * 4);
      cudaEventRecord(a);
      for (int i = 0; i < iters; ++i) {
        const int c = i % copies;
        if (sh.fp4) ling::kernels::gemv_nvfp4(x, 1, w + wbytes * c, s + sbytes * c, 1.f, y, sh.N, sh.K, nullptr);
        else ling::kernels::gemv_fp8(x, 1, w + wbytes * c, 1.f, y, sh.N, sh.K, nullptr);
      }
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      std::printf("  | v0 gemv M=1 %4.0f GB/s\n", (wbytes + sbytes) / (ms / iters * 1e-3) / 1e9);
    }
    cudaFree(w), cudaFree(s), cudaFree(x), cudaFree(xh), cudaFree(xinv), cudaFree(y);
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool ok = test_gemm();
  ok &= test_topk();
  if (argc > 1 && std::string(argv[1]) == "--bench") bench();
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf(ok ? "all stream tests passed\n" : "STREAM TESTS FAILED\n");
  return ok ? 0 : 1;
}
