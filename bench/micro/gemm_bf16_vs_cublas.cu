// The engine's BF16 GEMM (kernels::gemm_bf16) against cuBLAS's cublasGemmEx, the call it replaced
// (BACKLOG §19.23): bit for bit and speed, on random data, at the prefill's shapes. Built by hand, the
// one place cuBLAS is still linked (the engine itself no longer is):
//
//   nvcc -std=c++20 -O2 -gencode arch=compute_121a,code=sm_121a -Isrc bench/micro/gemm_bf16_vs_cublas.cu \
//        build/libling_core.a -lcublas -o build/gemm-bf16-vs-cublas && build/gemm-bf16-vs-cublas
//
// Prints per shape: the outputs that differ from cuBLAS's (bitwise), the largest difference relative to
// the outputs' RMS, and the median of 20 timed runs of each, in ms and TFLOPS.
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/kernels.cuh"

namespace {

void gemm_cublas(cublasHandle_t h, const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N, int K) {
  const float alpha = 1.f, beta = 0.f;
  if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &beta, y,
                   CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
    std::printf("cublasGemmEx failed\n");
}

template <typename F>
double median_ms(F&& f) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  f();
  cudaDeviceSynchronize();
  std::vector<float> t;
  for (int i = 0; i < 20; ++i) {
    cudaEventRecord(a);
    f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    t.push_back(ms);
  }
  std::sort(t.begin(), t.end());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const bool time_only = argc > 1 && std::strcmp(argv[1], "--time") == 0;
  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
  std::mt19937 rng(11);
  std::normal_distribution<float> normal(0.f, 1.f);
  struct Shape {
    const char* what;
    int N, K;
  };
  // The drafter's fc (the served path), its selector projection, and the DeltaNet's a/b projections.
  const Shape shapes[] = {{"drafter fc", 5120, 25600}, {"selector projection", 256, 5120}, {"deltanet a/b", 48, 5120}};
  const int rows[] = {33, 64, 100, 256, 512, 1000, 2048, 4096};
  bool all_equal = true;
  for (const Shape& s : shapes) {
    const int Mmax = 4096;
    std::vector<__nv_bfloat16> hx(size_t(Mmax) * s.K), hw(size_t(s.N) * s.K);
    for (auto& v : hx) v = __float2bfloat16(normal(rng));
    for (auto& v : hw) v = __float2bfloat16(0.02f * normal(rng));
    __nv_bfloat16 *x, *w;
    float *y0, *y1;
    cudaMalloc(&x, hx.size() * 2);
    cudaMalloc(&w, hw.size() * 2);
    cudaMalloc(&y0, size_t(Mmax) * s.N * 4);
    cudaMalloc(&y1, size_t(Mmax) * s.N * 4);
    cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    std::vector<float> a(size_t(Mmax) * s.N), b(size_t(Mmax) * s.N);
    for (int M : rows) {
      size_t differ = 0;
      double worst = 0, rms = 0;
      if (!time_only) {
        gemm_cublas(h, x, M, w, y0, s.N, s.K);
        ling::kernels::gemm_bf16(x, M, w, y1, s.N, s.K, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(a.data(), y0, size_t(M) * s.N * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), y1, size_t(M) * s.N * 4, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < size_t(M) * s.N; ++i) {
          differ += std::memcmp(&a[i], &b[i], 4) != 0;
          worst = std::max(worst, double(std::abs(a[i] - b[i])));
          rms += double(a[i]) * a[i];
        }
        rms = std::sqrt(rms / (size_t(M) * s.N));
        all_equal &= differ == 0;
      }
      const double tc = median_ms([&] { gemm_cublas(h, x, M, w, y0, s.N, s.K); });
      const double te = median_ms([&] { ling::kernels::gemm_bf16(x, M, w, y1, s.N, s.K, nullptr); });
      const double flop = 2.0 * M * s.N * s.K;
      std::printf("%-20s M=%-5d N=%-5d K=%-5d differ %zu/%zu (max %.2e x rms)  cublas %.3f ms %.1f TFLOPS  engine %.3f ms "
                  "%.1f TFLOPS  engine/cublas %.3f\n",
                  s.what, M, s.N, s.K, differ, size_t(M) * s.N, rms > 0 ? worst / rms : 0.0, tc, flop / tc / 1e9, te,
                  flop / te / 1e9, te / tc);
    }
    cudaFree(x);
    cudaFree(w);
    cudaFree(y0);
    cudaFree(y1);
  }
  std::printf(time_only ? "timed only\n" : all_equal ? "BIT-IDENTICAL to cuBLAS on every shape\n" : "DIFFERS from cuBLAS\n");
  return 0;
}
