// Checks the weight-streaming kernels against a host reference on random data, for every row count the
// GEMV path takes and for the cuBLAS path.
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <random>
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

bool close(const std::vector<float>& a, const std::vector<float>& b, const char* what, int M) {
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::abs(a[i] - b[i]) / (std::abs(b[i]) + 1.0));
  const bool ok = worst < 2e-3;
  std::printf("%-26s M=%-3d max rel err %.2e %s\n", what, M, worst, ok ? "ok" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> scale_byte(0x30, 0x40);  // modest positive E4M3 scales
  std::normal_distribution<float> normal(0.f, 1.f);
  const int N = 300, K = 2048;
  bool ok = true;

  std::vector<uint8_t> w4(size_t(N) * K / 2), s4(size_t(N) * K / 16), w8(size_t(N) * K);
  for (auto& b : w4) b = byte(rng);
  for (auto& b : s4) b = scale_byte(rng);
  for (auto& b : w8) {
    do b = byte(rng);
    while ((b & 0x7f) == 0x7f);  // skip NaN encodings
  }
  const float scale2 = 0.37f, scale8 = 0.021f;
  uint8_t *dw4 = upload(w4), *ds4 = upload(s4), *dw8 = upload(w8);
  cublasHandle_t h;
  cublasCreate(&h);

  for (int M : {1, 2, 3, 5, 8, 33}) {
    std::vector<float> x(size_t(M) * K);
    for (auto& v : x) v = normal(rng);
    float* dx = upload(x);
    float* dy = nullptr;
    cudaMalloc(&dy, size_t(M) * N * sizeof(float));
    std::vector<float> ref4(size_t(M) * N), ref8(size_t(M) * N), got(size_t(M) * N);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double a4 = 0, a8 = 0;
        for (int k = 0; k < K; ++k) {
          const uint8_t b = w4[(size_t(n) * K + k) / 2];
          const float q = kFp4[(k & 1) ? b >> 4 : b & 0xf];
          a4 += double(q) * fp8(s4[size_t(n) * (K / 16) + k / 16]) * x[size_t(m) * K + k];
          a8 += double(fp8(w8[size_t(n) * K + k])) * x[size_t(m) * K + k];
        }
        ref4[size_t(m) * N + n] = float(a4 * scale2);
        ref8[size_t(m) * N + n] = float(a8 * scale8);
      }
    if (M <= ling::kernels::kMaxGemvRows) {
      ling::kernels::gemv_nvfp4(dx, M, dw4, ds4, scale2, dy, N, K, nullptr);
      cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
      ok &= close(got, ref4, "gemv_nvfp4", M);
      ling::kernels::gemv_fp8(dx, M, dw8, scale8, dy, N, K, nullptr);
      cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
      ok &= close(got, ref8, "gemv_fp8", M);
    } else {
      __nv_bfloat16 *wb, *xb;
      cudaMalloc(&wb, size_t(N) * K * 2);
      cudaMalloc(&xb, size_t(M) * K * 2);
      ling::kernels::to_bf16(dx, xb, M * K, nullptr);
      ling::kernels::dequant_nvfp4(dw4, ds4, scale2, wb, N, K, nullptr);
      ling::kernels::gemm_bf16_cublas(h, xb, M, wb, dy, N, K);
      cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
      // BF16 inputs: a looser bound.
      // BF16 inputs: compare the worst error with the outputs' RMS, a looser bound.
      double worst = 0, rms = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, double(std::abs(got[i] - ref4[i])));
        rms += double(ref4[i]) * ref4[i];
      }
      worst /= std::sqrt(rms / got.size());
      std::printf("%-26s M=%-3d max err / rms %.2e %s\n", "dequant_nvfp4+cublas", M, worst, worst < 2e-2 ? "ok" : "FAIL");
      ok &= worst < 2e-2;
      cudaFree(wb);
      cudaFree(xb);
    }
    cudaFree(dx);
    cudaFree(dy);
  }
  // Bandwidth of the decode path (M = 1) on the model's largest shapes; informational, not a pass/fail.
  {
    const int BN = 17408, BK = 5120;
    uint8_t *bw4, *bs4, *bw8;
    float *bx, *by;
    cudaMalloc(&bw4, size_t(BN) * BK / 2);
    cudaMalloc(&bs4, size_t(BN) * BK / 16);
    cudaMalloc(&bw8, size_t(BN) * BK);
    cudaMalloc(&bx, BK * sizeof(float));
    cudaMalloc(&by, BN * sizeof(float));
    cudaMemset(bw4, 0x11, size_t(BN) * BK / 2);
    cudaMemset(bs4, 0x38, size_t(BN) * BK / 16);
    cudaMemset(bw8, 0x38, size_t(BN) * BK);
    cudaMemset(bx, 0, BK * sizeof(float));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int pass = 0; pass < 2; ++pass) {
      const bool fp4 = pass == 0;
      for (int i = 0; i < 3; ++i) {
        if (fp4) ling::kernels::gemv_nvfp4(bx, 1, bw4, bs4, 1.f, by, BN, BK, nullptr);
        else ling::kernels::gemv_fp8(bx, 1, bw8, 1.f, by, BN, BK, nullptr);
      }
      cudaEventRecord(a);
      const int iters = 50;
      for (int i = 0; i < iters; ++i) {
        if (fp4) ling::kernels::gemv_nvfp4(bx, 1, bw4, bs4, 1.f, by, BN, BK, nullptr);
        else ling::kernels::gemv_fp8(bx, 1, bw8, 1.f, by, BN, BK, nullptr);
      }
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      const double bytes = fp4 ? double(BN) * BK * (0.5 + 1.0 / 16) : double(BN) * BK;
      std::printf("bench %-8s M=1 %dx%d: %.3f ms, %.0f GB/s\n", fp4 ? "nvfp4" : "fp8", BN, BK, ms / iters,
                  bytes / (ms / iters * 1e-3) / 1e9);
    }
    cudaFree(bw4);
    cudaFree(bs4);
    cudaFree(bw8);
    cudaFree(bx);
    cudaFree(by);
  }
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf(ok ? "all kernel tests passed\n" : "KERNEL TESTS FAILED\n");
  return ok ? 0 : 1;
}
