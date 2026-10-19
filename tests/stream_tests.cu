// The tensor-core streaming GEMM (1-32 rows) and the row top-K: accuracy against a host reference,
// bit-for-bit row invariance across M (what makes greedy speculation reproduce plain decoding), and
// bandwidth on the model's own matrix shapes.
//
//   ling-stream-tests            correctness only
//   ling-stream-tests --bench    plus timings
//   LING_PDL=1 ling-stream-tests  the same with programmatic dependent launch (launch.cuh)
//   LING_NO_TIMING=1 ling-stream-tests  skips the attention timings (under compute-sanitizer)
//   ling-stream-tests --nvfp4-oracle DIR  only: the at-load NVFP4 rounding (BACKLOG 33) against the "<name>.nvfp4.*"
//                                        copies bench/quant/drafter_variants.py --nvfp4-copy wrote into DIR, byte for byte
#include <cuda.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "core/kernels.cuh"
#include "core/model.hpp"
#include "core/safetensors.hpp"

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
  if (fp4) ling::kernels::stream_gemm_nvfp4(xh, xinv, M, w, nullptr, gs, dy, N, K, nullptr);
  else ling::kernels::stream_gemm_fp8(xh, xinv, M, w, gs, dy, N, K, nullptr);
}

bool test_gemm() {
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> byte(0, 255), scale_byte(0x28, 0x48);
  std::normal_distribution<float> normal(0.f, 1.f);
  const int N = 528, K = 2048, MMAX = ling::kernels::kMaxStreamRows;
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
  uint8_t *rw4 = upload(w4), *ds4 = upload(s4), *rw8 = upload(w8);
  // The kernels read the tiled layout; the row-major copies check the tiled dequantizers.
  uint8_t *dw4, *dw8;
  cudaMalloc(&dw4, ling::kernels::tiled_bytes_nvfp4(N, K));
  cudaMalloc(&dw8, ling::kernels::tiled_bytes_fp8(N, K));
  ling::kernels::retile_nvfp4(rw4, ds4, dw4, N, K, nullptr);
  ling::kernels::retile_fp8(rw8, dw8, N, K, nullptr);
  float* dx = upload(x);
  bool ok = true;
  {
    __nv_bfloat16 *a, *b;
    cudaMalloc(&a, size_t(N) * K * 2);
    cudaMalloc(&b, size_t(N) * K * 2);
    ling::kernels::dequant_nvfp4(rw4, ds4, gs4, a, N, K, nullptr);
    ling::kernels::dequant_tiled_nvfp4(dw4, gs4, b, N, K, nullptr);
    bool same = download(a, size_t(N) * K) == download(b, size_t(N) * K);
    ling::kernels::dequant_fp8(rw8, gs8, a, N, K, nullptr);
    ling::kernels::dequant_tiled_fp8(dw8, gs8, b, N, K, nullptr);
    same &= download(a, size_t(N) * K) == download(b, size_t(N) * K);
    std::printf("retile + tiled dequant equal row-major dequant (NVFP4, FP8): %s\n", same ? "yes ok" : "NO FAIL");
    ok &= same;
    cudaFree(a), cudaFree(b);
  }
  __half* xh;
  float *xinv, *dy;
  cudaMalloc(&xh, size_t(MMAX) * K * sizeof(__half));
  cudaMalloc(&xinv, MMAX * sizeof(float));
  cudaMalloc(&dy, size_t(MMAX) * N * sizeof(float));
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
  // B26: the same NVFP4 matrix with 6-bit scale codes streams to the same bits, at every M, alone and in one launch
  // beside its raw copy; its tiled dequantization (with the raw scales) equals the raw blob's.
  {
    uint8_t table[ling::kernels::kScaleCodes], code_of[256];
    bool same = ling::kernels::scale_code_table(s4.data(), s4.size(), table, code_of);
    uint8_t *map = upload(std::vector<uint8_t>(code_of, code_of + 256)), *dtab = upload(std::vector<uint8_t>(table, table + 64));
    uint8_t *coded, *raw;
    cudaMalloc(&coded, ling::kernels::tiled_bytes_nvfp4_coded(N, K));
    cudaMalloc(&raw, s4.size());
    ling::kernels::retile_nvfp4_coded(rw4, ds4, map, coded, raw, N, K, nullptr);
    for (int M : {1, 8, 12, 16, 17, 24, 32}) {
      ling::kernels::to_half_rows(dx, M, K, xh, xinv, nullptr);
      ling::kernels::stream_gemm_nvfp4(xh, xinv, M, dw4, nullptr, gs4, dy, N, K, nullptr);
      const auto a = download(dy, size_t(M) * N);
      ling::kernels::stream_gemm_nvfp4(xh, xinv, M, coded, dtab, gs4, dy, N, K, nullptr);
      same &= download(dy, size_t(M) * N) == a;
    }
    {
      __nv_bfloat16 *a, *b;
      cudaMalloc(&a, size_t(N) * K * 2);
      cudaMalloc(&b, size_t(N) * K * 2);
      ling::kernels::dequant_tiled_nvfp4(dw4, gs4, a, N, K, nullptr);
      ling::kernels::dequant_tiled_nvfp4(coded, gs4, b, N, K, nullptr, raw);
      same &= download(a, size_t(N) * K) == download(b, size_t(N) * K);
      cudaFree(a), cudaFree(b);
    }
    {
      const int M = 12;
      float *ya, *yb;
      cudaMalloc(&ya, size_t(M) * N * sizeof(float));
      cudaMalloc(&yb, size_t(M) * N * sizeof(float));
      ling::kernels::to_half_rows(dx, M, K, xh, xinv, nullptr);
      const ling::kernels::StreamTarget t[2] = {{coded, dtab, 0.5f, ya, N}, {dw4, nullptr, 0.25f, yb, N}};
      ling::kernels::stream_gemm_multi(true, xh, xinv, M, t, 2, K, nullptr);
      const auto ga = download(ya, size_t(M) * N), gb = download(yb, size_t(M) * N);
      ling::kernels::stream_gemm_nvfp4(xh, xinv, M, dw4, nullptr, 0.5f, dy, N, K, nullptr);
      same &= download(dy, size_t(M) * N) == ga;
      ling::kernels::stream_gemm_nvfp4(xh, xinv, M, dw4, nullptr, 0.25f, dy, N, K, nullptr);
      same &= download(dy, size_t(M) * N) == gb;
      cudaFree(ya), cudaFree(yb);
    }
    ok &= same;
    std::printf("stream_gemm nvfp4 with 6-bit scale codes identical to raw scales (M = 1 .. 32, dequant, mixed launch): %s\n",
                same ? "yes ok" : "NO FAIL");
    cudaFree(map), cudaFree(dtab), cudaFree(coded), cudaFree(raw);
  }
  // Several matrices in one launch give exactly the separate launches' results.
  for (bool fp4 : {true, false}) {
    const int M = 16;
    float *ya, *yb;
    cudaMalloc(&ya, size_t(M) * N * sizeof(float));
    cudaMalloc(&yb, size_t(M) * (N - 144) * sizeof(float));
    ling::kernels::to_half_rows(dx, M, K, xh, xinv, nullptr);
    // The second target: the first N - 144 rows of the same weights with another scale.
    const ling::kernels::StreamTarget t[2] = {{fp4 ? dw4 : dw8, nullptr, 0.5f, ya, N},
                                              {fp4 ? dw4 : dw8, nullptr, 0.25f, yb, N - 144}};
    ling::kernels::stream_gemm_multi(fp4, xh, xinv, M, t, 2, K, nullptr);
    const auto ga = download(ya, size_t(M) * N), gb = download(yb, size_t(M) * (N - 144));
    run(fp4, dx, M, K, xh, xinv, fp4 ? dw4 : dw8, ds4, 0.5f, dy, N);
    const auto ra = download(dy, size_t(M) * N);
    run(fp4, dx, M, K, xh, xinv, fp4 ? dw4 : dw8, ds4, 0.25f, dy, N - 144);
    const auto rb = download(dy, size_t(M) * (N - 144));
    const bool same = ga == ra && gb == rb;
    ok &= same;
    std::printf("stream_gemm_multi %-5s two matrices in one launch identical to separate launches: %s\n",
                fp4 ? "nvfp4" : "fp8", same ? "yes ok" : "NO FAIL");
    cudaFree(ya), cudaFree(yb);
  }
  cudaFree(dw4), cudaFree(ds4), cudaFree(dw8), cudaFree(rw4), cudaFree(rw8), cudaFree(dx), cudaFree(xh), cudaFree(xinv),
      cudaFree(dy);
  return ok;
}

// Row invariance on a wide matrix too (N >= 8192 takes the wide launch shapes, B16): every M's rows are bit for bit
// the M = 32 launch's, NVFP4 and FP8, single and as two matrices in one launch.
bool test_gemm_wide() {
  std::mt19937 rng(12);
  std::uniform_int_distribution<int> byte(0, 255), scale_byte(0x28, 0x48);
  std::normal_distribution<float> normal(0.f, 1.f);
  const int N = 8448, K = 1024, MMAX = ling::kernels::kMaxStreamRows;
  std::vector<uint8_t> w4(size_t(N) * K / 2), s4(size_t(N) * K / 16), w8(size_t(N) * K);
  for (auto& b : w4) b = byte(rng);
  for (auto& b : s4) b = scale_byte(rng);
  for (auto& b : w8) {
    do b = byte(rng);
    while ((b & 0x7f) == 0x7f);
  }
  std::vector<float> x(size_t(MMAX) * K);
  for (float& v : x) v = normal(rng);
  uint8_t *rw4 = upload(w4), *ds4 = upload(s4), *rw8 = upload(w8), *dw4, *dw8;
  cudaMalloc(&dw4, ling::kernels::tiled_bytes_nvfp4(N, K));
  cudaMalloc(&dw8, ling::kernels::tiled_bytes_fp8(N, K));
  ling::kernels::retile_nvfp4(rw4, ds4, dw4, N, K, nullptr);
  ling::kernels::retile_fp8(rw8, dw8, N, K, nullptr);
  float* dx = upload(x);
  __half* xh;
  float *xinv, *dy;
  cudaMalloc(&xh, size_t(MMAX) * K * sizeof(__half));
  cudaMalloc(&xinv, MMAX * sizeof(float));
  cudaMalloc(&dy, size_t(MMAX) * N * sizeof(float));
  bool ok = true;
  for (bool fp4 : {true, false}) {
    run(fp4, dx, MMAX, K, xh, xinv, fp4 ? dw4 : dw8, ds4, 0.5f, dy, N);
    const std::vector<float> full = download(dy, size_t(MMAX) * N);
    bool same = true;
    for (int M : {1, 8, 9, 12, 16, 17, 24}) {
      run(fp4, dx, M, K, xh, xinv, fp4 ? dw4 : dw8, ds4, 0.5f, dy, N);
      const std::vector<float> got = download(dy, size_t(M) * N);
      same &= std::memcmp(got.data(), full.data(), got.size() * sizeof(float)) == 0;
    }
    ok &= same;
    std::printf("stream_gemm %-5s wide (N=%d) rows identical to M=32 for M = 1 .. 24: %s\n", fp4 ? "nvfp4" : "fp8", N,
                same ? "yes ok" : "NO FAIL");
  }
  cudaFree(dw4), cudaFree(dw8), cudaFree(rw4), cudaFree(ds4), cudaFree(rw8), cudaFree(dx), cudaFree(xh), cudaFree(xinv),
      cudaFree(dy);
  return ok;
}

bool test_bf16_rows() {
  std::mt19937 rng(9);
  std::normal_distribution<float> normal(0.f, 1.f);
  const int N = 48, K = 5120, MMAX = 32;
  std::vector<__nv_bfloat16> w(size_t(N) * K);
  for (auto& v : w) v = __float2bfloat16(normal(rng) * 0.05f);
  std::vector<float> x(size_t(MMAX) * K);
  for (auto& v : x) v = normal(rng);
  __nv_bfloat16* dw = upload(w);
  float* dx = upload(x);
  float* dy;
  cudaMalloc(&dy, size_t(MMAX) * N * sizeof(float));
  ling::kernels::bf16_rows(dx, MMAX, dw, dy, N, K, nullptr);
  const std::vector<float> full = download(dy, size_t(MMAX) * N);
  bool ok = true;
  for (int M : {1, 5, 16, 32}) {
    ling::kernels::bf16_rows(dx, M, dw, dy, N, K, nullptr);
    const std::vector<float> got = download(dy, size_t(M) * N);
    double worst = 0;
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double a = 0;
        for (int k = 0; k < K; ++k) a += double(__bfloat162float(w[size_t(n) * K + k])) * x[size_t(m) * K + k];
        worst = std::max(worst, std::abs(got[size_t(m) * N + n] - a) / (std::abs(a) + 1.0));
      }
    const bool same = std::memcmp(got.data(), full.data(), got.size() * sizeof(float)) == 0;
    const bool good = worst < 1e-4 && same;
    ok &= good;
    std::printf("bf16_rows M=%-2d max rel err %.2e, rows identical to M=32: %s %s\n", M, worst, same ? "yes" : "NO",
                good ? "ok" : "FAIL");
  }
  cudaFree(dw), cudaFree(dx), cudaFree(dy);
  return ok;
}

// The rows path's attention: against a host reference, and each row of a 32-row call bit for bit equal
// to a one-row call at that row's position (what decode and verify see).
bool test_attention_rows() {
  std::mt19937 rng(13);
  std::normal_distribution<float> normal(0.f, 1.f);
  const int Hq = 24, Hkv = 4, D = 256, P = 2900, MMAX = 32, ctx = P + MMAX;
  std::vector<__nv_bfloat16> kc(size_t(ctx) * Hkv * D), vc(kc.size());
  for (auto& v : kc) v = __float2bfloat16(normal(rng));
  for (auto& v : vc) v = __float2bfloat16(normal(rng));
  std::vector<float> q(size_t(MMAX) * Hq * D);
  for (auto& v : q) v = normal(rng) * 2.f;
  __nv_bfloat16 *dk = upload(kc), *dv = upload(vc);
  float* dq = upload(q);
  float *scratch, *out;
  cudaMalloc(&scratch, ling::kernels::attention_rows_scratch_floats(MMAX, Hq, D, ctx) * sizeof(float));
  cudaMalloc(&out, size_t(MMAX) * Hq * D * sizeof(float));
  ling::kernels::attention_rows(dq, dk, dv, P, MMAX, Hq, Hkv, D, scratch, out, nullptr);
  const std::vector<float> full = download(out, size_t(MMAX) * Hq * D);
  bool ok = true;
  double worst = 0;
  for (int m : {0, 7, 31})
    for (int h = 0; h < Hq; h += 5) {
      const int kvh = h / 6, pos = P + m;
      std::vector<double> sc(pos + 1);
      double mx = -1e300;
      for (int j = 0; j <= pos; ++j) {
        double a = 0;
        for (int d = 0; d < D; ++d)
          a += double(__bfloat162float(__float2bfloat16(q[(size_t(m) * Hq + h) * D + d] * 0.0625f))) *
               __bfloat162float(kc[(size_t(j) * Hkv + kvh) * D + d]);
        sc[j] = a;
        mx = std::max(mx, a);
      }
      double l = 0;
      for (double& v : sc) l += v = std::exp(v - mx);
      for (int d = 0; d < D; ++d) {
        double o = 0;
        for (int j = 0; j <= pos; ++j) o += sc[j] * __bfloat162float(vc[(size_t(j) * Hkv + kvh) * D + d]);
        o /= l;
        worst = std::max(worst, std::abs(full[(size_t(m) * Hq + h) * D + d] - o));
      }
    }
  ok &= worst < 2e-2;
  std::printf("attention_rows vs host: max abs err %.2e %s\n", worst, worst < 2e-2 ? "ok" : "FAIL");
  bool same = true;
  for (int m = 0; m < MMAX; ++m) {
    ling::kernels::attention_rows(dq + size_t(m) * Hq * D, dk, dv, P + m, 1, Hq, Hkv, D, scratch, out, nullptr);
    const std::vector<float> one = download(out, size_t(Hq) * D);
    same &= std::memcmp(one.data(), full.data() + size_t(m) * Hq * D, one.size() * sizeof(float)) == 0;
  }
  ling::kernels::attention_rows(dq, dk, dv, P, 16, Hq, Hkv, D, scratch, out, nullptr);
  const std::vector<float> sixteen = download(out, size_t(16) * Hq * D);
  same &= std::memcmp(sixteen.data(), full.data(), sixteen.size() * sizeof(float)) == 0;
  ok &= same;
  std::printf("attention_rows rows identical across M = 1, 16, 32: %s\n", same ? "yes ok" : "NO FAIL");
  // As a step graph launches it: the position read on the device (the host's value deliberately wrong),
  // every key range up to a fixed context. Must be bit for bit the eager launch.
  {
    const int fixed = 3 * 4096;
    float* gscratch;
    cudaMalloc(&gscratch, ling::kernels::attention_rows_scratch_floats(MMAX, Hq, D, fixed) * sizeof(float));
    int* dpos = upload(std::vector<int>{P});
    ling::kernels::attention_rows(dq, dk, dv, ling::kernels::DevPos(-1, dpos, 0), 16, Hq, Hkv, D, gscratch, out, nullptr,
                                  fixed);
    const std::vector<float> g = download(out, size_t(16) * Hq * D);
    const bool gsame = std::memcmp(g.data(), full.data(), g.size() * sizeof(float)) == 0;
    ok &= gsame;
    std::printf("attention_rows with a device position and fixed ranges identical: %s\n", gsame ? "yes ok" : "NO FAIL");
    cudaFree(gscratch), cudaFree(dpos);
  }
  // The bulk-copy tiles (M3): every row bit for bit the cp.async kernel's, at M = 32, 16 and 1, with a
  // range that ends inside a tile (the rows past it are not copied).
  {
    ling::kernels::set_attention_bulk(true);
    bool bsame = true;
    ling::kernels::attention_rows(dq, dk, dv, P, MMAX, Hq, Hkv, D, scratch, out, nullptr);
    const std::vector<float> b32 = download(out, size_t(MMAX) * Hq * D);
    bsame &= std::memcmp(b32.data(), full.data(), b32.size() * sizeof(float)) == 0;
    ling::kernels::attention_rows(dq, dk, dv, P, 16, Hq, Hkv, D, scratch, out, nullptr);
    const std::vector<float> b16 = download(out, size_t(16) * Hq * D);
    bsame &= std::memcmp(b16.data(), full.data(), b16.size() * sizeof(float)) == 0;
    for (int m : {0, 13, 31}) {
      ling::kernels::attention_rows(dq + size_t(m) * Hq * D, dk, dv, P + m, 1, Hq, Hkv, D, scratch, out, nullptr);
      const std::vector<float> one = download(out, size_t(Hq) * D);
      bsame &= std::memcmp(one.data(), full.data() + size_t(m) * Hq * D, one.size() * sizeof(float)) == 0;
    }
    ling::kernels::set_attention_bulk(false);
    ok &= bsame;
    std::printf("attention_rows bulk-copy tiles identical to cp.async: %s\n", bsame ? "yes ok" : "NO FAIL");
  }
  // Speed at the workload's median context, 16 rows (one layer).
  {
    const int big = 24576 + 32;
    __nv_bfloat16 *bk, *bv;
    float* bs;
    cudaMalloc(&bk, size_t(big) * Hkv * D * 2);
    cudaMalloc(&bv, size_t(big) * Hkv * D * 2);
    // Random-looking data (zeros would let the memory system flatter the read).
    {
      std::vector<__nv_bfloat16> h(size_t(big) * Hkv * D);
      for (size_t i = 0; i < h.size(); ++i) h[i] = __float2bfloat16(normal(rng));
      cudaMemcpy(bk, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
      for (size_t i = 0; i < h.size(); ++i) h[i] = __float2bfloat16(normal(rng));
      cudaMemcpy(bv, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
    }
    cudaMalloc(&bs, ling::kernels::attention_rows_scratch_floats(MMAX, Hq, D, big) * sizeof(float));
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    // The L2 prefetch (head-major only) moves data, never arithmetic: bit for bit the same output.
    {
      ling::kernels::set_kv_layout(size_t(big) * D, D);
      ling::kernels::attention_rows(dq, bk, bv, 24576, 16, Hq, Hkv, D, bs, out, nullptr);
      const std::vector<float> plain = download(out, size_t(16) * Hq * D);
      ling::kernels::set_attention_prefetch(2);
      ling::kernels::attention_rows(dq, bk, bv, 24576, 16, Hq, Hkv, D, bs, out, nullptr);
      const std::vector<float> pref = download(out, size_t(16) * Hq * D);
      ling::kernels::set_attention_prefetch(0);
      ling::kernels::set_kv_layout(0, 0);
      const bool psame = std::memcmp(plain.data(), pref.data(), plain.size() * sizeof(float)) == 0;
      ok &= psame;
      std::printf("attention_rows with L2 prefetch identical (24K, head-major): %s\n", psame ? "yes ok" : "NO FAIL");
    }
    for (int layout = 0; layout < (std::getenv("LING_NO_TIMING") ? 0 : 3); ++layout) {
      if (layout >= 1) ling::kernels::set_kv_layout(size_t(big) * D, D);  // head-major, as the engine uses
      ling::kernels::set_attention_prefetch(layout == 2 ? 2 : 0);
      for (int bulk = 0; bulk < 2; ++bulk) {
        ling::kernels::set_attention_bulk(bulk == 1);
        for (int M : {1, 16}) {
          cudaEventRecord(a);
          for (int i = 0; i < 20; ++i) ling::kernels::attention_rows(dq, bk, bv, 24576, M, Hq, Hkv, D, bs, out, nullptr);
          cudaEventRecord(b);
          cudaEventSynchronize(b);
          float ms = 0;
          cudaEventElapsedTime(&ms, a, b);
          std::printf("attention_rows 24K context, %s KV, %s, M=%-2d: %.3f ms per layer (%.0f GB/s of KV)\n",
                      layout == 0 ? "interleaved " : layout == 1 ? "head-major  " : "head-major+L2 prefetch", bulk ? "bulk   " : "cp.async", M, ms / 20,
                      2.0 * 24576 * Hkv * D * 2 / (ms / 20 * 1e-3) / 1e9);
        }
      }
      ling::kernels::set_attention_bulk(false);
    }
    ling::kernels::set_attention_prefetch(0);
    ling::kernels::set_kv_layout(0, 0);
    cudaFree(bk), cudaFree(bv), cudaFree(bs);
  }
  cudaFree(dk), cudaFree(dv), cudaFree(dq), cudaFree(scratch), cudaFree(out);
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
    // Each copy is one tiled blob (values and, for NVFP4, the scales after them), the size stream_gemm reads.
    const size_t blob = wbytes + sbytes;
    uint8_t *w, *s = nullptr;
    cudaMalloc(&w, blob * copies);
    cudaMemset(w, 0x22, blob * copies);
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
          if (sh.fp4) ling::kernels::stream_gemm_nvfp4(xh, xinv, M, w + blob * c, nullptr, 1.f, y, sh.N, sh.K, nullptr);
          else ling::kernels::stream_gemm_fp8(xh, xinv, M, w + blob * c, 1.f, y, sh.N, sh.K, nullptr);
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
        if (sh.fp4) ling::kernels::gemv_nvfp4(x, 1, w + blob * c, s + sbytes * c, 1.f, y, sh.N, sh.K, nullptr);
        else ling::kernels::gemv_fp8(x, 1, w + blob * c, 1.f, y, sh.N, sh.K, nullptr);
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

__global__ void fill_random_kernel(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
    p[i] = h & 0x77777777u;  // NVFP4 nibbles and FP8 bytes of moderate size; scales stay finite
  }
}

// BACKLOG §19.24 (B08's trace): LING_BENCH_SMS=N runs the bench on a stream of a green context holding N SMs
// (cuDevSmResourceSplitByCount), so the groups' bandwidth is what a subset of the GPU reaches. Unset: the default
// stream, the whole GPU. Returns the stream and prints the SM count the split gave.
cudaStream_t bench_stream() {
  const char* e = std::getenv("LING_BENCH_SMS");
  if (!e || std::atoi(e) <= 0) return nullptr;
  cudaFree(nullptr);  // the primary context first, so the green context shares its device state
  CUdevice dev;
  CUdevResource all, part, rest;
  unsigned groups = 1;
  CUdevResourceDesc desc;
  CUgreenCtx g;
  CUstream st;
  auto ok = [](CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
      const char* m = nullptr;
      cuGetErrorString(r, &m);
      std::fprintf(stderr, "green context: %s failed: %s\n", what, m ? m : "?");
      std::exit(2);
    }
  };
  ok(cuDeviceGet(&dev, 0), "cuDeviceGet");
  ok(cuDeviceGetDevResource(dev, &all, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
  ok(cuDevSmResourceSplitByCount(&part, &groups, &all, &rest, 0, static_cast<unsigned>(std::atoi(e))),
     "cuDevSmResourceSplitByCount");
  ok(cuDevResourceGenerateDesc(&desc, &part, 1), "cuDevResourceGenerateDesc");
  ok(cuGreenCtxCreate(&g, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
  ok(cuGreenCtxStreamCreate(&st, g, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate");
  std::printf("green context: %u of %u SMs\n", part.sm.smCount, all.sm.smCount);
  return reinterpret_cast<cudaStream_t>(st);
}

// The step's weight stream, launch for launch as the engine groups it (B16): each group timed over enough
// copies that L2 never holds the next one, and the step's GEMM total from the per-step launch counts.
void bench_groups() {
  const cudaStream_t bs = bench_stream();
  struct Group {
    const char* name;
    bool fp4;
    int K, n, count;
    int N[3];
  };
  const Group groups[] = {
      {"attn q|k|v fp8", false, 5120, 3, 16, {12288, 1024, 1024}}, {"attn o fp8", false, 6144, 1, 16, {5120}},
      {"gdn in_qkv|z fp8", false, 5120, 2, 48, {10240, 6144}},     {"gdn out fp8", false, 6144, 1, 48, {5120}},
      {"ffn gate|up fp4", true, 5120, 2, 64, {17408, 17408}},       {"ffn down fp4", true, 17408, 1, 64, {5120}},
      {"lm head fp4", true, 5120, 1, 1, {248320}},
  };
  const int Ms[] = {1, 12, 16, 24, 32};
  double step_ms[5] = {}, step_gb = 0;
  // LING_BENCH_CODED=1: the NVFP4 groups as B26's coded blobs (random codes, a 64-entry table).
  const bool coded = std::getenv("LING_BENCH_CODED") && std::atoi(std::getenv("LING_BENCH_CODED")) != 0;
  uint8_t* dtab = nullptr;
  {
    std::vector<uint8_t> tab(64);
    for (int i = 0; i < 64; ++i) tab[i] = static_cast<uint8_t>(0x28 + i / 2);
    dtab = upload(tab);
  }
  for (const Group& gr : groups) {
    size_t bytes = 0, sizes[3];
    for (int i = 0; i < gr.n; ++i)
      bytes += sizes[i] = gr.fp4 && coded ? ling::kernels::tiled_bytes_nvfp4_coded(gr.N[i], gr.K)
                                          : size_t(gr.N[i]) * gr.K * (gr.fp4 ? 9 : 16) / 16;
    const int copies = int(std::max<size_t>(2, (1024u << 20) / bytes));
    uint8_t* w;
    cudaMalloc(&w, bytes * copies);
    fill_random_kernel<<<1024, 256>>>(reinterpret_cast<uint32_t*>(w), bytes * copies / 4, 7);
    float *x, *y[3], *xinv;
    __half* xh;
    cudaMalloc(&x, size_t(32) * gr.K * sizeof(float));
    fill_random_kernel<<<64, 256>>>(reinterpret_cast<uint32_t*>(x), size_t(32) * gr.K, 9);  // finite floats
    cudaMalloc(&xh, size_t(32) * gr.K * sizeof(__half));
    cudaMalloc(&xinv, 32 * sizeof(float));
    for (int i = 0; i < gr.n; ++i) cudaMalloc(&y[i], size_t(32) * gr.N[i] * sizeof(float));
    ling::kernels::to_half_rows(x, 32, gr.K, xh, xinv, nullptr);
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    std::printf("group %-17s %6.1f MB x%-2d", gr.name, bytes / 1e6, gr.count);
    step_gb += double(bytes) * gr.count / 1e9;
    for (int mi = 0; mi < 5; ++mi) {
      const int M = Ms[mi];
      const int iters = std::max(40, copies * 4);
      float best = 1e30f;
      for (int pass = 0; pass < 3; ++pass) {
        // Host-timed around a synchronized queue (an event of the primary context cannot be recorded on a green
        // context's stream); a launch costs microseconds against the group's ~0.1-3 ms.
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
          const uint8_t* base = w + bytes * (it % copies);
          ling::kernels::StreamTarget t[3];
          size_t off = 0;
          for (int i = 0; i < gr.n; ++i)
            t[i] = {base + off, gr.fp4 && coded ? dtab : nullptr, 1.f, y[i], gr.N[i]}, off += sizes[i];
          ling::kernels::stream_gemm_multi(gr.fp4, xh, xinv, M, t, gr.n, gr.K, bs);
        }
        if (bs) cudaStreamSynchronize(bs);
        cudaDeviceSynchronize();
        const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          std::fprintf(stderr, "bench: %s\n", cudaGetErrorString(err));
          std::exit(2);
        }
        if (pass > 0) best = std::min(best, ms / iters);
      }
      step_ms[mi] += double(best) * gr.count;
      std::printf("  M=%-2d %4.0f", M, bytes / (best * 1e-3) / 1e9);
    }
    std::printf("  GB/s\n");
    cudaFree(w), cudaFree(x), cudaFree(xh), cudaFree(xinv);
    for (int i = 0; i < gr.n; ++i) cudaFree(y[i]);
  }
  std::printf("step GEMMs (%.2f GB):", step_gb);
  for (int mi = 0; mi < 5; ++mi) std::printf("  M=%-2d %6.2f ms %4.0f GB/s", Ms[mi], step_ms[mi], step_gb / step_ms[mi] * 1e3);
  std::printf("\n");
}


// BACKLOG 33: the drafter's NVFP4 copies, rounded at load by ling::quantize_nvfp4.
//   - a hand-made block: every halfway value takes the smaller magnitude, the sign bit only on a non-zero code, the
//     packing (first code in the low nibble), the E4M3 scale, and an all-zero block's scale clamped to 2^-9;
//   - a Gaussian matrix's relative squared error (the offline copies of the drafter measured 0.0090);
//   - the copy through upload_nvfp4 and the rows path (to_half_rows or bf16_to_half_rows, then stream_gemm_nvfp4)
//     against a host GEMM on the dequantized copy (as tight as test_gemm) and on the BF16 matrix (bounded by the
//     rounding), every row bit for bit the same at every M; bf16_to_half_rows equal to to_half_rows on the same values.
bool test_nvfp4_copy() {
  bool ok = true;
  {
    // amax 2688 = 448 * 6: the global scale 1 and the first block's scale 448 exactly, so x / 448 lands on the bounds.
    const float row[32] = {2688, 112, 336, 560, 784, 1120, 1568, 2240, 113, -336, -10, 0, -2688, 338, -1568, 448,
                           0,    0,   0,   0,   0,   0,    0,    0,    0,   0,    0,   0, 0,     0,   0,     0};
    const uint8_t want[16] = {7, 0, 1, 2, 3, 4, 5, 6, 1, 9, 0, 0, 15, 2, 13, 2};
    std::vector<__nv_bfloat16> w(32);
    for (int i = 0; i < 32; ++i) w[i] = __float2bfloat16(row[i]);
    const ling::Nvfp4Host q = ling::quantize_nvfp4(w.data(), 1, 32);
    bool good = q.scale2 == 1.f && q.scales.size() == 2 && q.scales[0] == 0x7e && q.scales[1] == 0x01;
    for (int i = 0; i < 16; ++i) good &= ((q.values[i / 2] >> (4 * (i & 1))) & 15) == want[i];
    for (int i = 8; i < 16; ++i) good &= q.values[i] == 0;
    ok &= good;
    std::printf("nvfp4 copy: hand-made block (ties to the smaller code, sign, packing, scales) %s\n", good ? "ok" : "FAIL");
  }
  std::mt19937 rng(33);
  std::normal_distribution<float> normal(0.f, 1.f);
  {
    const int N = 256, K = 1024;
    std::vector<__nv_bfloat16> w(size_t(N) * K);
    for (auto& v : w) v = __float2bfloat16(normal(rng) * 0.02f);
    const ling::Nvfp4Host q = ling::quantize_nvfp4(w.data(), N, K);
    const bool good = q.rel_sq_error > 0.007 && q.rel_sq_error < 0.011;
    ok &= good;
    std::printf("nvfp4 copy: Gaussian matrix, relative squared error %.5f %s\n", q.rel_sq_error, good ? "ok" : "FAIL");
  }
  // The conv kernel projection's shape (FP32 rows in) and a narrow slice of the fc's (BF16 rows in, K = 5 x 5120).
  for (auto [N, K, bf16_in] : {std::tuple{1280, 5120, false}, std::tuple{96, 25600, true}}) {
    const int MMAX = ling::kernels::kMaxStreamRows;
    std::vector<__nv_bfloat16> w(size_t(N) * K);
    for (auto& v : w) v = __float2bfloat16(normal(rng) * 0.02f);
    const ling::Nvfp4Host q = ling::quantize_nvfp4(w.data(), N, K);
    std::vector<float> wq(w.size()), wb(w.size());  // the copy dequantized, and the BF16 matrix
    for (size_t i = 0; i < w.size(); ++i) {
      const uint8_t c = q.values[i / 2];
      wq[i] = kFp4[(i & 1) ? c >> 4 : c & 15] * fp8(q.scales[i / 16]) * q.scale2;
      wb[i] = __bfloat162float(w[i]);
    }
    ling::TensorView tv, ts;
    tv.dtype = "U8", tv.shape = {N, K / 2}, tv.data = q.values.data(), tv.bytes = q.values.size();
    ts.dtype = "F8_E4M3", ts.shape = {N, K / 16}, ts.data = q.scales.data(), ts.bytes = q.scales.size();
    size_t bytes = 0;
    std::vector<void*> owned;
    const ling::Fp4Upload u = ling::upload_nvfp4(tv, ts, N, K, &bytes, owned);
    std::vector<float> x(size_t(MMAX) * K);
    std::vector<__nv_bfloat16> xb(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      xb[i] = __float2bfloat16(normal(rng) * (i % 97 == 0 ? 30.f : 1.f));
      x[i] = __bfloat162float(xb[i]);
    }
    float* dx = upload(x);
    __nv_bfloat16* dxb = upload(xb);
    __half *xh, *xh2;
    float *xinv, *xinv2, *dy;
    cudaMalloc(&xh, x.size() * sizeof(__half));
    cudaMalloc(&xh2, x.size() * sizeof(__half));
    cudaMalloc(&xinv, MMAX * sizeof(float));
    cudaMalloc(&xinv2, MMAX * sizeof(float));
    cudaMalloc(&dy, size_t(MMAX) * N * sizeof(float));
    auto gemm = [&](int M) {
      if (bf16_in) ling::kernels::bf16_to_half_rows(dxb, M, K, xh, xinv, nullptr);
      else ling::kernels::to_half_rows(dx, M, K, xh, xinv, nullptr);
      ling::kernels::stream_gemm_nvfp4(xh, xinv, M, u.w, u.table, q.scale2, dy, N, K, nullptr);
      return download(dy, size_t(M) * N);
    };
    if (bf16_in) {
      ling::kernels::bf16_to_half_rows(dxb, MMAX, K, xh, xinv, nullptr);
      ling::kernels::to_half_rows(dx, MMAX, K, xh2, xinv2, nullptr);
      const std::vector<__half> a = download(xh, x.size()), b = download(xh2, x.size());
      const bool same = std::memcmp(a.data(), b.data(), a.size() * sizeof(__half)) == 0 &&
                        download(xinv, MMAX) == download(xinv2, MMAX);
      ok &= same;
      std::printf("bf16_to_half_rows equals to_half_rows on the same values: %s\n", same ? "yes ok" : "NO FAIL");
    }
    const std::vector<float> full = gemm(MMAX);
    for (int M : {1, 7, 12, 16, 32}) {
      const std::vector<float> got = gemm(M);
      double worst = 0, e2 = 0, r2 = 0;
      for (int m = 0; m < M; ++m) {
        std::vector<double> ref(N), bf(N);
        double rms = 0;
        for (int n = 0; n < N; ++n) {
          double a = 0, b = 0;
          const float *pq = &wq[size_t(n) * K], *pb = &wb[size_t(n) * K], *px = &x[size_t(m) * K];
          for (int k = 0; k < K; ++k) a += double(pq[k]) * px[k], b += double(pb[k]) * px[k];
          ref[n] = a, bf[n] = b, rms += a * a;
        }
        rms = std::sqrt(rms / N);
        for (int n = 0; n < N; ++n) {
          const double y = got[size_t(m) * N + n];
          worst = std::max(worst, std::abs(y - ref[n]) / rms);
          e2 += (y - bf[n]) * (y - bf[n]), r2 += bf[n] * bf[n];
        }
      }
      const double vs_bf16 = std::sqrt(e2 / r2);
      const bool same = std::memcmp(got.data(), full.data(), got.size() * sizeof(float)) == 0;
      const bool good = worst < 2e-3 && vs_bf16 < 0.15 && same;
      ok &= good;
      std::printf("nvfp4 copy %dx%d %s M=%-2d err/rms vs the copy %.2e, vs BF16 %.3f, rows identical to M=32: %s %s\n", N,
                  K, bf16_in ? "BF16 rows" : "FP32 rows", M, worst, vs_bf16, same ? "yes" : "NO", good ? "ok" : "FAIL");
    }
    for (void* p : owned) cudaFree(p);
    cudaFree(dx), cudaFree(dxb), cudaFree(xh), cudaFree(xh2), cudaFree(xinv), cudaFree(xinv2), cudaFree(dy);
  }
  return ok;
}

// --nvfp4-oracle DIR: every "<name>.nvfp4.weight" in DIR's checkpoint (bench/quant/drafter_variants.py --nvfp4-copy,
// the copies BACKLOG 33 measured) against quantize_nvfp4 of "<name>.weight", byte for byte.
bool nvfp4_oracle(const std::string& dir) {
  ling::Checkpoint ck(dir);
  bool ok = true;
  int n = 0;
  std::vector<std::string> names = {"fc"};
  for (int l = 0; l < 64; ++l)
    for (const char* k : {"attention_conv.kernel_projection", "mlp_conv.kernel_projection"})
      names.push_back("layers." + std::to_string(l) + "." + k);
  for (const std::string& b : names) {
    if (!ck.has(b + ".nvfp4.weight")) continue;
    const ling::TensorView& t = ck.get(b + ".weight");
    const ling::Nvfp4Host q =
        ling::quantize_nvfp4(reinterpret_cast<const __nv_bfloat16*>(t.data), int(t.shape[0]), int(t.shape[1]));
    const ling::TensorView &v = ck.get(b + ".nvfp4.weight"), &s = ck.get(b + ".nvfp4.weight_scale"),
                           &g = ck.get(b + ".nvfp4.weight_scale_2");
    const bool same = v.bytes == q.values.size() && std::memcmp(v.data, q.values.data(), v.bytes) == 0 &&
                      s.bytes == q.scales.size() && std::memcmp(s.data, q.scales.data(), s.bytes) == 0 &&
                      g.bytes == 4 && std::memcmp(g.data, &q.scale2, 4) == 0;
    std::printf("%s: %s (relative squared error %.5f)\n", b.c_str(), same ? "identical" : "DIFFERS", q.rel_sq_error);
    ok &= same;
    ++n;
  }
  ok &= n > 0;
  std::printf(ok ? "nvfp4 oracle: %d copies identical\n" : "NVFP4 ORACLE FAILED (%d copies)\n", n);
  return ok;
}

}  // namespace

// M4 stage 4: each state kernel over two sequences in one launch (kernels::SeqBatch) against its launch per
// sequence, bit for bit: attn_prepare (q, gate, and the KV written into each sequence's caches), attention_rows
// (eager, and with device positions over fixed ranges as a graph launches it), the verify's gdn_conv and
// gdn_recurrent, and the drafter's draft_qk_rope and draft_attention over each sequence's ring.
bool test_batched_state_kernels() {
  namespace k = ling::kernels;
  std::mt19937 rng(29);
  std::normal_distribution<float> normal(0.f, 1.f);
  auto randv = [&](size_t n, float scale = 1.f) {
    std::vector<float> v(n);
    for (auto& x : v) x = normal(rng) * scale;
    return v;
  };
  auto randb = [&](size_t n, float scale = 1.f) {
    std::vector<__nv_bfloat16> v(n);
    for (auto& x : v) x = __float2bfloat16(normal(rng) * scale);
    return v;
  };
  auto same = [](const void* a, const void* b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; };
  bool ok = true;
  const int rows[2] = {12, 16}, pos[2] = {5000, 2900};  // the second sequence shorter and with more rows
  const int M = rows[0] + rows[1];
  k::SeqBatch sb;
  sb.n = 2;
  sb.row0[1] = rows[0];
  sb.row0[2] = M;
  sb.pos[0] = pos[0];
  sb.pos[1] = pos[1];
  int* dpos = upload(std::vector<int>{pos[0], pos[1]});
  // ---- attn_prepare and attention_rows ----
  {
    const int Hq = 24, Hkv = 4, D = 256, ctx = 8192, rot = 64;
    const size_t cache = size_t(ctx) * Hkv * D;
    float* qg = upload(randv(size_t(M) * Hq * 2 * D));
    float* kk = upload(randv(size_t(M) * Hkv * D));
    float* vv = upload(randv(size_t(M) * Hkv * D));
    __nv_bfloat16* qn = upload(randb(D, 0.1f));
    __nv_bfloat16* kn = upload(randb(D, 0.1f));
    const std::vector<__nv_bfloat16> c0 = randb(cache), c1 = randb(cache), c2 = randb(cache), c3 = randb(cache);
    // Per sequence: caches A (k0, v0) and B (k1, v1); the batched launch writes copies of them.
    __nv_bfloat16 *ka = upload(c0), *va = upload(c1), *kb = upload(c2), *vb = upload(c3);
    __nv_bfloat16 *ka2 = upload(c0), *va2 = upload(c1), *kb2 = upload(c2), *vb2 = upload(c3);
    float *q1, *g1, *q2, *g2;
    cudaMalloc(&q1, size_t(M) * Hq * D * 4), cudaMalloc(&g1, size_t(M) * Hq * D * 4);
    cudaMalloc(&q2, size_t(M) * Hq * D * 4), cudaMalloc(&g2, size_t(M) * Hq * D * 4);
    for (int s = 0; s < 2; ++s) {
      const size_t r0 = sb.row0[s];
      k::attn_prepare(qg + r0 * Hq * 2 * D, kk + r0 * Hkv * D, vv + r0 * Hkv * D, qn, kn, pos[s], rows[s], Hq, Hkv, D, rot,
                      1e7f, 1e-6f, q1 + r0 * Hq * D, g1 + r0 * Hq * D, s ? kb : ka, s ? vb : va, nullptr);
    }
    k::SeqPtrs<__nv_bfloat16> kc, vc;
    kc.p[0] = ka2, kc.p[1] = kb2, vc.p[0] = va2, vc.p[1] = vb2;
    k::attn_prepare_seqs(qg, kk, vv, qn, kn, sb, Hq, Hkv, D, rot, 1e7f, 1e-6f, q2, g2, kc, vc, nullptr);
    const size_t qbytes = size_t(M) * Hq * D * 4, cbytes = cache * 2;
    bool prep = same(download(q1, qbytes / 4).data(), download(q2, qbytes / 4).data(), qbytes) &&
                same(download(g1, qbytes / 4).data(), download(g2, qbytes / 4).data(), qbytes);
    for (auto [a, b] : {std::pair{ka, ka2}, {va, va2}, {kb, kb2}, {vb, vb2}})
      prep &= same(download(a, cache).data(), download(b, cache).data(), cbytes);
    ok &= prep;
    std::printf("attn_prepare over two sequences in one launch identical to per sequence: %s\n", prep ? "yes ok" : "NO FAIL");
    // attention_rows: per sequence against one launch, eager and as a graph launches it.
    float *scratch, *o1, *o2;
    cudaMalloc(&scratch, k::attention_rows_scratch_floats(32, Hq, D, ctx) * sizeof(float));
    cudaMalloc(&o1, qbytes), cudaMalloc(&o2, qbytes);
    for (int s = 0; s < 2; ++s) {
      const size_t r0 = sb.row0[s];
      k::attention_rows(q1 + r0 * Hq * D, s ? kb : ka, s ? vb : va, pos[s], rows[s], Hq, Hkv, D, scratch,
                        o1 + r0 * Hq * D, nullptr);
    }
    k::SeqPtrs<const __nv_bfloat16> kr, vr;
    kr.p[0] = ka, kr.p[1] = kb, vr.p[0] = va, vr.p[1] = vb;
    k::attention_rows_seqs(q1, kr, vr, sb, Hq, Hkv, D, scratch, o2, nullptr);
    const std::vector<float> ref = download(o1, qbytes / 4);
    bool att = same(ref.data(), download(o2, qbytes / 4).data(), qbytes);
    k::SeqBatch gb = sb;
    gb.pos[0] = gb.pos[1] = -1;  // deliberately wrong: the graph form reads the device's
    gb.pos_dev = dpos;
    cudaMemset(o2, 0, qbytes);
    k::attention_rows_seqs(q1, kr, vr, gb, Hq, Hkv, D, scratch, o2, nullptr, ctx);
    const bool attg = same(ref.data(), download(o2, qbytes / 4).data(), qbytes);
    ok &= att && attg;
    std::printf("attention_rows over two sequences in one launch identical to per sequence: %s; with device positions "
                "and fixed ranges: %s\n", att ? "yes ok" : "NO FAIL", attg ? "yes ok" : "NO FAIL");
    for (void* p : {(void*)qg, (void*)kk, (void*)vv, (void*)qn, (void*)kn, (void*)ka, (void*)va, (void*)kb, (void*)vb,
                    (void*)ka2, (void*)va2, (void*)kb2, (void*)vb2, (void*)q1, (void*)g1, (void*)q2, (void*)g2,
                    (void*)scratch, (void*)o1, (void*)o2})
      cudaFree(p);
  }
  // ---- the verify's gdn_conv and gdn_recurrent ----
  {
    const int H = 16, HV = 48, C = 2 * H * 128 + HV * 128;
    const std::vector<float> mixed = randv(size_t(M) * C);
    float *m1 = upload(mixed), *m2 = upload(mixed);
    __nv_bfloat16* w = upload(randb(size_t(C) * 4, 0.5f));
    const std::vector<float> s0 = randv(size_t(C) * 3), s1 = randv(size_t(C) * 3);
    float *st0 = upload(s0), *st1 = upload(s1), *tmp = upload(s0);
    for (int s = 0; s < 2; ++s) {
      cudaMemcpy(tmp, s ? st1 : st0, size_t(C) * 3 * 4, cudaMemcpyDeviceToDevice);  // the verify's copy of the window
      k::gdn_conv(m1 + size_t(sb.row0[s]) * C, tmp, w, rows[s], C, nullptr);
    }
    k::SeqPtrs<const float> cs;
    cs.p[0] = st0, cs.p[1] = st1;
    k::gdn_conv_seqs(m2, cs, w, sb, C, nullptr);
    const bool conv = same(download(m1, size_t(M) * C).data(), download(m2, size_t(M) * C).data(), size_t(M) * C * 4) &&
                      same(download(st0, size_t(C) * 3).data(), s0.data(), size_t(C) * 3 * 4) &&
                      same(download(st1, size_t(C) * 3).data(), s1.data(), size_t(C) * 3 * 4);
    ok &= conv;
    std::printf("gdn_conv over two sequences in one launch identical to per sequence (windows unwritten): %s\n",
                conv ? "yes ok" : "NO FAIL");
    // The recurrence over the conv's output, with decays and strengths in range.
    std::vector<float> g = randv(size_t(M) * HV, 0.3f), beta(size_t(M) * HV);
    for (auto& x : g) x = -std::abs(x);
    for (auto& x : beta) x = 1.f / (1.f + std::exp(-normal(rng)));
    float *dg = upload(g), *db = upload(beta);
    const size_t SS = size_t(HV) * 128 * 128;
    const std::vector<float> r0 = randv(SS, 0.05f), r1 = randv(SS, 0.05f);
    float *rs0 = upload(r0), *rs1 = upload(r1), *out1, *out2;
    cudaMalloc(&out1, size_t(M) * HV * 128 * 4), cudaMalloc(&out2, size_t(M) * HV * 128 * 4);
    for (int s = 0; s < 2; ++s) {
      const size_t a = sb.row0[s];
      k::gdn_recurrent(m1 + a * C, dg + a * HV, db + a * HV, s ? rs1 : rs0, nullptr, out1 + a * HV * 128, rows[s], H, HV,
                       nullptr);
    }
    k::SeqPtrs<const float> rs;
    rs.p[0] = rs0, rs.p[1] = rs1;
    k::gdn_recurrent_seqs(m1, dg, db, rs, out2, sb, H, HV, nullptr);
    const bool rec = same(download(out1, size_t(M) * HV * 128).data(), download(out2, size_t(M) * HV * 128).data(),
                          size_t(M) * HV * 128 * 4) &&
                     same(download(rs0, SS).data(), r0.data(), SS * 4) && same(download(rs1, SS).data(), r1.data(), SS * 4);
    ok &= rec;
    std::printf("gdn_recurrent over two sequences in one launch identical to per sequence (states unwritten): %s\n",
                rec ? "yes ok" : "NO FAIL");
    for (void* p : {(void*)m1, (void*)m2, (void*)w, (void*)st0, (void*)st1, (void*)tmp, (void*)dg, (void*)db, (void*)rs0,
                    (void*)rs1, (void*)out1, (void*)out2})
      cudaFree(p);
  }
  // ---- the drafter: draft_qk_rope and draft_attention over two blocks ----
  {
    const int Hkv = 8, Hq = 32, dd = 128, ring = 8224, window = 2048;
    const int B = 12;  // two full blocks
    k::SeqBatch db;
    db.n = 2;
    db.row0[1] = B;
    db.row0[2] = 2 * B;
    db.pos[0] = pos[0];
    db.pos[1] = 9000;  // past the ring: positions wrap
    const int R = 2 * B;
    const std::vector<float> xq = randv(size_t(R) * Hq * dd);
    float *x1 = upload(xq), *x2 = upload(xq);
    __nv_bfloat16* nw = upload(randb(dd, 0.1f));
    for (int s = 0; s < 2; ++s)
      k::draft_qk_rope(x1 + size_t(s) * B * Hq * dd, B, Hq, Hq * dd, nw, db.pos[s], 1e6f, 1e-6f, nullptr);
    k::draft_qk_rope_seqs(x2, Hq, Hq * dd, nw, db, 1e6f, 1e-6f, nullptr);
    const bool rope = same(download(x1, size_t(R) * Hq * dd).data(), download(x2, size_t(R) * Hq * dd).data(),
                           size_t(R) * Hq * dd * 4);
    float *kb = upload(randv(size_t(R) * Hkv * dd)), *vb = upload(randv(size_t(R) * Hkv * dd));
    __nv_bfloat16 *r0k = upload(randb(size_t(ring) * Hkv * dd)), *r0v = upload(randb(size_t(ring) * Hkv * dd));
    __nv_bfloat16 *r1k = upload(randb(size_t(ring) * Hkv * dd)), *r1v = upload(randb(size_t(ring) * Hkv * dd));
    float *a1, *a2;
    cudaMalloc(&a1, size_t(R) * Hq * dd * 4), cudaMalloc(&a2, size_t(R) * Hq * dd * 4);
    for (int s = 0; s < 2; ++s) {
      const size_t r0 = size_t(s) * B;
      k::draft_attention(x1 + r0 * Hq * dd, s ? r1k : r0k, s ? r1v : r0v, ring, kb + r0 * Hkv * dd, vb + r0 * Hkv * dd,
                         db.pos[s], B, window - 1, Hq, Hkv, a1 + r0 * Hq * dd, nullptr);
    }
    k::SeqPtrs<const __nv_bfloat16> rk, rv;
    rk.p[0] = r0k, rk.p[1] = r1k, rv.p[0] = r0v, rv.p[1] = r1v;
    k::draft_attention_seqs(x1, rk, rv, ring, kb, vb, db, window - 1, Hq, Hkv, a2, nullptr);
    const bool datt = same(download(a1, size_t(R) * Hq * dd).data(), download(a2, size_t(R) * Hq * dd).data(),
                           size_t(R) * Hq * dd * 4);
    ok &= rope && datt;
    std::printf("draft_qk_rope and draft_attention over two blocks in one launch identical to per sequence: %s, %s\n",
                rope ? "yes ok" : "NO FAIL", datt ? "yes ok" : "NO FAIL");
    for (void* p : {(void*)x1, (void*)x2, (void*)nw, (void*)kb, (void*)vb, (void*)r0k, (void*)r0v, (void*)r1k, (void*)r1v,
                    (void*)a1, (void*)a2})
      cudaFree(p);
  }
  cudaFree(dpos);
  return ok;
}

int main(int argc, char** argv) {
  if (const char* e = std::getenv("LING_PDL")) ling::kernels::set_pdl(std::atoi(e) != 0);
  if (argc > 2 && std::string(argv[1]) == "--nvfp4-oracle") return nvfp4_oracle(argv[2]) ? 0 : 1;
  if (argc > 1 && std::string(argv[1]) == "--bench-groups") {  // timings only (B16 sweeps)
    bench_groups();
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
  }
  bool ok = test_gemm();
  ok &= test_gemm_wide();
  ok &= test_bf16_rows();
  ok &= test_attention_rows();
  ok &= test_topk();
  ok &= test_batched_state_kernels();
  ok &= test_nvfp4_copy();
  if (argc > 1 && std::string(argv[1]) == "--bench") bench();
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf(ok ? "all stream tests passed\n" : "STREAM TESTS FAILED\n");
  return ok ? 0 : 1;
}
