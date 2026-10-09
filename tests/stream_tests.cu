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
  // Several matrices in one launch give exactly the separate launches' results.
  for (bool fp4 : {true, false}) {
    const int M = 16;
    float *ya, *yb;
    cudaMalloc(&ya, size_t(M) * N * sizeof(float));
    cudaMalloc(&yb, size_t(M) * (N - 144) * sizeof(float));
    ling::kernels::to_half_rows(dx, M, K, xh, xinv, nullptr);
    // The second target: the first N - 144 rows of the same weights with another scale.
    const ling::kernels::StreamTarget t[2] = {{fp4 ? dw4 : dw8, fp4 ? ds4 : nullptr, 0.5f, ya, N},
                                              {fp4 ? dw4 : dw8, fp4 ? ds4 : nullptr, 0.25f, yb, N - 144}};
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
  // Speed at the workload's median context, 16 rows (one layer).
  {
    const int big = 24576 + 32;
    __nv_bfloat16 *bk, *bv;
    float* bs;
    cudaMalloc(&bk, size_t(big) * Hkv * D * 2);
    cudaMalloc(&bv, size_t(big) * Hkv * D * 2);
    cudaMemset(bk, 0, size_t(big) * Hkv * D * 2);
    cudaMemset(bv, 0, size_t(big) * Hkv * D * 2);
    cudaMalloc(&bs, ling::kernels::attention_rows_scratch_floats(MMAX, Hq, D, big) * sizeof(float));
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int M : {1, 16}) {
      cudaEventRecord(a);
      for (int i = 0; i < 20; ++i) ling::kernels::attention_rows(dq, bk, bv, 24576, M, Hq, Hkv, D, bs, out, nullptr);
      cudaEventRecord(b);
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      std::printf("attention_rows 24K context, M=%-2d: %.3f ms per layer (%.0f GB/s of KV)\n", M, ms / 20,
                  2.0 * 24576 * Hkv * D * 2 / (ms / 20 * 1e-3) / 1e9);
    }
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

}  // namespace

int main(int argc, char** argv) {
  bool ok = test_gemm();
  ok &= test_bf16_rows();
  ok &= test_attention_rows();
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
