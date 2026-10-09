// Checks the prefill path's kernels against host references on random data:
//   - activation quantization (NVFP4 and FP8) against the recipe written out on the host, byte for byte;
//   - the block-scaled GEMMs against a double-precision product of the same quantized operands;
// and, with --bench, times the GEMMs at the model's shapes.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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

float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }

// E2M1 round-to-nearest-even with saturation, as cvt.rn.satfinite.
uint8_t e2m1(float x) {
  const float a = std::fabs(x);
  int best = 0;
  for (int i = 1; i < 8; ++i) {
    const float d = std::fabs(kFp4[i] - a), db = std::fabs(kFp4[best] - a);
    if (d < db || (d == db && (i % 2 == 0))) best = i;  // ties to the even mantissa
  }
  if (a > 6.f) best = 7;
  return static_cast<uint8_t>(best | (std::signbit(x) ? 8 : 0));  // negative values that round to 0 keep the sign
}

bool test_quant(std::mt19937& rng) {
  const int M = 67, K = 512;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(size_t(M) * K);
  for (size_t i = 0; i < x.size(); ++i) x[i] = nd(rng) * (i % 97 == 0 ? 40.f : 1.f) * 0.003f;
  const float in4 = 0.0019763766f, in8 = 0.2053571f;
  float* dx = upload(x);
  uint8_t *q4, *s4, *q8;
  cudaMalloc(&q4, size_t(M) * K / 2);
  cudaMalloc(&s4, size_t(M) * K / 16);
  cudaMalloc(&q8, size_t(M) * K);
  ling::kernels::quant_act_nvfp4(dx, M, K, in4, q4, s4, nullptr);
  ling::kernels::quant_act_fp8(dx, M, K, in8, q8, nullptr);
  const auto hq4 = download(q4, size_t(M) * K / 2), hs4 = download(s4, size_t(M) * K / 16), hq8 = download(q8, size_t(M) * K);
  size_t bad4 = 0, bads = 0, bad8 = 0;
  const float g = 1.f / in4;
  for (size_t b = 0; b < x.size() / 16; ++b) {
    float v[16], amax = 0;
    for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(v[i] = bf16_round(x[b * 16 + i])));
    const __nv_fp8_e4m3 s(g * (amax * (1.f / 6.f)));
    bads += s.__x != hs4[b];
    const float sv = static_cast<float>(s), os = sv != 0.f ? 1.f / (sv * (1.f / g)) : 0.f;
    for (int i = 0; i < 16; i += 2) {
      const uint8_t want = e2m1(v[i] * os) | (e2m1(v[i + 1] * os) << 4);
      bad4 += want != hq4[b * 8 + i / 2];
    }
  }
  for (size_t i = 0; i < x.size(); ++i)
    bad8 += __nv_cvt_float_to_fp8(bf16_round(x[i]) * (1.f / in8), __NV_SATFINITE, __NV_E4M3) != hq8[i];
  // The fused silu(g) * u quantization equals silu_mul followed by the plain quantization, byte for byte.
  std::vector<float> u(x.size());
  for (auto& v : u) v = nd(rng);
  float* du = upload(u);
  float* dm;
  cudaMalloc(&dm, x.size() * sizeof(float));
  ling::kernels::silu_mul(dx, du, dm, M * K, nullptr);
  ling::kernels::quant_act_nvfp4(dm, M, K, in4, q4, s4, nullptr);
  const auto rq = download(q4, size_t(M) * K / 2), rs = download(s4, size_t(M) * K / 16);
  ling::kernels::silu_mul_quant_nvfp4(dx, du, M, K, in4, q4, s4, nullptr);
  const bool fused = download(q4, size_t(M) * K / 2) == rq && download(s4, size_t(M) * K / 16) == rs;
  std::printf("fused silu_mul + NVFP4 quantization %s\n", fused ? "identical ok" : "DIFFERS FAIL");
  cudaFree(du);
  cudaFree(dm);
  const bool ok = bad4 == 0 && bads == 0 && bad8 == 0 && fused;
  std::printf("activation quantization: NVFP4 %zu bad bytes, %zu bad scales; FP8 %zu bad bytes %s\n", bad4, bads, bad8,
              ok ? "ok" : "FAIL");
  cudaFree(dx);
  cudaFree(q4);
  cudaFree(s4);
  cudaFree(q8);
  return ok;
}

struct Weights {
  std::vector<uint8_t> w, s;  // checkpoint layout: NVFP4 [N][K/2] + [N][K/16], or FP8 [N][K]
  uint8_t* tiled = nullptr;
};

Weights make_weights(std::mt19937& rng, bool fp4, int N, int K) {
  std::uniform_int_distribution<int> byte(0, 255), sbyte(0x28, 0x40);
  Weights W;
  W.w.resize(fp4 ? size_t(N) * K / 2 : size_t(N) * K);
  for (auto& b : W.w) {
    do b = static_cast<uint8_t>(byte(rng));
    while (!fp4 && (b & 0x7f) == 0x7f);
  }
  if (fp4) {
    W.s.resize(size_t(N) * K / 16);
    for (auto& b : W.s) b = static_cast<uint8_t>(sbyte(rng));
  }
  uint8_t* dw = upload(W.w);
  uint8_t* ds = fp4 ? upload(W.s) : nullptr;
  cudaMalloc(&W.tiled, fp4 ? ling::kernels::tiled_bytes_nvfp4(N, K) : ling::kernels::tiled_bytes_fp8(N, K));
  if (fp4) ling::kernels::retile_nvfp4(dw, ds, W.tiled, N, K, nullptr);
  else ling::kernels::retile_fp8(dw, W.tiled, N, K, nullptr);
  cudaDeviceSynchronize();
  cudaFree(dw);
  if (ds) cudaFree(ds);
  return W;
}

bool test_gemm(std::mt19937& rng, bool fp4, int M, int N, int K, bool accumulate) {
  Weights W = make_weights(rng, fp4, N, K);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(size_t(M) * K);
  for (auto& v : x) v = nd(rng);
  const float in_scale = fp4 ? 0.05f : 0.02f, alpha = 0.37f;
  float* dx = upload(x);
  uint8_t *q, *sf = nullptr;
  cudaMalloc(&q, fp4 ? size_t(M) * K / 2 : size_t(M) * K);
  if (fp4) cudaMalloc(&sf, size_t(M) * K / 16);
  if (fp4) ling::kernels::quant_act_nvfp4(dx, M, K, in_scale, q, sf, nullptr);
  else ling::kernels::quant_act_fp8(dx, M, K, in_scale, q, nullptr);
  const int ldy = N + 128;  // a wider output row: the GEMM writes columns [0, N) of each
  std::vector<float> y0(size_t(M) * ldy);
  for (auto& v : y0) v = nd(rng);
  float* dy = upload(y0);
  if (fp4) ling::kernels::prefill_gemm_nvfp4(q, sf, M, W.tiled, alpha, dy, ldy, N, K, accumulate, nullptr);
  else ling::kernels::prefill_gemm_fp8(q, M, W.tiled, alpha, dy, ldy, N, K, accumulate, nullptr);
  const cudaError_t e = cudaDeviceSynchronize();
  const auto y = download(dy, y0.size());
  const auto hq = download(q, fp4 ? size_t(M) * K / 2 : size_t(M) * K);
  const auto hs = fp4 ? download(sf, size_t(M) * K / 16) : std::vector<uint8_t>();
  // Dequantized operands, then the product in double.
  std::vector<double> xa(size_t(M) * K), wa(size_t(N) * K);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      const size_t i = size_t(m) * K + k;
      xa[i] = fp4 ? kFp4[(hq[i / 2] >> (4 * (k & 1))) & 15] * double(fp8(hs[i / 16])) : double(fp8(hq[i]));
    }
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const size_t i = size_t(n) * K + k;
      wa[i] = fp4 ? kFp4[(W.w[i / 2] >> (4 * (k & 1))) & 15] * double(fp8(W.s[i / 16])) : double(fp8(W.w[i]));
    }
  double worst = 0, rms = 0;
  size_t untouched = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < ldy; ++n) {
      const size_t i = size_t(m) * ldy + n;
      if (n >= N) {
        untouched += y[i] != y0[i];
        continue;
      }
      double ref = 0;
      for (int k = 0; k < K; ++k) ref += xa[size_t(m) * K + k] * wa[size_t(n) * K + k];
      ref = ref * alpha + (accumulate ? y0[i] : 0.0);
      rms += ref * ref;
      worst = std::max(worst, std::fabs(y[i] - ref));
    }
  rms = std::sqrt(rms / (double(M) * N));
  const bool ok = e == cudaSuccess && worst <= 1e-4 * rms + 1e-6 && untouched == 0;
  std::printf("prefill_gemm %s M=%-4d N=%-4d K=%-5d%s: max err %.2e of rms %.2e, %zu stray writes %s %s\n",
              fp4 ? "nvfp4" : "fp8  ", M, N, K, accumulate ? " +=" : "   ", worst, rms, untouched, cudaGetErrorString(e),
              ok ? "ok" : "FAIL");
  cudaFree(dx);
  cudaFree(q);
  if (sf) cudaFree(sf);
  cudaFree(dy);
  cudaFree(W.tiled);
  return ok;
}

// The prefill conv against the rows path's (then the same recurrence: close, the conv sums in another order),
// and against itself over a split prompt (bit for bit).
bool test_deltanet(std::mt19937& rng) {
  const int H = 16, HV = 48, C = 2 * H * 128 + HV * 128, M = 100, split = 37;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  std::vector<float> raw(size_t(M) * C), g(size_t(M) * HV), beta(size_t(M) * HV), conv0(size_t(C) * 3),
      state0(size_t(HV) * 128 * 128);
  std::vector<__nv_bfloat16> w(size_t(C) * 4);
  for (auto& v : raw) v = nd(rng);
  for (auto& v : g) v = -0.5f * ud(rng);
  for (auto& v : beta) v = ud(rng);
  for (auto& v : conv0) v = nd(rng);
  for (auto& v : state0) v = 0.1f * nd(rng);
  for (auto& v : w) v = __float2bfloat16(0.5f * nd(rng));
  float *draw = upload(raw), *dg = upload(g), *db = upload(beta);
  __nv_bfloat16* dw = upload(w);
  float *mixed, *conv, *state, *out;
  cudaMalloc(&mixed, raw.size() * 4);
  cudaMalloc(&conv, conv0.size() * 4);
  cudaMalloc(&state, state0.size() * 4);
  cudaMalloc(&out, size_t(M) * HV * 128 * 4);
  auto restore = [&] {
    cudaMemcpy(conv, conv0.data(), conv0.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(state, state0.data(), state0.size() * 4, cudaMemcpyHostToDevice);
  };
  // Rows-path reference.
  restore();
  cudaMemcpy(mixed, draw, raw.size() * 4, cudaMemcpyDeviceToDevice);
  ling::kernels::gdn_conv(mixed, conv, dw, M, C, nullptr);
  ling::kernels::gdn_recurrent(mixed, dg, db, state, state, out, M, H, HV, nullptr);
  const auto ref_out = download(out, size_t(M) * HV * 128), ref_state = download(state, state0.size()),
             ref_conv = download(conv, conv0.size());
  // Prefill path, whole.
  restore();
  ling::kernels::gdn_conv_prefill(draw, mixed, conv, dw, M, C, nullptr);
  ling::kernels::gdn_recurrent(mixed, dg, db, state, state, out, M, H, HV, nullptr);
  const auto p_out = download(out, size_t(M) * HV * 128), p_state = download(state, state0.size()),
             p_conv = download(conv, conv0.size());
  // Prefill path, in two calls.
  restore();
  ling::kernels::gdn_conv_prefill(draw, mixed, conv, dw, split, C, nullptr);
  ling::kernels::gdn_conv_prefill(draw + size_t(split) * C, mixed + size_t(split) * C, conv, dw, M - split, C, nullptr);
  ling::kernels::gdn_recurrent(mixed, dg, db, state, state, out, split, H, HV, nullptr);
  ling::kernels::gdn_recurrent(mixed + size_t(split) * C, dg + split * HV, db + split * HV, state, state,
                               out + size_t(split) * HV * 128, M - split, H, HV, nullptr);
  const cudaError_t e = cudaDeviceSynchronize();
  const bool same = download(out, size_t(M) * HV * 128) == p_out && download(state, state0.size()) == p_state &&
                    download(conv, conv0.size()) == p_conv;
  double worst = 0, rms = 0;
  for (size_t i = 0; i < p_out.size(); ++i) {
    worst = std::max(worst, double(std::fabs(p_out[i] - ref_out[i])));
    rms += double(ref_out[i]) * ref_out[i];
  }
  for (size_t i = 0; i < p_state.size(); ++i) worst = std::max(worst, double(std::fabs(p_state[i] - ref_state[i])));
  rms = std::sqrt(rms / p_out.size());
  const bool conv_same = p_conv == ref_conv;  // the window is a copy of raw inputs
  const bool ok = e == cudaSuccess && same && conv_same && worst < 1e-4 * std::max(rms, 1.0);
  std::printf("deltanet prefill: max diff against the rows path %.2e (rms %.2e), window %s, split run %s %s\n", worst,
              rms, conv_same ? "identical" : "DIFFERS", same ? "bit-identical" : "DIFFERS", ok ? "ok" : "FAIL");
  for (float* f : {draw, dg, db, mixed, conv, state, out}) cudaFree(f);
  cudaFree(dw);
  return ok;
}

void bench(std::mt19937& rng) {
  struct Shape {
    const char* name;
    bool fp4;
    int N, K;
  } shapes[] = {{"ffn gate|up", true, 17408, 5120},  {"ffn down", true, 5120, 17408}, {"gdn in_qkv", false, 10240, 5120},
                {"gdn in_z", false, 6144, 5120},     {"gdn out", false, 5120, 6144},   {"attn q", false, 12288, 5120},
                {"attn o", false, 5120, 6144}};
  for (int M : {256, 512, 2048}) {
    for (const Shape& sh : shapes) {
      Weights W = make_weights(rng, sh.fp4, sh.N, sh.K);
      uint8_t *q, *sf;
      float *x, *y;
      cudaMalloc(&x, size_t(M) * sh.K * 4);
      cudaMemset(x, 0, size_t(M) * sh.K * 4);
      cudaMalloc(&q, size_t(M) * sh.K);
      cudaMalloc(&sf, size_t(M) * sh.K / 16);
      cudaMemset(q, 0x22, size_t(M) * sh.K);
      cudaMemset(sf, 0x38, size_t(M) * sh.K / 16);
      cudaMalloc(&y, size_t(M) * sh.N * 4);
      auto run = [&] {
        if (sh.fp4) ling::kernels::prefill_gemm_nvfp4(q, sf, M, W.tiled, 1.f, y, sh.N, sh.N, sh.K, false, nullptr);
        else ling::kernels::prefill_gemm_fp8(q, M, W.tiled, 1.f, y, sh.N, sh.N, sh.K, false, nullptr);
      };
      run();
      cudaDeviceSynchronize();
      cudaEvent_t e0, e1;
      cudaEventCreate(&e0);
      cudaEventCreate(&e1);
      const int reps = 10;
      cudaEventRecord(e0);
      for (int r = 0; r < reps; ++r) run();
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      ms /= reps;
      // The quantization of the input, timed separately.
      cudaEventRecord(e0);
      for (int r = 0; r < reps; ++r) {
        if (sh.fp4) ling::kernels::quant_act_nvfp4(x, M, sh.K, 1.f, q, sf, nullptr);
        else ling::kernels::quant_act_fp8(x, M, sh.K, 1.f, q, nullptr);
      }
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float qms = 0;
      cudaEventElapsedTime(&qms, e0, e1);
      qms /= reps;
      const double flop = 2.0 * M * sh.N * sh.K;
      const double wbytes = sh.fp4 ? double(sh.N) * sh.K * (0.5 + 1.0 / 16) : double(sh.N) * sh.K;
      std::printf("bench M=%-4d %-12s %s N=%-5d K=%-5d %7.3f ms %6.1f TFLOPS %5.0f GB/s of weights | quant %.3f ms\n", M,
                  sh.name, sh.fp4 ? "nvfp4" : "fp8  ", sh.N, sh.K, ms, flop / ms / 1e9, wbytes / ms / 1e6, qms);
      cudaFree(x);
      cudaFree(q);
      cudaFree(sf);
      cudaFree(y);
      cudaFree(W.tiled);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::mt19937 rng(11);
  bool ok = test_quant(rng);
  ok &= test_deltanet(rng);
  for (bool fp4 : {true, false}) {
    ok &= test_gemm(rng, fp4, 200, 256, 512, false);
    ok &= test_gemm(rng, fp4, 33, 128, 256, true);
    ok &= test_gemm(rng, fp4, 300, 384, 1024, true);
  }
  if (argc > 1 && std::string(argv[1]) == "--bench") bench(rng);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf(ok ? "all prefill tests passed\n" : "PREFILL TESTS FAILED\n");
  return ok ? 0 : 1;
}
