// Checks the fused decode + prefill GEMM (fused_gemm.cu) against the two kernels it fuses, byte for byte: the decode
// rows against stream_gemm (stream_gemm_multi for gate|up, stream_gemm_nvfp4 for the down projection), the chunk rows
// against prefill_swiglu_nvfp4 / prefill_gemm_nvfp4 / prefill_gemm_fp8(_bf16out), at the model's shapes, NVFP4 blobs
// raw and coded (B26) alike.
// With --bench, times the fused launch against the two kernels back to back and on two concurrent streams.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.cuh"

namespace {

namespace k = ling::kernels;

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

// A random NVFP4 matrix tiled (retile_nvfp4), and the same matrix coded (B26) with its raw scales and code table.
struct Weights {
  uint8_t *tiled = nullptr, *coded = nullptr, *raw = nullptr, *table = nullptr;
};

Weights make_weights(std::mt19937& rng, int N, int K) {
  std::uniform_int_distribution<int> byte(0, 255), sbyte(0x28, 0x40);
  std::vector<uint8_t> w(size_t(N) * K / 2), s(size_t(N) * K / 16);
  for (auto& b : w) b = static_cast<uint8_t>(byte(rng));
  for (auto& b : s) b = static_cast<uint8_t>(sbyte(rng));
  uint8_t *dw = upload(w), *ds = upload(s);
  Weights W;
  cudaMalloc(&W.tiled, k::tiled_bytes_nvfp4(N, K));
  k::retile_nvfp4(dw, ds, W.tiled, N, K, nullptr);
  uint8_t table[k::kScaleCodes], code_of[256];
  if (k::scale_code_table(s.data(), s.size(), table, code_of)) {
    uint8_t* map = upload(std::vector<uint8_t>(code_of, code_of + 256));
    W.table = upload(std::vector<uint8_t>(table, table + k::kScaleCodes));
    cudaMalloc(&W.coded, k::tiled_bytes_nvfp4_coded(N, K));
    cudaMalloc(&W.raw, s.size());
    k::retile_nvfp4_coded(dw, ds, map, W.coded, W.raw, N, K, nullptr);
    cudaDeviceSynchronize();
    cudaFree(map);
  }
  cudaDeviceSynchronize();
  cudaFree(dw);
  cudaFree(ds);
  return W;
}

std::vector<float> activations(std::mt19937& rng, int M, int K, float scale) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> x(size_t(M) * K);
  for (size_t i = 0; i < x.size(); ++i) x[i] = nd(rng) * (i % 97 == 0 ? 30.f : 1.f) * scale;
  return x;
}

template <typename T>
bool same_bytes(const T* a, const T* b, size_t n) {
  return std::memcmp(download(a, n).data(), download(b, n).data(), n * sizeof(T)) == 0;
}

constexpr int kH = 5120, kI = 17408;  // Qwen3.8-27B: hidden, FFN intermediate
constexpr float kGate2 = 0.37f, kUp2 = 0.29f, kDown2 = 0.41f, kInGate = 0.011f, kInDown = 0.023f, kOutIn = 0.05f;

struct Buffers {
  float *xd, *xinv, *xp;  // decode FP32 input [32][K], its scales; prefill FP32 input [512][K]
  __half* xh;
  uint8_t *xq, *xsf;
  float *yg[2], *yu[2], *yp[2];   // [0] the reference, [1] the fused launch
  uint8_t *q2[2], *sf2[2];
};

Buffers make_buffers(std::mt19937& rng, int K, int N) {
  Buffers b;
  b.xd = upload(activations(rng, 32, K, 1.f));
  b.xp = upload(activations(rng, 512, K, 0.02f));
  cudaMalloc(&b.xh, size_t(32) * K * sizeof(__half));
  cudaMalloc(&b.xinv, 32 * sizeof(float));
  cudaMalloc(&b.xq, size_t(512) * K);  // NVFP4 needs K / 2 bytes per row, FP8 K
  cudaMalloc(&b.xsf, size_t(512) * K / 16);
  for (int i = 0; i < 2; ++i) {
    cudaMalloc(&b.yg[i], size_t(32) * N * sizeof(float));
    cudaMalloc(&b.yu[i], size_t(32) * N * sizeof(float));
    cudaMalloc(&b.yp[i], size_t(512) * N * sizeof(float));
    cudaMalloc(&b.q2[i], size_t(512) * N / 2);
    cudaMalloc(&b.sf2[i], size_t(512) * N / 16);
  }
  return b;
}

// gate|up: K = kH, N = kI.
bool test_swiglu(std::mt19937& rng) {
  const int K = kH, N = kI;
  const Weights G = make_weights(rng, N, K), U = make_weights(rng, N, K);
  Buffers b = make_buffers(rng, K, N);
  bool ok = true;
  for (bool coded : {false, true}) {
    if (coded && (!G.coded || !U.coded)) continue;
    const uint8_t *gw = coded ? G.coded : G.tiled, *uw = coded ? U.coded : U.tiled;
    const uint8_t *gsf = coded ? G.raw : nullptr, *usf = coded ? U.raw : nullptr;
    for (int Md : {0, 1, 8, 12, 16, 17, 24, 32}) {
      for (int Mp : {0, 1, 128, 200, 256, 512}) {
        if (Md == 0 && Mp == 0) continue;
        for (int i = 0; i < 2; ++i) {
          cudaMemset(b.yg[i], 0xff, size_t(32) * N * sizeof(float));
          cudaMemset(b.yu[i], 0xff, size_t(32) * N * sizeof(float));
          cudaMemset(b.q2[i], 0xee, size_t(512) * N / 2);
          cudaMemset(b.sf2[i], 0xee, size_t(512) * N / 16);
        }
        if (Md > 0) {
          k::to_half_rows(b.xd, Md, K, b.xh, b.xinv, nullptr);
          const k::StreamTarget t[2] = {{gw, coded ? G.table : nullptr, kGate2, b.yg[0], N},
                                        {uw, coded ? U.table : nullptr, kUp2, b.yu[0], N}};
          k::stream_gemm_multi(true, b.xh, b.xinv, Md, t, 2, K, nullptr);
        }
        if (Mp > 0) {
          k::quant_act_nvfp4(b.xp, Mp, K, kInGate, b.xq, b.xsf, nullptr);
          k::prefill_swiglu_nvfp4(b.xq, b.xsf, Mp, gw, kInGate * kGate2, uw, kInGate * kUp2, N, K, kOutIn, b.q2[0],
                                  b.sf2[0], nullptr, gsf, usf);
        }
        k::fused_swiglu_nvfp4(b.xh, b.xinv, Md, b.yg[1], b.yu[1], kGate2, kUp2, b.xq, b.xsf, Mp, gw, kInGate * kGate2, uw,
                              kInGate * kUp2, N, K, kOutIn, b.q2[1], b.sf2[1], nullptr, gsf, usf);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          std::printf("CUDA error: %s\n", cudaGetErrorString(e));
          return false;
        }
        const size_t nd = size_t(Md) * N;
        const bool dec = same_bytes(b.yg[0], b.yg[1], nd) && same_bytes(b.yu[0], b.yu[1], nd);
        const bool pre = same_bytes(b.q2[0], b.q2[1], size_t(Mp) * N / 2) &&
                         same_bytes(b.sf2[0], b.sf2[1], size_t(Mp) * N / 16);
        ok &= dec && pre;
        std::printf("fused gate|up %-5s decode %2d rows vs stream_gemm_multi: %-9s chunk %3d rows vs prefill_swiglu: %s\n",
                    coded ? "coded" : "raw", Md, dec ? "identical" : "DIFFER", Mp, pre ? "identical" : "DIFFER");
      }
    }
  }
  return ok;
}

// down: K = kI, N = kH; the chunk's rows added into a residual, as the engine runs it.
bool test_down(std::mt19937& rng) {
  const int K = kI, N = kH;
  const Weights D = make_weights(rng, N, K);
  Buffers b = make_buffers(rng, K, N);
  const std::vector<float> resid = activations(rng, 512, N, 1.f);
  bool ok = true;
  for (bool coded : {false, true}) {
    if (coded && !D.coded) continue;
    const uint8_t* w = coded ? D.coded : D.tiled;
    const uint8_t* sf = coded ? D.raw : nullptr;
    for (int Md : {0, 12, 16, 24, 32}) {
      for (int Mp : {0, 128, 256, 512}) {
        if (Md == 0 && Mp == 0) continue;
        for (int i = 0; i < 2; ++i) {
          cudaMemset(b.yg[i], 0xff, size_t(32) * N * sizeof(float));
          cudaMemcpy(b.yp[i], resid.data(), resid.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        if (Md > 0) {
          k::to_half_rows(b.xd, Md, K, b.xh, b.xinv, nullptr);
          k::stream_gemm_nvfp4(b.xh, b.xinv, Md, w, coded ? D.table : nullptr, kDown2, b.yg[0], N, K, nullptr);
        }
        if (Mp > 0) {
          k::quant_act_nvfp4(b.xp, Mp, K, kInDown, b.xq, b.xsf, nullptr);
          k::prefill_gemm_nvfp4(b.xq, b.xsf, Mp, w, kInDown * kDown2, b.yp[0], N, N, K, true, nullptr, sf);
        }
        k::fused_gemm_nvfp4(b.xh, b.xinv, Md, b.yg[1], kDown2, b.xq, b.xsf, Mp, w, kInDown * kDown2, b.yp[1], N, N, K,
                            true, nullptr, sf);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          std::printf("CUDA error: %s\n", cudaGetErrorString(e));
          return false;
        }
        const bool dec = same_bytes(b.yg[0], b.yg[1], size_t(Md) * N);
        const bool pre = same_bytes(b.yp[0], b.yp[1], size_t(512) * N);
        ok &= dec && pre;
        std::printf("fused down    %-5s decode %2d rows vs stream_gemm_nvfp4: %-9s chunk %3d rows vs prefill_gemm: %s\n",
                    coded ? "coded" : "raw", Md, dec ? "identical" : "DIFFER", Mp, pre ? "identical" : "DIFFER");
      }
    }
  }
  return ok;
}

// A random FP8 E4M3 matrix (no NaN encodings), tiled (retile_fp8).
uint8_t* make_fp8_weights(std::mt19937& rng, int N, int K) {
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<uint8_t> w(size_t(N) * K);
  for (auto& b : w) {
    do b = static_cast<uint8_t>(byte(rng));
    while ((b & 0x7f) == 0x7f);
  }
  uint8_t *dw = upload(w), *tiled;
  cudaMalloc(&tiled, k::tiled_bytes_fp8(N, K));
  k::retile_fp8(dw, tiled, N, K, nullptr);
  cudaDeviceSynchronize();
  cudaFree(dw);
  return tiled;
}

constexpr float kWs8 = 0.021f, kIn8 = 0.035f;

// The FP8 projections: a wide one with BF16 output (DeltaNet in_qkv: N 10,240, K 5,120) and a narrow one added into
// the residual (the attention and DeltaNet out-projections: N 5,120, K 6,144).
bool test_fp8(std::mt19937& rng) {
  bool ok = true;
  for (bool wide : {true, false}) {
    const int N = wide ? 10240 : kH, K = wide ? kH : 6144;
    uint8_t* w = make_fp8_weights(rng, N, K);
    Buffers b = make_buffers(rng, K, N);
    const std::vector<float> resid = activations(rng, 512, N, 1.f);
    for (int Md : {0, 1, 12, 16, 17, 24, 32}) {
      for (int Mp : {0, 1, 128, 200, 256, 512}) {
        if (Md == 0 && Mp == 0) continue;
        for (int i = 0; i < 2; ++i) {
          cudaMemset(b.yg[i], 0xff, size_t(32) * N * sizeof(float));
          cudaMemcpy(b.yp[i], resid.data(), resid.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        auto* yb0 = reinterpret_cast<__nv_bfloat16*>(b.yp[0]);
        auto* yb1 = reinterpret_cast<__nv_bfloat16*>(b.yp[1]);
        if (Md > 0) {
          k::to_half_rows(b.xd, Md, K, b.xh, b.xinv, nullptr);
          k::stream_gemm_fp8(b.xh, b.xinv, Md, w, kWs8, b.yg[0], N, K, nullptr);
        }
        if (Mp > 0) {
          k::quant_act_fp8(b.xp, Mp, K, kIn8, b.xq, nullptr);
          if (wide) k::prefill_gemm_fp8_bf16out(b.xq, Mp, w, kIn8 * kWs8, yb0, N, N, K, nullptr);
          else k::prefill_gemm_fp8(b.xq, Mp, w, kIn8 * kWs8, b.yp[0], N, N, K, true, nullptr);
        }
        if (wide) k::fused_gemm_fp8_bf16out(b.xh, b.xinv, Md, b.yg[1], kWs8, b.xq, Mp, w, kIn8 * kWs8, yb1, N, N, K, nullptr);
        else k::fused_gemm_fp8(b.xh, b.xinv, Md, b.yg[1], kWs8, b.xq, Mp, w, kIn8 * kWs8, b.yp[1], N, N, K, true, nullptr);
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          std::printf("CUDA error: %s\n", cudaGetErrorString(e));
          return false;
        }
        const bool dec = same_bytes(b.yg[0], b.yg[1], size_t(Md) * N);
        const bool pre = same_bytes(b.yp[0], b.yp[1], size_t(512) * N);
        ok &= dec && pre;
        std::printf("fused fp8 %-6s decode %2d rows vs stream_gemm_fp8: %-9s chunk %3d rows vs prefill_gemm_fp8%s: %s\n",
                    wide ? "wide" : "narrow", Md, dec ? "identical" : "DIFFER", Mp, wide ? "_bf16out" : " +=",
                    pre ? "identical" : "DIFFER");
      }
    }
  }
  return ok;
}

// ---- --bench ----

struct Timer {
  cudaEvent_t a, b;
  Timer() {
    cudaEventCreate(&a);
    cudaEventCreate(&b);
  }
};

// Median of `reps` timings of fn (ms), each after the L2 is flushed (a 96 MB write), so every run streams its
// weights from DRAM as a step's projection does.
template <typename F>
float time_ms(F&& fn, int reps = 15) {
  static uint8_t* flush = nullptr;
  constexpr size_t kFlush = size_t(96) << 20;
  if (!flush) cudaMalloc(&flush, kFlush);
  static Timer t;
  std::vector<float> v;
  fn();  // warm-up
  for (int i = 0; i < reps; ++i) {
    cudaMemsetAsync(flush, i, kFlush, nullptr);
    cudaEventRecord(t.a, nullptr);
    fn();
    cudaEventRecord(t.b, nullptr);
    cudaEventSynchronize(t.b);
    float ms = 0;
    cudaEventElapsedTime(&ms, t.a, t.b);
    v.push_back(ms);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void bench() {
  std::mt19937 rng(7);
  cudaStream_t s1, s2;
  cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
  cudaEvent_t fork, j1, j2;
  cudaEventCreateWithFlags(&fork, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&j1, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&j2, cudaEventDisableTiming);
  // Two kernels on two streams, forked from and joined into the default stream (which the timer records on).
  auto concurrent = [&](auto&& a, auto&& b) {
    cudaEventRecord(fork, nullptr);
    cudaStreamWaitEvent(s1, fork);
    cudaStreamWaitEvent(s2, fork);
    a(s1);
    b(s2);
    cudaEventRecord(j1, s1);
    cudaEventRecord(j2, s2);
    cudaStreamWaitEvent(nullptr, j1);
    cudaStreamWaitEvent(nullptr, j2);
  };
  for (int pass = 0; pass < 4; ++pass) {
    // gate|up, down, then the FP8 DeltaNet in_qkv|z (one 16,384-row matrix: a launch's matrices are independent)
    // and out-projection.
    const bool swiglu = pass == 0, fp8 = pass >= 2, wide8 = pass == 2;
    const int K = pass == 0 || pass == 2 ? kH : pass == 1 ? kI : 6144, N = pass == 0 ? kI : pass == 2 ? 16384 : kH;
    uint8_t* w8 = fp8 ? make_fp8_weights(rng, N, K) : nullptr;
    const Weights A = fp8 ? Weights{} : make_weights(rng, N, K), B = swiglu ? make_weights(rng, N, K) : A;
    const bool coded = A.coded && B.coded;
    const uint8_t *aw = coded ? A.coded : A.tiled, *bw = coded ? B.coded : B.tiled;
    const uint8_t *asf = coded ? A.raw : nullptr, *bsf = coded ? B.raw : nullptr;
    Buffers b = make_buffers(rng, K, N);
    const double gb = fp8 ? double(N) * K / 1e9 : (swiglu ? 2.0 : 1.0) * N * K * (0.5 + 1.0 / 16) / 1e9;
    const char* name = pass == 0 ? "gate|up" : pass == 1 ? "down" : pass == 2 ? "fp8 in_qkv|z, BF16 out" : "fp8 out +=";
    std::printf("\n%s (N %d, K %d, %s, %.1f MB of weights): ms (weight GB/s)\n", name, N, K,
                fp8 ? "FP8" : coded ? "coded blobs" : "raw blobs", gb * 1e3);
    std::printf("%-6s %-6s %-14s %-14s %-14s %-14s %-14s\n", "decode", "chunk", "stream", "prefill", "back-to-back",
                "concurrent", "fused");
    for (int Md : {12, 16, 24, 32}) {
      k::to_half_rows(b.xd, Md, K, b.xh, b.xinv, nullptr);
      for (int Mp : {128, 256, 512}) {
        if (fp8) k::quant_act_fp8(b.xp, Mp, K, kIn8, b.xq, nullptr);
        else k::quant_act_nvfp4(b.xp, Mp, K, kInGate, b.xq, b.xsf, nullptr);
        auto* yb0 = reinterpret_cast<__nv_bfloat16*>(b.yp[0]);
        auto* yb1 = reinterpret_cast<__nv_bfloat16*>(b.yp[1]);
        auto dec = [&](cudaStream_t s) {
          if (fp8) {
            k::stream_gemm_fp8(b.xh, b.xinv, Md, w8, kWs8, b.yg[0], N, K, s);
          } else if (swiglu) {
            const k::StreamTarget t[2] = {{aw, coded ? A.table : nullptr, kGate2, b.yg[0], N},
                                          {bw, coded ? B.table : nullptr, kUp2, b.yu[0], N}};
            k::stream_gemm_multi(true, b.xh, b.xinv, Md, t, 2, K, s);
          } else {
            k::stream_gemm_nvfp4(b.xh, b.xinv, Md, aw, coded ? A.table : nullptr, kDown2, b.yg[0], N, K, s);
          }
        };
        auto pre = [&](cudaStream_t s) {
          if (wide8)
            k::prefill_gemm_fp8_bf16out(b.xq, Mp, w8, 1.f, yb0, N, N, K, s);
          else if (fp8)
            k::prefill_gemm_fp8(b.xq, Mp, w8, 1.f, b.yp[0], N, N, K, true, s);
          else if (swiglu)
            k::prefill_swiglu_nvfp4(b.xq, b.xsf, Mp, aw, 1.f, bw, 1.f, N, K, kOutIn, b.q2[0], b.sf2[0], s, asf, bsf);
          else
            k::prefill_gemm_nvfp4(b.xq, b.xsf, Mp, aw, 1.f, b.yp[0], N, N, K, false, s, asf);
        };
        auto fused = [&](cudaStream_t s) {
          if (wide8)
            k::fused_gemm_fp8_bf16out(b.xh, b.xinv, Md, b.yg[1], kWs8, b.xq, Mp, w8, 1.f, yb1, N, N, K, s);
          else if (fp8)
            k::fused_gemm_fp8(b.xh, b.xinv, Md, b.yg[1], kWs8, b.xq, Mp, w8, 1.f, b.yp[1], N, N, K, true, s);
          else if (swiglu)
            k::fused_swiglu_nvfp4(b.xh, b.xinv, Md, b.yg[1], b.yu[1], kGate2, kUp2, b.xq, b.xsf, Mp, aw, 1.f, bw, 1.f, N,
                                  K, kOutIn, b.q2[1], b.sf2[1], s, asf, bsf);
          else
            k::fused_gemm_nvfp4(b.xh, b.xinv, Md, b.yg[1], kDown2, b.xq, b.xsf, Mp, aw, 1.f, b.yp[1], N, N, K, false, s,
                                asf);
        };
        const float t_dec = time_ms([&] { dec(nullptr); });
        const float t_pre = time_ms([&] { pre(nullptr); });
        const float t_seq = time_ms([&] {
          dec(nullptr);
          pre(nullptr);
        });
        const float t_con = time_ms([&] { concurrent(dec, pre); });
        const float t_fus = time_ms([&] { fused(nullptr); });
        auto cell = [&](float ms) {
          static char buf[8][32];
          static int i = 0;
          char* c = buf[i++ % 8];
          std::snprintf(c, 32, "%.3f (%3.0f)", ms, gb / (ms * 1e-3));
          return c;
        };
        std::printf("%-6d %-6d %-14s %-14s %-14s %-14s %-14s\n", Md, Mp, cell(t_dec), cell(t_pre), cell(t_seq),
                    cell(t_con), cell(t_fus));
      }
    }
  }
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) std::printf("CUDA error: %s\n", cudaGetErrorString(e));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--bench") {
    bench();
    return 0;
  }
  std::mt19937 rng(2026);
  bool ok = test_swiglu(rng);
  ok &= test_down(rng);
  ok &= test_fp8(rng);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    return 1;
  }
  std::printf(ok ? "all fused tests passed\n" : "FUSED TESTS FAILED\n");
  return ok ? 0 : 1;
}
