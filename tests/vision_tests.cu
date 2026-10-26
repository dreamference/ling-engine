// The vision tower's kernels against host references on random data (specs/DREAMFERENCE_LING_ENGINE_VISION.md §20.8):
// the attention (bidirectional, head dimension 72, lengths that end inside a tile), the LayerNorm, the rotary, the
// epilogue and the position embedding's host side (torch.linspace and the merge-window order). The whole tower is
// checked against reference dumps by `ling-run --vision-check`.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/vision.hpp"

namespace {

float bfr(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

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

bool report(const char* what, double err, double tol) {
  const bool ok = err <= tol;
  std::printf("%-44s max err %.3e (tol %.1e) %s\n", what, err, tol, ok ? "ok" : "FAIL");
  return ok;
}

bool test_attention(int n, int heads, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.f, 1.f);
  const int D = 72, H = heads * D;
  std::vector<__nv_bfloat16> qkv(size_t(n) * 3 * H);
  std::vector<float> f(qkv.size());
  for (size_t i = 0; i < qkv.size(); ++i) {
    qkv[i] = __float2bfloat16_rn(nd(rng) * 1.5f);
    f[i] = __bfloat162float(qkv[i]);
  }
  __nv_bfloat16* dq = upload(qkv);
  __nv_bfloat16* dout = nullptr;
  cudaMalloc(&dout, size_t(n) * H * 2);
  ling::vision_kernels::attention(dq, dout, n, heads, nullptr);
  cudaDeviceSynchronize();
  std::vector<__nv_bfloat16> out = download(dout, size_t(n) * H);
  double worst = 0;
  const double scale = 1.0 / std::sqrt(double(D));
  std::vector<double> p(n), ref(size_t(n) * H);
  double sumsq = 0;
  for (int h = 0; h < heads; ++h)
    for (int i = 0; i < n; ++i) {
      double mx = -1e300;
      for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int d = 0; d < D; ++d) s += double(f[size_t(i) * 3 * H + h * D + d]) * f[size_t(j) * 3 * H + H + h * D + d];
        p[j] = s * scale;
        mx = std::max(mx, p[j]);
      }
      double sum = 0;
      for (int j = 0; j < n; ++j) sum += (p[j] = std::exp(p[j] - mx));
      for (int d = 0; d < D; ++d) {
        double o = 0;
        for (int j = 0; j < n; ++j) o += p[j] * f[size_t(j) * 3 * H + 2 * H + h * D + d];
        o /= sum;
        ref[size_t(i) * H + h * D + d] = o;
        sumsq += o * o;
      }
    }
  // The error against the output's RMS: P is rounded to BF16 for the PV product and the output to BF16, so an element
  // near zero carries an absolute error of the order of 2^-9 of the large ones (FlashAttention's own numerics).
  // Both measures, and where the large ones sit: the first run's relative error |e| / (|o| + 0.05), its worst element's
  // |o|, and its worst among outputs of at least a quarter of the RMS against those below; then |e| / RMS(o), which
  // decides (P is rounded to BF16 for the PV product and the output to BF16, as in FlashAttention, so an element near
  // zero carries an absolute error of the order of 2^-9 of the large ones). The tower's comparison with the reference
  // dumps (ling-run --vision-check) is the test that matters.
  const double rms = std::sqrt(sumsq / ref.size());
  double rel_worst = 0, rel_at = 0, rel_big = 0, rel_small = 0, abs_worst = 0;
  for (size_t k = 0; k < ref.size(); ++k) {
    const double e = std::abs(double(__bfloat162float(out[k])) - ref[k]);
    const double rel = e / (std::abs(ref[k]) + 0.05);
    if (rel > rel_worst) rel_worst = rel, rel_at = ref[k];
    (std::abs(ref[k]) >= 0.25 * rms ? rel_big : rel_small) = std::max(std::abs(ref[k]) >= 0.25 * rms ? rel_big : rel_small, rel);
    abs_worst = std::max(abs_worst, e);
    worst = std::max(worst, e / rms);
  }
  std::printf("attention n=%-4d relative (|o|+0.05) worst %.3e at |o| = %.4f; among |o| >= RMS/4 %.3e, below %.3e; "
              "worst |e| %.3e, RMS(o) %.3f\n", n, rel_worst, std::abs(rel_at), rel_big, rel_small, abs_worst, rms);
  cudaFree(dq);
  cudaFree(dout);
  char what[64];
  std::snprintf(what, sizeof what, "attention n=%d heads=%d (|e| / RMS)", n, heads);
  return report(what, worst, 1e-2);
}

bool test_layernorm(std::mt19937& rng) {
  std::normal_distribution<float> nd(0.f, 1.f);
  const int rows = 37, cols = 1152;
  std::vector<__nv_bfloat16> x(size_t(rows) * cols), w(cols), b(cols);
  for (auto& v : x) v = __float2bfloat16_rn(nd(rng) * 3.f + 0.5f);
  for (auto& v : w) v = __float2bfloat16_rn(1.f + 0.1f * nd(rng));
  for (auto& v : b) v = __float2bfloat16_rn(0.1f * nd(rng));
  __nv_bfloat16 *dx = upload(x), *dw = upload(w), *db = upload(b), *dout = nullptr;
  cudaMalloc(&dout, x.size() * 2);
  ling::vision_kernels::layernorm(dx, dw, db, dout, rows, cols, 1e-6f, nullptr);
  cudaDeviceSynchronize();
  auto out = download(dout, x.size());
  double worst = 0;
  for (int r = 0; r < rows; ++r) {
    double m = 0, v = 0;
    for (int c = 0; c < cols; ++c) m += __bfloat162float(x[size_t(r) * cols + c]);
    m /= cols;
    for (int c = 0; c < cols; ++c) {
      const double d = __bfloat162float(x[size_t(r) * cols + c]) - m;
      v += d * d;
    }
    const double rstd = 1.0 / std::sqrt(v / cols + 1e-6);
    for (int c = 0; c < cols; ++c) {
      const double ref = (__bfloat162float(x[size_t(r) * cols + c]) - m) * rstd * __bfloat162float(w[c]) +
                         __bfloat162float(b[c]);
      worst = std::max(worst, std::abs(__bfloat162float(out[size_t(r) * cols + c]) - ref) / (std::abs(ref) + 1e-2));
    }
  }
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(db);
  cudaFree(dout);
  return report("layernorm", worst, 8e-3);
}

bool test_rotary(std::mt19937& rng) {
  std::normal_distribution<float> nd(0.f, 1.f);
  const int n = 50, heads = 3, H = heads * 72;
  std::vector<__nv_bfloat16> qkv(size_t(n) * 3 * H);
  for (auto& v : qkv) v = __float2bfloat16_rn(nd(rng));
  std::vector<int> pos(2 * n);
  for (int i = 0; i < n; ++i) pos[2 * i] = i / 7, pos[2 * i + 1] = (i * 13) % 40;
  __nv_bfloat16* dq = upload(qkv);
  int* dp = upload(pos);
  ling::vision_kernels::rotary(dq, dp, n, heads, nullptr);
  cudaDeviceSynchronize();
  auto out = download(dq, qkv.size());
  double worst = 0;
  for (int i = 0; i < n; ++i)
    for (int part = 0; part < 3; ++part)
      for (int h = 0; h < heads; ++h)
        for (int d = 0; d < 72; ++d) {
          const size_t base = size_t(i) * 3 * H + size_t(part) * H + h * 72;
          double ref = __bfloat162float(qkv[base + d]);
          if (part < 2) {
            const int dd = d % 36, j = dd < 18 ? dd : dd - 18;
            const double ang = (dd < 18 ? pos[2 * i] : pos[2 * i + 1]) * std::pow(10000.0, -2.0 * j / 36.0);
            const double x1 = __bfloat162float(qkv[base + dd]), x2 = __bfloat162float(qkv[base + dd + 36]);
            ref = d < 36 ? x1 * std::cos(ang) - x2 * std::sin(ang) : x2 * std::cos(ang) + x1 * std::sin(ang);
          }
          worst = std::max(worst, std::abs(__bfloat162float(out[base + d]) - ref) / (std::abs(ref) + 1e-2));
        }
  cudaFree(dq);
  cudaFree(dp);
  return report("rotary (q, k turned; v untouched)", worst, 8e-3);
}

bool test_bias_act() {
  const int M = 3, N = 6, ld = 8;
  std::vector<float> y(M * N);
  for (int i = 0; i < M * N; ++i) y[i] = (i - 9) * 0.37f;
  std::vector<__nv_bfloat16> b(N), base(M * ld);
  for (int i = 0; i < N; ++i) b[i] = __float2bfloat16_rn(0.1f * i);
  for (int i = 0; i < M * ld; ++i) base[i] = __float2bfloat16_rn(1.f + i);
  bool ok = true;
  for (int act = 0; act < 3; ++act)
    for (int res = 0; res < 2; ++res) {
      float* dy = upload(y);
      __nv_bfloat16 *db = upload(b), *dout = upload(base);
      ling::vision_kernels::bias_act(dy, db, M, N, dout, ld, act, res != 0, nullptr);
      cudaDeviceSynchronize();
      auto out = download(dout, base.size());
      double worst = 0;
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < ld; ++c) {
          const float got = __bfloat162float(out[r * ld + c]);
          float ref;
          if (c >= N) {
            ref = res ? __bfloat162float(base[r * ld + c]) : 0.f;
          } else {
            float v = bfr(y[r * N + c] + __bfloat162float(b[c]));
            if (act == 1) v = bfr(0.5f * v * (1.f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))));
            if (act == 2) v = bfr(0.5f * v * (1.f + std::erf(v * 0.7071067811865476f)));
            ref = res ? bfr(__bfloat162float(base[r * ld + c]) + v) : v;
          }
          worst = std::max(worst, double(std::abs(got - ref)));
        }
      char what[64];
      std::snprintf(what, sizeof what, "bias_act act=%d residual=%d", act, res);
      ok &= report(what, worst, 1e-2);
      cudaFree(dy);
      cudaFree(db);
      cudaFree(dout);
    }
  return ok;
}

bool test_positions() {
  bool ok = true;
  // torch.linspace(0, 47, 64, dtype=float32): the first half from 0, the second from 47.
  auto v = ling::torch_linspace(0.f, 47.f, 64);
  ok &= v.front() == 0.f && v.back() == 47.f && v.size() == 64;
  const float step = 47.f / 63.f;
  ok &= v[10] == 0.f + step * 10.f && v[40] == 47.f - step * 23.f;
  // Merge-window order on a 4 x 6 grid: patch 0..3 are (0,0) (0,1) (1,0) (1,1), then window (0,1).
  std::vector<int> idx, rc;
  std::vector<float> wt;
  ling::vision_positions(4, 6, 2, 48, idx, wt, rc);
  const int expect[8][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}};
  for (int p = 0; p < 8; ++p) ok &= rc[2 * p] == expect[p][0] && rc[2 * p + 1] == expect[p][1];
  // The four weights sum to 1 and the corners are the grid's at the linspace's sample points.
  const int n = 24;
  for (int p = 0; p < n; ++p) {
    float s = 0;
    for (int i = 0; i < 4; ++i) s += wt[size_t(i) * n + p];
    ok &= std::abs(s - 1.f) < 1e-5f;
  }
  ok &= idx[0] == 0;                                // patch (0, 0): corner (0, 0)
  ok &= idx[size_t(3) * n + 23] == 47 * 48 + 47;    // the last patch, (3, 5): its far corner is the table's last
  std::printf("%-44s %s\n", "linspace and merge-window positions", ok ? "ok" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  bool ok = true;
  ok &= test_positions();
  ok &= test_bias_act();
  ok &= test_layernorm(rng);
  ok &= test_rotary(rng);
  for (int n : {1, 63, 64, 65, 200, 257}) ok &= test_attention(n, 2, rng);
  std::printf(ok ? "vision tests passed\n" : "VISION TESTS FAILED\n");
  return ok ? 0 : 1;
}
