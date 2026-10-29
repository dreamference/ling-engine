// ling-lora-kernel-tests: the LoRA kernels (src/core/lora_kernels.cuh) against a host reference in double, and the
// properties the engine relies on bit for bit: a row's delta is the same alone or in a batch of any size beside rows
// on other adapters or none (row invariance, so a verify's rows equal the decode's); a row on no adapter is never
// written; a graph's device slots give what the eager slots give; a site the adapter does not touch is skipped.
// Exits 0 when everything holds.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/lora_kernels.cuh"

namespace k = ling::kernels;

namespace {

int failures = 0;

void report(const char* what, bool ok) {
  std::printf("%-100s %s\n", what, ok ? "ok" : "FAILED");
  if (!ok) ++failures;
}

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, v.size() * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

uint16_t bf16_bits(float f) {
  const __nv_bfloat16 b = __float2bfloat16(f);
  uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}

float bf16_value(uint16_t u) {
  __nv_bfloat16 b;
  std::memcpy(&b, &u, 2);
  return __bfloat162float(b);
}

constexpr int K = 512;
const int kN[3] = {96, 32, 32};  // the site's three outputs (q_proj's, k_proj's and v_proj's, scaled down)
constexpr int kMaxN = 96;

// One slot's site on the host: its A, B (BF16 bits) and segments, and its device copy.
struct HostSite {
  int rank = 0, blocks = 0;
  float scale = 0.f;
  std::vector<uint16_t> a, b;
  std::vector<k::LoraSegDev> segs;
};

HostSite make_site(std::mt19937& rng, int rank, const std::vector<std::pair<int, int>>& outs_blocks, float scale) {
  std::normal_distribution<float> nd(0.f, 0.05f);
  HostSite s;
  s.rank = rank;
  s.scale = scale;
  for (const auto& [out, block] : outs_blocks) s.blocks = std::max(s.blocks, block + 1);
  s.a.resize(size_t(s.blocks) * rank * K);
  for (auto& v : s.a) v = bf16_bits(nd(rng));
  for (const auto& [out, block] : outs_blocks) {
    k::LoraSegDev g;
    g.out = out;
    g.n0 = 0;
    g.n1 = kN[out];
    g.block = block;
    g.b_off = static_cast<long long>(s.b.size());
    for (int i = 0; i < kN[out] * rank; ++i) s.b.push_back(bf16_bits(nd(rng)));
    s.segs.push_back(g);
  }
  return s;
}

k::LoraSiteDev to_device(const HostSite& s, std::vector<void*>& owned) {
  k::LoraSiteDev d;
  if (s.rank == 0) return d;
  __nv_bfloat16* a = reinterpret_cast<__nv_bfloat16*>(upload(s.a));
  __nv_bfloat16* b = reinterpret_cast<__nv_bfloat16*>(upload(s.b));
  owned.push_back(a);
  owned.push_back(b);
  d.a = a;
  d.b = b;
  d.K = K;
  d.rank = s.rank;
  d.blocks = s.blocks;
  d.nsegs = static_cast<int>(s.segs.size());
  d.scale = s.scale;
  for (int i = 0; i < d.nsegs; ++i) d.segs[i] = s.segs[i];
  return d;
}

// The host reference: y[r] += scale * B (A x[r]), in double.
void reference(const HostSite& s, const float* x, std::vector<double>* y) {
  std::vector<double> t(size_t(s.blocks) * s.rank);
  for (size_t j = 0; j < t.size(); ++j) {
    double v = 0;
    for (int kk = 0; kk < K; ++kk) v += double(x[kk]) * bf16_value(s.a[j * K + kk]);
    t[j] = v;
  }
  for (const k::LoraSegDev& g : s.segs)
    for (int n = g.n0; n < g.n1; ++n) {
      double v = 0;
      for (int j = 0; j < s.rank; ++j)
        v += double(bf16_value(s.b[g.b_off + size_t(n - g.n0) * s.rank + j])) * t[size_t(g.block) * s.rank + j];
      y[g.out][n] += s.scale * v;
    }
}

// The device state of one run: x, t and the three outputs over M rows, with a slot table of `sites`.
struct Run {
  int M;
  float *x, *t, *y[3];
  k::LoraSiteDev* table;
  int slots;
  Run(const std::vector<float>& hx, const std::vector<float> hy[3], int M_, const std::vector<k::LoraSiteDev>& sites)
      : M(M_), slots(static_cast<int>(sites.size())) {
    x = upload(hx);
    cudaMalloc(&t, size_t(M) * k::kLoraRankRows * sizeof(float));
    cudaMemset(t, 0, size_t(M) * k::kLoraRankRows * sizeof(float));
    for (int o = 0; o < 3; ++o) y[o] = upload(hy[o]);
    table = upload(sites);
  }
  ~Run() {
    cudaFree(x);
    cudaFree(t);
    for (float* p : y) cudaFree(p);
    cudaFree(table);
  }
  void go(const k::LoraRows& rows, bool skip_q = false) {
    k::LoraOuts outs;
    for (int o = 0; o < 3; ++o) {
      outs.y[o] = skip_q && o == 0 ? nullptr : y[o];
      outs.ld[o] = kN[o];
    }
    k::lora_shrink(x, K, M, rows, table, 1, slots, t, nullptr);
    k::lora_expand(t, M, rows, table, 1, slots, outs, kMaxN, nullptr);
  }
  std::vector<float> out(int o) {
    cudaDeviceSynchronize();
    return download(y[o], size_t(M) * kN[o]);
  }
};

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  // Slot 0: q, k and v at rank 16 (three blocks); slot 1: q and v at rank 8 (two blocks, rsLoRA's scale); slot 2: the
  // adapter does not touch this site.
  std::vector<HostSite> host = {make_site(rng, 16, {{0, 0}, {1, 1}, {2, 2}}, 2.f),
                                make_site(rng, 8, {{0, 0}, {2, 1}}, 16.f / std::sqrt(8.f)), HostSite{}};
  std::vector<void*> owned;
  std::vector<k::LoraSiteDev> sites;
  for (const HostSite& s : host) sites.push_back(to_device(s, owned));

  // Four sequences of 5, 1, 9 and 3 rows on slots 1, -1, 0 and 2: 18 rows, tiles that straddle sequences.
  const int seq_rows[4] = {5, 1, 9, 3}, seq_slot[4] = {1, -1, 0, 2};
  const int M = 18;
  std::vector<float> hx(size_t(M) * K);
  for (auto& v : hx) v = nd(rng);
  std::vector<float> hy[3];
  for (int o = 0; o < 3; ++o) {
    hy[o].resize(size_t(M) * kN[o]);
    for (auto& v : hy[o]) v = nd(rng);
  }
  k::LoraRows rows;
  rows.n = 4;
  for (int i = 0; i < 4; ++i) {
    rows.row0[i + 1] = rows.row0[i] + seq_rows[i];
    rows.slot[i] = seq_slot[i];
  }
  std::vector<float> batch[3];
  {
    Run r(hx, hy, M, sites);
    r.go(rows);
    for (int o = 0; o < 3; ++o) batch[o] = r.out(o);
  }

  // Against the reference; rows on no adapter, and on slot 2's untouched site, bit for bit as they were.
  {
    double worst = 0;
    bool untouched = true;
    for (int seq = 0; seq < 4; ++seq)
      for (int r = rows.row0[seq]; r < rows.row0[seq + 1]; ++r) {
        std::vector<double> want[3];
        for (int o = 0; o < 3; ++o) want[o].assign(hy[o].begin() + size_t(r) * kN[o], hy[o].begin() + size_t(r + 1) * kN[o]);
        const int s = seq_slot[seq];
        if (s >= 0 && host[s].rank > 0) reference(host[s], hx.data() + size_t(r) * K, want);
        for (int o = 0; o < 3; ++o)
          for (int n = 0; n < kN[o]; ++n) {
            const float got = batch[o][size_t(r) * kN[o] + n];
            if (s < 0 || host[s].rank == 0) untouched &= got == hy[o][size_t(r) * kN[o] + n];
            worst = std::max(worst, std::abs(got - want[o][n]) / (1.0 + std::abs(want[o][n])));
          }
      }
    std::printf("worst relative error against the double reference: %.3g\n", worst);
    report("shrink and expand over four sequences on slots 1, -1, 0, 2 match the reference (1e-5)", worst < 1e-5);
    report("the base sequence's rows and the untouched site's rows are not written", untouched);
  }

  // Row invariance: each row alone (M = 1, its own slot) gives the batch's bits.
  {
    bool same = true;
    for (int seq = 0; seq < 4; ++seq)
      for (int r = rows.row0[seq]; r < rows.row0[seq + 1]; ++r) {
        std::vector<float> x1(hx.begin() + size_t(r) * K, hx.begin() + size_t(r + 1) * K), y1[3];
        for (int o = 0; o < 3; ++o) y1[o].assign(hy[o].begin() + size_t(r) * kN[o], hy[o].begin() + size_t(r + 1) * kN[o]);
        Run one(x1, y1, 1, sites);
        k::LoraRows r1;
        r1.n = 1;
        r1.row0[1] = 1;
        r1.slot[0] = seq_slot[seq];
        one.go(r1);
        for (int o = 0; o < 3; ++o)
          same &= std::memcmp(one.out(o).data(), batch[o].data() + size_t(r) * kN[o], kN[o] * sizeof(float)) == 0;
      }
    report("every row alone (M = 1) is bit for bit its row of the 18-row batch", same);
  }

  // A graph's slots (slot_dev) against the eager ones; then the slots changed on the device under the same launch.
  {
    int* dslot = upload(std::vector<int>(seq_slot, seq_slot + 4));
    k::LoraRows g = rows;
    for (int& s : g.slot) s = -7;  // must not be read
    g.slot_dev = dslot;
    Run r(hx, hy, M, sites);
    r.go(g);
    bool same = true;
    for (int o = 0; o < 3; ++o) same &= std::memcmp(r.out(o).data(), batch[o].data(), batch[o].size() * 4) == 0;
    report("device slots (a captured graph's) give the eager slots' bits", same);
    const std::vector<int> base(4, -1);
    cudaMemcpy(dslot, base.data(), 4 * sizeof(int), cudaMemcpyHostToDevice);
    Run r2(hx, hy, M, sites);
    r2.go(g);
    bool none = true;
    for (int o = 0; o < 3; ++o) none &= std::memcmp(r2.out(o).data(), hy[o].data(), hy[o].size() * 4) == 0;
    report("every slot -1 on the device: no output written", none);
    cudaFree(dslot);
  }

  // A null output (the drafter's context rows: k and v, no q) is skipped; the others get the batch's bits.
  {
    Run r(hx, hy, M, sites);
    r.go(rows, true);
    const bool q_kept = r.out(0) == hy[0];
    bool same = true;
    for (int o = 1; o < 3; ++o) same &= std::memcmp(r.out(o).data(), batch[o].data(), batch[o].size() * 4) == 0;
    report("a null output is skipped, the site's other outputs as in the batch", q_kept && same);
  }

  for (void* p : owned) cudaFree(p);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    ++failures;
  }
  std::printf(failures ? "LORA KERNEL TESTS FAILED (%d)\n" : "all lora kernel tests passed\n", failures);
  return failures ? 1 : 0;
}
