// Tree verification's kernels (BACKLOG 26) against the chain's, bit for bit, on random data: every row of a tree
// verify (attention with its q/k norms and RoPE, the DeltaNet conv and recurrence) must equal what the chain kernels
// compute for that row's root-to-row path laid out as a chain. Model-free: a context of random keys and values,
// random projections, trees from build_tree on random drafter scores.
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <tuple>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.cuh"
#include "core/tree.hpp"

namespace k = ling::kernels;

namespace {

template <typename T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <typename T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

ling::DraftTree random_tree(std::mt19937_64& rng, int E, int nodes, int width, int chain = 1000) {
  const int K16 = 16;
  std::uniform_real_distribution<float> u(-3.f, 3.f);
  std::vector<int> cand(size_t(E) * K16);
  for (size_t i = 0; i < cand.size(); ++i) cand[i] = int(i);
  std::vector<float> scores(size_t(E) * K16 * K16);
  for (float& x : scores) x = u(rng);
  std::vector<int> toks, picks;
  for (int e = 0, prev = 0; e < E; ++e) {
    const float* sc = scores.data() + (size_t(e) * K16 + (e ? prev : 0)) * K16;
    int pick = 0;
    for (int kk = 1; kk < K16; ++kk)
      if (sc[kk] > sc[pick]) pick = kk;
    toks.push_back(cand[size_t(e) * K16 + pick]);
    picks.push_back(pick);
    prev = pick;
  }
  ling::TreeShape shape;
  shape.nodes = nodes;
  shape.width = width;
  shape.chain = chain;
  return ling::build_tree(0, cand.data(), scores.data(), E, K16, toks, picks, shape);
}

int compare(const float* a, const float* b, size_t n) {
  return std::memcmp(a, b, n * sizeof(float)) == 0 ? 0 : 1;
}

// Attention: tree rows against each leaf path's chain.
bool check_attention(std::mt19937_64& rng, int pos0, int E, int nodes, int width) {
  const int Hq = 24, Hkv = 4, D = 256, rot = 64, ctx = 8192;
  const float theta = 1e7f, eps = 1e-6f;
  const ling::DraftTree t = random_tree(rng, E, nodes, width);
  k::TreeDesc td;
  ling::describe_tree(t, td);
  const int V = t.rows();
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__nv_bfloat16> kc(size_t(Hkv) * ctx * D), vc(kc.size());
  for (size_t i = 0; i < kc.size(); ++i) kc[i] = __float2bfloat16(nd(rng)), vc[i] = __float2bfloat16(nd(rng));
  std::vector<float> qg(size_t(V) * Hq * 2 * D), kk(size_t(V) * Hkv * D), vv(kk.size());
  for (float& x : qg) x = nd(rng);
  for (float& x : kk) x = nd(rng);
  for (float& x : vv) x = nd(rng);
  std::vector<__nv_bfloat16> qn(D), kn(D);
  for (int i = 0; i < D; ++i) qn[i] = __float2bfloat16(0.1f * nd(rng)), kn[i] = __float2bfloat16(0.1f * nd(rng));
  k::set_kv_layout(size_t(ctx) * D, D);
  __nv_bfloat16 *dkc = upload(kc), *dvc = upload(vc), *dqn = upload(qn), *dkn = upload(kn);
  float *dqg = upload(qg), *dk = upload(kk), *dv = upload(vv);
  float *dq = nullptr, *dgate = nullptr, *dout = nullptr, *dscr = nullptr;
  cudaMalloc(&dq, size_t(k::kMaxStreamRows) * Hq * D * 4);
  cudaMalloc(&dgate, size_t(k::kMaxStreamRows) * Hq * D * 4);
  cudaMalloc(&dout, size_t(k::kMaxStreamRows) * Hq * D * 4);
  cudaMalloc(&dscr, k::attention_rows_scratch_floats(k::kMaxStreamRows, Hq, D, ctx) * 4);
  k::TreeDesc* dtd = upload(std::vector<k::TreeDesc>{td});
  k::attn_prepare(dqg, dk, dv, dqn, dkn, pos0, V, Hq, Hkv, D, rot, theta, eps, dq, dgate, dkc, dvc, 0, dtd);
  k::attention_rows(dq, dkc, dvc, pos0, V, Hq, Hkv, D, dscr, dout, 0, 0, dtd);
  const std::vector<float> tree_out = download(dout, size_t(V) * Hq * D);
  const std::vector<float> tree_q = download(dq, size_t(V) * Hq * D);
  int bad = 0, checked = 0;
  for (int v = 0; v < td.nvar; ++v) {
    int leaf = 0;
    for (int r = 0; r < V; ++r)
      if (t.children[r].empty() && td.var[r] == v && td.path[v][t.depth[r]] == r) leaf = r;
    const int P = t.depth[leaf] + 1;
    std::vector<float> cqg(size_t(P) * Hq * 2 * D), ck(size_t(P) * Hkv * D), cv(ck.size());
    for (int j = 0; j < P; ++j) {
      const int r = td.path[v][j];
      std::memcpy(&cqg[size_t(j) * Hq * 2 * D], &qg[size_t(r) * Hq * 2 * D], size_t(Hq) * 2 * D * 4);
      std::memcpy(&ck[size_t(j) * Hkv * D], &kk[size_t(r) * Hkv * D], size_t(Hkv) * D * 4);
      std::memcpy(&cv[size_t(j) * Hkv * D], &vv[size_t(r) * Hkv * D], size_t(Hkv) * D * 4);
    }
    __nv_bfloat16 *ckc = upload(kc), *cvc = upload(vc);
    float *dcqg = upload(cqg), *dck = upload(ck), *dcv = upload(cv);
    k::attn_prepare(dcqg, dck, dcv, dqn, dkn, pos0, P, Hq, Hkv, D, rot, theta, eps, dq, dgate, ckc, cvc, 0);
    k::attention_rows(dq, ckc, cvc, pos0, P, Hq, Hkv, D, dscr, dout, 0, 0);
    const std::vector<float> chain_out = download(dout, size_t(P) * Hq * D);
    const std::vector<float> chain_q = download(dq, size_t(P) * Hq * D);
    for (int j = 0; j < P; ++j) {
      const int r = td.path[v][j];
      bad += compare(&tree_out[size_t(r) * Hq * D], &chain_out[size_t(j) * Hq * D], size_t(Hq) * D);
      bad += compare(&tree_q[size_t(r) * Hq * D], &chain_q[size_t(j) * Hq * D], size_t(Hq) * D);
      ++checked;
    }
    cudaFree(ckc), cudaFree(cvc), cudaFree(dcqg), cudaFree(dck), cudaFree(dcv);
  }
  cudaFree(dkc), cudaFree(dvc), cudaFree(dqn), cudaFree(dkn), cudaFree(dqg), cudaFree(dk), cudaFree(dv);
  cudaFree(dq), cudaFree(dgate), cudaFree(dout), cudaFree(dscr), cudaFree(dtd);
  const cudaError_t e = cudaGetLastError();
  std::printf("attention tree pos %d, %d rows, %d variants: %d path rows checked, %d differ%s %s\n", pos0, V, td.nvar,
              checked, bad, e != cudaSuccess ? cudaGetErrorString(e) : "", bad == 0 && e == cudaSuccess ? "ok" : "FAIL");
  return bad == 0 && e == cudaSuccess;
}

// DeltaNet: the tree conv and recurrence against each leaf path's chain (gdn_conv in place, gdn_recurrent).
bool check_gdn(std::mt19937_64& rng, int E, int nodes, int width) {
  const int H = 16, HV = 48, Dk = 128, Dv = 128, C = 2 * H * Dk + HV * Dv;
  const ling::DraftTree t = random_tree(rng, E, nodes, width);
  k::TreeDesc td;
  const int ngdn = ling::describe_tree(t, td);
  const int V = t.rows();
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> uu(0.f, 1.f);
  std::vector<float> pre(size_t(V) * C), g(size_t(V) * HV), beta(g.size()), state(size_t(HV) * Dk * Dv), conv(size_t(C) * 3);
  for (float& x : pre) x = nd(rng);
  for (float& x : g) x = -0.5f * uu(rng);
  for (float& x : beta) x = uu(rng);
  for (float& x : state) x = 0.05f * nd(rng);
  for (float& x : conv) x = nd(rng);
  std::vector<__nv_bfloat16> w(size_t(C) * 4);
  for (auto& x : w) x = __float2bfloat16(0.5f * nd(rng));
  float *dpre = upload(pre), *dg = upload(g), *dbeta = upload(beta), *dstate = upload(state), *dconv = upload(conv);
  __nv_bfloat16* dw = upload(w);
  float *dpost = nullptr, *dout = nullptr;
  cudaMalloc(&dpost, size_t(V) * C * 4);
  cudaMalloc(&dout, size_t(V) * HV * Dv * 4);
  k::TreeDesc* dtd = upload(std::vector<k::TreeDesc>{td});
  k::gdn_conv_tree(dpre, dpost, dconv, dw, V, C, dtd, 0);
  k::gdn_recurrent_tree(dpost, dg, dbeta, dstate, dout, H, HV, dtd, k::kMaxTreeVariants, 0);
  const std::vector<float> tpost = download(dpost, size_t(V) * C), tout = download(dout, size_t(V) * HV * Dv);
  const bool state_kept = download(dstate, state.size()) == state && download(dconv, conv.size()) == conv;
  int bad = 0, checked = 0;
  for (int v = 0; v < td.nvar; ++v) {
    int leaf = 0;
    for (int r = 0; r < V; ++r)
      if (t.children[r].empty() && td.var[r] == v && td.path[v][t.depth[r]] == r) leaf = r;
    const int P = t.depth[leaf] + 1;
    std::vector<float> cm(size_t(P) * C), cg(size_t(P) * HV), cb(cg.size());
    for (int j = 0; j < P; ++j) {
      const int r = td.path[v][j];
      std::memcpy(&cm[size_t(j) * C], &pre[size_t(r) * C], size_t(C) * 4);
      std::memcpy(&cg[size_t(j) * HV], &g[size_t(r) * HV], size_t(HV) * 4);
      std::memcpy(&cb[size_t(j) * HV], &beta[size_t(r) * HV], size_t(HV) * 4);
    }
    float *dcm = upload(cm), *dcg = upload(cg), *dcb = upload(cb), *dtmp = upload(conv);
    k::gdn_conv(dcm, dtmp, dw, P, C, 0);
    k::gdn_recurrent(dcm, dcg, dcb, dstate, nullptr, dout, P, H, HV, 0);
    const std::vector<float> cpost = download(dcm, size_t(P) * C), cout = download(dout, size_t(P) * HV * Dv);
    for (int j = 0; j < P; ++j) {
      const int r = td.path[v][j];
      bad += compare(&tpost[size_t(r) * C], &cpost[size_t(j) * C], C);
      bad += compare(&tout[size_t(r) * HV * Dv], &cout[size_t(j) * HV * Dv], size_t(HV) * Dv);
      ++checked;
    }
    cudaFree(dcm), cudaFree(dcg), cudaFree(dcb), cudaFree(dtmp);
  }
  cudaFree(dpre), cudaFree(dg), cudaFree(dbeta), cudaFree(dstate), cudaFree(dconv), cudaFree(dw), cudaFree(dpost);
  cudaFree(dout), cudaFree(dtd);
  const cudaError_t e = cudaGetLastError();
  const bool ok = bad == 0 && state_kept && e == cudaSuccess;
  std::printf("deltanet tree %d rows, %d gdn paths: %d path rows checked, %d differ, state %s%s %s\n", V, ngdn, checked,
              bad, state_kept ? "unwritten" : "WRITTEN", e != cudaSuccess ? cudaGetErrorString(e) : "", ok ? "ok" : "FAIL");
  return ok;
}

// tree/rows: the recurrence's time over 48 layers by tree size, on trees of the engine's shape (12 drafted positions, the
// chain's first 8, N nodes), each launch on the next of 48 states. (A depth-first walk with the state saved at branch
// points, every row computed once, measured slower, 2026-10-20: 1.77-1.98 ms at 16 rows and 3.03-3.27 at 32 against the
// paths' 1.63-1.65 and 2.21-2.29: the paths run side by side, the walk is one sequence of every row.)
void bench_gdn() {
  const int H = 16, HV = 48, Dk = 128, Dv = 128, C = 2 * H * Dk + HV * Dv, L = 48, reps = 20;
  std::mt19937_64 rng(7);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> uu(0.f, 1.f);
  std::vector<float> pre(size_t(k::kMaxStreamRows) * C), g(size_t(k::kMaxStreamRows) * HV), beta(g.size());
  for (float& x : pre) x = nd(rng);
  for (float& x : g) x = -0.5f * uu(rng);
  for (float& x : beta) x = uu(rng);
  std::vector<float> state(size_t(L) * HV * Dk * Dv);
  for (float& x : state) x = 0.05f * nd(rng);
  float *dpre = upload(pre), *dg = upload(g), *dbeta = upload(beta), *dstate = upload(state), *dout = nullptr;
  cudaMalloc(&dout, size_t(k::kMaxStreamRows) * HV * Dv * 4);
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  for (int nodes : {0, 3, 7, 11, 15, 19, 23}) {
    for (int trial = 0; trial < 3; ++trial) {
      const ling::DraftTree t = random_tree(rng, 12, nodes, 16, nodes ? 8 : 1000);
      k::TreeDesc td;
      const int ngdn = ling::describe_tree(t, td);
      k::TreeDesc* dtd = upload(std::vector<k::TreeDesc>{td});
      float ms = 0;
      for (int pass = 0; pass < 2; ++pass) {
        cudaEventRecord(a);
        for (int r = 0; r < reps; ++r)
          for (int l = 0; l < L; ++l)
            k::gdn_recurrent_tree(dpre, dg, dbeta, dstate + size_t(l) * HV * Dk * Dv, dout, H, HV, dtd, ngdn, 0);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
      }
      std::printf("bench gdn %2d rows, %d paths: %.3f ms per 48 layers\n", t.rows(), ngdn, ms / reps);
      cudaFree(dtd);
    }
  }
  cudaFree(dpre), cudaFree(dg), cudaFree(dbeta), cudaFree(dstate), cudaFree(dout);
}

// tree/rows: the attention's time per full-attention layer at agentlong's contexts, by rows (one block of 96 query rows
// holds 16 tree rows; 17 to 32 take a second block), on trees of the engine's shape.
void bench_attention() {
  const int Hq = 24, Hkv = 4, D = 256, rot = 64, ctx = 65536, reps = 20;
  const float theta = 1e7f, eps = 1e-6f;
  std::mt19937_64 rng(11);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<__nv_bfloat16> kc(size_t(Hkv) * ctx * D);
  for (auto& x : kc) x = __float2bfloat16(nd(rng));
  k::set_kv_layout(size_t(ctx) * D, D);
  __nv_bfloat16 *dkc = upload(kc), *dvc = upload(kc);
  std::vector<float> qg(size_t(k::kMaxStreamRows) * Hq * 2 * D), kk(size_t(k::kMaxStreamRows) * Hkv * D);
  for (float& x : qg) x = nd(rng);
  for (float& x : kk) x = nd(rng);
  std::vector<__nv_bfloat16> qn(D), kn(D);
  for (int i = 0; i < D; ++i) qn[i] = __float2bfloat16(0.1f * nd(rng)), kn[i] = __float2bfloat16(0.1f * nd(rng));
  __nv_bfloat16 *dqn = upload(qn), *dkn = upload(kn);
  float *dqg = upload(qg), *dk = upload(kk), *dv = upload(kk), *dq, *dgate, *dout, *dscr;
  cudaMalloc(&dq, size_t(k::kMaxStreamRows) * Hq * D * 4);
  cudaMalloc(&dgate, size_t(k::kMaxStreamRows) * Hq * D * 4);
  cudaMalloc(&dout, size_t(k::kMaxStreamRows) * Hq * D * 4);
  cudaMalloc(&dscr, k::attention_rows_scratch_floats(k::kMaxStreamRows, Hq, D, ctx) * 4);
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  for (int pos0 : {28000, 57000})
    for (int nodes : {0, 7, 8, 11, 15, 23}) {
      const ling::DraftTree t = random_tree(rng, 12, nodes, 16, nodes ? 8 : 1000);
      k::TreeDesc td;
      ling::describe_tree(t, td);
      k::TreeDesc* dtd = upload(std::vector<k::TreeDesc>{td});
      const int V = t.rows();
      k::attn_prepare(dqg, dk, dv, dqn, dkn, pos0, V, Hq, Hkv, D, rot, theta, eps, dq, dgate, dkc, dvc, 0, dtd);
      float ms = 0;
      for (int pass = 0; pass < 2; ++pass) {
        cudaEventRecord(a);
        for (int r = 0; r < reps; ++r) k::attention_rows(dq, dkc, dvc, pos0, V, Hq, Hkv, D, dscr, dout, 0, 0, dtd);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b);
      }
      std::printf("bench attention pos %d, %2d rows, %2d variants: %.3f ms per layer (x16 layers %.2f ms)\n", pos0, V,
                  td.nvar, ms / reps, 16 * ms / reps);
      cudaFree(dtd);
    }
  cudaFree(dkc), cudaFree(dvc), cudaFree(dqn), cudaFree(dkn), cudaFree(dqg), cudaFree(dk), cudaFree(dv);
  cudaFree(dq), cudaFree(dgate), cudaFree(dout), cudaFree(dscr);
}

}  // namespace

// Batching P1: tree_gather moves a batched sequence's accepted path within its own rows, row0 on in the pass's buffers
// and from its position in its caches; every other row is left as it was.
bool check_gather(int row0, int pos) {
  const int rows = 64, cap = 256, width = 16;  // bytes per row
  std::vector<uint8_t> a(size_t(rows) * width), b(size_t(cap) * width);
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<uint8_t>(i * 7 + 3);
  for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>(i * 13 + 5);
  const std::vector<int> path{0, 2, 3, 7, 11};
  const int n = static_cast<int>(path.size());
  uint8_t *da, *db;
  int* dr;
  cudaMalloc(&da, a.size());
  cudaMalloc(&db, b.size());
  cudaMalloc(&dr, n * sizeof(int));
  cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dr, path.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  const k::GatherBuf bufs[2] = {{reinterpret_cast<char*>(da), size_t(width), width, 0},
                                {reinterpret_cast<char*>(db), size_t(width), width, 1}};
  k::GatherBuf* dbufs;
  cudaMalloc(&dbufs, sizeof(bufs));
  cudaMemcpy(dbufs, bufs, sizeof(bufs), cudaMemcpyHostToDevice);
  k::tree_gather(dbufs, 2, dr, n, k::DevPos(pos), nullptr, row0);
  std::vector<uint8_t> ga(a.size()), gb(b.size());
  cudaMemcpy(ga.data(), da, a.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(gb.data(), db, b.size(), cudaMemcpyDeviceToHost);
  std::vector<uint8_t> ea = a, eb = b;
  for (int j = 0; j < n; ++j) {
    std::memcpy(&ea[size_t(row0 + j) * width], &a[size_t(row0 + path[j]) * width], width);
    std::memcpy(&eb[size_t(pos + j) * width], &b[size_t(pos + path[j]) * width], width);
  }
  const bool ok = ga == ea && gb == eb && cudaGetLastError() == cudaSuccess;
  std::printf("tree_gather row0 %d, position %d: %s\n", row0, pos, ok ? "ok" : "FAIL");
  cudaFree(da), cudaFree(db), cudaFree(dr), cudaFree(dbufs);
  return ok;
}

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--bench") {
    k::prepare_kernels();
    bench_gdn();
    bench_attention();
    return 0;
  }
  std::mt19937_64 rng(26);
  k::prepare_kernels();
  bool ok = true;
  for (int pos0 : {100, 3000, 4090, 4085, 8150})
    for (auto [E, nodes, width] : {std::tuple{11, 20, 4}, std::tuple{11, 4, 16}, std::tuple{7, 24, 2}, std::tuple{11, 0, 1}})
      if (pos0 + E + 1 + nodes <= 8192) ok &= check_attention(rng, pos0, E, nodes, width);
  for (auto [E, nodes, width] : {std::tuple{11, 20, 4}, std::tuple{11, 4, 16}, std::tuple{7, 24, 2}, std::tuple{15, 16, 3},
                                 std::tuple{8, 7, 16}, std::tuple{8, 23, 16}, std::tuple{11, 0, 1}, std::tuple{3, 28, 2},
                                 std::tuple{1, 30, 16}, std::tuple{15, 16, 16}})
    ok &= check_gdn(rng, E, nodes, width);
  for (auto [row0, pos] : {std::pair{0, 100}, std::pair{12, 100}, std::pair{24, 200}, std::pair{40, 7}})
    ok &= check_gather(row0, pos);
  std::printf(ok ? "all tree tests passed\n" : "TREE TESTS FAILED\n");
  return ok ? 0 : 1;
}
