// The megakernel's persistent layers (BACKLOG 28) against the graph path's kernels, bit for bit, on random data at the
// model's shapes. Stage 3: a full-attention layer (q|k|v, the prepare step, the attention over 4096-key ranges, the
// combine with the sigmoid gate, the FP16 copy, the out-projection) at 12 and 16 rows, at positions inside a range,
// across a range boundary and at 24K and 59K, twice in a row (the counters reset themselves). Stage 4: the same layer
// with its tail (the residual add and the post-attention norm inside the launch) against attention, add_inplace and
// rmsnorm_half; the FFN with its tail (the down projection into the residual stream, then the next layer's norm)
// against the four FFN kernels, add_inplace and rmsnorm_half, NVFP4 raw. Model-free.
//   ling-mk-tests                       every case
//   LING_MK_ATTN_BLOCKS=3 ling-mk-tests the same on three blocks (the ticket queue's progress with few resident)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "core/kernels.cuh"

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
template <typename T>
T* alloc(size_t n) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(n, 1) * sizeof(T));
  cudaMemset(d, 0, std::max<size_t>(n, 1) * sizeof(T));
  return d;
}

// A random FP8 matrix [N][K] in the tiled layout.
uint8_t* fp8_matrix(std::mt19937_64& rng, int N, int K, float range) {
  std::uniform_real_distribution<float> u(-range, range);
  std::vector<uint8_t> h(size_t(N) * K);
  for (uint8_t& b : h) b = __nv_fp8_e4m3(u(rng)).__x;
  uint8_t* raw = upload(h);
  uint8_t* tiled = nullptr;
  cudaMalloc(&tiled, k::tiled_bytes_fp8(N, K));
  k::retile_fp8(raw, tiled, N, K, nullptr);
  cudaDeviceSynchronize();
  cudaFree(raw);
  return tiled;
}

std::vector<__nv_bfloat16> bf16_random(std::mt19937_64& rng, size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> u(lo, hi);
  std::vector<__nv_bfloat16> h(n);
  for (auto& x : h) x = __float2bfloat16(u(rng));
  return h;
}

template <typename T>
int same(const char* what, const T* a, const T* b, size_t n) {
  const std::vector<T> ha = download(a, n), hb = download(b, n);
  if (std::memcmp(ha.data(), hb.data(), n * sizeof(T)) == 0) return 0;
  size_t i = 0;
  while (std::memcmp(&ha[i], &hb[i], sizeof(T)) == 0) ++i;
  std::printf("    %s differs from element %zu\n", what, i);
  return 1;
}

// A random NVFP4 matrix [N][K] in the tiled layout (raw scales).
uint8_t* fp4_matrix(std::mt19937_64& rng, int N, int K) {
  std::vector<uint8_t> v(size_t(N) * K / 2), sc(size_t(N) * K / 16);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_real_distribution<float> u(0.002f, 0.05f);
  for (uint8_t& b : v) b = uint8_t(byte(rng));
  for (uint8_t& b : sc) b = __nv_fp8_e4m3(u(rng)).__x;
  uint8_t* dv = upload(v);
  uint8_t* ds = upload(sc);
  uint8_t* tiled = nullptr;
  cudaMalloc(&tiled, k::tiled_bytes_nvfp4(N, K));
  k::retile_nvfp4(dv, ds, tiled, N, K, nullptr);
  cudaDeviceSynchronize();
  cudaFree(dv);
  cudaFree(ds);
  return tiled;
}

int attention_layer(std::mt19937_64& rng, int M, int pos0, bool tail) {
  const int H = 5120, Hq = 24, Hkv = 4, D = 256, rot = 64, qsize = Hq * D, kvsize = Hkv * D, ctx = 65536;
  const float theta = 1e7f, eps = 1e-6f;
  k::set_kv_layout(size_t(ctx) * D, D);  // head-major, as the engine sets it
  // Weights (per-tensor scales as a checkpoint's), norms, and the input norm's output.
  uint8_t* wq = fp8_matrix(rng, 2 * qsize, H, 0.4f);
  uint8_t* wk = fp8_matrix(rng, kvsize, H, 0.4f);
  uint8_t* wv = fp8_matrix(rng, kvsize, H, 0.4f);
  uint8_t* wo = fp8_matrix(rng, H, qsize, 0.4f);
  const float sq = 0.003f, sk = 0.0031f, sv = 0.0029f, so = 0.0027f;
  __nv_bfloat16* q_norm = upload(bf16_random(rng, D, -0.3f, 0.3f));
  __nv_bfloat16* k_norm = upload(bf16_random(rng, D, -0.3f, 0.3f));
  __nv_bfloat16* in_norm = upload(bf16_random(rng, H, -0.3f, 0.3f));
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hx(size_t(M) * H);
  for (float& x : hx) x = nd(rng);
  float* h = upload(hx);
  float* xn = alloc<float>(size_t(M) * H);
  __half* xh = alloc<__half>(size_t(M) * H);
  float* xinv = alloc<float>(M);
  k::rmsnorm_half(h, in_norm, xn, xh, xinv, M, H, eps, true, nullptr);
  // The caches: random keys and values before pos0, the same in both copies.
  const size_t cache = size_t(ctx) * kvsize;
  std::vector<__nv_bfloat16> hk = bf16_random(rng, cache, -2.f, 2.f), hv = bf16_random(rng, cache, -2.f, 2.f);
  __nv_bfloat16 *kc[2], *vc[2];
  for (int i = 0; i < 2; ++i) {
    kc[i] = upload(hk);
    vc[i] = upload(hv);
  }
  // Outputs, per path.
  float *qg[2], *kk[2], *vv[2], *q[2], *gate[2], *scratch[2], *attn[2], *y[2], *xinvo[2];
  __half* xho[2];
  const size_t sc = k::attention_rows_scratch_floats(M, Hq, D, ctx);
  for (int i = 0; i < 2; ++i) {
    qg[i] = alloc<float>(size_t(M) * 2 * qsize);
    kk[i] = alloc<float>(size_t(M) * kvsize);
    vv[i] = alloc<float>(size_t(M) * kvsize);
    q[i] = alloc<float>(size_t(M) * qsize);
    gate[i] = alloc<float>(size_t(M) * qsize);
    scratch[i] = alloc<float>(sc);
    attn[i] = alloc<float>(size_t(M) * qsize);
    y[i] = alloc<float>(size_t(M) * H);
    xho[i] = alloc<__half>(size_t(32) * qsize);
    xinvo[i] = alloc<float>(32);
  }
  int* pos_dev = upload(std::vector<int>{pos0});
  const k::DevPos dpos(pos0, pos_dev, 0);  // as a captured graph passes it
  // The graph path (engine.cpp's attention branch, rows path).
  const k::StreamTarget tq{wq, nullptr, sq, qg[0], 2 * qsize}, tk{wk, nullptr, sk, kk[0], kvsize},
      tv{wv, nullptr, sv, vv[0], kvsize};
  const k::StreamTarget t3[3] = {tq, tk, tv};
  k::stream_gemm_multi(false, xh, xinv, M, t3, 3, H, nullptr);
  k::attn_prepare(qg[0], kk[0], vv[0], q_norm, k_norm, dpos, M, Hq, Hkv, D, rot, theta, eps, q[0], gate[0], kc[0], vc[0],
                  nullptr);
  k::attention_rows(q[0], kc[0], vc[0], dpos, M, Hq, Hkv, D, scratch[0], attn[0], nullptr, ctx);
  k::sigmoid_mul(attn[0], gate[0], M * qsize, nullptr);
  k::to_half_rows(attn[0], M, qsize, xho[0], xinvo[0], nullptr);
  k::stream_gemm_fp8(xho[0], xinvo[0], M, wo, so, y[0], H, qsize, nullptr);
  // Stage 4's reference: the residual add and the post-attention norm as their own kernels.
  __nv_bfloat16* post_norm = upload(bf16_random(rng, H, -0.3f, 0.3f));
  float *hres[2], *pn[2], *pinv[2];
  __half* pxh[2];
  for (int i = 0; i < 2; ++i) {
    hres[i] = upload(hx);
    pn[i] = alloc<float>(size_t(M) * H);
    pxh[i] = alloc<__half>(size_t(M) * H);
    pinv[i] = alloc<float>(M);
  }
  if (tail) {
    k::add_inplace(hres[0], y[0], M * H, nullptr);
    k::rmsnorm_half(hres[0], post_norm, pn[0], pxh[0], pinv[0], M, H, eps, true, nullptr);
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("    graph path: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  // The persistent kernel, twice (the second launch runs on the counters the first one reset).
  unsigned* counters = alloc<unsigned>(k::kMkAttnWords);
  int bad = 0;
  for (int run = 0; run < 2; ++run) {
    cudaMemcpy(kc[1], hk.data(), cache * sizeof(hk[0]), cudaMemcpyHostToDevice);
    cudaMemcpy(vc[1], hv.data(), cache * sizeof(hv[0]), cudaMemcpyHostToDevice);
    cudaMemset(y[1], 0, size_t(M) * H * sizeof(float));
    k::MkAttnParams p{};
    p.xh = xh;
    p.xinv = xinv;
    p.M = M;
    p.K = H;
    p.q = {wq, nullptr, sq, qg[1], 2 * qsize};
    p.k = {wk, nullptr, sk, kk[1], kvsize};
    p.v = {wv, nullptr, sv, vv[1], kvsize};
    p.o = {wo, nullptr, so, y[1], H};
    p.Hq = Hq;
    p.Hkv = Hkv;
    p.D = D;
    p.rot = rot;
    p.theta = theta;
    p.eps = eps;
    p.q_norm = q_norm;
    p.k_norm = k_norm;
    p.q_out = q[1];
    p.gate = gate[1];
    p.kcache = kc[1];
    p.vcache = vc[1];
    p.pos0 = dpos;
    p.scratch = scratch[1];
    p.attn = attn[1];
    p.xh2 = xho[1];
    p.xinv2 = xinvo[1];
    p.counters = counters;
    const k::MkTail tl{hres[1], post_norm, pn[1], pxh[1], pinv[1], eps};
    if (tail) {
      cudaMemcpy(hres[1], hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice);
      p.tail = &tl;
    }
    if (!k::attn_persistent(p, nullptr)) {
      std::printf("    attn_persistent refused the shape\n");
      return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::printf("    attn_persistent: %s\n", cudaGetErrorString(cudaGetLastError()));
      return 1;
    }
    bad += same("q|gate", qg[0], qg[1], size_t(M) * 2 * qsize);
    bad += same("q", q[0], q[1], size_t(M) * qsize);
    bad += same("gate", gate[0], gate[1], size_t(M) * qsize);
    bad += same("k cache", kc[0], kc[1], cache);
    bad += same("v cache", vc[0], vc[1], cache);
    bad += same("attn", attn[0], attn[1], size_t(M) * qsize);
    bad += same("FP16 copy", xho[0], xho[1], size_t(M) * qsize);
    bad += same("row scales", xinvo[0], xinvo[1], M);
    bad += same("out-projection", y[0], y[1], size_t(M) * H);
    if (tail) {
      bad += same("residual", hres[0], hres[1], size_t(M) * H);
      bad += same("post norm", pn[0], pn[1], size_t(M) * H);
      bad += same("post norm FP16", pxh[0], pxh[1], size_t(M) * H);
      bad += same("post norm scales", pinv[0], pinv[1], M);
    }
    const std::vector<unsigned> c = download(counters, k::kMkAttnWords);
    for (unsigned w : c) bad += w != 0;
  }
  for (void* ptr : {(void*)wq, (void*)wk, (void*)wv, (void*)wo, (void*)q_norm, (void*)k_norm, (void*)in_norm, (void*)h,
                    (void*)xn, (void*)xh, (void*)xinv, (void*)pos_dev, (void*)counters})
    cudaFree(ptr);
  cudaFree(post_norm);
  for (int i = 0; i < 2; ++i)
    for (void* ptr : {(void*)hres[i], (void*)pn[i], (void*)pxh[i], (void*)pinv[i]}) cudaFree(ptr);
  for (int i = 0; i < 2; ++i)
    for (void* ptr : {(void*)kc[i], (void*)vc[i], (void*)qg[i], (void*)kk[i], (void*)vv[i], (void*)q[i], (void*)gate[i],
                      (void*)scratch[i], (void*)attn[i], (void*)y[i], (void*)xho[i], (void*)xinvo[i]})
      cudaFree(ptr);
  return bad;
}

// Stage 4: the FFN with its tail (the down projection added into the residual stream, then the next layer's input
// norm) against gate|up, silu_mul, to_half_rows, down, add_inplace and rmsnorm_half.
int ffn_tail(std::mt19937_64& rng, int M, bool norm) {
  const int H = 5120, I = 17408;
  const float eps = 1e-6f;
  uint8_t* wg = fp4_matrix(rng, I, H);
  uint8_t* wu = fp4_matrix(rng, I, H);
  uint8_t* wd = fp4_matrix(rng, H, I);
  const float sg = 0.9f, su = 1.1f, sd = 0.7f;
  __nv_bfloat16* pre = upload(bf16_random(rng, H, -0.3f, 0.3f));
  __nv_bfloat16* next = upload(bf16_random(rng, H, -0.3f, 0.3f));
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hx(size_t(M) * H);
  for (float& x : hx) x = nd(rng);
  float *h[2], *xn[2], *xinv[2], *g[2], *u[2], *t2[2], *xinv2[2];
  __half *xh[2], *xh2[2];
  for (int i = 0; i < 2; ++i) {
    h[i] = upload(hx);
    xn[i] = alloc<float>(size_t(M) * H);
    xh[i] = alloc<__half>(size_t(M) * H);
    xinv[i] = alloc<float>(32);
    g[i] = alloc<float>(size_t(M) * I);
    u[i] = alloc<float>(size_t(M) * I);
    t2[i] = alloc<float>(size_t(M) * H);
    xh2[i] = alloc<__half>(size_t(32) * I);
    xinv2[i] = alloc<float>(32);
    k::rmsnorm_half(h[i], pre, xn[i], xh[i], xinv[i], M, H, eps, true, nullptr);  // the FFN's input
  }
  const k::StreamTarget tg{wg, nullptr, sg, g[0], I}, tu{wu, nullptr, su, u[0], I};
  const k::StreamTarget gu[2] = {tg, tu};
  k::stream_gemm_multi(true, xh[0], xinv[0], M, gu, 2, H, nullptr);
  k::silu_mul(g[0], u[0], g[0], M * I, nullptr);
  k::to_half_rows(g[0], M, I, xh2[0], xinv2[0], nullptr);
  k::stream_gemm_nvfp4(xh2[0], xinv2[0], M, wd, nullptr, sd, t2[0], H, I, nullptr);
  k::add_inplace(h[0], t2[0], M * H, nullptr);
  if (norm) k::rmsnorm_half(h[0], next, xn[0], xh[0], xinv[0], M, H, eps, true, nullptr);
  unsigned* counters = alloc<unsigned>(k::kMkFfnWords);
  const k::MkTail tl{h[1], norm ? next : nullptr, xn[1], xh[1], xinv[1], eps};
  const bool ok = k::ffn_persistent(xh[1], xinv[1], M, {wg, nullptr, sg, g[1], I}, {wu, nullptr, su, u[1], I},
                                    {wd, nullptr, sd, t2[1], H}, H, I, g[1], u[1], xh2[1], xinv2[1], counters, nullptr,
                                    nullptr, &tl);
  if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("    ffn_persistent: %s\n", ok ? cudaGetErrorString(cudaGetLastError()) : "refused");
    return 1;
  }
  int bad = same("residual", h[0], h[1], size_t(M) * H);
  if (norm) {
    bad += same("next norm", xn[0], xn[1], size_t(M) * H);
    bad += same("next norm FP16", xh[0], xh[1], size_t(M) * H);
    bad += same("next norm scales", xinv[0], xinv[1], M);
  }
  for (unsigned w : download(counters, k::kMkFfnWords)) bad += w != 0;
  for (void* ptr : {(void*)wg, (void*)wu, (void*)wd, (void*)pre, (void*)next, (void*)counters}) cudaFree(ptr);
  for (int i = 0; i < 2; ++i)
    for (void* ptr : {(void*)h[i], (void*)xn[i], (void*)xh[i], (void*)xinv[i], (void*)g[i], (void*)u[i], (void*)t2[i],
                      (void*)xh2[i], (void*)xinv2[i]})
      cudaFree(ptr);
  return bad;
}

// Stage 4 (LING_MK_SPAN): a DeltaNet layer's mixer and FFN in one launch against gdn_persistent and ffn_persistent with
// their tails (each checked against the graph path's kernels), bit for bit: the residual stream, the next norm and the
// mixer's intermediates.
int span_layer(std::mt19937_64& rng, int M) {
  const int H = 5120, I = 17408, Hk = 16, HV = 48, Dk = 128, C = 2 * Hk * Dk + HV * Dk, Vs = HV * Dk;
  const float eps = 1e-6f;
  uint8_t* wqkv = fp8_matrix(rng, C, H, 0.4f);
  uint8_t* wz = fp8_matrix(rng, Vs, H, 0.4f);
  uint8_t* wout = fp8_matrix(rng, H, Vs, 0.4f);
  uint8_t* wg = fp4_matrix(rng, I, H);
  uint8_t* wu = fp4_matrix(rng, I, H);
  uint8_t* wd = fp4_matrix(rng, H, I);
  __nv_bfloat16* wa = upload(bf16_random(rng, size_t(HV) * H, -0.02f, 0.02f));
  __nv_bfloat16* wb = upload(bf16_random(rng, size_t(HV) * H, -0.02f, 0.02f));
  __nv_bfloat16* A_log = upload(bf16_random(rng, HV, -1.f, 1.f));
  __nv_bfloat16* dt_bias = upload(bf16_random(rng, HV, -1.f, 1.f));
  __nv_bfloat16* conv_w = upload(bf16_random(rng, size_t(C) * 4, -0.5f, 0.5f));
  __nv_bfloat16* lin_norm = upload(bf16_random(rng, Dk, 0.5f, 1.5f));
  __nv_bfloat16* in_norm = upload(bf16_random(rng, H, -0.3f, 0.3f));
  __nv_bfloat16* post_norm = upload(bf16_random(rng, H, -0.3f, 0.3f));
  __nv_bfloat16* next_norm = upload(bf16_random(rng, H, -0.3f, 0.3f));
  std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> hcs(size_t(C) * 3), hst(size_t(HV) * Dk * Dk), hx(size_t(M) * H);
  for (float& x : hcs) x = nd(rng);
  for (float& x : hst) x = 0.05f * nd(rng);
  for (float& x : hx) x = nd(rng);
  float* conv_state = upload(hcs);
  float* state = upload(hst);
  struct Bufs {
    float *h, *xn, *xinv, *vpre, *z, *a, *b, *g, *beta, *mixed, *core, *normed, *xinv2, *xinv3, *t1, *t2, *xinv4;
    __half *xh, *xh2, *xh3, *xh4;
    unsigned *cg, *cf;
  } B[2];
  for (Bufs& b : B) {
    b.h = upload(hx);
    b.xn = alloc<float>(size_t(M) * H);
    b.xh = alloc<__half>(size_t(M) * H);
    b.xinv = alloc<float>(32);
    k::rmsnorm_half(b.h, in_norm, b.xn, b.xh, b.xinv, M, H, eps, true, nullptr);
    b.vpre = alloc<float>(size_t(M) * C);
    b.z = alloc<float>(size_t(M) * Vs);
    for (float** p : {&b.a, &b.b, &b.g, &b.beta}) *p = alloc<float>(size_t(M) * HV);
    b.mixed = alloc<float>(size_t(M) * C);
    b.core = alloc<float>(size_t(M) * Vs);
    b.normed = alloc<float>(size_t(M) * Vs);
    b.xh2 = alloc<__half>(size_t(32) * Vs);
    b.xinv2 = alloc<float>(32);
    b.xh3 = alloc<__half>(size_t(32) * H);
    b.xinv3 = alloc<float>(32);
    b.t1 = alloc<float>(size_t(M) * I);
    b.t2 = alloc<float>(size_t(M) * I);
    b.xh4 = alloc<__half>(size_t(32) * I);
    b.xinv4 = alloc<float>(32);
    b.cg = alloc<unsigned>(k::kMkGdnWords);
    b.cf = alloc<unsigned>(k::kMkFfnWords);
  }
  auto params = [&](Bufs& b, const k::MkTail* tail) {
    k::MkGdnParams p{};
    p.xh = b.xh;
    p.xinv = b.xinv;
    p.xn = b.xn;
    p.M = M;
    p.K = H;
    p.in_qkv = {wqkv, nullptr, 0.003f, b.vpre, C};
    p.in_z = {wz, nullptr, 0.0031f, b.z, Vs};
    p.out = {wout, nullptr, 0.0029f, nullptr, H};
    p.wa = wa;
    p.wb = wb;
    p.a = b.a;
    p.b = b.b;
    p.g = b.g;
    p.beta = b.beta;
    p.A_log = A_log;
    p.dt_bias = dt_bias;
    p.mixed = b.mixed;
    p.conv_state = conv_state;
    p.conv_w = conv_w;
    p.state = state;
    p.core = b.core;
    p.norm_w = lin_norm;
    p.eps = eps;
    p.normed = b.normed;
    p.xh2 = b.xh2;
    p.xinv2 = b.xinv2;
    p.C = C;
    p.H = Hk;
    p.HV = HV;
    p.Dk = Dk;
    p.Dv = Dk;
    p.Vs = Vs;
    p.counters = b.cg;
    p.tail = tail;
    return p;
  };
  const k::StreamTarget tg[2] = {{wg, nullptr, 0.9f, B[0].t1, I}, {wg, nullptr, 0.9f, B[1].t1, I}};
  const k::StreamTarget tu[2] = {{wu, nullptr, 1.1f, B[0].t2, I}, {wu, nullptr, 1.1f, B[1].t2, I}};
  const k::StreamTarget td[2] = {{wd, nullptr, 0.7f, B[0].t2, H}, {wd, nullptr, 0.7f, B[1].t2, H}};
  // The two launches.
  const k::MkTail post0{B[0].h, post_norm, B[0].xn, B[0].xh3, B[0].xinv3, eps};
  const k::MkTail next0{B[0].h, next_norm, B[0].xn, B[0].xh, B[0].xinv, eps};
  const k::MkGdnParams p0 = params(B[0], &post0);
  bool ok = k::gdn_persistent(p0, nullptr) &&
            k::ffn_persistent(B[0].xh3, B[0].xinv3, M, tg[0], tu[0], td[0], H, I, B[0].t1, B[0].t2, B[0].xh4,
                              B[0].xinv4, B[0].cf, nullptr, nullptr, &next0);
  // The spanning one.
  const k::MkTail post1{B[1].h, post_norm, B[1].xn, B[1].xh3, B[1].xinv3, eps};
  const k::MkTail next1{B[1].h, next_norm, B[1].xn, B[1].xh, B[1].xinv, eps};
  const k::MkGdnParams p1 = params(B[1], &post1);
  ok = ok && k::gdn_ffn_persistent(p1, tg[1], tu[1], td[1], I, B[1].t1, B[1].t2, B[1].xh4, B[1].xinv4, B[1].cf, nullptr,
                                   &next1, nullptr);
  if (!ok || cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("    %s\n", ok ? cudaGetErrorString(cudaGetLastError()) : "a launch refused the shape");
    return 1;
  }
  int bad = same("in_qkv", B[0].vpre, B[1].vpre, size_t(M) * C);
  bad += same("core", B[0].core, B[1].core, size_t(M) * Vs);
  bad += same("post norm FP16", B[0].xh3, B[1].xh3, size_t(M) * H);
  bad += same("residual", B[0].h, B[1].h, size_t(M) * H);
  bad += same("next norm", B[0].xn, B[1].xn, size_t(M) * H);
  bad += same("next norm FP16", B[0].xh, B[1].xh, size_t(M) * H);
  bad += same("next norm scales", B[0].xinv, B[1].xinv, M);
  for (const Bufs& b : B) {
    for (unsigned w : download(b.cg, k::kMkGdnWords)) bad += w != 0;
    for (unsigned w : download(b.cf, k::kMkFfnWords)) bad += w != 0;
  }
  return bad;  // (test buffers are left to the process's exit)
}

}  // namespace

int main() {
  std::mt19937_64 rng(28);
  k::set_pdl(true);
  k::set_attention_prefetch(2);
  k::prepare_kernels();
  k::attn_persistent_prepare();
  int failed = 0;
  for (int M : {16, 12})
    for (int pos : {100, 4090, 24000, 59000}) {
      const int bad = attention_layer(rng, M, pos, false);
      std::printf("attention layer, %d rows at %d: %s\n", M, pos, bad ? "DIFFERS" : "bit for bit");
      failed += bad != 0;
    }
  // Stage 4 (LING_MK_LAYERS).
  for (int M : {16, 12}) {
    int bad = attention_layer(rng, M, 24000, true);
    std::printf("attention layer with its tail, %d rows at 24000: %s\n", M, bad ? "DIFFERS" : "bit for bit");
    failed += bad != 0;
    for (bool norm : {true, false}) {
      bad = ffn_tail(rng, M, norm);
      std::printf("FFN with its tail (%s), %d rows: %s\n", norm ? "the next norm" : "the add only", M,
                  bad ? "DIFFERS" : "bit for bit");
      failed += bad != 0;
    }
    bad = span_layer(rng, M);
    std::printf("DeltaNet layer in one launch, %d rows: %s\n", M, bad ? "DIFFERS" : "bit for bit");
    failed += bad != 0;
  }
  std::printf("%s\n", failed ? "FAILED" : "all passed");
  return failed ? 1 : 0;
}
