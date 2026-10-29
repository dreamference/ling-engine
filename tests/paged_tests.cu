// ling-paged-tests: P4's paged KV (specs/DREAMFERENCE_LING_ENGINE_BATCHING.md §5.3) against the flat head-major
// layout, bit for bit. The same random KV is laid out flat ([Hkv][max_context][D] per layer, as the engine's slots)
// and in 64-position pages of a pool behind a shuffled page table (as --kv-pages), and every kernel that addresses the
// KV runs on both: attention_rows (rows 1-32, contexts across page, tile and key-range boundaries, cp.async and bulk
// tiles, L2 prefetch on and off, eager and with a device position over fixed ranges as a graph launches it, a tree
// verify whose path crosses a page), attention_rows_seqs and attn_prepare_seqs over three sequences, attn_prepare's
// writes, tree_gather and kv_pages_copy. memcmp of the outputs, and of the caches read back position by position.
// P4b: kv_table_set, the page table's update. Exits 0 when everything is identical. With the argument `head-major`,
// the pool is laid out as LING_KV_HEAD_MAJOR=1 lays it out ([layer][K, V][kv head][page][PT][D], the table in units of
// one head's run).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <numeric>
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
  cudaDeviceSynchronize();
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
bool same(const void* a, const void* b, size_t bytes) { return std::memcmp(a, b, bytes) == 0; }

constexpr int Hq = 24, Hkv = 4, D = 256, PT = k::kKvPageTokens;
constexpr int NL = 2;  // full-attention layers in a page; the tests read layer 1 (a non-zero offset inside the page)
constexpr int LAYER = 1;
constexpr size_t kPageElems = size_t(NL) * 2 * Hkv * PT * D;

std::mt19937 rng(2027);
std::normal_distribution<float> normal(0.f, 1.f);

std::vector<float> randv(size_t n, float scale = 1.f) {
  std::vector<float> v(n);
  for (auto& x : v) x = normal(rng) * scale;
  return v;
}
std::vector<__nv_bfloat16> randb(size_t n, float scale = 1.f) {
  std::vector<__nv_bfloat16> v(n);
  for (auto& x : v) x = __float2bfloat16(normal(rng) * scale);
  return v;
}

// The pool's layout: its head stride and the elements per page id in a table (set by Pool).
bool g_head_major = false;
size_t g_hs = size_t(PT) * D, g_unit = kPageElems;

void flat_layout(int max_ctx) { k::set_kv_layout(size_t(max_ctx) * D, D); }
void paged_layout() { k::set_kv_layout(g_hs, D); }

// One sequence's KV of one layer in both layouts. The pool holds every sequence's pages (page 0 unused); a sequence's
// table maps its pages in a shuffled order, so neighbouring positions across a page boundary are far apart.
struct Pool {
  std::vector<__nv_bfloat16> host;
  __nv_bfloat16* dev = nullptr;
  int pages = 0;
  explicit Pool(int n) : host(kPageElems * (n + 1)), pages(n + 1) {
    std::memset(host.data(), 0x3f, host.size() * 2);  // 0.746 wherever nothing is mapped (page 0, the other layer)
    g_hs = g_head_major ? size_t(pages) * PT * D : size_t(PT) * D;
    g_unit = g_head_major ? size_t(PT) * D : kPageElems;
  }
  __nv_bfloat16* kbase() const { return dev + size_t(LAYER) * 2 * Hkv * g_hs; }
  __nv_bfloat16* vbase() const { return kbase() + size_t(Hkv) * g_hs; }
  size_t at(uint64_t page_off, int kv, int h, int p) const {
    return page_off + (size_t(LAYER) * 2 + kv) * Hkv * g_hs + size_t(h) * g_hs + size_t(p % PT) * D;
  }
};

struct Seq {
  int max_ctx = 0;
  std::vector<__nv_bfloat16> k, v;  // flat [Hkv][max_ctx][D]
  std::vector<uint64_t> table;      // element offsets into the pool
  __nv_bfloat16 *dk = nullptr, *dv = nullptr;
  uint64_t* dtable = nullptr;
};

// A sequence of max_ctx positions with random KV, its pages taken from `ids` (consumed), written into the pool.
Seq make_seq(Pool& pool, int max_ctx, std::vector<int>& ids) {
  Seq s;
  s.max_ctx = max_ctx;
  s.k = randb(size_t(Hkv) * max_ctx * D);
  s.v = randb(s.k.size());
  const int np = (max_ctx + PT - 1) / PT;
  for (int j = 0; j < np; ++j) {
    s.table.push_back(uint64_t(ids.back()) * g_unit);
    ids.pop_back();
  }
  for (int h = 0; h < Hkv; ++h)
    for (int p = 0; p < max_ctx; ++p)
      for (int d = 0; d < D; ++d) {
        pool.host[pool.at(s.table[p / PT], 0, h, p) + d] = s.k[(size_t(h) * max_ctx + p) * D + d];
        pool.host[pool.at(s.table[p / PT], 1, h, p) + d] = s.v[(size_t(h) * max_ctx + p) * D + d];
      }
  return s;
}
void upload_seq(Seq& s) {
  s.dk = upload(s.k);
  s.dv = upload(s.v);
  s.dtable = upload(s.table);
}
// The pool's KV of sequence s read back into the flat layout.
std::vector<__nv_bfloat16> read_back(const Pool& pool, const std::vector<__nv_bfloat16>& dev_pool, const Seq& s, int kv) {
  std::vector<__nv_bfloat16> out(size_t(Hkv) * s.max_ctx * D);
  for (int h = 0; h < Hkv; ++h)
    for (int p = 0; p < s.max_ctx; ++p)
      std::memcpy(&out[(size_t(h) * s.max_ctx + p) * D], &dev_pool[pool.at(s.table[p / PT], kv, h, p)], D * 2);
  return out;
}

std::vector<int> shuffled_ids(int n) {
  std::vector<int> ids(n);
  std::iota(ids.begin(), ids.end(), 1);
  std::shuffle(ids.begin(), ids.end(), rng);
  return ids;
}

int failures = 0;
void report(const char* what, bool ok) {
  std::printf("%-100s %s\n", what, ok ? "identical" : "DIFFERENT");
  if (!ok) ++failures;
}

// attention_rows at M rows from position P, flat against paged, under the current tile settings.
bool attention_case(const Seq& s, const Pool& pool, const float* dq, int P, int M, float* scratch, float* out,
                    const k::TreeDesc* td = nullptr, bool graph = false) {
  const size_t n = size_t(M) * Hq * D;
  int* dpos = graph ? upload(std::vector<int>{P}) : nullptr;
  const k::DevPos pos = graph ? k::DevPos(-1, dpos, 0) : k::DevPos(P);
  const int fixed = graph ? s.max_ctx : 0;
  flat_layout(s.max_ctx);
  k::attention_rows(dq, s.dk, s.dv, pos, M, Hq, Hkv, D, scratch, out, nullptr, fixed, td);
  const std::vector<float> a = download(out, n);
  cudaMemset(out, 0, n * 4);
  paged_layout();
  k::attention_rows(dq, pool.kbase(), pool.vbase(), pos, M, Hq, Hkv, D, scratch, out, nullptr, fixed, td, s.dtable);
  const std::vector<float> b = download(out, n);
  if (dpos) cudaFree(dpos);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) std::printf("CUDA error: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess && same(a.data(), b.data(), n * 4);
}

}  // namespace

int main(int argc, char** argv) {
  g_head_major = argc > 1 && std::strcmp(argv[1], "head-major") == 0;
  std::printf("pool layout: %s\n", g_head_major ? "head-major" : "interleaved");
  k::prepare_kernels();
  const int kMaxCtx = 131072 + 64;
  Pool pool((kMaxCtx / PT + 1) + 3 * (8192 / PT) + 8);
  std::vector<int> ids = shuffled_ids(pool.pages - 1);
  Seq big = make_seq(pool, kMaxCtx, ids);
  std::vector<Seq> seqs;
  for (int i = 0; i < 3; ++i) seqs.push_back(make_seq(pool, 8192, ids));
  pool.dev = upload(pool.host);
  upload_seq(big);
  for (Seq& s : seqs) upload_seq(s);

  float* dq = upload(randv(size_t(32) * Hq * D, 2.f));
  float *scratch, *out;
  cudaMalloc(&scratch, k::attention_rows_scratch_floats(256, Hq, D, kMaxCtx) * sizeof(float));
  cudaMalloc(&out, size_t(256) * Hq * D * sizeof(float));

  // ---- attention_rows ----
  const int contexts[] = {1, 33, 63, 64, 65, 127, 4095, 4096, 4097, 27000, 131072};
  for (int bulk = 0; bulk < 2; ++bulk)
    for (int pf : {0, 2}) {
      k::set_attention_bulk(bulk != 0);
      k::set_attention_prefetch(pf);
      bool ok = true;
      for (int ctx : contexts)
        for (int M : {1, 12, 16, 24, 32})
          if (ctx >= M) ok &= attention_case(big, pool, dq, ctx - M, M, scratch, out);
      char what[128];
      std::snprintf(what, sizeof what, "attention_rows, %s tiles, prefetch %d, 11 contexts x rows 1-32", bulk ? "bulk" : "cp.async",
                    pf);
      report(what, ok);
    }
  k::set_attention_bulk(false);
  k::set_attention_prefetch(2);
  {
    bool ok = true;
    for (int P : {60, 4090, 27001}) ok &= attention_case(big, pool, dq, P, 16, scratch, out, nullptr, true);
    report("attention_rows as a graph launches it (device position, every range up to the context)", ok);
  }
  // A prefill slice: 256 rows (the engine's kAttnSlice) straddling pages and a key range.
  {
    float* dq256 = upload(randv(size_t(256) * Hq * D, 2.f));
    bool ok = true;
    for (int P : {0, 3900, 70000}) ok &= attention_case(big, pool, dq256, P, 256, scratch, out);
    cudaFree(dq256);
    report("attention_rows over a 256-row prefill slice", ok);
  }
  // ---- a tree verify whose path crosses a page: rows 0-4 one variant (depths 0-4), 5-7 another (depths 2-4) ----
  {
    k::TreeDesc td;
    td.n = 8;
    const int depth[8] = {0, 1, 2, 3, 4, 2, 3, 4}, var[8] = {0, 0, 0, 0, 0, 1, 1, 1};
    for (int m = 0; m < 8; ++m) td.depth[m] = depth[m], td.var[m] = var[m];
    td.nvar = 2;
    const int p0[5] = {0, 1, 2, 3, 4}, p1[5] = {0, 1, 5, 6, 7};
    for (int j = 0; j < 5; ++j) td.path[0][j] = p0[j], td.path[1][j] = p1[j];
    k::TreeDesc* dtd = upload(std::vector<k::TreeDesc>{td});
    bool ok = true;
    for (int P : {58, 60, 62, 4093}) ok &= attention_case(big, pool, dq, P, 8, scratch, out, dtd);
    report("attention_rows, a tree verify (paths across a page boundary)", ok);
    // attn_prepare with the tree: rows to slots P + m, RoPE at the depth's position.
    const int M = 8, P = 61, rot = 64;
    float* qg = upload(randv(size_t(M) * Hq * 2 * D));
    float* kk = upload(randv(size_t(M) * Hkv * D));
    float* vv = upload(randv(size_t(M) * Hkv * D));
    __nv_bfloat16* qn = upload(randb(D, 0.1f));
    __nv_bfloat16* kn = upload(randb(D, 0.1f));
    float *q1, *g1, *q2, *g2;
    const size_t qn_ = size_t(M) * Hq * D;
    cudaMalloc(&q1, qn_ * 4), cudaMalloc(&g1, qn_ * 4), cudaMalloc(&q2, qn_ * 4), cudaMalloc(&g2, qn_ * 4);
    flat_layout(big.max_ctx);
    k::attn_prepare(qg, kk, vv, qn, kn, P, M, Hq, Hkv, D, rot, 1e7f, 1e-6f, q1, g1, big.dk, big.dv, nullptr, dtd);
    paged_layout();
    k::attn_prepare(qg, kk, vv, qn, kn, P, M, Hq, Hkv, D, rot, 1e7f, 1e-6f, q2, g2, pool.kbase(), pool.vbase(), nullptr,
                    dtd, nullptr, big.dtable);
    const std::vector<__nv_bfloat16> dev_pool = download(pool.dev, pool.host.size());
    bool prep = same(download(q1, qn_).data(), download(q2, qn_).data(), qn_ * 4) &&
                same(download(g1, qn_).data(), download(g2, qn_).data(), qn_ * 4);
    report("attn_prepare (a tree's rows into slots across a page): q and gate", prep);
    // The caches compared as bytes (operator== on __nv_bfloat16 compares values; use memcmp).
    const std::vector<__nv_bfloat16> fk = download(big.dk, big.k.size()), fv = download(big.dv, big.v.size());
    const std::vector<__nv_bfloat16> pk = read_back(pool, dev_pool, big, 0), pv = read_back(pool, dev_pool, big, 1);
    report("attn_prepare: the K and V caches, position by position",
           same(fk.data(), pk.data(), fk.size() * 2) && same(fv.data(), pv.data(), fv.size() * 2));
    // tree_gather: the accepted path 0, 1, 5, 6, 7 moved to slots P .. P + 4, every head of K and V.
    const std::vector<int> rows = {0, 1, 5, 6, 7};
    int* drows = upload(rows);
    std::vector<k::GatherBuf> gf, gp;
    for (int h = 0; h < Hkv; ++h) {
      gf.push_back({reinterpret_cast<char*>(big.dk + size_t(h) * big.max_ctx * D), D * 2, D * 2, 1});
      gf.push_back({reinterpret_cast<char*>(big.dv + size_t(h) * big.max_ctx * D), D * 2, D * 2, 1});
      gp.push_back({reinterpret_cast<char*>(pool.kbase() + size_t(h) * g_hs), D * 2, D * 2, 1, big.dtable, 2});
      gp.push_back({reinterpret_cast<char*>(pool.vbase() + size_t(h) * g_hs), D * 2, D * 2, 1, big.dtable, 2});
    }
    k::GatherBuf* dgf = upload(gf);
    k::GatherBuf* dgp = upload(gp);
    k::tree_gather(dgf, int(gf.size()), drows, int(rows.size()), k::DevPos(P), nullptr);
    k::tree_gather(dgp, int(gp.size()), drows, int(rows.size()), k::DevPos(P), nullptr);
    const std::vector<__nv_bfloat16> dev_pool2 = download(pool.dev, pool.host.size());
    const std::vector<__nv_bfloat16> gk = download(big.dk, big.k.size()), gv = download(big.dv, big.v.size());
    const std::vector<__nv_bfloat16> qk = read_back(pool, dev_pool2, big, 0), qv = read_back(pool, dev_pool2, big, 1);
    report("tree_gather (rows moved across a page boundary)",
           same(gk.data(), qk.data(), gk.size() * 2) && same(gv.data(), qv.data(), gv.size() * 2));
    for (void* p : {(void*)qg, (void*)kk, (void*)vv, (void*)qn, (void*)kn, (void*)q1, (void*)g1, (void*)q2, (void*)g2,
                    (void*)dtd, (void*)drows, (void*)dgf, (void*)dgp})
      cudaFree(p);
  }
  // ---- three sequences in one launch: attn_prepare_seqs and attention_rows_seqs, each with its own table ----
  {
    const int rows[3] = {4, 12, 12}, pos[3] = {100, 5000, 63};
    const int M = 28, rot = 64;
    k::SeqBatch sb;
    sb.n = 3;
    for (int i = 0; i < 3; ++i) sb.row0[i + 1] = sb.row0[i] + rows[i], sb.pos[i] = pos[i];
    float* qg = upload(randv(size_t(M) * Hq * 2 * D));
    float* kk = upload(randv(size_t(M) * Hkv * D));
    float* vv = upload(randv(size_t(M) * Hkv * D));
    __nv_bfloat16* qn = upload(randb(D, 0.1f));
    __nv_bfloat16* kn = upload(randb(D, 0.1f));
    const size_t qn_ = size_t(M) * Hq * D;
    float *q1, *g1, *q2, *g2, *o1, *o2;
    for (float** p : {&q1, &g1, &q2, &g2, &o1, &o2}) cudaMalloc(p, qn_ * 4);
    k::SeqPtrs<__nv_bfloat16> kf, vf, kp, vp;
    k::SeqPtrs<const __nv_bfloat16> kfr, vfr, kpr, vpr;
    k::SeqPtrs<const uint64_t> pg;
    for (int i = 0; i < 3; ++i) {
      kfr.p[i] = kf.p[i] = seqs[i].dk, vfr.p[i] = vf.p[i] = seqs[i].dv;
      kpr.p[i] = kp.p[i] = pool.kbase(), vpr.p[i] = vp.p[i] = pool.vbase();
      pg.p[i] = seqs[i].dtable;
    }
    flat_layout(8192);
    k::attn_prepare_seqs(qg, kk, vv, qn, kn, sb, Hq, Hkv, D, rot, 1e7f, 1e-6f, q1, g1, kf, vf, nullptr);
    k::attention_rows_seqs(q1, kfr, vfr, sb, Hq, Hkv, D, scratch, o1, nullptr);
    paged_layout();
    k::attn_prepare_seqs(qg, kk, vv, qn, kn, sb, Hq, Hkv, D, rot, 1e7f, 1e-6f, q2, g2, kp, vp, nullptr, pg);
    k::attention_rows_seqs(q2, kpr, vpr, sb, Hq, Hkv, D, scratch, o2, nullptr, 0, pg);
    const std::vector<__nv_bfloat16> dev_pool = download(pool.dev, pool.host.size());
    bool ok = same(download(q1, qn_).data(), download(q2, qn_).data(), qn_ * 4) &&
              same(download(o1, qn_).data(), download(o2, qn_).data(), qn_ * 4);
    for (const Seq& s : seqs) {
      const std::vector<__nv_bfloat16> fk = download(s.dk, s.k.size()), fv = download(s.dv, s.v.size());
      const std::vector<__nv_bfloat16> pk = read_back(pool, dev_pool, s, 0), pv = read_back(pool, dev_pool, s, 1);
      ok &= same(fk.data(), pk.data(), fk.size() * 2) && same(fv.data(), pv.data(), fv.size() * 2);
    }
    // As a graph launches it: device positions, every range up to the context.
    int* dpos = upload(std::vector<int>{pos[0], pos[1], pos[2]});
    k::SeqBatch gb = sb;
    gb.pos[0] = gb.pos[1] = gb.pos[2] = -1;
    gb.pos_dev = dpos;
    cudaMemset(o2, 0, qn_ * 4);
    k::attention_rows_seqs(q2, kpr, vpr, gb, Hq, Hkv, D, scratch, o2, nullptr, 8192, pg);
    ok &= same(download(o1, qn_).data(), download(o2, qn_).data(), qn_ * 4);
    report("attn_prepare_seqs and attention_rows_seqs over three sequences (eager and graph form), caches too", ok);
    for (void* p : {(void*)qg, (void*)kk, (void*)vv, (void*)qn, (void*)kn, (void*)q1, (void*)g1, (void*)q2, (void*)g2,
                    (void*)o1, (void*)o2, (void*)dpos})
      cudaFree(p);
  }
  // ---- kv_pages_copy: positions [from, to) of a sequence to a store-layout buffer and back ----
  {
    // Layer-and-K/V count NL * 2: the store layout holds every layer's K and V. Compare layer LAYER's K and V.
    const int from = 37, to = 4200, buf_start = 30, buf_len = 4200 - 30 + 5;
    const size_t bn = size_t(NL) * 2 * Hkv * buf_len * D;
    __nv_bfloat16* buf;
    cudaMalloc(&buf, bn * 2);
    cudaMemset(buf, 0, bn * 2);
    paged_layout();  // kv_pages_copy reads the pool's head stride
    k::kv_pages_copy(pool.dev, big.dtable, NL * 2, Hkv, D, buf, buf_start, buf_len, from, to, true, nullptr);
    const std::vector<__nv_bfloat16> b = download(buf, bn), fk = download(big.dk, big.k.size()),
                                     fv = download(big.dv, big.v.size());
    bool ok = true;
    for (int kv = 0; kv < 2; ++kv)
      for (int h = 0; h < Hkv; ++h)
        for (int p = from; p < to; ++p) {
          const __nv_bfloat16* src = &(kv ? fv : fk)[(size_t(h) * big.max_ctx + p) * D];
          const __nv_bfloat16* dst = &b[((size_t(LAYER * 2 + kv) * Hkv + h) * buf_len + (p - buf_start)) * D];
          ok &= same(src, dst, D * 2);
        }
    // Back into the pages of the last of the three sequences (a different table), then read back.
    k::kv_pages_copy(pool.dev, seqs[2].dtable, NL * 2, Hkv, D, buf, buf_start, buf_len, from, std::min(to, 8192), false,
                     nullptr);
    const std::vector<__nv_bfloat16> dev_pool = download(pool.dev, pool.host.size());
    const std::vector<__nv_bfloat16> rk = read_back(pool, dev_pool, seqs[2], 0);
    for (int h = 0; h < Hkv; ++h)
      for (int p = from; p < std::min(to, 8192); ++p)
        ok &= same(&rk[(size_t(h) * 8192 + p) * D], &fk[(size_t(h) * big.max_ctx + p) * D], D * 2);
    cudaFree(buf);
    report("kv_pages_copy to the store's layout and back into another table", ok);
  }
  // ---- kv_table_set (P4b): table entries as kernel parameters, several launches queued back to back ----
  {
    // A table of 4,096 entries; three updates queued without a synchronization in between (the first of 700 entries,
    // three launches), each built in the same host vectors, which change before the earlier launches have run: every
    // launch must still write the values it was given (no copy reads a host buffer later).
    const int n_entries = 4096;
    uint64_t* table = nullptr;
    cudaMalloc(&table, n_entries * sizeof(uint64_t));
    cudaMemset(table, 0, n_entries * sizeof(uint64_t));
    std::vector<uint64_t> model(n_entries, 0);
    std::vector<int> idx;
    std::vector<uint64_t> val;
    std::vector<int> order(n_entries);
    std::iota(order.begin(), order.end(), 0);
    for (int round = 0; round < 3; ++round) {
      std::shuffle(order.begin(), order.end(), rng);
      const int m = round == 0 ? 700 : round == 1 ? 256 : 1;
      idx.assign(order.begin(), order.begin() + m);
      val.resize(m);
      for (int i = 0; i < m; ++i) model[idx[i]] = val[i] = (uint64_t(rng()) << 20) * (round + 1) + i;
      k::kv_table_set(table, idx.data(), val.data(), m, nullptr);
    }
    k::kv_table_set(table, idx.data(), val.data(), 0, nullptr);  // nothing to write: no launch
    const std::vector<uint64_t> got = download(table, n_entries);
    report("kv_table_set: 957 entries over five launches queued back to back, read back", got == model);
    cudaFree(table);
  }
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("CUDA error: %s\n", cudaGetErrorString(e));
    ++failures;
  }
  std::printf(failures ? "PAGED TESTS FAILED (%d)\n" : "all paged tests passed\n", failures);
  return failures ? 1 : 0;
}
