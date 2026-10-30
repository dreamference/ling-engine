// paged_attn_bench: attention_rows_seqs timed over the flat head-major KV and over the paged pool, with the engine's
// geometry (Qwen3.8-27B: 24 query heads, 4 KV heads, head_dim 256, 64-position pages holding every full-attention
// layer). Four sequences of 12 rows (a batched verify at --draft-block 12), each at the given context. Each call reads
// the next of `cycle` layers, as a pass does, so no layer's KV is still in L2. Timing only: every buffer holds the
// same constant, the identity of the layouts is ling-paged-tests' job.
//
// Layouts (-l):
//   flat    one allocation per sequence and layer, [Hkv][max_ctx][D] (the engine without --kv-pages)
//   page    the P4 pool: page = [layer][K, V][head][64][D], the table holds page id * page elements
//   head    a pool per (layer, K/V, head): base(l, kv, h) + page id * 64 * D; kv_off's formula unchanged, hs is the
//           sub-pool's size and the table holds page id * 64 * D
// Page orders (-o): seq (each sequence's pages consecutive), rr (round robin across sequences, as passes that grow
// in step), shuf (shuffled).
//
// Usage: paged_attn_bench CTX [LAYERS_IN_PAGE=16] [CYCLE=16] [REPS=64] [layouts=flat,page,head] [orders=seq,rr,shuf]
// One line per case: layout, order, prefetch, bulk, ms per layer call (median of REPS), effective GB/s of KV read.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/kernels.cuh"

namespace k = ling::kernels;

namespace {

constexpr int Hq = 24, Hkv = 4, D = 256, PT = k::kKvPageTokens, NSEQ = 4, ROWS = 12;

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

template <typename T>
T* dalloc(size_t n) {
  T* p = nullptr;
  ck(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMalloc");
  ck(cudaMemset(p, 0x3f, std::max<size_t>(n, 1) * sizeof(T)), "cudaMemset");
  return p;
}

std::vector<int> page_order(const std::string& order, int np) {
  // ids[i * np + j]: sequence i's page j (ids from 1; page 0 is the masked keys' page)
  std::vector<int> ids(size_t(NSEQ) * np);
  for (int i = 0; i < NSEQ; ++i)
    for (int j = 0; j < np; ++j) ids[size_t(i) * np + j] = order == "rr" ? 1 + j * NSEQ + i : 1 + i * np + j;
  if (order == "shuf") {
    std::mt19937 rng(2027);
    std::shuffle(ids.begin(), ids.end(), rng);
  }
  return ids;
}

std::vector<std::string> split(const char* s) {
  std::vector<std::string> out;
  std::string cur;
  for (const char* p = s; *p; ++p) {
    if (*p == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += *p;
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s CTX [LAYERS_IN_PAGE=16] [CYCLE=16] [REPS=64] [layouts] [orders]\n", argv[0]);
    return 2;
  }
  const int ctx = std::atoi(argv[1]);
  const int NL = argc > 2 ? std::atoi(argv[2]) : 16;
  const int cycle = std::min(NL, argc > 3 ? std::atoi(argv[3]) : 16);
  const int reps = argc > 4 ? std::atoi(argv[4]) : 64;
  const auto layouts = split(argc > 5 ? argv[5] : "flat,page,head");
  const auto orders = split(argc > 6 ? argv[6] : "seq,rr,shuf");
  k::prepare_kernels();
  const int np = (ctx + PT - 1) / PT;
  const size_t P = size_t(NSEQ) * np + 1;  // pool pages, page 0 included
  const size_t kvsize = size_t(Hkv) * D;

  float* q = dalloc<float>(size_t(NSEQ) * ROWS * Hq * D);
  ck(cudaMemset(q, 0, size_t(NSEQ) * ROWS * Hq * D * sizeof(float)), "q");
  float* scratch = dalloc<float>(k::attention_rows_scratch_floats(NSEQ * ROWS, Hq, D, ctx));
  float* out = dalloc<float>(size_t(NSEQ) * ROWS * Hq * D);
  k::SeqBatch sb;
  sb.n = NSEQ;
  for (int i = 0; i < NSEQ; ++i) {
    sb.row0[i + 1] = (i + 1) * ROWS;
    sb.pos[i] = ctx - ROWS;
  }
  const double kv_bytes = double(NSEQ) * ctx * kvsize * 2 * 2;  // K and V, BF16, every sequence
  cudaEvent_t e0, e1;
  ck(cudaEventCreate(&e0), "event");
  ck(cudaEventCreate(&e1), "event");

  for (const std::string& layout : layouts) {
    const bool flat = layout == "flat";
    // The buffers: flat, one [Hkv][ctx][D] K and V per sequence and cycled layer; paged, one pool.
    std::vector<__nv_bfloat16*> fk, fv;
    __nv_bfloat16* pool = nullptr;
    size_t page_elems = 0, sub = 0;
    if (flat) {
      for (int l = 0; l < cycle; ++l)
        for (int i = 0; i < NSEQ; ++i) {
          fk.push_back(dalloc<__nv_bfloat16>(size_t(ctx) * kvsize));
          fv.push_back(dalloc<__nv_bfloat16>(size_t(ctx) * kvsize));
        }
    } else {
      page_elems = size_t(NL) * 2 * kvsize * PT;
      sub = P * PT * D;  // "head": one (layer, K/V, head) sub-pool
      pool = dalloc<__nv_bfloat16>(page_elems * P);
    }
    for (const std::string& order : flat ? std::vector<std::string>{"-"} : orders) {
      std::vector<uint64_t*> tables;
      if (!flat) {
        const std::vector<int> ids = page_order(order, np);
        for (int i = 0; i < NSEQ; ++i) {
          std::vector<uint64_t> t(np);
          for (int j = 0; j < np; ++j)
            t[j] = uint64_t(ids[size_t(i) * np + j]) * (layout == "head" ? uint64_t(PT) * D : page_elems);
          uint64_t* dt = nullptr;
          ck(cudaMalloc(&dt, np * sizeof(uint64_t)), "table");
          ck(cudaMemcpy(dt, t.data(), np * sizeof(uint64_t), cudaMemcpyHostToDevice), "table");
          tables.push_back(dt);
        }
      }
      if (flat) k::set_kv_layout(size_t(ctx) * D, D);
      else if (layout == "head") k::set_kv_layout(sub, D);
      else k::set_kv_layout(size_t(PT) * D, D);
      for (int bulk = 0; bulk < 2; ++bulk)
        for (int pf : {0, 2}) {
          k::set_attention_bulk(bulk != 0);
          k::set_attention_prefetch(pf);
          std::vector<float> ms;
          for (int r = -cycle; r < reps; ++r) {  // the first cycle warms up
            const int l = (r + cycle) % cycle;
            k::SeqPtrs<const __nv_bfloat16> kc, vc;
            k::SeqPtrs<const uint64_t> pg;
            for (int i = 0; i < NSEQ; ++i) {
              if (flat) {
                kc.p[i] = fk[size_t(l) * NSEQ + i];
                vc.p[i] = fv[size_t(l) * NSEQ + i];
              } else if (layout == "head") {
                kc.p[i] = pool + (size_t(l) * 2 + 0) * Hkv * sub;
                vc.p[i] = pool + (size_t(l) * 2 + 1) * Hkv * sub;
                pg.p[i] = tables[i];
              } else {
                kc.p[i] = pool + size_t(l) * 2 * kvsize * PT;
                vc.p[i] = pool + (size_t(l) * 2 + 1) * kvsize * PT;
                pg.p[i] = tables[i];
              }
            }
            ck(cudaEventRecord(e0), "record");
            k::attention_rows_seqs(q, kc, vc, sb, Hq, Hkv, D, scratch, out, nullptr, 0, pg);
            ck(cudaEventRecord(e1), "record");
            ck(cudaEventSynchronize(e1), "sync");
            float t = 0.f;
            ck(cudaEventElapsedTime(&t, e0, e1), "elapsed");
            if (r >= 0) ms.push_back(t);
          }
          ck(cudaGetLastError(), "attention");
          std::sort(ms.begin(), ms.end());
          const double med = ms[ms.size() / 2];
          std::printf("ctx %d layout %-4s order %-4s prefetch %d bulk %d  %.3f ms/layer (min %.3f)  %.0f GB/s\n", ctx,
                      layout.c_str(), order.c_str(), pf, bulk, med, ms.front(), kv_bytes / (med * 1e-3) / 1e9);
          std::fflush(stdout);
        }
      for (uint64_t* t : tables) cudaFree(t);
    }
    for (auto* p : fk) cudaFree(p);
    for (auto* p : fv) cudaFree(p);
    if (pool) cudaFree(pool);
  }
  return 0;
}
