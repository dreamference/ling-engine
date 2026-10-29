// The LoRA kernels (specs/DREAMFERENCE_LING_ENGINE_LORA.md §3): a site's shrink, t = x A^T over its stacked A blocks,
// and its expand, y += scale * B t, each B segment into its rows of its output. The base projection has run over the
// whole batch already; these touch only the rows whose sequence is on an adapter, each with its own slot's weights.
//
// Row-invariant like the rest of the engine: a row's t and its delta do not depend on how many rows the launch has,
// where the row sits or which other adapters share the batch (a fixed per-thread order over K and a fixed reduction
// tree, no atomics), so a verify's rows equal the decode's and speculation stays exact.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "core/kernels.cuh"

namespace ling::kernels {

constexpr int kLoraRankRows = 64;  // t's row stride: kLoraMaxBlocks (4) x kLoraMaxRank (16)
constexpr int kLoraSegs = 4;       // kLoraMaxSegs
constexpr int kLoraOuts = 3;       // kLoraMaxOuts

// One B segment: rows [n0, n1) of output `out`, from A block `block`; its B, [n1 - n0][rank], at b + b_off.
struct LoraSegDev {
  int out = 0, n0 = 0, n1 = 0, block = 0;
  long long b_off = 0;
};

// One site of one layer in one slot of the GPU pool. a == nullptr: the slot's adapter does not touch the site.
struct LoraSiteDev {
  const __nv_bfloat16* a = nullptr;  // [blocks * rank][K]
  const __nv_bfloat16* b = nullptr;
  int K = 0, rank = 0, blocks = 0, nsegs = 0;
  float scale = 0.f;
  LoraSegDev segs[kLoraSegs];
};

// Which slot each row is on: the batch's sequences (row0 as SeqBatch's), each one slot or -1 (the base). A captured
// graph reads the slots from slot_dev, written before every replay beside the positions; eager launches take slot.
struct LoraRows {
  int n = 0;
  int row0[kMaxSeqs + 1] = {};
  int slot[kMaxSeqs] = {-1, -1, -1, -1};
  const int* slot_dev = nullptr;
};

// The outputs a site's segments add into, FP32 [M][ld].
struct LoraOuts {
  float* y[kLoraOuts] = {};
  int ld[kLoraOuts] = {};
};

// t[r][j] = sum_k x[r][k] A[j][k] for every row r on a slot, j < blocks * rank of that slot's site. `site` is the
// site in slot 0; slot s's is site[s * slot_stride]. x is FP32 [M][ldx]; K a multiple of 4.
void lora_shrink(const float* x, int ldx, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride,
                 int slots, float* t, cudaStream_t stream);

// y_out[r][n] += scale * sum_j B[n - n0][j] t[r][block * rank + j] for every row r on a slot and every segment of its
// site; max_n bounds every segment's n1 - n0 (the launch geometry).
void lora_expand(const float* t, int M, const LoraRows& rows, const LoraSiteDev* site, int slot_stride, int slots,
                 const LoraOuts& outs, int max_n, cudaStream_t stream);

}  // namespace ling::kernels
