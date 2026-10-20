// The v0 kernels. Activations are FP32 between kernels, weights stay in the checkpoint's own encoding
// (NVFP4 for the MLP and the LM head, FP8 E4M3 for the attention projections, BF16 for the rest) and are
// dequantized in registers. Rows (M) are the tokens of one step: 1 for decode, up to a prefill chunk.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ling::kernels {

// A sequence position as a kernel argument. Eager launches pass the host's value (an int converts
// implicitly). A launch captured into a CUDA graph is replayed at every step, at a different position each
// time, so it passes `dev`: the kernel then reads the position from device memory (*dev + off), where the
// engine writes it before each replay. `host` is the host's view at launch time; it is never used for
// launch geometry inside a graph (kernels whose grid would depend on it take a fixed grid instead).
struct DevPos {
  int host = 0;
  int off = 0;               // added to *dev
  const int* dev = nullptr;  // nullptr: the position is `host`
  DevPos(int p) : host(p) {}  // NOLINT: implicit on purpose, so eager call sites pass a plain int
  DevPos(int p, const int* d, int o) : host(p), off(o), dev(d) {}
  DevPos plus(int k) const { return DevPos(host + k, dev, off + k); }
};
#ifdef __CUDACC__
__device__ __forceinline__ int pos_value(const DevPos& p) { return p.dev ? *p.dev + p.off : p.host; }
#endif

// Several sequences' rows in one launch (M4 stage 4, the batched verify and drafter): sequence i owns rows
// [row0[i], row0[i + 1]) of the launch, at positions pos(i) + (row - row0[i]). Passed by value, so a launch captured
// into a CUDA graph keeps the row split; the positions come from pos_dev (written before each replay) when set.
constexpr int kMaxSeqs = 4;
struct SeqBatch {
  int n = 0;
  int row0[kMaxSeqs + 1] = {};
  int pos[kMaxSeqs] = {};          // the host's positions (eager launches; also the launch geometry)
  const int* pos_dev = nullptr;    // a captured graph: sequence i's position is pos_dev[i]
};
template <typename T>
struct SeqPtrs {
  T* p[kMaxSeqs] = {};
};
#ifdef __CUDACC__
__device__ __forceinline__ int seq_of(const SeqBatch& b, int row) {
  int s = 0;
  while (s + 1 < b.n && row >= b.row0[s + 1]) ++s;
  return s;
}
__device__ __forceinline__ int seq_pos(const SeqBatch& b, int s) { return b.pos_dev ? b.pos_dev[s] : b.pos[s]; }
#endif

// Programmatic dependent launch (launch.cuh) for the rows path's kernels (decode, verify, drafter, commit):
// off by default. Set before launching; it applies to every later launch, captured or not.
void set_pdl(bool on);
bool pdl_enabled();

// Largest M the weight-streaming GEMV kernels handle; larger M dequantizes to BF16 and uses gemm_bf16.
constexpr int kMaxGemvRows = 8;

// y[M][N] = scale2 * x[M][K] . W[N][K]^T, W NVFP4: packed [N][K/2] (low nibble first), E4M3 block
// scales [N][K/16]. K must be a multiple of 1024.
void gemv_nvfp4(const float* x, int M, const uint8_t* w, const uint8_t* wscale, float scale2, float* y,
                int N, int K, cudaStream_t s);
// y = wscale * x . W^T, W FP8 E4M3 [N][K]. K must be a multiple of 512.
void gemv_fp8(const float* x, int M, const uint8_t* w, float wscale, float* y, int N, int K,
              cudaStream_t s);
// y = x . W^T for narrow BF16 matrices and up to 32 rows, row-invariant (the rows path). K % 1024 == 0.
// With w2/y2: a second matrix of the same shape and input in the same launch.
void bf16_rows(const float* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s,
               const __nv_bfloat16* w2 = nullptr, float* y2 = nullptr);
// The same with a BF16 input (rows of `x` K apart): the drafter's fc over the target features.
void bf16_rows_bf16in(const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s);
// y = x . W^T, W BF16 [N][K].
void gemv_bf16(const float* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s);

// ---- Weight-streaming GEMM on tensor cores for 1 to 32 rows (stream_gemm.cu): decode, verify, drafter. ----
// Row-invariant: a row's result is bit-for-bit the same whatever M is, so a token verified in a block of
// 16 gets exactly the numbers it would get decoded alone.
constexpr int kMaxStreamRows = 32;

// Tree verification (BACKLOG 26): one verify's rows form a tree under the anchor (row 0) instead of a chain. Row m
// sits at depth depth[m] (its position is pos0 + depth[m]) and its key and value at cache slot pos0 + m, as a chain's
// do; every ancestor of a row has a lower index. Each row is computed exactly as a chain holding its root-to-row path
// would compute it, bit for bit:
//   - attention: a row's keys from pos0 on are its path's (variant var[m], a root-to-leaf path through m, whose row
//     at depth j is path[var][j]), loaded at the positions the chain would hold them in;
//   - the conv: its taps are its ancestors' inputs (from the variant's path);
//   - the recurrence: walked along gdn paths, each a root-to-node path advanced row by row from the state, with a
//     leaf child computed from its parent's state without advancing it; gdn_step[p][i] = row | kLeaf | kWrite.
// Written by the host before each tree verify (a graph's memcpy node copies it from pinned memory).
constexpr int kMaxTreeVariants = 32;
struct TreeDesc {
  int n = 0;
  int depth[kMaxStreamRows] = {};
  int var[kMaxStreamRows] = {};
  int nvar = 0;
  int path[kMaxTreeVariants][kMaxStreamRows] = {};
  int ngdn = 0;
  int gdn_len[kMaxTreeVariants] = {};
  int gdn_step[kMaxTreeVariants][2 * kMaxStreamRows] = {};
  static constexpr int kLeaf = 1 << 8, kWrite = 1 << 9;
};
// The verify's conv over a tree: post[m] from pre[m] and its ancestors' pre rows (or the window `state`, unwritten).
void gdn_conv_tree(const float* pre, float* post, const float* state, const __nv_bfloat16* w, int M, int C,
                   const TreeDesc* td, cudaStream_t s);
// The verify's recurrence over a tree, from state_in (unwritten).
void gdn_recurrent_tree(const float* mixed, const float* g, const float* beta, const float* state_in, float* out,
                        int H, int HV, const TreeDesc* td, int ngdn, cudaStream_t s);
// After a tree verify's accept: the accepted path's rows rows[j] (j < n, rows[j] >= j, rows[0] = 0) moved to row j in
// every buffer of `bufs` (row r of buffer i at base + (r + (pos_rel ? pos : 0)) * stride bytes, row_bytes long), in
// place, j ascending.
struct GatherBuf {
  char* base;
  size_t stride;
  int row_bytes;
  int pos_rel;
};
void tree_gather(const GatherBuf* bufs, int nbufs, const int* rows, int n, DevPos pos, cudaStream_t s);
// The weights stream_gemm reads are tiled (SPEC.md §8, stage 4): each (16-row tile, 128-value chunk) is
// one contiguous block in the order the warp's lanes read it, so every load covers 512 contiguous bytes
// (measured: the row-major pattern reads at 215 GB/s, contiguous blocks at 232). NVFP4 blocks carry
// their E4M3 scales after the values. N % 16 == 0 and K % 128 == 0.
size_t tiled_bytes_nvfp4(int N, int K);
size_t tiled_bytes_fp8(int N, int K);
void retile_nvfp4(const uint8_t* w, const uint8_t* wscale, uint8_t* out, int N, int K, cudaStream_t s);
void retile_fp8(const uint8_t* w, uint8_t* out, int N, int K, cudaStream_t s);
// B26, the NVFP4 scales as 6-bit codes: a tensor whose block scales take at most 64 distinct E4M3 values (every
// matrix of the target but one, and the drafter's) stores every (tile, chunk) block's 1,024 value bytes, then every
// block's 96 bytes of codes (a lane slot's two 6-bit codes are 12 bits, the first scale's low: lane L's slots L and
// L + 32 keep their low bytes at 2L and 2L + 1 and their high nibbles in byte 64 + L, so a lane reads one 16-bit
// word and one byte per tile, as many loads as the raw scales; the value blocks stay 128-byte aligned), 1,120 bytes
// per block instead of 1,152, with the tensor's 64-entry table (code -> E4M3 byte) beside it; the streaming GEMM
// looks each code up in shared memory, so the scales it multiplies by are bit for bit the checkpoint's (lossless).
// The prefill GEMM reads the raw scales from a separate copy (`scales`: 128 bytes per block, in the raw blob's
// order), which only the prefill path touches.
constexpr int kScaleCodes = 64;
// The code of every scale byte (0xff: unused) and the table, or false when the scales take more than 64 values.
bool scale_code_table(const uint8_t* scales, size_t n, uint8_t (&table)[kScaleCodes], uint8_t (&code_of)[256]);
size_t tiled_bytes_nvfp4_coded(int N, int K);
void retile_nvfp4_coded(const uint8_t* w, const uint8_t* wscale, const uint8_t* code_of_dev, uint8_t* out,
                        uint8_t* raw_scales, int N, int K, cudaStream_t s);
// Tiled weights -> BF16 row-major [N][K] (the prefill path's BF16 GEMM input).
void dequant_tiled_nvfp4(const uint8_t* tw, float scale2, __nv_bfloat16* out, int N, int K, cudaStream_t s,
                         const uint8_t* raw_scales = nullptr);  // raw_scales: the coded blob's (B26)
void dequant_tiled_fp8(const uint8_t* tw, float wscale, __nv_bfloat16* out, int N, int K, cudaStream_t s);

// ---- Prefill GEMMs for more than 32 rows (prefill_gemm.cu): block-scaled tensor cores over the same tiled
// weights, with the activations quantized as production quantizes them (SPEC.md §11). ----
// x [M][K] -> NVFP4: q packed [M][K/2] (low nibble first), E4M3 scales sf [M][K/16], global scale
// 1 / input_scale. K % 16 == 0.
void quant_act_nvfp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s);
// The same for silu(g) * u (the FFN's activation, never written in FP32).
void silu_mul_quant_nvfp4(const float* g, const float* u, int M, int K, float input_scale, uint8_t* q, uint8_t* sf,
                          cudaStream_t s);
// x [M][K] -> FP8 E4M3 [M][K], x / input_scale saturated. K % 4 == 0.
void quant_act_fp8(const float* x, int M, int K, float input_scale, uint8_t* q, cudaStream_t s);
// Producers of the prefill GEMMs' inputs, fused with the quantization (the FP32 tensor is never written, except
// `out` of rmsnorm_quant when given): Gemma RMSNorm (H % 1024 == 0), the DeltaNet gated norm (D = 128) and the
// attention's sigmoid gate (n % 4 == 0).
void rmsnorm_quant(const float* x, const __nv_bfloat16* w, float* out, int M, int H, float eps, bool nvfp4,
                   float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s);
void gated_rmsnorm_fp8(const float* x, const float* z, const __nv_bfloat16* w, int rows, int D, float eps,
                       float input_scale, uint8_t* q, cudaStream_t s);
void sigmoid_mul_fp8(const float* x, const float* gate, size_t n, float input_scale, uint8_t* q, cudaStream_t s);
// y[M][0..N) (rows ldy apart) = alpha * xq . W^T, or += with `accumulate`; W tiled (retile_*), alpha =
// input_scale * the weight's global scale. N % 128 == 0, K % 128 == 0.
// wsf (B26): a coded blob's raw scales (retile_nvfp4_coded); nullptr when the blob carries them.
void prefill_gemm_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, float alpha, float* y, int ldy,
                        int N, int K, bool accumulate, cudaStream_t s, const uint8_t* wsf = nullptr);
void prefill_gemm_fp8(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                      bool accumulate, cudaStream_t s);
// The FFN's gate and up projections in one GEMM, straight into the down projection's input:
// NVFP4(silu(alpha_gate * xq . G^T) * (alpha_up * xq . U^T)) with global scale 1 / out_input_scale, into q_out
// [M][N/2] and sf_out [M][N/16]; the FP32 products are never written. N % 128 == 0, K % 128 == 0.
void prefill_swiglu_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* gate, float alpha_gate,
                          const uint8_t* up, float alpha_up, int N, int K, float out_input_scale, uint8_t* q_out,
                          uint8_t* sf_out, cudaStream_t s, const uint8_t* gate_sf = nullptr,
                          const uint8_t* up_sf = nullptr);

// ya = x . Wa^T, yb = x . Wb^T for two BF16 [48][K] matrices and any M, row-invariant (the prefill path's
// DeltaNet a and b projections).
void narrow_bf16(const float* x, int M, const __nv_bfloat16* wa, const __nv_bfloat16* wb, float* ya, float* yb, int N,
                 int K, cudaStream_t s);

// ---- --prefill-bf16 (EngineOptions::prefill_bf16, off by default; MILESTONES §14's exactness policy): the
// prefill's DeltaNet intermediates in BF16 where production keeps BF16 (M2 §8 item 3, cited as §7.3 in the
// backlog's B34 row). Each is a new instantiation of the kernel above or below it over the element type
// (elem.cuh); the FP32 kernels are untouched. ----
// rmsnorm_quant with the FP32 copy replaced by a BF16 one (the a/b projections' input; the same values
// narrow_bf16 rounds its FP32 input to).
void rmsnorm_quant_bf16(const float* x, const __nv_bfloat16* w, __nv_bfloat16* out, int M, int H, float eps, bool nvfp4,
                        float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s);
// quant_act_fp8 over the BF16 copy (the same bytes as from the FP32 values it was rounded from).
void quant_act_fp8_bf16in(const __nv_bfloat16* x, int M, int K, float input_scale, uint8_t* q, cudaStream_t s);
// prefill_gemm_fp8 with its FP32 output rounded to BF16 (the in_qkv and in_z projections; never accumulated).
void prefill_gemm_fp8_bf16out(const uint8_t* xq, int M, const uint8_t* w, float alpha, __nv_bfloat16* y, int ldy, int N,
                              int K, cudaStream_t s);
// narrow_bf16 over the BF16 copy of the norm (the same MMA fragments).
void narrow_bf16_bf16in(const __nv_bfloat16* x, int M, const __nv_bfloat16* wa, const __nv_bfloat16* wb, float* ya,
                        float* yb, int N, int K, cudaStream_t s);
// gated_rmsnorm_fp8 over the recurrence's BF16 output and the BF16 z projection.
void gated_rmsnorm_fp8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* z, const __nv_bfloat16* w, int rows, int D,
                            float eps, float input_scale, uint8_t* q, cudaStream_t s);

// ---- The batched verify's state kernels (M4 stage 4): one launch over several sequences, each row computed
// exactly as the per-sequence launch computes it (ling-stream-tests compares them bit for bit). ----
// attn_prepare over every row of b, each into its sequence's caches at its sequence's position.
void attn_prepare_seqs(const float* q_gate, const float* k, const float* v, const __nv_bfloat16* q_norm,
                       const __nv_bfloat16* k_norm, const SeqBatch& b, int Hq, int Hkv, int D, int rot, float theta,
                       float eps, float* q, float* gate, SeqPtrs<__nv_bfloat16> kc, SeqPtrs<__nv_bfloat16> vc,
                       cudaStream_t s);
// attention_rows over every row of b (q and out [rows][Hq][D]), each sequence over its own caches.
void attention_rows_seqs(const float* q, SeqPtrs<const __nv_bfloat16> kc, SeqPtrs<const __nv_bfloat16> vc,
                         const SeqBatch& b, int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s,
                         int fixed_ctx = 0);
// The verify's conv: each sequence's rows of mixed from its own window (read, not written).
void gdn_conv_seqs(float* mixed, SeqPtrs<const float> state, const __nv_bfloat16* w, const SeqBatch& b, int C,
                   cudaStream_t s);
// The verify's recurrence: each sequence's rows from its own state (read, not written).
void gdn_recurrent_seqs(const float* mixed, const float* g, const float* beta, SeqPtrs<const float> state, float* out,
                        const SeqBatch& b, int H, int HV, cudaStream_t s);

// RMSNorm (as rmsnorm) that also writes the FP16 copy and scales to_half_rows would make from its output.
void rmsnorm_half(const float* x, const __nv_bfloat16* w, float* out, __half* xh, float* xinv, int rows, int H, float eps,
                  bool gemma, cudaStream_t s);
// x [M][K] FP32 -> FP16 with a power-of-two scale per row (xinv[m] undoes it).
void to_half_rows(const float* x, int M, int K, __half* out, float* xinv, cudaStream_t s);
// y[M][N] = scale2 * x . W^T, W NVFP4 in the tiled layout (wscale is unused: the scales are in the blob).
// wscale: the scale-code table of a coded blob (B26), nullptr when the blob carries its raw scales.
void stream_gemm_nvfp4(const __half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
                       float scale2, float* y, int N, int K, cudaStream_t s);
// y[M][N] = wscale * x . W^T, W FP8 E4M3 in the tiled layout.
void stream_gemm_fp8(const __half* x, const float* xinv, int M, const uint8_t* w, float wscale, float* y, int N,
                     int K, cudaStream_t s);

// Several matrices that share the input x, in one launch (fewer launch ramps and tails): each target
// gets y = scale * x . W^T. Row-invariant like the single-matrix calls, and bit-identical to them.
struct StreamTarget {
  const uint8_t* w;
  const uint8_t* wscale;  // NVFP4: the scale-code table of a coded blob (B26); nullptr for a raw blob and for FP8
  float scale;            // NVFP4 global scale, or the FP8 per-tensor scale
  float* y;
  int N;
};
void stream_gemm_multi(bool nvfp4, const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K,
                       cudaStream_t s);
// The megakernel's stage 1 in the step (LING_MK_FFN): the rows path's FFN (gate|up from xh, silu(g) * u into g, its FP16
// copy into xh2 with xinv2, down into down.y) as one persistent kernel, bit for bit the four kernels. `counters` is
// kMkFfnWords zeroed words, owned by this call site (the kernel leaves them zeroed). False: a shape or a stream GEMM
// override (LING_SG_*, LING_KSPLIT) the kernel does not reproduce; nothing was launched.
constexpr int kMkFfnWords = 5 + kMaxStreamRows;
// The megakernel's stage 4 (LING_MK_LAYERS, BACKLOG 28): a persistent kernel's last GEMM adds its output into the
// residual stream h itself (add_inplace's rounding), and, with norm_w, the next norm (rmsnorm_half's row, exactly)
// follows in the same launch into out (FP32), xh and xinv. Passed as a pointer: nullptr keeps the separate kernels.
struct MkTail {
  float* h;
  const __nv_bfloat16* norm_w = nullptr;  // nullptr: the residual add only
  float* out = nullptr;
  __half* xh = nullptr;
  float* xinv = nullptr;
  float eps = 0.f;
};
// The rows the fused tails are built for (9 to 16: the verify's draft blocks of 12 and 16); other row counts keep the
// separate kernels (the persistent kernels then refuse a tail).
inline bool mk_tail_rows(int M) { return M > 8 && M <= 16; }
// The megakernel's stage 2 in the step (LING_MK_GDN): one verify sequence's DeltaNet mixer, from the input norm's output
// (xn in FP32, xh/xinv its FP16 copy) to the out-projection's output (out.y), as one persistent kernel: in_qkv's output
// into in_qkv.y (kept for the commit), in_z's into in_z.y, the a/b projections and the gating into a, b, g, beta, the
// conv's output into mixed (from conv_state, left unwritten), the recurrence's into core (from state, left unwritten),
// the gated norm's into normed and its FP16 copy into xh2/xinv2. Bit for bit the graph path's kernels. `counters` is
// kMkGdnWords zeroed words owned by the call site. False: a shape or override it does not reproduce; nothing launched.
constexpr int kMkGdnWords = 12 + kMaxStreamRows;
struct MkGdnParams {
  const __half* xh;
  const float* xinv;
  const float* xn;
  int M, K;  // rows; the hidden size (the projections' K and the out-projection's N)
  StreamTarget in_qkv, in_z, out;  // FP8
  const __nv_bfloat16 *wa, *wb;    // [HV][K] each
  float *a, *b, *g, *beta;
  const __nv_bfloat16 *A_log, *dt_bias;
  float* mixed;
  const float* conv_state;
  const __nv_bfloat16* conv_w;
  const float* state;
  float* core;
  const __nv_bfloat16* norm_w;
  float eps;
  float* normed;
  __half* xh2;
  float* xinv2;
  int C, H, HV, Dk, Dv, Vs;
  unsigned* counters;
  unsigned long long* times = nullptr;  // LING_MK_TIMES: [blocks][kMkTimesPerBlock] stamps of the last launch
  const MkTail* tail = nullptr;  // stage 4: the out-projection into tail->h, then tail's norm
  const TreeDesc* td = nullptr;  // a tree verify (BACKLOG 26): the tree conv and the per-path recurrence
};
bool gdn_persistent(const MkGdnParams& p, cudaStream_t s);
// The megakernel's stage 4 (LING_MK_SPAN): a verify's DeltaNet layer, its mixer (as gdn_persistent with gp.tail) and
// its FFN (as ffn_persistent with tail), in one persistent launch. gp.tail must carry the post-attention norm into its
// own FP16 buffer (not gp.xh or gp.xh2), and xh2/xinv2 (the down projection's input) must not be the mixer's. False:
// not built for these shapes or options (LING_MK_SPAN_PFG 2 or 3, LING_MK_SPAN_PFF 3 or 4); nothing launched.
bool gdn_ffn_persistent(const MkGdnParams& gp, const StreamTarget& gate, const StreamTarget& up, const StreamTarget& down,
                        int I, float* g, float* u, __half* xh2, float* xinv2, unsigned* counters,
                        unsigned long long* times, const MkTail* tail, cudaStream_t s);
// tail (stage 4): the down projection into tail->h, then tail's norm (down.y unused).
bool ffn_persistent(const __half* xh, const float* xinv, int M, const StreamTarget& gate, const StreamTarget& up,
                    const StreamTarget& down, int H, int I, float* g, float* u, __half* xh2, float* xinv2,
                    unsigned* counters, cudaStream_t s, unsigned long long* times = nullptr,
                    const MkTail* tail = nullptr);
constexpr int kMkTimesPerBlock = 10;  // LING_MK_TIMES: globaltimer stamps per block of a persistent kernel
// The megakernel probe (BACKLOG 28): about `bytes` of a tiled weight blob (the first chunks of every tile, what a stream
// GEMM's blocks read first) prefetched into L2 by one PDL launch, so the latency-bound kernels before that GEMM run
// beside its first bytes. Writes nothing; the GEMM computes the same bits.
void l2_prefetch_weights(const uint8_t* w, bool fp4, bool coded, int N, int K, size_t bytes, cudaStream_t s);

// Top-K (K <= 64) of each row of x [rows][V]: values descending, ties to the lower index (so entry 0 is
// what std::max_element returns). `scratch` needs topk_scratch_floats(rows, K) floats.
size_t topk_scratch_floats(int rows, int K);
void topk_rows(const float* x, int rows, int V, int K, float* scratch, float* vals, int* ids, cudaStream_t s);

// Dequantize a whole matrix to BF16 (for the BF16 GEMM path).
void dequant_nvfp4(const uint8_t* w, const uint8_t* wscale, float scale2, __nv_bfloat16* out, int N, int K,
                   cudaStream_t s);
void dequant_fp8(const uint8_t* w, float wscale, __nv_bfloat16* out, int N, int K, cudaStream_t s);
void to_bf16(const float* x, __nv_bfloat16* out, int n, cudaStream_t s);
// y[M][N] = x[M][K] . W[N][K]^T with BF16 inputs and FP32 output, on tensor cores (gemm_bf16.cu; it
// replaced cuBLAS, BACKLOG §19.23). K % 64 == 0, N even, x and W 16-byte aligned; any M. A row's result
// does not depend on M. Graph-capturable.
void gemm_bf16(const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s);

void embed(const __nv_bfloat16* table, const int* ids, int M, int H, float* out, cudaStream_t s);
// out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * (gemma ? 1 + w : w), rows of length H.
void rmsnorm(const float* x, const __nv_bfloat16* w, float* out, int rows, int H, float eps, bool gemma,
             cudaStream_t s);
void add_inplace(float* h, const float* d, int n, cudaStream_t s);
// out = silu(g) * u
void silu_mul(const float* g, const float* u, float* out, int n, cudaStream_t s);
// x *= sigmoid(gate)
void sigmoid_mul(float* x, const float* gate, int n, cudaStream_t s);

// Gated DeltaNet. `mixed` is [M][C] (q | k | v, C = 2*Kd + Vd): causal depthwise conv (kernel 4) with
// state [C][3], then SiLU, in place.
void gdn_conv(float* mixed, float* conv_state, const __nv_bfloat16* w, int M, int C, cudaStream_t s);
// g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b); a, b, g, beta are [M][HV].
void gdn_gating(const float* a, const float* b, const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
                float* g, float* beta, int M, int HV, cudaStream_t s);
// The delta rule over M tokens with q/k L2 normalization: state [HV][Dk][Dv] (k-major), out [M][HV*Dv].
// Reads the state from state_in and writes the advanced state to state_out (the same pointer to update in
// place; nullptr to leave it unwritten, as a verify does). Each token's arithmetic is the same whatever M
// is, so replaying accepted rows reproduces sequential decoding bit for bit.
void gdn_recurrent(const float* mixed, const float* g, const float* beta, const float* state_in, float* state_out,
                   float* out, int M, int H, int HV, cudaStream_t s);
// The prefill path's conv (prefill_gdn.cu), parallel over tokens: out of place (in: the raw projections,
// out: after the conv and SiLU), and the window updated. Each token is computed the same way whatever M is.
// window_at (optional): also the window after the first `at` tokens (a state kept for a later prompt).
void gdn_conv_prefill(const float* in, float* out, float* conv_state, const __nv_bfloat16* w, int M, int C,
                      cudaStream_t s, float* window_at = nullptr, int at = -1);
// The prefill path's recurrence: the same math as gdn_recurrent in its chunked form (32-token chunks at absolute
// positions: pos0 is the first token's position) on tensor cores, the state updated in place. Its result for a
// prompt does not depend on how the prompt was split, as long as every split falls on a multiple of 32.
// state_at (optional): also the state after the first `at` tokens, which must end a chunk (pos0 + at % 32 == 0).
void gdn_recurrent_prefill(const float* mixed, const float* g, const float* beta, float* state, float* out, int M, int H,
                           int HV, int pos0, cudaStream_t s, float* state_at = nullptr, int at = -1);
// The two above over BF16 activations (--prefill-bf16; see the block after narrow_bf16): the conv reads the
// in_qkv projection in BF16 and writes its output in BF16 (the state window then holds BF16 values, as
// production's conv state does), the recurrence reads that and writes its output in BF16. The conv and
// recurrent state, g and beta stay FP32; the same chunk invariance holds.
void gdn_conv_prefill_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, float* conv_state, const __nv_bfloat16* w, int M,
                           int C, cudaStream_t s, float* window_at = nullptr, int at = -1);
void gdn_recurrent_prefill_bf16(const __nv_bfloat16* mixed, const float* g, const float* beta, float* state,
                                __nv_bfloat16* out, int M, int H, int HV, int pos0, cudaStream_t s,
                                float* state_at = nullptr, int at = -1);
// out = rmsnorm(x) * w * silu(z), rows of length D.
void gated_rmsnorm(const float* x, const float* z, const __nv_bfloat16* w, float* out, int rows, int D,
                   float eps, cudaStream_t s);

// The KV caches' layout (elements): head h, position p at h * head_stride + p * pos_stride. The default
// is interleaved [pos][head][dim]; the engine sets head-major [head][pos][dim], so one head's keys are
// one contiguous stream for the attention kernels.
void set_kv_layout(size_t head_stride, size_t pos_stride);

// Full attention: q_gate [M][Hq*2*D] (per head: q then gate), k/v [M][Hkv*D]. Applies the Gemma-style
// q/k norms and NeoX RoPE on the first `rot` dims, writes q [M][Hq][D] and gate [M][Hq][D], and appends
// k/v to the BF16 caches [pos][Hkv][D] at positions pos0 .. pos0+M-1.
void attn_prepare(const float* q_gate, const float* k, const float* v, const __nv_bfloat16* q_norm,
                  const __nv_bfloat16* k_norm, DevPos pos0, int M, int Hq, int Hkv, int D, int rot, float theta,
                  float eps, float* q, float* gate, __nv_bfloat16* kcache, __nv_bfloat16* vcache,
                  cudaStream_t s, const TreeDesc* td = nullptr);
// Causal attention of M queries (positions pos0..) over the cache, on tensor cores over fixed 4096-key ranges at
// fixed positions (combined afterwards), so each row's result is independent of M: decode, verify and prefill
// (in slices of queries). `scratch` needs attention_rows_scratch_floats(M, Hq, D, ctx) floats.
// fixed_ctx > 0 (a captured graph, pos0 read on the device): launch the ranges covering fixed_ctx keys
// whatever the position; ranges past the last visible key write an empty partial that the combine skips,
// so the result is bit for bit the eager launch's.
size_t attention_rows_scratch_floats(int M, int Hq, int D, int ctx);
void attention_rows(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, DevPos pos0, int M,
                    int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s, int fixed_ctx = 0,
                    const TreeDesc* td = nullptr);
// Sets the kernels' shared-memory attributes once, outside any graph capture.
void prepare_kernels();
// M3: attention_rows loads its key/value tiles with bulk copies (TMA, one 512-byte copy per key row) instead
// of 16-byte cp.async. Same tiles, same arithmetic: the output is bit for bit the same. Off by default.
void set_attention_bulk(bool on);
// M3: attention_rows prefetches its KV tiles into L2 this many tiles past its shared-memory ring (head-major KV
// only; 0 off). Moves data only: the output is bit for bit the same.
void set_attention_prefetch(int tiles);
// What attention_rows uses (for the megakernel's attention layer, which reproduces it): the KV strides, the L2
// prefetch depth, whether the bulk-copy tiles are on, and the key range size (attention_chunk()).
size_t kv_head_stride(int D);
size_t kv_pos_stride(int Hkv, int D);
int attention_prefetch_tiles();
bool attention_bulk_enabled();
int attention_chunk();
// The megakernel's stage 3 in the step (LING_MK_ATTN, BACKLOG 28): one verify sequence's full-attention layer, from the
// input norm's FP16 copy (xh/xinv) to the out-projection's output (o.y), as one persistent kernel: q|k|v into q.y,
// k.y, v.y, the prepare step (q/k norms, RoPE, the rows' K and V into the caches at pos0 ..) into q_out, gate and the
// caches, the attention over the session's real key ranges into scratch, the combine with the sigmoid gate into attn,
// its FP16 copy into xh2/xinv2, then the out-projection. Bit for bit the graph path's kernels (attn_prepare,
// attention_rows, sigmoid_mul, to_half_rows, the stream GEMMs); work is taken from one ordered ticket queue, so it
// completes with any number of resident blocks. `counters` is kMkAttnWords zeroed words owned by the call site (left
// zeroed). False: a shape or override it does not reproduce (bulk tiles, LING_SG_*, LING_KSPLIT); nothing launched.
constexpr int kMkAttnWords = 32;
constexpr int kMkAttnEvents = 16;  // LING_MK_TIMES: per launch, the globaltimer of each counter's completion
struct MkAttnParams {
  const __half* xh;
  const float* xinv;
  int M, K;                     // rows; the hidden size (q|k|v's K, o's N)
  StreamTarget q, k, v, o;      // FP8; q is q|gate per head [Hq][2 D]
  int Hq, Hkv, D, rot;
  float theta, eps;
  const __nv_bfloat16 *q_norm, *k_norm;
  float *q_out, *gate;          // [M][Hq][D]
  __nv_bfloat16 *kcache, *vcache;
  DevPos pos0 = DevPos(0);
  float* scratch;               // attention_rows_scratch_floats(M, Hq, D, the context) floats
  float* attn;                  // [M][Hq D]
  __half* xh2;                  // the out-projection's input and row scales
  float* xinv2;
  unsigned* counters;
  unsigned long long* times = nullptr;  // LING_MK_TIMES: kMkAttnEvents stamps of the last launch
  const TreeDesc* td = nullptr;  // a tree verify (BACKLOG 26): its rows' depths and the attention's per-variant tail
  const MkTail* tail = nullptr;  // stage 4: the out-projection into tail->h (o.y unused), then tail's norm
};
bool attn_persistent(const MkAttnParams& p, cudaStream_t s);
void attn_persistent_prepare();  // its shared-memory attribute, once, outside any graph capture
// ---- The DFlash2 drafter (draft_kernels.cu). ----
// dst[r][0..H) = bf16(src[r][0..H)), rows dst_stride apart (the target features the drafter conditions on).
void copy_rows_bf16(const float* src, int rows, int H, __nv_bfloat16* dst, int dst_stride, cudaStream_t s);
// to_half_rows from BF16 rows [M][K] (BACKLOG 33: the captured features into the fc's NVFP4 copy).
void bf16_to_half_rows(const __nv_bfloat16* x, int M, int K, __half* out, float* xinv, cudaStream_t s);
// Per-head RMSNorm (plain weight) and NeoX RoPE over the full 128-dim head at positions pos0 + row, in place.
void draft_qk_rope(float* x, int rows, int heads, int row_stride, const __nv_bfloat16* norm, DevPos pos0, float theta,
                   float eps, cudaStream_t s);
// k/v rows [rows][kv_size] into the BF16 rings [ring][kv_size] at positions pos0 ... (position p in row p % ring;
// M4 §1). rows <= ring.
void draft_store_kv(const float* k, const float* v, int rows, int kv_size, DevPos pos0, __nv_bfloat16* kc,
                    __nv_bfloat16* vc, int ring, cudaStream_t s);
// The block's B queries (positions L .. L + B - 1) over the cached context [L + j - window_left, L) and the
// whole block (bidirectional within it). q [B][Hq][128], kb/vb [B][Hkv][128]. The context is read from the
// rings kc/vc (position p in row p % ring, ring > window_left).
void draft_attention(const float* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, int ring, const float* kb,
                     const float* vb, DevPos L, int B, int window_left, int Hq, int Hkv, float* out, cudaStream_t s);
// The drafter over several sequences' blocks at once (M4 stage 4): draft_qk_rope with each row at its sequence's
// position, and draft_attention with each sequence's queries over its own ring and its own block (b's rows).
void draft_qk_rope_seqs(float* x, int heads, int row_stride, const __nv_bfloat16* norm, const SeqBatch& b, float theta,
                        float eps, cudaStream_t s);
void draft_attention_seqs(const float* q, SeqPtrs<const __nv_bfloat16> kc, SeqPtrs<const __nv_bfloat16> vc, int ring,
                          const float* kb, const float* vb, const SeqBatch& b, int window_left, int Hq, int Hkv,
                          float* out, cudaStream_t s);
// DFlash2's two-tap grouped dynamic convolution over a block (side 0 wraps a sublayer's input, 1 its output).
void grouped_conv(const float* h, const float* coef, const __nv_bfloat16* base, int side, float* out, int rows,
                  int C, int groups, int block, cudaStream_t s);
// The candidate selector's 16 x 16 transition scores for E positions (hp: [E][256], cand/unary: [E][16]).
// anchor_dev (a captured graph): read the anchor from device memory instead of `anchor`.
void selector_lattice(const float* hp, const int* cand, const float* unary, const __nv_bfloat16* P,
                      const __nv_bfloat16* S, int anchor, int E, float* scores, cudaStream_t s,
                      const int* anchor_dev = nullptr);

}  // namespace ling::kernels
