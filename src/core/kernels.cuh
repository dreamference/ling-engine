// The v0 kernels. Activations are FP32 between kernels, weights stay in the checkpoint's own encoding
// (NVFP4 for the MLP and the LM head, FP8 E4M3 for the attention projections, BF16 for the rest) and are
// dequantized in registers. Rows (M) are the tokens of one step: 1 for decode, up to a prefill chunk.
#pragma once

#include <cstdint>

#include <cublas_v2.h>
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

// Largest M the weight-streaming GEMV kernels handle; larger M dequantizes to BF16 and uses cuBLAS.
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
// The weights stream_gemm reads are tiled (SPEC.md §8, stage 4): each (16-row tile, 128-value chunk) is
// one contiguous block in the order the warp's lanes read it, so every load covers 512 contiguous bytes
// (measured: the row-major pattern reads at 215 GB/s, contiguous blocks at 232). NVFP4 blocks carry
// their E4M3 scales after the values. N % 16 == 0 and K % 128 == 0.
size_t tiled_bytes_nvfp4(int N, int K);
size_t tiled_bytes_fp8(int N, int K);
void retile_nvfp4(const uint8_t* w, const uint8_t* wscale, uint8_t* out, int N, int K, cudaStream_t s);
void retile_fp8(const uint8_t* w, uint8_t* out, int N, int K, cudaStream_t s);
// Tiled weights -> BF16 row-major [N][K] (the prefill path's cuBLAS input).
void dequant_tiled_nvfp4(const uint8_t* tw, float scale2, __nv_bfloat16* out, int N, int K, cudaStream_t s);
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
void prefill_gemm_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, float alpha, float* y, int ldy,
                        int N, int K, bool accumulate, cudaStream_t s);
void prefill_gemm_fp8(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                      bool accumulate, cudaStream_t s);
// The FFN's gate and up projections in one GEMM, straight into the down projection's input:
// NVFP4(silu(alpha_gate * xq . G^T) * (alpha_up * xq . U^T)) with global scale 1 / out_input_scale, into q_out
// [M][N/2] and sf_out [M][N/16]; the FP32 products are never written. N % 128 == 0, K % 128 == 0.
void prefill_swiglu_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* gate, float alpha_gate,
                          const uint8_t* up, float alpha_up, int N, int K, float out_input_scale, uint8_t* q_out,
                          uint8_t* sf_out, cudaStream_t s);

// ya = x . Wa^T, yb = x . Wb^T for two BF16 [48][K] matrices and any M, row-invariant (the prefill path's
// DeltaNet a and b projections).
void narrow_bf16(const float* x, int M, const __nv_bfloat16* wa, const __nv_bfloat16* wb, float* ya, float* yb, int N,
                 int K, cudaStream_t s);

// RMSNorm (as rmsnorm) that also writes the FP16 copy and scales to_half_rows would make from its output.
void rmsnorm_half(const float* x, const __nv_bfloat16* w, float* out, __half* xh, float* xinv, int rows, int H, float eps,
                  bool gemma, cudaStream_t s);
// x [M][K] FP32 -> FP16 with a power-of-two scale per row (xinv[m] undoes it).
void to_half_rows(const float* x, int M, int K, __half* out, float* xinv, cudaStream_t s);
// y[M][N] = scale2 * x . W^T, W NVFP4 in the tiled layout (wscale is unused: the scales are in the blob).
void stream_gemm_nvfp4(const __half* x, const float* xinv, int M, const uint8_t* w, const uint8_t* wscale,
                       float scale2, float* y, int N, int K, cudaStream_t s);
// y[M][N] = wscale * x . W^T, W FP8 E4M3 in the tiled layout.
void stream_gemm_fp8(const __half* x, const float* xinv, int M, const uint8_t* w, float wscale, float* y, int N,
                     int K, cudaStream_t s);

// Several matrices that share the input x, in one launch (fewer launch ramps and tails): each target
// gets y = scale * x . W^T. Row-invariant like the single-matrix calls, and bit-identical to them.
struct StreamTarget {
  const uint8_t* w;
  const uint8_t* wscale;  // NVFP4 block scales (nullptr for FP8)
  float scale;            // NVFP4 global scale, or the FP8 per-tensor scale
  float* y;
  int N;
};
void stream_gemm_multi(bool nvfp4, const __half* x, const float* xinv, int M, const StreamTarget* t, int n, int K,
                       cudaStream_t s);

// Top-K (K <= 64) of each row of x [rows][V]: values descending, ties to the lower index (so entry 0 is
// what std::max_element returns). `scratch` needs topk_scratch_floats(rows, K) floats.
size_t topk_scratch_floats(int rows, int K);
void topk_rows(const float* x, int rows, int V, int K, float* scratch, float* vals, int* ids, cudaStream_t s);

// Dequantize a whole matrix to BF16 (for the cuBLAS path).
void dequant_nvfp4(const uint8_t* w, const uint8_t* wscale, float scale2, __nv_bfloat16* out, int N, int K,
                   cudaStream_t s);
void dequant_fp8(const uint8_t* w, float wscale, __nv_bfloat16* out, int N, int K, cudaStream_t s);
void to_bf16(const float* x, __nv_bfloat16* out, int n, cudaStream_t s);
// y[M][N] = x[M][K] . W[N][K]^T with BF16 inputs and FP32 output, on tensor cores.
void gemm_bf16_cublas(cublasHandle_t h, const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y,
                      int N, int K);

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
                  cudaStream_t s);
// Causal attention of M queries (positions pos0..) over the cache, on tensor cores over fixed 4096-key ranges at
// fixed positions (combined afterwards), so each row's result is independent of M: decode, verify and prefill
// (in slices of queries). `scratch` needs attention_rows_scratch_floats(M, Hq, D, ctx) floats.
// fixed_ctx > 0 (a captured graph, pos0 read on the device): launch the ranges covering fixed_ctx keys
// whatever the position; ranges past the last visible key write an empty partial that the combine skips,
// so the result is bit for bit the eager launch's.
size_t attention_rows_scratch_floats(int M, int Hq, int D, int ctx);
void attention_rows(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, DevPos pos0, int M,
                    int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s, int fixed_ctx = 0);
// Sets the kernels' shared-memory attributes once, outside any graph capture.
void prepare_kernels();
// ---- The DFlash2 drafter (draft_kernels.cu). ----
// dst[r][0..H) = bf16(src[r][0..H)), rows dst_stride apart (the target features the drafter conditions on).
void copy_rows_bf16(const float* src, int rows, int H, __nv_bfloat16* dst, int dst_stride, cudaStream_t s);
// Per-head RMSNorm (plain weight) and NeoX RoPE over the full 128-dim head at positions pos0 + row, in place.
void draft_qk_rope(float* x, int rows, int heads, int row_stride, const __nv_bfloat16* norm, DevPos pos0, float theta,
                   float eps, cudaStream_t s);
// k/v rows [rows][kv_size] into the BF16 caches [position][kv_size] at positions pos0 ...
void draft_store_kv(const float* k, const float* v, int rows, int kv_size, DevPos pos0, __nv_bfloat16* kc,
                    __nv_bfloat16* vc, cudaStream_t s);
// The block's B queries (positions L .. L + B - 1) over the cached context [L + j - window_left, L) and the
// whole block (bidirectional within it). q [B][Hq][128], kb/vb [B][Hkv][128].
void draft_attention(const float* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const float* kb,
                     const float* vb, DevPos L, int B, int window_left, int Hq, int Hkv, float* out, cudaStream_t s);
// DFlash2's two-tap grouped dynamic convolution over a block (side 0 wraps a sublayer's input, 1 its output).
void grouped_conv(const float* h, const float* coef, const __nv_bfloat16* base, int side, float* out, int rows,
                  int C, int groups, int block, cudaStream_t s);
// The candidate selector's 16 x 16 transition scores for E positions (hp: [E][256], cand/unary: [E][16]).
// anchor_dev (a captured graph): read the anchor from device memory instead of `anchor`.
void selector_lattice(const float* hp, const int* cand, const float* unary, const __nv_bfloat16* P,
                      const __nv_bfloat16* S, int anchor, int E, float* scores, cudaStream_t s,
                      const int* anchor_dev = nullptr);

}  // namespace ling::kernels
