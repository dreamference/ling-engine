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
void bf16_rows(const float* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s);
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
// out = rmsnorm(x) * w * silu(z), rows of length D.
void gated_rmsnorm(const float* x, const float* z, const __nv_bfloat16* w, float* out, int rows, int D,
                   float eps, cudaStream_t s);

// Full attention: q_gate [M][Hq*2*D] (per head: q then gate), k/v [M][Hkv*D]. Applies the Gemma-style
// q/k norms and NeoX RoPE on the first `rot` dims, writes q [M][Hq][D] and gate [M][Hq][D], and appends
// k/v to the BF16 caches [pos][Hkv][D] at positions pos0 .. pos0+M-1.
void attn_prepare(const float* q_gate, const float* k, const float* v, const __nv_bfloat16* q_norm,
                  const __nv_bfloat16* k_norm, int pos0, int M, int Hq, int Hkv, int D, int rot, float theta,
                  float eps, float* q, float* gate, __nv_bfloat16* kcache, __nv_bfloat16* vcache,
                  cudaStream_t s);
// Causal attention of M queries (positions pos0..) over the cache, split over key ranges and combined.
// `scratch` needs attention_scratch_floats(M, Hq, Hkv, D, ctx) floats.
size_t attention_scratch_floats(int M, int Hq, int Hkv, int D, int ctx);
// The rows path's attention (M <= 32): fixed 1024-key ranges, so each row's result is independent of M.
size_t attention_rows_scratch_floats(int M, int Hq, int D, int ctx);
void attention_rows(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, int pos0, int M, int Hq,
                    int Hkv, int D, float* scratch, float* out, cudaStream_t s);
void attention(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, int pos0, int M,
               int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s);

// ---- The DFlash2 drafter (draft_kernels.cu). ----
// dst[r][0..H) = bf16(src[r][0..H)), rows dst_stride apart (the target features the drafter conditions on).
void copy_rows_bf16(const float* src, int rows, int H, __nv_bfloat16* dst, int dst_stride, cudaStream_t s);
// Per-head RMSNorm (plain weight) and NeoX RoPE over the full 128-dim head at positions pos0 + row, in place.
void draft_qk_rope(float* x, int rows, int heads, int row_stride, const __nv_bfloat16* norm, int pos0, float theta,
                   float eps, cudaStream_t s);
// k/v rows [rows][kv_size] into the BF16 caches [position][kv_size] at positions pos0 ...
void draft_store_kv(const float* k, const float* v, int rows, int kv_size, int pos0, __nv_bfloat16* kc,
                    __nv_bfloat16* vc, cudaStream_t s);
// The block's B queries (positions L .. L + B - 1) over the cached context [L + j - window_left, L) and the
// whole block (bidirectional within it). q [B][Hq][128], kb/vb [B][Hkv][128].
void draft_attention(const float* q, const __nv_bfloat16* kc, const __nv_bfloat16* vc, const float* kb,
                     const float* vb, int L, int B, int window_left, int Hq, int Hkv, float* out, cudaStream_t s);
// DFlash2's two-tap grouped dynamic convolution over a block (side 0 wraps a sublayer's input, 1 its output).
void grouped_conv(const float* h, const float* coef, const __nv_bfloat16* base, int side, float* out, int rows,
                  int C, int groups, int block, cudaStream_t s);
// The candidate selector's 16 x 16 transition scores for E positions (hp: [E][256], cand/unary: [E][16]).
void selector_lattice(const float* hp, const int* cand, const float* unary, const __nv_bfloat16* P,
                      const __nv_bfloat16* S, int anchor, int E, float* scores, cudaStream_t s);

}  // namespace ling::kernels
