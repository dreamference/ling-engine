// The v0 kernels. Activations are FP32 between kernels, weights stay in the checkpoint's own encoding
// (NVFP4 for the MLP and the LM head, FP8 E4M3 for the attention projections, BF16 for the rest) and are
// dequantized in registers. Rows (M) are the tokens of one step: 1 for decode, up to a prefill chunk.
#pragma once

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_bf16.h>
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
// y = x . W^T, W BF16 [N][K].
void gemv_bf16(const float* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s);

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
void gdn_recurrent(const float* mixed, const float* g, const float* beta, float* state, float* out, int M,
                   int H, int HV, cudaStream_t s);
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
void attention(const float* q, const __nv_bfloat16* kcache, const __nv_bfloat16* vcache, int pos0, int M,
               int Hq, int Hkv, int D, float* scratch, float* out, cudaStream_t s);

}  // namespace ling::kernels
