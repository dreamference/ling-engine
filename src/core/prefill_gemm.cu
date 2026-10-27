// Prefill GEMMs (M > 32 rows) on block-scaled tensor cores, reading the weights in the tiled layout the
// decode kernels read (stream_gemm.cu), so no second copy of any weight exists.
//
// Numerics are production's (specs/DREAMFERENCE_LING_ENGINE_SESSIONS.md §11): the NVFP4 FFN multiplies NVFP4 activations (E2M1 values, one
// E4M3 scale per 16, the checkpoint's input_scale as the global scale) with mma.sync kind::mxf4nvf4, and
// the FP8 projections multiply FP8 E4M3 activations (static per-tensor input_scale) with kind::f8f6f4.
// Accumulation is FP32; y = alpha * acc with alpha = input_scale * weight scale, as SGLang computes it.
//
// Tiling: a block computes 128 tokens x 256 weight rows (3 stages) or, when that leaves under two waves,
// 128 x 128 (4 stages), with 8 warps (2 x 4). One pipeline stage holds 64 bytes of K per row (128 NVFP4
// values, one tiled block's width; or 64 FP8 values, half of one): activations and weights in shared
// memory with their 16-byte units XOR-swizzled by row pair, so every ldmatrix phase is free of bank
// conflicts, plus the rows' E4M3 block scales; cp.async fills the stages ahead of the MMAs. The FFN's gate
// and up run as one GEMM whose epilogue applies SwiGLU and writes the down projection's NVFP4 input.
//
// Measured at 2,048 rows: NVFP4 230-290 TFLOPS (CUTLASS 4.5.1's SM120 example: 325 on the gate shape, 262
// on the down shape), FP8 ~133. A warp-specialized variant (one producer warp issuing every cp.async of a
// stage, mbarriers between it and 8 consumer warps) ran FP8 ~8% faster and NVFP4 3-4x slower: one warp
// cannot issue a stage's ~1,800 copies in the time the MMAs take. TMA would issue them in a few
// instructions, but the FP8 tiled layout (decode's) has no 64-byte row runs for a swizzled box.
// FP8 blocks also prefetch their weight tiles into L2 two chunks ahead of the MMAs (l2_prefetch_fp8, a
// hint): with the stage copies alone the FP8 weights streamed at ~100 GB/s at 128-512 rows, with it ~150-175.
#include "core/kernels.cuh"

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "core/elem.cuh"
#include "core/prefill_tile.cuh"

namespace ling::kernels {
namespace {

using namespace pf;  // the block tile (prefill_tile.cuh)

template <bool FP4, int BN, int STAGES, bool SWIGLU = false, bool OUT_BF16 = false>
__global__ void __launch_bounds__(kThreads) prefill_gemm_kernel(GemmArgs p) {
  using S = Shape<BN, STAGES>;
  constexpr int NF = S::kNF;
  extern __shared__ __align__(128) uint8_t smem[];
  const int m0 = blockIdx.x * kBM, n0 = blockIdx.y * BN;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int wm = warp & 1, wn = warp >> 1;
  const int KT = p.K / kRowBytes / (FP4 ? 2 : 1);  // stages along K

  float acc[4][NF][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NF; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const Loader<FP4, BN, STAGES, SWIGLU> loader(p, m0, n0);
  if constexpr (!FP4) {  // the weight tiles kL2LeadFp8 chunks ahead of the MMAs, into L2 (prefill_tile.cuh)
#pragma unroll
    for (int c = 0; c < kL2LeadFp8; ++c) l2_prefetch_fp8<BN>(p, n0, c);
  }
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) loader.load(smem, s, s);
    cp_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    const int nk = kt + STAGES - 1;
    if (nk < KT) loader.load(smem, nk % STAGES, nk);
    cp_commit();
    if constexpr (!FP4) {
      if ((kt & 1) == 0) l2_prefetch_fp8<BN>(p, n0, (kt >> 1) + kL2LeadFp8);
    }
    compute_stage<FP4, BN, STAGES>(smem + (kt % STAGES) * S::kStageBytes, acc, lane, wm, wn);
  }
  cp_wait<0>();
  epilogue<BN, STAGES, SWIGLU, OUT_BF16>(p, acc, m0, n0, lane, wm, wn);
}

// ---- Activation quantization, as production's kernels do it (activations are BF16 there, so each value is
// rounded to BF16 first). ----


// The NVFP4 encoding of four values held by each of four adjacent lanes (one block of 16).
__device__ __forceinline__ void put_nvfp4(const float (&v)[4], float gscale, size_t quad, uint8_t* __restrict__ q,
                                          uint8_t* __restrict__ sf) {
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  const __nv_fp8_e4m3 sc(gscale * (amax * (1.f / 6.f)));
  const float sv = static_cast<float>(sc);
  const float os = sv != 0.f ? 1.f / (sv * (1.f / gscale)) : 0.f;
  const uint32_t lo = e2m1x2(v[0] * os, v[1] * os), hi = e2m1x2(v[2] * os, v[3] * os);
  reinterpret_cast<uint16_t*>(q)[quad] = static_cast<uint16_t>(lo | (hi << 8));
  if ((quad & 3) == 0) sf[quad >> 2] = sc.__x;
}

__device__ __forceinline__ uint32_t put_fp8x4(const float (&v)[4], float inv_scale) {
  uint32_t w = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k)
    w |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[k] * inv_scale, __NV_SATFINITE, __NV_E4M3)) << (8 * k);
  return w;
}

// Four lanes per block of 16 values (each loads one float4, coalesced): scale = E4M3(gscale * amax / 6),
// values = E2M1(x * gscale / scale). With `up`, the value quantized is silu(x) * up (the FFN's gate and up
// products, fused so the 17,408-wide activation is never written in FP32).
template <bool SILU>
__global__ void quant_nvfp4_kernel(const float* __restrict__ x, const float* __restrict__ up, size_t quads, float gscale,
                                   uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads; i += stride) {
    float4 f = reinterpret_cast<const float4*>(x)[i];
    if constexpr (SILU) {
      const float4 u = reinterpret_cast<const float4*>(up)[i];
      f = make_float4(silu(f.x) * u.x, silu(f.y) * u.y, silu(f.z) * u.z, silu(f.w) * u.w);
    }
    const float v[4] = {bf16_round(f.x), bf16_round(f.y), bf16_round(f.z), bf16_round(f.w)};
    put_nvfp4(v, gscale, i, q, sf);
  }
}

// RMSNorm of each row (Gemma's x * rstd * (1 + w)) straight into the next GEMM's encoding, and into FP32 too
// when `out` is given (or into BF16, OutT = __nv_bfloat16: the --prefill-bf16 copy, rounded as the quantizer
// rounds). One block per row; each thread takes whole groups of four values, so four adjacent lanes hold one
// NVFP4 block. H % 1024 == 0.
template <bool FP4, typename OutT = float>
__global__ void __launch_bounds__(256)
    norm_quant_kernel(const float* __restrict__ x, const __nv_bfloat16* __restrict__ w, OutT* __restrict__ out, int H,
                      float eps, float scale, uint8_t* __restrict__ q, uint8_t* __restrict__ sf) {
  __shared__ float red[8];
  const size_t r = blockIdx.x;
  const float4* row = reinterpret_cast<const float4*>(x + r * H);
  float ss = 0.f;
  for (int i = threadIdx.x; i < H / 4; i += 256) {
    const float4 f = row[i];
    ss = fmaf(f.x, f.x, fmaf(f.y, f.y, fmaf(f.z, f.z, fmaf(f.w, f.w, ss))));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) tot += red[i];
  const float rstd = rsqrtf(tot / H + eps);
  for (int i = threadIdx.x; i < H / 4; i += 256) {
    const float4 f = row[i];
    const float xs[4] = {f.x, f.y, f.z, f.w};
    float v[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = xs[k] * rstd * (1.f + __bfloat162float(w[4 * i + k]));
    if (out) elem::store4(out + r * H, i, make_float4(v[0], v[1], v[2], v[3]));
#pragma unroll
    for (int k = 0; k < 4; ++k) v[k] = bf16_round(v[k]);
    const size_t quad = r * (H / 4) + i;
    if constexpr (FP4) put_nvfp4(v, scale, quad, q, sf);
    else reinterpret_cast<uint32_t*>(q)[quad] = put_fp8x4(v, scale);
  }
}

// The DeltaNet output norm (rmsnorm(x) * w * silu(z) per head of D = 128) straight into FP8 for the
// out-projection: one warp per head row, four values per lane. InT: the recurrence's output and the z
// projection, float or (--prefill-bf16) __nv_bfloat16.
template <typename InT>
__global__ void gated_norm_fp8_kernel(const InT* __restrict__ x, const InT* __restrict__ z,
                                      const __nv_bfloat16* __restrict__ w, size_t rows, float eps, float inv_scale,
                                      uint8_t* __restrict__ q) {
  const size_t r = blockIdx.x * static_cast<size_t>(blockDim.x / 32) + threadIdx.x / 32;
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const float4 f = elem::load4(x + r * 128, lane);
  const float4 g = elem::load4(z + r * 128, lane);
  float ss = f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float rstd = rsqrtf(ss / 128 + eps);
  const float xs[4] = {f.x, f.y, f.z, f.w}, zs[4] = {g.x, g.y, g.z, g.w};
  float v[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) v[k] = bf16_round(xs[k] * rstd * __bfloat162float(w[4 * lane + k]) * silu(zs[k]));
  reinterpret_cast<uint32_t*>(q)[r * 32 + lane] = put_fp8x4(v, inv_scale);
}

// x * sigmoid(gate) straight into FP8 (the attention output, for the o-projection).
__global__ void sigmoid_mul_fp8_kernel(const float* __restrict__ x, const float* __restrict__ gate, size_t quads,
                                       float inv_scale, uint8_t* __restrict__ q) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float4 f = reinterpret_cast<const float4*>(x)[i], g = reinterpret_cast<const float4*>(gate)[i];
    const float v[4] = {bf16_round(f.x / (1.f + __expf(-g.x))), bf16_round(f.y / (1.f + __expf(-g.y))),
                        bf16_round(f.z / (1.f + __expf(-g.z))), bf16_round(f.w / (1.f + __expf(-g.w)))};
    reinterpret_cast<uint32_t*>(q)[i] = put_fp8x4(v, inv_scale);
  }
}

// Four values per thread: E4M3(x / input_scale), saturated to +-448. InT = __nv_bfloat16 (--prefill-bf16)
// reads the BF16 copy of the norm, which the BF16 rounding below leaves as it is.
template <typename InT>
__global__ void quant_fp8_kernel(const InT* __restrict__ x, size_t quads, float inv_scale, uint8_t* __restrict__ q) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < quads;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const float4 f = elem::load4(x, i);
    const float v[4] = {bf16_round(f.x) * inv_scale, bf16_round(f.y) * inv_scale, bf16_round(f.z) * inv_scale,
                        bf16_round(f.w) * inv_scale};
    uint32_t w = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
      w |= static_cast<uint32_t>(__nv_cvt_float_to_fp8(v[k], __NV_SATFINITE, __NV_E4M3)) << (8 * k);
    reinterpret_cast<uint32_t*>(q)[i] = w;
  }
}

// Narrow BF16 matrices (the DeltaNet a and b projections, 48 rows each) on tensor cores: a one-warp block
// takes 16 rows of x and a quarter of the 2N output columns (blockIdx.y), m16n8k16 MMAs over K in order. x is
// rounded to BF16 (production's activations), or (InT = __nv_bfloat16, --prefill-bf16) is BF16 already: the
// same fragments either way. A row's result does not depend on M.
template <typename InT>
__global__ void __launch_bounds__(32)
    narrow_bf16_kernel(const InT* __restrict__ x, int M, const __nv_bfloat16* __restrict__ wa,
                       const __nv_bfloat16* __restrict__ wb, float* __restrict__ ya, float* __restrict__ yb, int N, int K) {
  constexpr int NF = 3;  // n8 fragments per block: 4 column groups x 3 x 8 = 96 = 2 x 48 columns
  const int lane = threadIdx.x & 31, warp = blockIdx.y, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const InT* xa = x + static_cast<size_t>(min(r0 + g, M - 1)) * K;
  const InT* xb = x + static_cast<size_t>(min(r0 + g + 8, M - 1)) * K;
  const __nv_bfloat16* wrow[NF];
#pragma unroll
  for (int j = 0; j < NF; ++j) {
    const int col = (warp * NF + j) * 8 + g;  // this lane's B column
    wrow[j] = col < N ? wa + static_cast<size_t>(col) * K : wb + static_cast<size_t>(col - N) * K;
  }
  float acc[NF][4] = {};
  for (int k = 0; k < K; k += 16) {
    const uint32_t a0 = elem::pack_bf16x2(xa + k + 2 * t);
    const uint32_t a1 = elem::pack_bf16x2(xb + k + 2 * t);
    const uint32_t a2 = elem::pack_bf16x2(xa + k + 8 + 2 * t);
    const uint32_t a3 = elem::pack_bf16x2(xb + k + 8 + 2 * t);
#pragma unroll
    for (int j = 0; j < NF; ++j) {
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(wrow[j] + k + 2 * t);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(wrow[j] + k + 8 + 2 * t);
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
          "{%0, %1, %2, %3};"
          : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
#pragma unroll
  for (int j = 0; j < NF; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = r0 + g + 8 * h;
      if (row >= M) continue;
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int col = (warp * NF + j) * 8 + 2 * t + e;
        if (col < N) ya[static_cast<size_t>(row) * N + col] = acc[j][2 * h + e];
        else yb[static_cast<size_t>(row) * N + col - N] = acc[j][2 * h + e];
      }
    }
}

int grid_for(size_t n) { return static_cast<int>(std::min<size_t>((n + 255) / 256, 48 * 64)); }

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <bool FP4, int BN, int STAGES, bool SWIGLU = false, bool OUT_BF16 = false>
void launch_shape(const GemmArgs& p, cudaStream_t s) {
  using S = Shape<BN, STAGES>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(prefill_gemm_kernel<FP4, BN, STAGES, SWIGLU, OUT_BF16>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSmemBytes);
    configured = true;
  }
  const dim3 grid((p.M + kBM - 1) / kBM, (SWIGLU ? 2 * p.N : p.N) / BN);
  prefill_gemm_kernel<FP4, BN, STAGES, SWIGLU, OUT_BF16><<<grid, kThreads, S::kSmemBytes, s>>>(p);
}

// LING_GEMM_SHAPE (experiments): 0 automatic, 1: 128 x 128 x 4 stages, 2: 128 x 256 x 3 stages.
int gemm_shape() {
  static const int v = [] {
    const char* e = std::getenv("LING_GEMM_SHAPE");
    return e ? std::atoi(e) : 0;
  }();
  return v;
}

template <bool FP4, bool OUT_BF16 = false>
void launch_gemm(const GemmArgs& p, cudaStream_t s) {
  if (p.N % 128 != 0 || p.K % 128 != 0) throw std::runtime_error("prefill_gemm: N % 128 and K % 128 must be 0");
  if (p.M < 1) return;
  int shape = gemm_shape();
  // 128 x 256 tiles (measured +12-25% on the NVFP4 matrices at 512-2,048 rows) when they still give two waves.
  if (shape == 0) shape = (p.M + kBM - 1) / kBM * (p.N / 256) >= 96 ? 2 : 1;
  if (shape == 2 && p.N % 256 == 0) launch_shape<FP4, 256, 3, false, OUT_BF16>(p, s);
  else launch_shape<FP4, 128, 4, false, OUT_BF16>(p, s);
  check_launch(FP4 ? "prefill_gemm_nvfp4" : "prefill_gemm_fp8");
}

}  // namespace

void quant_act_nvfp4(const float* x, int M, int K, float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  quant_nvfp4_kernel<false><<<grid_for(quads), 256, 0, s>>>(x, nullptr, quads, 1.f / input_scale, q, sf);
  check_launch("quant_act_nvfp4");
}

void silu_mul_quant_nvfp4(const float* g, const float* u, int M, int K, float input_scale, uint8_t* q, uint8_t* sf,
                          cudaStream_t s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  quant_nvfp4_kernel<true><<<grid_for(quads), 256, 0, s>>>(g, u, quads, 1.f / input_scale, q, sf);
  check_launch("silu_mul_quant_nvfp4");
}

void quant_act_fp8(const float* x, int M, int K, float input_scale, uint8_t* q, cudaStream_t s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  quant_fp8_kernel<float><<<grid_for(quads), 256, 0, s>>>(x, quads, 1.f / input_scale, q);
  check_launch("quant_act_fp8");
}

void quant_act_fp8_bf16in(const __nv_bfloat16* x, int M, int K, float input_scale, uint8_t* q, cudaStream_t s) {
  const size_t quads = static_cast<size_t>(M) * K / 4;
  quant_fp8_kernel<__nv_bfloat16><<<grid_for(quads), 256, 0, s>>>(x, quads, 1.f / input_scale, q);
  check_launch("quant_act_fp8_bf16in");
}

void rmsnorm_quant(const float* x, const __nv_bfloat16* w, float* out, int M, int H, float eps, bool nvfp4,
                   float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s) {
  if (H % 1024 != 0) throw std::runtime_error("rmsnorm_quant: H % 1024 must be 0");
  if (nvfp4) norm_quant_kernel<true><<<M, 256, 0, s>>>(x, w, out, H, eps, 1.f / input_scale, q, sf);
  else norm_quant_kernel<false><<<M, 256, 0, s>>>(x, w, out, H, eps, 1.f / input_scale, q, nullptr);
  check_launch("rmsnorm_quant");
}

void rmsnorm_quant_bf16(const float* x, const __nv_bfloat16* w, __nv_bfloat16* out, int M, int H, float eps, bool nvfp4,
                        float input_scale, uint8_t* q, uint8_t* sf, cudaStream_t s) {
  if (H % 1024 != 0) throw std::runtime_error("rmsnorm_quant_bf16: H % 1024 must be 0");
  if (nvfp4) norm_quant_kernel<true, __nv_bfloat16><<<M, 256, 0, s>>>(x, w, out, H, eps, 1.f / input_scale, q, sf);
  else norm_quant_kernel<false, __nv_bfloat16><<<M, 256, 0, s>>>(x, w, out, H, eps, 1.f / input_scale, q, nullptr);
  check_launch("rmsnorm_quant_bf16");
}

void gated_rmsnorm_fp8(const float* x, const float* z, const __nv_bfloat16* w, int rows, int D, float eps,
                       float input_scale, uint8_t* q, cudaStream_t s) {
  if (D != 128) throw std::runtime_error("gated_rmsnorm_fp8: D must be 128");
  gated_norm_fp8_kernel<float><<<(rows + 7) / 8, 256, 0, s>>>(x, z, w, rows, eps, 1.f / input_scale, q);
  check_launch("gated_rmsnorm_fp8");
}

void gated_rmsnorm_fp8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* z, const __nv_bfloat16* w, int rows, int D,
                            float eps, float input_scale, uint8_t* q, cudaStream_t s) {
  if (D != 128) throw std::runtime_error("gated_rmsnorm_fp8_bf16: D must be 128");
  gated_norm_fp8_kernel<__nv_bfloat16><<<(rows + 7) / 8, 256, 0, s>>>(x, z, w, rows, eps, 1.f / input_scale, q);
  check_launch("gated_rmsnorm_fp8_bf16");
}

void sigmoid_mul_fp8(const float* x, const float* gate, size_t n, float input_scale, uint8_t* q, cudaStream_t s) {
  sigmoid_mul_fp8_kernel<<<grid_for(n / 4), 256, 0, s>>>(x, gate, n / 4, 1.f / input_scale, q);
  check_launch("sigmoid_mul_fp8");
}

void prefill_gemm_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* w, float alpha, float* y, int ldy,
                        int N, int K, bool accumulate, cudaStream_t s, const uint8_t* wsf) {
  GemmArgs p{xq, xs, w, alpha, y, ldy, M, N, K, accumulate};
  p.wsf = wsf;
  launch_gemm<true>(p, s);
}

void narrow_bf16(const float* x, int M, const __nv_bfloat16* wa, const __nv_bfloat16* wb, float* ya, float* yb, int N,
                 int K, cudaStream_t s) {
  if (2 * N != 96 || K % 16 != 0) throw std::runtime_error("narrow_bf16: built for two 48-row matrices, K % 16 == 0");
  narrow_bf16_kernel<float><<<dim3((M + 15) / 16, 4), 32, 0, s>>>(x, M, wa, wb, ya, yb, N, K);
  check_launch("narrow_bf16");
}

void narrow_bf16_bf16in(const __nv_bfloat16* x, int M, const __nv_bfloat16* wa, const __nv_bfloat16* wb, float* ya,
                        float* yb, int N, int K, cudaStream_t s) {
  if (2 * N != 96 || K % 16 != 0) throw std::runtime_error("narrow_bf16: built for two 48-row matrices, K % 16 == 0");
  narrow_bf16_kernel<__nv_bfloat16><<<dim3((M + 15) / 16, 4), 32, 0, s>>>(x, M, wa, wb, ya, yb, N, K);
  check_launch("narrow_bf16_bf16in");
}

void prefill_swiglu_nvfp4(const uint8_t* xq, const uint8_t* xs, int M, const uint8_t* gate, float alpha_gate,
                          const uint8_t* up, float alpha_up, int N, int K, float out_input_scale, uint8_t* q_out,
                          uint8_t* sf_out, cudaStream_t s, const uint8_t* gate_sf, const uint8_t* up_sf) {
  if (N % 128 != 0 || K % 128 != 0) throw std::runtime_error("prefill_swiglu_nvfp4: N % 128 and K % 128 must be 0");
  if (M < 1) return;
  GemmArgs p{xq, xs, gate, alpha_gate, nullptr, 0, M, N, K, false};
  p.w2 = up;
  p.alpha2 = alpha_up;
  p.gscale_out = 1.f / out_input_scale;
  p.q_out = q_out;
  p.sf_out = sf_out;
  p.wsf = gate_sf;
  p.wsf2 = up_sf;
  if ((M + kBM - 1) / kBM * (2 * N / 256) >= 96) launch_shape<true, 256, 3, true>(p, s);
  else launch_shape<true, 128, 4, true>(p, s);
  check_launch("prefill_swiglu_nvfp4");
}

void prefill_gemm_fp8(const uint8_t* xq, int M, const uint8_t* w, float alpha, float* y, int ldy, int N, int K,
                      bool accumulate, cudaStream_t s) {
  launch_gemm<false>(GemmArgs{xq, nullptr, w, alpha, y, ldy, M, N, K, accumulate}, s);
}

void prefill_gemm_fp8_bf16out(const uint8_t* xq, int M, const uint8_t* w, float alpha, __nv_bfloat16* y, int ldy, int N,
                              int K, cudaStream_t s) {
  GemmArgs p{xq, nullptr, w, alpha, nullptr, ldy, M, N, K, false};
  p.y_bf16 = y;
  launch_gemm<false, true>(p, s);
}

}  // namespace ling::kernels
