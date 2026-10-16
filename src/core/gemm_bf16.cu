// The BF16 GEMM for more rows than the rows path takes (BACKLOG §19.23; it replaced cuBLAS's cublasGemmEx,
// the engine's one vendor GEMM): y[M][N] = x[M][K] . W[N][K]^T, BF16 inputs (both K-contiguous), FP32
// accumulation and output. Its one caller on the served path is the drafter's fc over a prompt's context
// rows (speculate.cpp); Engine::linear_bf16 and linear_bf16_drafter keep it for more rows than kMaxStreamRows.
//
// Tiling: a block computes 256 rows x 128 columns with 8 warps (4 along M x 2 along N, each 64 x 64) and
// m16n8k16 MMAs over two pipeline stages of 64 values of K. One thread issues each stage as two tensor copies
// (TMA: the activation and weight tiles, 128-byte swizzled so every ldmatrix phase is free of bank conflicts,
// rows past M or N zero-filled); the stage's full barrier completes when their bytes land and each warp
// releases it on its empty barrier after its MMAs, so the loop has no block-wide barrier. Blocks are rastered
// in groups of 4 row tiles, so the blocks resident at once share their tiles in L2. Measured against cuBLAS on
// the drafter's fc (bench/micro/gemm_bf16_vs_cublas.cu, 2026-10-16): x0.91 at 256 rows, x0.96 at 1,000, x0.84
// at 1,500, x1.03 at 2,048 (a prefill chunk); the cp.async form of the same tiles was x1.07 there.
//
// Numerics: each output is one FP32 accumulator taking the MMAs over K in ascending order of 16, whatever
// the tile; cuBLAS's kernel on the drafter's fc took the same order (bench/micro/gemm_bf16_vs_cublas.cu: bit
// for bit at 33 to 4,096 rows before cuBLAS left the link). No split-K, so a row's result does not depend
// on M or on the rows beside it.
#include "core/kernels.cuh"

#include <cuda.h>
#include <cudaTypedefs.h>

#include <stdexcept>
#include <string>

namespace ling::kernels {
namespace {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// Not volatile: a register-only instruction the compiler may interleave with the fragment loads.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int BM, int BN, int BK, int STAGES, int WM, int WN>
struct Cfg {
  static constexpr int kThreads = WM * WN * 32;
  static constexpr int kUnits = BK / 8;                 // 16-byte units per row per stage
  static constexpr int kRowBytes = BK * 2;
  static constexpr int kStageA = BM * kRowBytes, kStageBytes = (BM + BN) * kRowBytes;
  static constexpr int kSmem = STAGES * kStageBytes;
  static constexpr int kWMR = BM / WM, kMF = kWMR / 16;  // a warp's rows and m16 fragments
  static constexpr int kWN = BN / WN, kNF = kWN / 8;      // a warp's columns and n8 fragments
  // Byte offset of (row, 16-byte unit) in a tile: the unit XOR the row's low bits, so the 8 rows one
  // ldmatrix phase reads land on 8 different 16-byte bank groups.
  static __device__ __forceinline__ int swz(int r, int u) {
    if constexpr (kUnits == 8) return r * kRowBytes + ((u ^ (r & 7)) << 4);
    else return r * kRowBytes + ((u ^ ((r >> 1) & 3)) << 4);
  }
};

// ---- The TMA mainloop: one thread issues each stage as two tensor copies (the activation and weight tiles,
// swizzled by the copy engine exactly as swz() reads them, rows past M or N zero-filled), the stage's full
// barrier completes when their bytes land, and each warp releases the stage on its empty barrier after its
// MMAs: no block-wide barrier in the loop. ----
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, int parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}" ::"r"(bar), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}

template <int BM, int BN, int BK, int STAGES, int GROUP_M, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32) gemm_bf16_tma_kernel(const __grid_constant__ CUtensorMap tx,
                                                                 const __grid_constant__ CUtensorMap tw,
                                                                 float* __restrict__ y, int M, int N, int K) {
  using C = Cfg<BM, BN, BK, STAGES, WM, WN>;
  static_assert(BK == 64 || BK == 32, "a 128- or 64-byte swizzled row");
  extern __shared__ unsigned char smem_raw[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int tm = (M + BM - 1) / BM, tn = (N + BN - 1) / BN;
  const int pid = blockIdx.x, in_group = GROUP_M * tn, group = pid / in_group;
  const int first_m = group * GROUP_M, gsz = min(tm - first_m, GROUP_M);
  const int pm = first_m + (pid % in_group) % gsz, pn = (pid % in_group) / gsz;
  const int m0 = pm * BM, n0 = pn * BN;
  const uint32_t base = (smem_addr(smem_raw) + 1023) & ~1023u;  // the swizzle repeats every 1,024 bytes
  const uint32_t bars = base + STAGES * C::kStageBytes;          // full[STAGES], then empty[STAGES]
  const int KT = K / BK;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(bars + 8 * s, 1);
      mbar_init(bars + 8 * (STAGES + s), WM * WN);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto issue = [&](int kt, int s) {
    const uint32_t full = bars + 8 * s, dst = base + s * C::kStageBytes;
    mbar_expect_tx(full, C::kStageBytes);
    tma_2d(dst, &tx, kt * BK, m0, full);
    tma_2d(dst + C::kStageA, &tw, kt * BK, n0, full);
  };
  if (tid == 0)
    for (int s = 0; s < STAGES && s < KT; ++s) issue(s, s);

  float acc[C::kMF][C::kNF][4] = {};
  const int a_row = wm * C::kWMR + (lane & 15), a_u = lane >> 4;
  const int b_row = wn * C::kWN + (lane & 7) + ((lane >> 4) << 3), b_u = (lane >> 3) & 1;
  for (int kt = 0; kt < KT; ++kt) {
    const int slot = kt % STAGES, phase = (kt / STAGES) & 1;
    mbar_wait(bars + 8 * slot, phase);
    const uint32_t s = base + slot * C::kStageBytes;
    uint32_t a[2][C::kMF][4], b[2][C::kNF / 2][4];
    auto frags = [&](int kk, int buf) {
#pragma unroll
      for (int mf = 0; mf < C::kMF; ++mf) ldsm_x4(a[buf][mf], s + C::swz(a_row + mf * 16, kk * 2 + a_u));
#pragma unroll
      for (int nb = 0; nb < C::kNF / 2; ++nb)
        ldsm_x4(b[buf][nb], s + C::kStageA + C::swz(b_row + nb * 16, kk * 2 + b_u));
    };
    frags(0, 0);
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {  // ascending K: the accumulation order the numerics depend on
      if (kk + 1 < BK / 16) frags(kk + 1, (kk + 1) & 1);
#pragma unroll
      for (int mf = 0; mf < C::kMF; ++mf)
#pragma unroll
        for (int nf = 0; nf < C::kNF; ++nf)
          mma_bf16(acc[mf][nf], a[kk & 1][mf], b[kk & 1][nf >> 1][(nf & 1) * 2], b[kk & 1][nf >> 1][(nf & 1) * 2 + 1]);
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(bars + 8 * (STAGES + slot));
    if (tid == 0 && kt + STAGES < KT) {
      mbar_wait(bars + 8 * (STAGES + slot), phase);
      issue(kt + STAGES, slot);
    }
  }

  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mf = 0; mf < C::kMF; ++mf)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int row = m0 + wm * C::kWMR + mf * 16 + g + 8 * h;
      if (row >= M) continue;
      float* yr = y + static_cast<size_t>(row) * N;
#pragma unroll
      for (int nf = 0; nf < C::kNF; ++nf) {
        const int col = n0 + wn * C::kWN + nf * 8 + 2 * t;
        if (col + 1 < N) {
          *reinterpret_cast<float2*>(yr + col) = make_float2(acc[mf][nf][2 * h], acc[mf][nf][2 * h + 1]);
        } else if (col < N) {
          yr[col] = acc[mf][nf][2 * h];
        }
      }
    }
}

// A row-major BF16 matrix [rows][K] as a tensor map with boxes of BK x box_rows, swizzled for the kernel.
CUtensorMap tensor_map(const __nv_bfloat16* p, int rows, int K, int box_rows, int BK) {
  static PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult q;
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess || !fn)
      throw std::runtime_error("gemm_bf16: cuTensorMapEncodeTiled not available");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn);
  }();
  CUtensorMap m;
  const cuuint64_t dims[2] = {cuuint64_t(K), cuuint64_t(rows)};
  const cuuint64_t strides[1] = {cuuint64_t(K) * 2};
  const cuuint32_t box[2] = {cuuint32_t(BK), cuuint32_t(box_rows)};
  const cuuint32_t elem[2] = {1, 1};
  const CUresult r = encode(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<__nv_bfloat16*>(p), dims, strides, box,
                            elem, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            BK == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) throw std::runtime_error("gemm_bf16: cuTensorMapEncodeTiled failed: " + std::to_string(int(r)));
  return m;
}

template <int BM, int BN, int BK, int STAGES, int GROUP_M, int WM = 2, int WN = 4>
void launch_tma(const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s) {
  using C = Cfg<BM, BN, BK, STAGES, WM, WN>;
  constexpr int smem = C::kSmem + 1024 + 16 * STAGES;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(gemm_bf16_tma_kernel<BM, BN, BK, STAGES, GROUP_M, WM, WN>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    configured = true;
  }
  const CUtensorMap tx = tensor_map(x, M, K, BM, BK), tw = tensor_map(w, N, K, BN, BK);
  const int blocks = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
  gemm_bf16_tma_kernel<BM, BN, BK, STAGES, GROUP_M, WM, WN><<<blocks, C::kThreads, smem, s>>>(tx, tw, y, M, N, K);
}

}  // namespace

void gemm_bf16(const __nv_bfloat16* x, int M, const __nv_bfloat16* w, float* y, int N, int K, cudaStream_t s) {
  if (M < 1 || N < 1) return;
  if (K % 64 != 0 || N % 2 != 0)
    throw std::runtime_error("gemm_bf16: K % 64 and N % 2 must be 0 (K " + std::to_string(K) + ", N " +
                             std::to_string(N) + ")");
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16 != 0)
    throw std::runtime_error("gemm_bf16: x and W must be 16-byte aligned");
  launch_tma<256, 128, 64, 2, 4, 4, 2>(x, M, w, y, N, K, s);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("gemm_bf16: ") + cudaGetErrorString(e));
}

}  // namespace ling::kernels
