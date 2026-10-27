// One K chunk of the weight-streaming GEMM (stream_gemm.cu) for one lane: its weights' FP16 dequantization and the
// MMAs of the chunk's eight K steps into the accumulators. Shared by stream_gemm_body and the fused verify + prefill
// kernel (fused_gemm.cu), so both compute every element with the same instructions on the same operands. The
// numerics, the layout and the K permutation are described at the top of stream_gemm.cu.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

#include "core/kernels.cuh"

namespace ling::kernels::sg {
namespace {

constexpr int kWarps = 8;
constexpr int kChunk = 128;
// Bytes of one (16-row tile, 128-value chunk) block: NVFP4 values (16 x 64) then their E4M3 scales
// (16 x 8); FP8 values (16 x 128).
constexpr int kTileBytesFp4 = 16 * 64 + 16 * 8, kTileBytesFp8 = 16 * 128;
// B26: a coded NVFP4 blob is every block's 1,024 value bytes (aligned, as the raw blob's 1,152-byte blocks are), then
// every block's 96 code bytes (kernels.cuh, retile_nvfp4_coded).
constexpr int kValueBytesFp4 = 16 * 64, kCodeBytesFp4 = 128 * 6 / 8;

__device__ __forceinline__ uint32_t e2m1x2_to_h2(uint32_t byte) {
  uint32_t out;
  asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %1;\n cvt.rn.f16x2.e2m1x2 %0, b0;\n}"
      : "=r"(out)
      : "r"(byte));
  return out;
}

__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t pair) {
  uint32_t out;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(out) : "r"(pair));
  return out;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t out;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
  return out;
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Per lane and chunk: the weights of rows g and g + 8 of every tile.
template <int TILES, bool FP4, bool CODED>
struct WeightRegs {
  uint4 v[TILES][2][FP4 ? 1 : 2];
  uint32_t sc[TILES][2];  // NVFP4: the two E4M3 block scales of the lane's 32 values (low byte first)
  uint32_t cw[TILES][CODED ? 2 : 1];  // CODED: the lane's code bytes for both of its slots (low bytes, high nibbles)
};

__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Chunk c of x's rows g + 8j (j < NT; rows >= M are zeros) times the lane's weights `cur` (rows g and g + 8 of each
// of TILES tiles), added into acc. tab: a coded matrix's scale-code table (CODED).
template <int NT, int TILES, bool FP4, bool CODED>
__device__ __forceinline__ void stream_chunk(const __half* x, int M, int K, int c, int g, int t,
                                             const WeightRegs<TILES, FP4, CODED>& cur,
                                             const uint8_t (&tab)[CODED ? kScaleCodes : 1], float (&acc)[TILES][NT][4]) {
  // This chunk's activations: 32 halves (16 words) per token for tokens g + 8j.
  uint32_t xw[NT][16];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    const int n = g + 8 * j;
    if (n < M) {
      const uint4* p = reinterpret_cast<const uint4*>(x + static_cast<size_t>(n) * K + c * kChunk + t * 32);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = __ldg(p + q);
        xw[j][4 * q] = v.x;
        xw[j][4 * q + 1] = v.y;
        xw[j][4 * q + 2] = v.z;
        xw[j][4 * q + 3] = v.w;
      }
    } else {
#pragma unroll
      for (int q = 0; q < 16; ++q) xw[j][q] = 0u;
    }
  }
  uint32_t sc_h2[TILES][2][2];
  if constexpr (FP4) {
#pragma unroll
    for (int i = 0; i < TILES; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t pair = cur.sc[i][h];
        if constexpr (CODED) {  // the two codes -> the two E4M3 scale bytes, exactly the raw blob's
          const uint32_t f = ((cur.cw[i][0] >> (8 * h)) & 0xffu) | (((cur.cw[i][1] >> (4 * h)) & 0xfu) << 8);
          pair = uint32_t(tab[f & 63u]) | (uint32_t(tab[(f >> 6) & 63u]) << 8);
        }
        const uint32_t s2 = e4m3x2_to_h2(pair);  // low half: first 16 values, high: next 16
        sc_h2[i][h][0] = (s2 & 0xffffu) | (s2 << 16);
        sc_h2[i][h][1] = (s2 >> 16) | (s2 & 0xffff0000u);
      }
  }
#pragma unroll
  for (int s = 0; s < 8; ++s) {
#pragma unroll
    for (int i = 0; i < TILES; ++i) {
      uint32_t a[2][2];  // [row g / g + 8][k pair 0 / 1]
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        if constexpr (FP4) {
          const uint32_t word = word_of(cur.v[i][h][0], s >> 1);
          const uint32_t bytes = word >> (16 * (s & 1));
          const uint32_t scale = sc_h2[i][h][s >> 2];
          a[h][0] = hmul2(e2m1x2_to_h2(bytes & 0xffu), scale);
          a[h][1] = hmul2(e2m1x2_to_h2((bytes >> 8) & 0xffu), scale);
        } else {
          const uint32_t word = word_of(cur.v[i][h][s >> 2], s & 3);
          a[h][0] = e4m3x2_to_h2(word & 0xffffu);
          a[h][1] = e4m3x2_to_h2(word >> 16);
        }
      }
#pragma unroll
      for (int j = 0; j < NT; ++j) mma16816(acc[i][j], a[0][0], a[1][0], a[0][1], a[1][1], xw[j][2 * s], xw[j][2 * s + 1]);
    }
  }
}

}  // namespace
}  // namespace ling::kernels::sg
