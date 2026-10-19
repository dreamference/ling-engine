// BACKLOG §19 item 30, stage 2: the FP8 projections' exponent Huffman decoded inside the streaming GEMM's tile loop,
// on the real matrices of the served checkpoint, against today's stream_gemm (the library's kernel) at 1, 12 and
// 24 rows, launch group by launch group in the order a step runs them.
//
// The coded form (stage 1's "lane" format; the warp-quarter streams of stage 1 cannot be found by a lane without
// decoding the lanes before it, so the decodable form carries a length per lane):
//   plane  1,024 bytes a (16-row tile, 128-value chunk) block at a fixed stride: each value's sign and mantissa as
//          a nibble (s << 3 | m), lane L's 64 values (the four 16-byte pieces it loads today, rows g and g + 8)
//          as two 16-byte words at 16 L and 512 + 16 L; within a 32-bit word value 8w + i (i < 4) is nibble 2i and
//          value 8w + 4 + i nibble 2i + 1, so one mask and one shift give the bytes in today's order
//   seg    per block, 4-byte aligned, found by a per-block word offset: 32 length bytes (lane L's stream in
//          bytes), then the 32 lanes' exponent streams, each byte-padded, LSB first, Huffman held to 10 bits
//   table  1,024 16-bit entries per matrix (low byte exponent << 3, high byte code length), in shared memory
// A lane finds its stream by a prefix scan of the 32 lengths over the warp and decodes its 64 exponents into the
// same 16 words of E4M3 bytes the raw kernel loads, so the MMA sees the same operands and the output is the raw
// kernel's bit for bit (checked here on every group and row count).
//
// Usage: fp8_coded_gemm SNAPSHOT_DIR [--reps N] [--layers N]
//   FP8C_TILES 1|2 (16-row tiles a block), FP8C_PF 1-4 (chunks in flight a warp; 3-4 with one tile), FP8C_ILP 0|1
//   (two tiles' streams decoded side by side), FP8C_M (one row count, for profiling); compile with -DFP8C_MINB=2
//   for two blocks an SM (128 registers). The pair-table form is fp8_coded_gemm_pair.cu; both are run by
//   fp8_coded_sweep.sh.
#include "core/kernels.cuh"
#include "core/launch.cuh"
#include "core/safetensors.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

namespace k = ling::kernels;

namespace {

constexpr int kWarps = 8, kChunk = 128, kTabBits = 10, kTab = 1 << kTabBits;
constexpr int kSegRegs = 9;                 // staged segment words per lane: 1,152 bytes a block at most
constexpr int kStageWords = kSegRegs * 32 + 8;

#define CK(x)                                                                                          \
  do {                                                                                                 \
    cudaError_t e_ = (x);                                                                              \
    if (e_ != cudaSuccess) {                                                                           \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
      std::exit(2);                                                                                    \
    }                                                                                                  \
  } while (0)

struct CodedProblems {
  const uint8_t* plane[4];
  const uint32_t* seg[4];
  const uint32_t* off[4];
  const uint16_t* tab[4];
  float gscale[4];
  float* y[4];
  int N[4];
  int block0[4];
  int count;
};

__device__ __forceinline__ uint32_t e4m3x2_to_h2(uint32_t pair) {
  uint32_t out;
  asm("{\n .reg .b16 lo, hi;\n mov.b32 {lo, hi}, %1;\n cvt.rn.f16x2.e4m3x2 %0, lo;\n}" : "=r"(out) : "r"(pair));
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

__device__ __forceinline__ uint32_t ld_nc(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

template <int TILES>
struct CodedRegs {
  uint4 pl[TILES][2];
  uint32_t sg[TILES][kSegRegs];
};

// A block's segment offset and length, loaded one chunk earlier than its words (so the dependent load of the words
// never waits for them).
template <int TILES>
struct SegOff {
  uint32_t o0[TILES], n[TILES];
};

template <int TILES>
__device__ __forceinline__ void load_off(SegOff<TILES>& o, const uint32_t* __restrict__ off, int row0, int N, int K,
                                         int chunk) {
  const int chunks = K / kChunk, last_tile = N / 16 - 1;
#pragma unroll
  for (int i = 0; i < TILES; ++i) {
    const size_t blk = static_cast<size_t>(min((row0 >> 4) + i, last_tile)) * chunks + chunk;
    o.o0[i] = __ldg(off + blk);
    o.n[i] = __ldg(off + blk + 1) - o.o0[i];
  }
}

template <int TILES>
__device__ __forceinline__ void load_coded(CodedRegs<TILES>& r, const uint8_t* __restrict__ plane,
                                           const uint32_t* __restrict__ seg, const SegOff<TILES>& so, int row0,
                                           int N, int K, int chunk, int lane) {
  const int chunks = K / kChunk, last_tile = N / 16 - 1;
#pragma unroll
  for (int i = 0; i < TILES; ++i) {
    const int tile = min((row0 >> 4) + i, last_tile);
    const size_t blk = static_cast<size_t>(tile) * chunks + chunk;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint8_t* p = plane + blk * 1024 + h * 512 + lane * 16;
      asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(r.pl[i][h].x), "=r"(r.pl[i][h].y), "=r"(r.pl[i][h].z), "=r"(r.pl[i][h].w)
                   : "l"(p));
    }
    const uint32_t n = so.n[i];
    const uint32_t* s = seg + so.o0[i];
#pragma unroll
    for (int j = 0; j < kSegRegs; ++j) {
      const uint32_t idx = j * 32 + lane;
      r.sg[i][j] = idx < n ? ld_nc(s + idx) : 0u;
    }
  }
}

// Lane `lane`'s 64 E4M3 bytes of T blocks (decoded side by side, T independent chains), as the 16 words the raw
// kernel holds in v[h][q] (piece h * 2 + q, word j at index (h * 2 + q) * 4 + j).
template <int T>
__device__ __forceinline__ void decode_blocks(const uint4 (*pl)[2], const uint32_t* __restrict__ st0, int stride,
                                              const uint16_t* __restrict__ tab, int lane, uint32_t (*w)[16]) {
  uint32_t wi[T], p[T], lo[T], hi[T];
#pragma unroll
  for (int b = 0; b < T; ++b) {
    const uint32_t* st = st0 + b * stride;
    const uint32_t lb = (st[lane >> 2] >> (8 * (lane & 3))) & 0xffu;
    uint32_t incl = lb;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += v;
    }
    const uint32_t start = 32 + incl - lb;
    wi[b] = start >> 2, p[b] = (start & 3) * 8;
    lo[b] = st[wi[b]], hi[b] = st[wi[b] + 1];
    wi[b] += 2;
#pragma unroll
    for (int i = 0; i < 16; ++i) w[b][i] = 0u;
  }
#pragma unroll
  for (int kk = 0; kk < 64; ++kk)
#pragma unroll
    for (int b = 0; b < T; ++b) {
      const uint32_t idx = __funnelshift_r(lo[b], hi[b], p[b]) & (kTab - 1);
      const uint32_t e = tab[idx];
      p[b] += e >> 8;
      w[b][kk >> 2] |= (e & 0x78u) << (8 * (kk & 3));
      if (p[b] >= 32) {
        lo[b] = hi[b];
        hi[b] = st0[b * stride + wi[b]++];
        p[b] -= 32;
      }
    }
#pragma unroll
  for (int b = 0; b < T; ++b)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint32_t P = q == 0 ? pl[b][h].x : q == 1 ? pl[b][h].y : q == 2 ? pl[b][h].z : pl[b][h].w;
        const uint32_t A = P & 0x0f0f0f0fu, B = (P >> 4) & 0x0f0f0f0fu;
        const int o = h * 8 + 2 * q;
        w[b][o] |= (A & 0x07070707u) | ((A & 0x08080808u) << 4);
        w[b][o + 1] |= (B & 0x07070707u) | ((B & 0x08080808u) << 4);
      }
}

template <int NT, int TILES, int PF, bool ILP>
#ifndef FP8C_MINB
#define FP8C_MINB 1
#endif
__global__ void __launch_bounds__(kWarps * 32, FP8C_MINB)
    coded_gemm_kernel(const __half* __restrict__ x, const float* __restrict__ xinv, int M, CodedProblems pr, int K) {
  int pi = 0;
  while (pi + 1 < pr.count && static_cast<int>(blockIdx.x) >= pr.block0[pi + 1]) ++pi;
  const uint8_t* __restrict__ plane = pr.plane[pi];
  const uint32_t* __restrict__ seg = pr.seg[pi];
  const uint32_t* __restrict__ off = pr.off[pi];
  const float gscale = pr.gscale[pi];
  float* __restrict__ y = pr.y[pi];
  const int N = pr.N[pi];
  __shared__ float red[kWarps][TILES * 16][NT * 8 + 1];
  __shared__ uint16_t tab[kTab];
  __shared__ uint32_t stage[kWarps][ILP ? TILES : 1][kStageWords];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int row0 = (blockIdx.x - pr.block0[pi]) * TILES * 16;
  const int chunks = K / kChunk;

  float acc[TILES][NT][4];
#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  CodedRegs<TILES> buf[PF + 1];
#pragma unroll
  for (int p = 0; p < PF; ++p)
    if (warp + p * kWarps < chunks) {
      SegOff<TILES> o;
      load_off<TILES>(o, off, row0, N, K, warp + p * kWarps);
      load_coded<TILES>(buf[p], plane, seg, o, row0, N, K, warp + p * kWarps, lane);
    }
  SegOff<TILES> nxt;
  if (warp + PF * kWarps < chunks) load_off<TILES>(nxt, off, row0, N, K, warp + PF * kWarps);
  for (int i = threadIdx.x; i < kTab; i += blockDim.x) tab[i] = pr.tab[pi][i];
  __syncthreads();
  k::pdl_begin();
  uint32_t* st = stage[warp][0];
  for (int c = warp; c < chunks; c += kWarps) {
    if (c + PF * kWarps < chunks) load_coded<TILES>(buf[PF], plane, seg, nxt, row0, N, K, c + PF * kWarps, lane);
    if (c + (PF + 1) * kWarps < chunks) load_off<TILES>(nxt, off, row0, N, K, c + (PF + 1) * kWarps);
    const CodedRegs<TILES>& cur = buf[0];
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
    auto mma_tile = [&](int i, const uint32_t (&wv)[16]) {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        uint32_t a[2][2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t word = wv[(h * 2 + (s >> 2)) * 4 + (s & 3)];
          a[h][0] = e4m3x2_to_h2(word & 0xffffu);
          a[h][1] = e4m3x2_to_h2(word >> 16);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
          mma16816(acc[i][j], a[0][0], a[1][0], a[0][1], a[1][1], xw[j][2 * s], xw[j][2 * s + 1]);
      }
    };
    if constexpr (ILP) {  // every tile's segment staged, then the tiles' streams decoded side by side
      __syncwarp();
#pragma unroll
      for (int i = 0; i < TILES; ++i)
#pragma unroll
        for (int j = 0; j < kSegRegs; ++j) st[i * kStageWords + j * 32 + lane] = cur.sg[i][j];
      __syncwarp();
      uint32_t wv[TILES][16];
      decode_blocks<TILES>(cur.pl, st, kStageWords, tab, lane, wv);
#pragma unroll
      for (int i = 0; i < TILES; ++i) mma_tile(i, wv[i]);
    } else {
#pragma unroll
      for (int i = 0; i < TILES; ++i) {
        __syncwarp();
#pragma unroll
        for (int j = 0; j < kSegRegs; ++j) st[j * 32 + lane] = cur.sg[i][j];
        __syncwarp();
        uint32_t wv[1][16];
        decode_blocks<1>(&cur.pl[i], st, kStageWords, tab, lane, wv);
        mma_tile(i, wv[0]);
      }
    }
#pragma unroll
    for (int p = 0; p < PF; ++p) buf[p] = buf[p + 1];
  }

#pragma unroll
  for (int i = 0; i < TILES; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      red[warp][16 * i + g][8 * j + 2 * t] = acc[i][j][0];
      red[warp][16 * i + g][8 * j + 2 * t + 1] = acc[i][j][1];
      red[warp][16 * i + g + 8][8 * j + 2 * t] = acc[i][j][2];
      red[warp][16 * i + g + 8][8 * j + 2 * t + 1] = acc[i][j][3];
    }
  __syncthreads();
  for (int e = threadIdx.x; e < TILES * 16 * NT * 8; e += blockDim.x) {
    const int r = e / (NT * 8), n = e % (NT * 8);
    if (n >= M || row0 + r >= N) continue;
    float sum = 0.f;
#pragma unroll
    for (int wi = 0; wi < kWarps; ++wi) sum += red[wi][r][n];
    y[static_cast<size_t>(n) * N + row0 + r] = sum * gscale * xinv[n];
  }
}

// ---- host side ----

struct CodedMatrix {
  int N, K;
  uint8_t* plane;
  uint32_t* seg;
  uint32_t* off;
  uint16_t* tab;
  size_t bytes;  // plane + seg + offsets + table: what a step reads
  size_t max_seg;
};

struct Matrix {
  std::string name;
  int N, K;
  uint8_t* tiled;  // today's blob
  CodedMatrix coded;
};

// Length-limited Huffman (package-merge), 16 symbols.
std::vector<int> limited_lengths(const std::vector<uint64_t>& freq, int limit) {
  std::vector<int> used;
  for (int i = 0; i < 16; ++i)
    if (freq[i]) used.push_back(i);
  std::vector<int> len(16, 0);
  if (used.size() == 1) {
    len[used[0]] = 1;
    return len;
  }
  struct Item {
    uint64_t w;
    std::vector<int> syms;
  };
  std::vector<Item> leaves;
  for (int i : used) leaves.push_back({freq[i], {i}});
  std::stable_sort(leaves.begin(), leaves.end(), [](const Item& a, const Item& b) { return a.w < b.w; });
  std::vector<Item> level = leaves;
  for (int it = 0; it < limit - 1; ++it) {
    std::vector<Item> pk;
    for (size_t j = 0; j + 1 < level.size(); j += 2) {
      Item m{level[j].w + level[j + 1].w, level[j].syms};
      m.syms.insert(m.syms.end(), level[j + 1].syms.begin(), level[j + 1].syms.end());
      pk.push_back(std::move(m));
    }
    level = leaves;
    level.insert(level.end(), pk.begin(), pk.end());
    std::stable_sort(level.begin(), level.end(), [](const Item& a, const Item& b) { return a.w < b.w; });
  }
  for (size_t j = 0; j < 2 * used.size() - 2; ++j)
    for (int s : level[j].syms) ++len[s];
  return len;
}

uint32_t reverse_bits(uint32_t v, int n) {
  uint32_t r = 0;
  for (int i = 0; i < n; ++i) r |= ((v >> i) & 1u) << (n - 1 - i);
  return r;
}

CodedMatrix encode(const std::vector<uint8_t>& tiled, int N, int K) {
  const int chunks = K / kChunk;
  const size_t blocks = static_cast<size_t>(N / 16) * chunks;
  std::vector<uint64_t> freq(16, 0);
  for (uint8_t b : tiled) ++freq[(b >> 3) & 15];
  const std::vector<int> len = limited_lengths(freq, kTabBits);
  // Canonical codes (MSB-first), sent LSB-first: the bit-reversed code is what the table is indexed by.
  std::vector<uint32_t> code(16, 0);
  {
    uint32_t c = 0;
    for (int l = 1; l <= kTabBits; ++l) {
      for (int s = 0; s < 16; ++s)
        if (len[s] == l) code[s] = reverse_bits(c++, l);
      c <<= 1;
    }
  }
  std::vector<uint16_t> tab(kTab, 0);
  for (int s = 0; s < 16; ++s)
    if (len[s])
      for (uint32_t hi = 0; hi < (1u << (kTabBits - len[s])); ++hi)
        tab[(hi << len[s]) | code[s]] = static_cast<uint16_t>((len[s] << 8) | (s << 3));
  std::vector<uint8_t> plane(blocks * 1024, 0);
  std::vector<std::vector<uint32_t>> segs(blocks);
  size_t max_seg = 0;
#pragma omp parallel for schedule(dynamic, 256) reduction(max : max_seg)
  for (size_t blk = 0; blk < blocks; ++blk) {
    const uint8_t* b = tiled.data() + blk * 2048;
    std::vector<uint8_t> out(32, 0);
    for (int L = 0; L < 32; ++L) {
      const int g = L >> 2, t = L & 3;
      uint64_t acc = 0;
      int nb = 0;
      size_t first = out.size();
      for (int kk = 0; kk < 64; ++kk) {
        const int hq = kk >> 4, bi = kk & 15;
        const uint8_t v = b[((hq * 8 + g) * 4 + t) * 16 + bi];
        const int e = (v >> 3) & 15;
        const uint8_t nib = static_cast<uint8_t>(((v >> 7) << 3) | (v & 7));
        const int half = hq >> 1, local = kk - 32 * half, wd = local >> 3, i = local & 7;
        const int pos = i < 4 ? 2 * i : 2 * (i - 4) + 1;
        plane[blk * 1024 + half * 512 + L * 16 + wd * 4 + (pos >> 1)] |= static_cast<uint8_t>(nib << (4 * (pos & 1)));
        acc |= static_cast<uint64_t>(code[e]) << nb;
        nb += len[e];
        while (nb >= 8) {
          out.push_back(static_cast<uint8_t>(acc & 0xff));
          acc >>= 8;
          nb -= 8;
        }
      }
      if (nb > 0) out.push_back(static_cast<uint8_t>(acc & 0xff));
      out[L] = static_cast<uint8_t>(out.size() - first);
    }
    while (out.size() % 4) out.push_back(0);
    max_seg = std::max(max_seg, out.size());
    segs[blk].resize(out.size() / 4);
    std::memcpy(segs[blk].data(), out.data(), out.size());
  }
  if (max_seg > kSegRegs * 128) {
    std::fprintf(stderr, "a block's segment is %zu bytes, over the %d staged\n", max_seg, kSegRegs * 128);
    std::exit(3);
  }
  std::vector<uint32_t> off(blocks + 1, 0);
  for (size_t blk = 0; blk < blocks; ++blk) off[blk + 1] = off[blk] + static_cast<uint32_t>(segs[blk].size());
  std::vector<uint32_t> seg(off[blocks] + 16, 0);  // slack: a lane's look-ahead word past the end
  for (size_t blk = 0; blk < blocks; ++blk) std::copy(segs[blk].begin(), segs[blk].end(), seg.begin() + off[blk]);
  CodedMatrix m{N, K, nullptr, nullptr, nullptr, nullptr, 0, max_seg};
  CK(cudaMalloc(&m.plane, plane.size()));
  CK(cudaMalloc(&m.seg, seg.size() * 4));
  CK(cudaMalloc(&m.off, off.size() * 4));
  CK(cudaMalloc(&m.tab, tab.size() * 2));
  CK(cudaMemcpy(m.plane, plane.data(), plane.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(m.seg, seg.data(), seg.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(m.off, off.data(), off.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(m.tab, tab.data(), tab.size() * 2, cudaMemcpyHostToDevice));
  m.bytes = plane.size() + size_t(off[blocks]) * 4 + off.size() * 4 + tab.size() * 2;
  return m;
}

int g_tiles = 2, g_pf = 2, g_ilp = 0;

template <int NT, int TILES, int PF, bool ILP>
void launch_coded4(const __half* x, const float* xinv, int M, const Matrix* const* ms, float* const* ys, int n, int K,
                   cudaStream_t s) {
  CodedProblems pr{};
  int blocks = 0;
  for (int i = 0; i < n; ++i) {
    const CodedMatrix& c = ms[i]->coded;
    pr.plane[i] = c.plane, pr.seg[i] = c.seg, pr.off[i] = c.off, pr.tab[i] = c.tab;
    pr.gscale[i] = 1.f, pr.y[i] = ys[i], pr.N[i] = c.N, pr.block0[i] = blocks;
    blocks += (c.N + TILES * 16 - 1) / (TILES * 16);
  }
  pr.count = n;
  k::launch_kernel("coded_gemm", coded_gemm_kernel<NT, TILES, PF, ILP>, dim3(blocks), dim3(kWarps * 32), 0, s, x, xinv, M, pr,
                   K);
}

template <int NT, int TILES, int PF>
void launch_coded3(const __half* x, const float* xinv, int M, const Matrix* const* ms, float* const* ys, int n, int K,
                   cudaStream_t s) {
  if (g_ilp && TILES > 1) launch_coded4<NT, TILES, PF, true>(x, xinv, M, ms, ys, n, K, s);
  else launch_coded4<NT, TILES, PF, false>(x, xinv, M, ms, ys, n, K, s);
}

template <int NT>
void launch_coded(const __half* x, const float* xinv, int M, const Matrix* const* ms, float* const* ys, int n, int K,
                  cudaStream_t s) {
  if (g_tiles == 1) {
    if (g_pf == 1) launch_coded3<NT, 1, 1>(x, xinv, M, ms, ys, n, K, s);
    else if (g_pf == 3) launch_coded3<NT, 1, 3>(x, xinv, M, ms, ys, n, K, s);
    else if (g_pf == 4) launch_coded3<NT, 1, 4>(x, xinv, M, ms, ys, n, K, s);
    else launch_coded3<NT, 1, 2>(x, xinv, M, ms, ys, n, K, s);
  } else {
    if (g_pf == 1) launch_coded3<NT, 2, 1>(x, xinv, M, ms, ys, n, K, s);
    else launch_coded3<NT, 2, 2>(x, xinv, M, ms, ys, n, K, s);
  }
}

void coded_multi(const __half* x, const float* xinv, int M, const Matrix* const* ms, float* const* ys, int n, int K,
                 cudaStream_t s) {
  if (M <= 8) launch_coded<1>(x, xinv, M, ms, ys, n, K, s);
  else if (M <= 16) launch_coded<2>(x, xinv, M, ms, ys, n, K, s);
  else launch_coded<4>(x, xinv, M, ms, ys, n, K, s);
}

void raw_multi(const __half* x, const float* xinv, int M, const Matrix* const* ms, float* const* ys, int n, int K,
               cudaStream_t s) {
  k::StreamTarget t[4];
  for (int i = 0; i < n; ++i) t[i] = {ms[i]->tiled, nullptr, 1.f, ys[i], ms[i]->N};
  k::stream_gemm_multi(false, x, xinv, M, t, n, K, s);
}

struct Group {
  std::string kind;
  std::vector<Matrix*> ms;
  int K;
};

__global__ void fill_x_kernel(float* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
    p[i] = (static_cast<float>(h & 0xffffu) / 65536.f - 0.5f) * 4.f;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: fp8_coded_gemm SNAPSHOT_DIR [--reps N] [--layers N]\n");
    return 1;
  }
  int reps = 5, layers = 64;
  for (int i = 2; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--reps")) reps = std::atoi(argv[i + 1]);
    if (!std::strcmp(argv[i], "--layers")) layers = std::atoi(argv[i + 1]);
  }
  if (const char* e = std::getenv("FP8C_TILES")) g_tiles = std::atoi(e);
  if (const char* e = std::getenv("FP8C_PF")) g_pf = std::atoi(e);
  if (const char* e = std::getenv("FP8C_ILP")) g_ilp = std::atoi(e);
  std::printf("coded kernel: tiles %d, chunks in flight %d, tiles decoded side by side %d\n", g_tiles, g_pf, g_ilp);
  k::set_pdl(true);
  ling::Checkpoint ck(argv[1]);
  const std::string pre = "model.language_model.layers.";
  std::vector<Matrix*> all;
  std::vector<Group> groups;
  size_t raw_bytes = 0, coded_bytes = 0, max_seg = 0;
  auto load = [&](const std::string& name) {
    const ling::TensorView& tv = ck.get(name + ".weight");
    Matrix* m = new Matrix{name, static_cast<int>(tv.shape[0]), static_cast<int>(tv.shape[1]), nullptr, {}};
    uint8_t* rowmajor;
    CK(cudaMalloc(&rowmajor, tv.bytes));
    CK(cudaMemcpy(rowmajor, tv.data, tv.bytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&m->tiled, tv.bytes));
    k::retile_fp8(rowmajor, m->tiled, m->N, m->K, nullptr);
    CK(cudaDeviceSynchronize());
    CK(cudaFree(rowmajor));
    std::vector<uint8_t> host(tv.bytes);
    CK(cudaMemcpy(host.data(), m->tiled, tv.bytes, cudaMemcpyDeviceToHost));
    m->coded = encode(host, m->N, m->K);
    raw_bytes += tv.bytes;
    coded_bytes += m->coded.bytes;
    max_seg = std::max(max_seg, m->coded.max_seg);
    all.push_back(m);
    return m;
  };
  const auto t0 = std::chrono::steady_clock::now();
  for (int l = 0; l < layers; ++l) {
    const std::string p = pre + std::to_string(l) + ".";
    if (ck.has(p + "self_attn.q_proj.weight")) {
      groups.push_back({"attn q|k|v", {load(p + "self_attn.q_proj"), load(p + "self_attn.k_proj"), load(p + "self_attn.v_proj")}, 0});
      groups.push_back({"attn o", {load(p + "self_attn.o_proj")}, 0});
    } else {
      groups.push_back({"gdn in_qkv|z", {load(p + "linear_attn.in_proj_qkv"), load(p + "linear_attn.in_proj_z")}, 0});
      groups.push_back({"gdn out", {load(p + "linear_attn.out_proj")}, 0});
    }
  }
  for (Group& g : groups) g.K = g.ms[0]->K;
  std::printf("%zu FP8 matrices, %.3f GB raw, %.3f GB coded (%.4f), largest block segment %zu bytes, encoded in %.0f s\n",
              all.size(), raw_bytes / 1e9, coded_bytes / 1e9, double(coded_bytes) / raw_bytes, max_seg,
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

  const int Kmax = 6144;
  float* x;
  __half* xh;
  float* xinv;
  CK(cudaMalloc(&x, size_t(32) * Kmax * 4));
  CK(cudaMalloc(&xh, size_t(32) * Kmax * 2));
  CK(cudaMalloc(&xinv, 32 * 4));
  fill_x_kernel<<<64, 256>>>(x, size_t(32) * Kmax, 11);
  size_t ymax = 0;
  for (Matrix* m : all) ymax = std::max(ymax, size_t(m->N));
  float *ya[4], *yb[4];
  for (int i = 0; i < 4; ++i) {
    CK(cudaMalloc(&ya[i], size_t(32) * ymax * 4));
    CK(cudaMalloc(&yb[i], size_t(32) * ymax * 4));
  }
  std::vector<float> ha(size_t(32) * ymax), hb(size_t(32) * ymax);
  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t ea, eb;
  CK(cudaEventCreate(&ea));
  CK(cudaEventCreate(&eb));

  std::vector<int> Ms = {1, 12, 24};
  if (const char* e = std::getenv("FP8C_M")) Ms = {std::atoi(e)};  // one row count (profiling)
  bool all_exact = true;
  for (int M : Ms) {
    // Exactness: every group, coded against raw, bit for bit.
    int bad = 0;
    for (const Group& g : groups) {
      k::to_half_rows(x, M, g.K, xh, xinv, s);
      raw_multi(xh, xinv, M, g.ms.data(), ya, static_cast<int>(g.ms.size()), g.K, s);
      coded_multi(xh, xinv, M, g.ms.data(), yb, static_cast<int>(g.ms.size()), g.K, s);
      CK(cudaStreamSynchronize(s));
      for (size_t i = 0; i < g.ms.size(); ++i) {
        const size_t n = size_t(M) * g.ms[i]->N;
        CK(cudaMemcpy(ha.data(), ya[i], n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), yb[i], n * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(ha.data(), hb.data(), n * 4) != 0) ++bad;
      }
    }
    all_exact &= bad == 0;
    // Timing: the step's FP8 launches in order, the input converted once per K as the step does (once per group).
    std::vector<double> per_group_raw(4, 0), per_group_coded(4, 0);
    auto run = [&](bool coded) {
      for (const Group& g : groups) {
        if (coded) coded_multi(xh, xinv, M, g.ms.data(), yb, static_cast<int>(g.ms.size()), g.K, s);
        else raw_multi(xh, xinv, M, g.ms.data(), ya, static_cast<int>(g.ms.size()), g.K, s);
      }
    };
    k::to_half_rows(x, M, Kmax, xh, xinv, s);  // K = 5,120 groups read a prefix of each row: timing only
    float best[2] = {1e30f, 1e30f};
    std::vector<float> all_t[2];
    for (int r = 0; r < reps + 1; ++r)
      for (int coded = 0; coded < 2; ++coded) {
        CK(cudaEventRecord(ea, s));
        run(coded);
        CK(cudaEventRecord(eb, s));
        CK(cudaEventSynchronize(eb));
        float ms;
        CK(cudaEventElapsedTime(&ms, ea, eb));
        if (r > 0) {
          best[coded] = std::min(best[coded], ms);
          all_t[coded].push_back(ms);
        }
      }
    for (auto& v : all_t) std::sort(v.begin(), v.end());
    const float med_raw = all_t[0][all_t[0].size() / 2], med_cod = all_t[1][all_t[1].size() / 2];
    std::printf("M=%-2d raw %7.3f ms (%5.1f GB/s, median %7.3f)  coded %7.3f ms (%5.1f GB/s of coded bytes, median %7.3f)"
                "  net %+6.3f ms (median %+6.3f)  exact %s\n",
                M, best[0], raw_bytes / (best[0] * 1e6), med_raw, best[1], coded_bytes / (best[1] * 1e6), med_cod,
                best[0] - best[1], med_raw - med_cod, bad == 0 ? "yes" : "NO");
    // Per launch kind.
    const char* kinds[] = {"attn q|k|v", "attn o", "gdn in_qkv|z", "gdn out"};
    for (const char* kind : kinds) {
      double tr = 0, tc = 0, br = 0, bc = 0;
      for (int coded = 0; coded < 2; ++coded) {
        float b = 1e30f;
        for (int r = 0; r < 3; ++r) {
          CK(cudaEventRecord(ea, s));
          for (const Group& g : groups)
            if (g.kind == kind) {
              if (coded) coded_multi(xh, xinv, M, g.ms.data(), yb, static_cast<int>(g.ms.size()), g.K, s);
              else raw_multi(xh, xinv, M, g.ms.data(), ya, static_cast<int>(g.ms.size()), g.K, s);
            }
          CK(cudaEventRecord(eb, s));
          CK(cudaEventSynchronize(eb));
          float ms;
          CK(cudaEventElapsedTime(&ms, ea, eb));
          b = std::min(b, ms);
        }
        (coded ? tc : tr) = b;
      }
      for (const Group& g : groups)
        if (g.kind == kind)
          for (Matrix* m : g.ms) br += double(m->N) * m->K, bc += m->coded.bytes;
      std::printf("    %-13s raw %6.3f ms %5.1f GB/s   coded %6.3f ms %5.1f GB/s (%.4f of the bytes)   net %+6.3f ms\n", kind,
                  tr, br / (tr * 1e6), tc, bc / (tc * 1e6), bc / br, tr - tc);
    }
  }
  std::printf("exact on every group and row count: %s\n", all_exact ? "yes" : "NO");
  return all_exact ? 0 : 1;
}
