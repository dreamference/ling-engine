// Milestone 1 probe: the verify GEMM on gfx1151. Weights stream from memory in the served formats
// (NVFP4: E2M1 codes [N][K/2] low nibble first + E4M3 scales [N][K/16]; FP8: E4M3 [N][K]), are decoded
// to FP16 in registers and multiplied on WMMA 16x16x16 against M <= 16*MT activation rows staged in LDS.
// One wave owns one 16-column tile of N and walks all of K. Header-free (no HIP headers on the box).
// The fastest configuration on 2026-10-26 (specs/DREAMFERENCE_LING_ENGINE_AMD.md §6.5): -DKMAJOR -DNT -DKC=256,
// with -DBWAVES=32 -DXROWS=16 and 1024-thread blocks for FP8. The fp4_/fp8_ kernels are the first, row-major
// attempt; the fp4w_/fp8w_ ones give each wave two tiles; NODEC, NOWMMA, NOX and NOBAR strip parts for attribution.
#define G extern "C" __attribute__((global))
#define D static __attribute__((device)) inline
typedef unsigned int u32;
typedef unsigned long long u64;
typedef _Float16 h;
typedef h v16h __attribute__((ext_vector_type(16)));
typedef h v2h __attribute__((ext_vector_type(2)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef u32 v4u __attribute__((ext_vector_type(4)));
typedef u32 v2u __attribute__((ext_vector_type(2)));
#define LDS __attribute__((address_space(3)))
#define TX __builtin_amdgcn_workitem_id_x()
#define BX __builtin_amdgcn_workgroup_id_x()
#ifndef BWAVES
#define BWAVES 8
#endif
#ifndef KC
#define KC 512
#endif
#ifndef XROWS
#define XROWS 32
#endif
#ifndef UNRT
#define UNRT 2
#endif
#ifdef NT
#define WLD(p) __builtin_nontemporal_load(p)
#else
#define WLD(p) (*(p))
#endif
#ifndef XPAD
#define XPAD 16
#endif
#define XST (KC + XPAD)
__attribute__((shared)) h XS[XROWS * XST];
D void bar() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup"); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup"); }

D u32 as_u(v2h x) { return __builtin_bit_cast(u32, x); }
D v2h as_h(u32 x) { return __builtin_bit_cast(v2h, x); }

// 8 E2M1 codes (one u32, low nibble = first value) -> 8 halves times the block scale (pre-multiplied by 2^7).
D void dec_fp4(u32 c, v2h sc, v2h* o) {
  u32 lo = c & 0x0F0F0F0Fu, hi = (c >> 4) & 0x0F0F0F0Fu;
  const v2h k14 = {(h)16384.0f, (h)16384.0f};
#pragma unroll
  for (int j = 0; j < 4; j++) {
    u32 d = __builtin_amdgcn_perm(hi, lo, 0x0C040C00u + j * 0x00010001u);  // [lo_j, 0, hi_j, 0]
    u32 r = ((d & 0x00070007u) << 9) | ((d & 0x00080008u) << 12);
    o[j] = as_h(r) * k14 * sc;
  }
}
// 4 E4M3 bytes -> 4 halves.
D void dec_fp8(u32 c, v2h* o) {
  const v2h k8 = {(h)256.0f, (h)256.0f};
#pragma unroll
  for (int j = 0; j < 2; j++) {
    u32 d = __builtin_amdgcn_perm(0u, c, 0x0C010C00u + j * 0x00020002u);  // [b_2j, 0, b_2j+1, 0]
    u32 r = ((d & 0x007F007Fu) << 7) | ((d & 0x00800080u) << 8);
    o[j] = as_h(r) * k8;
  }
}
D v2h e4m3_to_h2(u32 b) {  // one E4M3 byte -> half, broadcast to both halves, times 2^7 folded in
  u32 r = ((b & 0x7Fu) << 7) | ((b & 0x80u) << 8);
  h v = __builtin_bit_cast(h, (unsigned short)r) * (h)256.0f * (h)128.0f;
  v2h o = {v, v};
  return o;
}

template <int MT, bool FP4>
D void body(const unsigned char* __restrict w, const unsigned char* __restrict sc, const h* __restrict x,
            float* __restrict y, u32 K, u32 N, u32 M, float gscale) {
  h* xs = XS;  // [MT*16][KC] in LDS
  u32 lane = TX & 31, wave = TX >> 5;
  u32 n0 = (BX * 8 + wave) * 16;
  u32 row = n0 + (lane & 15);
  v8f acc[MT];
#pragma unroll
  for (int t = 0; t < MT; t++) acc[t] = (v8f){};
  for (u32 k0 = 0; k0 < K; k0 += KC) {
    bar();
    for (u32 i = TX; i < MT * 16 * KC / 8; i += 256) {  // 8 halves per thread per iteration
      u32 r = i / (KC / 8), c = (i % (KC / 8)) * 8;
      v4u v = {0, 0, 0, 0};
      if (r < M) v = *(const v4u*)(x + (u64)r * K + k0 + c);
      *(v4u*)(xs + r * KC + c) = v;
    }
    bar();
#pragma unroll 2
    for (u32 kk = 0; kk < KC; kk += 64) {  // four 16-wide K chunks per step
      u32 k = k0 + kk;
      v16h b[4];
      if (FP4) {
        const v4u* wp = (const v4u*)(w + (u64)row * (K / 2) + k / 2);
        v4u c0 = __builtin_nontemporal_load(wp), c1 = __builtin_nontemporal_load(wp + 1);
        u32 s4 = __builtin_nontemporal_load((const u32*)(sc + (u64)row * (K / 16) + k / 16));
        u32 cw[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
        for (int q = 0; q < 4; q++) {
          v2h s = e4m3_to_h2((s4 >> (8 * q)) & 0xFF);
          v2h o[8];
          dec_fp4(cw[2 * q], s, o);
          dec_fp4(cw[2 * q + 1], s, o + 4);
#pragma unroll
          for (int e = 0; e < 8; e++) { b[q][2 * e] = o[e].x; b[q][2 * e + 1] = o[e].y; }
        }
      } else {
        const v4u* wp = (const v4u*)(w + (u64)row * K + k);
        v4u c[4];
#pragma unroll
        for (int q = 0; q < 4; q++) c[q] = __builtin_nontemporal_load(wp + q);
#pragma unroll
        for (int q = 0; q < 4; q++) {
          u32 cw[4] = {c[q].x, c[q].y, c[q].z, c[q].w};
          v2h o[8];
#pragma unroll
          for (int e = 0; e < 4; e++) dec_fp8(cw[e], o + 2 * e);
#pragma unroll
          for (int e = 0; e < 8; e++) { b[q][2 * e] = o[e].x; b[q][2 * e + 1] = o[e].y; }
        }
      }
#pragma unroll
      for (int q = 0; q < 4; q++) {
#pragma unroll
        for (int t = 0; t < MT; t++) {
          v16h a = *(v16h*)(xs + (t * 16 + (lane & 15)) * KC + kk + q * 16);
          acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[q], acc[t]);
        }
      }
    }
  }
  float post = FP4 ? gscale * (1.0f / 128.0f) : gscale;
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int r = 0; r < 8; r++) {
      u32 m = t * 16 + 2 * r + (lane >> 4);
      if (m < M) y[(u64)m * N + n0 + (lane & 15)] = acc[t][r] * post;
    }
}

G void __attribute__((amdgpu_flat_work_group_size(256, 256))) fp4_m16(const unsigned char* w, const unsigned char* sc, const h* x, float* y, u32 K, u32 N, u32 M, float g) { body<1, true>(w, sc, x, y, K, N, M, g); }
#if XROWS >= 32
G void __attribute__((amdgpu_flat_work_group_size(256, 256))) fp4_m32(const unsigned char* w, const unsigned char* sc, const h* x, float* y, u32 K, u32 N, u32 M, float g) { body<2, true>(w, sc, x, y, K, N, M, g); }
#endif
G void __attribute__((amdgpu_flat_work_group_size(256, 256))) fp8_m16(const unsigned char* w, const unsigned char* sc, const h* x, float* y, u32 K, u32 N, u32 M, float g) { body<1, false>(w, sc, x, y, K, N, M, g); }
#if XROWS >= 32
G void __attribute__((amdgpu_flat_work_group_size(256, 256))) fp8_m32(const unsigned char* w, const unsigned char* sc, const h* x, float* y, u32 K, u32 N, u32 M, float g) { body<2, false>(w, sc, x, y, K, N, M, g); }
#endif


// v2: weights retiled at load (as ling-engine already does for the GB10, ARCHITECTURE §8 stage 4) so that one
// wave's load is contiguous. Tile = 16 rows x 64 K. NVFP4 codes: 512 B per tile, row r's 32 B at r*32; lanes
// 0-15 load the first 16 B of rows 0-15 (K 0..31), lanes 16-31 the second 16 B (K 32..63). Scales: 64 B per
// tile, [row][4]. FP8: 1 KB per tile, row r's 64 B at r*64; lanes 0-15 take K 0..31, lanes 16-31 K 32..63.
// Each half decodes only its own K and the halves swap decoded fragments with v_permlanex16.
D u32 xswap(u32 v) { return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xFEDCBA98u, false, false); }
typedef u32 v8u __attribute__((ext_vector_type(8)));
D v16h frag(v8u d) { return __builtin_bit_cast(v16h, d); }

template <int MT, bool FP4, int NTW, int UNR>
D void tbody(const unsigned char* __restrict w, const unsigned char* __restrict sc, const h* __restrict x,
             float* __restrict y, u32 K, u32 N, u32 M, float gscale) {
  h* xs = XS;
  u32 lane = TX & 31, wave = TX >> 5, half = lane >> 4, r16 = lane & 15;
  u32 ntb = (BX * BWAVES + wave) * NTW;
  u32 kg_per_row = K / 64;
  v8f acc[NTW][MT];
#pragma unroll
  for (int j = 0; j < NTW; j++)
#pragma unroll
  for (int t = 0; t < MT; t++) acc[j][t] = (v8f){};
  for (u32 k0 = 0; k0 < K; k0 += KC) {
#ifndef NOBAR
    bar();
#endif
#ifndef NOX
    for (u32 i = TX; i < MT * 16 * KC / 8; i += BWAVES * 32) {
      u32 r = i / (KC / 8), c = (i % (KC / 8)) * 8;
      v4u v = {0, 0, 0, 0};
      if (r < M) v = *(const v4u*)(x + (u64)r * K + k0 + c);
      *(v4u*)(xs + r * XST + c) = v;
    }
#endif
#ifndef NOBAR
    bar();
#endif
#pragma unroll UNR
    for (u32 kk = 0; kk < KC; kk += 64) {
      v16h b[NTW][4];
#pragma unroll
      for (int j = 0; j < NTW; j++) {
      #ifdef KMAJOR
      u64 tile = (u64)((k0 + kk) / 64) * (N / 16) + (ntb + j);
#else
      u64 tile = (u64)(ntb + j) * kg_per_row + (k0 + kk) / 64;
#endif
      v8u own0, own1;
      if (FP4) {
        v4u c = WLD((const v4u*)(w + tile * 512 + r16 * 32 + half * 16));
        u32 s2 = *(const unsigned short*)(sc + tile * 64 + r16 * 4 + half * 2);
        v2h sa = e4m3_to_h2(s2 & 0xFF), sb = e4m3_to_h2(s2 >> 8);
        v2h o[8];
#ifdef NODEC
        own0 = (v8u){c.x, c.y, c.z, c.w, c.x ^ s2, c.y, c.z, c.w}; own1 = (v8u){c.w, c.z, c.y, c.x, c.w, c.z, c.y, c.x ^ s2};
        (void)sa; (void)sb; (void)o;
#else
        dec_fp4(c.x, sa, o); dec_fp4(c.y, sa, o + 4);
#pragma unroll
        for (int e = 0; e < 8; e++) own0[e] = as_u(o[e]);
        dec_fp4(c.z, sb, o); dec_fp4(c.w, sb, o + 4);
#pragma unroll
        for (int e = 0; e < 8; e++) own1[e] = as_u(o[e]);
#endif
      } else {
        const unsigned char* base = w + tile * 1024 + r16 * 64 + half * 32;
        v4u c0 = WLD((const v4u*)base), c1 = WLD((const v4u*)(base + 16));
        v2h o[8];
        dec_fp8(c0.x, o); dec_fp8(c0.y, o + 2); dec_fp8(c0.z, o + 4); dec_fp8(c0.w, o + 6);
#pragma unroll
        for (int e = 0; e < 8; e++) own0[e] = as_u(o[e]);
        dec_fp8(c1.x, o); dec_fp8(c1.y, o + 2); dec_fp8(c1.z, o + 4); dec_fp8(c1.w, o + 6);
#pragma unroll
        for (int e = 0; e < 8; e++) own1[e] = as_u(o[e]);
      }
      v8u oth0, oth1;
#pragma unroll
      for (int e = 0; e < 8; e++) { oth0[e] = xswap(own0[e]); oth1[e] = xswap(own1[e]); }
      b[j][0] = frag(half ? oth0 : own0);
      b[j][1] = frag(half ? oth1 : own1);
      b[j][2] = frag(half ? own0 : oth0);
      b[j][3] = frag(half ? own1 : oth1);
      }
#pragma unroll
      for (int q = 0; q < 4; q++)
#pragma unroll
        for (int t = 0; t < MT; t++) {
          v16h a = *(v16h*)(xs + (t * 16 + r16) * XST + kk + q * 16);
#pragma unroll
          for (int j = 0; j < NTW; j++) {
#ifdef NOWMMA
            v8u bb = __builtin_bit_cast(v8u, b[j][q]) ^ __builtin_bit_cast(v8u, a);
            acc[j][t] += __builtin_bit_cast(v8f, bb);
#else
            acc[j][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b[j][q], acc[j][t]);
#endif
          }
        }
    }
  }
  float post = FP4 ? gscale * (1.0f / 128.0f) : gscale;
#pragma unroll
  for (int j = 0; j < NTW; j++)
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int r = 0; r < 8; r++) {
      u32 m = t * 16 + 2 * r + half;
      if (m < M) y[(u64)m * N + (ntb + j) * 16 + r16] = acc[j][t][r] * post;
    }
}
#define KAT __attribute__((amdgpu_flat_work_group_size(BWAVES * 32, BWAVES * 32)))
#define KARGS const unsigned char* w, const unsigned char* sc, const h* x, float* y, u32 K, u32 N, u32 M, float g
G void KAT fp4t_m16(KARGS) { tbody<1, true, 1, UNRT>(w, sc, x, y, K, N, M, g); }
#if XROWS >= 32
G void KAT fp4t_m32(KARGS) { tbody<2, true, 1, 2>(w, sc, x, y, K, N, M, g); }
#endif
G void KAT fp8t_m16(KARGS) { tbody<1, false, 1, UNRT>(w, sc, x, y, K, N, M, g); }
#if XROWS >= 32
G void KAT fp8t_m32(KARGS) { tbody<2, false, 1, 2>(w, sc, x, y, K, N, M, g); }
#endif
G void KAT fp4w_m16(KARGS) { tbody<1, true, 2, 1>(w, sc, x, y, K, N, M, g); }
#if XROWS >= 32
G void KAT fp4w_m32(KARGS) { tbody<2, true, 2, 1>(w, sc, x, y, K, N, M, g); }
#endif
G void KAT fp8w_m16(KARGS) { tbody<1, false, 2, 1>(w, sc, x, y, K, N, M, g); }
#if XROWS >= 32
G void KAT fp8w_m32(KARGS) { tbody<2, false, 2, 1>(w, sc, x, y, K, N, M, g); }
#endif

// Pure read of the same bytes, for the ratio.
G void rd(const v4u* __restrict p, u64 n, u64 stride, u32* out) {
  v4u a = {0, 0, 0, 0};
  for (u64 i = (u64)BX * 256 + TX; i < n; i += stride) a ^= __builtin_nontemporal_load(p + i);
  if ((a.x ^ a.y ^ a.z ^ a.w) == 0x9e3779b9u) out[0] = 1;
}
// Data: hashed codes; E4M3 scales in [2^-2, 2^5]; activations in [-1, 1).
D u32 hsh(u64 i) { u64 z = (i + 1) * 0x9e3779b97f4a7c15ull; z ^= z >> 29; z *= 0xbf58476d1ce4e5b9ull; z ^= z >> 32; return (u32)z; }
G void fill_bytes(unsigned char* p, u64 n, u32 seed, u32 mode) {
  for (u64 i = (u64)BX * 256 + TX; i < n; i += (u64)__builtin_amdgcn_grid_size_x()) {
    u32 v = hsh(i ^ ((u64)seed << 40));
    if (mode == 1) v = 0x28 + (v % 0x38);          // E4M3 scale, exponent 5..11
    else if (mode == 2) v = (v & 0xF7) == 0x7F ? 0x7E : (v & 0xF7);  // FP8 weight, |x| <= 240, no NaN
    p[i] = (unsigned char)v;
  }
}
G void fill_x(h* p, u64 n) {
  for (u64 i = (u64)BX * 256 + TX; i < n; i += (u64)__builtin_amdgcn_grid_size_x())
    p[i] = (h)((float)(hsh(i ^ 0x5555ull) & 0xFFFF) / 32768.0f - 1.0f);
}
// Naive reference in FP32.
D float e2m1f(u32 n) { const float t[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f}; float v = t[n & 7]; return (n & 8) ? -v : v; }
D float e4m3f(u32 b) {
  u32 e = (b >> 3) & 15, m = b & 7; float v = e ? __builtin_ldexpf(1.0f + m / 8.0f, (int)e - 7) : __builtin_ldexpf(m / 8.0f, -6);
  return (b & 0x80) ? -v : v;
}
G void ref(const unsigned char* w, const unsigned char* sc, const h* x, float* y, u32 K, u32 N, u32 M, u32 fp4, float g, u32 tiled) {
  u64 id = (u64)BX * 256 + TX; if (id >= (u64)M * N) return;
  u32 m = id / N, n = id % N; float s = 0.f;
  for (u32 k = 0; k < K; k++) {
    float wv;
    #ifdef KMAJOR
    u64 tile = (u64)(k / 64) * (N / 16) + n / 16;
#else
    u64 tile = (u64)(n / 16) * (K / 64) + k / 64;
#endif
    u32 r = n % 16, kt = k % 64;
    if (fp4) {
      u64 cb = tiled ? tile * 512 + r * 32 + kt / 2 : (u64)n * (K / 2) + k / 2;
      u64 sb = tiled ? tile * 64 + r * 4 + kt / 16 : (u64)n * (K / 16) + k / 16;
      u32 byte = w[cb]; wv = e2m1f((k & 1) ? byte >> 4 : byte & 15) * e4m3f(sc[sb]);
    } else wv = e4m3f(w[tiled ? tile * 1024 + r * 64 + kt : (u64)n * K + k]);
    s += (float)x[(u64)m * K + k] * wv;
  }
  y[id] = s * g;
}
