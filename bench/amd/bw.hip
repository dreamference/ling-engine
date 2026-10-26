// Header-free HIP device code: read, write and copy bandwidth kernels for gfx1151.
#define G extern "C" __attribute__((global))
typedef unsigned int u32;
typedef unsigned long long u64;
typedef u32 u4 __attribute__((ext_vector_type(4)));
static __attribute__((device)) inline u64 tid() {
  return (u64)__builtin_amdgcn_workgroup_id_x() * 256ull + __builtin_amdgcn_workitem_id_x();
}
G void rd(const u4* __restrict p, u64 n, u64 stride, u32* out) {
  u4 acc = {0, 0, 0, 0};
  for (u64 i = tid(); i < n; i += stride) acc ^= __builtin_nontemporal_load(p + i);
  u32 x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x9e3779b9u) out[0] = x;  // never true for a zero buffer; keeps loads live
}
G void rd4(const u4* __restrict p, u64 n, u64 stride, u32* out) {
  u4 a = {0,0,0,0}, b = a, c = a, d = a;
  u64 i = tid();
  for (; i + 3 * stride < n; i += 4 * stride) {
    a ^= __builtin_nontemporal_load(p + i);
    b ^= __builtin_nontemporal_load(p + i + stride);
    c ^= __builtin_nontemporal_load(p + i + 2 * stride);
    d ^= __builtin_nontemporal_load(p + i + 3 * stride);
  }
  for (; i < n; i += stride) a ^= p[i];
  u4 acc = a ^ b ^ c ^ d;
  u32 x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x9e3779b9u) out[0] = x;
}
G void wr(u4* __restrict p, u64 n, u64 stride, u32* out) {
  u4 v = {1, 2, 3, 4};
  for (u64 i = tid(); i < n; i += stride) __builtin_nontemporal_store(v, p + i);
}
G void cp(const u4* __restrict s, u4* __restrict d, u64 n, u64 stride) {
  for (u64 i = tid(); i < n; i += stride) __builtin_nontemporal_store(__builtin_nontemporal_load(s + i), d + i);
}
G void fill(u4* __restrict p, u64 n, u64 stride, u32* out) {
  for (u64 i = tid(); i < n; i += stride) {
    u64 h = (i + 1) * 0x9e3779b97f4a7c15ull; h ^= h >> 29; h *= 0xbf58476d1ce4e5b9ull; h ^= h >> 32;
    u4 v = {(u32)h, (u32)(h >> 32), (u32)(h * 3), (u32)(h >> 17)};
    p[i] = v;
  }
}
