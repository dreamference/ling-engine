// Header-free WMMA throughput probe for gfx1151 (wave32). Each wave keeps 8 independent accumulators.
#define G extern "C" __attribute__((global))
typedef float v8f __attribute__((ext_vector_type(8)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v2i __attribute__((ext_vector_type(2)));
#define TID __builtin_amdgcn_workitem_id_x()
G void f16(float* out, int iters) {
  v16h a, b; for (int i = 0; i < 16; i++) { a[i] = (_Float16)(TID * 0.001f + i); b[i] = (_Float16)(i * 0.5f); }
  v8f c[8] = {};
  for (int it = 0; it < iters; it++)
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[j]);
  float s = 0; for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) s += c[j][i];
  if (s == 1.2345f) out[0] = s;
}
G void bf16(float* out, int iters) {
  v16s a, b; for (int i = 0; i < 16; i++) { a[i] = (short)(0x3f80 + TID + i); b[i] = (short)(0x3f80 + i); }
  v8f c[8] = {};
  for (int it = 0; it < iters; it++)
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c[j]);
  float s = 0; for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) s += c[j][i];
  if (s == 1.2345f) out[0] = s;
}
G void iu8(float* out, int iters) {
  v4i a, b; for (int i = 0; i < 4; i++) { a[i] = TID * 7 + i; b[i] = i * 3 + 1; }
  v8i c[8] = {};
  for (int it = 0; it < iters; it++)
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c[j], false);
  int s = 0; for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) s += c[j][i];
  if (s == 12345) out[0] = (float)s;
}
G void iu4(float* out, int iters) {
  v2i a, b; for (int i = 0; i < 2; i++) { a[i] = TID * 7 + i; b[i] = i * 3 + 1; }
  v8i c[8] = {};
  for (int it = 0; it < iters; it++)
#pragma unroll
    for (int j = 0; j < 8; j++) c[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c[j], false);
  int s = 0; for (int j = 0; j < 8; j++) for (int i = 0; i < 8; i++) s += c[j][i];
  if (s == 12345) out[0] = (float)s;
}
