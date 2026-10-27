// Which 2:4 sparse tensor-core MMAs sm_121 accepts and runs (the low-bit study, spec LOWBIT section 6).
//
// One variant per build (-DVARIANT=n), because ptxas rejects a whole file for one instruction it does
// not know; sparse_isa_probe.sh builds each, records accepted / rejected, and runs the accepted ones.
// Each kernel multiplies an all-ones compressed A (every group of four keeping positions 0 and 1,
// metadata nibble 0b0100) by an all-ones B, so every output must be K / 2.
//   1  mma.sp::ordered_metadata m16n8k32 f16 x f16 -> f32           (the engine's FP16 path, sparse)
//   2  mma.sp::ordered_metadata m16n8k64 kind::f8f6f4 e4m3 x e4m3   (sparse FP8)
//   3  mma.sp::ordered_metadata m16n8k128 kind::mxf4nvf4 block_scale scale_vec::4X e2m1, ue4m3 (sparse NVFP4)
//   4  the same with scale_vec::2X
//   5  mma.sp::ordered_metadata m16n8k64 kind::mxf8f6f4 block_scale scale_vec::1X e4m3, ue8m0 (sparse MXFP8)
//   6  mma.sp::ordered_metadata m16n8k64 kind::f8f6f4 e2m1 x e2m1   (sparse FP4 without block scales)
// Build: nvcc -O3 -std=c++20 -gencode arch=compute_121a,code=sm_121a -DVARIANT=n sparse_isa_probe.cu
#include <cstdint>
#include <cstdio>

__global__ void probe(float* out, int* expect) {
  const uint32_t meta = 0x44444444u;  // every group of four keeps positions 0 and 1
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f, c = 0.f;
#if VARIANT == 1
  const uint32_t a = 0x3C003C00u, b = a;  // f16 1.0
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b), "r"(b), "r"(b), "f"(c), "f"(c), "f"(c), "f"(c), "r"(meta));
  *expect = 16;
#elif VARIANT == 2 || VARIANT == 6
#if VARIANT == 2
  const uint32_t a = 0x38383838u;  // e4m3 1.0
#define KIND "e4m3.e4m3"
#else
  const uint32_t a = 0x08080808u;  // e2m1 1.0 (0b0010) in bits 2-5 of each byte (f8f6f4's FP4 container)
#define KIND "e2m1.e2m1"
#endif
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32." KIND ".f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "f"(c), "f"(c), "f"(c), "f"(c), "r"(meta));
  *expect = 32;
#elif VARIANT == 3 || VARIANT == 4
  const uint32_t a = 0x22222222u;   // e2m1 1.0, two per byte
  const uint32_t sf = 0x38383838u;  // UE4M3 scales of 1.0
  const uint16_t z = 0;
#if VARIANT == 3
#define SV "4X"
#else
#define SV "2X"
#endif
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::" SV
      ".f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17}, {%18,%19}, {%20}, {%21,%22};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "f"(c), "f"(c), "f"(c), "f"(c), "r"(meta),
        "r"(sf), "h"(z), "h"(z), "r"(sf), "h"(z), "h"(z));
  *expect = 64;
#elif VARIANT == 5
  const uint32_t a = 0x38383838u;   // e4m3 1.0
  const uint32_t sf = 0x7f7f7f7fu;  // UE8M0 scales of 1.0
  const uint16_t z = 0;
  asm volatile(
      "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
      ".f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0, {%17}, {%18,%19}, {%20}, {%21,%22};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "f"(c), "f"(c), "f"(c), "f"(c), "r"(meta),
        "r"(sf), "h"(z), "h"(z), "r"(sf), "h"(z), "h"(z));
  *expect = 32;
#endif
  out[threadIdx.x * 4 + 0] = d0;
  out[threadIdx.x * 4 + 1] = d1;
  out[threadIdx.x * 4 + 2] = d2;
  out[threadIdx.x * 4 + 3] = d3;
}

int main() {
  float* d;
  int* e;
  cudaMalloc(&d, 128 * sizeof(float));
  cudaMallocManaged(&e, sizeof(int));
  probe<<<1, 32>>>(d, e);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("variant %d: launch failed: %s\n", VARIANT, cudaGetErrorString(err));
    return 1;
  }
  float h[128];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (float v : h) bad += v != (float)*e;
  printf("variant %d: ran, %d of 128 outputs differ from %d (first %g)\n", VARIANT, bad, *e, h[0]);
  return bad ? 2 : 0;
}
