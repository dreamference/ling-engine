// The element type of the prefill path's intermediate tensors (CUDA source files only).
//
// The exact path keeps every activation between kernels in FP32. Behind --prefill-bf16 (off by default;
// specs/DREAMFERENCE_LING_ENGINE_MILESTONES.md §14, the exactness policy) the prefill's DeltaNet intermediates
// are BF16 where production keeps BF16 (reports/M2-prefill.md §8 item 3, cited as §7.3 in the backlog's B34
// row). The kernels that produce or consume them are templates over the element type; their `float`
// instantiations are the kernels as they were, since every helper here is the identity on float. New
// instantiations, not edits, carry the BF16 mode.
#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ling::kernels::elem {

template <typename T>
__device__ __forceinline__ float load(const T* p) {
  return *p;
}
template <>
__device__ __forceinline__ float load<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}

template <typename T>
__device__ __forceinline__ void store(T* p, float v) {
  *p = v;
}
template <>
__device__ __forceinline__ void store<__nv_bfloat16>(__nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}

// Four consecutive values at base[4 * i .. 4 * i + 3] (16-byte aligned base for float, 8-byte for BF16).
template <typename T>
__device__ __forceinline__ float4 load4(const T* base, size_t i) {
  return reinterpret_cast<const float4*>(base)[i];
}
template <>
__device__ __forceinline__ float4 load4<__nv_bfloat16>(const __nv_bfloat16* base, size_t i) {
  const uint2 u = reinterpret_cast<const uint2*>(base)[i];
  const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
  const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

template <typename T>
__device__ __forceinline__ void store4(T* base, size_t i, float4 v) {
  reinterpret_cast<float4*>(base)[i] = v;
}
template <>
__device__ __forceinline__ void store4<__nv_bfloat16>(__nv_bfloat16* base, size_t i, float4 v) {
  const __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
  uint2 u;
  u.x = *reinterpret_cast<const uint32_t*>(&lo);
  u.y = *reinterpret_cast<const uint32_t*>(&hi);
  reinterpret_cast<uint2*>(base)[i] = u;
}

// Two consecutive values at p (8-byte aligned for float, 4-byte for BF16).
template <typename T>
__device__ __forceinline__ void store2(T* p, float a, float b) {
  *reinterpret_cast<float2*>(p) = make_float2(a, b);
}
template <>
__device__ __forceinline__ void store2<__nv_bfloat16>(__nv_bfloat16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

// Two consecutive values at p as one packed BF16 pair (an MMA A-fragment register): FP32 values are rounded
// to BF16, BF16 values are read as they are, so both give the same register for the same values.
template <typename T>
__device__ __forceinline__ uint32_t pack_bf16x2(const T* p) {
  const float2 v = *reinterpret_cast<const float2*>(p);
  const __nv_bfloat162 h = __floats2bfloat162_rn(v.x, v.y);
  return *reinterpret_cast<const uint32_t*>(&h);
}
template <>
__device__ __forceinline__ uint32_t pack_bf16x2<__nv_bfloat16>(const __nv_bfloat16* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

}  // namespace ling::kernels::elem
