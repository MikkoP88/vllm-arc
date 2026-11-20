// MX KV-cache format traits + in-register dequant bit-math for the
// arc7 MX-native paged-decode kernel (P6.47 Phase 8, wheel section).
//
// One source of truth for the byte geometry of the TQ-AoS MX slot
// (mirrors vllm_gaudi/xpu/patches/mx_kv/mx_layout.py — the python side
// owns the law; these constexpr must not drift from it):
//
//   slot = [ K bytes ][ Ks e8m0 x D/32 ][ V bytes ][ Vs e8m0 x D/32 ]
//
//   mxfp8 : K e4m3 x D          | V e4m3 x D
//   mxfp4 : K e2m1 nibbles D/2  | V e2m1 nibbles D/2
//   mxk8v4: K e4m3 x D          | V e2m1 nibbles D/2
//
// All surfaces are uint8 with seq stride Hk*slot_bytes, head stride
// slot_bytes, page stride block*Hk*slot_bytes — the kernel consumes the
// cache tensor's own strides, so no re-layout ever happens.
//
// Dequant law (identical to the Triton v2/v5 kernels and mx_quant.py):
//   value = fp(payload) * 2^(e8m0_code - 127)
//   nibble order: element 2j = LOW nibble of byte j, 2j+1 = HIGH
// The e8m0 factor is exact in f32 via bit assembly: code<<23 is the
// IEEE-754 encoding of 2^(code-127) (code != 0xFF; the quantizer clamps
// codes to [0, 254], so the NaN/inf band is unreachable in-cache).
#pragma once

#include <cstdint>

#include <cutlass/numeric_types.h>
#include <cute/config.hpp>

namespace cutlass::fmha::collective::mx {

enum class MxFormat { mxfp8, mxfp4, mxk8v4 };

constexpr int kMxGroupSize = 32;  // elements per e8m0 scale group (MX law)

template <MxFormat F>
struct MxTraits;

// D = head_size in elements (128 on this lane); all offsets in bytes.
template <>
struct MxTraits<MxFormat::mxfp8> {
  static constexpr int k_bytes(int d) { return d; }
  static constexpr int v_bytes(int d) { return d; }
  static constexpr int ks_off(int d) { return d; }
  static constexpr int vs_off(int d) { return 2 * d + d / kMxGroupSize; }
  static constexpr int slot_bytes(int d) {
    return 2 * d + 2 * (d / kMxGroupSize);
  }
  static constexpr bool k_is_e4m3 = true;   // K payload type
  static constexpr bool v_is_e4m3 = true;   // V payload type
};

template <>
struct MxTraits<MxFormat::mxfp4> {
  static constexpr int k_bytes(int d) { return d / 2; }
  static constexpr int v_bytes(int d) { return d / 2; }
  static constexpr int ks_off(int d) { return d / 2; }
  static constexpr int vs_off(int d) { return d + d / kMxGroupSize; }
  static constexpr int slot_bytes(int d) {
    return d + 2 * (d / kMxGroupSize);
  }
  static constexpr bool k_is_e4m3 = false;
  static constexpr bool v_is_e4m3 = false;
};

template <>
struct MxTraits<MxFormat::mxk8v4> {
  static constexpr int k_bytes(int d) { return d; }
  static constexpr int v_bytes(int d) { return d / 2; }
  static constexpr int ks_off(int d) { return d; }
  static constexpr int vs_off(int d) {
    return 3 * d / 2 + d / kMxGroupSize;
  }
  static constexpr int slot_bytes(int d) {
    return 3 * d / 2 + 2 * (d / kMxGroupSize);
  }
  static constexpr bool k_is_e4m3 = true;
  static constexpr bool v_is_e4m3 = false;
};

// ---------------------------------------------------------------------------
// In-register dequant primitives (scalar; the dequant loops over
// fragment values are fully unrolled and vectorize to the same convert
// class the stock fp8 reorder() path emits).
// ---------------------------------------------------------------------------

// e8m0 scale code -> exact float factor 2^(code-127). code in [0,254].
CUTE_HOST_DEVICE
inline float
e8m0_to_scale(uint8_t code) {
  uint32_t bits = uint32_t(code) << 23;
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

// e2m1 nibble (4 bits: s|ee|m) -> float. Values: 0, .5, 1, 1.5, 2, 3,
// 4, 6 and negatives; subnormals at ee==0 (0 / 0.5). r204 lean
// BRANCHLESS assembly (mirrors the r203b e4m3 lean — the ee==0 branch
// defeated predication on the e2m1-K/V hot walks): normals assemble
// directly, bits = s | (126+ee)<<23 | m<<22; the subnormal band is
// m * 0.5, exact as a float multiply, with the sign on a negate
// select (keeps -0.0 bitwise — the faces gate is bit-exact).
CUTE_HOST_DEVICE
inline float
e2m1_to_float(uint8_t nibble) {
  const uint32_t s = uint32_t(nibble >> 3) & 1u;
  const uint32_t ee = uint32_t(nibble >> 1) & 3u;
  const uint32_t m = uint32_t(nibble) & 1u;
  const uint32_t bits = (s << 31) | ((126u + ee) << 23) | (m << 22);
  float fn;
  __builtin_memcpy(&fn, &bits, 4);
  float sub = float(m) * 0.5f;
  sub = s ? -sub : sub;
  return (ee == 0) ? sub : fn;
}

// One packed byte -> two dequantized floats (element 2j = low nibble,
// 2j+1 = high; mx_quant.py nibble law), scaled by the group factor.
CUTE_HOST_DEVICE
inline void
e2m1_byte_to_pair(uint8_t byte, float scale, float& lo, float& hi) {
  lo = e2m1_to_float(uint8_t(byte & 0xF)) * scale;
  hi = e2m1_to_float(uint8_t(byte >> 4)) * scale;
}

}  // namespace cutlass::fmha::collective::mx
