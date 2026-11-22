#!/usr/bin/env python3
"""p212m_patch.py -- r212m: direct-bf16 decode with the e8m0 exponent
folded into the bf16 exponent field.

The r212m probe-9 measurement (r212m_probedec.sh, 2026-11-21): severing
ONLY the decode tail (e2m1/e4m3 bit assembly + scale multiply +
f32->bf16 convert) dropped the 32K walk from 1027.8 to 291.0 us
(mxfp4, 71.7%) and 1361.8 to 589.0 us (mxfp8, 56.7%) -- the decode
math is THE bottleneck, fully exposed (the m9 floor sits at the
memory-bandwidth floor).

Bit-exactness argument: every e2m1/e4m3 value times a power of two is
exactly representable in bf16 (<= 4 significant mantissa bits vs 8), so
whenever the biased exponent stays in the normal band [1, 254] the
directly assembled bf16 bits are IDENTICAL to float-assembly + multiply
+ convert (no rounding anywhere; e8m0 scales are pure powers of two and
always positive, so the product's sign is the payload's sign).
  e2m1: bits = s<<15 | (126+ee+k)<<7 | m<<6, base_e = code - 1
        (subnormal m=1 is exactly 0.5 = 2^(k-1), landing on base_e;
        ee=0 m=0 is +-0)
  e4m3: bits = s<<15 | (120+ee+k)<<7 | m<<4, base_e = code - 7
        (normals only; payload subnormals ee==0 take the float path)
Band edges (E' < 1 denormal band, E' > 254 overflow, code 0 whose
bit-assembly float semantics the legacy path preserves) route slow via
per-group band checks -- one predictable branch per group. The r198
bitwise faces gate the claim exhaustively at seq_len==1.

Env switch ARC_MXKV_DIRECTBF16 (default 1; 0 = legacy float path) gives
the A/B oracle. Sits ON TOP of the probe patch (probe 9 unchanged).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

HELPERS = """
// r212m DIRECT-BF16 DECODE LAW (the probe-9 measurement: the decode
// tail -- bit assembly + scale multiply + f32->bf16 convert -- is
// 56-72% of the 32K-context walk): assemble the bf16 BITS of
// payload * 2^k directly. Every e2m1/e4m3 value times a power of
// two is exactly representable in bf16 (<= 4 significant bits vs 8),
// so whenever the biased exponent lands in the normal band [1, 254]
// the direct bits are IDENTICAL to the float-assembly + multiply +
// convert bits -- no rounding anywhere. base_e is the biased bf16
// exponent of the group's smallest normal payload value scaled:
// e2m1 base_e = 126 + k = code - 1 (the subnormal m=1 is exactly
// 0.5, landing on base_e itself; ee=0 m=0 is +-0); e4m3
// base_e = 120 + k = code - 7 (normals only -- payload subnormals
// ee==0 take the float path). Out-of-band codes (including code 0,
// whose bit-assembly float semantics the legacy path preserves)
// route slow via the band checks. The r198 bitwise faces gate the
// whole claim exhaustively at seq_len==1.
CUTE_HOST_DEVICE
inline uint16_t
e2m1_scaled_bf16(uint32_t nib, int base_e) {
  uint32_t s = (nib >> 3) & 1u;
  uint32_t ee = (nib >> 1) & 3u;
  uint32_t m = nib & 1u;
  uint16_t nz = uint16_t(
      (s << 15) | (uint32_t(base_e + int(ee)) << 7) | (m << 6));
  return (m | ee) ? nz : uint16_t(s << 15);
}

CUTE_HOST_DEVICE
inline uint16_t
e4m3_scaled_bf16(uint32_t b, int base_e) {
  uint32_t s = (b >> 7) & 1u;
  uint32_t ee = (b >> 3) & 0xFu;
  uint32_t m = b & 7u;
  return uint16_t(
      (s << 15) | (uint32_t(base_e + int(ee)) << 7) | (m << 4));
}

// Group-band checks: every payload value of a group whose code passes
// stays in the bf16 normal band after scaling, so the fast path covers
// the whole group (one predictable branch per group; e2m1 needs no
// per-element predication at all).
CUTE_HOST_DEVICE
inline bool
e2m1_band_ok(uint8_t code) {
  int k = int(code) - 127;
  return k >= -125 && k <= 125;  // E' = 126+ee+k in [1, 254]
}

CUTE_HOST_DEVICE
inline bool
e4m3_band_ok(uint8_t code) {
  int k = int(code) - 127;
  return k >= -120 && k <= 119;  // E' = 120+ee+k in [1, 254], ee >= 1
}
"""

EDITS = {
    "mx_dequant.hpp": [
        # (1) helpers after e4m3_to_float
        (
            """  float sub = float(m) * 0x1p-9f;
  sub = s ? -sub : sub;
  return (e == 0) ? sub : fn;
}
""",
            """  float sub = float(m) * 0x1p-9f;
  sub = s ? -sub : sub;
  return (e == 0) ? sub : fn;
}
""" + HELPERS,
        ),
        # (2) dequant_reg signature
        (
            """dequant_reg(
    ScaleTensor const& scales,
    SrcFrag const& src,
    DstFrag& dst,
    CoordView const& src_coords,
    bool probe_val = false) {""",
            """dequant_reg(
    ScaleTensor const& scales,
    SrcFrag const& src,
    DstFrag& dst,
    CoordView const& src_coords,
    bool probe_val = false,
    bool direct_bf16 = true) {""",
        ),
        # (3) dequant_reg scale vectors -> codes + band + bases
        (
            """  int grp0 = org_col / kMxGroupSize;
  float sc0[8], sc1[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    sc0[r] = e8m0_to_scale(scales(kr[r], grp0));
    sc1[r] = e8m0_to_scale(scales(kr[r], grp0 + 1));
  }
  bool direct = true;""",
            """  int grp0 = org_col / kMxGroupSize;
  // r212m CODE-DOMAIN SCALES: the walk consumes e8m0 CODES -- the
  // fast path folds them into the bf16 exponent (base_e = code - 7 =
  // 120 + k, k = code - 127); the rare slow element derives its
  // float scale via e8m0_to_scale(code) on demand. Identical gmem
  // traffic (same 16 byte reads).
  uint8_t kc0[8], kc1[8];
  bool fast_band = direct_bf16;
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    kc0[r] = uint8_t(scales(kr[r], grp0));
    kc1[r] = uint8_t(scales(kr[r], grp0 + 1));
    fast_band = fast_band && detail::e4m3_band_ok(kc0[r]) &&
                detail::e4m3_band_ok(kc1[r]);
  }
  int be0[8], be1[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    be0[r] = int(kc0[r]) - 7;
    be1[r] = int(kc1[r]) - 7;
  }
  bool direct = true;""",
        ),
        # (4) dequant_reg per-element tail: fast path + code-derived slow
        (
            """    float scale = 0.f;
    if (direct) {
      // Direct index: slot = (i&1) + 2*(i>>4) selects the row slot
      // (compile-time under the unroll; the kr guard itself folds to
      // a compile-time compare of the two tv offsets), the column's
      // group offset goff = grp - grp0 selects between the two flat
      // vectors — the only runtime index is this 0/1 select.
      int slot = (i & 1) + 2 * (i >> 4);
      int goff = grp - grp0;
      if (kr[slot] != row || goff < 0 || goff > 1) {
        direct = false;
        scale = e8m0_to_scale(scales(row, grp));
      } else {
        scale = goff ? sc1[slot] : sc0[slot];
      }
    } else {
      scale = e8m0_to_scale(scales(row, grp));
    }

    // Byte from the loaded fragment register — no per-element memory.
    // r212m probe 9: sever ONLY the decode tail (assembly + scale mul);
    // loads/extracts/selects/writes stay.
    float val = probe_val
        ? float(uint8_t(src(i)))
        : detail::e4m3_to_float(uint8_t(src(i)));
    // bfloat16_t's float ctor is explicit (no implicit conversion on
    // assignment); build the element type directly.
    dst(i) = typename DstFrag::value_type(probe_val ? val : val * scale);""",
            """    float scale = 0.f;
    uint8_t byte = uint8_t(src(i));
    if (direct) {
      // Direct index: slot = (i&1) + 2*(i>>4) selects the row slot
      // (compile-time under the unroll; the kr guard itself folds to
      // a compile-time compare of the two tv offsets), the column's
      // group offset goff = grp - grp0 selects between the two flat
      // vectors — the only runtime index is this 0/1 select.
      int slot = (i & 1) + 2 * (i >> 4);
      int goff = grp - grp0;
      if (kr[slot] != row || goff < 0 || goff > 1) {
        direct = false;
        scale = e8m0_to_scale(scales(row, grp));
      } else if (fast_band && ((byte >> 3) & 0xF) != 0) {
        // r212m DIRECT-BF16: fold the e8m0 code into the bf16 exponent
        // (base_e = code - 7; bit-identical to float assembly + mul +
        // convert in the normal band — see e4m3_scaled_bf16). Payload
        // subnormals (ee == 0) and out-of-band groups take the float
        // path below.
        uint16_t bits = detail::e4m3_scaled_bf16(
            byte, goff ? be1[slot] : be0[slot]);
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(i) = o;
        continue;
      } else {
        scale = e8m0_to_scale(goff ? kc1[slot] : kc0[slot]);
      }
    } else {
      scale = e8m0_to_scale(scales(row, grp));
    }

    // Float path (slow band edges / fallback): byte from the loaded
    // fragment register — no per-element memory.
    // r212m probe 9: sever ONLY the decode tail (assembly + scale mul);
    // loads/extracts/selects/writes stay.
    float val = probe_val
        ? float(byte)
        : detail::e4m3_to_float(byte);
    // bfloat16_t's float ctor is explicit (no implicit conversion on
    // assignment); build the element type directly.
    dst(i) = typename DstFrag::value_type(probe_val ? val : val * scale);""",
        ),
        # (5) dequant_vslm signature
        (
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false) {""",
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true) {""",
        ),
        # (6) dequant_vslm scale vector -> codes + band + bases
        (
            """  int grp0 = v_base / kMxGroupSize;
  uint32_t gsh = 8 * (grp0 & 3);
  const uint32_t* gl = (grp0 < 4) ? &line[0][0] : &line[0][1];
  float sc[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    sc[r] = skip_scales ? 1.f : e8m0_to_scale(
        uint8_t((gl[2 * r] >> gsh) & 0xFF));
  }""",
            """  int grp0 = v_base / kMxGroupSize;
  uint32_t gsh = 8 * (grp0 & 3);
  const uint32_t* gl = (grp0 < 4) ? &line[0][0] : &line[0][1];
  // r212m CODE-DOMAIN SCALES: kc[r] feeds the direct-bf16 fast path
  // (base_e = code - 7 for e4m3, code - 1 for e2m1); the float
  // vector is derived ONLY when a slow element can occur — e4m3
  // always (its payload-subnormal bytes fall back per byte), e2m1
  // only when a group sits outside the normal band — or a probe
  // demands it.
  uint8_t kc[8];
  bool fast_band = direct_bf16;
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    kc[r] = skip_scales ? uint8_t(127)
                        : uint8_t((gl[2 * r] >> gsh) & 0xFF);
    fast_band = fast_band && (VIsE4m3 ? detail::e4m3_band_ok(kc[r])
                                      : detail::e2m1_band_ok(kc[r]));
  }
  int be[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    be[r] = int(kc[r]) - (VIsE4m3 ? 7 : 1);
  }
  float sc[8] = {};
  if (!fast_band || probe_val || VIsE4m3) {
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      sc[r] = skip_scales ? 1.f : e8m0_to_scale(kc[r]);
    }
  }""",
        ),
        # (7) dequant_vslm per-element tail: fast path
        (
            """    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    // r212m probe 9: sever ONLY the decode tail (assembly + scale
    // mul); the SLM reads, extracts and fragment writes stay.
    float val = probe_val
        ? float(b)
        : (VIsE4m3 ? detail::e4m3_to_float(b)
                   : e2m1_to_float(uint8_t(
                         nib_phase ? (b >> 4) : (b & 0xF))));

    dst(i) = typename DstFrag::value_type(
        probe_val ? val : val * sc[r]);""",
            """    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    // r212m DIRECT-BF16 fast path: e2m1 covers every in-band nibble
    // (zero nibbles return +-0); e4m3 covers only ee != 0 bytes —
    // payload subnormals fall back to the float path below.
    if (!probe_val && fast_band &&
        (!VIsE4m3 || ((b >> 3) & 0xF) != 0)) {
      uint16_t bits = VIsE4m3
          ? detail::e4m3_scaled_bf16(b, be[r])
          : detail::e2m1_scaled_bf16(
                uint8_t(nib_phase ? (b >> 4) : (b & 0xF)), be[r]);
      typename DstFrag::value_type o;
      __builtin_memcpy(&o, &bits, 2);
      dst(i) = o;
      continue;
    }
    // r212m probe 9: sever ONLY the decode tail (assembly + scale
    // mul); the SLM reads, extracts and fragment writes stay.
    float val = probe_val
        ? float(b)
        : (VIsE4m3 ? detail::e4m3_to_float(b)
                   : e2m1_to_float(uint8_t(
                         nib_phase ? (b >> 4) : (b & 0xF))));

    dst(i) = typename DstFrag::value_type(
        probe_val ? val : val * sc[r]);""",
        ),
        # (8) dequant_dtile signature
        (
            """dequant_dtile(
    ByteView const& bytes,
    ScaleTensor const& scales,
    DstFrag& dst,
    CoordView const& dst_coords,
    bool probe_val = false) {""",
            """dequant_dtile(
    ByteView const& bytes,
    ScaleTensor const& scales,
    DstFrag& dst,
    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true) {""",
        ),
        # (9) dequant_dtile scale quads -> codes + band + bases
        (
            """    int row0 = org_row + t1;
    float sc00 = e8m0_to_scale(scales(row0, grp0));
    float sc01 = e8m0_to_scale(scales(row0, grp0 + 1));
    float sc10 = e8m0_to_scale(scales(row0 + 8, grp0));
    float sc11 = e8m0_to_scale(scales(row0 + 8, grp0 + 1));""",
            """    int row0 = org_row + t1;
    // r212m CODE-DOMAIN SCALES: base_e = code - 1 (= 126 + k); the
    // slow path derives its float on demand. Identical gmem traffic
    // (same 4 byte reads).
    uint8_t c00 = uint8_t(scales(row0, grp0));
    uint8_t c01 = uint8_t(scales(row0, grp0 + 1));
    uint8_t c10 = uint8_t(scales(row0 + 8, grp0));
    uint8_t c11 = uint8_t(scales(row0 + 8, grp0 + 1));
    bool fast_band = direct_bf16 && detail::e2m1_band_ok(c00) &&
                     detail::e2m1_band_ok(c01) &&
                     detail::e2m1_band_ok(c10) &&
                     detail::e2m1_band_ok(c11);
    int be00 = int(c00) - 1, be01 = int(c01) - 1;
    int be10 = int(c10) - 1, be11 = int(c11) - 1;""",
        ),
        # (10) dequant_dtile staged walk: drop hoisted float select,
        # add the fast path, derive slow scale on demand
        (
            """      int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
      int goff = (off + org_emod) >> 5;
      float scale =
          goff ? (dv & 1 ? sc11 : sc01) : (dv & 1 ? sc10 : sc00);""",
            """      int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
      int goff = (off + org_emod) >> 5;""",
        ),
        (
            """      uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;

      // r212m probe 9: sever ONLY the decode tail (assembly + scale
      // mul); the staged window reads and decode arithmetic on the
      // addresses stay.
      float val = probe_val
          ? float(uint8_t(b))
          : e2m1_to_float(uint8_t(t0 ? (b >> 4) : (b & 0xF)));
      // bfloat16_t's float ctor is explicit (no implicit conversion on
      // assignment); build the element type directly.
      dst(dv) = typename DstFrag::value_type(
          probe_val ? val : val * scale);""",
            """      uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;

      // r212m DIRECT-BF16 fast path: every in-band nibble is covered
      // (zero nibbles return +-0); out-of-band groups fall through.
      if (!probe_val && fast_band) {
        uint16_t bits = detail::e2m1_scaled_bf16(
            uint8_t(t0 ? (b >> 4) : (b & 0xF)),
            (dv & 1) ? (goff ? be11 : be10) : (goff ? be01 : be00));
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(dv) = o;
        continue;
      }
      // Slow path: derive the float scale on demand.
      float scale = e8m0_to_scale(
          goff ? (dv & 1 ? c11 : c01) : (dv & 1 ? c10 : c00));
      // r212m probe 9: sever ONLY the decode tail (assembly + scale
      // mul); the staged window reads and decode arithmetic on the
      // addresses stay.
      float val = probe_val
          ? float(uint8_t(b))
          : e2m1_to_float(uint8_t(t0 ? (b >> 4) : (b & 0xF)));
      // bfloat16_t's float ctor is explicit (no implicit conversion on
      // assignment); build the element type directly.
      dst(dv) = typename DstFrag::value_type(
          probe_val ? val : val * scale);""",
        ),
    ],
    "mx_paged_decode_mainloop.hpp": [
        # (11) Arguments field
        (
            """    // r200c debug lane (args.debug_buf layout documented at the args
    // struct; SINGLE-workgroup shapes only — WGs race on the buffer).
    void* debug_buf = nullptr;
  };""",
            """    // r200c debug lane (args.debug_buf layout documented at the args
    // struct; SINGLE-workgroup shapes only — WGs race on the buffer).
    void* debug_buf = nullptr;
    // r212m direct-bf16 decode switch (env ARC_MXKV_DIRECTBF16; the
    // fast path folds the e8m0 code into the bf16 exponent — bit-
    // identical in the normal band, see mx_dequant.hpp). Default on;
    // 0 forces the float path (A/B oracle).
    int direct_bf16 = 1;
  };""",
        ),
        # (12) to_underlying_arguments pack
        (
            """        args.vlane_probe,
        args.debug_buf};""",
            """        args.vlane_probe,
        args.debug_buf,
        args.direct_bf16};""",
        ),
        # (13) dequant_reg call site
        (
            """          mx::dequant_reg<true, true>(
              ks_surf,
              tKrB,
              tKrBq,
              tKBgK_cache(_, _, _, D),
              params.vlane_probe == 9);""",
            """          mx::dequant_reg<true, true>(
              ks_surf,
              tKrB,
              tKrBq,
              tKBgK_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0);""",
        ),
        # (14) dequant_dtile call site
        (
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9);""",
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0);""",
        ),
        # (15) dequant_vslm call site
        (
            """              params.vlane_probe == 5,
              params.vlane_probe == 9);""",
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0);""",
        ),
    ],
    "mx_paged_decode.hpp": [
        # (16) args_t field
        (
            """  // Null = off, zero cost.
  void* debug_buf = nullptr;
};""",
            """  // Null = off, zero cost.
  void* debug_buf = nullptr;
  // r212m direct-bf16 decode switch (env ARC_MXKV_DIRECTBF16; the
  // fast path folds the e8m0 code into the bf16 exponent — bit-
  // identical in the normal band, see mx_dequant.hpp). Default on;
  // 0 forces the float path (A/B oracle).
  int direct_bf16 = 1;
};""",
        ),
        # (17) config pack
        (
            """         args.vlane_probe,
         args.debug_buf},""",
            """         args.vlane_probe,
         args.debug_buf,
         args.direct_bf16},""",
        ),
        # (18) env parse
        (
            """    mx_paged_decode_args_t args_p = args;
    if (const char* vprobe = std::getenv("ARC_MXKV_VPROBE")) {
      args_p.vlane_probe = std::atoi(vprobe);
    }""",
            """    mx_paged_decode_args_t args_p = args;
    if (const char* vprobe = std::getenv("ARC_MXKV_VPROBE")) {
      args_p.vlane_probe = std::atoi(vprobe);
    }
    // r212m: direct-bf16 decode switch (default on; 0 = float path).
    if (const char* dbf = std::getenv("ARC_MXKV_DIRECTBF16")) {
      args_p.direct_bf16 = std::atoi(dbf);
    }""",
        ),
    ],
}


def main():
    for fname, pairs in EDITS.items():
        path = SRC / fname
        text = path.read_text()
        for old, new in pairs:
            if text.count(old) != 1:
                print(f"PATCH_ANCHOR_MISS {fname}: count={text.count(old)}")
                print(f"  anchor head: {old[:72]!r}")
                sys.exit(1)
            text = text.replace(old, new)
        path.write_text(text)
        print(f"PATCHED {fname} ({len(pairs)} edits)")
    print("P212M_PATCH_OK")


if __name__ == "__main__":
    main()
