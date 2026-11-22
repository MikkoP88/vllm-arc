#!/usr/bin/env python3
"""p212m3_patch.py -- r212m3: gate direct-bf16 to the e2m1 lanes only.

Evidence chain: the e4m3 fast path gains +17.4% same-wheel, but the
restructured e4m3 tail costs the float path monotonically more each
wheel (1359 -> 1750 -> 1862 at 32K dbf0) — net mxfp8 fast 1537.6 is
+13% WORSE than the r212l float path, while the e2m1 lanes sit at a
clean -27% (750.5 us at 32K). The e4m3 slow tail's per-element
e8m0_to_scale(code-select) (r212l paid a plain float select) plus the
codegen shifts from carrying the dual machinery are the suspects.

r212m3 reverts the e4m3 decode to the r212l-equivalent float path
(byte-equivalent logic; the probe-wheel build measured 1361.8 =
r212l-class layout) and keeps direct-bf16 for e2m1 only:
  - dequant_reg (e4m3-only: mxfp8 K, mxk8v4 K): reverted to the
    probe-wheel text (probe 9 kept, fast path dropped)
  - dequant_vslm: if constexpr (!VIsE4m3) keeps the trimmed fast path;
    the e4m3 branch restores r212l's exact scale-vector prep and tail
  - dequant_dtile (e2m1-only): unchanged (the winning state)
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) dequant_reg: revert scale prep to r212l floats
        (
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
  // r212m2 REGISTER TRIM: no be arrays — base_e derives inline
  // from the code select (int(code) - 7); 16 fewer live ints in the
  // hot loop (the r212m mxfp8 float-path regression suspect).
  bool direct = true;""",
            """  int grp0 = org_col / kMxGroupSize;
  // r212m3 E4M3 REVERT: the e4m3 lanes keep the r212l float path —
  // the direct-bf16 gain (+17.4% same-wheel) did not cover the
  // restructured tail's cost (float path 1359 -> 1862 monatone across
  // r212m/r212m2; net fast = +13% over the old float path). e2m1
  // lanes keep direct-bf16 (see dequant_vslm / dequant_dtile).
  float sc0[8], sc1[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    sc0[r] = e8m0_to_scale(scales(kr[r], grp0));
    sc1[r] = e8m0_to_scale(scales(kr[r], grp0 + 1));
  }
  bool direct = true;""",
        ),
        # (2) dequant_reg: revert the per-element tail to the
        # probe-wheel text (r212l + probe 9)
        (
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
            byte, int(goff ? kc1[slot] : kc0[slot]) - 7);
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
        ),
        # (3) dequant_vslm: dtype-split scale prep
        (
            """  // r212m CODE-DOMAIN SCALES + r212m2 REGISTER TRIM: kc[r] (two
  // u32 registers) is the ONLY added live state — base_e derives
  // inline (int(code) - 7 for e4m3, int(code) - 1 for e2m1) and the
  // RARE slow element converts its own code on demand (e2m1:
  // out-of-band groups only; e4m3: payload-subnormal bytes only).
  uint8_t kc[8];
  bool fast_band = direct_bf16;
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    kc[r] = skip_scales ? uint8_t(127)
                        : uint8_t((gl[2 * r] >> gsh) & 0xFF);
    fast_band = fast_band && (VIsE4m3 ? detail::e4m3_band_ok(kc[r])
                                      : detail::e2m1_band_ok(kc[r]));
  }""",
            """  // r212m3 DTYPE SPLIT: e2m1 (mxfp4 V, mxk8v4 V) keeps the trimmed
  // direct-bf16 machinery — kc[r] (two u32 registers) is the only
  // added live state, base_e derives inline (int(code) - 1), the rare
  // out-of-band element converts its own code on demand. e4m3
  // (mxfp8 V) restores the r212l float-vector prep and tail — the
  // direct-bf16 gain did not cover the restructured tail's cost
  // (see dequant_reg's r212m3 note).
  uint8_t kc[8];
  bool fast_band = false;
  float sc[8] = {};
  if constexpr (VIsE4m3) {
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      sc[r] = skip_scales ? 1.f : e8m0_to_scale(
          uint8_t((gl[2 * r] >> gsh) & 0xFF));
    }
  } else {
    bool fb = direct_bf16;
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      kc[r] = skip_scales ? uint8_t(127)
                          : uint8_t((gl[2 * r] >> gsh) & 0xFF);
      fb = fb && detail::e2m1_band_ok(kc[r]);
    }
    fast_band = fb;
    if (!fast_band || probe_val) {
      CUTE_UNROLL
      for (int r = 0; r < 8; ++r) {
        sc[r] = skip_scales ? 1.f : e8m0_to_scale(kc[r]);
      }
    }
  }""",
        ),
        # (4) dequant_vslm: dtype-split walk tail
        (
            """    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    // r212m DIRECT-BF16 fast path: e2m1 covers every in-band nibble
    // (zero nibbles return +-0); e4m3 covers only ee != 0 bytes —
    // payload subnormals fall back to the float path below.
    if (!probe_val && fast_band &&
        (!VIsE4m3 || ((b >> 3) & 0xF) != 0)) {
      uint16_t bits = VIsE4m3
          ? detail::e4m3_scaled_bf16(b, int(kc[r]) - 7)
          : detail::e2m1_scaled_bf16(
                uint8_t(nib_phase ? (b >> 4) : (b & 0xF)),
                int(kc[r]) - 1);
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
        probe_val ? val
                  : val * (skip_scales ? 1.f : e8m0_to_scale(kc[r])));""",
            """    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    if constexpr (!VIsE4m3) {
      // r212m DIRECT-BF16 fast path (e2m1 only): every in-band nibble
      // is covered (zero nibbles return +-0); out-of-band groups fall
      // back to the float path below.
      if (!probe_val && fast_band) {
        uint16_t bits = detail::e2m1_scaled_bf16(
            uint8_t(nib_phase ? (b >> 4) : (b & 0xF)), int(kc[r]) - 1);
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(i) = o;
        continue;
      }
    }
    // r212m probe 9: sever ONLY the decode tail (assembly + scale
    // mul); the SLM reads, extracts and fragment writes stay.
    float val = probe_val
        ? float(b)
        : (VIsE4m3 ? detail::e4m3_to_float(b)
                   : e2m1_to_float(uint8_t(
                         nib_phase ? (b >> 4) : (b & 0xF))));

    if constexpr (VIsE4m3) {
      dst(i) = typename DstFrag::value_type(
          probe_val ? val : val * sc[r]);
    } else {
      dst(i) = typename DstFrag::value_type(
          probe_val ? val
                    : val * (skip_scales ? 1.f : e8m0_to_scale(kc[r])));
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
    print("P212M3_PATCH_OK")


if __name__ == "__main__":
    main()
