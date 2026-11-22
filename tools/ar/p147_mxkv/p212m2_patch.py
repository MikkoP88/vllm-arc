#!/usr/bin/env python3
"""p212m2_patch.py -- r212m2 register-pressure trim on the r212m tree.

The r212m validation showed the fast path strictly better same-wheel
(+12.6/+13.5% at 32K) but this wheel's mxfp8 FLOAT path regressed +29%
vs r212l — layout dice plus a structural suspect: the e4m3 lanes keep
kc[8]+be[8]+sc[8] live in the hot loop (the e2m1 lane, which skips sc
when in-band, is the one that GAINED). This trim removes the be arrays
(base_e derives inline: int(code)-7 / int(code)-1) and the vslm sc
vector (the rare slow element converts its own code on demand; kc[8]
in 2 registers is the only added live state).

Also closes the paired-u32-assembly idea on layout grounds: all of a
WI's elements share one nibble phase (vslm: t1&1; dtile: t0), so the
two nibbles of a byte always belong to DIFFERENT WIs — no intra-WI
pairing exists; cross-WI pairing is the closed staging/reorder class.
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # dequant_reg: drop the be arrays
        (
            """  int be0[8], be1[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    be0[r] = int(kc0[r]) - 7;
    be1[r] = int(kc1[r]) - 7;
  }
  bool direct = true;""",
            """  // r212m2 REGISTER TRIM: no be arrays — base_e derives inline
  // from the code select (int(code) - 7); 16 fewer live ints in the
  // hot loop (the r212m mxfp8 float-path regression suspect).
  bool direct = true;""",
        ),
        (
            """        uint16_t bits = detail::e4m3_scaled_bf16(
            byte, goff ? be1[slot] : be0[slot]);""",
            """        uint16_t bits = detail::e4m3_scaled_bf16(
            byte, int(goff ? kc1[slot] : kc0[slot]) - 7);""",
        ),
        # dequant_vslm: drop the be AND sc arrays
        (
            """  // r212m CODE-DOMAIN SCALES: kc[r] feeds the direct-bf16 fast path
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
        ),
        (
            """      uint16_t bits = VIsE4m3
          ? detail::e4m3_scaled_bf16(b, be[r])
          : detail::e2m1_scaled_bf16(
                uint8_t(nib_phase ? (b >> 4) : (b & 0xF)), be[r]);""",
            """      uint16_t bits = VIsE4m3
          ? detail::e4m3_scaled_bf16(b, int(kc[r]) - 7)
          : detail::e2m1_scaled_bf16(
                uint8_t(nib_phase ? (b >> 4) : (b & 0xF)),
                int(kc[r]) - 1);""",
        ),
        (
            """    dst(i) = typename DstFrag::value_type(
        probe_val ? val : val * sc[r]);""",
            """    dst(i) = typename DstFrag::value_type(
        probe_val ? val
                  : val * (skip_scales ? 1.f : e8m0_to_scale(kc[r])));""",
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
    print("P212M2_PATCH_OK")


if __name__ == "__main__":
    main()
