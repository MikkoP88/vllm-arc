#!/usr/bin/env python3
"""p212t_patch.py -- r212t: STORE-SHAPE SPLIT (probes 14/15).

The r212s ladder came back FLAT (m11 379.6 / m12 379.2 / m13 380.2):
scalar int ops are FREE. Every arithmetic theory is now eliminated
(branches r212o, loop form r212r, domain r212q -24 us, int ops r212s,
converts r212p). The m0-m11 pool (~337 us) can only be:
  (a) the 2-byte memcpy into the 2-byte local + store-from-local
      (the fast loop's store shape), or
  (b) the ternary SELECT inside e2m1_scaled_bf16_u32 (the ladder's
      dummy ops had no select), or
  (c) an assembly-specific interaction.
Probe 14 = light tail + the REAL helper call (full assembly incl.
the select, base 120) with the CONVERT-shaped store -> prices (b)+(c).
Probe 15 = light tail + trivial bits with the MEMCPY-shaped store
-> prices (a).
m14 - m11 ~= 0 AND m15 - m11 big  -> MEMCPY CONFIRMED -> store-
  constructor fix next (bitcast/from_bits, no 2-byte local).
m14 - m11 big AND m15 - m11 ~= 0  -> SELECT/assembly -> algebraic
  select removal next.
Both run on the same wheel with m11/m0 rerun (same-binary law).
Scope: the two e2m1 light loops only (fast path byte-identical).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) vslm signature: probe_kops -> probe_mode (the int v_base
        # prefix disambiguates from dtile's identical tail)
        (
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_kops = 0) {""",
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_mode = 0) {""",
        ),
        # (2) vslm light loop: mode dispatch with rungs 14/15
        (
            """      // r212s INT-OP LADDER (probes 12/13): each rung adds 4
      // dependent scalar int ops (2 shifts + xor + and) chained
      // through the value so nothing folds; the convert/store tail
      // is the m11 shape. Prices int-op throughput against the
      // m0-m11 assembly pool.
      uint32_t nib = VIsE4m3
          ? uint32_t(b)
          : uint32_t(nib_phase ? (b >> 4) : (b & 0xF));
      for (int k = 0; k + 3 <= probe_kops; k += 3) {
        uint32_t t = nib;
        nib = ((t << 1) ^ (t >> 1)) & 0xFu;
      }
      dst(i) = typename DstFrag::value_type(float(uint8_t(nib)));""",
            """      // r212s/r212t PROBE LADDER: 11 = light tail; 12/13 = +4/+8
      // dependent scalar int ops (2 shifts + xor + and, chained so
      // nothing folds); 14 = + the REAL helper call (full assembly
      // incl. the ternary select, base 120) with the convert-shaped
      // store; 15 = trivial bits with the memcpy-shaped store. The
      // rungs split the m0-m11 pool into assembly/select vs store
      // shape (r212s: int ops alone are FREE — the ladder is flat).
      uint32_t nib = VIsE4m3
          ? uint32_t(b)
          : uint32_t(nib_phase ? (b >> 4) : (b & 0xF));
      int kops = probe_mode == 12 ? 4 : (probe_mode == 13 ? 8 : 0);
      for (int k = 0; k + 3 <= kops; k += 3) {
        uint32_t t = nib;
        nib = ((t << 1) ^ (t >> 1)) & 0xFu;
      }
      if (probe_mode == 15) {
        typename DstFrag::value_type o;
        uint32_t bits = nib;
        __builtin_memcpy(&o, &bits, 2);
        dst(i) = o;
      } else {
        if (probe_mode == 14) {
          uint32_t bits = detail::e2m1_scaled_bf16_u32(nib, 120u);
          nib = bits & 0xFu;
        }
        dst(i) = typename DstFrag::value_type(float(uint8_t(nib)));
      }""",
        ),
        # (3) dtile signature: probe_kops -> probe_mode (CoordView
        # prefix disambiguates)
        (
            """    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_kops = 0) {""",
            """    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_mode = 0) {""",
        ),
        # (4) dtile light loop: mode dispatch with rungs 14/15
        (
            """        // r212s INT-OP LADDER (see dequant_vslm).
        uint32_t nib = t0 ? (b >> 4) : (b & 0xF);
        for (int k = 0; k + 3 <= probe_kops; k += 3) {
          uint32_t t = nib;
          nib = ((t << 1) ^ (t >> 1)) & 0xFu;
        }
        dst(dv) = typename DstFrag::value_type(float(uint8_t(nib)));""",
            """        // r212s/r212t PROBE LADDER (see dequant_vslm).
        uint32_t nib = t0 ? (b >> 4) : (b & 0xF);
        int kops = probe_mode == 12 ? 4 : (probe_mode == 13 ? 8 : 0);
        for (int k = 0; k + 3 <= kops; k += 3) {
          uint32_t t = nib;
          nib = ((t << 1) ^ (t >> 1)) & 0xFu;
        }
        if (probe_mode == 15) {
          typename DstFrag::value_type o;
          uint32_t bits = nib;
          __builtin_memcpy(&o, &bits, 2);
          dst(dv) = o;
        } else {
          if (probe_mode == 14) {
            uint32_t bits = detail::e2m1_scaled_bf16_u32(nib, 120u);
            nib = bits & 0xFu;
          }
          dst(dv) = typename DstFrag::value_type(float(uint8_t(nib)));
        }""",
        ),
    ],
    "mx_paged_decode_mainloop.hpp": [
        # (5) dtile call site: light range widened, mode passthrough
        (
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 13,
              params.vlane_probe == 12 ? 4
                  : (params.vlane_probe == 13 ? 8 : 0));""",
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 15,
              params.vlane_probe);""",
        ),
        # (6) vslm call site: light range widened, mode passthrough
        (
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 13,
              params.vlane_probe == 12 ? 4
                  : (params.vlane_probe == 13 ? 8 : 0));""",
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 15,
              params.vlane_probe);""",
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
    print("P212T_PATCH_OK")


if __name__ == "__main__":
    main()
