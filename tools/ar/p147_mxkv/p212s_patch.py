#!/usr/bin/env python3
"""p212s_patch.py -- r212s: INT-OP PRICING LADDER (probes 12/13).

Evidence chain: r212o falsified branches; r212p priced the tail
machinery free (m11 == m9); r212q's u32 domain bought only -3.2%;
r212r's no-unroll was a PERFECT null (-0.0%) — code size and loop
form are innocent. The wall lives in the per-element assembly work
itself: m0 - m11 = ~337 us buys ~8 extra scalar int ops + a 2-byte
memcpy per element.

The ladder prices the ops directly: probe 12 = the m11 light tail +
4 dependent int ops per element; probe 13 = + 8 ops (a rung = 2
shifts + xor + and, chained through the value so nothing folds).
Same-wheel cells m11/m12/m13/m0 give:
    slope  = (m13 - m12) / 4 us per int op
    ops share = slope * 8 (the m0-m11 op delta)
    residual = (m0 - m11) - slope * 8 = the memcpy/store/other pool
ops share >> residual  -> algebraic op-reduction path (r212t)
residual >> ops share  -> store/memcpy path (fragment packing)
The disasm route (parked: offload-bundle extraction) would confirm;
this measures the same fact for one build cycle.

Scope: the two e2m1 light-probe loops only (the m0 fast path text is
byte-identical; dequant_reg untouched — m8 discipline).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) vslm signature: append probe_kops
        (
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false) {""",
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_kops = 0) {""",
        ),
        # (2) dtile signature: append probe_kops
        (
            """    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false) {""",
            """    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_kops = 0) {""",
        ),
        # (3) vslm light loop: the ladder on the extracted nibble
        (
            """  if (probe_light) {
    CUTE_UNROLL
    for (int i = 0; i < dst.size(); ++i) {
      int r = (i >> 1) & 7;
      int j = (i & 1) + 2 * (i >> 4);
      uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
      dst(i) = typename DstFrag::value_type(
          float(VIsE4m3 ? b
                        : uint8_t(nib_phase ? (b >> 4) : (b & 0xF))));
    }
    return;
  }""",
            """  if (probe_light) {
    CUTE_UNROLL
    for (int i = 0; i < dst.size(); ++i) {
      int r = (i >> 1) & 7;
      int j = (i & 1) + 2 * (i >> 4);
      uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
      // r212s INT-OP LADDER (probes 12/13): each rung adds 4
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
      dst(i) = typename DstFrag::value_type(float(uint8_t(nib)));
    }
    return;
  }""",
        ),
        # (4) dtile light loop: the ladder
        (
            """    if (probe_light) {
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;
        dst(dv) = typename DstFrag::value_type(
            float(uint8_t(t0 ? (b >> 4) : (b & 0xF))));
      }
      return;
    }""",
            """    if (probe_light) {
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;
        // r212s INT-OP LADDER (see dequant_vslm).
        uint32_t nib = t0 ? (b >> 4) : (b & 0xF);
        for (int k = 0; k + 3 <= probe_kops; k += 3) {
          uint32_t t = nib;
          nib = ((t << 1) ^ (t >> 1)) & 0xFu;
        }
        dst(dv) = typename DstFrag::value_type(float(uint8_t(nib)));
      }
      return;
    }""",
        ),
    ],
    "mx_paged_decode_mainloop.hpp": [
        # (5) dtile call site: light range + kops (the tKDgKD_cache
        # prefix disambiguates from the vslm site's identical tail)
        (
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe == 11);""",
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 13,
              params.vlane_probe == 12 ? 4
                  : (params.vlane_probe == 13 ? 8 : 0));""",
        ),
        # (6) vslm call site: light range + kops
        (
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe == 11);""",
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 13,
              params.vlane_probe == 12 ? 4
                  : (params.vlane_probe == 13 ? 8 : 0));""",
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
    print("P212S_PATCH_OK")


if __name__ == "__main__":
    main()
