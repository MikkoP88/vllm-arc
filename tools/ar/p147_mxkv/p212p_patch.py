#!/usr/bin/env python3
"""p212p_patch.py -- r212p: LIGHT probe-9 tail (vlane_probe 11) on the
r212o tree.

Why: r212o falsified the branch hypothesis and left a puzzle -- the
probe-9 cell (380.6 us) carries a HEAVIER per-element tail (float
convert + inline e8m0_to_scale + mul + f32->bf16 convert) than the fast
path's ~6 u16 bit ops, yet sits 361 us BELOW the fast path. The
residual is not op count and not branches. Remaining suspects: u8/u16
scalar bit-surgery codegen, dependency-chain serialization, and the
per-element dst(i) fragment stores (both paths pay those).

Probe 11 is the deciding instrument: it severs MORE than probe 9 --
the light tail keeps ONLY the byte/nibble extracts + a float convert +
the fragment store, and the prep (scale-code extraction + band chain)
is severed too. Then on the same wheel:
    m0 - m11  = the FULL decode machinery (assembly + scales + prep)
    m9 - m11  = the heavy tail's extra machinery (scale re-derivation)
    m11 - c8  = extracts + stores + walk loads + softmax/P + staging
If m11 ~= m9 the extracts+stores+walk dominate the tail lanes and
store packing is the lever; if m11 ~= c8+small the decode machinery
dominates and the bit-surgery codegen is the lever.

Scope: the two e2m1 sites only (dequant_vslm, dequant_dtile) -- the
e4m3 dequant_reg keeps its byte-identical source (m8 discipline); m8's
m11 cell is therefore partial (its K lane rides the full decode) and
the leg prices m4 only.
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) vslm signature: append probe_light
        (
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true) {""",
            """    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false) {""",
        ),
        # (2) vslm prep: sever under probe_light
        (
            """  uint8_t kc[8];
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
    // r212o: the e2m1 sc vector was dead (the slow tail derives its
    // scale inline from kc[r]) — removed; only the e4m3 arm reads sc.
  }""",
            """  uint8_t kc[8];
  bool fast_band = false;
  float sc[8] = {};
  // r212p LIGHT PROBE (probe 11) severs the prep too — the light
  // tail needs no scale codes, no band, no float vector.
  if (!probe_light) {
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
      // r212o: the e2m1 sc vector was dead (the slow tail derives
      // its scale inline from kc[r]) — removed; only the e4m3 arm
      // reads sc.
    }
  }""",
        ),
        # (3) vslm: the light loop after the u loads
        (
            """  uint32_t u[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    u[r] = *reinterpret_cast<const uint32_t*>(
        slm + kr[r] * slm_pitch + qbase);
  }
  // r212o LOOP SPLIT: the per-element runtime branch""",
            """  uint32_t u[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    u[r] = *reinterpret_cast<const uint32_t*>(
        slm + kr[r] * slm_pitch + qbase);
  }
  // r212p LIGHT PROBE TAIL (probe 11): extracts + float convert +
  // store ONLY — severs more than probe 9 (whose tail kept the nibble
  // decode arithmetic and scale re-derivation). Prices the
  // extracts/store floor vs the decode machinery (r212o's puzzle).
  if (probe_light) {
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
  }
  // r212o LOOP SPLIT: the per-element runtime branch""",
        ),
        # (4) dtile signature: append probe_light
        (
            """    DstFrag& dst,
    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true) {""",
            """    DstFrag& dst,
    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false) {""",
        ),
        # (5) dtile prep: sever scale codes + band under probe_light
        (
            """    uint8_t c00 = uint8_t(scales(row0, grp0));
    uint8_t c01 = uint8_t(scales(row0, grp0 + 1));
    uint8_t c10 = uint8_t(scales(row0 + 8, grp0));
    uint8_t c11 = uint8_t(scales(row0 + 8, grp0 + 1));
    bool fast_band = direct_bf16 && detail::e2m1_band_ok(c00) &&
                     detail::e2m1_band_ok(c01) &&
                     detail::e2m1_band_ok(c10) &&
                     detail::e2m1_band_ok(c11);
    int be00 = int(c00) - 1, be01 = int(c01) - 1;
    int be10 = int(c10) - 1, be11 = int(c11) - 1;""",
            """    // r212p LIGHT PROBE (probe 11): prep severed — the light tail
    // needs no scale codes and no band.
    uint8_t c00 = 127, c01 = 127, c10 = 127, c11 = 127;
    bool fast_band = false;
    int be00 = 0, be01 = 0, be10 = 0, be11 = 0;
    if (!probe_light) {
      c00 = uint8_t(scales(row0, grp0));
      c01 = uint8_t(scales(row0, grp0 + 1));
      c10 = uint8_t(scales(row0 + 8, grp0));
      c11 = uint8_t(scales(row0 + 8, grp0 + 1));
      fast_band = direct_bf16 && detail::e2m1_band_ok(c00) &&
                  detail::e2m1_band_ok(c01) &&
                  detail::e2m1_band_ok(c10) &&
                  detail::e2m1_band_ok(c11);
      be00 = int(c00) - 1;
      be01 = int(c01) - 1;
      be10 = int(c10) - 1;
      be11 = int(c11) - 1;
    }""",
        ),
        # (6) dtile: the light loop before the r212o split
        (
            """    // r212o LOOP SPLIT (see dequant_vslm): the per-element runtime
    // branch hoists to per-d-tile; the fast loop is branch-free and
    // drops the jb decode comments (same arithmetic verbatim).
    if (!probe_val && fast_band) {""",
            """    // r212p LIGHT PROBE TAIL (probe 11): extracts + float convert +
    // store ONLY (see dequant_vslm). The st[] window loads above stay
    // — they ARE the walk's load path.
    if (probe_light) {
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;
        dst(dv) = typename DstFrag::value_type(
            float(uint8_t(t0 ? (b >> 4) : (b & 0xF))));
      }
      return;
    }
    // r212o LOOP SPLIT (see dequant_vslm): the per-element runtime
    // branch hoists to per-d-tile; the fast loop is branch-free and
    // drops the jb decode comments (same arithmetic verbatim).
    if (!probe_val && fast_band) {""",
        ),
    ],
    "mx_paged_decode_mainloop.hpp": [
        # (7) dtile call site: pass probe 11
        (
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0);""",
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe == 11);""",
        ),
        # (8) vslm call site: pass probe 11
        (
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0);""",
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe == 11);""",
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
    print("P212P_PATCH_OK")


if __name__ == "__main__":
    main()
