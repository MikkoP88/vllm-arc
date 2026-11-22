#!/usr/bin/env python3
"""p212u_patch.py -- r212u: VERBATIM-BODY rung (probe 16).

The r212t split came back BOTH CHEAP: full assembly + select (m14
379.6) and the memcpy store shape (m15 379.5) are each FREE in the
unrolled light tail — yet the fast loop pays +337 us. The components
are innocent INDIVIDUALLY in the light context.

The one difference no rung has replicated: the fast loop is a
RUNTIME loop (r212r) whose body reads RUNTIME-INDEXED local arrays
— kc[r] in vslm, st[v0][jb>>2] in dtile — while every probe rung
runs UNROLLED (compile-time indices -> SROA promotes arrays to
registers). This codebase documents the hazard in situ: "a runtime
index here would push st[] to scratch". Scratch loads per element
would be exactly a 300-us-class wall, and the r212q==r212r equality
is consistent with SROA failing to promote kc in BOTH builds (the
prep writes/reads kc[r] with a runtime loop index before the walk).

Probe 16 = the VERBATIM fast-loop body as a runtime loop inside the
light path (kc initialized to 120 under probe_light so the read
cannot be UB-folded away):
  m16 ~= m11  -> body innocent even runtime-indexed -> CONTEXT
                 (placement/branch) -> disasm route decides
  m16 ~= m0   -> BODY: the runtime-indexed array reads (scratch)
                 -> fix = constant-index everything (unroll prep +
                 walk) so the arrays fully promote
Same wheel reruns m11/m0 (same-binary law). m8 rides m0.
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) vslm: insert the mode-16 runtime loop before the
        # unrolled light loop (kc is function-scope, declared before
        # the prep guard — 120-initialized INSIDE the rung so the
        # runtime-indexed read cannot be UB-folded)
        (
            """  if (probe_light) {
    CUTE_UNROLL
    for (int i = 0; i < dst.size(); ++i) {
      int r = (i >> 1) & 7;
      int j = (i & 1) + 2 * (i >> 4);
      uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);""",
            """  if (probe_light) {
    // r212u VERBATIM-BODY RUNG (probe 16): the fast loop's exact
    // body as a RUNTIME loop (runtime-indexed kc[r] read included)
    // inside the light path — the only fast-loop feature no unrolled
    // rung replicates. kc is 120-initialized under probe_light so
    // the read cannot be UB-folded. Splits body-vs-context for the
    // m0-m11 pool after r212t's both-cheap split.
    if (probe_mode == 16) {
      CUTE_UNROLL
      for (int r = 0; r < 8; ++r) {
        kc[r] = uint8_t(120);
      }
      for (int i = 0; i < dst.size(); ++i) {
        int r = (i >> 1) & 7;
        int j = (i & 1) + 2 * (i >> 4);
        uint32_t nib = (u[r] >> (8 * j + 4 * nib_phase)) & 0xFu;
        uint32_t bits = detail::e2m1_scaled_bf16_u32(
            nib, uint32_t(int(kc[r]) - 1));
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(i) = o;
      }
      return;
    }
    CUTE_UNROLL
    for (int i = 0; i < dst.size(); ++i) {
      int r = (i >> 1) & 7;
      int j = (i & 1) + 2 * (i >> 4);
      uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);""",
        ),
        # (2) dtile: insert the mode-16 runtime loop (be-s stay 0
        # under probe_light — values do not price)
        (
            """    if (probe_light) {
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;""",
            """    if (probe_light) {
      // r212u VERBATIM-BODY RUNG (probe 16, see dequant_vslm): the
      // dtile fast body as a runtime loop — st[v0][jb >> 2] is read
      // with RUNTIME indices here (the unrolled rungs read it with
      // compile-time indices and promote).
      if (probe_mode == 16) {
        for (int dv = 0; dv < dst.size(); ++dv) {
          int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
          int goff = (off + org_emod) >> 5;
          int v0 = dv & 1;
          int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
          uint32_t nib =
              (st[v0][jb >> 2] >> ((jb & 3) * 8 + 4 * t0)) & 0xFu;
          uint32_t bits = detail::e2m1_scaled_bf16_u32(
              nib,
              uint32_t((dv & 1) ? (goff ? be11 : be10)
                                : (goff ? be01 : be00)));
          typename DstFrag::value_type o;
          __builtin_memcpy(&o, &bits, 2);
          dst(dv) = o;
        }
        return;
      }
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;""",
        ),
    ],
    "mx_paged_decode_mainloop.hpp": [
        # (3) dtile call site: widen the light range to 16
        (
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 15,
              params.vlane_probe);""",
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 16,
              params.vlane_probe);""",
        ),
        # (4) vslm call site: widen the light range to 16
        (
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 15,
              params.vlane_probe);""",
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 16,
              params.vlane_probe);""",
        ),
    ],
}


def main():
    for fname, plist in EDITS.items():
        path = SRC / fname
        text = path.read_text()
        for old, new in plist:
            if text.count(old) != 1:
                print(f"PATCH_ANCHOR_MISS {fname}: count={text.count(old)}")
                print(f"  anchor head: {old[:72]!r}")
                sys.exit(1)
            text = text.replace(old, new)
        path.write_text(text)
        print(f"PATCHED {fname} ({len(plist)} edits)")
    print("P212U_PATCH_OK")


if __name__ == "__main__":
    main()
