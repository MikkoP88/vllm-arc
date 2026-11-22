#!/usr/bin/env python3
"""p212w_patch.py -- r212w: K-DECOMPOSITION RUNGS (probes 18/20).

r212v established (directionally): the m0-m11 pool is the K WALK
(k_lane = 104% of pool) and V is at its floor (v_pool ~= 0). The
r212v wheel was dice-regressed (m8 +33.8%), so r212w re-baselines
on a fresh wheel AND splits the K lane into its components:

  K lane = (a) K byte loads + window staging + walk extracts
         + (b) dequant_dtile decode (scales + assembly + stores)
         + (c) reorder(tKrDeq, tSrK) layout change into the MMA
             operand + gemm interleave

  m18 = K real + reorder SEVERED (dtile runs fully; tSrK
        const-filled after) -> m17-m18 = (c) the reorder share
  m20 = K real + dtile LIGHT TAIL + REAL reorder -> m20-m11 =
        (a)+(c); a = (m20-m11) - (m17-m18); b = (m18-m11) - a
  m0/m11/m17 ride as same-wheel anchors; m8 gates the wheel's
  absolutes (r212v FAILed at +33.8%).

Mainloop-only, three edits. The m18 severance sits inside the e2m1
else-branch (the e4m3 arm and its reorder are untouched -> m8
byte-identical).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_paged_decode_mainloop.hpp": [
        # (1) K const-fill gate: exempt 20 alongside 17 (ASCII-only
        # anchor: the if-line is unique)
        (
            """        if (params.vlane_probe >= 6 && params.vlane_probe != 17) {""",
            """        if (params.vlane_probe >= 6 && params.vlane_probe != 17 &&
            params.vlane_probe != 20) {""",
        ),
        # (2) m18: sever ONLY the reorder after a REAL dtile run
        (
            """          reorder(tKrDeq, tSrK);""",
            """          if (params.vlane_probe == 18) {
            // r212w: sever ONLY the reorder (layout change) — dtile
            // ran fully above; tSrK const-filled so GEMM 1 consumes.
            // m17-m18 = the reorder's share of the K lane.
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < tSrK.size(); ++i) {
              tSrK(i) = typename decltype(tSrK)::value_type(1.f);
            }
          } else {
            reorder(tKrDeq, tSrK);
          }""",
        ),
        # (3) dtile call site: extend the dtile light range to 20
        # (the tKDgKD_cache prefix disambiguates from vslm's tail;
        # vslm keeps (>=11 && <=16) || == 17 — m20's V is const)
        (
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 16,
              params.vlane_probe);""",
            """              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              (params.vlane_probe >= 11 && params.vlane_probe <= 16) ||
                  params.vlane_probe == 20,
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
    print("P212W_PATCH_OK")


if __name__ == "__main__":
    main()
