#!/usr/bin/env python3
"""p212v_patch.py -- r212v: K-RESTORE RUNG (probe 17).

The r212u attribution correction: every m-series cell (9, 11-16)
severs the K walk too (GEMM 1's `vlane_probe >= 6` const-fills
tSrK; GEMM 2's `>= 2 && != 4,5,6` const-fills tArV). So the pool
m0-m11 (~338 us) is the K walk's full cost PLUS the V fast-light
delta — never a V-only quantity.

Probe 17 = K REAL + V LIGHT (the m11 tail), the missing cell:
  k_lane = m17 - m11  (the K walk's real cost; V light both arms)
  v_pool = m0 - m17   (the V fast-light delta; K real both arms)
  k_lane + v_pool = m0 - m11 (telescoping check)
  k_lane > 0.7*pool -> K DOMINANT -> attack the K walk (dtile
    u32/SWAR/window surgery); V is already at its floor
  v_pool > 0.7*pool -> V dominant after all
  split -> attack the bigger half first
Three mainloop edits, ZERO dequant edits (mode 17 falls to the
plain light tail; dtile keeps probe_light false so the K walk runs
its real fast loop; m8 e4m3 lanes byte-identical).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_paged_decode_mainloop.hpp": [
        # (1) K const-fill gate: exclude 17 so the K walk runs REAL
        # (ASCII-only anchor: the comment line carries an em-dash)
        (
            """        if (params.vlane_probe >= 6) {""",
            """        if (params.vlane_probe >= 6 && params.vlane_probe != 17) {
          // r212v: 17 is EXEMPT from the r200g const-K sever (probe
          // 17 = K real + V light; splits the m0-m11 pool into K vs
          // V; the K walk below runs its REAL fast loop).""",
        ),
        # (2) V const-fill gate: exclude 17 so the V walk runs the
        # vslm call (which takes the LIGHT tail for 17)
        (
            """        if (params.vlane_probe >= 2 && params.vlane_probe != 4 &&
            params.vlane_probe != 5 && params.vlane_probe != 6) {""",
            """        if (params.vlane_probe >= 2 && params.vlane_probe != 4 &&
            params.vlane_probe != 5 && params.vlane_probe != 6 &&
            params.vlane_probe != 17) {""",
        ),
        # (3) vslm call site: extend ONLY the vslm light range to 17
        # (the `== 5` prefix disambiguates from dtile's identical
        # tail; dtile keeps <= 16 so K stays real for 17)
        (
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              params.vlane_probe >= 11 && params.vlane_probe <= 16,
              params.vlane_probe);""",
            """              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              (params.vlane_probe >= 11 && params.vlane_probe <= 16) ||
                  params.vlane_probe == 17,
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
    print("P212V_PATCH_OK")


if __name__ == "__main__":
    main()
