#!/usr/bin/env python3
"""p212y_patch.py -- r212y: ATTRACTOR RE-ROLL (mainloop only).

r212x closed the reorder hypothesis (deletion neutral, canary clean)
and left the m8 +34% GEMM-1 codegen attractor as the dominant
confound: three consecutive wheels (r212v/w/x) regressed m8 by the
same +34 after edits to the probe-gate text inside the GEMM regions,
while the clean r212u wheel measured the whole m0-m11 pool at 338 us
(vs attractor k_lane ~600). r212y re-rolls the dice with ZERO
semantic change: the K/V sever-gate comparison chains hoist to
precomputed bools before the K loop, and both gates read the bools.
The dead `if (false && probe==18)` block stays (unreachable code
emits nothing). Cells 0/11/17 + m8 oracle:
  m8 in band  -> CLEAN WHEEL: k_lane becomes the calibrated K-lane
                 cost; compare vs 338 (r212u pool) and 600 (attractor)
  m8 still +34 -> attractor held; k_lane directional only
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

ANCHOR_LOOP = """    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {"""

HOIST = """    /* Main loop, blocked in k. */
    // r212y ATTRACTOR RE-ROLL: the probe gates' comparison chains in
    // the GEMM regions pinned a bad codegen layout for three
    // consecutive wheels (m8 +33.8/+34.1/+34.5 pct after r212v's
    // null gate edit; k_lane 595.5/600.1 vs the clean r212u pool
    // 338). Hoist the sever conditions to precomputed bools --
    // semantically identical, fresh layout dice for both regions.
    const int vprobe = params.vlane_probe;
    const bool kp_sever = vprobe >= 6 && vprobe != 17 && vprobe != 20;
    const bool vp_sever =
        vprobe >= 2 && vprobe != 4 && vprobe != 5 && vprobe != 6 &&
        vprobe != 17;
    for (int K = blk_k0; K < blk_k1; K++) {"""

ANCHOR_KGATE = """        if (params.vlane_probe >= 6 && params.vlane_probe != 17 &&
            params.vlane_probe != 20) {"""

KGATE = """        if (kp_sever) {"""

ANCHOR_VGATE = """        if (params.vlane_probe >= 2 && params.vlane_probe != 4 &&
            params.vlane_probe != 5 && params.vlane_probe != 6 &&
            params.vlane_probe != 17) {"""

VGATE = """        if (vp_sever) {"""


def main():
    path = SRC / "mx_paged_decode_mainloop.hpp"
    text = path.read_text()
    if text.count("r212y ATTRACTOR RE-ROLL") != 0:
        print("PATCH_ALREADY_APPLIED (r212y ATTRACTOR RE-ROLL present)")
        sys.exit(1)
    for name, anchor in (("LOOP", ANCHOR_LOOP), ("KGATE", ANCHOR_KGATE),
                         ("VGATE", ANCHOR_VGATE)):
        n = text.count(anchor)
        if n != 1:
            print(f"PATCH_ANCHOR_MISS {name}: count={n}")
            sys.exit(1)
    text = text.replace(ANCHOR_LOOP, HOIST)
    text = text.replace(ANCHOR_KGATE, KGATE)
    text = text.replace(ANCHOR_VGATE, VGATE)
    path.write_text(text)
    print("PATCHED mx_paged_decode_mainloop.hpp (3 edits)")
    print("P212Y_PATCH_OK")


if __name__ == "__main__":
    main()
