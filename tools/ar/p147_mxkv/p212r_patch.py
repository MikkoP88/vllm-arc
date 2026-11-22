#!/usr/bin/env python3
"""p212r_patch.py -- r212r: NO-UNROLL probe on the r212q fast loops.

Evidence chain: r212o falsified branches; r212p priced the tail
machinery free (m11 == m9); r212q's u32 domain recovered only -3.2%
(717.7 vs 741.8) of the ~361 us bit-assembly pool. The surviving
~337 us is structural. The r212q fast loop is 32x unrolled (~12 ops
per element -> ~400 instructions) vs the light probe's ~100; the
leading suspects are unrolled CODE SIZE (icache), register-pressure
serialization, or chain ILP.

r212r isolates code size / cross-element interleave: the SAME u32
body under a RUNTIME loop (no CUTE_UNROLL). Loop body ~15
instructions, dst.size() is the immediate bound, index math (r, j)
becomes runtime shifts (cheap, ALU is not the wall).
  big drop  -> unroll class confirmed; next = tuned partial unroll
  flat     -> code size innocent; next = SWAR pairing (r212s)
Bit-identical by construction. Scope: the two e2m1 fast loops only.
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) vslm fast loop: drop the unroll
        (
            """    if (!probe_val && fast_band) {
      CUTE_UNROLL
      for (int i = 0; i < dst.size(); ++i) {
        int r = (i >> 1) & 7;
        int j = (i & 1) + 2 * (i >> 4);
        // r212q U32-DOMAIN ASSEMBLY: one fused native shift for the
        // nibble (byte extract folded away), u32 field math, single
        // 2-byte touch at the store.
        uint32_t nib = (u[r] >> (8 * j + 4 * nib_phase)) & 0xFu;""",
            """    if (!probe_val && fast_band) {
      // r212r NO-UNROLL PROBE: the identical r212q body under a
      // runtime loop — prices the 32x-unrolled code size (icache /
      // cross-element interleave) against the m11-class light loops.
      for (int i = 0; i < dst.size(); ++i) {
        int r = (i >> 1) & 7;
        int j = (i & 1) + 2 * (i >> 4);
        // r212q U32-DOMAIN ASSEMBLY: one fused native shift for the
        // nibble (byte extract folded away), u32 field math, single
        // 2-byte touch at the store.
        uint32_t nib = (u[r] >> (8 * j + 4 * nib_phase)) & 0xFu;""",
        ),
        # (2) dtile fast loop: drop the unroll
        (
            """    if (!probe_val && fast_band) {
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
        int goff = (off + org_emod) >> 5;
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        // r212q U32-DOMAIN ASSEMBLY (see dequant_vslm).""",
            """    if (!probe_val && fast_band) {
      // r212r NO-UNROLL PROBE (see dequant_vslm).
      for (int dv = 0; dv < dst.size(); ++dv) {
        int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
        int goff = (off + org_emod) >> 5;
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        // r212q U32-DOMAIN ASSEMBLY (see dequant_vslm).""",
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
    print("P212R_PATCH_OK")


if __name__ == "__main__":
    main()
