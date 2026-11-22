#!/usr/bin/env python3
"""p212x2_guard.py -- r212x part 2: NESTED-FORM GUARD in dequant_dtile.

Part 1 (p212x_patch.py) swapped the dtile dst to tSrK and deleted the
reorder; the first attempt added a dedicated MMA-form walk branch --
unnecessary and wrong (dv>>3): the MMA B operand's tv is the SAME walk
law as the pinned staged form, only nested differently. Rank-2
((2,8),((2,8),(1,4))) with strides ((1e1,1e0),((8e0,2e1),(0,16e1)):
idx2crd through the nested val gives dv = v0 + 2*v1 + 16*v2 (the (1)
mode is stride-0 degenerate), row = t_b + 8*v0, elem = t_a + 2*v1 +
16*v2 -- byte = v1 + 8*v2, nibble phase t_a, group v2>>1: exactly the
staged body's own arithmetic (its loops compute v2 = dv >> 4,
jb = v1 + 8*v2). The r212w reorder cost (595 us = 100% of the K lane)
was the stock generic scalar implementation, not the permutation --
the r200a/r200b story. So the staged body runs verbatim into tSrK and
the ONLY edit is the form pin: rank-3 keeps the full assert (old
copy-D dst forms stay protected); rank-2 gets a positive shape pin.
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

ANCHOR = """    detail::assert_staged_tv_form(tv);
    using TV = decltype(tv);"""

GUARD = """    // r212x NESTED-FORM GUARD: dst may be the MMA B operand (tSrK --
    // the K reorder deletion). Its tv is rank-2 with the rest mode
    // folded into val as a stride-0 degenerate: ((2,8),((2,8),(1,4)))
    // -- the SAME walk law as the pinned rank-3 form (dv = v0 + 2*v1
    // + 16*v2; row = t_b + 8*v0, elem = t_a + 2*v1 + 16*v2), so the
    // staged body below applies verbatim. Only the form pin differs.
    if constexpr (cute::rank_v<decltype(tv)> == 3) {
      detail::assert_staged_tv_form(tv);
    } else {
      constexpr auto vsh = cute::shape<1>(decltype(tv){});
      static_assert(
          cute::rank_v<decltype(vsh)> == 2 &&
              cute::size(cute::get<0>(vsh)) == 16 &&
              cute::size(vsh) % 16 == 0,
          "r212x: unexpected rank-2 tv form (expected the nested "
          "((2,8),((2,8),(1,4))) MMA B operand)");
    }
    using TV = decltype(tv);"""


def main():
    path = SRC / "mx_dequant.hpp"
    text = path.read_text()
    if text.count("r212x NESTED-FORM GUARD") != 0:
        print("PATCH_ALREADY_APPLIED (r212x NESTED-FORM GUARD present)")
        sys.exit(1)
    if text.count(ANCHOR) != 1:
        print(f"PATCH_ANCHOR_MISS mx_dequant.hpp: count={text.count(ANCHOR)}")
        sys.exit(1)
    path.write_text(text.replace(ANCHOR, GUARD))
    print("PATCHED mx_dequant.hpp (1 edit)")
    print("P212X2_PATCH_OK")


if __name__ == "__main__":
    main()
