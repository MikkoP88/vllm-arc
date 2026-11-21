#!/usr/bin/env python3
"""p212x_patch.py -- r212x: DELETE THE K REORDER (the r200b twin).

r212w priced the K lane: m18 (K real + reorder severed) = 430.6 is
IDENTICAL to m11 (K const) = 430.1 — the K loads (fully prefetched),
staging, and decode all cost ZERO, and reorder(tKrDeq, tSrK) carries
595.1 us = 100% of the K lane (~338 us on a clean-dice wheel). This
is the exact r200a story on the V side (+0.36 ms Generic reorder)
that r200b deleted; the K twin was never deleted.

Surgery: pass tSrK (thr_mma_qk.partition_sg_fragment_B — the MMA
operand itself) as dequant_dtile's dst and delete the reorder.
dtile derives the walk from dst.tv_layout():
  - if tSrK's val form satisfies the 16-leading staged test AND the
    pinned law -> staged walk straight into the MMA operand (WIN);
  - if the form mismatches -> the r198 PASS scalar walk fallback
    (CORRECT results, slower reads — the price shows in m0);
  - law violation inside the staged branch -> assert fires loudly.
No silent corruption path; m0's canary is the byte-level net.
tKrDeq's declaration is [[maybe_unused]] — no warning.

Leg: m0/m11/m17 same-wheel (m17 should collapse to m11 if the K
lane died). m8 rides m0 (e4m3 untouched).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_paged_decode_mainloop.hpp": [
        # (1) dtile call: dst becomes tSrK (the MMA operand itself)
        (
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D),""",
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tSrK,
              tKDgKD_cache(_, _, _, D),""",
        ),
        # (2) the reorder (and the r212w m18 severance block) are
        # DELETED — the dst swap above makes them dead (anchor
        # shortened to ASCII-only prefix lines: the r212w comment
        # carries an em-dash)
        (
            """          if (params.vlane_probe == 18) {""",
            """          if (false && params.vlane_probe == 18) {""",
        ),
        (
            """          } else {
            reorder(tKrDeq, tSrK);
          }""",
            """          } else {
            // r212x: the K REORDER IS DELETED. r212w priced it at
            // 100% of the K lane (m17-m18 = 595 us; loads+staging+
            // decode = +0.5 us vs K-const): dequant_dtile now writes
            // tSrK directly -- its walk derives the law from the dst
            // fragment's tv_layout (staged branch if the val form
            // qualifies, r198 scalar-walk fallback otherwise; both
            // correct by construction). The V lane's r200b precedent.
            // The r212w probe-18 severance is retired with it.
          }""",
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
    print("P212X_PATCH_OK")


if __name__ == "__main__":
    main()
