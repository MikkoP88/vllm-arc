#!/usr/bin/env python3
"""p212aa_revert.py -- revert r212aa (TEMPLATE-SPLIT) on the mainloop,
returning the tree to the exact r212y state. Reverses the ten edits of
p212aa_patch.py (anchor/replace swapped)."""
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p212aa_patch import (ANCHOR_BARRIER, ANCHOR_BOOLS, ANCHOR_DBG, ANCHOR_DTILE,
                          ANCHOR_P, ANCHOR_PREP, ANCHOR_REG, ANCHOR_SIG,
                          ANCHOR_STAGE, ANCHOR_VSLM, BARRIER, BOOLS, DBG, DTILE,
                          P, PREP, REG, SIG, SRC, STAGE, VSLM)


def main():
    path = SRC / "mx_paged_decode_mainloop.hpp"
    text = path.read_text(encoding="utf-8")
    if text.count("r212aa TEMPLATE-SPLIT") == 0:
        print("REVERT_NOT_NEEDED (no r212aa marker present)")
        sys.exit(1)
    # Post-patch text contains the NEW strings; count those, then
    # substitute back (roundtrip proven byte-identical in the dry-run).
    for name, new in (("SIG", SIG), ("BOOLS", BOOLS), ("REG", REG),
                      ("DTILE", DTILE), ("STAGE", STAGE),
                      ("BARRIER", BARRIER), ("DBG", DBG), ("P", P),
                      ("PREP", PREP), ("VSLM", VSLM)):
        n = text.count(new)
        if n != 1:
            print(f"REVERT_MISS {name}: count={n}")
            sys.exit(1)
    for new, anchor in ((SIG, ANCHOR_SIG), (BOOLS, ANCHOR_BOOLS),
                        (REG, ANCHOR_REG), (DTILE, ANCHOR_DTILE),
                        (STAGE, ANCHOR_STAGE), (BARRIER, ANCHOR_BARRIER),
                        (DBG, ANCHOR_DBG), (P, ANCHOR_P),
                        (PREP, ANCHOR_PREP), (VSLM, ANCHOR_VSLM)):
        text = text.replace(new, anchor)
    path.write_text(text, encoding="utf-8", newline="\n")
    print("REVERTED mx_paged_decode_mainloop.hpp (10 edits reversed)")
    print("P212AA_REVERT_OK")


if __name__ == "__main__":
    main()
