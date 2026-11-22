#!/usr/bin/env python3
"""p212z_revert.py -- revert r212z (DOUBLE-BUFFERED K DECODE) on the
mainloop, returning the tree to the exact r212y state (r212z was a
net-loss wheel: k_lane collapsed 600->112.4 proving the overlap, but
the runtime-parity fragment selection collapsed the whole GEMM-1
region -- m11 1107 vs 414, m8 +109.4%). Reverses the five edits of
p212z_patch.py by string substitution (anchor/replace swapped).
"""
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p212z_patch import (ANCHOR_CALL, ANCHOR_E4M3, ANCHOR_FILL, ANCHOR_GEMM,
                         ANCHOR_HEAD, CALL, E4M3, FILL, GEMM, HEAD, SRC)


def main():
    path = SRC / "mx_paged_decode_mainloop.hpp"
    text = path.read_text()
    if text.count("r212z DOUBLE-BUFFERED K DECODE") == 0:
        print("REVERT_NOT_NEEDED (no r212z marker present)")
        sys.exit(1)
    # Post-patch text contains the NEW strings (HEAD/FILL/...), not the
    # anchors -- count those, then substitute back.
    for name, new in (("HEAD", HEAD), ("FILL", FILL), ("E4M3", E4M3),
                      ("CALL", CALL), ("GEMM", GEMM)):
        n = text.count(new)
        if n != 1:
            print(f"REVERT_MISS {name}: count={n}")
            sys.exit(1)
    text = text.replace(HEAD, ANCHOR_HEAD)
    text = text.replace(FILL, ANCHOR_FILL)
    text = text.replace(E4M3, ANCHOR_E4M3)
    text = text.replace(CALL, ANCHOR_CALL)
    text = text.replace(GEMM, ANCHOR_GEMM)
    path.write_text(text)
    print("REVERTED mx_paged_decode_mainloop.hpp (5 edits reversed)")
    print("P212Z_REVERT_OK")


if __name__ == "__main__":
    main()
