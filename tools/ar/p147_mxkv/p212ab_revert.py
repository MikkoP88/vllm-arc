#!/usr/bin/env python3
"""p212ab_revert.py -- revert r212ab (peeled K-decode pipeline),
returning the tree to the exact r212aa state. The original loop bytes
are preserved verbatim inside the wrapper, so reverting is dropping
the WRAP_OPEN prefix and the PIPE block."""
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p212ab_patch import MARK, PIPE, SRC, WRAP_OPEN


def main():
    path = SRC / "mx_paged_decode_mainloop.hpp"
    text = path.read_text(encoding="utf-8")
    if text.count(MARK) == 0:
        print("REVERT_NOT_NEEDED (no r212ab marker present)")
        sys.exit(1)
    if text.count(WRAP_OPEN) != 1 or text.count(PIPE) != 1:
        print("REVERT_MISS (wrapper or pipeline block not unique)")
        sys.exit(1)
    text = text.replace(WRAP_OPEN, "").replace(PIPE, "")
    path.write_text(text, encoding="utf-8", newline="\n")
    print("REVERTED mx_paged_decode_mainloop.hpp (wrapper + pipeline dropped)")
    print("P212AB_REVERT_OK")


if __name__ == "__main__":
    main()
