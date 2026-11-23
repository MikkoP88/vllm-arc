#!/usr/bin/env python3
"""r231b_extract.py -- pull the MX-decode .so artifacts out of the
canonical arc7mx wheel (exact-name match: the wheels dir also carries
_r210keeper and *.bak siblings whose globs must not match).

The MX paged-decode kernels live in libattn_kernels_xe_2.so (37
mx_paged_decode dynsyms; libmqa_logits carries none -- the r231b-1
misextraction lesson); mx_attn_interface.py reaches them through the
_vllm_fa2_C pybind stub, so both artifacts are extracted. Run INSIDE
the builder.
"""

import sys
import zipfile

WHEEL = "/wheels/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl"
OUT = "/tmp/r231b_so"
WANTED = ("libattn_kernels_xe_2.so", "_vllm_fa2_C.abi3.so")


def main() -> int:
    z = zipfile.ZipFile(WHEEL)
    names = [
        n for n in z.namelist()
        if n.rsplit("/", 1)[-1] in WANTED
    ]
    if len(names) != len(WANTED):
        print(f"R231B_EXTRACT_ABORT: {sorted(names)}")
        return 1
    for n in sorted(names):
        z.extract(n, OUT)
        print(f"R231B_EXTRACTED {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
