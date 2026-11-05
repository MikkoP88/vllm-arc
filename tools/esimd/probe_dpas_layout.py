#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P6.21 r92 round-6b: decode the dpas_probe output into the true
(C lane, k) -> (B word, half) map for xmx dpas N=8 (raw + SLM-roundtrip
variants) and the N=16 control. B ids: word f LO=f, HI=f+base
(base=128 for N=8, 256 for N=16). A runs: c0 row0-ones (which C lanes
belong to A row 0), c1 delta(j==t) (k=t row t), c2 delta(j==8+t).
"""
from __future__ import annotations

import os
import sys

import torch

_so = os.environ.get("ARC_DENSE_FA_ESIMD_SO", "")
if _so:
    sys.path.insert(0, os.path.dirname(os.path.abspath(_so)))
import dpas_probe_ops  # noqa: E402


def decode(name, c0, c1, c2, N, base):
    n_lanes = 8 * N
    print(f"== {name} (N={N})")
    lit = [i for i in range(n_lanes) if abs(float(c0[i])) > 1e-3]
    print(f"  row0-ones lit lanes: {lit}")
    ok = True
    for i in range(n_lanes):
        v1, v2 = int(round(float(c1[i]))), int(round(float(c2[i])))
        f1, h1 = v1 % base, v1 // base
        f2, h2 = v2 % base, v2 // base
        m, n = i // N, i % N
        e1 = ((m // 2) * N + n, m % 2)
        e2 = (((8 + m) // 2) * N + n, (8 + m) % 2)
        got = (f1, h1, f2, h2)
        exp = (e1[0], e1[1], e2[0], e2[1])
        if got != exp:
            ok = False
            hs = lambda h: "HI" if h else "LO"  # noqa: E731
            print(f"  lane {i:3d} (m={m},n={n}): k<8 -> w{f1}.{hs(h1)}"
                  f"  k>=8 -> w{f2}.{hs(h2)}   my-law exp w{e1[0]}.{hs(e1[1])}"
                  f" / w{e2[0]}.{hs(e2[1])}")
    print(f"  my-law holds everywhere: {ok}")
    if not ok:
        print("  full map (lane: k<8 word.half | k>=8 word.half):")
        for i in range(n_lanes):
            v1, v2 = int(round(float(c1[i]))), int(round(float(c2[i])))
            hs = lambda h: "HI" if h else "LO"  # noqa: E731
            print(f"   {i:3d}: w{v1 % base}.{hs(v1 // base)} | "
                  f"w{v2 % base}.{hs(v2 // base)}")


def main() -> int:
    torch.xpu.set_device("xpu:0")
    out = torch.zeros(768, device="xpu", dtype=torch.float32)
    dpas_probe_ops.dpas_probe(out)
    torch.xpu.synchronize()
    o = out.cpu().tolist()
    decode("a: N=8 raw-register B", o[0:64], o[64:128], o[128:192], 8, 128)
    decode("b: N=8 SLM-roundtrip B", o[192:256], o[256:320], o[320:384],
           8, 128)
    decode("c: N=16 control", o[384:512], o[512:640], o[640:768], 16, 256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
