# SPDX-License-Identifier: Apache-2.0
"""dump_diff_pair.py — P6.47 leg B: direct payload-pair differential.
Picks the spec mid-verify p1 payload (mql==2) and the eager p1-prefill
payload (mql==sl==14) from two dump directories and compares the
prompt-position bytes (0..13) through each boot's own block table.
Tokens are identical there in both lanes (the text starts coherent), so
byte differences at these positions are wrong stores."""

import sys
from pathlib import Path

import torch


def load(d):
    return [
        torch.load(f, map_location="cpu", weights_only=False)
        for f in sorted(Path(d).glob("*.pt"))
    ]


S = load(sys.argv[1])
E = load(sys.argv[2])
s = max((p for p in S if p["mql"] == 2), key=lambda p: int(p["seq_lens"][0]))
e = next(p for p in E if p["mql"] == 14 and int(p["seq_lens"][0]) == 14)
bs, be = s["blocks"], e["blocks"]
bts, bte = s["bt"][0].tolist(), e["bt"][0].tolist()
print("spec: ", "sl=", int(s["seq_lens"][0]), "bt=", bts[:4], "smap=", s["smap"][-4:].tolist())
print("eager:", "sl=", int(e["seq_lens"][0]), "bt=", bte[:4], "smap=", e["smap"][-4:].tolist())

d_ = 256
ng = 8
slot = int(bs.shape[-1])
bounds = {"K": (0, d_), "Ks": (d_, d_ + ng), "V": (d_ + ng, 2 * d_ + ng), "Vs": (2 * d_ + ng, slot)}
for p in range(14):
    a = bs[bts[p // 64], p % 64]
    b = be[bte[p // 64], p % 64]
    d = a != b
    n = int(d.sum())
    parts = [f"{k}:{int(d[:, lo:hi].sum())}" for k, (lo, hi) in bounds.items() if int(d[:, lo:hi].sum())]
    print(f"pos {p:2d}: {n:4d} B differ ({', '.join(parts) if parts else 'IDENTICAL'})")
