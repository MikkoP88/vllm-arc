#!/usr/bin/env python3
"""r231b_pickprobe.py -- enumerate the MX payload banks' pick geometry
(sl vs banked window capacity nbt*64) so the r229 cell picks can be
chosen to fit the fp64 walk (the r231b-2 lesson: idx 1/12 banks carry
sl~8198 over a 96-block window -> walk IndexError). CPU-only load.
"""

import glob
import math
import os

import torch

BANKS = [
    ("r224a", "/root/build/p647/r224a", "2447"),
    ("r229a", "/root/build/p647/r229a", "1030"),
    ("r229b", "/root/build/p647/r229b", "1030"),
]


def main() -> int:
    for name, d, pre in BANKS:
        for f in sorted(
            glob.glob(os.path.join(d, f"mxkv_ab.{pre}.*.pt")),
            key=lambda p: int(p.rsplit(".", 2)[1]),
        ):
            idx = int(f.rsplit(".", 2)[1])
            p = torch.load(f, map_location="cpu", weights_only=False)
            try:
                r = int(p["q"].shape[0])
                sl = p["sl"][:r].tolist()
                nbt = int(p["bt_head"].numel())
            except KeyError:
                print(f"R231B_PICK {name} idx={idx} SCHEMA_SKIP")
                continue
            need = math.ceil(max(sl) / 64)
            fit = "FIT" if need <= nbt else "OVER"
            print(
                f"R231B_PICK {name} idx={idx} fmt={p['fmt']} "
                f"sl0={sl[0]} slmax={max(sl)} need_blocks={need} "
                f"nbt={nbt} {fit}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
