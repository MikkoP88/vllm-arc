# SPDX-License-Identifier: Apache-2.0
"""dump_inspect.py — P6.47 leg B: full-field dump payload inspector.

Prints every payload's mql/msl/seq_lens/bt rows/smap tail so recurring
forward classes (the 97/97 and 6656/6656 prefill-shaped forwards that
eat ord-0 sentinels) can be identified: seq_lens shows batch
composition, smap shows WHERE the forward wrote, bt shows WHICH
blocks it walks (aliasing a live request's blocks = the clobber
signature)."""

import sys
from pathlib import Path

import torch

for f in sorted(Path(sys.argv[1]).glob("*.pt")):
    p = torch.load(f, map_location="cpu", weights_only=False)
    print(f"== {f.name}")
    print(f"  mql={p['mql']} msl={p['msl']} impl={p.get('impl')} ord={p.get('ord')}")
    print(f"  seq_lens[:8]={p['seq_lens'][:8].tolist()}")
    for r in range(p["bt"].shape[0]):
        row = p["bt"][r].tolist()
        nz = [b for b in row if b]
        print(f"  bt[{r}] nonzero={nz[:12]}{'...' if len(nz) > 12 else ''}")
    sm = p["smap"].tolist()
    print(f"  smap tail={sm}")
