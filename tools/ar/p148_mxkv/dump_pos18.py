# SPDX-License-Identifier: Apache-2.0
"""dump_pos18.py — P6.47 leg B: the position-sl-1 rewrite discriminator.

trace18 series fact: C(verify sl=19)-vs-B(sl=75) is byte-IDENTICAL at
positions 0..17 but position 18 (C's last row) is wholesale-rewritten
(1022/1056 B). Two candidate wrong-writers: C's own draft-row store,
or the next step's re-process store. The tokens are IDENTICAL in both
lanes below the divergence point (~pos 29), so diffing position 18
against the EAGER lane's deepest decode names the wrong write:
  spec-B pos-18 ~ eager pos-18  -> the BETWEEN-writer is correct and
                                   C's original store was wrong;
  spec-B pos-18 ~ C   pos-18    -> the between-writer poisoned it.
Also probes position 19..23 (later re-stores) for the pattern."""

import sys
from collections import Counter
from pathlib import Path

import torch

D = 256
NG = 8
BOUNDS = {"K": (0, D), "Ks": (D, D + NG), "V": (D + NG, 2 * D + NG), "Vs": (2 * D + NG, None)}


def load(d):
    out = []
    for f in sorted(Path(d).glob("*.pt")):
        p = torch.load(f, map_location="cpu", weights_only=False)
        pid, n = f.name.split(".")[1:3]
        p["_pid"], p["_n"] = int(pid), int(n)
        out.append(p)
    return out


def row(p, pos):
    bt = p["bt"][0].tolist()
    return p["blocks"][bt[pos // 64], pos % 64]


def diff(x, y, label):
    d = x != y
    n = int(d.sum())
    parts = {k: int(d[:, lo:hi or d.shape[-1]].sum()) for k, (lo, hi) in BOUNDS.items()}
    print(f"{label}: {n}/1056 B  parts={parts}")
    return n


S = load(sys.argv[1])
E = load(sys.argv[2])
dom = Counter(p["_pid"] for p in S).most_common(1)[0][0]
sp = [p for p in S if p["_pid"] == dom]
sp_ver = [p for p in sp if p["mql"] in (1, 2)]
C = min(sp_ver, key=lambda p: int(p["seq_lens"][0]))
B = max(sp_ver, key=lambda p: int(p["seq_lens"][0]))
edom = Counter(p["_pid"] for p in E).most_common(1)[0][0]
ep = [p for p in E if p["_pid"] == edom]
e_ver = [p for p in ep if p["mql"] in (1, 2)]
EB = max(e_ver, key=lambda p: int(p["seq_lens"][0]))
print(f"C: sl={int(C['seq_lens'][0])}  B: sl={int(B['seq_lens'][0])}  EB: sl={int(EB['seq_lens'][0])}")
for pos in [16, 17, 18, 19, 20, 22]:
    diff(row(B, pos), row(EB, pos), f"pos {pos}: specB-vs-eagerB")
print()
for pos in [16, 17, 18, 19, 20, 22]:
    diff(row(B, pos), row(C, pos), f"pos {pos}: specB-vs-C")
print()
for pos in [16, 17, 18, 19, 20, 22]:
    diff(row(C, pos), row(EB, pos), f"pos {pos}: C-vs-eagerB")
