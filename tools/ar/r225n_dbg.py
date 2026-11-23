#!/usr/bin/env python3
"""r225n_dbg2.py — per-row H0-vs-ref discrepancy hunt."""
import sys

import torch

from r225m_splitsim import D, dmax, logits_for_payload
from r225n_bound import softmax_out

p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
r = p["q"].shape[0]
per_row = logits_for_payload(p)
print("q shape", p["q"].shape, "sl", p["sl"].tolist(), "r", r)

o_ref = torch.empty(r, p["q"].shape[1], D, dtype=torch.float64)
for i in range(r):
    for h in range(p["q"].shape[1]):
        s, v = per_row[i][h]
        o_ref[i, h] = torch.softmax(s, dim=-1) @ v

n = int(p["sl"][0])
h0 = list(range(n))
for i in range(r):
    d_row = 0.0
    for h in range(p["q"].shape[1]):
        s, v = per_row[i][h]
        got = softmax_out(s, v, h0)
        d_row = max(d_row, dmax(got, o_ref[i, h]))
    print(f"row {i}: H0-vs-ref dmax={d_row:.3e}  (logits len "
          f"{per_row[i][0][0].shape[0]})")
print("DBG2_DONE")
