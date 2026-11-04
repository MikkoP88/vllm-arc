#!/usr/bin/env python3
"""P619d norms — who is the zero vector in the t8f dump?"""
import torch

d = torch.load("/root/p619f_dump.pt", map_location="cpu", weights_only=False)
print("mq_out  rows:", [round(float(r.float().norm()), 6) for r in d["mq_out"]])
print("ref0    rows:", [round(float(r.float().norm()), 6) for r in d["ref0"]])
cp = d["cache_pages"]
print("cache_pages: shape", tuple(cp.shape), "dtype", cp.dtype,
      "nonzero_frac", round(float((cp != 0).float().mean()), 6))
# page 117 = bt[0,0]: the request's first page (seq 9 -> tokens 0..8 live here)
p117 = cp[int(d["bt_row"][0, 0])]
print("page117: nonzero_frac", round(float((p117 != 0).float().mean()), 6),
      "first-slot bytes", p117[0, 0, :16].tolist())
print("bt_row first 12:", d["bt_row"][0, :12].tolist())
print("q rows norms:", [round(float(r.float().norm()), 4) for r in d["q"]])
