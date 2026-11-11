#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p147_dbg.py — one-slot mxfp4 byte-law divergence probe (leg A debug).
Prints kernel bytes vs reference bytes for token/head (0,0) so the
transform difference is visible directly."""
import importlib.util
import sys
from pathlib import Path

import torch

_MXKV_DIR = Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
spec = importlib.util.spec_from_file_location(
    "mx_kv_lane", _MXKV_DIR / "__init__.py", submodule_search_locations=[str(_MXKV_DIR)]
)
mxkv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mxkv
spec.loader.exec_module(mxkv)

D, HK, BLOCK = 128, 2, 64
gen = torch.Generator().manual_seed(7)
kx = torch.randn(1, HK, D, generator=gen)
vx = torch.randn(1, HK, D, generator=gen) * 0.5

enc, codes = mxkv.encode_mxfp4(kx)
ref = torch.cat([enc, codes], dim=-1)[0, 0]  # [68]

cache = torch.zeros(1, BLOCK, HK, mxkv.slot_bytes(D, "mxfp4"), dtype=torch.uint8, device="xpu")
tb = torch.zeros(1, dtype=torch.int32, device="xpu")
tp = torch.zeros(1, dtype=torch.int32, device="xpu")
mxkv.mx_store(kx.to("xpu"), vx.to("xpu"), cache, tb, tp)
torch.xpu.synchronize()
got = cache[0, 0, 0, :].cpu()

print("K row (first 8 elems):", [f"{v:.4f}" for v in kx[0, 0, :8].tolist()])
print("ref  bytes[0:16]:", ref[:16].tolist())
print("kern bytes[0:16]:", got[:16].tolist())
print("ref  bytes[64:68] (codes):", ref[64:68].tolist())
print("kern bytes[64:68] (codes):", got[64:68].tolist())
nd = int((got[:68] != ref).sum())
print(f"K block mismatches: {nd}/68")
# element-level: decode both
k_ref, _ = mxkv.decode_kv(ref.unsqueeze(0).unsqueeze(0), ref.unsqueeze(0).unsqueeze(0), "mxfp4", D)
k_got, _ = mxkv.decode_kv(got[:68].unsqueeze(0).unsqueeze(0).contiguous(), got[:68].unsqueeze(0).unsqueeze(0).contiguous(), "mxfp4", D)
print("ref-decoded [0:8]:", [f"{v:.4f}" for v in k_ref[0, 0, :8].tolist()])
print("got-decoded [0:8]:", [f"{v:.4f}" for v in k_got[0, 0, :8].tolist()])
