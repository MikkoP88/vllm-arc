# SPDX-License-Identifier: Apache-2.0
"""store_vstride_probe.py — P6.47 leg B regression guard (the V-stride bug).

ROOT CAUSE (trace24, 2026-11-12): mx_store passed keys' token+head strides
for BOTH loads; in-engine key and value arrive with INDEPENDENT layouts
(upstream law — the wheel's store kernels take stride_val_tok /
stride_val_head). Stored-V rel-L2 was 42-80% with K clean at 2-3%
(chunk_snr xcheck), invisible to every earlier check because they compared
store-vs-store or stored-K-vs-raw-K, never stored-V-vs-raw-V.

This probe stores one batch where v is a transpose VIEW (token/head
strides swapped relative to k) and asserts the dequantized cache V
matches the v CONTENT (not the v view's memory order). Runs offline on
xpu (rig law: /work PYTHONPATH shadow).
"""

import sys
from pathlib import Path

WORK = Path("/work")
sys.path.insert(0, str(WORK))

import torch  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import slot_bytes  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_store import mx_store  # noqa: E402

torch.manual_seed(0)
DEV = "xpu"
BLOCKS, BS, HK, D, NG = 4, 64, 4, 256, 8
SLOT = slot_bytes(D, "mxfp8")
VO = D + NG  # mxfp8 v payload offset (mx_layout law)
N = 50

# k: standard contiguous (N, Hk, D) — strides (Hk*D, D, 1).
k = torch.randn(N, HK, D, device=DEV, dtype=torch.float32) * 3.0
# v CONTENT in (Hk, N, D) contiguous memory, viewed as (N, Hk, D):
# token/head strides swap vs k — the in-engine mismatch shape.
v_mem = torch.randn(HK, N, D, device=DEV, dtype=torch.float32) * 3.0
v = v_mem.transpose(0, 1)
print(f"k strides {k.stride()}  v strides {v.stride()} (mismatched by design)")

cache = torch.zeros(BLOCKS, BS, HK, SLOT, dtype=torch.uint8, device=DEV)
slots = torch.arange(0, N, device=DEV, dtype=torch.int64)
tb = torch.div(slots, BS, rounding_mode="floor").to(torch.int32)
tp = (slots - tb.long() * BS).to(torch.int32)
fmt = mx_store(k, v, cache, tb, tp)

# Decode every stored V head-row and compare against the CONTENT v[n, h].
worst = 0.0
for n in range(N):
    for h in range(HK):
        row = cache[slots[n] // BS, slots[n] % BS, h]
        _, vv = decode_kv(row[:VO], row[VO:], fmt, D)
        rel = float((vv - v[n, h]).norm() / v[n, h].norm().clamp_min(1e-6))
        worst = max(worst, rel)
print(f"worst stored-V rel-L2 across {N * HK} rows: {worst:.4f}")
# e4m3 per-32-group theory: ~3% RMS, bound at 8% for 50-row worst case.
print("VERDICT", "VSTRIDE_OK" if worst < 0.08 else "VSTRIDE_BROKEN (V stored from wrong addresses)")
sys.exit(0 if worst < 0.08 else 1)
