#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p199_ablation.py — arc7 Phase-8 r199a: attribution telemetry for the
MX-NATIVE CUTLASS/Xe2 paged-decode kernel (arc7mx wheel).

Telemetry-only companion to p198_native_offline (correctness stays
owned by the r198 faces). Splits each native call into HOST time
(wall - device) and DEVICE time (XPU events) so the 0.10-0.26x v2
telemetry gap decomposes into per-call host overhead (workspace /
buffer allocation churn in the C++ entry) vs the device-side walk
(the dequant's scattered per-element gmem reads). Dequant-ablation
wheels (constant-fill kernel bodies) reuse this script unchanged.

Runs INSIDE a cert25zc-family container with the arc7mx wheel
installed (same posture as r198). Usage: p199_ablation.py OUT_DIR.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

# Load the mx_kv package by FILE PATH under a synthetic top-level name
# (p147 law: the wheel's vllm_gaudi would win resolution otherwise).
_MXKV_DIR = (
    Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
)


def _load_mxkv():
    spec = importlib.util.spec_from_file_location(
        "mx_kv_lane",
        _MXKV_DIR / "__init__.py",
        submodule_search_locations=[str(_MXKV_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mxkv = _load_mxkv()
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

D = 128
HK = 2
HREP = 4
HQ = HK * HREP
BLOCK = 16
DEV = "xpu"
FMTS = ("mxfp8", "mxfp4", "mxk8v4")


def build_paged(total_tokens, seed=7):
    """Fragmented block-16 tables: shuffled physical blocks, 2 sequences
    (p198's geometry, verbatim)."""
    n_blocks = total_tokens // BLOCK + 2
    perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed))
    order = perm.tolist()
    lens = (
        [total_tokens]
        if total_tokens < 2
        else [max(1, total_tokens // 3), total_tokens - max(1, total_tokens // 3)]
    )
    tables, tok_block, tok_pos = [], [], []
    cur = 0
    for ln in lens:
        ntb = (ln + BLOCK - 1) // BLOCK
        tab = order[cur : cur + ntb]
        tables.append(torch.tensor(tab, dtype=torch.int32, device=DEV))
        for t in range(ln):
            tok_block.append(tab[t // BLOCK])
            tok_pos.append(t % BLOCK)
        cur += ntb
    return (
        lens,
        tables,
        torch.tensor(tok_block, dtype=torch.int32, device=DEV),
        torch.tensor(tok_pos, dtype=torch.int32, device=DEV),
        n_blocks,
    )


def flat_tables(lens, tables):
    mb = max(t.shape[0] for t in tables)
    bt = torch.zeros(len(lens), mb, dtype=torch.int32, device=DEV)
    for i, t in enumerate(tables):
        bt[i, : t.shape[0]] = t
    return bt


def native_ms_split(q_bf, cache, fmt, lens, bt, iters=20, warm=3):
    """(wall_ms, device_ms, host_ms) per native call, averaged."""
    cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    for _ in range(warm):
        mx_paged_decode(
            q_bf, cache, fmt, cu_q, sl, bt,
            max_seqlen_k=max(lens), softmax_scale=1.0 / math.sqrt(D),
        )
    torch.xpu.synchronize()
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    cu_q2 = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
    sl2 = torch.tensor(lens, dtype=torch.int32, device=DEV)
    t0 = time.perf_counter()
    ev0.record()
    for _ in range(iters):
        mx_paged_decode(
            q_bf, cache, fmt, cu_q2, sl2, bt,
            max_seqlen_k=max(lens), softmax_scale=1.0 / math.sqrt(D),
        )
    ev1.record()
    torch.xpu.synchronize()
    wall = (time.perf_counter() - t0) / iters
    dev = ev0.elapsed_time(ev1) / 1e3 / iters
    return wall, dev, wall - dev


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p199_ablation — device={DEV}, torch={torch.__version__}")
    R = {}
    for fmt in FMTS:
        rows = {}
        for ctx in (1024, 8192, 32768):
            gen = torch.Generator().manual_seed(ctx)
            xk = torch.randn(ctx, HK, D, generator=gen).to(DEV)
            xv = torch.randn(ctx, HK, D, generator=gen).to(DEV)
            lens, tables, tb, tp, n_blocks = build_paged(ctx, seed=ctx % 97)
            q = (
                torch.randn(len(lens), HQ, D, generator=gen)
                .to(torch.bfloat16)
                .to(DEV)
            )
            cache = torch.zeros(
                n_blocks,
                BLOCK,
                HK,
                slot_bytes(D, fmt),
                dtype=torch.uint8,
                device=DEV,
            )
            mx_store(xk, xv, cache, tb, tp)
            bt = flat_tables(lens, tables)
            torch.xpu.synchronize()
            wall, dev, host = native_ms_split(q, cache, fmt, lens, bt)
            b_read = ctx * HK * slot_bytes(D, fmt) * HREP
            rows[ctx] = {
                "wall_ms": round(wall * 1e3, 4),
                "device_ms": round(dev * 1e3, 4),
                "host_ms": round(host * 1e3, 4),
                "device_gbps": round(b_read / dev / 1e9, 2),
            }
            print(
                f"  {fmt} ctx={ctx}: wall {wall*1e3:.3f} ms = "
                f"device {dev*1e3:.3f} + host {host*1e3:.3f} ms "
                f"({b_read/dev/1e9:.1f} GB/s device-effective)"
            )
        R[fmt] = rows
    with open(out / "p199_ablation.json", "w", newline="\n") as f:
        json.dump(R, f, indent=1)
    print("P199_DONE")


if __name__ == "__main__":
    main()
