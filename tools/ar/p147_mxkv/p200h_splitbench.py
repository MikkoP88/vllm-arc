#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p200h_splitbench.py -- arc7 Phase-8 r200h/r207: num_kv_splits sweep for
the MX-NATIVE CUTLASS/Xe2 paged-decode kernel (arc7mx wheel).

The r199/p198 telemetry compares native (kernel default num_kv_splits=1:
a serial 2048-k-tile walk per (head, seq) workgroup -- 16 WGs total at
HQ=8 x batch=2) against the Triton v2 decoder, which launches its own
block-split grid. This bench sweeps the kernel's num_splits_kv path at
1/2/4/8/16/24/32/48/64. num_splits_kv acts as the CAP on the per-seq
host split plan (work-list path), and ARC_MXKV_WGTARGET (set to 64
here) scales the plan's tiles-per-WG target so per-seq splits actually
track the cap -- t(s) therefore measures true split-scaling to 64.

r207 law (the first extended run's lesson): s>16 used to die inside
can_implement (max_num_kv_splits was 16) with a Status the torch op
swallows -- NO kernel launched, the "measured" time was host overhead,
and the canary FALSE-PASSED because torch.empty_like recycled the
allocator block still holding the previous correct output. The canary
now writes into a NaN-poisoned buffer: a non-launching cell shows
CANARY_NAN and is invalid on its face.

Also records |splits_n - splits_1| max-abs as a cheap corruption canary
per cell (the real gate stays the r198 forced8 face) and prints the
per-seq split plan per cell for evidence.

Runs INSIDE a cert25zc-family container with the arc7mx wheel installed
(same posture as r198/p199). Usage: p200h_splitbench.py OUT_DIR.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

# r207: make the per-seq plan track the sweep cap (see docstring).
os.environ.setdefault("ARC_MXKV_WGTARGET", "64")

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

from vllm_xpu_kernels.mx_attn_interface import (  # noqa: E402
    _infer_num_xe_cores,
    _kv_tile_from_block_size,
    _mx_build_decode_split_plan,
    mx_paged_decode,
)

D = 128
HK = 2
HREP = 4
HQ = HK * HREP
BLOCK = 16
DEV = "xpu"
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
# r210: ctx list and split list are env-tunable so the same binary runs
# the 1K calibration sweep (P200H_CTX=1024 P200H_SPLITS=1,2,3,4,5,6,8,
# 10,12,16 -- the vendor plan's 4-tiles-per-WG floor caps plannable
# splits at n_tiles/4, so >16 is unreachable at 64 tiles/seq).
SPLITS = tuple(
    int(s)
    for s in os.environ.get(
        "P200H_SPLITS", "1,2,4,8,16,24,32,48,64"
    ).split(",")
    if s.strip()
)
CTXS = tuple(
    int(c)
    for c in os.environ.get("P200H_CTX", "8192,32768").split(",")
    if c.strip()
)


def build_paged(total_tokens, seed=7):
    """Fragmented block-16 tables: shuffled physical blocks, 2 sequences
    (p198/p199 geometry, verbatim)."""
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


def native_call(q_bf, cache, fmt, lens, bt, splits, out=None):
    cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    return mx_paged_decode(
        q_bf,
        cache,
        fmt,
        cu_q,
        sl,
        bt,
        max_seqlen_k=max(lens),
        softmax_scale=1.0 / math.sqrt(D),
        num_splits_kv=splits,
        host_kv_lens=torch.tensor(lens) if splits > 1 else None,
        out=out,
    )


def bench(fn, iters=20, warm=3):
    """(wall_ms, device_ms, host_ms) per call, averaged -- p199 posture."""
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    t0 = time.perf_counter()
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.xpu.synchronize()
    wall = (time.perf_counter() - t0) / iters
    dev = ev0.elapsed_time(ev1) / 1e3 / iters
    return wall, dev, wall - dev


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p200h_splitbench -- device={DEV}, torch={torch.__version__}")
    R = {}
    for fmt in FMTS:
        rows = {}
        for ctx in CTXS:
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
            ref1 = (
                native_call(q, cache, fmt, lens, bt, 1).float().cpu()
            )  # splits=1 reference for the canary
            cell = {}
            # r207: the plan the sweep cap + WGTARGET actually produce
            # (evidence per cell -- t(s) is a plan-splits curve).
            ncores = _infer_num_xe_cores(torch.device(DEV))
            kt = _kv_tile_from_block_size(BLOCK)
            for sp in SPLITS:
                # B023 law: bind the loop variables through defaults so
                # the closure cannot capture a later iteration's binding.
                call = lambda q=q, cache=cache, fmt=fmt, lens=lens, bt=bt, sp=sp: (  # noqa: E731
                    native_call(q, cache, fmt, lens, bt, sp)
                )
                wall, dev, host = bench(call)
                # NaN-poisoned canary (r207 law): a kernel that never
                # launches leaves the poison intact -> CANARY_NAN.
                poison = torch.full_like(q, float("nan"))
                d_canary = (
                    native_call(q, cache, fmt, lens, bt, sp, out=poison)
                    .float()
                    .cpu()
                    .sub(ref1)
                    .abs()
                    .max()
                    .item()
                )
                if math.isnan(d_canary):
                    print(f"  !! {fmt} ctx={ctx} splits={sp}: CANARY_NAN "
                          "(kernel did not launch?)")
                    d_canary = -1.0
                plan_splits, _wl = _mx_build_decode_split_plan(
                    lens, kt, sp, ncores, HK)
                plan_str = list(plan_splits.tolist())
                cell[str(sp)] = {
                    "wall_ms": round(wall * 1e3, 4),
                    "device_ms": round(dev * 1e3, 4),
                    "host_ms": round(host * 1e3, 4),
                    "canary_vs_splits1": round(d_canary, 6),
                    "plan_splits": plan_str,
                }
                print(
                    f"  {fmt} ctx={ctx} splits={sp:2d} plan={plan_str}: "
                    f"device {dev*1e3:.3f} ms (wall {wall*1e3:.3f}, "
                    f"host {host*1e3:.3f}), canary {d_canary:.2e}"
                )
            rows[str(ctx)] = cell
        R[fmt] = rows
    with open(out / "p200h_splits.json", "w", newline="\n") as f:
        json.dump(R, f, indent=1)
    print("P200H_DONE")


if __name__ == "__main__":
    main()
