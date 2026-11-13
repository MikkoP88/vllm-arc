#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p173_v6sweep.py — P6.47 A14 v6 deep-split sweep probe.

Owner PRIORITY 1 + "if kernel is issue patch it" (2026-11-13): the
m8 32K decode gap is kernel-explained (16 x v5 8.79 = 140.7 ms ~= the
130 ms e2e delta) and the v5 grid is parallelism-starved — 128 CTAs
serially walking 32-tile bt->slot->gather chains at ~3.8 GB/s
effective (~1% HBM). The v6 patch: split cap 32 -> 256 with the
adaptive >= 4-serial-tile rule, a v5-only BLOCK_N=64 MMA tune, and the
chunked combine. This probe validates and sweeps it:

  1. correctness — v6 default (BN=64, splits=256) vs per-row v2 at
     the LIMIT-LAW boundary shapes (ctx 549 odd tail; 2051 final-tile
     straddle; mixed 549/1500/2051/8192), per format. Gate d <= 2e-3
     + finite (fp16-operand class, the ctrl lane's own).
  2. A/B + sweep — G 1/8 x ctx 8192/32768 per format: the A13 v5
     baseline (ARC_MXKV_V5_SPLITS=32 BN=32, same grid as cert25z
     shipping) vs v6 default vs the BN {32,64} x splits {32,64,128,
     256} grid. Winner table -> the bake decision.

Loads mx_kv from MXKV_DIR (the PATCHED tree mount; falls back to the
repo tree). Env knobs are call-time reads in the patched launcher —
the sweep walks them without re-imports.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch

_TREE = (
    Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
)
_MXKV_DIR = Path(os.environ.get("MXKV_DIR", "") or _TREE)


def _load_mxkv():
    spec = importlib.util.spec_from_file_location(
        "mx_kv_lane6",
        _MXKV_DIR / "__init__.py",
        submodule_search_locations=[str(_MXKV_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mxkv = _load_mxkv()
mx_decode = mxkv.mx_decode
mx_decode_rows_dot = mxkv.mx_decode_rows_dot
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
RTILE = 4
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp8", "mxfp4", "mxk8v4")


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """G requests with per-request ctx; rows of request g sit at
    g*rtile..+r. Returns the v4/v5 call args (p160 law)."""
    g = len(ctxs)
    n_blocks = sum((c + BLOCK - 1) // BLOCK for c in ctxs) + 2
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_blocks, generator=gen)
    cache = torch.zeros(
        n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
    )
    bt = torch.zeros(g, (max(ctxs) + BLOCK - 1) // BLOCK,
                     dtype=torch.int32, device=DEV)
    cursor = 0
    for gi, c in enumerate(ctxs):
        blocks = perm[cursor: cursor + (c + BLOCK - 1) // BLOCK]
        cursor += (c + BLOCK - 1) // BLOCK
        bt[gi, : blocks.numel()] = blocks.to(torch.int32).to(DEV)
        tb = torch.tensor([int(b) for b in blocks])
        tp = torch.tensor([t % BLOCK for t in range(c)])
        tb_full = torch.tensor([int(tb[t // BLOCK]) for t in range(c)])
        xk = torch.randn(c, HK, D, generator=gen).to(DEV)
        xv = torch.randn(c, HK, D, generator=gen).to(DEV)
        mx_store(xk, xv, cache, tb_full.to(DEV), tp.to(DEV))
    torch.xpu.synchronize()
    q = torch.randn(g * rtile, HK * HREP, D, generator=gen).to(DEV)
    lims = torch.cat(
        [
            (c - 3 + torch.arange(rtile)).to(torch.int32).to(DEV)
            for c in ctxs
        ]
    )
    return q, cache, bt.contiguous(), lims


def timeit(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def v5env(bn, splits):
    os.environ["ARC_MXKV_V5_BN"] = str(bn)
    os.environ["ARC_MXKV_V5_SPLITS"] = str(splits)


def v5clean():
    os.environ.pop("ARC_MXKV_V5_BN", None)
    os.environ.pop("ARC_MXKV_V5_SPLITS", None)


def main():
    print(f"p173_v6sweep — device={DEV}, torch={torch.__version__}")
    print(f"mxkv_src={_MXKV_DIR}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    best: dict[tuple, tuple] = {}
    for fmt in FMTS:
        # ---- face 1: correctness — v6 default vs per-row v2
        v5clean()
        for name, ctxs in {
            "ctx549": [549] * 2,
            "ctx2051": [2051] * 2,
            "mixed": [549, 1500, 2051, 8192],
        }.items():
            q, cache, bt, lims = build_multi(ctxs, fmt, 31)
            o6 = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            o2 = torch.empty_like(o6)
            for r_ in range(q.shape[0]):
                g_ = r_ // RTILE
                o2[r_: r_ + 1] = mx_decode(
                    q[r_: r_ + 1],
                    cache,
                    bt[g_: g_ + 1],
                    lims[r_: r_ + 1],
                    SCALE,
                )
            torch.xpu.synchronize()
            d = float((o6 - o2).abs().amax().item())
            ok = d <= 2e-3 and torch.isfinite(o6).all()
            print(f"[{fmt}] v6 vs per-row v2 {name}: max|d|={d:.3e} "
                  f"{'OK' if ok else 'FAIL'}")
            rc |= 0 if ok else 1
        # ---- face 2: A/B + sweep at the r162 problem cells
        for g_n, ctx in ((1, 8192), (1, 32768), (8, 8192), (8, 32768)):
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)
            reps = 5 if ctx >= 32768 else 10
            cell = (fmt, g_n, ctx)

            def call(_q=q, _c=cache, _b=bt, _l=lims):
                return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

            def run(bn, sp, _r=reps):
                v5env(bn, sp)
                return timeit(call, _r)

            t_base = run(32, 32)  # A13 v5 grid (cert25z shipping shape)
            results = []
            for bn in (32, 64):
                for sp in (32, 64, 128, 256):
                    t = run(bn, sp)
                    results.append((bn, sp, t))
            v5clean()
            t_def = timeit(call, reps)
            for bn, sp, t in results:
                print(f"[{fmt}] G={g_n} ctx={ctx} bn={bn} splits={sp}: "
                      f"{t:.3f} ms ({t_base / t:.2f}x)")
            print(f"[{fmt}] G={g_n} ctx={ctx} v6-default: {t_def:.3f} ms "
                  f"({t_base / t_def:.2f}x vs A13 v5 {t_base:.3f})")
            bn_b, sp_b, t_b = min(results, key=lambda x: x[2])
            best[cell] = (bn_b, sp_b, t_b, t_base)
    print("--- A14 winner table (best swept config per cell) ---")
    for (fmt, g_n, ctx), (bn_b, sp_b, t_b, t_base) in best.items():
        print(f"[{fmt}] G={g_n} ctx={ctx}: bn={bn_b} splits={sp_b} "
              f"{t_b:.3f} ms = {t_base / t_b:.2f}x A13-v5")
    print(f"A14_V6_SWEEP_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
