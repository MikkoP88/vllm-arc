#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p159_v4rows.py — P6.47 A11 lever 2: the v4 verify-row-shared decode.

mx_decode_rows (triton_mx_decode.py, A11-NEXT-LEVERS.md lever 2): one
program per (row-group, q-head, split) walks the request's blocks ONCE
and serves all R_TILE=4 MTP verify rows — shared K/V dequant, per-row
limits, half of v3's spilled GQA state. This probe earns (or refutes)
the engine default flip:

Faces per format:
  1. correctness vs per-row v2 (bare shipping-config calls) at boundary
     shapes: ctx 549 (odd tail), 2051 (limits straddle the final tile),
     and a MIXED-ctx group set (549/1500/2051/8192 — stresses walk_end =
     max(lims) with per-row masking and unequal splits).
  2. timing at the A10 problem cells: G=1 and G=8 row-groups at ctx
     {2048, 8192, 32768} — v4 vs the v2 batched call (row-duplicated
     bt), the shipping incumbent.

Gate: d <= 2e-3 + finite everywhere (expected class: split regrouping,
2-3e-8 per the A11 S-sweep; per-row math is identical over valid lanes).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

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
mx_decode = mxkv.mx_decode
mx_decode_rows = mxkv.mx_decode_rows
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
    """G requests with per-request ctx; returns the v4 call args plus the
    per-row v2 reference operands. Rows of request g sit at g*rtile..+r."""
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
    # per-row limits: the MTP verify pattern ctx-3..ctx
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


def main():
    print(f"p159_v4rows — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    for fmt in FMTS:
        # face 1: correctness vs per-row v2 at boundary shapes
        cases = {
            "ctx549": [549] * 2,
            "ctx2051": [2051] * 2,
            "mixed": [549, 1500, 2051, 8192],
        }
        for name, ctxs in cases.items():
            q, cache, bt, lims = build_multi(ctxs, fmt, 31)
            o4 = mx_decode_rows(q, cache, bt, lims, SCALE)
            o2 = torch.empty_like(o4)
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
            d = float((o4 - o2).abs().amax().item())
            ok = d <= 2e-3 and torch.isfinite(o4).all()
            print(f"[{fmt}] v4 vs per-row v2 {name}: max|d|={d:.3e} "
                  f"{'OK' if ok else 'FAIL'}")
            rc |= 0 if ok else 1
        # face 2: timing at the A10 problem cells, v4 vs v2 batched
        for g_n, ctx in ((1, 2048), (1, 8192), (1, 32768),
                         (8, 2048), (8, 8192), (8, 32768)):
            ctxs = [ctx] * g_n
            q, cache, bt, lims = build_multi(ctxs, fmt, 19)
            bt_rows = bt.repeat_interleave(RTILE, dim=0).contiguous()

            def v2(q=q, c=cache, b=bt_rows, s=lims):
                return mx_decode(q, c, b, s, SCALE)

            def v4(q=q, c=cache, b=bt, s=lims):
                return mx_decode_rows(q, c, b, s, SCALE)

            reps = 5 if ctx >= 32768 else 10
            t2 = timeit(v2, reps)
            t4 = timeit(v4, reps)
            print(f"[{fmt}] G={g_n} ctx={ctx}: v4 {t4:.3f} ms vs "
                  f"v2 {t2:.3f} ms ({t2 / t4:.2f}x)")
    print(f"A12_PROBE_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
