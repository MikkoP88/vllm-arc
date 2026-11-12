#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p160_v5dot.py — P6.47 A11 lever 3: the v5 XMX tl.dot decode.

mx_decode_rows_dot (triton_mx_decode.py, A11-NEXT-LEVERS.md lever 3):
the v4 row-group walk with the score path in tl.dot — fp16 operands,
f32 accumulate, M packs R_TILE x N_REP = 24 lanes (padded to 32). The
counter-operand economy the ctrl fp8 lane's fused FA kernel uses.

Faces per format:
  1. correctness vs per-row v2 (bare shipping-config calls) at the
     LIMIT-LAW boundary shapes (ctx 549 odd tail; 2051 final-tile
     straddle; mixed-ctx groups 549/1500/2051/8192). Numerics: fp16
     operand products with f32 accumulation (the ctrl lane's class)
     — expected d ~1e-3, gate d <= 2e-3 + finite.
  2. timing at the A10 problem cells (G 1/8 x ctx 2048/8192/32768):
     v5 vs v4 vs the v2 batched incumbent.

Gate: d <= 2e-3 + finite everywhere.
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
    g*rtile..+r. Returns the v4/v5 call args."""
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


def main():
    print(f"p160_v5dot — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE} "
          f"M_LANES={RTILE * HREP}")
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
            o5 = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            o2 = torch.empty_like(o5)
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
            d = float((o5 - o2).abs().amax().item())
            ok = d <= 2e-3 and torch.isfinite(o5).all()
            print(f"[{fmt}] v5 vs per-row v2 {name}: max|d|={d:.3e} "
                  f"{'OK' if ok else 'FAIL'}")
            rc |= 0 if ok else 1
        # face 2: timing at the A10 problem cells — v5 vs v4 vs v2
        for g_n, ctx in ((1, 2048), (1, 8192), (1, 32768),
                         (8, 2048), (8, 8192), (8, 32768)):
            ctxs = [ctx] * g_n
            q, cache, bt, lims = build_multi(ctxs, fmt, 19)
            bt_rows = bt.repeat_interleave(RTILE, dim=0).contiguous()

            def v2(q=q, c=cache, b=bt_rows, s=lims):
                return mx_decode(q, c, b, s, SCALE)

            def v4(q=q, c=cache, b=bt, s=lims):
                return mx_decode_rows(q, c, b, s, SCALE)

            def v5(q=q, c=cache, b=bt, s=lims):
                return mx_decode_rows_dot(q, c, b, s, SCALE)

            reps = 5 if ctx >= 32768 else 10
            t2 = timeit(v2, reps)
            t4 = timeit(v4, reps)
            t5 = timeit(v5, reps)
            print(f"[{fmt}] G={g_n} ctx={ctx}: v5 {t5:.3f} vs "
                  f"v4 {t4:.3f} vs v2 {t2:.3f} ms "
                  f"({t2 / t5:.2f}x / {t4 / t5:.2f}x)")
    print(f"A13_PROBE_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
