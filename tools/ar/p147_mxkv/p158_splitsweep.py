#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p158_splitsweep.py — P6.47 A11 lever 1: the _MX_SPLIT_MAX sweep.

A10 attribution (FP8-PATH-REFERENCE section 7): the 32K decode cliff is
the split cap — s_eff freezes at 32 splits, so each stage-1 program
serially walks 1024 tokens at 32K while ctrl's FA kernel redistributes
work in-kernel. _MX_SPLIT_MAX is a module global read at launch time
(grid z + NUM_SPLITS constexpr + the mid-cache key), so the sweep
monkeypatches it — no kernel edit, every S compiles its own constexpr
instantiation.

Faces: correctness spots (S=32 output vs S=64/128 at ctx 549 — odd
boundary, empty splits park markers — and 8195 — tail boundary; splits
only regroup the online-softmax partials, so the gate is reassociation
noise d <= 2e-3 + finite), then timing at the A10 problem cells:
(8192, 32K) x bsz {1, 8} x S {32, 64, 128}, the verify shapes R=32 @
2048 (A9 continuity) and R=32 @ 32K (the 3714 ms m4 collapse cell).
Bare mx_decode calls — the shipping decoder config.
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
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes
tmx = sys.modules["mx_kv_lane.triton_mx_decode"]

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
SPLITS = (32, 64, 128)


def build(ctx, fmt, seed, bsz=1):
    n_blocks = ctx // BLOCK + 2
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_blocks, generator=gen)
    tab = perm[: (ctx + BLOCK - 1) // BLOCK]
    tb = torch.tensor([int(tab[t // BLOCK]) for t in range(ctx)])
    tp = torch.tensor([t % BLOCK for t in range(ctx)])
    cache = torch.zeros(
        n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
    )
    xk = torch.randn(ctx, HK, D, generator=gen).to(DEV)
    xv = torch.randn(ctx, HK, D, generator=gen).to(DEV)
    mx_store(xk, xv, cache, tb.to(DEV), tp.to(DEV))
    torch.xpu.synchronize()
    bt = tab[None, :].to(DEV).to(torch.int32).expand(bsz, -1).contiguous()
    sl = torch.full((bsz,), ctx, dtype=torch.int32, device=DEV)
    q = torch.randn(bsz, HK * HREP, D, generator=gen).to(DEV)
    return q, cache, bt, sl


def decode_at(s, q, cache, bt, sl):
    tmx._MX_SPLIT_MAX = s
    return mx_decode(q, cache, bt, sl, SCALE)


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
    print(f"p158_splitsweep — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} N_REP={HREP} (true qwen cfg)")
    rc = 0
    for fmt in FMTS:
        # correctness: S=64/128 vs S=32 — reassociation-only regrouping
        for ctx_c in (549, 8195):
            q, cache, bt, sl = build(ctx_c, fmt, 31)
            o32 = decode_at(32, q, cache, bt, sl)
            for s in (64, 128):
                o = decode_at(s, q, cache, bt, sl)
                torch.xpu.synchronize()
                d = float((o - o32).abs().amax().item())
                ok = d <= 2e-3 and torch.isfinite(o).all()
                print(
                    f"[{fmt}] S={s} vs 32 @ctx={ctx_c}: max|d|={d:.3e} "
                    f"{'OK' if ok else 'FAIL'}"
                )
                rc |= 0 if ok else 1
        # timing: the A10 problem cells
        for ctx in (8192, 32768):
            for bsz in (1, 8):
                q2, cache2, bt2, sl2 = build(ctx, fmt, 7, bsz=bsz)
                rows = {}
                for s in SPLITS:
                    rows[s] = round(
                        timeit(
                            lambda q=q2, c=cache2, b=bt2, ls=sl2, s_=s: decode_at(
                                s_, q, c, b, ls
                            ),
                            reps=8,
                        ),
                        4,
                    )
                best = min((v, k) for k, v in rows.items())
                print(
                    f"[{fmt}] ctx={ctx} B={bsz}: best S={best[1]} "
                    f"{best[0]:.3f} ms  {rows}"
                )
        # verify shapes: R=32 @2048 (A9 continuity) and @32K (collapse cell)
        for ctx_v in (2048, 32768):
            bsz = 8
            qv, cachev, btv, _ = build(ctx_v, fmt, 19, bsz=bsz)
            rv = bsz * 4
            gen = torch.Generator().manual_seed(23)
            q_v = torch.randn(rv, HK * HREP, D, generator=gen).to(DEV)
            bt_rows = btv.repeat_interleave(4, dim=0).contiguous()
            sl_v = (ctx_v - 3 + (torch.arange(rv) % 4)).to(torch.int32).to(DEV)
            rows = {}
            for s in SPLITS:
                rows[s] = round(
                    timeit(
                        lambda q=q_v, c=cachev, b=bt_rows, ls=sl_v, s_=s: decode_at(
                            s_, q, c, b, ls
                        ),
                        reps=5,
                        warm=1,
                    ),
                    4,
                )
            best = min((v, k) for k, v in rows.items())
            print(
                f"[{fmt}] verify R={rv} @ctx={ctx_v}: best S={best[1]} "
                f"{best[0]:.3f} ms  {rows}"
            )
    tmx._MX_SPLIT_MAX = 32
    print(f"A11_PROBE_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
