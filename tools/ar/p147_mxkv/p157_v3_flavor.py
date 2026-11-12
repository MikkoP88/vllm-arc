#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p157_v3_flavor.py — P6.47: close the A7b tune gap. The A7b tune face
swept v3join only; telemetry showed v3shift is the mxk8v4 champion
(2.247 vs 3.899 ms @32K) because the shared K/V walk makes the V-leg
e2m1 dequant the hot op (shift = no cross-lane join), while mxfp4
prefers join. This probe sweeps (BLOCK_N, num_warps, num_stages) x
(join, shift) for v3 on mxk8v4 and mxfp8 @32K to fix the final
per-format table (flavor included). Times only — correctness is the
A/B battery's gate (v3 flavors are bit-equal to v2 per legA7b exact).
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

D = 128
HK = 8
BLOCK = 64
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
COMBOS = ((64, 4, 3), (64, 4, 2), (64, 8, 2), (64, 8, 3), (128, 4, 3))


def build(ctx, fmt, seed):
    n_blocks = ctx // BLOCK + 2
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_blocks, generator=gen)
    tab = perm[: (ctx + BLOCK - 1) // BLOCK]
    tb = torch.tensor([tab[t // BLOCK].item() for t in range(ctx)])
    tp = torch.tensor([t % BLOCK for t in range(ctx)])
    cache = torch.zeros(
        n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
    )
    xk = torch.randn(ctx, HK, D, generator=gen).to(DEV)
    xv = torch.randn(ctx, HK, D, generator=gen).to(DEV)
    mx_store(xk, xv, cache, tb.to(DEV), tp.to(DEV))
    torch.xpu.synchronize()
    bt = tab[None, :].to(DEV).to(torch.int32)
    sl = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    q = torch.randn(1, HK * 4, D, generator=gen).to(DEV)
    return q, cache, bt, sl


def main():
    print(f"p157_v3_flavor — device={DEV}, torch={torch.__version__}")
    for fmt in ("mxk8v4", "mxfp8", "mxfp4"):
        q, cache, bt, sl = build(32768, fmt, 7)
        rows = {}
        for shift in (False, True):
            for bn, nw, ns in COMBOS:
                tmx._SHIFT = shift
                tag = f"{'shift' if shift else 'join'}n{bn}w{nw}s{ns}"
                try:
                    for _ in range(3):
                        mx_decode(
                            q, cache, bt, sl, SCALE, decoder="v3",
                            block_n=bn, num_warps=nw, num_stages=ns,
                        )
                    torch.xpu.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(20):
                        mx_decode(
                            q, cache, bt, sl, SCALE, decoder="v3",
                            block_n=bn, num_warps=nw, num_stages=ns,
                        )
                    torch.xpu.synchronize()
                    rows[tag] = round((time.perf_counter() - t0) / 20 * 1e3, 4)
                except Exception as exc:  # noqa: BLE001 — record, keep sweeping
                    rows[tag] = f"err:{type(exc).__name__}"
        best = min((v, k) for k, v in rows.items() if not isinstance(v, str))
        print(f"[{fmt}] best {best[1]} {best[0]:.3f} ms")
        print(f"[{fmt}] sweep {rows}")
    print("V3_FLAVOR_DONE")


if __name__ == "__main__":
    main()
