#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p157_v3_probe.py — P6.47: fast compile + numeric probe for the v3
GQA-batched split decoder (triton_mx_decode._mx_decode_split_gqa) before
a full A7 battery. Compiles both unpack flavors for every format, then
checks v3 vs v2 device-side on a 549-token fragmented case at the qwen
head config (HK=8, Hq=32, N_REP=4) — max|d| <= 2e-3 gate. Also times one
32K telemetry point per format as an early performance signal.

Standalone (not part of the A/B gate battery): exit 0 = probe clean.
"""

from __future__ import annotations

import importlib
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
FMTS = ("mxfp8", "mxfp4", "mxk8v4")


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
    print(f"p157_v3_probe — device={DEV}, torch={torch.__version__}")
    rc = 0
    for fmt in FMTS:
        # compile both flavors on a small case first
        q, cache, bt, sl = build(549, fmt, 31)
        for shift in (False, True):
            tmx._SHIFT = shift
            try:
                o3 = mx_decode(q, cache, bt, sl, SCALE, decoder="v3")
                torch.xpu.synchronize()
            except Exception as exc:  # noqa: BLE001 — probe reports and continues
                print(f"[{fmt} shift={shift}] v3 COMPILE/RUN FAIL: {exc}")
                rc = 1
                continue
            o2 = mx_decode(q, cache, bt, sl, SCALE, decoder="v2")
            torch.xpu.synchronize()
            d = float((o3 - o2).abs().amax().item())
            ok = d <= 2e-3 and torch.isfinite(o3).all()
            print(
                f"[{fmt} shift={shift}] v3-vs-v2 max|d|={d:.3e} "
                f"{'OK' if ok else 'FAIL'}"
            )
            rc |= 0 if ok else 1
        # 32K timing signal (join flavor, tuned defaults)
        tmx._SHIFT = False
        q2, cache2, bt2, sl2 = build(32768, fmt, 7)
        for name in ("v2", "v3"):
            for _ in range(3):
                mx_decode(q2, cache2, bt2, sl2, SCALE, decoder=name)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                mx_decode(q2, cache2, bt2, sl2, SCALE, decoder=name)
            torch.xpu.synchronize()
            ms = (time.perf_counter() - t0) / 20 * 1e3
            print(f"[{fmt}] {name} @32K: {ms:.3f} ms")
    print(f"V3_PROBE_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
