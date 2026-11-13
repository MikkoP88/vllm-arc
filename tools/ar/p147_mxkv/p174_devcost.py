#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p174_devcost.py — P6.47 A15: device-vs-launcher attribution of the
v6 decode call (K2 evidence; P6.47 parity goal, 2026-11-13).

The A14 sweep left mxfp8 G1@32K at 5.25 ms/call vs a ~0.2 ms HBM
envelope — 25x above roofline. Before restructuring the kernel (K2:
persistent CTAs, SLM staging) this probe splits WHERE the 5.25 ms
lives, per face:

  A. wall/call — the p172-style timeit (launcher python + enqueue +
     device, one sync per rep). The number every earlier probe banked.
  B. device/call — XPU event pairs around the call with the queue
     pre-drained: elapsed = device-busy time of the call's kernels.
  C. amortized/call — N back-to-back calls, ONE sync (queue depth N):
     wall/N = launcher+enqueue only (device overlaps behind CPU).
     A - C ~= the device-bound share when CPU can't run ahead.
  D. profiler kernel table — torch.profiler over R reps, kernel-name
     rows with device durations (main walk vs combine vs anything
     else the launcher hides). Skipped gracefully if XPU kineto is
     unavailable (empty trace).
  E. pointer-chase share — identity-bt variant (blocks in natural
     order, zero indirection) vs the standard randperm bt at 32K:
     the delta = the bt->slot->gather dependent-load chain's cost.

Print law: one [fmt] row per (cell, face); A15 summary table at the
end. Loads the INSTALLED mx_kv (cert25za shipping kernels).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

_INSTALLED = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv"
)
_here = Path(__file__).resolve()
_TREE = (
    _here.parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
    if len(_here.parents) > 3
    else _INSTALLED
)


def _load_mxkv():
    src = _INSTALLED if (_INSTALLED / "__init__.py").exists() else _TREE
    spec = importlib.util.spec_from_file_location(
        "mx_kv_lane", src / "__init__.py", submodule_search_locations=[str(src)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, str(src)


mxkv, MXKV_SRC = _load_mxkv()
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


def build_multi(ctxs, fmt, seed, identity_bt=False):
    """p160-law v4/v5 call args; identity_bt orders blocks naturally
    (face E: removes the gather indirection, keeps the byte count)."""
    g = len(ctxs)
    n_blocks = sum((c + BLOCK - 1) // BLOCK for c in ctxs) + 2
    gen = torch.Generator().manual_seed(seed)
    perm = (
        torch.arange(n_blocks)
        if identity_bt
        else torch.randperm(n_blocks, generator=gen)
    )
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
    q = torch.randn(g * RTILE, HK * HREP, D, generator=gen).to(DEV)
    lims = torch.cat(
        [
            (c - 3 + torch.arange(RTILE)).to(torch.int32).to(DEV)
            for c in ctxs
        ]
    )
    return q, cache, bt.contiguous(), lims


def wall_per_call(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
        torch.xpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def device_per_call(fn, reps, warm=3):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    acc = 0.0
    for _ in range(reps):
        ev0.record()
        fn()
        ev1.record()
        torch.xpu.synchronize()
        acc += ev0.elapsed_time(ev1)
    return acc / reps


def amortized_per_call(fn, reps, warm=3, depth=16):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        for _ in range(depth):
            fn()
        torch.xpu.synchronize()
    return (time.perf_counter() - t0) / (reps * depth) * 1e3


def profiler_table(fn, reps, warm=3):
    """Face D: kernel-name device table via torch.profiler; empty list
    when XPU kineto yields no kernel rows (skipped gracefully)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as prof:
            for _ in range(reps):
                fn()
            torch.xpu.synchronize()
    except Exception as exc:  # noqa: BLE001 — optional face
        print(f"  profiler unavailable: {exc}")
        return []
    rows = {}
    for ev in prof.key_averages():
        if getattr(ev, "device_type", None) is not None and "xpu" in str(
            ev.device_type
        ).lower() and ev.self_device_time_total > 0:
            rows[ev.key] = ev.self_device_time_total / reps / 1e3
    if not rows:  # fallback: any rows with device time at all
        rows = {
            ev.key: ev.self_device_time_total / reps / 1e3
            for ev in prof.key_averages()
            if ev.self_device_time_total > 0
            and ev.count >= reps
        }
    return sorted(rows.items(), key=lambda kv: -kv[1])[:6]


def main():
    print(f"p174_devcost — device={DEV}, torch={torch.__version__}")
    print(f"mxkv_src={MXKV_SRC}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    summary = []
    for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
        for ctx in (8192, 32768):
            q, cache, bt, lims = build_multi([ctx], fmt, 19)
            reps = 8 if ctx >= 32768 else 16

            def call(_q=q, _c=cache, _b=bt, _l=lims):
                return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

            a = wall_per_call(call, reps)
            b = device_per_call(call, reps)
            c = amortized_per_call(call, reps)
            fin = torch.isfinite(call().float()).all()
            if not bool(fin):
                rc = 1
            print(f"[{fmt}] G=1 ctx={ctx}: wall {a:.3f} | device {b:.3f} "
                  f"| amortized {c:.3f} ms/call finite={bool(fin)}")
            summary.append((fmt, ctx, a, b, c))
            if ctx == 32768:
                qi, cache_i, bt_i, lims_i = build_multi(
                    [ctx], fmt, 19, identity_bt=True
                )

                def call_i(_q=qi, _c=cache_i, _b=bt_i, _l=lims_i):
                    return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

                bi = device_per_call(call_i, reps)
                print(f"[{fmt}] G=1 ctx={ctx} identity-bt: device {bi:.3f} "
                      f"ms/call (pointer-chase delta {b - bi:.3f})")
        # profiler face once per fmt at 32K
        q, cache, bt, lims = build_multi([32768], fmt, 19)

        def callp(_q=q, _c=cache, _b=bt, _l=lims):
            return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

        print(f"[{fmt}] profiler kernel table (device ms/call):")
        for name, ms in profiler_table(callp, 6):
            print(f"  {ms:.3f}  {name[:88]}")
    print("--- A15 device-cost summary ---")
    for fmt, ctx, a, b, c in summary:
        print(f"[{fmt}] ctx={ctx}: wall {a:.3f} device {b:.3f} "
              f"amort {c:.3f} -> launcher+overlap share {a - b:.3f}")
    print(f"A15_DEVCOST_DONE rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
