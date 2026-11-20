#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p202_timeline.py -- arc7 Phase-8 r211 E1': 1K timeline decomposition
for the MX-NATIVE paged-decode kernel (mxfp8 short-ctx floor pathology).

The r210 discovery: at ctx=1024 the mxfp8 floor cells read ~0.87 ms
(probe m2/m3/m5/m6) vs mxfp4 ~0.24 for a grid of only B2 x HK2 x splits
workgroups -- ~100x any plausible empty-kernel time. The harness events
in p199/p200h bracket the WHOLE rep loop, so inter-launch turnaround
lands inside the "device" number. This tool decomposes per call:

  dev[i]  = elapsed(ev_s[i], ev_e[i])   -- kernel occupancy per call
  gap[i]  = elapsed(ev_e[i], ev_s[i+1]) -- device-timeline idle between
            calls (dispatch/queue turnaround)
  host[i] = perf_counter_ns around (record + launch + record), never
            synchronizing inside the loop -- enqueue cost per call

plus one classic bracketed cell (ev0, N calls, ev1) on the same tensors
so the decomposition sums against the number the r208/r210 chains
actually reported. Verdict logic (r211 plan):
  bracketed ~= dev            -> the kernel really is slow (H2'/in-kernel)
  dev tiny, gap ~= host       -> host-side dispatch turnaround (H1' host)
  dev tiny, gap >> host       -> queue/scheduler launch cost (H1' device)

Mode comes from the environment (ARC_MXKV_VPROBE, set by r211_e1.sh --
one process per mode; the kernel reads it at launch). NaN-poison canary
per cell (r207 law). Usage: p202_timeline.py OUT_DIR.
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
    mx_paged_decode,
)

D = 128
HK = 2
HREP = 4
HQ = HK * HREP
BLOCK = 16
DEV = "xpu"
FMTS = tuple(
    f
    for f in os.environ.get("P202_FMTS", "mxfp8,mxfp4,mxk8v4").split(",")
    if f.strip()
)
CTX = int(os.environ.get("P202_CTX", "1024"))
# r210 postures at 1K: 5 = the heuristic optimum, 1 = the single-split
# floor posture (latency-bound, 1.5x worse end-to-end but the cleanest
# turnaround probe -- one workgroup chain, no reduce dependency).
SPLITS = tuple(
    int(s)
    for s in os.environ.get("P202_SPLITS", "5,1").split(",")
    if s.strip()
)
REPS = int(os.environ.get("P202_REPS", "60"))
WARM = int(os.environ.get("P202_WARM", "5"))
VPROBE = os.environ.get("ARC_MXKV_VPROBE", "0")


def build_paged(total_tokens, seed=7):
    """Fragmented block-16 tables, p198/p199/p200h geometry verbatim."""
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


def stats(xs):
    xs = sorted(xs)
    n = len(xs)

    def pct(p):
        return xs[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "med": round(xs[n // 2], 4),
        "p10": round(pct(0.10), 4),
        "p90": round(pct(0.90), 4),
        "mean": round(sum(xs) / n, 4),
    }


def decompose(call, reps, warm):
    """Per-rep event pairs with no intermediate sync + bracketed cell."""
    for _ in range(warm):
        call()
    torch.xpu.synchronize()
    ev_s = [torch.xpu.Event(enable_timing=True) for _ in range(reps)]
    ev_e = [torch.xpu.Event(enable_timing=True) for _ in range(reps)]
    host = []
    t_loop0 = time.perf_counter()
    for i in range(reps):
        h0 = time.perf_counter_ns()
        ev_s[i].record()
        call()
        ev_e[i].record()
        host.append((time.perf_counter_ns() - h0) / 1e3)
    wall_loop = (time.perf_counter() - t_loop0) / reps
    torch.xpu.synchronize()
    dev = [ev_s[i].elapsed_time(ev_e[i]) * 1e3 for i in range(reps)]
    gap = [ev_e[i].elapsed_time(ev_s[i + 1]) * 1e3 for i in range(reps - 1)]
    # classic p199/p200h posture: bracket N calls with one event pair
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    t0 = time.perf_counter()
    ev0.record()
    for _ in range(reps):
        call()
    ev1.record()
    torch.xpu.synchronize()
    wall = (time.perf_counter() - t0) / reps
    bracketed = ev0.elapsed_time(ev1) * 1e3 / reps
    return {
        "dev_us": stats(dev),
        "gap_us": stats(gap),
        "host_us": stats(host),
        "bracketed_us": round(bracketed, 4),
        "wall_us": round(wall * 1e6, 4),
        "wall_loop_us": round(wall_loop * 1e6, 4),
        "reps": reps,
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"p202_timeline -- device={DEV}, vprobe={VPROBE}, ctx={CTX}, "
        f"reps={REPS}, torch={torch.__version__}"
    )
    R = {}
    for fmt in FMTS:
        gen = torch.Generator().manual_seed(CTX)
        xk = torch.randn(CTX, HK, D, generator=gen).to(DEV)
        xv = torch.randn(CTX, HK, D, generator=gen).to(DEV)
        lens, tables, tb, tp, n_blocks = build_paged(CTX, seed=CTX % 97)
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
        ref1 = native_call(q, cache, fmt, lens, bt, 1).float().cpu()
        cell = {}
        for sp in SPLITS:
            call = lambda q=q, cache=cache, fmt=fmt, lens=lens, bt=bt, sp=sp: (  # noqa: E731
                native_call(q, cache, fmt, lens, bt, sp)
            )
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
                print(f"  !! {fmt} splits={sp}: CANARY_NAN (no launch?)")
                d_canary = -1.0
            r = decompose(call, REPS, WARM)
            r["canary_vs_splits1"] = round(d_canary, 6)
            cell[str(sp)] = r
            d, g, h = r["dev_us"], r["gap_us"], r["host_us"]
            print(
                f"  {fmt} splits={sp:2d}: dev {d['med']:.1f}us "
                f"[{d['p10']:.1f}-{d['p90']:.1f}] gap {g['med']:.1f}us "
                f"host {h['med']:.1f}us bracketed {r['bracketed_us']:.1f}us "
                f"canary {d_canary:.2e}"
            )
        R[fmt] = cell
    mode = VPROBE if VPROBE else "0"
    with open(out / f"p202_timeline_m{mode}.json", "w", newline="\n") as f:
        json.dump(R, f, indent=1)
    print("P202_DONE")


if __name__ == "__main__":
    main()
