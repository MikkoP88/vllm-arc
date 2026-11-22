#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p203_hostab.py -- arc7 Phase-8 r211 E2': host-path A/B for the
MX-NATIVE paged decode interface at ctx=1024 (the 1K blocker).

r211 E1' verdict: 1K decode is HOST-LAUNCH-BOUND (dev==host, device
gap 2us = python loop overhead; even the mode-3 empty floor kernel
"reads" 330-840us). The host cost decomposes into: harness rebuilds
(arange cu_q, tensor(lens) H2D, host_kv_lens), the wrapper's per-call
work (_as_int32_device_tensor H2D, _infer_num_xe_cores, python split
plan + two CPU tensors + two H2D moves, empty_like), and the C++ op
entry + launch. This tool prices each segment and A/Bs call variants:

  V0_rebuild         native_call verbatim (rebuild + full wrapper)
  V1_prebuilt        wrapper entry with cu_q/sl/bt/host_kv_lens
                     prebuilt (plan still per call)
  V2_prebuilt_s1     V1 at splits=1 (no plan path at all)
  V3_direct_op       torch.ops._vllm_fa2_C.mx_paged_decode with
                     everything prebuilt incl. splits/work/out
                     (pure C++ entry + kernel launch)

Also the graph-capturability corollary (pageable-H2D-under-capture
law): V3's shape is what the engine needs for XPU-graph capture --
per-call H2D in V0-V2 is the blocker for the engine lane. NaN-poison
canary per variant (r207 law). Usage: p203_hostab.py OUT_DIR.
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

from vllm_xpu_kernels.flash_attn_interface import (  # noqa: E402
    _as_int32_device_tensor,
    _infer_num_xe_cores,
    _kv_tile_from_block_size,
)
from vllm_xpu_kernels.mx_attn_interface import (  # noqa: E402
    _FORMATS,
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
CTX = int(os.environ.get("P203_CTX", "1024"))
SPLITS = int(os.environ.get("P203_SPLITS", "5"))
REPS = int(os.environ.get("P203_REPS", "60"))
SCALE = 1.0 / math.sqrt(D)


def build_paged(total_tokens, seed=7):
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
        softmax_scale=SCALE,
        num_splits_kv=splits,
        host_kv_lens=torch.tensor(lens) if splits > 1 else None,
        out=out,
    )


def med_us(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2], 2)


def time_segment(fn, n=100):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        ts.append((time.perf_counter_ns() - t0) / 1e3)
    return med_us(ts)


def stats(xs):
    xs = sorted(xs)
    n = len(xs)

    def pct(p):
        return xs[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "med": round(xs[n // 2], 4),
        "p10": round(pct(0.10), 4),
        "p90": round(pct(0.90), 4),
    }


def decompose(call, reps, warm):
    for _ in range(warm):
        call()
    torch.xpu.synchronize()
    ev_s = [torch.xpu.Event(enable_timing=True) for _ in range(reps)]
    ev_e = [torch.xpu.Event(enable_timing=True) for _ in range(reps)]
    host = []
    for i in range(reps):
        h0 = time.perf_counter_ns()
        ev_s[i].record()
        call()
        ev_e[i].record()
        host.append((time.perf_counter_ns() - h0) / 1e3)
    torch.xpu.synchronize()
    dev = [ev_s[i].elapsed_time(ev_e[i]) * 1e3 for i in range(reps)]
    gap = [ev_e[i].elapsed_time(ev_s[i + 1]) * 1e3 for i in range(reps - 1)]
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    ev0.record()
    for _ in range(reps):
        call()
    ev1.record()
    torch.xpu.synchronize()
    return {
        "dev_us": stats(dev),
        "gap_us": stats(gap),
        "host_us": stats(host),
        "bracketed_us": round(ev0.elapsed_time(ev1) * 1e3 / reps, 4),
    }


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"p203_hostab -- device={DEV}, ctx={CTX}, splits={SPLITS}, "
        f"reps={REPS}, torch={torch.__version__}"
    )
    R = {"segments": {}}
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
            n_blocks, BLOCK, HK, slot_bytes(D, fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        bt = flat_tables(lens, tables)
        torch.xpu.synchronize()

        # ---- segment pricing (once per format; all host-side) --------
        host_lens = torch.tensor(lens)
        kt = _kv_tile_from_block_size(BLOCK)
        ncores = _infer_num_xe_cores(torch.device(DEV))
        seg = {
            "arange_cuq": time_segment(
                lambda lens=lens: torch.arange(
                    len(lens) + 1, dtype=torch.int32, device=DEV)),
            "sl_dev_h2d": time_segment(
                lambda lens=lens: torch.tensor(
                    lens, dtype=torch.int32, device=DEV)),
            "host_lens_build": time_segment(
                lambda lens=lens: torch.tensor(lens)),
            "as_int32_h2d": time_segment(
                lambda hl=host_lens: _as_int32_device_tensor(
                    hl, torch.device(DEV))),
            "infer_xe_cores": time_segment(
                lambda: _infer_num_xe_cores(torch.device(DEV))),
            "plan_build_plus_h2d": time_segment(
                lambda hl=host_lens, kt=kt, nc=ncores: (
                    _mx_build_decode_split_plan(
                        hl, kt, SPLITS, nc, HK)[0].to(
                            device=DEV, non_blocking=True))),
            "empty_like_q": time_segment(
                lambda q=q: torch.empty_like(q)),
        }
        R["segments"][fmt] = seg
        print(f"  segments/{fmt}: " + " ".join(
            f"{k}={v}us" for k, v in seg.items()))

        # ---- prebuilt constants --------------------------------------
        cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
        sl_dev = _as_int32_device_tensor(host_lens, torch.device(DEV))
        out_pre = torch.empty_like(q)
        fmt_id = _FORMATS[fmt]
        splits_cpu, work_cpu = _mx_build_decode_split_plan(
            host_lens, kt, SPLITS, ncores, HK)
        splits_dev = splits_cpu.to(device=DEV, non_blocking=True)
        work_dev = work_cpu.to(device=DEV, non_blocking=True)
        op = torch.ops._vllm_fa2_C.mx_paged_decode

        ref1 = native_call(q, cache, fmt, lens, bt, 1).float().cpu()
        torch.xpu.synchronize()

        mseq = max(lens)
        variants = {
            "V0_rebuild": lambda q=q, cache=cache, fmt=fmt, lens=lens, bt=bt: (  # noqa: E731
                native_call(q, cache, fmt, lens, bt, SPLITS)),
            "V1_prebuilt": (
                lambda q=q, cache=cache, fmt=fmt, bt=bt, cu_q=cu_q,
                    sl_dev=sl_dev, hl=host_lens, mk=mseq:
                mx_paged_decode(
                    q, cache, fmt, cu_q, sl_dev, bt,
                    max_seqlen_k=mk, softmax_scale=SCALE,
                    num_splits_kv=SPLITS, host_kv_lens=hl)),
            "V2_prebuilt_s1": (
                lambda q=q, cache=cache, fmt=fmt, bt=bt, cu_q=cu_q,
                    sl_dev=sl_dev, mk=mseq:
                mx_paged_decode(
                    q, cache, fmt, cu_q, sl_dev, bt,
                    max_seqlen_k=mk, softmax_scale=SCALE,
                    num_splits_kv=1, host_kv_lens=None)),
            "V3_direct_op": (
                lambda q=q, cache=cache, fmt_id=fmt_id, cu_q=cu_q,
                    sl_dev=sl_dev, bt=bt, mk=mseq, sd=splits_dev,
                    wd=work_dev, opre=out_pre, op=op:
                op(q, cache, fmt_id, cu_q, sl_dev, bt, 1, mk,
                   SCALE, SPLITS, sd, wd, opre)),
        }
        cell = {}
        for name, call in variants.items():
            if name == "V3_direct_op":
                out_pre.fill_(float("nan"))
                call()
                d = out_pre.float().cpu().sub(ref1).abs().max().item()
            else:
                d = call().float().cpu().sub(ref1).abs().max().item()
            if math.isnan(d):
                print(f"  !! {fmt} {name}: CANARY_NAN")
                d = -1.0
            r = decompose(call, REPS, 5)
            r["canary_vs_splits1"] = round(d, 6)
            cell[name] = r
            dv, gv, hv = r["dev_us"], r["gap_us"], r["host_us"]
            print(
                f"  {fmt} {name}: host {hv['med']:.1f}us dev {dv['med']:.1f}us "
                f"gap {gv['med']:.1f}us bracketed {r['bracketed_us']:.1f}us "
                f"canary {d:.2e}"
            )
        R[fmt] = cell
    with open(out / "p203_hostab.json", "w", newline="\n") as f:
        json.dump(R, f, indent=1)
    print("P203_DONE")


if __name__ == "__main__":
    main()
