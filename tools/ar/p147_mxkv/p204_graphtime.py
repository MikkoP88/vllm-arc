#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p204_graphtime.py -- arc7 Phase-8 r211 E2'': device-only per-call
times via XPU-graph capture+replay (the engine posture) for the
MX-native decode vs the Triton v2 lane, at short ctx.

E2' left the fair-comparison gap: V3 device times (224-332us at 1K)
compare against v2's ~218us HARNESS number, which still contains v2's
own launch host. The engine replays decode under XPU graphs
(VLLM_XPU_ENABLE_XPU_GRAPH), so the honest 1K comparison is
graph-replay vs graph-replay. Capture N calls into one graph, replay
R times, event-bracket each replay: per-call device = elapsed/(R*N).
This is also the op-level capture SMOKE for the native kernel: V3's
shape has no pageable H2D (pageable-H2D-under-capture law), so capture
must succeed -- failure here would block the engine lane outright.

Cells per format: v2 (Triton, prebuilt tables), native_s5 (direct op,
prebuilt plan, splits=5), native_auto (num_splits=-1, no plan -- the
engine bridge shape). Canary: captured output vs eager reference
after replay. Usage: p204_graphtime.py OUT_DIR. Exit 3 if
torch.xpu.XPUGraph/CUDAGraph is absent (graceful -- rig torch face
probe; this face ships XPUGraph).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch

_MXKV_DIR = (
    Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
)

D = 128
HK = 2
HREP = 4
HQ = HK * HREP
BLOCK = 16
DEV = "xpu"
FMTS = tuple(
    f
    for f in os.environ.get("P204_FMTS", "mxfp8,mxfp4,mxk8v4").split(",")
    if f.strip()
)
CTX = int(os.environ.get("P204_CTX", "1024"))
NCAP = int(os.environ.get("P204_NCAP", "32"))
NREPLAY = int(os.environ.get("P204_REPLAY", "50"))
# "auto" = num_splits=-1 (heuristic; engine bridge shape); ints = forced
# split postures with prebuilt plans. The r210 1K optimum (s=5) was
# calibrated on the HOST-BOUND harness; graph-replay re-prices the
# ladder device-only (E2'' finding: native 1K = 5-7x v2 kernel-side).
SPLIT_CELLS = tuple(
    s.strip()
    for s in os.environ.get("P204_SPLITS", "auto,5,10,16,32").split(",")
    if s.strip()
)
SCALE = 1.0 / math.sqrt(D)


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
mx_decode = mxkv.mx_decode
slot_bytes = mxkv.slot_bytes

from vllm_xpu_kernels.flash_attn_interface import (  # noqa: E402
    _infer_num_xe_cores,
    _kv_tile_from_block_size,
)
from vllm_xpu_kernels.mx_attn_interface import (  # noqa: E402
    _FORMATS,
    _mx_build_decode_split_plan,
)


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


def graph_time(call, ncap, nreplay):
    """Capture ncap calls in one graph; replay-bracketed device time."""
    for _ in range(3):
        call()
    torch.xpu.synchronize()
    g = _GRAPH_CLS()
    with torch.xpu.graph(g):
        for _ in range(ncap):
            call()
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    g.replay()
    torch.xpu.synchronize()
    ev0.record()
    for _ in range(nreplay):
        g.replay()
    ev1.record()
    torch.xpu.synchronize()
    return ev0.elapsed_time(ev1) * 1e3 / (nreplay * ncap)


# torch 2.13+xpu face: the graph class is XPUGraph (CUDAGraph absent)
_GRAPH_CLS = getattr(torch.xpu, "XPUGraph", None) or getattr(
    torch.xpu, "CUDAGraph", None)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    if _GRAPH_CLS is None or not hasattr(torch.xpu, "graph"):
        print("P204_NO_XPU_GRAPH (torch face lacks XPUGraph/CUDAGraph)")
        sys.exit(3)
    print(
        f"p204_graphtime -- device={DEV}, ctx={CTX}, ncap={NCAP}, "
        f"replays={NREPLAY}, torch={torch.__version__}"
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
        qf = q.float()
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        bt = flat_tables(lens, tables)
        torch.xpu.synchronize()

        cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
        sl_dev = torch.tensor(lens, dtype=torch.int32, device=DEV)
        fmt_id = _FORMATS[fmt]
        host_lens = torch.tensor(lens)
        kt = _kv_tile_from_block_size(BLOCK)
        ncores = _infer_num_xe_cores(torch.device(DEV))
        op = torch.ops._vllm_fa2_C.mx_paged_decode

        ref_out = torch.empty_like(q)
        op(q, cache, fmt_id, cu_q, sl_dev, bt, 1, max(lens),
           SCALE, -1, None, None, ref_out)
        ref = ref_out.float().cpu()
        ref_v2 = mx_decode(qf, cache, bt, sl_dev, SCALE,
                           block_size=BLOCK).float().cpu()
        torch.xpu.synchronize()

        cell = {}
        # v2 Triton lane (prebuilt tables -- its engine posture)
        out_v2 = None

        def call_v2(qf=qf, cache=cache, bt=bt, sl_dev=sl_dev):
            nonlocal out_v2
            out_v2 = mx_decode(qf, cache, bt, sl_dev, SCALE, block_size=BLOCK)

        mseq = max(lens)

        # native cells: "auto" (heuristic, engine bridge shape) + forced
        # split postures with prebuilt plans (device-only re-calibration
        # of the r210 1K ladder, which was priced host-bound)
        calls = []
        for spec in SPLIT_CELLS:
            out_c = torch.empty_like(q)
            if spec == "auto":
                def call_c(q=q, cache=cache, fmt_id=fmt_id, cu_q=cu_q,
                           sl_dev=sl_dev, bt=bt, op=op, mk=mseq, oc=out_c):
                    op(q, cache, fmt_id, cu_q, sl_dev, bt, 1, mk,
                       SCALE, -1, None, None, oc)
            else:
                sp = int(spec)
                sc, wc = _mx_build_decode_split_plan(
                    host_lens, kt, sp, ncores, HK)
                sd = sc.to(device=DEV, non_blocking=True)
                wd = wc.to(device=DEV, non_blocking=True)

                def call_c(q=q, cache=cache, fmt_id=fmt_id, cu_q=cu_q,
                           sl_dev=sl_dev, bt=bt, op=op, mk=mseq, oc=out_c,
                           sp=sp, sd=sd, wd=wd):
                    op(q, cache, fmt_id, cu_q, sl_dev, bt, 1, mk,
                       SCALE, sp, sd, wd, oc)
            calls.append((f"native_{spec}", call_c, out_c))

        for name, call, obuf in (("v2", call_v2, None), *calls):
            try:
                us = graph_time(call, NCAP, NREPLAY)
            except Exception as e:  # capture failure is itself the datum
                cell[name] = {"capture_error": str(e)[:200]}
                print(f"  {fmt} {name}: CAPTURE_FAIL {str(e)[:120]}")
                continue
            got = (out_v2 if name == "v2" else obuf).float().cpu()
            tgt = ref_v2 if name == "v2" else ref
            d = got.sub(tgt).abs().max().item()
            cell[name] = {"dev_us": round(us, 2), "canary": round(d, 6)}
            print(
                f"  {fmt} {name}: device {us:.1f}us/call "
                f"(graph {NCAP}x{NREPLAY}) canary {d:.2e}"
            )
        R[fmt] = cell
    vprobe = os.environ.get("ARC_MXKV_VPROBE", "")
    suffix = f"_m{vprobe}" if vprobe else ""
    with open(out / f"p204_graphtime{suffix}.json", "w", newline="\n") as f:
        json.dump(R, f, indent=1)
    print("P204_DONE")


if __name__ == "__main__":
    main()
