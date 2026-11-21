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

# Geometry: env-parameterized since r212b (fail2 law) — the r208-r212
# ladder ran D=128/HREP=4/BLOCK=16, calibrated off the config's GDN
# linear head dim. The e2e target qwen3.8-27b-fp8 (TP2) is head_dim 256,
# HQ 12 / HK 2 (G 6), engine pool page 64: P204_D=256 P204_HREP=6
# P204_BLOCK=64 reproduces the engine's real kernel geometry.
D = int(os.environ.get("P204_D", "128"))
HK = 2
HREP = int(os.environ.get("P204_HREP", "4"))
HQ = HK * HREP
BLOCK = int(os.environ.get("P204_BLOCK", "16"))
DEV = "xpu"
FMTS = tuple(
    f
    for f in os.environ.get("P204_FMTS", "mxfp8,mxfp4,mxk8v4").split(",")
    if f.strip()
)
CTX = int(os.environ.get("P204_CTX", "1024"))
# r212d-diag engine-shape knobs (2026-11-21): r212b/r212c falsified
# the split/walk-extent theories e2e (max_seqlen_k 9 vs 262400 gave
# byte-identical ITLs; the walk is device-limit-bounded). The remaining
# engine-vs-harness deltas these knobs reproduce one at a time:
#   P204_POOLBLKS  over-allocate the pool (engine: 11281 blocks =
#                  721994 tokens; harness: ctx/64+2)
#   P204_BTWIDTH   pad the block table to WIDTH columns (engine: 4100
#                  = max_model_len 262144/block, constant per bucket)
#   P204_BTPAD     zero (harness default) | stale (random valid block
#                  ids — the engine's stale beyond-fill entries)
#   P204_MAXSK     ctx (harness default: max real len) | cap
#                  (bt_width x block — the r212c bridge posture) | any
#                  int literal (r212f: the engine's DUMMY-diluted
#                  freeze — 9/13/18/5 observed across buckets)
POOLBLKS = int(os.environ.get("P204_POOLBLKS", "0"))
BTWIDTH = int(os.environ.get("P204_BTWIDTH", "0"))
BTPAD = os.environ.get("P204_BTPAD", "zero")
MAXSK = os.environ.get("P204_MAXSK", "ctx")
# r212i2: forced-split plans must COVER max_seqlen_k. Plans built from
# real lens under a larger mk (r212i: MAXSK=cap, cap = bt_width*64 >
# max real len by up to BLOCK-1) left outputs unwritten — every forced
# cell canaried at |0 - ref|max while auto stayed 0 (r210 passed only
# because MAXSK=ctx made plan lens == mk exactly). "cap" builds plans
# from the capacity lens so the plan covers the mk-priced partition.
PLANLEN = os.environ.get("P204_PLANLEN", "real")
# r212i3 diagnostics (default OFF — banked default behavior unchanged):
#   P204_POISON    1 = NaN-fill native out buffers pre-capture
#   P204_ROWPROBE  1 = per-row match matrix for canary-failing cells
POISON = os.environ.get("P204_POISON", "0") == "1"
ROWPROBE = os.environ.get("P204_ROWPROBE", "0") == "1"
NCAP = int(os.environ.get("P204_NCAP", "32"))
NREPLAY = int(os.environ.get("P204_REPLAY", "50"))
# E5b (r212a): ENGINE VERIFY SHAPE — B requests x r_tile=4 draft rows,
# the spec-k3 batch the engine actually decodes with. The E5 map's
# independent-row geometry under-prices the short-ctx side twice: both
# kernels walk R=B*r rows (not 2), and mxfp8/mxk8v4's engine short lane
# is v5 (mx_decode_rows_dot), not v2. Adds cells v2_es/v5_es/native_es.
ESHAPE = os.environ.get("P204_ESHAPE", "0") == "1"
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
mx_decode_rows_dot = mxkv.mx_decode_rows_dot
slot_bytes = mxkv.slot_bytes

from vllm_xpu_kernels.flash_attn_interface import (  # noqa: E402
    _infer_num_xe_cores,
)
from vllm_xpu_kernels.mx_attn_interface import (  # noqa: E402
    _FORMATS,
    _mx_build_decode_split_plan,
)


def build_paged(total_tokens, seed=7):
    n_blocks = total_tokens // BLOCK + 2
    if n_blocks < POOLBLKS:
        # r212d-diag: engine-sized pool (surface-split arithmetic sees
        # the real block count; tables stay at their own lengths)
        n_blocks = POOLBLKS
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
        if bt.shape[1] < BTWIDTH:
            # r212d-diag: engine-width block table (4100 columns);
            # beyond-fill entries zero (harness default) or stale
            # random-valid ids (engine reality)
            pad_w = BTWIDTH - bt.shape[1]
            if BTPAD == "stale":
                g2 = torch.Generator().manual_seed(99)
                pad = torch.randint(
                    0, cache.shape[0], (bt.shape[0], pad_w),
                    generator=g2, dtype=torch.int32,
                ).to(DEV)
            else:
                pad = torch.zeros(
                    bt.shape[0], pad_w, dtype=torch.int32, device=DEV)
            bt = torch.cat([bt, pad], dim=1)
        torch.xpu.synchronize()

        cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
        sl_dev = torch.tensor(lens, dtype=torch.int32, device=DEV)
        fmt_id = _FORMATS[fmt]
        host_lens = torch.tensor(lens)
        if PLANLEN == "cap":
            host_lens = torch.full_like(host_lens, bt.shape[1] * BLOCK)
        # r212j TILE-UNIT LAW (2026-11-21, source-proven): the MX decode
        # kernel walks 16-row tiles at EVERY page (mx_paged_decode_xe2.cpp
        # dispatch — page-64 pools ride the _16 policy per the r212c SG
        # law; "every admitted geometry walks 16-row tiles"). The stock
        # mirror _kv_tile_from_block_size returns the STOCK kernel's unit
        # (16/32/64 by page) — at BLOCK=64 the builder partitioned in
        # 64-token tiles while the kernel reads wl_tile_start/count as
        # 16-row indices -> plans covered 1/4 of the KV -> the r212i/i2/i3
        # wrong-values law. BLOCK=16 coincided (16==16), which is why the
        # r210 ladder passed.
        kt = int(os.environ.get("P204_KVT", "16"))
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
        if MAXSK == "cap":
            # r212d-diag: the r212c bridge posture — capacity-priced
            mseq = bt.shape[1] * BLOCK
        elif MAXSK != "ctx":
            # r212f: literal — the engine's DUMMY-diluted freeze
            # (r212b telemetry observed 9/13/18/5 across buckets).
            # The no-op-falsification audit (2026-11-21): the r212c
            # bridge "fix" only changed the TELEMETRY line — the call
            # still passed int(max_seq_len) — so r212c_min never
            # tested capacity pricing, and the r212d cells (ctx/cap)
            # never tested the value the engine actually freezes.
            # This cell closes that hole.
            mseq = int(MAXSK)

        # native cells: "auto" (heuristic, engine bridge shape) + forced
        # split postures with prebuilt plans (device-only re-calibration
        # of the r210 1K ladder, which was priced host-bound)
        calls = []
        for spec in SPLIT_CELLS:
            out_c = torch.empty_like(q)
            # r212i3: NaN-poison native out buffers (the stale-buffer
            # canary law) — a NaN that SURVIVES warmup+capture+replay
            # marks elements the op never wrote; distinguishing
            # never-written from written-wrong for the forced-plan
            # canaries. ref_absmax prices the |0-ref| scale.
            if POISON:
                out_c.fill_(float("nan"))
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
            if POISON and name != "v2":
                nan_left = int(torch.isnan(got).sum())
                cell[name]["nan_left"] = nan_left
                cell[name]["ref_absmax"] = round(
                    float(tgt.abs().max()), 6)
            if ROWPROBE and name != "v2" and d > 2e-3:
                # r212i3: permutation signature — per (i, j) row-pair
                # max |got_i - ref_j|. A layout/order bug shows 0 at
                # the permuted partner; wrong math shows none clean.
                n = got.shape[0]
                m = [
                    [
                        round(float(got[i].sub(tgt[j]).abs().max()), 6)
                        for j in range(n)
                    ]
                    for i in range(n)
                ]
                cell[name]["rowmatch"] = m
                print(f"  {fmt} {name}: rowmatch {m}")
            print(
                f"  {fmt} {name}: device {us:.1f}us/call "
                f"(graph {NCAP}x{NREPLAY}) canary {d:.2e}"
            )
        if ESHAPE:
            # engine verify shape: same pool/tables, r_tile=4 rows per
            # request (v5's grouped tables + per-row limits; native's
            # per-row one-query sequences). Per-lane eager canaries.
            r_t = 4
            q_e = q.repeat_interleave(r_t, dim=0)
            qf_e = qf.repeat_interleave(r_t, dim=0)
            bt_rows_e = bt.repeat_interleave(r_t, dim=0)
            sl_e = sl_dev.repeat_interleave(r_t)
            cu_e = torch.arange(
                len(lens) * r_t + 1, dtype=torch.int32, device=DEV)
            out_n = torch.empty_like(q_e)
            out_es: dict[str, torch.Tensor] = {}

            def c_v2e(qf_e=qf_e, cache=cache, btre=bt_rows_e, sle=sl_e,
                      store=out_es):
                store["v2_es"] = mx_decode(
                    qf_e, cache, btre, sle, SCALE, block_size=BLOCK)

            def c_v5e(qe=q_e, cache=cache, btg=bt, sle=sl_e, rt=r_t,
                      store=out_es):
                store["v5_es"] = mx_decode_rows_dot(
                    qe, cache, btg, sle, SCALE, block_size=BLOCK,
                    r_tile=rt)

            def c_nae(qe=q_e, cache=cache, fmt_id=fmt_id, cu_e=cu_e,
                      sle=sl_e, btre=bt_rows_e, op=op, mk=mseq, oc=out_n):
                op(qe, cache, fmt_id, cu_e, sle, btre, 1, mk,
                   SCALE, -1, None, None, oc)

            torch.xpu.synchronize()
            ref_es = {}
            ref_es["v2_es"] = mx_decode(
                qf_e, cache, bt_rows_e, sl_e, SCALE,
                block_size=BLOCK).float().cpu()
            ref_es["v5_es"] = mx_decode_rows_dot(
                q_e, cache, bt, sl_e, SCALE, block_size=BLOCK,
                r_tile=r_t).float().cpu()
            oer = torch.empty_like(q_e)
            op(q_e, cache, fmt_id, cu_e, sl_e, bt_rows_e, 1, mseq,
               SCALE, -1, None, None, oer)
            ref_es["native_es"] = oer.float().cpu()
            torch.xpu.synchronize()
            # cross-lane truth for the v5 block-16 walk (fp16-dot class
            # vs v2: gate ~2e-3 per the kernel's own docstring)
            cell["v5_vs_v2_ref"] = round(
                float(ref_es["v5_es"].sub(ref_es["v2_es"]).abs().max()), 6)
            print(f"  {fmt} v5_es-vs-v2_es eager cross: "
                  f"{cell['v5_vs_v2_ref']:.2e}")
            for name, call in (
                ("v2_es", c_v2e), ("v5_es", c_v5e), ("native_es", c_nae),
            ):
                try:
                    us = graph_time(call, NCAP, NREPLAY)
                except Exception as e:
                    cell[name] = {"capture_error": str(e)[:200]}
                    print(f"  {fmt} {name}: CAPTURE_FAIL {str(e)[:120]}")
                    continue
                got = (
                    out_n if name == "native_es" else out_es[name]
                ).float().cpu()
                d = got.sub(ref_es[name]).abs().max().item()
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
