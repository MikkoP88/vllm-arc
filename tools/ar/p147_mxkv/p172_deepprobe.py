#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p172_deepprobe.py — P6.47 A14 deep-attribution kernel probe.

Owner PRIORITY 1 directive 2026-11-13: locate the ROOT delays of the MX
KV decode lane vs the fp8 ctrl. The r162 e2e ladder isolated conc-1 ITL
deltas (32K ctx: m8 172.9 ms / m4 112.0 vs q8 42.6; per-attn-layer
deltas 8.15 / 4.34 ms) but could not split WHERE inside the step the
time goes. This probe measures every device phase of the MX lane at the
exact r162 problem shapes (qwen3.8-27b TP2: D=256 HK=4 Hq=24,
MTP spec=3 -> R_TILE=4 verify rows):

  A. verify-decode kernel — the SHIPPING dispatch per format (v5
     mx_decode_rows_dot for mxfp8/mxk8v4, the v2 batch for mxfp4) plus
     the alternates, G 1/8 x ctx 1024/8192/32768. Kernel x 16 attn
     layers is the graph-replay attn share of the step.
  B. per-layer AUX pipeline — q index_select gather + output
     index_copy_ + dtype cast around the kernel (what
     _split_verify_decode runs once per attn layer). aux =
     (kernel+aux) - kernel.
  C. plan-build cost — the once-per-step device-op chain (arange/full/
     cat/index_select/div/argmax, engine.py _split_verify_decode),
     amortized /16 layers. Under XPU-graph replay this replays as
     captured kernels every step.
  D. mx_store (write-quant) at prefill shapes ctx 8192/32768 — the
     per-layer first-chunk prefill tax (m8 TTFT +2.95 s at 8K suspect).
  E. mx_dequant_cache (continuation-prefill prefix dequant) at cached
     2048/8192 — the re-dequant tax every continuation chunk pays per
     layer; quadratic over chunked prefill.

Loads the INSTALLED site-packages mx_kv (the shipping cert25z kernels)
when present, else the repo tree. Measurement leg: the only gate is
finite outputs (no NaN/Inf decode) — deltas are banked, never passed/
failed here. Print law: `[fmt] ...` rows + A14 summary table.
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
# repo-tree fallback only when the probe actually sits 4-deep in the
# tree (in-container copies at /tmp or a mount root have no parents[3]
# — eager evaluation crashed the phase-0 probe before the installed
# check could win; r172 leg 07:17)
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
mx_decode = mxkv.mx_decode
mx_decode_rows = mxkv.mx_decode_rows
mx_decode_rows_dot = mxkv.mx_decode_rows_dot
mx_store = mxkv.mx_store
mx_dequant_cache = getattr(mxkv, "mx_dequant_cache", None)
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
# cert25z engine module top: v5 default for these, v2 batch otherwise
V5_DEFAULT = ("mxfp8", "mxk8v4")


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """G requests with per-request ctx; rows of request g sit at
    g*rtile..+r. Returns the v4/v5 call args (p160 law)."""
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


def plan_build_cost(g, ctx):
    """C: the _split_verify_decode once-per-step device chain, timed
    standalone (arange/full/cat x2, smeta+bt index_select, div, eq,
    argmax, mul+add) on the same shapes the engine builds."""
    n_blk = ctx // BLOCK
    bt = torch.arange(
        g * n_blk, device=DEV, dtype=torch.int32
    ).reshape(g, n_blk).contiguous()
    # plausible slot_mapping: row r of request r//RTILE sits at some
    # cached position of that request (block from bt, offset in block)
    req = torch.arange(g * RTILE, device=DEV, dtype=torch.long) // RTILE
    pos = (torch.arange(g * RTILE, device=DEV, dtype=torch.int32)
           * 137) % ctx
    blk = bt[req, (pos // BLOCK).to(torch.long)]
    smeta = blk * BLOCK + pos % BLOCK
    stride = BLOCK

    def build():
        idx_parts = []
        req_parts = []
        for i in range(g):
            qs = i * RTILE
            idx_parts.append(torch.arange(qs, qs + RTILE, device=DEV))
            req_parts.append(
                torch.full((RTILE,), i, device=DEV, dtype=torch.long)
            )
        idx = torch.cat(idx_parts)
        req_idx = torch.cat(req_parts)
        sm = smeta.index_select(0, idx)
        bt_rows = bt.index_select(0, req_idx)
        sm_blk = torch.div(sm, stride, rounding_mode="floor")
        sm_pos = sm - sm_blk * stride
        blk_idx = (
            (bt_rows == sm_blk.unsqueeze(1))
            .to(torch.int32)
            .argmax(dim=1)
            .to(sm.dtype)
        )
        sl_rows = blk_idx * stride + sm_pos + 1
        btg = bt.index_select(
            0, torch.arange(g, device=DEV, dtype=torch.long)
        )
        return idx, bt_rows, sl_rows, btg

    return timeit(build, 20), bt.shape[1]


def main():
    print(f"p172_deepprobe — device={DEV}, torch={torch.__version__}")
    print(f"mxkv_src={MXKV_SRC}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    if mx_dequant_cache is None:
        print("WARN: mx_dequant_cache not exported — face E skipped")
    rc = 0
    rows = []
    for fmt in FMTS:
        ship = "v5" if fmt in V5_DEFAULT else "v2batch"
        # ---- A + B: kernel and kernel+aux at the r162 problem cells
        for g_n, ctx in ((1, 1024), (1, 8192), (1, 32768),
                         (8, 8192), (8, 32768)):
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)
            bt_rows = bt.repeat_interleave(RTILE, dim=0).contiguous()
            idx = torch.arange(g_n * RTILE, device=DEV)
            output = torch.empty(g_n * RTILE, HK * HREP, D,
                                 dtype=torch.float16, device=DEV)

            def v2(q=q, c=cache, b=bt_rows, s=lims):
                return mx_decode(q, c, b, s, SCALE)

            def v4(q=q, c=cache, b=bt, s=lims):
                return mx_decode_rows(q, c, b, s, SCALE)

            def v5(q=q, c=cache, b=bt, s=lims):
                return mx_decode_rows_dot(q, c, b, s, SCALE)

            kern = v5 if ship == "v5" else v2
            barg = bt if ship == "v5" else bt_rows
            reps = 5 if ctx >= 32768 else 10
            t2 = timeit(v2, reps)
            t4 = timeit(v4, reps)
            t5 = timeit(v5, reps)
            tk = timeit(kern, reps)

            def full(q=q, c=cache, b=barg, s=lims,
                     o=output, i=idx, k=kern):
                q_rows = q.index_select(0, i)
                out = k(q_rows, c, b, s)
                o.index_copy_(0, i, out.to(o.dtype))

            tf = timeit(full, reps)
            out_now = kern(q, cache, barg, lims)
            fin = bool(torch.isfinite(out_now.float()).all())
            if not fin:
                rc = 1
            rows.append((fmt, g_n, ctx, ship, tk, tf, t2, t4, t5))
            print(f"[{fmt}] G={g_n} ctx={ctx}: ship({ship}) {tk:.3f} "
                  f"+aux {tf:.3f} (aux {tf - tk:.3f}) | v2 {t2:.3f} "
                  f"v4 {t4:.3f} v5 {t5:.3f} ms finite={fin}")
        # ---- D: mx_store per-layer prefill write-quant
        for ctx in (8192, 32768):
            gen = torch.Generator().manual_seed(23)
            n_blocks = ctx // BLOCK + 2
            cache = torch.zeros(
                n_blocks, BLOCK, HK, slot_bytes(D, fmt),
                dtype=torch.uint8, device=DEV,
            )
            tb_full = torch.arange(ctx, device=DEV, dtype=torch.int64) \
                // BLOCK % n_blocks
            tp = torch.arange(ctx, device=DEV, dtype=torch.int64) % BLOCK
            xk = torch.randn(ctx, HK, D, generator=gen).to(DEV)
            xv = torch.randn(ctx, HK, D, generator=gen).to(DEV)

            def st(xk=xk, xv=xv, c=cache, tb=tb_full, tp=tp):
                return mx_store(xk, xv, c, tb, tp)

            ts = timeit(st, 5 if ctx >= 32768 else 10)
            print(f"[{fmt}] store ctx={ctx}: {ts:.3f} ms/layer")
        # ---- E: continuation-prefill prefix dequant per layer
        if mx_dequant_cache is not None:
            for cached in (2048, 8192):
                _, cache, bt, _ = build_multi([cached], fmt, 29)
                alloc = math.ceil(cached / BLOCK) * BLOCK
                k_buf = torch.empty(1, HK, alloc, D,
                                    dtype=torch.float16, device=DEV)
                v_buf = torch.empty_like(k_buf)

                def dq(c=cache, b=bt, k=k_buf, v=v_buf, n=cached):
                    return mx_dequant_cache(c, b, k, v, n)

                td = timeit(dq, 10)
                print(f"[{fmt}] dequant cached={cached}: {td:.3f} ms/layer")
    # ---- C: plan build (fmt-independent shapes)
    for g_n, ctx in ((1, 8192), (8, 8192)):
        tc, n_blk = plan_build_cost(g_n, ctx)
        print(f"[plan] G={g_n} ctx={ctx} bt_cols={n_blk}: {tc:.3f} "
              f"ms/step ({tc / 16:.3f} ms/layer amortized)")
    print(f"A14_PROBE_DONE rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
