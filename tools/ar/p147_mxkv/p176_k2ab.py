#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p176_k2ab.py — P6.47 K2 cycle 1: packed-fp16 dequant A/B (m8 lane).

r177 stage-split evidence: the v6 walk at 32K is 98-99% stage 1,
FLAT across splits 32..256 (shared-pipe wall), num_stages a no-op —
the cost is per-element scalar ALU in the f32 dequant chain. The K2
patch moves e4m3-payload dequant into packed fp16 (e4m3->fp16 exact,
pow2 scale exact in fp16 — single rounding point unchanged) behind
ARC_MXKV_K2=1. This probe validates it on the PATCHED installed tree
(the wrapper bind-mounts the shipped triton_mx_decode.py over the
image's copy — the exact shipping path):

  F0 correctness, per format: (a) K2 on vs off — the bit-identical
     design claim (report the actual max|d|), (b) K2 on vs per-row v2
     — the standing gate d <= 2e-3 (fp16-operand class).
  F1 perf, G1 x {8192, 32768} + G8 x 32768 per format: wall ms with
     K2 off vs on, speedup, and the r177-banked installed-v6 full-call
     baselines for cross-check (m8 1.787 / 5.909; m4 2.409 / 8.633;
     mkv4 2.415 / 8.559 ms at 8K / 32K G1).

Print law: one row per face cell; K2 winner table at the end.
"""

from __future__ import annotations

import inspect as _inspect
import math
import os
import sys
import time
from pathlib import Path

import torch

_INSTALLED = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv"
)


def _load_mxkv():
    import importlib.util

    src = _INSTALLED if (_INSTALLED / "__init__.py").exists() else None
    if src is None:
        raise SystemExit("P176_NO_INSTALLED_MXKV (run inside the image)")
    spec = importlib.util.spec_from_file_location(
        "mx_kv_k2ab", src / "__init__.py", submodule_search_locations=[str(src)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mxkv = _load_mxkv()
mx_decode = mxkv.mx_decode
mx_decode_rows_dot = mxkv.mx_decode_rows_dot
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes

# K2 patch identity (refuse a stale image tree)
_src_text = _inspect.getsource(mxkv.triton_mx_decode)
for marker in ("_f16_scale", "F16OUT", "ARC_MXKV_K2"):
    assert marker in _src_text, f"P176_STALE_TREE (no {marker})"

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
RTILE = 4
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
# r177 k2stageA16 bank (installed v6, full-call wall, G1) — reference
_R177_BANK = {
    ("mxfp8", 8192): 1.787,
    ("mxfp8", 32768): 5.909,
    ("mxfp4", 8192): 2.409,
    ("mxfp4", 32768): 8.633,
    ("mxk8v4", 8192): 2.415,
    ("mxk8v4", 32768): 8.559,
}


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """p160-law v5 call args; identical builder to p173/p175 (seed law)."""
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


def wall_ms(fn, reps, warm=2):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def v2_reference(q, cache, bt, lims):
    o2 = torch.empty_like(q)
    for r_ in range(q.shape[0]):
        g_ = r_ // RTILE
        o2[r_: r_ + 1] = mx_decode(
            q[r_: r_ + 1], cache, bt[g_: g_ + 1], lims[r_: r_ + 1], SCALE,
        )
    torch.xpu.synchronize()
    return o2


def k2on():
    os.environ["ARC_MXKV_K2"] = "1"


def k2off():
    # "0" (not pop) — the lane is default-ON since cert25zb; popping
    # would silently re-enable it against the flipped kernel.
    os.environ["ARC_MXKV_K2"] = "0"


def main():
    print(f"p176_k2ab — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    summary = []
    for fmt in FMTS:
        # ---- F0: correctness — K2 on vs off (bit-identity face), then
        # K2 on vs per-row v2 (the standing d <= 2e-3 gate)
        for name, ctxs in {
            "ctx549": [549] * 2,
            "ctx2051": [2051] * 2,
            "mixed": [549, 1500, 2051, 8192],
        }.items():
            q, cache, bt, lims = build_multi(ctxs, fmt, 31)
            k2off()
            o_off = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            k2on()
            o_on = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            o2 = v2_reference(q, cache, bt, lims)
            d_bit = float((o_on - o_off).abs().amax().item())
            d_gate = float((o_on - o2).abs().amax().item())
            ok = d_gate <= 2e-3 and torch.isfinite(o_on).all()
            print(f"[{fmt}] F0 {name}: K2-vs-off max|d|={d_bit:.3e} | "
                  f"K2-vs-v2 max|d|={d_gate:.3e} {'OK' if ok else 'FAIL'}")
            rc |= 0 if ok else 1

        # ---- F1: perf — K2 off vs on at the K2 problem cells
        for g_n, ctx in ((1, 8192), (1, 32768), (8, 32768)):
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)
            reps = 5 if ctx >= 32768 else 10

            def call(_q=q, _c=cache, _b=bt, _l=lims):
                return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

            k2off()
            t_off = wall_ms(call, reps)
            k2on()
            t_on = wall_ms(call, reps)
            bank = _R177_BANK.get((fmt, ctx)) if g_n == 1 else None
            ref = f" | r177 bank {bank:.3f}" if bank else ""
            print(f"[{fmt}] F1 G={g_n} ctx={ctx}: off {t_off:.3f} -> "
                  f"on {t_on:.3f} ms ({t_off / t_on:.2f}x){ref}")
            if g_n == 1:
                summary.append((fmt, ctx, t_off, t_on))
    k2off()
    print("--- K2 cycle-1 winner table (G1, wall ms) ---")
    for fmt, ctx, t_off, t_on in summary:
        bank = _R177_BANK[(fmt, ctx)]
        print(f"[{fmt}] ctx={ctx}: off {t_off:.3f} -> on {t_on:.3f} "
              f"({t_off / t_on:.2f}x; r177 bank {bank:.3f} -> "
              f"{bank / t_on:.2f}x)")
    print(f"K2_AB_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
