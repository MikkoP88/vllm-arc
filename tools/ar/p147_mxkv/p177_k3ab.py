#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p177_k3ab.py — P6.47 K2 cycle 2: e2m1 fast-unpack A/B (m4/mkv4).

After K2 (r178 bank: m8 4.871 ms @32K, 1.22x) the e2m1 lanes still
pay the f32 decode chain — m4 8.679 / mkv4 8.281 vs m8's 4.871 ms at
32K. The K3 patch maps each e2m1 nibble to its e4m3 byte with ~10
integer ops (exact map — 16/16 host-verified) and rides the K2
hardware-cvt -> packed-fp16 multiply, replacing the ~18-op
where/exp2 chain. Behind ARC_MXKV_K3=1 (independent of ARC_MXKV_K2
so the A/B attributes). This probe validates on the bind-mounted
shipping tree:

  F0 correctness, per format: K3 on vs off (bit-identity face — the
     map and the pow2 scale are exact, single rounding point
     unchanged), K3 on vs per-row v2 (standing gate d <= 2e-3), and
     the shipping combo K2+K3 vs v2.
  F1 perf, per format: legs {off, K3, K2+K3} at G1 x {8192, 32768}
     + G8 x 32768. r178 K2-on banks for reference: m8 1.527/4.871;
     m4 2.411/8.679; mkv4 2.354/8.281 (8K/32K, G1).

Print law: one row per face cell; K3 winner table at the end.
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
        raise SystemExit("P177_NO_INSTALLED_MXKV (run inside the image)")
    spec = importlib.util.spec_from_file_location(
        "mx_kv_k3ab", src / "__init__.py", submodule_search_locations=[str(src)]
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

# K2+K3 patch identity (refuse a stale image tree)
_src_text = _inspect.getsource(mxkv.triton_mx_decode)
for marker in ("_f16_scale", "_e2m1_byte", "F16OUT", "E2M1FAST",
               "ARC_MXKV_K2", "ARC_MXKV_K3"):
    assert marker in _src_text, f"P177_STALE_TREE (no {marker})"

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
RTILE = 4
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
# r178 k2cycleA17 bank (K2 on, full-call wall, G1) — reference
_R178_BANK = {
    ("mxfp8", 8192): 1.527,
    ("mxfp8", 32768): 4.871,
    ("mxfp4", 8192): 2.411,
    ("mxfp4", 32768): 8.679,
    ("mxk8v4", 8192): 2.354,
    ("mxk8v4", 32768): 8.281,
}


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """p160-law v5 call args; identical builder to p173/p175/p176."""
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


def envs(k2, k3):
    # "0" (not pop) — the lanes are default-ON since cert25zb; popping
    # would silently re-enable them against the flipped kernel.
    for k, on in (("ARC_MXKV_K2", k2), ("ARC_MXKV_K3", k3)):
        os.environ[k] = "1" if on else "0"


def main():
    print(f"p177_k3ab — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    summary = []
    for fmt in FMTS:
        # ---- F0: correctness — K3 bit-identity + K3 and K2+K3 vs v2
        for name, ctxs in {
            "ctx549": [549] * 2,
            "ctx2051": [2051] * 2,
            "mixed": [549, 1500, 2051, 8192],
        }.items():
            q, cache, bt, lims = build_multi(ctxs, fmt, 31)
            envs(False, False)
            o_off = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            envs(False, True)
            o_k3 = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            envs(True, True)
            o_k23 = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            o2 = v2_reference(q, cache, bt, lims)
            d_bit = float((o_k3 - o_off).abs().amax().item())
            d_gate = float((o_k3 - o2).abs().amax().item())
            d_gate23 = float((o_k23 - o2).abs().amax().item())
            ok = (d_gate <= 2e-3 and d_gate23 <= 2e-3
                  and torch.isfinite(o_k3).all() and torch.isfinite(o_k23).all())
            print(f"[{fmt}] F0 {name}: K3-vs-off max|d|={d_bit:.3e} | "
                  f"K3-vs-v2 {d_gate:.3e} | K2K3-vs-v2 {d_gate23:.3e} "
                  f"{'OK' if ok else 'FAIL'}")
            rc |= 0 if ok else 1

        # ---- F1: perf — legs {off, K3, K2+K3}
        for g_n, ctx in ((1, 8192), (1, 32768), (8, 32768)):
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)
            reps = 5 if ctx >= 32768 else 10

            def call(_q=q, _c=cache, _b=bt, _l=lims):
                return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

            legs = {}
            for leg, k2, k3 in (("off", False, False),
                                ("k3", False, True),
                                ("k2k3", True, True)):
                envs(k2, k3)
                legs[leg] = wall_ms(call, reps)
            envs(False, False)
            bank = _R178_BANK.get((fmt, ctx)) if g_n == 1 else None
            ref = f" | r178 k2-bank {bank:.3f}" if bank else ""
            print(f"[{fmt}] F1 G={g_n} ctx={ctx}: off {legs['off']:.3f} | "
                  f"k3 {legs['k3']:.3f} ({legs['off'] / legs['k3']:.2f}x) | "
                  f"k2k3 {legs['k2k3']:.3f} ({legs['off'] / legs['k2k3']:.2f}x)"
                  f"{ref}")
            if g_n == 1:
                summary.append((fmt, ctx, legs))
    envs(False, False)
    print("--- K3 cycle-2 winner table (G1, wall ms) ---")
    for fmt, ctx, legs in summary:
        bank = _R178_BANK[(fmt, ctx)]
        print(f"[{fmt}] ctx={ctx}: off {legs['off']:.3f} -> k3 {legs['k3']:.3f}"
              f" -> k2k3 {legs['k2k3']:.3f} (r178 k2-bank {bank:.3f} -> "
              f"{bank / legs['k2k3']:.2f}x)")
    print(f"K3_AB_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
