#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p178_k2tune.py — P6.47 K2 cycle 3: launch-config re-sweep under the
packed lanes.

The A14 sweep's refutations (BLOCK_N=64 re-climbing the register
spill slope, warps law 4/8/8) were measured on the PRE-K2 kernel —
f32 dequant temporaries and their conversions set that pressure. The
K2/K3 packed lanes removed the f32 chain (r178/r179 banks: m8 4.885 /
m4 7.618 / mkv4 7.623 ms @32K G1), so the (BN, warps, splits) space
is re-measured here on the bind-mounted shipping tree, per format at
its packed lane (m8: K2; m4/mkv4: K2+K3):

  F0 correctness once per format (defaults; d <= 2e-3 vs per-row v2).
  F1 config sweep G1 @32768: BN {32, 64} x warps {4, 8} at
     splits=256, then splits {64, 128, 256} at BN=32/warps=4.
  F2 the per-format winner at G8 @32768 (conc face).

All knobs are call-time env reads in the launcher (capture law) — no
code changes between legs. Print law: one row per cell; winner table
at the end.
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
        raise SystemExit("P178_NO_INSTALLED_MXKV (run inside the image)")
    spec = importlib.util.spec_from_file_location(
        "mx_kv_k2tune", src / "__init__.py", submodule_search_locations=[str(src)]
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

# patch identity (refuse a stale image tree)
_src_text = _inspect.getsource(mxkv.triton_mx_decode)
for marker in ("_f16_scale", "_e2m1_byte", "ARC_MXKV_K3"):
    assert marker in _src_text, f"P178_STALE_TREE (no {marker})"

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
RTILE = 4
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
LANES = {
    "mxfp8": (True, False),  # (K2, K3)
    "mxfp4": (True, True),
    "mxk8v4": (True, True),
}


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """p160-law v5 call args; identical builder to p173/p175/p176/p177."""
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


def envs(k2, k3, bn=None, nw=None, sp=None):
    # "0" (not pop) for the lane knobs — default-ON since cert25zb;
    # popping would silently re-enable them against the flipped kernel.
    for k, on in (("ARC_MXKV_K2", k2), ("ARC_MXKV_K3", k3)):
        os.environ[k] = "1" if on else "0"
    for k, v in (("ARC_MXKV_V5_BN", bn),
                 ("ARC_MXKV_V5_WARPS", nw),
                 ("ARC_MXKV_V5_SPLITS", sp)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)


def main():
    print(f"p178_k2tune — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    winners = {}
    for fmt, (k2, k3) in LANES.items():
        # ---- F0: correctness at the packed defaults
        envs(k2, k3)
        q, cache, bt, lims = build_multi([2051] * 2, fmt, 31)
        o = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
        o2 = torch.empty_like(o)
        for r_ in range(q.shape[0]):
            g_ = r_ // RTILE
            o2[r_: r_ + 1] = mx_decode(
                q[r_: r_ + 1], cache, bt[g_: g_ + 1], lims[r_: r_ + 1], SCALE,
            )
        torch.xpu.synchronize()
        d = float((o - o2).abs().amax().item())
        ok = d <= 2e-3 and torch.isfinite(o).all()
        print(f"[{fmt}] F0 packed-default vs per-row v2 ctx2051x2: "
              f"max|d|={d:.3e} {'OK' if ok else 'FAIL'}")
        rc |= 0 if ok else 1

        # ---- F1: config sweep G1 @32K
        q, cache, bt, lims = build_multi([32768], fmt, 19)

        def call(_q=q, _c=cache, _b=bt, _l=lims):
            return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

        results = []
        for bn in (32, 64):
            for nw in (4, 8):
                envs(k2, k3, bn=bn, nw=nw, sp=256)
                t = wall_ms(call, 5)
                results.append(((bn, nw, 256), t))
        for sp in (64, 128):
            envs(k2, k3, bn=32, nw=4, sp=sp)
            results.append(((32, 4, sp), wall_ms(call, 5)))
        envs(k2, k3)  # default (table tune) leg
        t_def = wall_ms(call, 5)
        for (bn, nw, sp), t in results:
            print(f"[{fmt}] F1 G=1 ctx=32768 bn={bn} w={nw} sp={sp}: "
                  f"{t:.3f} ms")
        print(f"[{fmt}] F1 G=1 ctx=32768 tune-default: {t_def:.3f} ms")
        cfg_b, t_b = min(results, key=lambda x: x[1])
        if t_def < t_b:
            cfg_b, t_b = "default", t_def
        winners[fmt] = (cfg_b, t_b, t_def)

        # ---- F2: winner at G8 @32K (conc face)
        q8, cache8, bt8, lims8 = build_multi([32768] * 8, fmt, 19)

        def call8(_q=q8, _c=cache8, _b=bt8, _l=lims8):
            return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

        envs(k2, k3)
        t8_def = wall_ms(call8, 4)
        if cfg_b == "default":
            t8_b = t8_def
        else:
            bn, nw, sp = cfg_b
            envs(k2, k3, bn=bn, nw=nw, sp=sp)
            t8_b = wall_ms(call8, 4)
        print(f"[{fmt}] F2 G=8 ctx=32768: default {t8_def:.3f} | winner "
              f"{cfg_b} {t8_b:.3f} ms ({t8_def / t8_b:.2f}x)")
    envs(False, False)
    print("--- K2 cycle-3 tune re-sweep winners (G1 @32K) ---")
    for fmt, (cfg_b, t_b, t_def) in winners.items():
        print(f"[{fmt}] winner {cfg_b}: {t_b:.3f} ms "
              f"(default {t_def:.3f}, {t_def / t_b:.2f}x)")
    print(f"K2_TUNE_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
