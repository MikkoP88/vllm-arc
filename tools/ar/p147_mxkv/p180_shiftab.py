#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p180_shiftab.py — P6.47 final kernel rung: join vs duplicated-load
shift flavor on the K3 e2m1 lane (m4 / mkv4).

r182 closed the walk decomposition: m8 is CONVERGED (66% dot+load
floor; the 1.62 ms f8->f16 cvt has no lawful removal on Xe2; scale and
softmax free). The true-LUT rung is DEAD (m4 byte-map share -0.09 ms =
noise zero). The remaining m4/mkv4 residual (+2.8 ms over m8) is
structurally suspected: the tl.join + reshape cross-lane shuffle in
the nibble extraction. The kernel already ships a semantics-identical
duplicated-load SHIFT flavor (v2 A/B heritage; K3 lane implements both
— module header). The launcher flavor comes from the _V5_TUNE 4th
field; the A/B harness flips the module attr at call time (the
launcher reads it per call).

  F0 correctness, m4 + mkv4: shift vs join bit-identity (the flavor
     claim) + shift vs per-row v2 gate d <= 2e-3, on {549x2, 2051x2,
     mixed} ctx sets.
  F1 perf, m4 + mkv4: G1 x {8192, 32768} + G8 x 32768, legs
     [join, shift, join] — adjacent pairs with a repeat bracket (the
     r180 position-noise law).

K2/K3 stay default-on (cert25zb). SHIFT_AB_PASS = F0 clean.
"""

from __future__ import annotations

import importlib.util
import inspect as _inspect
import math
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
        raise SystemExit("P180_NO_INSTALLED_MXKV (run inside the image)")
    spec = importlib.util.spec_from_file_location(
        "mx_kv_shiftab", src / "__init__.py", submodule_search_locations=[str(src)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mxkv = _load_mxkv()
mx_decode = mxkv.mx_decode
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes
_tmd = getattr(mxkv, "triton_mx_decode", None)
if _tmd is None:
    _tmd = importlib.import_module(".triton_mx_decode", mxkv.__name__)


def set_shift(on):
    _tmd._SHIFT = bool(on)


# cert25zb kernel identity (refuse a stale image tree)
_src_text = _inspect.getsource(_tmd)
for marker in ("_f16_scale", "_e2m1_byte", "F16OUT", "E2M1FAST",
               'ARC_MXKV_K2", "1") != "0"', 'ARC_MXKV_K3", "1") != "0"'):
    assert marker in _src_text, f"P180_STALE_TREE (no {marker})"

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
RTILE = 4
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp4", "mxk8v4")


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """p160-law v5 call args; identical builder to p173..p179."""
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


def main():
    print(f"p180_shiftab — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    summary = []
    for fmt in FMTS:
        # ---- F0: shift bit-identity vs join + gate vs per-row v2
        for name, ctxs in {
            "ctx549": [549] * 2,
            "ctx2051": [2051] * 2,
            "mixed": [549, 1500, 2051, 8192],
        }.items():
            q, cache, bt, lims = build_multi(ctxs, fmt, 31)
            set_shift(False)
            o_join = mxkv.mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            set_shift(True)
            o_shift = mxkv.mx_decode_rows_dot(q, cache, bt, lims, SCALE)
            o2 = torch.empty_like(o_shift)
            for r_ in range(q.shape[0]):
                g_ = r_ // RTILE
                o2[r_: r_ + 1] = mx_decode(
                    q[r_: r_ + 1], cache, bt[g_: g_ + 1], lims[r_: r_ + 1],
                    SCALE,
                )
            torch.xpu.synchronize()
            d_bit = float((o_shift - o_join).abs().amax().item())
            d_gate = float((o_shift - o2).abs().amax().item())
            ok = (d_bit <= 2e-3 and d_gate <= 2e-3
                  and torch.isfinite(o_shift).all())
            print(f"[{fmt}] F0 {name}: shift-vs-join max|d|={d_bit:.3e} | "
                  f"shift-vs-v2 {d_gate:.3e} {'OK' if ok else 'FAIL'}")
            rc |= 0 if ok else 1

        # ---- F1: perf — [join, shift, join] interleaved legs
        for g_n, ctx in ((1, 8192), (1, 32768), (8, 32768)):
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)
            reps = 5 if ctx >= 32768 else 10

            def call(_q=q, _c=cache, _b=bt, _l=lims):
                return mxkv.mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

            ts = {}
            for k in ("join", "shift", "join"):
                set_shift(k == "shift")
                ts.setdefault(k, []).append(wall_ms(call, reps))
            set_shift(False)
            j1, s, j2 = ts["join"][0], ts["shift"][0], ts["join"][1]
            print(f"[{fmt}] F1 G={g_n} ctx={ctx}: join {j1:.3f}/{j2:.3f} | "
                  f"shift {s:.3f} ({j2 / s:.2f}x vs last-join)")
            if g_n == 1:
                summary.append((fmt, ctx, j1, j2, s))
    print("--- shift-flavor summary (G1) ---")
    for fmt, ctx, j1, j2, s in summary:
        print(f"[{fmt}] ctx={ctx}: join {j1:.3f}/{j2:.3f} -> shift {s:.3f} "
              f"({min(j1, j2) / s:.2f}x best-join)")
    print(f"SHIFT_AB_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
