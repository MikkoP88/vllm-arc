#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p179_kdiag.py — P6.47 convergence decider: dequant-free / softmax-free
diagnostic bounds on the packed-lane walk residual (K2/K3 default-on,
cert25zb).

The walk after the K2/K3 bake: m8 4.87 / m4 7.62 / mkv4 7.62 ms at 32K
(r178/r179 adjacent A/B banks). Whether another kernel rung exists
depends on what owns the residual. This probe patches the INSTALLED
kernel source in-process (per-variant tmp copy, anchored replaces
sliced to the split-dot kernel and the dequant helper) and times the
variants back-to-back — the r180 law: adjacent legs are the only
trustworthy structure, and the full leg repeats at the end bracket
clock drift.

Variants (timing-only; patched numerics are intentionally garbage):
  full     — the shipping kernel (K2/K3 defaults on).
  noscale  — e4m3 lane drops the scale chain + packed mul (DCE removes
             the exp2/clamp/broadcast; bounds the scale share).
  rawbits  — bytes -> u16 & 0x3C00 -> f16 1.0 (drops the f8->f16 cvt;
             bounds the whole per-element dequant share).
  nosftexp — the split-dot softmax exps -> arithmetic (bounds the SFU
             share; the per-group scale exp2 stays).
  floor    — rawbits + nosftexp stacked (the removable floor; the
             residual = dot + loads + addressing + reductions).
  nomap    — m4 only: _e2m1_byte -> trivial widen*16 (bounds the byte-
             map op share = the true-LUT rung ceiling).

Faces:
  F1 m8 G1 @8192 and @32768: [full, noscale, rawbits, nosftexp, floor,
  full2]. F2 m8 G8 @32768: [full, floor, full2]. F3 m4 G1 @32768:
  [full, nomap, full2]. F0 sanity: the full variant's output is finite
  and matches the per-row v2 gate (d <= 2e-3); patched variants only
  need to complete without raising.

Print law: one row per leg; share table at the end.
KDIAG_PASS = no crash + F0 clean.
"""

from __future__ import annotations

import importlib.util
import inspect as _inspect
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch

_INSTALLED = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv"
)

# anchored patch substrings (unique within their slice, indent-free)
_F16_RET = "return raw.to(tl.float8e4nv, bitcast=True).to(tl.float16) * sc16"
_RAWBITS_RET = (
    "return (raw.to(tl.uint16) & 0x3C00).to(tl.float16, bitcast=True)"
)
_P_EXP = "p = tl.exp(scores - m_safe[:, None])"
_P_ARI = "p = (scores - m_safe[:, None]) * 0.5"
_A_EXP = "alpha = tl.exp(m - m_safe)"
_A_ARI = "alpha = (m - m_safe) * 0.5 + 1.0"
_MAP_JOIN_LO = "lo = _e2m1_byte(raw & 0xF)"
_MAP_JOIN_HI = "hi = _e2m1_byte(raw >> 4)"
_MAP_SHIFT = "b = _e2m1_byte(nib)"
_NOMAP_JOIN_LO = "lo = (raw & 0xF) * 16"
_NOMAP_JOIN_HI = "hi = (raw >> 4) * 16"
_NOMAP_SHIFT = "b = nib.to(tl.uint8) * 16"


def _patch(src, kind):
    """Slice-scoped anchored replaces. The helper slice carries the
    dequant returns; the split-dot slice carries the softmax exps (the
    per-row kernels share the softmax text — slice or corrupt them)."""
    ha = src.index("def _mx_dequant_block")
    hb = src.index("def _mx_decode(")
    sa = src.index("def _mx_decode_split_dot")
    sb = src.index("def _mx_decode_combine")

    helper, split = src[ha:hb], src[sa:sb]
    if kind in ("noscale",):
        assert helper.count(_F16_RET) == 1, "F16 anchor"
        helper = helper.replace(
            _F16_RET, "return raw.to(tl.float8e4nv, bitcast=True)"
            ".to(tl.float16)"
        )
    elif kind in ("rawbits", "floor"):
        assert helper.count(_F16_RET) == 1, "F16 anchor"
        helper = helper.replace(_F16_RET, _RAWBITS_RET)
    if kind == "nomap":
        for old, new in ((_MAP_JOIN_LO, _NOMAP_JOIN_LO),
                         (_MAP_JOIN_HI, _NOMAP_JOIN_HI),
                         (_MAP_SHIFT, _NOMAP_SHIFT)):
            assert helper.count(old) == 1, f"map anchor {old}"
            helper = helper.replace(old, new)
    if kind in ("nosftexp", "floor"):
        assert split.count(_P_EXP) == 1 and split.count(_A_EXP) == 1, "exp anchor"
        split = split.replace(_P_EXP, _P_ARI).replace(_A_EXP, _A_ARI)
    out = src[:ha] + helper + src[hb:sa] + split + src[sb:]
    assert out != src or kind == "full", "no-op patch"
    return out


def load_variant(kind):
    d = tempfile.mkdtemp(prefix=f"mxkv_diag_{kind}_")
    for fn in os.listdir(_INSTALLED):
        if fn.endswith(".py"):
            shutil.copy(_INSTALLED / fn, d)
    path = Path(d) / "triton_mx_decode.py"
    path.write_text(_patch(path.read_text(), kind))
    spec = importlib.util.spec_from_file_location(
        f"mxkv_diag_{kind}", Path(d) / "__init__.py",
        submodule_search_locations=[d],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mxkv = load_variant("full")
mx_decode = mxkv.mx_decode
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes

# cert25zb kernel identity (refuse a stale image tree)
_src_text = _inspect.getsource(mxkv.triton_mx_decode)
for marker in ("_f16_scale", "_e2m1_byte", "F16OUT", "E2M1FAST",
               'ARC_MXKV_K2", "1") != "0"', 'ARC_MXKV_K3", "1") != "0"'):
    assert marker in _src_text, f"P179_STALE_TREE (no {marker})"

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
RTILE = 4
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """p160-law v5 call args; identical builder to p173..p178."""
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


def leg(mod, q, cache, bt, lims, reps):
    def fn():
        return mod.mx_decode_rows_dot(q, cache, bt, lims, SCALE)

    return wall_ms(fn, reps)


def main():
    print(f"p179_kdiag — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    mods = {k: (mxkv if k == "full" else load_variant(k))
            for k in ("full", "noscale", "rawbits", "nosftexp",
                      "floor", "nomap")}

    # ---- F0: shipping lane sanity (full vs per-row v2 gate)
    for fmt in ("mxfp8", "mxfp4"):
        q, cache, bt, lims = build_multi([2051] * 2, fmt, 31)
        o = mxkv.mx_decode_rows_dot(q, cache, bt, lims, SCALE)
        o2 = torch.empty_like(o)
        for r_ in range(q.shape[0]):
            g_ = r_ // RTILE
            o2[r_: r_ + 1] = mx_decode(
                q[r_: r_ + 1], cache, bt[g_: g_ + 1], lims[r_: r_ + 1],
                SCALE,
            )
        torch.xpu.synchronize()
        d = float((o - o2).abs().amax().item())
        ok = d <= 2e-3 and torch.isfinite(o).all()
        print(f"[{fmt}] F0 full vs per-row v2 ctx2051x2: max|d|={d:.3e} "
              f"{'OK' if ok else 'FAIL'}")
        rc |= 0 if ok else 1

    # ---- F1: m8 walk decomposition
    for ctx, reps in ((8192, 10), (32768, 5)):
        q, cache, bt, lims = build_multi([ctx], "mxfp8", 19)
        order = ("full", "noscale", "rawbits", "nosftexp", "floor", "full")
        ts = {}
        for k in order:
            ts[k] = leg(mods[k], q, cache, bt, lims, reps)
            print(f"[mxfp8] F1 G=1 ctx={ctx} {k}: {ts[k]:.3f} ms")
        if ctx == 32768:
            f, ns, rb, se, fl = (ts["full"], ts["noscale"], ts["rawbits"],
                                 ts["nosftexp"], ts["floor"])
            print(f"[mxfp8] F1 shares @32K: scale {f - ns:.3f} | cvt+mul "
                  f"{ns - rb:.3f} | softmax-exp {f - se:.3f} | removable "
                  f"{f - fl:.3f} | floor(dot+load+addr+red) {fl:.3f} ms")

    # ---- F2: m8 conc face bound
    q, cache, bt, lims = build_multi([32768] * 8, "mxfp8", 19)
    for k in ("full", "floor", "full"):
        t = leg(mods[k], q, cache, bt, lims, 4)
        print(f"[mxfp8] F2 G=8 ctx=32768 {k}: {t:.3f} ms")

    # ---- F3: m4 byte-map share (the true-LUT rung ceiling)
    q, cache, bt, lims = build_multi([32768], "mxfp4", 19)
    ts = {}
    for k in ("full", "nomap", "full"):
        ts[k] = leg(mods[k], q, cache, bt, lims, 5)
        print(f"[mxfp4] F3 G=1 ctx=32768 {k}: {ts[k]:.3f} ms")
    print(f"[mxfp4] F3 byte-map share: {ts['full'] - ts['nomap']:.3f} ms "
          f"(LUT ceiling)")

    print(f"KDIAG_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
