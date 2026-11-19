"""p183_v7ab.py -- P6.47 Phase 7 attempt 1: the v7 FA-shaped walk
(_mx_decode_split_dot2, raw-tile double buffering + dequant-at-consume)
correctness + perf A/B against the certified v5, on cert25zb with the
v7-bearing triton_mx_decode.py swapped over the installed one by the
r192 wrapper (ephemeral --rm container; the image is untouched).

FACES
  correctness (the gate): v7 vs v5 max-abs on the f16 outputs across
    all three formats (K4/V4 payload combinations: mxfp8 0/0, mxfp4
    1/1, mxk8v4 0/1) x edge cells -- G1 sub-tile ctx 200 (s_eff=1,
    masked tail), G1 1K, G1 8K, G2 MIXED ctx [4096, 960] (uneven
    walk_end per group), G8 32K. Env-flipped in-process
    (ARC_MXKV_V7=1/0 -- call-time read in mx_decode_rows_dot);
    d <= 2e-3 + isfinite required at every cell (the v5-class gate).
  perf: timeit v5 vs v7 at the T1 attribution cells (G1 x
    1K/8K/32K, G8 x 8K/32K per format) -- directly comparable to the
    banked r191 kbench table. Threshold (evaluated in the analysis,
    not here): <= 1.0 ms at G1@32K.

Exit 0 = V7AB_PASS (correctness clean); perf prints regardless.
python -u (the r191 buffering law).
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch

_INSTALLED = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv"
)
_here = Path(__file__).resolve()
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
mx_decode_rows_dot = mxkv.mx_decode_rows_dot
mx_store = mxkv.mx_store
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

# correctness edge cells: (G, [ctx per request])
CORS_CELLS = (
    (1, [200]),
    (1, [1024]),
    (1, [8192]),
    (2, [4096, 960]),
    (8, [32768] * 8),
)
# perf cells: the r191/T1 grid
PERF_CELLS = (
    (1, 1024), (1, 8192), (1, 32768), (8, 8192), (8, 32768),
)


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


def run_v7(q, c, b, s):
    os.environ["ARC_MXKV_V7"] = "1"
    return mx_decode_rows_dot(q, c, b, s, SCALE)


def run_v5(q, c, b, s):
    os.environ["ARC_MXKV_V7"] = "0"
    return mx_decode_rows_dot(q, c, b, s, SCALE)


def main():
    print(f"p183_v7ab — device={DEV}, torch={torch.__version__}")
    print(f"mxkv_src={MXKV_SRC}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    # ---- correctness face (the gate)
    print("== correctness: v7 vs v5, d = max|out7 - out5| ==")
    worst = 0.0
    for fmt in FMTS:
        for g_n, ctxs in CORS_CELLS:
            q, cache, bt, lims = build_multi(ctxs, fmt, 19)
            o5 = run_v5(q, cache, bt, lims)
            o7 = run_v7(q, cache, bt, lims)
            torch.xpu.synchronize()
            d = (o7.float() - o5.float()).abs().max().item()
            fin = bool(torch.isfinite(o7.float()).all())
            ok = fin and d <= 2e-3
            worst = max(worst, d)
            if not ok:
                rc = 1
            tag = ",".join(str(c) for c in ctxs)
            print(f"[{fmt}] G={g_n} ctx=[{tag}]: d={d:.3e} "
                  f"finite={fin} {'OK' if ok else 'FAIL'}")
    print(f"CORRECTNESS_WORST_d={worst:.3e}")
    if rc:
        print("V7AB_FAIL")
        return rc
    # ---- perf face: the T1 cells, v5 vs v7
    print("== perf: v5 vs v7 at the T1 cells (ms/call) ==")
    for fmt in FMTS:
        for g_n, ctx in PERF_CELLS:
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)

            def v5(q=q, c=cache, b=bt, s=lims):
                return run_v5(q, c, b, s)

            def v7(q=q, c=cache, b=bt, s=lims):
                return run_v7(q, c, b, s)

            reps = 5 if ctx >= 32768 else 10
            t5 = timeit(v5, reps)
            t7 = timeit(v7, reps)
            print(f"[{fmt}] G={g_n} ctx={ctx}: v5 {t5:.3f} v7 {t7:.3f} "
                  f"ms  x{t5 / t7:.2f}")
    os.environ["ARC_MXKV_V7"] = "0"
    print("V7AB_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
