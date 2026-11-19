"""p184_tfloor.py -- P6.47 Phase 7 attempt 2: deep-split re-sweep of the
v5 split-dot on cert25zb + the TILE_FLOOR-bearing triton_mx_decode.py
(in-container swap, r193 -- image untouched).

WHY: the v6 >=4-serial-tile floor (the A14 rule) was tuned pre-K2/K3,
when per-tile scalar-ALU dequant dominated the walk; the cert25zb
packed lanes cut that cost, and the r192 toolchain verdict closed the
in-language prefetch lane (num_stages no-op AND manual double
buffering counterproductive on Triton/XPU). The remaining serial
bt->slot->gather chain tiles can instead be REMOVED: TILE_FLOOR 1-2
raises s_eff so each CTA walks fewer (ideally one) tiles and the chain
latency rides concurrency, traded against the combine reduce over more
splits. This probe sweeps the U-curve.

CONFIGS (TILE_FLOOR, SPLITS cap): (4,256) = certified baseline;
(2,256), (2,512), (1,256), (1,512), (1,1024).
CELLS: G1/G8 x 8K/32K x 3 formats (the T1 money cells).
GATE: correctness d <= 2e-3 vs the baseline config at G1 8K + G8 32K
per format (split regrouping reassociation is the A11 S-sweep class,
2-3e-8); perf table banks for the winner call.

Exit 0 = TFL_PASS. python -u (r191 buffering law).
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

CONFIGS = ((4, 256), (2, 256), (2, 512), (1, 256), (1, 512), (1, 1024))
PERF_CELLS = ((1, 8192), (1, 32768), (8, 8192), (8, 32768))
COR_CELLS = ((1, 8192), (8, 32768))


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


def run_cfg(tfl, splits, q, c, b, s):
    os.environ["ARC_MXKV_V7"] = "0"
    os.environ["ARC_MXKV_V5_TFLOOR"] = str(tfl)
    os.environ["ARC_MXKV_V5_SPLITS"] = str(splits)
    return mx_decode_rows_dot(q, c, b, s, SCALE)


def main():
    print(f"p184_tfloor — device={DEV}, torch={torch.__version__}")
    print(f"mxkv_src={MXKV_SRC}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    # ---- correctness face: every non-baseline config vs (4,256)
    print("== correctness vs certified (4,256), d = max|out - base| ==")
    worst = 0.0
    for fmt in FMTS:
        for g_n, ctx in COR_CELLS:
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)
            base = run_cfg(4, 256, q, cache, bt, lims)
            for tfl, splits in CONFIGS[1:]:
                o = run_cfg(tfl, splits, q, cache, bt, lims)
                torch.xpu.synchronize()
                d = (o.float() - base.float()).abs().max().item()
                fin = bool(torch.isfinite(o.float()).all())
                ok = fin and d <= 2e-3
                worst = max(worst, d)
                if not ok:
                    rc = 1
                print(f"[{fmt}] G={g_n} ctx={ctx} tf={tfl},sp={splits}: "
                      f"d={d:.3e} finite={fin} "
                      f"{'OK' if ok else 'FAIL'}")
    print(f"CORRECTNESS_WORST_d={worst:.3e}")
    if rc:
        print("TFL_FAIL")
        return rc
    # ---- perf face: the T1 money cells, all configs
    print("== perf: (TILE_FLOOR, SPLITS) sweep, ms/call ==")
    for fmt in FMTS:
        for g_n, ctx in PERF_CELLS:
            q, cache, bt, lims = build_multi([ctx] * g_n, fmt, 19)
            reps = 5 if ctx >= 32768 else 10
            line = []
            for tfl, splits in CONFIGS:
                def fn(q=q, c=cache, b=bt, s=lims, t=tfl, p=splits):
                    return run_cfg(t, p, q, c, b, s)

                t_ms = timeit(fn, reps)
                line.append(f"tf{tfl}/sp{splits} {t_ms:.3f}")
            print(f"[{fmt}] G={g_n} ctx={ctx}: " + " | ".join(line))
    os.environ["ARC_MXKV_V5_TFLOOR"] = "4"
    os.environ["ARC_MXKV_V5_SPLITS"] = "256"
    print("TFL_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
