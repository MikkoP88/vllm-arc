#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p175_k2stage.py — P6.47 K2: stage-1/stage-2 split of the v6 decode
call + splits/num_stages scaling (K2 lever decision probe, 2026-11-13).

The r176/p174 devcost faces closed 32K as 100% device-bound with the
profiler table empty (XPU kineto yields no kernel rows), so WHERE the
6.05 ms lives inside the call is still unmeasured — and it decides
K2's shape. The combine reads the whole mid buffer ([r, hq, splits,
D+1] f32 = 25.2 MB at 32K/splits=256) on a (r, hq) = 96-CTA grid; if
it owns a large share, combine parallelism (a math-neutral grid
change) is the first lever, not walk-body pipelining.

Faces (installed v6 kernels, no patch — measurement only):
  F0 correctness: v6 default vs per-row v2 at the LIMIT-LAW boundary
     shape (ctx2051 x2) per format; gate d <= 2e-3 + finite.
  F1 stage split: event-timed stage-1-only and stage-2-only launches
     (the wrapper's exact launch args, replicated) + full-call wall,
     per format x ctx {8192, 32768} at the v6 default splits=256.
     Cross-check: s1 + s2 ~= full (device-bound at 32K).
  F2 splits scaling: splits {32, 64, 128, 256} at 32K — t_s1 flat vs
     splits = throughput/latency wall (pipelining attack); t_s1 ~ 1/s
     = still occupancy-starved (cap/rule rebalance).
  F3 num_stages: ns {2, 3, 4} on the dot walk at 32K/splits=256.
     A6's num_stages no-op verdict was measured on the scalar v2 walk;
     the v5 dot lane has tl.dot in the loop, where the Triton pipeliner
     can prefetch loads feeding the MMA — untested face.

Loads mx_kv from MXKV_DIR (mount-aware; falls back to the repo tree).
Print law: one row per (fmt, face cell); K2 decision table at the end.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import torch
import triton

_INSTALLED = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv"
)
_here = Path(__file__).resolve()
# repo-tree fallback only when 4-deep in the tree AND the installed
# lane is absent (bare-container /tmp copies resolve the fallback to
# the probe file itself — the p174 _INSTALLED-first law)
_TREE = (
    _here.parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
    if len(_here.parents) > 3
    else _here
)


def _mxkv_src():
    if os.environ.get("MXKV_DIR"):
        return Path(os.environ["MXKV_DIR"])
    return _INSTALLED if (_INSTALLED / "__init__.py").exists() else _TREE


_MXKV_DIR = _mxkv_src()


def _load_mxkv():
    spec = importlib.util.spec_from_file_location(
        "mx_kv_k2stage",
        _MXKV_DIR / "__init__.py",
        submodule_search_locations=[str(_MXKV_DIR)],
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
# private kernels live on the submodule (the package __init__ re-exports
# only the public API; from-import binds the submodule attribute)
_tmd = getattr(mxkv, "triton_mx_decode", None)
if _tmd is None:
    import importlib as _il

    _tmd = _il.import_module(".triton_mx_decode", mxkv.__name__)
_split_dot = _tmd._mx_decode_split_dot  # noqa: SLJ001 — probe face
_combine = _tmd._mx_decode_combine  # noqa: SLJ001 — probe face
_V5_TUNE = _tmd._V5_TUNE

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
RTILE = 4
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp8", "mxfp4", "mxk8v4")


def build_multi(ctxs, fmt, seed, rtile=RTILE):
    """p160-law v5 call args; identical builder to p173/p174 (seed law)."""
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


class StagedCall:
    """Replicates mx_decode_rows_dot's launch exactly (tune table, fmt
    detection, strides, M packing) but exposes stage-1-only and
    stage-2-only entry points, each with its own preallocated mid —
    nothing allocates inside a timed region."""

    def __init__(self, q, cache, block_tables, limits, sm_scale, splits):
        self.q, self.cache = q, cache
        self.bt, self.lims = block_tables, limits
        self.scale = sm_scale
        r, hq, d = q.shape
        g = r // RTILE
        hk = cache.shape[2]
        n_rep = hq // hk
        sb = cache.shape[-1]
        want = {f: slot_bytes(d, f) for f in FMTS}
        self.fmt = next(f for f, w in want.items() if w == sb)
        self.bn, self.nw, self.ns, self.sh = _V5_TUNE[self.fmt]
        self.splits = splits
        self.grid1 = (g, hk, splits)
        self.grid2 = (r, hq)
        self.mid = torch.empty(
            r, hq, splits, d + 1, dtype=torch.float32, device=q.device
        )
        self.out = torch.empty_like(q)
        self.m_lanes = RTILE * n_rep
        self.consts = dict(
            D=d,
            N_GROUPS=d // 32,
            K4=self.fmt == "mxfp4",
            V4=self.fmt in ("mxfp4", "mxk8v4"),
            R_TILE=RTILE,
            N_REP=n_rep,
            M_LANES=self.m_lanes,
            M_PAD=triton.next_power_of_2(self.m_lanes),
            BLOCK_SIZE=BLOCK,
            BLOCK_N=self.bn,
            NUM_SPLITS=splits,
            SHIFT=self.sh,
        )

    def stage1(self, num_stages=None):
        _split_dot[self.grid1](
            self.q, self.cache, self.bt, self.lims, self.mid, self.scale,
            self.q.stride(0), self.q.stride(1),
            self.mid.stride(0), self.mid.stride(1), self.mid.stride(2),
            self.bt.stride(0),
            self.cache.stride(0), self.cache.stride(1), self.cache.stride(2),
            num_warps=self.nw,
            num_stages=self.ns if num_stages is None else num_stages,
            **self.consts,
        )

    def stage2(self):
        _combine[self.grid2](
            self.mid, self.out,
            self.mid.stride(0), self.mid.stride(1), self.mid.stride(2),
            self.out.stride(0), self.out.stride(1),
            D=self.consts["D"],
            NUM_SPLITS=self.splits,
            BLOCK_S=32,
            num_warps=1,
        )


def event_ms(fn, reps, warm=3):
    """Device-busy time per call (XPU event pairs, queue pre-drained —
    the p174 face-B law)."""
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    ev0 = torch.xpu.Event(enable_timing=True)
    ev1 = torch.xpu.Event(enable_timing=True)
    acc = 0.0
    for _ in range(reps):
        ev0.record()
        fn()
        ev1.record()
        torch.xpu.synchronize()
        acc += ev0.elapsed_time(ev1)
    return acc / reps


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
    print(f"p175_k2stage — device={DEV}, torch={torch.__version__}")
    print(f"mxkv_src={_MXKV_DIR}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} R_TILE={RTILE}")
    rc = 0
    summary = []

    # ---- F0: correctness (v6 default vs per-row v2, LIMIT-LAW shape)
    for fmt in FMTS:
        q, cache, bt, lims = build_multi([2051] * 2, fmt, 31)
        o6 = mx_decode_rows_dot(q, cache, bt, lims, SCALE)
        o2 = torch.empty_like(o6)
        for r_ in range(q.shape[0]):
            g_ = r_ // RTILE
            o2[r_: r_ + 1] = mx_decode(
                q[r_: r_ + 1], cache, bt[g_: g_ + 1], lims[r_: r_ + 1], SCALE,
            )
        torch.xpu.synchronize()
        d = float((o6 - o2).abs().amax().item())
        ok = d <= 2e-3 and torch.isfinite(o6).all()
        print(f"[{fmt}] F0 v6 vs per-row v2 ctx2051x2: max|d|={d:.3e} "
              f"{'OK' if ok else 'FAIL'}")
        rc |= 0 if ok else 1

    # ---- F1+F2: stage split + splits scaling (G1 problem cells)
    for fmt in FMTS:
        for ctx in (8192, 32768):
            q, cache, bt, lims = build_multi([ctx], fmt, 19)
            reps = 6 if ctx >= 32768 else 10

            def full(_q=q, _c=cache, _b=bt, _l=lims):
                return mx_decode_rows_dot(_q, _c, _b, _l, SCALE)

            for sp in (256, 128, 64, 32):
                os.environ["ARC_MXKV_V5_SPLITS"] = str(sp)
                sc = StagedCall(q, cache, bt, lims, SCALE, sp)
                t1 = event_ms(sc.stage1, reps)
                t2 = event_ms(sc.stage2, reps)
                tf = wall_ms(full, reps)
                share2 = t2 / tf if tf > 0 else 0.0
                tag = "F1" if sp == 256 else "F2"
                print(f"[{fmt}] {tag} ctx={ctx} splits={sp}: "
                      f"s1 {t1:.3f} | s2 {t2:.3f} | full {tf:.3f} ms "
                      f"(s2 share {share2:.0%})")
                if sp == 256:
                    summary.append((fmt, ctx, t1, t2, tf))
            os.environ.pop("ARC_MXKV_V5_SPLITS", None)

            # ---- F3: num_stages on the dot walk (splits=256)
            if ctx == 32768:
                sc = StagedCall(q, cache, bt, lims, SCALE, 256)
                for ns in (2, 3, 4):
                    t1 = event_ms(
                        lambda sc=sc, ns=ns: sc.stage1(num_stages=ns), reps
                    )
                    print(f"[{fmt}] F3 ctx=32768 splits=256 ns={ns}: "
                          f"s1 {t1:.3f} ms")

    print("--- K2 stage-split decision table (v6 default splits=256) ---")
    for fmt, ctx, t1, t2, tf in summary:
        print(f"[{fmt}] ctx={ctx}: s1 {t1:.3f} ({t1 / tf:.0%}) "
              f"s2 {t2:.3f} ({t2 / tf:.0%}) full {tf:.3f} ms")
    print(f"K2_STAGE_{'DONE' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
