#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p157_v3_shortctx.py — P6.47 A9: the TRUE-SHAPE short-ctx probe.

Two gaps the A8 battery exposed once the r159 e2e cells landed:
  1. Serve decode runs at ctx ~1-3K where v3's Hk-grid starves (A8: v3
     0.531 vs v2join 0.210 ms @1K, D=128 shape) — the s_eff floor now
     gives v3 one tile per split; this probe measures whether that
     restores the short-ctx win.
  2. The offline batteries tuned at HK=8/Hq=32/D=128 (N_REP=4) — the
     REAL qwen3.8-27b-fp8 config is HK=4/Hq=24/D=256 (N_REP=6). At D=256
     a BLOCK_N=64 K-tile holds 64KB fp32 of live state — the spill cliff
     moves and BLOCK_N=32 was never swept. This probe sweeps at the true
     shape.

Faces: correctness spot (v3 vs v2, 549 tokens, per fmt, d <= 2e-3),
decode timing (ctx {1024, 4096, 32768} x decoder {v2,v3} x combos
{(32,4,s2),(32,8,s2),(64,4,s2),(64,8,s2)} x fmt, table flavor),
verify-batch face (R=32 batched split-verify call shape, BARE decoder
calls = the shipping config, vs the old per-row loop: d <= 2e-3 +
launch-count timing), the fp16-q dtype face (<= 2e-3, out dtype law),
and mx_store timing at the true shape (N=4 spec rows, per fmt). A9 run
2 (2026-11-12) validates the rebake it produced: default v2 + BLOCK_N=32
table on every serve cell.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

_MXKV_DIR = (
    Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
)


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
mx_decode = mxkv.mx_decode
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes
tmx = sys.modules["mx_kv_lane.triton_mx_decode"]

# TRUE qwen3.8-27b-fp8 attention shape (config.json, rig /models).
D = 256
HK = 4
HREP = 6  # Hq = 24
BLOCK = 64
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
COMBOS = ((32, 4, 2), (32, 8, 2), (64, 4, 2), (64, 8, 2))


def build(ctx, fmt, seed, bsz=1):
    n_blocks = ctx // BLOCK + 2
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_blocks, generator=gen)
    tab = perm[: (ctx + BLOCK - 1) // BLOCK]
    tb = torch.tensor([int(tab[t // BLOCK]) for t in range(ctx)])
    tp = torch.tensor([t % BLOCK for t in range(ctx)])
    cache = torch.zeros(
        n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
    )
    xk = torch.randn(ctx, HK, D, generator=gen).to(DEV)
    xv = torch.randn(ctx, HK, D, generator=gen).to(DEV)
    mx_store(xk, xv, cache, tb.to(DEV), tp.to(DEV))
    torch.xpu.synchronize()
    bt = tab[None, :].to(DEV).to(torch.int32).expand(bsz, -1).contiguous()
    sl = torch.full((bsz,), ctx, dtype=torch.int32, device=DEV)
    q = torch.randn(bsz, HK * HREP, D, generator=gen).to(DEV)
    return q, cache, bt, sl


def timeit(fn, reps=20, warm=3):
    for _ in range(warm):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    print(f"p157_v3_shortctx — device={DEV}, torch={torch.__version__}")
    print(f"shape: D={D} HK={HK} Hq={HK * HREP} N_REP={HREP} (true qwen cfg)")
    rc = 0
    for fmt in FMTS:
        # correctness spot: v3 vs v2 at 549 (split boundaries differ ->
        # reassociation only; gate d <= 2e-3)
        q, cache, bt, sl = build(549, fmt, 31)
        tmx._SHIFT = None  # table flavor
        o3 = mx_decode(q, cache, bt, sl, SCALE, decoder="v3")
        o2 = mx_decode(q, cache, bt, sl, SCALE, decoder="v2")
        torch.xpu.synchronize()
        d = float((o3 - o2).abs().amax().item())
        ok = d <= 2e-3 and torch.isfinite(o3).all()
        print(f"[{fmt}] v3-vs-v2 @549 true-shape: max|d|={d:.3e} {'OK' if ok else 'FAIL'}")
        rc |= 0 if ok else 1
        # dtype face: native-fp16 q vs the rounded-q fp32 path. fp16->fp32
        # promotion at the load is exact, but the kernel's intermediate
        # dtype follows the q/out element type at part of the pipeline, so
        # the two paths differ by O(one fp16 rounding) — A9 measured
        # 0.7-1.2e-4, two orders inside the MX quant-noise envelope (the
        # v3-vs-v2 reassociation gate is 2e-3). Gate: <= 2e-3 AND out
        # follows q's dtype (empty_like). Evidence for the engine's
        # .float()/out.to() cast removal.
        q16 = q.to(torch.float16)
        o16 = mx_decode(q16, cache, bt, sl, SCALE, decoder="v3")
        oref = mx_decode(q16.float(), cache, bt, sl, SCALE, decoder="v3")
        torch.xpu.synchronize()
        d16 = float((o16.float() - oref).abs().amax().item())
        ok16 = d16 <= 2e-3 and o16.dtype == torch.float16
        print(
            f"[{fmt}] fp16-q native vs rounded-fp32: max|d|={d16:.3e} "
            f"out.dtype={o16.dtype} {'OK' if ok16 else 'FAIL'}"
        )
        rc |= 0 if ok16 else 1
        # decode timing sweep
        for ctx in (1024, 4096, 32768):
            q2, cache2, bt2, sl2 = build(ctx, fmt, 7)
            rows = {}
            for dec in ("v2", "v3"):
                for bn, nw, ns in COMBOS:
                    tag = f"{dec}n{bn}w{nw}"
                    try:
                        def call(q=q2, c=cache2, b=bt2, s=sl2, dn=dec,
                                 bn_=bn, nw_=nw, ns_=ns):
                            return mx_decode(
                                q, c, b, s, SCALE, decoder=dn,
                                block_n=bn_, num_warps=nw_, num_stages=ns_,
                            )
                        rows[tag] = round(timeit(call), 4)
                    except Exception as exc:  # noqa: BLE001
                        rows[tag] = f"err:{type(exc).__name__}"
            best = min((v, k) for k, v in rows.items() if not isinstance(v, str))
            print(f"[{fmt}] ctx={ctx}: best {best[1]} {best[0]:.3f} ms")
            print(f"[{fmt}] ctx={ctx}: {rows}")
        # agg8 face: B=8 @2048 — the battery's agg8 cell (where the e2e
        # deficit is largest); same constexprs as the sweep, timing only.
        q8, cache8, bt8, sl8 = build(2048, fmt, 13, bsz=8)
        rows8 = {}
        for dec in ("v2", "v3"):
            for bn, nw, ns in COMBOS:
                tag = f"{dec}n{bn}w{nw}"

                def call8(q=q8, c=cache8, b=bt8, s=sl8, dn=dec,
                          bn_=bn, nw_=nw, ns_=ns):
                    return mx_decode(
                        q, c, b, s, SCALE, decoder=dn,
                        block_n=bn_, num_warps=nw_, num_stages=ns_,
                    )
                rows8[tag] = round(timeit(call8, reps=10), 4)
        best8 = min((v, k) for k, v in rows8.items())
        print(f"[{fmt}] B=8 ctx=2048: best {best8[1]} {best8[0]:.3f} ms")
        print(f"[{fmt}] B=8 ctx=2048: {rows8}")
        # verify-batch face: the engine's batched split-verify call shape
        # (_split_verify_decode, 2026-11-12): R = 8 reqs x 4 verify rows,
        # ONE mx_decode over (R, Hq, D) with duplicated bt rows and
        # per-row limits ctx-3..ctx (the LIMIT-LAW one-ahead allocator
        # pattern). BARE calls (decoder=None) — the shipping config the
        # engine actually launches. Must be BIT-EQUAL to the old per-row
        # call shape (same per-row math; the batch only widens grid dim
        # 0); the timing delta is the launch-count win the engine fix
        # banks.
        ctx_v = 2048
        qv, cachev, btv, _ = build(ctx_v, fmt, 19, bsz=8)
        RV = 32
        gen = torch.Generator().manual_seed(23)
        q_v = torch.randn(RV, HK * HREP, D, generator=gen).to(DEV)
        bt_rows = btv.repeat_interleave(4, dim=0).contiguous()
        sl_v = (ctx_v - 3 + (torch.arange(RV) % 4)).to(torch.int32).to(DEV)
        o_batch = mx_decode(q_v, cachev, bt_rows, sl_v, SCALE)
        o_rows = torch.empty_like(o_batch)
        for r in range(RV):
            o_rows[r : r + 1] = mx_decode(
                q_v[r : r + 1],
                cachev,
                btv[r // 4 : r // 4 + 1],
                sl_v[r : r + 1],
                SCALE,
            )
        torch.xpu.synchronize()
        dv = float((o_batch.float() - o_rows.float()).abs().amax().item())
        okv = dv <= 2e-3

        def callb(q=q_v, c=cachev, b=bt_rows, s=sl_v):
            return mx_decode(q, c, b, s, SCALE)

        def calll(q=q_v, c=cachev, b=btv, s=sl_v, rv=RV):
            for r in range(rv):
                mx_decode(
                    q[r : r + 1], c, b[r // 4 : r // 4 + 1],
                    s[r : r + 1], SCALE,
                )

        t_b = timeit(callb, reps=10)
        t_l = timeit(calll, reps=10)
        print(
            f"[{fmt}] verify-batch R=32 @2048: batched {t_b:.3f} ms vs "
            f"32-row loop {t_l:.3f} ms ({t_l / t_b:.1f}x) max|d|={dv:.3e} "
            f"{'OK' if okv else 'FAIL'}"
        )
        rc |= 0 if okv else 1
        # store timing at the true shape (spec verify rows: k+1 = 4)
        gen = torch.Generator().manual_seed(11)
        key = torch.randn(4, HK, D, generator=gen).to(DEV)
        val = torch.randn(4, HK, D, generator=gen).to(DEV)
        cache3 = torch.zeros(
            8, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        tb3 = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEV)
        tp3 = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEV)
        ms = timeit(
            lambda k=key, v=val, c=cache3, b=tb3, p=tp3: mx_store(k, v, c, b, p)
        )
        print(f"[{fmt}] store N=4 @D={D}: {ms:.4f} ms")
    print(f"A9_PROBE_{'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
