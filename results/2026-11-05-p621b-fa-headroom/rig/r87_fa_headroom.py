#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p621b r87 — dense-FA headroom microbench (the P6.21 decision leg).

Question this leg answers (plan P6.21, FA lane): can ANY dense-FA kernel
beat the stock ``_vllm_fa2_C.varlen_fwd`` (vxk) by >= 2x on the serve
slice? The banked p72 economics (results/2026-10-31-p72-rungb-dense-fa-
esimd/LEGS.md) measured v1 ESIMD at ~26x SLOWER and derived "sharing 6x
x block-load 4x = ~24x = parity" — i.e. stock effective bandwidth ~=
v1_rate x 26/6 ~= 173 GB/s. Whether >= 2x headroom EXISTS above stock
was never measured. This leg measures it directly.

Method (telemetry, end-to-end at the op boundary):
  * Reconstruct the captured serve contract (p72 leg-F ROUTED capture):
    bs=1 varlen, q [rows,12,256] fp16 contiguous, K/V paged fp8-e4m3
    page 832 [pages,832,2,256] contiguous, block_table int32, causal,
    uniform stride-0 descales (value 1.0 at serve; stock mis-indexing is
    invisible under uniform — p72 PROBE.md law).
  * Shapes = the real U32k/U64k chunked-prefill call pattern (MNBT 8192,
    16 dense FA layers): chunk j attends to j x 8192 KV. U32k aggregate =
    4 chunks x 16 layers = 64 calls. Plus the captured real probe call
    (6656 rows / 3072 kv) as a sanity anchor.
  * Time stock per shape (xpu events, 30 iters) + v1 ESIMD .so (same
    shapes, fro-sanity first) -> the three-way record.
  * Measure the card's copy ceilings: 512 MiB HBM stream + the exact
    per-call working sets (KV 33.5 MB @32k, 67.1 MB @64k; q+out) -> if
    those ride L2 the ideal rate is the working-set rate.
  * S_lo = t_stock / t_ideal under the PERFECT-HBM-once model (each KV
    byte leaves HBM once per call; q+out once; everything else cached) at
    the FASTEST measured rate. S_lo is an upper bound on ANY kernel's
    speedup: no implementation can beat it. Ideal rate uses
    max(one-way HBM, working-set) so the bound stays honest (faster ideal
    -> larger S_lo -> harder decision, not easier).

PRE-REGISTERED DECISION RULE (plan P6.21, written before the leg):
  S_lo(U32k aggregate) < 2.0  -> FA lane CLOSED on physics (no kernel can
                                 halve the slice) -> bank negative.
  S_lo(U32k aggregate) >= 2.0 -> v2 build lane opens with the measured
                                 gap as its target.

Run (rig cert9 container, GPUs quiet — W14 stopped for the window):
  /opt/venv/bin/python r87_fa_headroom.py [--out out.json] [--quick]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

# register the stock op before the first torch.ops._vllm_fa2_C call (the
# serve chain registers it via vllm._xpu_ops; standalone containers may
# only have vllm_xpu_kernels — either import works; p72 probe law)
try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401,E402
except ImportError:
    from vllm import _xpu_ops  # noqa: F401,E402

QH, KVH, HD = 12, 2, 256  # per-rank GQA (TP2 split of 24q/4kv), group 6
PAGE = 832  # dense-layer serve page (p72 leg-F capture; kernel dynamic)
SCALE = HD ** -0.5

# (rows, kv_len) — the serve call pattern. chunk j: rows 8192 (MNBT),
# kv = j*8192 causal context. U64k tail chunks included. (6656,3072) is
# the captured real leg-F probe call (sanity anchor).
SHAPES = [
    (6656, 3072),
    (8192, 8192),
    (8192, 16384),
    (8192, 24576),
    (8192, 32768),
    (8192, 40960),
    (8192, 49152),
    (8192, 57344),
    (8192, 65536),
]
# the U32k serve slice = chunks 1..4 x 16 layers (r67b: 3.122 s, 20.3% of
# rank0 device time) — the aggregate the decision rule uses.
U32K_PATTERN = [(8192, 8192), (8192, 16384), (8192, 24576), (8192, 32768)]
LAYERS = 16


def make_case(rows: int, kv_len: int, seed: int, device: str):
    """Serve-contract tensors; distinct random pages (blindness law)."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(rows, QH, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + PAGE - 1) // PAGE
    tp = npages + 32  # table pad (serve allocates wide; stock ignores 0s)
    k = (torch.randn(tp, PAGE, KVH, HD, device=device, generator=g) * 0.5
         ).to(torch.float8_e4m3fn)
    v = (torch.randn(tp, PAGE, KVH, HD, device=device, generator=g) * 0.5
         ).to(torch.float8_e4m3fn)
    # randperm needs a CPU generator (xpu generator rejected — r87a fix);
    # the xpu generator g keeps serving the tensor randoms above.
    gcpu = torch.Generator().manual_seed(seed * 7 + 1)
    ids = list(range(npages))
    ids = [ids[i] for i in torch.randperm(npages, generator=gcpu).tolist()]
    table = torch.tensor([ids + [0] * (tp - npages)], dtype=torch.int32,
                         device=device)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=device)
    ks = torch.ones(1, KVH, device=device, dtype=torch.float32)
    vs = torch.ones(1, KVH, device=device, dtype=torch.float32)
    return q, k, v, cu_q, seqused, table, ks, vs


def stock_call(q, k, v, cu_q, seqused, table, ks, vs):
    """Positional stock contract (p72 stock_ref verbatim)."""
    dummy = torch.zeros_like(cu_q)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        SCALE, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


def timed(fn, iters: int, warmup: int = 5):
    """Median/min per-call ms via xpu events (fallback: wall + sync)."""
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    try:
        ev = [(torch.xpu.Event(enable_timing=True),
               torch.xpu.Event(enable_timing=True)) for _ in range(iters)]
        for s, e in ev:
            s.record()
            fn()
            e.record()
        torch.xpu.synchronize()
        ts = [s.elapsed_time(e) for s, e in ev]
    except Exception:
        import time as _t
        ts = []
        for _ in range(iters):
            t0 = _t.perf_counter()
            fn()
            torch.xpu.synchronize()
            ts.append((_t.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def copy_gbps(nbytes: int, device: str, iters: int = 30):
    """d2d copy rate counting read+write (GB/s)."""
    src = torch.empty(nbytes, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    med, _ = timed(lambda: dst.copy_(src), iters)
    return 2 * nbytes / (med * 1e-3) / 1e9


def kv_bytes(kv_len: int) -> int:
    return kv_len * KVH * HD * 2 * 1  # K+V fp8: tokens x 2 heads x 256 x 2


def qo_bytes(rows: int) -> int:
    return 2 * rows * QH * HD * 2  # q fp16 in + out fp16 (weights not attn)


def v1_fro(v1, q, k, v, cu_q, seqused, table, ks, vs, ref):
    o1 = torch.empty_like(q)
    v1.fa_dense_varlen_fwd(q, k, v, cu_q, seqused, table, o1, ks, vs, SCALE)
    fro = ((o1.float() - ref.float()).norm()
           / ref.float().norm().clamp_min(1e-9)).item()
    return fro, o1


def v1_diag(v1, device):
    """Isolate the v1 NaN trigger: probe-passing small case vs the r87
    serve case, page-multiple vs partial-tail, table width 32 vs wide."""
    print("V1_DIAG (fro vs stock; probe bank: 2.5-2.9e-3):")
    variants = [
        ("probe p832 small (bs2 512/512 kv1664/832 tbl32)",
         dict(bs=2, rows=[512, 512], kvs=[1664, 832], pad=32, seed=11)),
        ("r87 anchor (bs1 6656 kv3072 tbl+32 pad)",
         dict(bs=1, rows=[6656], kvs=[3072], pad=32, seed=1000)),
        ("tail-page mult (bs1 8192 kv2496=3x832)",
         dict(bs=1, rows=[8192], kvs=[2496], pad=32, seed=1001)),
        ("tail-page partial (bs1 2048 kv1900)",
         dict(bs=1, rows=[2048], kvs=[1900], pad=32, seed=1002)),
        ("r87 anchor tbl32-exact",
         dict(bs=1, rows=[6656], kvs=[3072], pad=0, seed=1000)),
    ]
    for name, vv in variants:
        g = torch.Generator(device=device).manual_seed(vv["seed"])
        rows = sum(vv["rows"])
        q = torch.randn(rows, QH, HD, device=device, dtype=torch.float16,
                        generator=g) * 0.5
        pages = [(l + PAGE - 1) // PAGE for l in vv["kvs"]]
        tp = sum(pages) + vv["pad"]
        k = (torch.randn(tp, PAGE, KVH, HD, device=device, generator=g)
             * 0.5).to(torch.float8_e4m3fn)
        v = (torch.randn(tp, PAGE, KVH, HD, device=device, generator=g)
             * 0.5).to(torch.float8_e4m3fn)
        gcpu = torch.Generator().manual_seed(vv["seed"] * 7 + 1)
        tab = []
        base = 0
        for np_ in pages:
            ids = list(range(base, base + np_))
            ids = [ids[i] for i in torch.randperm(
                np_, generator=gcpu).tolist()]
            tab.append(ids + [0] * (max(pages) + vv["pad"] - np_))
            base += np_
        table = torch.tensor(tab, dtype=torch.int32, device=device)
        cu_q = torch.tensor([0] + list(torch.cumsum(
            torch.tensor(vv["rows"]), 0)), dtype=torch.int32, device=device)
        seqused = torch.tensor(vv["kvs"], dtype=torch.int32, device=device)
        ks = torch.ones(vv["bs"], KVH, device=device, dtype=torch.float32)
        vs = torch.ones_like(ks)
        ref = stock_call(q, k, v, cu_q, seqused, table, ks, vs)
        fro, _ = v1_fro(v1, q, k, v, cu_q, seqused, table, ks, vs, ref)
        nan = bool(torch.isnan(ref).any().item())
        print(f"  {name:48s} fro={fro:.3e} stock_nan={nan}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="r87_fa_headroom.json")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        SHAPES[:] = SHAPES[:3]

    # v1 kernel (optional): staged p72 .so — loader law: torch already
    # imported above, so dlopen resolves libc10 in-process.
    v1 = None
    try:
        import fa_dense_ops  # noqa: E402  (sys.path from the leg wrapper)
        v1 = fa_dense_ops
    except Exception as e:  # noqa: BLE001 — record, never block the bench
        print(f"V1_LOAD: SKIP ({e!r})")

    devices = ["xpu:0", "xpu:1"]
    report = {"devices": {}, "decision": {}}
    for dev in devices:
        torch.xpu.set_device(dev)
        if v1 is not None and dev == devices[0]:
            v1_diag(v1, dev)
        props = torch.xpu.get_device_properties(torch.xpu.current_device())
        d = {"name": props.name, "total_mem_gb": round(
            getattr(props, "total_memory", 0) / 2**30, 2)}
        for k in ("l2_cache_size", "L2_cache_size", "cache_size"):
            if getattr(props, k, None):
                d[k] = getattr(props, k)
        print(f"\n=== {dev}: {d} ===")

        # ceilings: HBM stream + the two per-call working-set sizes
        hbm = copy_gbps(512 * 2**20, dev, args.iters)
        ws32 = copy_gbps(kv_bytes(32768), dev, args.iters)  # 33.55 MB
        ws64 = copy_gbps(kv_bytes(65536), dev, args.iters)  # 67.11 MB
        d["copy_hbm_gbps"] = round(hbm, 1)
        d["copy_ws32k_gbps"] = round(ws32, 1)
        d["copy_ws64k_gbps"] = round(ws64, 1)
        print(f"copy GB/s: hbm512M={hbm:.1f} ws32k={ws32:.1f} "
              f"ws64k={ws64:.1f}  (read+write counted)")
        # ideal one-way stream rate = max(hbm/2, best working-set/2):
        # copy counts a read AND a write per byte; attention is ~all-read
        # on KV, so one-way ~= copy/2 at the same working set.
        ideal_rate = max(hbm, ws32, ws64) / 2
        d["ideal_oneway_gbps"] = round(ideal_rate, 1)

        rows_out = []
        agg_stock_ms = agg_ideal_ms = 0.0
        for idx, (rows, kv_len) in enumerate(SHAPES):
            q, k, v, cu_q, seqused, table, ks, vs = make_case(
                rows, kv_len, seed=1000 + idx, device=dev)
            out = stock_call(q, k, v, cu_q, seqused, table, ks, vs)
            med, mn = timed(
                lambda: stock_call(q, k, v, cu_q, seqused, table, ks, vs),
                args.iters)
            traffic = kv_bytes(kv_len) + qo_bytes(rows)
            ideal_ms = traffic / (ideal_rate * 1e9) * 1e3
            s_lo = med / ideal_ms
            row = {"rows": rows, "kv": kv_len, "stock_ms_med": round(med, 3),
                   "stock_ms_min": round(mn, 3),
                   "mandatory_mb": round(traffic / 2**20, 1),
                   "ideal_ms": round(ideal_ms, 3), "s_lo": round(s_lo, 2)}
            # effective stock rate on mandatory traffic (info only)
            row["stock_eff_gbps"] = round(
                traffic / (med * 1e-3) / 1e9, 1)
            if v1 is not None:
                fro, o1 = v1_fro(v1, q, k, v, cu_q, seqused, table, ks,
                                 vs, out)
                row["v1_fro"] = round(fro, 6) if fro == fro else "nan"
                if fro == fro and fro < 5e-3:
                    med1, _ = timed(
                        lambda: v1.fa_dense_varlen_fwd(
                            q, k, v, cu_q, seqused, table, o1, ks, vs,
                            SCALE),
                        max(3, args.iters // 4))  # v1 slow: fewer iters
                    row["v1_ms_med"] = round(med1, 3)
                    row["v1_over_stock"] = round(med1 / med, 1)
                else:  # r87c: NaN on some serve shapes — diagnostic only,
                    # never fatal (the decision runs on stock + ceilings)
                    row["v1_timing"] = "SKIPPED(numerics)"
            rows_out.append(row)
            print(f"rows={rows:5d} kv={kv_len:6d} stock={med:9.3f} ms "
                  f"(min {mn:9.3f}) eff={row['stock_eff_gbps']:7.1f} GB/s "
                  f"ideal={ideal_ms:8.3f} ms S_lo={s_lo:6.2f}"
                  + (f" v1={row.get('v1_ms_med', '-'):>10} ms "
                     f"x{row.get('v1_over_stock', '-')}" if v1 else ""))
            if (rows, kv_len) in U32K_PATTERN:
                agg_stock_ms += med * LAYERS
                agg_ideal_ms += ideal_ms * LAYERS
        d["shapes"] = rows_out
        agg = {"u32k_stock_slice_ms": round(agg_stock_ms, 1),
               "u32k_ideal_slice_ms": round(agg_ideal_ms, 1),
               "u32k_s_lo": round(
                   agg_stock_ms / agg_ideal_ms, 2) if agg_ideal_ms else None,
               "banked_r67b_slice_ms": 3122.0}
        d["u32k_aggregate"] = agg
        print(f"U32k aggregate (4 chunks x 16 layers): stock="
              f"{agg_stock_ms:.0f} ms vs banked r67b 3122 ms | "
              f"S_lo={agg['u32k_s_lo']}")
        report["devices"][dev] = d

    # decision (pre-registered rule): worst device decides (TP2 = both
    # must improve; the slower bound is the honest one)
    s_vals = [v["u32k_aggregate"]["u32k_s_lo"]
              for v in report["devices"].values() if v["u32k_aggregate"][
                  "u32k_s_lo"]]
    s_min = min(s_vals) if s_vals else None
    report["decision"] = {
        "rule": "S_lo(U32k agg) >= 2.0 opens the v2 build lane; < 2.0 "
                "closes the FA lane on physics",
        "s_lo_min_across_devices": s_min,
        "verdict": ("OPEN v2 lane" if (s_min or 0) >= 2.0
                    else "CLOSE FA lane (physics)"),
    }
    print(f"\nDECISION: S_lo(min)={s_min} -> {report['decision']['verdict']}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
