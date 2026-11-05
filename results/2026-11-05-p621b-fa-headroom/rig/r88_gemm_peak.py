#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p621b r88 — XMX peak + dual-roofline decision for the FA lane.

r87 measured stock ``_vllm_fa2_C.varlen_fwd`` at serve shapes: 1.0-1.7
GB/s effective on mandatory traffic (memory roofline S ~97-265x — a
fantasy bound) but ~37-43 TFLOP/s raw on the QK+PV arithmetic. The
problem's arithmetic intensity is ~24 KFLOP/B => COMPUTE-bound; the
honest ceiling is max(memory, compute). This leg measures the practical
fp16 XMX peak at the exact attention bmm shape classes and computes the
dual-roofline speedup bound:

  S(shape) = t_stock / max(t_mem, t_flop)
  t_mem  = mandatory_bytes / one_way_copy_rate   (r87: ~264-268 GB/s)
  t_flop = useful_flops / measured bmm rate

PRE-REGISTERED RULE (extends r87's, plan P6.21): the lane opens for a
v2 build iff S(U32k aggregate) >= 2.0 on BOTH cards under the DUAL
roofline. Compute-bound verdicts close lanes honestly: if stock sits at
>=50% of the measured bmm peak, no attention kernel on this silicon can
halve the slice.

Also: the r87 v1-adjacent-corruption proof (stock byte-compare around a
v1 call — probe's ref was .float()-copied before v1 ran, which is why
p72 never saw it).

Run (rig cert9 container, GPUs quiet): PYTHONPATH=/work/v1so \
  /opt/venv/bin/python r88_gemm_peak.py [--out out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401,E402
except ImportError:
    from vllm import _xpu_ops  # noqa: F401,E402

QH, KVH, HD, PAGE = 12, 2, 256, 832
SCALE = HD ** -0.5
U32K_PATTERN = [(8192, 8192), (8192, 16384), (8192, 24576), (8192, 32768)]
# r87 measured one-way copy ceilings (GB/s) per device — reuse for t_mem
# unless re-measured here (this leg re-measures; r87 kept as fallback)
R87_ONEWAY = {"xpu:0": 268.3, "xpu:1": 268.5}


def timed(fn, iters=20, warmup=5):
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


def stock_call(q, k, v, cu_q, seqused, table, ks, vs):
    dummy = torch.zeros_like(cu_q)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        SCALE, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


def make_case(rows, kv_len, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(rows, QH, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + PAGE - 1) // PAGE
    tp = npages + 32
    k = (torch.randn(tp, PAGE, KVH, HD, device=device, generator=g)
         * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(tp, PAGE, KVH, HD, device=device, generator=g)
         * 0.5).to(torch.float8_e4m3fn)
    gcpu = torch.Generator().manual_seed(seed * 7 + 1)
    ids = [i for i in torch.randperm(npages, generator=gcpu).tolist()]
    table = torch.tensor([ids + [0] * (tp - npages)], dtype=torch.int32,
                         device=device)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=device)
    ks = torch.ones(1, KVH, device=device, dtype=torch.float32)
    vs = torch.ones_like(ks)
    return q, k, v, cu_q, seqused, table, ks, vs


def flops_useful(rows, kv_len):
    """QK + PV MACs x2, causal bottom-right (chunked prefill: q-rows
    attend prefix + own causal block => pairs = rows*(kv - rows/2))."""
    pairs = rows * max(1, kv_len - rows // 2)
    macs = 2 * pairs * QH * HD  # QK + PV over 12 q-heads, hd 256
    return 2 * macs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="r88_gemm_peak.json")
    a = ap.parse_args()

    v1 = None
    try:
        import fa_dense_ops  # noqa: E402
        v1 = fa_dense_ops
    except Exception as e:  # noqa: BLE001
        print(f"V1_LOAD: SKIP ({e!r})")

    report = {"devices": {}, "decision": {}}
    for dev in ["xpu:0", "xpu:1"]:
        torch.xpu.set_device(dev)
        d = {}
        print(f"\n=== {dev} ===")

        # ---- copy ceiling (re-measure; one-way = /2) ----
        src = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)
        dst = torch.empty_like(src)
        med, _ = timed(lambda: dst.copy_(src))
        copy_rw = 2 * src.numel() / (med * 1e-3) / 1e9
        oneway = copy_rw / 2
        d["copy_rw_gbps"] = round(copy_rw, 1)
        d["oneway_gbps"] = round(oneway, 1)

        # ---- fp16 XMX peaks at attention shape classes ----
        # (a) big square matmul — the classic peak probe
        m = torch.randn(8192, 8192, device=dev, dtype=torch.float16) * 0.1
        n = torch.randn(8192, 8192, device=dev, dtype=torch.float16) * 0.1
        med, _ = timed(lambda: m @ n)
        square = 2 * 8192**3 / (med * 1e-3) / 1e12
        # (b) QK^T bmm shape class: [12, 8192, 256] @ [12, 256, 32768]
        q3 = torch.randn(12, 8192, 256, device=dev,
                         dtype=torch.float16) * 0.1
        k3 = torch.randn(12, 256, 32768, device=dev,
                         dtype=torch.float16) * 0.1
        med, _ = timed(lambda: torch.bmm(q3, k3))
        qk = 2 * 12 * 8192 * 256 * 32768 / (med * 1e-3) / 1e12
        # (c) PV bmm shape class: [12, 8192, 32768] @ [12, 32768, 256]
        # (scores fp16 6.4 GB — fits; PV is the transpose-density side)
        p3 = torch.randn(12, 8192, 16384, device=dev,
                         dtype=torch.float16) * 0.1
        v3 = torch.randn(12, 16384, 256, device=dev,
                         dtype=torch.float16) * 0.1
        med, _ = timed(lambda: torch.bmm(p3, v3))
        pv = 2 * 12 * 8192 * 16384 * 256 / (med * 1e-3) / 1e12
        d["tflops_square8k"] = round(square, 1)
        d["tflops_qk_bmm"] = round(qk, 1)
        d["tflops_pv_bmm"] = round(pv, 1)
        peak = max(square, qk, pv)
        d["peak_tflops"] = round(peak, 1)
        print(f"copy(r+w)={copy_rw:.1f} GB/s oneway={oneway:.1f} | "
              f"TFLOP/s fp16: square={square:.1f} qk_bmm={qk:.1f} "
              f"pv_bmm={pv:.1f} -> peak={peak:.1f}")
        del m, n, q3, k3, p3, v3
        torch.xpu.empty_cache()

        # ---- v1 adjacent-corruption proof (r87 side finding) ----
        if v1 is not None:
            q, k, v, cu_q, seqused, table, ks, vs = make_case(
                2048, 1900, seed=1002, device=dev)
            ref1 = stock_call(q, k, v, cu_q, seqused, table, ks, vs)
            ref1_copy = ref1.clone()  # snapshot BEFORE v1 runs
            o1 = torch.empty_like(q)
            v1.fa_dense_varlen_fwd(q, k, v, cu_q, seqused, table, o1,
                                   ks, vs, SCALE)
            ref2 = stock_call(q, k, v, cu_q, seqused, table, ks, vs)
            d["v1_oob_proof"] = {
                "ref1_vs_snapshot_after_v1": bool(
                    not torch.equal(ref1, ref1_copy)),
                "fresh_stock_after_v1_nan": bool(
                    torch.isnan(ref2).any().item()),
            }
            print(f"v1 OOB proof: ref1 mutated by v1 run = "
                  f"{d['v1_oob_proof']['ref1_vs_snapshot_after_v1']}, "
                  f"fresh-stock-after-v1 nan = "
                  f"{d['v1_oob_proof']['fresh_stock_after_v1_nan']}")

        # ---- stock FA at the U32k pattern + dual roofline ----
        rows_out = []
        agg_stock = agg_ideal = 0.0
        for idx, (rows, kv_len) in enumerate(U32K_PATTERN):
            q, k, v, cu_q, seqused, table, ks, vs = make_case(
                rows, kv_len, seed=2000 + idx, device=dev)
            stock_call(q, k, v, cu_q, seqused, table, ks, vs)
            med, mn = timed(
                lambda: stock_call(q, k, v, cu_q, seqused, table, ks, vs),
                20)
            bytes_min = kv_len * KVH * HD * 2 + 2 * rows * QH * HD * 2
            t_mem = bytes_min / (oneway * 1e9) * 1e3
            t_flop = flops_useful(rows, kv_len) / (peak * 1e12) * 1e3
            t_ideal = max(t_mem, t_flop)
            row = {"rows": rows, "kv": kv_len,
                   "stock_ms": round(med, 3),
                   "t_mem_ms": round(t_mem, 3),
                   "t_flop_ms": round(t_flop, 3),
                   "bound": "mem" if t_mem > t_flop else "flop",
                   "s_dual": round(med / t_ideal, 2),
                   "stock_pct_of_peak": round(
                       flops_useful(rows, kv_len)
                       / (med * 1e-3) / (peak * 1e12) * 100, 1)}
            rows_out.append(row)
            print(f"rows={rows} kv={kv_len}: stock={med:.2f} ms "
                  f"t_mem={t_mem:.3f} t_flop={t_flop:.3f} "
                  f"[{row['bound']}] S={row['s_dual']} "
                  f"({row['stock_pct_of_peak']}% of peak)")
            agg_stock += med * 16
            agg_ideal += t_ideal * 16
        d["u32k"] = {"rows": rows_out,
                     "agg_stock_ms": round(agg_stock, 1),
                     "agg_ideal_ms": round(agg_ideal, 1),
                     "agg_s_dual": round(agg_stock / agg_ideal, 2),
                     "agg_pct_of_peak": round(
                         agg_ideal / agg_stock * 100, 1)}
        print(f"U32k agg: stock={agg_stock:.0f} ms ideal(dual)="
              f"{agg_ideal:.0f} ms -> S_dual={d['u32k']['agg_s_dual']} "
              f"(stock at {d['u32k']['agg_pct_of_peak']}% of dual-ideal)")
        report["devices"][dev] = d

    s_min = min(v["u32k"]["agg_s_dual"] for v in report["devices"].values())
    report["decision"] = {
        "rule": "S_dual(U32k agg) >= 2.0 on BOTH cards opens the v2 "
                "build lane; < 2.0 closes the FA lane (compute-bound "
                "silicon limit)",
        "s_dual_min": s_min,
        "verdict": ("OPEN v2 lane" if s_min >= 2.0
                    else "CLOSE FA lane (roofline)"),
    }
    print(f"\nDECISION: S_dual(min)={s_min} -> "
          f"{report['decision']['verdict']}")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
