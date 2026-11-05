#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P6.21 v2 numerics + rate probe: ESIMD SLM-tile dense-FA kernel vs
stock ``_vllm_fa2_C.varlen_fwd``.

v2 contract deltas vs the p72 v1 probe: bs == 1 only, pageSize % 32 == 0
(832 serve page + 1664 padded probe page both legal), kvLen >= qLen.
Probe discipline inherited (W14 laws): uniform-token blindness (distinct
random pages, shuffled table), fro/cos class gates, one-dimension-at-a-
time ladder.

The r87/r88 evidence demands an explicit OOB regression: every big case
snapshots inputs + a canary pool allocated around the output BEFORE the
v2 call and re-verifies after (v1 corrupted adjacent allocations at big
bs=1 shapes; its probe's .float() ref-copy had masked it).

Run (rig, GPUs quiet — stop W14 first):
  ARC_DENSE_FA_ESIMD_SO=/work/v2so/fa_dense_v2_ops.so \
  /opt/venv/bin/python probe_fa_dense_v2.py [--quick] [--out r.json]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401,E402
except ImportError:
    from vllm import _xpu_ops  # noqa: F401,E402

_so = os.environ.get("ARC_DENSE_FA_ESIMD_SO", "")
if _so:
    sys.path.insert(0, os.path.dirname(os.path.abspath(_so)))
elif os.path.isdir("/src/esimd-fa/python"):
    sys.path.insert(0, "/src/esimd-fa/python")
else:
    sys.path.insert(0, "/root")
import fa_dense_v2_ops  # noqa: E402  (loader law: name == basename)

QH, KVH, HD = 12, 2, 256
SCALE = HD ** -0.5
U32K_PATTERN = [(8192, 8192), (8192, 16384), (8192, 24576), (8192, 32768)]
SERVE_PAGE = 832
PROBE_PAGE = 1664


def make_case(rows, kv_len, seed, page=SERVE_PAGE, descale=True):
    """Serve contract (r87): paged fp8 KV, block table, varlen metadata,
    distinct random pages, shuffled within the single sequence."""
    g = torch.Generator(device="xpu").manual_seed(seed)
    q = torch.randn(rows, QH, HD, device="xpu", dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + page - 1) // page
    pad = max(1, npages // 8)  # headroom pages, table zero-padded
    k = (torch.randn(npages + pad, page, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(npages + pad, page, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    ids = list(range(npages))
    gcpu = torch.Generator().manual_seed(seed * 7 + 1)  # cpu gen law
    perm = torch.randperm(len(ids), generator=gcpu).tolist()
    table = torch.tensor([ids[i] for i in perm] + [0] * pad,
                         dtype=torch.int32, device="xpu").unsqueeze(0)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
    seqused = torch.tensor([kv_len], dtype=torch.int32, device="xpu")
    ks = vs = None
    if descale:
        kv = (torch.rand(1, device="xpu", generator=g) + 0.5).item()
        ks = torch.full((1, KVH), kv, device="xpu")
        vv = (torch.rand(1, device="xpu", generator=g) + 0.5).item()
        vs = torch.full((1, KVH), vv, device="xpu")
    return q, k, v, cu_q, seqused, table, ks, vs


def stock_ref(q, k, v, cu_q, seqused, table, ks, vs):
    dummy = torch.zeros_like(cu_q)
    if ks is None:
        ks = torch.ones(1, k.shape[2], device="xpu", dtype=torch.float32)
    if vs is None:
        vs = torch.ones(1, k.shape[2], device="xpu", dtype=torch.float32)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        SCALE, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


def v2_call(q, k, v, cu_q, seqused, table, ks, vs):
    out = torch.empty_like(q)
    fa_dense_v2_ops.fa_dense_varlen_fwd_v2(
        q, k, v, cu_q, seqused, table, out, ks, vs, SCALE)
    return out


def oob_guard(q, k, v, call):
    """r88 law: snapshot around the custom call and verify nothing the
    call should not touch changed. Canary pool sits next to `out` in the
    caching allocator (best-effort adjacency — the input snapshots are
    the hard guarantee)."""
    canary = torch.arange(1 << 21, dtype=torch.int32, device="xpu")
    sq, sk, sv, sc = (t.clone() for t in (q, k, v, canary))
    out = call()
    torch.xpu.synchronize()
    errs = []
    for name, a, b in (("q", q, sq), ("k", k, sk), ("v", v, sv),
                       ("canary", canary, sc)):
        if not torch.equal(a, b):
            errs.append(name)
    return out, errs


def fro_cos(o, ref):
    o, ref = o.float(), ref.float()
    fro = (o - ref).norm() / ref.norm().clamp_min(1e-9)
    cos = torch.nn.functional.cosine_similarity(
        o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    return fro.item(), cos


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    ev = [(torch.xpu.Event(enable_timing=True),
           torch.xpu.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.xpu.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in ev])


def flops_useful(rows, kv_len):
    return 4 * rows * max(1, kv_len - rows // 2) * QH * HD


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="probe_v2.json")
    a = ap.parse_args()
    report = {"numerics": [], "oob": [], "rates": [], "aggregate": {}}
    torch.xpu.set_device("xpu:0")

    # ladder: tiny -> slab boundary -> serve shapes; both page geometries;
    # descale on/off. rows=128 = the exact WG slab; 6656x3072 = the real
    # U32k chunk-1 geometry (52 slabs).
    cases = [
        ("tiny 16x64 p832", dict(rows=16, kv_len=64, seed=1,
                                 page=SERVE_PAGE)),
        ("slab 128x512 p832", dict(rows=128, kv_len=512, seed=2,
                                   page=SERVE_PAGE)),
        ("slab+1 136x544 p832", dict(rows=136, kv_len=544, seed=3,
                                     page=SERVE_PAGE)),
        ("512x2048 p832", dict(rows=512, kv_len=2048, seed=4,
                               page=SERVE_PAGE)),
        ("2048x4096 p1664", dict(rows=2048, kv_len=4096, seed=5,
                                 page=PROBE_PAGE)),
        ("3072x6656 p832 serve", dict(rows=3072, kv_len=6656, seed=6,
                                      page=SERVE_PAGE)),
        ("8192x16384 p832 nodesc", dict(rows=8192, kv_len=16384, seed=7,
                                        page=SERVE_PAGE, descale=False)),
        ("8192x32768 p832", dict(rows=8192, kv_len=32768, seed=8,
                                 page=SERVE_PAGE)),
    ]
    if a.quick:
        cases = cases[:3] + cases[5:6]
    allok = True
    for name, kw in cases:
        try:
            q, k, v, cu_q, seqused, table, ks, vs = make_case(**kw)
            ref = stock_ref(q, k, v, cu_q, seqused, table, ks, vs).float()
            out, errs = oob_guard(
                q, k, v, lambda: v2_call(q, k, v, cu_q, seqused, table,
                                         ks, vs))
            fro, cos = fro_cos(out, ref)
            ok = fro < 5e-3 and cos > 0.99999 and not errs
            allok &= ok
            report["numerics"].append({"case": name, "fro": round(fro, 6),
                                       "cos": round(cos, 8),
                                       "oob": errs, "pass": ok})
            print(f"[{'PASS' if ok else 'FAIL'}] {name:26s} "
                  f"fro={fro:.2e} cos={cos:.8f} oob={errs or 'clean'}")
            del q, k, v, ref, out
            torch.xpu.empty_cache()
        except Exception as e:  # noqa: BLE001 — the raise IS the datum
            allok = False
            report["numerics"].append({"case": name,
                                       "err": f"{type(e).__name__}: {e}"[:300]})
            print(f"[FAIL] {name:26s} EXC {type(e).__name__}: "
                  f"{str(e)[:200]}")
            torch.xpu.empty_cache()

    # ---- rates: U32k pattern, stock vs v2 (the >=2x bar) ----
    if allok or not a.quick:
        agg_s = agg_v = 0.0
        for rows, kv_len in (U32K_PATTERN if not a.quick
                             else U32K_PATTERN[:1]):
            q, k, v, cu_q, seqused, table, ks, vs = make_case(
                rows, kv_len, seed=41)
            t_s = timed(lambda: stock_ref(q, k, v, cu_q, seqused, table,
                                          ks, vs))
            t_v = timed(lambda: v2_call(q, k, v, cu_q, seqused, table,
                                        ks, vs))
            agg_s += t_s
            agg_v += t_v
            row = {"rows": rows, "kv": kv_len,
                   "stock_ms": round(t_s, 3), "v2_ms": round(t_v, 3),
                   "speedup": round(t_s / t_v, 2),
                   "v2_tflops": round(
                       flops_useful(rows, kv_len) / (t_v * 1e-3) / 1e12, 1)}
            report["rates"].append(row)
            print(f"RATE rows={rows} kv={kv_len}: stock={t_s:.2f} ms "
                  f"v2={t_v:.2f} ms x{t_s / t_v:.2f} "
                  f"({row['v2_tflops']} TF/s useful)")
            del q, k, v
            torch.xpu.empty_cache()
        report["aggregate"] = {
            "u32k_stock_ms": round(agg_s, 1), "u32k_v2_ms": round(agg_v, 1),
            "u32k_speedup": round(agg_s / agg_v, 2),
            "u32k_stock_tflops": round(
                sum(flops_useful(r, k) for r, k in U32K_PATTERN)
                / (agg_s * 1e-3) / 1e12, 1),
            "u32k_v2_tflops": round(
                sum(flops_useful(r, k) for r, k in U32K_PATTERN)
                / (agg_v * 1e-3) / 1e12, 1),
        }
        print(f"U32k agg: stock={agg_s:.0f} ms v2={agg_v:.0f} ms -> "
              f"{agg_s / agg_v:.2f}x "
              f"(bar >= 2.0x; {report['aggregate']['u32k_v2_tflops']} "
              f"TF/s useful vs 147 peak)")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"wrote {a.out}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
