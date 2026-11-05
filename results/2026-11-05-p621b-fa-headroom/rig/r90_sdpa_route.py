#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p621b r90 — SDPA-route prototype: rate + numerics at serve shapes.

r89 established: torch SDPA (XPU fused path) runs real flash attention
at ~98-104 TF/s when Lq==Lk (2.6-2.8x the stock vxk 37 TF/s), but
``is_causal`` is TOP-LEFT aligned — chunked prefill needs BOTTOM-RIGHT
(fro 4.7 = garbage on the chunk case). The route fix: explicit bool
mask ``tril(ones(Lq,Lk), diagonal=Lk-Lq)``.

This leg measures, at the four U32k shapes + numerics gates:
  1. SDPA with explicit bottom-right bool mask (fused-path rate? or
     mask-path collapse?) — THE decisive number for the route.
  2. The full route prototype end-to-end: gather paged fp8 KV via the
     block table -> dequant fp16 (scales applied) -> contiguous
     [1,KVH,Lk,256] -> SDPA(mask) -> out fp16 -> compare wall vs stock.
     Includes the q permute/contiguous and a per-(Lq,Lk) cached mask
     (the serve calls repeat the same geometry per chunk position).
  3. Numerics: route vs stock fro on small + one serve shape (both read
     the SAME paged fp8 KV; class gate fro<5e-3 like p72).
  4. bf16 variant rate (kv dequant to bf16 + q cast) for the record.

PRE-REGISTERED ROUTE RULE (plan P6.21): route ships as a patch iff
prototype end-to-end >= 1.8x stock on the U32k aggregate AND numerics
fro < 5e-3 vs stock on all gate shapes (then the ESIMD v2 build becomes
a later rung, not the P6.21 vehicle).

Run (rig cert9 container, GPUs quiet):
  /opt/venv/bin/python r90_sdpa_route.py [--out out.json]
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
U32K_PATTERN = [(8192, 8192), (8192, 16384), (8192, 24576), (8192, 32768)]
GATE_SHAPES = [(512, 2048), (2048, 4096), (6656, 3072), (8192, 16384),
               (8192, 32768)]
SCALE = HD ** -0.5


def timed(fn, iters=20, warmup=5):
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


def make_paged(rows, kv_len, seed, device):
    """The serve contract (r87): paged fp8 KV + block table + varlen
    metadata. Table entries 0..npages-1 in order (gather correctness is
    about pages, not shuffle; r87 proved shuffle-invariance for stock)."""
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(rows, QH, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + PAGE - 1) // PAGE
    k = (torch.randn(npages + 8, PAGE, KVH, HD, device=device, generator=g)
         * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(npages + 8, PAGE, KVH, HD, device=device, generator=g)
         * 0.5).to(torch.float8_e4m3fn)
    table = torch.nn.functional.pad(
        torch.arange(npages, dtype=torch.int32, device=device),
        (0, 8), value=0).unsqueeze(0)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=device)
    ks = torch.ones(1, KVH, device=device, dtype=torch.float32)
    vs = torch.ones_like(ks)
    return q, k, v, cu_q, seqused, table, ks, vs


def stock(q, k, v, cu_q, seqused, table, ks, vs):
    dummy = torch.zeros_like(cu_q)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        SCALE, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


_MASK_CACHE: dict = {}


def route(q, k, v, cu_q, seqused, table, ks, vs, dtype=torch.float16,
          use_mask=True):
    """The patch prototype: bs=1 path. Returns fp16 out [rows, QH, HD].
    Steps: gather+dequant KV -> [1,KVH,Lk,256]; q -> [1,QH,Lq,256];
    SDPA (explicit bottom-right mask, cached); out -> [rows,QH,HD]."""
    rows = q.shape[0]
    kv_len = int(seqused[0].item())
    npages = (kv_len + PAGE - 1) // PAGE
    ids = table[0, :npages].long()
    # gather pages to contiguous [npages*PAGE, KVH, HD] then flatten
    kg = k.index_select(0, ids).reshape(npages * PAGE, KVH, HD)
    vg = v.index_select(0, ids).reshape(npages * PAGE, KVH, HD)
    kc = kg[:kv_len].permute(1, 0, 2).unsqueeze(0)  # [1,KVH,Lk,HD] view
    vc = vg[:kv_len].permute(1, 0, 2).unsqueeze(0)
    ks_h = ks[0].view(1, KVH, 1, 1)
    vs_h = vs[0].view(1, KVH, 1, 1)
    if dtype == torch.bfloat16:
        kc = (kc.float() * ks_h).to(dtype)
        vc = (vc.float() * vs_h).to(dtype)
        qc = q.permute(1, 0, 2).unsqueeze(0).to(dtype)
    else:
        # fp16: dequant via float to apply scales, back to fp16
        kc = (kc.float() * ks_h).to(dtype)
        vc = (vc.float() * vs_h).to(dtype)
        qc = q.permute(1, 0, 2).unsqueeze(0)
    key = (rows, kv_len, dtype, use_mask)
    if use_mask:
        m = _MASK_CACHE.get(key)
        if m is None:
            m = torch.ones(rows, kv_len, dtype=torch.bool,
                           device=q.device)
            m = torch.tril(m, diagonal=kv_len - rows)
            _MASK_CACHE[key] = m
        o = torch.nn.functional.scaled_dot_product_attention(
            qc, kc, vc, attn_mask=m, enable_gqa=True)
    else:  # only correct when rows == kv_len (full attend)
        o = torch.nn.functional.scaled_dot_product_attention(
            qc, kc, vc, is_causal=True, enable_gqa=True)
    return o.squeeze(0).permute(1, 0, 2).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="r90_sdpa_route.json")
    a = ap.parse_args()
    report = {"numerics": [], "rates": [], "masks": {}}
    dev = "xpu:0"
    torch.xpu.set_device(dev)

    # ---- numerics gates (route vs stock, same paged fp8 KV) ----
    print("NUMERICS (route fp16 vs stock, same KV):")
    allok = True
    for rows, kv_len in GATE_SHAPES:
        q, k, v, cu_q, seqused, table, ks, vs = make_paged(
            rows, kv_len, seed=31, device=dev)
        ref = stock(q, k, v, cu_q, seqused, table, ks, vs).float()
        o = route(q, k, v, cu_q, seqused, table, ks, vs).float()
        fro = ((o - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        ok = fro < 5e-3
        allok &= ok
        report["numerics"].append({"rows": rows, "kv": kv_len,
                                   "fro": round(fro, 6), "pass": ok})
        print(f"  rows={rows:5d} kv={kv_len:6d} fro={fro:.3e} "
              f"[{'PASS' if ok else 'FAIL'}]")

    # ---- mask-only rate probe (no gather): the decisive fused-path
    # question — does SDPA keep ~98 TF/s with an explicit bool mask?
    for rows, kv_len in [(8192, 8192), (8192, 32768)]:
        g = torch.Generator(device=dev).manual_seed(7)
        q3 = torch.randn(1, QH, rows, HD, device=dev, dtype=torch.float16,
                         generator=g) * 0.5
        k3 = torch.randn(1, KVH, kv_len, HD, device=dev,
                         dtype=torch.float16, generator=g) * 0.5
        v3 = torch.randn(1, KVH, kv_len, HD, device=dev,
                         dtype=torch.float16, generator=g) * 0.5
        m = torch.ones(rows, kv_len, dtype=torch.bool, device=dev)
        m = torch.tril(m, diagonal=kv_len - rows)
        t = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
            q3, k3, v3, attn_mask=m, enable_gqa=True))
        tf = flops_useful(rows, kv_len) / (t * 1e-3) / 1e12
        report["masks"][f"{rows}x{kv_len}"] = {"ms": round(t, 3),
                                               "tflops": round(tf, 1)}
        print(f"MASK-ONLY {rows}x{kv_len}: {t:.2f} ms ({tf:.1f} TF/s "
              f"on bottom-right useful flops)")
        del q3, k3, v3, m
        torch.xpu.empty_cache()

    # ---- full route vs stock end-to-end ----
    print("RATES (end-to-end route fp16 vs stock):")
    agg_stock = agg_route = 0.0
    for rows, kv_len in U32K_PATTERN:
        q, k, v, cu_q, seqused, table, ks, vs = make_paged(
            rows, kv_len, seed=41, device=dev)
        stock(q, k, v, cu_q, seqused, table, ks, vs)
        route(q, k, v, cu_q, seqused, table, ks, vs)
        t_s = timed(lambda: stock(q, k, v, cu_q, seqused, table, ks, vs))
        t_r = timed(lambda: route(q, k, v, cu_q, seqused, table, ks, vs))
        agg_stock += t_s * 16
        agg_route += t_r * 16
        row = {"rows": rows, "kv": kv_len, "stock_ms": round(t_s, 3),
               "route_ms": round(t_r, 3),
               "speedup": round(t_s / t_r, 2),
               "route_tflops": round(
                   flops_useful(rows, kv_len) / (t_r * 1e-3) / 1e12, 1)}
        report["rates"].append(row)
        print(f"  rows={rows} kv={kv_len}: stock={t_s:.2f} ms "
              f"route={t_r:.2f} ms speedup={row['speedup']} "
              f"({row['route_tflops']} TF/s)")

    report["aggregate"] = {
        "u32k_stock_ms": round(agg_stock, 1),
        "u32k_route_ms": round(agg_route, 1),
        "u32k_speedup": round(agg_stock / agg_route, 2),
        "numerics_all_pass": allok,
    }
    print(f"U32k agg: stock={agg_stock:.0f} ms route={agg_route:.0f} ms "
          f"-> {report['aggregate']['u32k_speedup']}x "
          f"(numerics {'GREEN' if allok else 'RED'})")
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
