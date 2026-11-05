#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p621b r91 — LSE-capable aten ops + the two-part split-route prototype.

r90b killed the naive routes: XPU fused SDPA rejects every explicit
mask AND requires equal q/kv heads (enable_gqa unsupported; r89's fast
"SDPA" was the MATH backend, which memory-blows at serve shapes). The
remaining route idea rests on one fact: attention splits EXACTLY into

    part A: q x self-block   (Lq==Lk self KV, causal — top-left ==
            bottom-right when square, no mask needed)
    part B: q x prefix KV    (fully dense, no causal structure, no mask)
    merge : out = (outA*eA + outB*eB)/(eA+eB) row-wise, where
            eA = exp(lseA - m), eB = exp(lseB - m), m = max(lseA, lseB)

— IF both parts can run on a fast kernel that RETURNS the row
logsumexp. This leg:

  A. inventory: aten scaled_dot_product ops on this XPU build + which
     accept (equal-heads causal square, equal-heads dense Lq!=Lk) and
     return lse.
  B. rates of the survivors at serve shapes (iters=3, big shapes).
  C. split-merge numerics gate on (512, 2048) vs eager bottom-right
     reference (fro < 5e-3).
  D. full split-route prototype vs stock on the U32k pattern (gather
     paged fp8 KV -> dequant fp16 -> expand heads 2->12 for parts that
     need it -> A/B -> merge), end-to-end ms + fro vs stock.

RULE (r90, unchanged): >= 1.8x on the U32k aggregate with numerics
green ships the route patch; else the ESIMD v2 build opens.

Run: python -u r91_split_route.py [--out out.json]
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


def make_dense(rows, kv_len, heads, device="xpu:0", seed=7):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, heads, rows, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    k = torch.randn(1, heads, kv_len, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    v = torch.randn(1, heads, kv_len, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    return q, k, v


def eager_ref(q, k, v, rows, kv_len):
    kf = k.float().repeat_interleave(QH // KVH, dim=1)
    vf = v.float().repeat_interleave(QH // KVH, dim=1)
    att = torch.einsum("bhqd,bhkd->bhqk", q.float(), kf) * SCALE
    m = torch.ones(rows, kv_len, dtype=torch.bool, device=q.device)
    att = att.masked_fill(~torch.tril(m, diagonal=kv_len - rows),
                          float("-inf"))
    return torch.einsum("bhqk,bhkd->bhqd", att.softmax(-1), vf)


def stock_paged(q, k, v, cu_q, seqused, table, ks, vs):
    dummy = torch.zeros_like(cu_q)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        SCALE, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


def make_paged(rows, kv_len, seed, device="xpu:0"):
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


def probe_aten_ops(report):
    print("ATEN OPS:")
    names = [n for n in dir(torch.ops.aten)
             if "scaled_dot_product" in n]
    print(f"  available: {names}")
    report["aten_ops"] = names
    return names


def try_efficient_attn(q, k, v, is_causal):
    """aten::_scaled_dot_product_efficient_attention — CUDA/XPU signature
    (q, k, v, attn_bias, compute_log_sumexp, dropout_p, is_causal,
    scale) -> (output, log_sumexp, philox_seed, philox_offset,
    seed_el). Returns (out, lse) or raises."""
    r = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, is_causal, scale=SCALE)
    return r[0], r[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="r91_split_route.json")
    a = ap.parse_args()
    report = {}
    dev = "xpu:0"
    torch.xpu.set_device(dev)
    probe_aten_ops(report)

    # ---- B: efficient-attention acceptance + rates (equal heads) ----
    rows, kv_len = 8192, 8192
    q12, k12, v12 = make_dense(rows, kv_len, QH, seed=7)  # equal heads
    eff = {}
    try:
        out, lse = try_efficient_attn(q12, k12, v12, True)
        t = timed(lambda: try_efficient_attn(q12, k12, v12, True), 5)
        eff["causal_8kx8k"] = {"ms": round(t, 2), "tflops": round(
            flops_useful(rows, kv_len) / (t * 1e-3) / 1e12, 1),
            "lse_shape": list(lse.shape)}
    except Exception as e:  # noqa: BLE001
        eff["causal_8kx8k"] = f"{type(e).__name__}: {e}"[:200]
    print(f"B efficient causal 8kx8k: {eff['causal_8kx8k']}")
    # dense Lq != Lk (the prefix part): q 8192 x kv 24576, no causal
    try:
        qd, kd, vd = make_dense(rows, 24576, QH, seed=8)
        out, lse = try_efficient_attn(qd, kd, vd, False)
        t = timed(lambda: try_efficient_attn(qd, kd, vd, False), 5)
        fl = 2 * 2 * rows * 24576 * QH * HD
        eff["dense_8kx24k"] = {"ms": round(t, 2), "tflops": round(
            fl / (t * 1e-3) / 1e12, 1)}
        del qd, kd, vd
        torch.xpu.empty_cache()
    except Exception as e:  # noqa: BLE001
        eff["dense_8kx24k"] = f"{type(e).__name__}: {e}"[:200]
    print(f"B efficient dense 8kx24k: {eff['dense_8kx24k']}")
    # GQA without expansion (does XPU efficient take 12/2?)
    try:
        qg, kg, vg = make_dense(2048, 4096, QH, seed=9)
        k2 = kg[:, :KVH]
        v2 = vg[:, :KVH]
        out, lse = try_efficient_attn(qg, k2, v2, True)
        eff["gqa_2048x4096"] = "ACCEPTED"
        del qg, kg, vg
    except Exception as e:  # noqa: BLE001
        eff["gqa_2048x4096"] = f"{type(e).__name__}: {e}"[:200]
    print(f"B efficient gqa 2048x4096: {eff['gqa_2048x4096']}")
    report["efficient"] = eff
    del q12, k12, v12
    torch.xpu.empty_cache()

    # ---- C: split-merge numerics on (512, 2048) ----
    ok = False
    try:
        rows, kv_len = 512, 2048
        q, k, v = make_dense(rows, kv_len, KVH, seed=11)  # KVH heads; q?
        qh = make_dense(rows, kv_len, QH, seed=11)[0]  # same seed q, 12 h
        # q comes from the 12-head gen; build 12/2 pair explicitly:
        g = torch.Generator(device=dev).manual_seed(11)
        q12 = torch.randn(1, QH, rows, HD, device=dev, dtype=torch.float16,
                          generator=g) * 0.5
        k2 = torch.randn(1, KVH, kv_len, HD, device=dev,
                         dtype=torch.float16, generator=g) * 0.5
        v2 = torch.randn(1, KVH, kv_len, HD, device=dev,
                         dtype=torch.float16, generator=g) * 0.5
        ref = eager_ref(q12, k2, v2, rows, kv_len)
        # split: prefix kv [0:kv_len-rows], self kv [kv_len-rows:]
        pl = kv_len - rows
        kp = k2[:, :, :pl].repeat_interleave(QH // KVH, dim=1)
        vp = v2[:, :, :pl].repeat_interleave(QH // KVH, dim=1)
        ks_ = k2[:, :, pl:].repeat_interleave(QH // KVH, dim=1)
        vs_ = v2[:, :, pl:].repeat_interleave(QH // KVH, dim=1)
        if pl > 0:
            oB, lB = try_efficient_attn(q12, kp, vp, False)
        oA, lA = try_efficient_attn(q12, ks_, vs_, True)
        # lse shapes: [B, H, Lq] or [B, H, Lq, 1] — normalize
        lA = lA.float().flatten(2).squeeze(-1) if lA.dim() == 4 \
            else lA.float()
        m = lA
        if pl > 0:
            lB = lB.float().flatten(2).squeeze(-1) if lB.dim() == 4 \
                else lB.float()
            m = torch.maximum(lA, lB)
            eA = torch.exp(lA - m).unsqueeze(-1)
            eB = torch.exp(lB - m).unsqueeze(-1)
            out = (oA.float() * eA + oB.float() * eB) \
                / (eA + eB).clamp_min(1e-20)
        else:
            out = oA.float()
        fro = ((out - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        ok = fro < 5e-3
        report["split_merge_gate"] = {"fro": round(fro, 6), "pass": ok}
        print(f"C split-merge gate: fro={fro:.3e} "
              f"[{'PASS' if ok else 'FAIL'}]")
    except Exception as e:  # noqa: BLE001
        report["split_merge_gate"] = f"{type(e).__name__}: {e}"[:250]
        print(f"C split-merge gate: EXCEPTION {e}")
    torch.xpu.empty_cache()

    # ---- D: full split-route prototype vs stock on U32k pattern ----
    if ok:
        rates = []
        agg_s = agg_r = 0.0
        for rows, kv_len in U32K_PATTERN:
            q, k, v, cu_q, seqused, table, ks, vs = make_paged(
                rows, kv_len, seed=41)
            ref = stock_paged(q, k, v, cu_q, seqused, table, ks, vs)

            def route():
                kvl = int(seqused[0].item())
                np_ = (kvl + PAGE - 1) // PAGE
                ids = table[0, :np_].long()
                kg = k.index_select(0, ids).reshape(-1, KVH, HD)[:kvl]
                vg = v.index_select(0, ids).reshape(-1, KVH, HD)[:kvl]
                # -> [1, KVH, Lk, HD] fp16 dequant (scales are 1.0
                # class at serve; applied here when present)
                kc = (kg.permute(1, 0, 2).unsqueeze(0).float()
                      * ks[0].view(1, KVH, 1, 1)).to(torch.float16)
                vc = (vg.permute(1, 0, 2).unsqueeze(0).float()
                      * vs[0].view(1, KVH, 1, 1)).to(torch.float16)
                q3 = q.permute(1, 0, 2).unsqueeze(0)  # [1,12,Lq,256]
                pl = kvl - rows
                if pl > 0:
                    kp = kc[:, :, :pl].repeat_interleave(6, dim=1)
                    vp = vc[:, :, :pl].repeat_interleave(6, dim=1)
                    oB, lB = try_efficient_attn(q3, kp, vp, False)
                    lB = lB.float()
                kt = kc[:, :, pl:].repeat_interleave(6, dim=1)
                vt = vc[:, :, pl:].repeat_interleave(6, dim=1)
                oA, lA = try_efficient_attn(q3, kt, vt, True)
                lA = lA.float()
                m = torch.maximum(lA, lB)
                eA = torch.exp(lA - m).unsqueeze(-1)
                eB = torch.exp(lB - m).unsqueeze(-1)
                o = ((oA.float() * eA + oB.float() * eB)
                     / (eA + eB).clamp_min(1e-20))
                return o.squeeze(0).permute(1, 0, 2).to(
                    torch.float16).contiguous()

            o = route()
            fro = ((o.float() - ref.float()).norm()
                   / ref.float().norm().clamp_min(1e-9)).item()
            t_s = timed(lambda: stock_paged(q, k, v, cu_q, seqused,
                                            table, ks, vs), 5)
            t_r = timed(route, 5)
            agg_s += t_s * 16
            agg_r += t_r * 16
            rates.append({"rows": rows, "kv": kv_len,
                          "stock_ms": round(t_s, 2),
                          "route_ms": round(t_r, 2),
                          "speedup": round(t_s / t_r, 2),
                          "fro_vs_stock": round(fro, 5)})
            print(f"D rows={rows} kv={kv_len}: stock={t_s:.1f} ms "
                  f"route={t_r:.1f} ms x{t_s / t_r:.2f} fro={fro:.2e}")
            torch.xpu.empty_cache()
        report["rates"] = rates
        report["aggregate"] = {
            "u32k_stock_ms": round(agg_s, 1),
            "u32k_route_ms": round(agg_r, 1),
            "u32k_speedup": round(agg_s / agg_r, 2)}
        print(f"D U32k agg: stock={agg_s:.0f} route={agg_r:.0f} ms -> "
              f"{agg_s / agg_r:.2f}x")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
