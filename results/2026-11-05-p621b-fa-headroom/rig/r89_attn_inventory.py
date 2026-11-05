#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p621b r89 — on-image attention-kernel inventory + rate table.

r88 decision: stock ``_vllm_fa2_C.varlen_fwd`` sits at 25% of the
practical compute roofline (147 TFLOP/s fp16 square; blended attention
bmm ceiling ~90). Before committing to a from-scratch ESIMD flash build
(p72 v1 was 26x slower — one thread per (row, head)), rung-C's own law:
INVENTORY the on-image kernels first. This leg rates every candidate at
the serve shapes:

  * stock vxk varlen_fwd (reference, = r87/r88)
  * torch SDPA dense fp16/bf16 causal with enable_gqa (the oneDNN
    fusion path — also the ceiling reference for a
    gather-KV-then-dense-attention route)
  * every flash-ish entry point importable from vllm_xpu_kernels /
    vllm_flash_attn (probed generically; failures recorded, not fatal)

A candidate that (a) accepts the geometry (GQA 12/2, hd 256, causal,
fp8-or-fp16 KV) and (b) rates >= 2x stock becomes the route target
(a p44-style reroute patch — days cheaper than a kernel build). Else the
v2 ESIMD build opens with the measured ceiling as its bar.

Run (rig cert9 container, GPUs quiet):
  /opt/venv/bin/python r89_attn_inventory.py [--out out.json]
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

QH, KVH, HD = 12, 2, 256
U32K_PATTERN = [(8192, 8192), (8192, 16384), (8192, 24576), (8192, 32768)]


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
    pairs = rows * max(1, kv_len - rows // 2)
    return 4 * pairs * QH * HD  # QK+PV MACs x2, 12 q-heads, hd 256


def stock_ms(rows, kv_len, device):
    """Reference: the serve kernel, paged fp8 KV (r87 contract)."""
    g = torch.Generator(device=device).manual_seed(7)
    q = torch.randn(rows, QH, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + 831) // 832
    k = (torch.randn(npages + 32, 832, KVH, HD, device=device,
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(npages + 32, 832, KVH, HD, device=device,
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    table = torch.nn.functional.pad(
        torch.arange(npages, dtype=torch.int32, device=device),
        (0, 32), value=0).unsqueeze(0)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device=device)
    seqused = torch.tensor([kv_len], dtype=torch.int32, device=device)
    ks = torch.ones(1, KVH, device=device, dtype=torch.float32)
    vs = torch.ones_like(ks)
    dummy = torch.zeros_like(cu_q)

    def call():
        torch.ops._vllm_fa2_C.varlen_fwd(
            q, k, v, None, cu_q, dummy, seqused, None, table, None,
            rows, kv_len, 0.0, ks, vs, HD ** -0.5, None, False, True,
            -1, -1, 0.0, False, None, 0, False, None, None)
    t = timed(call)
    return t, (q, k, v)


def sdpa_ms(rows, kv_len, device, dtype):
    """Dense SDPA fp16/bf16 GQA causal — the oneDNN fusion reference.
    k/v materialized contiguous fp16 (a gather route would pay this)."""
    g = torch.Generator(device=device).manual_seed(7)
    q = (torch.randn(1, QH, rows, HD, device=device, dtype=dtype,
                     generator=g) * 0.5)
    k = (torch.randn(1, KVH, kv_len, HD, device=device, dtype=dtype,
                     generator=g) * 0.5)
    v = (torch.randn(1, KVH, kv_len, HD, device=device, dtype=dtype,
                     generator=g) * 0.5)

    def call():
        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=QH != KVH)
    try:
        t = timed(call)
    except Exception as e:  # noqa: BLE001 — record, continue
        return None, f"{type(e).__name__}: {e}"
    return t, None


def sdpa_correctness(device):
    """Gates the SDPA route BEFORE trusting any rate: (a) small eager
    reference, both full-attend (Lq==Lk) and chunk (Lq<Lk, bottom-right
    causal — the serve semantics; torch is_causal alignment must prove
    itself), (b) sync-hardened timing cross-check on one serve shape."""
    print("SDPA_CORRECTNESS:")
    torch.manual_seed(7)
    ok_all = True
    for name, rows, kv in (("full Lq==Lk", 512, 512),
                           ("chunk Lq<Lk", 512, 2048)):
        q = (torch.randn(1, QH, rows, HD, device=device)
             * 0.5).to(torch.float16)
        k = (torch.randn(1, KVH, kv, HD, device=device)
             * 0.5).to(torch.float16)
        v = (torch.randn(1, KVH, kv, HD, device=device)
             * 0.5).to(torch.float16)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True)
        # eager bottom-right reference (serve semantics); GQA: kv head
        # serves 6 consecutive q heads -> repeat_interleave to 12
        qf = q.float()
        kf = k.float().repeat_interleave(QH // KVH, dim=1)
        vf = v.float().repeat_interleave(QH // KVH, dim=1)
        att = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * HD ** -0.5
        Lq, Lk = rows, kv
        mask = torch.ones(Lq, Lk, device=device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=Lk - Lq)
        att = att.masked_fill(~mask, float("-inf"))
        p = att.softmax(-1)
        ref = torch.einsum("bhqk,bhkd->bhqd", p, vf)
        fro = ((out.float() - ref).norm()
               / ref.norm().clamp_min(1e-9)).item()
        ok = fro < 5e-2  # fp16 attention vs fp32 eager: loose class gate
        ok_all &= ok
        print(f"  {name:14s} fro_vs_bottomright_ref={fro:.3e} "
              f"[{'PASS' if ok else 'FAIL'}]")
    # sync-hardened cross-check on a serve shape: wall vs event timing
    rows, kv = 8192, 32768
    q = (torch.randn(1, QH, rows, HD, device=device) * 0.5).to(torch.float16)
    k = (torch.randn(1, KVH, kv, HD, device=device) * 0.5).to(torch.float16)
    v = (torch.randn(1, KVH, kv, HD, device=device) * 0.5).to(torch.float16)
    import time as _t
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True)
    torch.xpu.synchronize()
    t0 = _t.perf_counter()
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True)
    torch.xpu.synchronize()
    wall = (_t.perf_counter() - t0) / 10 * 1e3
    ev = timed(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True), 10)
    print(f"  sync-hardened 8192x32768: wall={wall:.2f} ms event={ev:.2f} "
          f"ms (r89 event-only said 4.2 — if both now >>4.2, the r89 "
          "number was stream-escape; if both ~4.2 the op is not doing "
          "the math and correctness above must explain it)")
    return ok_all, wall, ev


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="r89_attn_inventory.json")
    a = ap.parse_args()
    report = {"inventory": {}, "rates": {}}

    # ---- inventory: what attention machinery exists on this image ----
    inv = report["inventory"]
    inv["torch"] = torch.__version__
    inv["sdp_kernel_backends"] = []
    try:
        from torch.nn.attention import SDPBackend  # noqa: F401
        inv["sdp_kernel_backends"] = [b.name for b in SDPBackend]
    except Exception as e:  # noqa: BLE001
        inv["sdp_kernel_backends"] = f"enum fail {e!r}"
    try:
        import vllm_xpu_kernels as vxk
        inv["vllm_xpu_kernels"] = [m for m in dir(vxk)
                                   if not m.startswith("_")]
        import inspect
        import vllm_xpu_kernels.flash_attn_interface as fai
        inv["flash_attn_interface_fns"] = [
            n for n, f in inspect.getmembers(fai, inspect.isfunction)
            if not n.startswith("_")]
    except Exception as e:  # noqa: BLE001
        inv["vllm_xpu_kernels"] = f"fail {e!r}"
    try:
        import vllm_flash_attn as vfa
        inv["vllm_flash_attn"] = [m for m in dir(vfa)
                                  if not m.startswith("_")][:30]
    except Exception as e:  # noqa: BLE001
        inv["vllm_flash_attn"] = f"fail {e!r}"
    print(json.dumps(inv, indent=1, default=str))

    for dev in ["xpu:0"]:
        torch.xpu.set_device(dev)
        ok, wall, ev = sdpa_correctness(dev)
        report["sdpa_correctness"] = {"bottomright_gates": ok,
                                      "wall_ms_8kx32k": round(wall, 2),
                                      "event_ms_8kx32k": round(ev, 2)}
        rates = []
        for rows, kv_len in U32K_PATTERN:
            fl = flops_useful(rows, kv_len)
            row = {"rows": rows, "kv": kv_len, "flips_tflop":
                   round(fl / 1e12, 3)}
            t_stock, _ = stock_ms(rows, kv_len, dev)
            row["stock_ms"] = round(t_stock, 3)
            row["stock_tflops"] = round(fl / (t_stock * 1e-3) / 1e12, 1)
            for name, dtype in (("sdpa_fp16", torch.float16),
                                ("sdpa_bf16", torch.bfloat16)):
                t, err = sdpa_ms(rows, kv_len, dev, dtype)
                if t is not None:
                    row[f"{name}_ms"] = round(t, 3)
                    row[f"{name}_tflops"] = round(
                        fl / (t * 1e-3) / 1e12, 1)
                    row[f"{name}_vs_stock"] = round(t_stock / t, 2)
                else:
                    row[f"{name}_ms"] = err[:120]
            rates.append(row)
            print(f"rows={rows} kv={kv_len}: stock={row['stock_ms']} ms "
                  f"({row['stock_tflops']} TF/s) | "
                  f"sdpa16={row.get('sdpa_fp16_ms')} ms "
                  f"({row.get('sdpa_fp16_tflops')} TF/s) | "
                  f"sdpabf16={row.get('sdpa_bf16_ms')} ms "
                  f"({row.get('sdpa_bf16_tflops')} TF/s)")
        report["rates"][dev] = rates

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
