#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p621b r90b — SDPA backend/mask matrix (fast-fail) + python
flash_attn_varlen_func variant. Run with python -u.

r90 lesson: default-context SDPA with an explicit bool mask at serve
shapes silently falls to a catastrophic path (>26 min for a handful of
calls). This leg makes the backend selection EXPLICIT so unsupported
combos fail fast with an error instead of hanging, and rates whatever
survives:

  A. forced FLASH / EFFICIENT / MATH x {bool mask, additive fp16 mask,
     mask[1,1,Lq,Lk]} at (512,2048) — accept-or-raise, then ms.
  B. survivors rated at (8192, 32768), iters=3, 60s cap per call via
     small iters (the r90 stall cannot recur at 3 iters x small probes).
  C. vllm_xpu_kernels.flash_attn_varlen_func (python): signature,
     correctness class vs stock op, rate at (8192, 32768).

Decision input: any variant >= 1.8x stock => route patch; none => the
ESIMD v2 build opens (r88 rule).
"""
from __future__ import annotations

import inspect
import json
import statistics
import sys

import torch
import torch.nn.attention as tna

try:
    import vllm_xpu_kernels.flash_attn_interface as fai
except ImportError:
    from vllm import _xpu_ops  # noqa: F401,E402
    fai = None

QH, KVH, HD, PAGE = 12, 2, 256, 832
SCALE = HD ** -0.5


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


def make_dense(rows, kv_len, device="xpu:0", seed=7):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(1, QH, rows, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    k = torch.randn(1, KVH, kv_len, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    v = torch.randn(1, KVH, kv_len, HD, device=device, dtype=torch.float16,
                    generator=g) * 0.5
    return q, k, v


def bottomright(rows, kv_len, device="xpu:0", dtype=torch.bool, shape2d=True):
    m = torch.ones(rows, kv_len, dtype=dtype, device=device)
    m = torch.tril(m, diagonal=kv_len - rows)
    return m if shape2d else m.unsqueeze(0).unsqueeze(0)


def sdpa(q, k, v, mask, backend):
    kw = dict(enable_gqa=True)
    if mask is not None:
        kw["attn_mask"] = mask
    else:
        kw["is_causal"] = True
    if backend is None:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw)
    with tna.sdpa_kernel([backend]):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, **kw)


def fro_vs_ref(out, ref):
    return ((out.float() - ref.float()).norm()
            / ref.float().norm().clamp_min(1e-9)).item()


def eager_ref(q, k, v, rows, kv_len):
    kf = k.float().repeat_interleave(QH // KVH, dim=1)
    vf = v.float().repeat_interleave(QH // KVH, dim=1)
    att = torch.einsum("bhqd,bhkd->bhqk", q.float(), kf) * SCALE
    mask = bottomright(rows, kv_len)
    att = att.masked_fill(~mask, float("-inf"))
    return torch.einsum("bhqk,bhkd->bhqd", att.softmax(-1), vf)


def main() -> int:
    report = {"A_small_matrix": [], "B_survivors": [], "C_python_fn": {}}
    dev = "xpu:0"
    torch.xpu.set_device(dev)

    # ---- A: forced-backend matrix at (512, 2048) ----
    rows, kv_len = 512, 2048
    q, k, v = make_dense(rows, kv_len)
    ref = eager_ref(q, k, v, rows, kv_len)
    variants = []
    for bname, backend in (("FLASH", tna.SDPBackend.FLASH_ATTENTION),
                           ("EFFICIENT", tna.SDPBackend.EFFICIENT_ATTENTION),
                           ("MATH", tna.SDPBackend.MATH)):
        for mname, mask in (
                ("bool2d", bottomright(rows, kv_len)),
                ("bool4d", bottomright(rows, kv_len, shape2d=False)),
                ("add16", None)):
            if mname == "add16":
                bb = bottomright(rows, kv_len)  # bool
                bm = torch.zeros_like(bb, dtype=torch.float16)
                mask = bm.masked_fill(~bb, float("-inf"))
            variants.append((f"{bname}/{mname}", backend, mask))
    for name, backend, mask in variants:
        try:
            out = sdpa(q, k, v, mask, backend)
            fro = fro_vs_ref(out, ref)
            t = timed(lambda: sdpa(q, k, v, mask, backend), 10)
            rec = {"variant": name, "accept": True, "fro": round(fro, 5),
                   "ms_small": round(t, 3)}
        except Exception as e:  # noqa: BLE001 — the raise IS the datum
            rec = {"variant": name, "accept": False,
                   "err": f"{type(e).__name__}: {e}"[:160]}
        report["A_small_matrix"].append(rec)
        print(f"A {name:16s} -> {rec}")

    # ---- B: correct + accepted mask variants at (8192, 32768) ----
    survivors = [r for r in report["A_small_matrix"]
                 if r.get("accept") and r.get("fro", 1) < 5e-2
                 and r["variant"].split("/")[0] != "MATH"]
    rows, kv_len = 8192, 32768
    if survivors:
        q2, k2, v2 = make_dense(rows, kv_len, seed=8)
        for s in survivors:
            bname, mname = s["variant"].split("/")
            backend = {"FLASH": tna.SDPBackend.FLASH_ATTENTION,
                       "EFFICIENT": tna.SDPBackend.EFFICIENT_ATTENTION}[bname]
            if mname == "bool2d":
                mask = bottomright(rows, kv_len)
            elif mname == "bool4d":
                mask = bottomright(rows, kv_len, shape2d=False)
            else:
                bb = bottomright(rows, kv_len)
                bm = torch.zeros_like(bb, dtype=torch.float16)
                mask = bm.masked_fill(~bb, float("-inf"))
            try:
                t = timed(lambda: sdpa(q2, k2, v2, mask, backend), 3)
                rec = {"variant": s["variant"], "ms": round(t, 2),
                       "tflops": round(
                           flops_useful(rows, kv_len)
                           / (t * 1e-3) / 1e12, 1)}
            except Exception as e:  # noqa: BLE001
                rec = {"variant": s["variant"],
                       "err": f"{type(e).__name__}: {e}"[:160]}
            report["B_survivors"].append(rec)
            print(f"B {s['variant']:16s} -> {rec}")
        del q2, k2, v2
        torch.xpu.empty_cache()

    # ---- C: python flash_attn_varlen_func ----
    if fai is not None:
        c = report["C_python_fn"]
        try:
            c["signature"] = str(inspect.signature(
                fai.flash_attn_varlen_func))[:400]
        except Exception as e:  # noqa: BLE001
            c["signature"] = repr(e)
        # rate it on the r87 paged contract at one serve shape
        rows, kv_len = 8192, 32768
        g = torch.Generator(device=dev).manual_seed(9)
        qp = torch.randn(rows, QH, HD, device=dev, dtype=torch.float16,
                         generator=g) * 0.5
        npages = (kv_len + PAGE - 1) // PAGE
        kk = (torch.randn(npages + 8, PAGE, KVH, HD, device=dev,
                          generator=g) * 0.5).to(torch.float8_e4m3fn)
        vv = (torch.randn(npages + 8, PAGE, KVH, HD, device=dev,
                          generator=g) * 0.5).to(torch.float8_e4m3fn)
        table = torch.nn.functional.pad(
            torch.arange(npages, dtype=torch.int32, device=dev),
            (0, 8), value=0).unsqueeze(0)
        cu = torch.tensor([0, rows], dtype=torch.int32, device=dev)
        su = torch.tensor([kv_len], dtype=torch.int32, device=dev)
        ones = torch.ones(1, KVH, device=dev, dtype=torch.float32)
        try:
            out = fai.flash_attn_varlen_func(
                qp, kk, vv, cu, su, max_seqlen_q=rows,
                max_seqlen_k=kv_len, softmax_scale=SCALE, causal=True,
                block_table=table, k_descale=ones, v_descale=ones)
            t = timed(lambda: fai.flash_attn_varlen_func(
                qp, kk, vv, cu, su, max_seqlen_q=rows,
                max_seqlen_k=kv_len, softmax_scale=SCALE, causal=True,
                block_table=table, k_descale=ones, v_descale=ones), 3)
            c["rate_8kx32k"] = {"ms": round(t, 2), "tflops": round(
                flops_useful(rows, kv_len) / (t * 1e-3) / 1e12, 1)}
        except Exception as e:  # noqa: BLE001
            c["rate_8kx32k"] = f"{type(e).__name__}: {e}"[:200]
        print(f"C python fn -> {c}")

    with open("r90b_backend_matrix.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    print("wrote r90b_backend_matrix.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
