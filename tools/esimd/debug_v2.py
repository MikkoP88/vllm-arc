#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P6.21 v2 numerics bisection probe (first GPU run: fro 0.28-0.54,
cos 0.86-0.96 on every shape — structural bug, OOB clean).

Ladder discriminates the failure plane:
  8x64     one thread (lid 0) only        -> per-thread math (dpas operands,
                                             softmax, epilogue)
  16x64    one thread + row clamp
  64x64    all 8 threads, one tile        -> cross-thread staging (QV/KG/VV)
  64x128   two tiles                      -> online state / rescale
  64x192   three tiles + partial tail
  128x64   two slabs                      -> grid mapping (head/slab)
  128x128  two slabs, two tiles
  12x12    kvLen == qLen (no offset)
Prints fro/cos per case and a first-difference dump (head 0, rows 0-3,
first 8 dims) for the first failing case, plus ratio stats.
"""
from __future__ import annotations

import os
import sys

import torch

try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401,E402
except ImportError:
    from vllm import _xpu_ops  # noqa: E402

_so = os.environ.get("ARC_DENSE_FA_ESIMD_SO", "")
if _so:
    sys.path.insert(0, os.path.dirname(os.path.abspath(_so)))
import fa_dense_v2_ops  # noqa: E402

QH, KVH, HD = 12, 2, 256
SCALE = HD ** -0.5


def make_case(rows, kv_len, seed):
    g = torch.Generator(device="xpu").manual_seed(seed)
    q = torch.randn(rows, QH, HD, device="xpu", dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + 831) // 832
    pad = max(1, npages // 8)
    k = (torch.randn(npages + pad, 832, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(npages + pad, 832, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    ids = list(range(npages))
    gcpu = torch.Generator().manual_seed(seed * 7 + 1)
    perm = torch.randperm(len(ids), generator=gcpu).tolist()
    table = torch.tensor([ids[i] for i in perm] + [0] * pad,
                         dtype=torch.int32, device="xpu").unsqueeze(0)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
    seqused = torch.tensor([kv_len], dtype=torch.int32, device="xpu")
    kv = (torch.rand(1, device="xpu", generator=g) + 0.5).item()
    ks = torch.full((1, KVH), kv, device="xpu")
    vv = (torch.rand(1, device="xpu", generator=g) + 0.5).item()
    vs = torch.full((1, KVH), vv, device="xpu")
    return q, k, v, cu_q, seqused, table, ks, vs


def stock_ref(q, k, v, cu_q, seqused, table, ks, vs):
    dummy = torch.zeros_like(cu_q)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        SCALE, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


def main() -> int:
    torch.xpu.set_device("xpu:0")
    first_fail = None
    for rows, kv in [(8, 64), (16, 64), (12, 12), (64, 64), (64, 128),
                     (64, 192), (128, 64), (128, 128)]:
        q, k, v, cu_q, seqused, table, ks, vs = make_case(rows, kv, seed=11)
        ref = stock_ref(q, k, v, cu_q, seqused, table, ks, vs).float()
        out = torch.empty_like(q)
        fa_dense_v2_ops.fa_dense_varlen_fwd_v2(
            q, k, v, cu_q, seqused, table, out, ks, vs, SCALE)
        torch.xpu.synchronize()
        o = out.float()
        fro = ((o - ref).norm() / ref.norm().clamp_min(1e-9)).item()
        cos = torch.nn.functional.cosine_similarity(
            o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
        print(f"rows={rows:4d} kv={kv:4d} fro={fro:.3e} cos={cos:.7f} "
              f"{'PASS' if fro < 5e-3 else 'FAIL'}")
        if fro >= 5e-3 and first_fail is None:
            first_fail = (rows, kv, o, ref)
    if first_fail is not None:
        rows, kv, o, ref = first_fail
        print(f"\n--- first fail {rows}x{kv}: head 0, rows 0-3, dims 0-7 ---")
        for r in range(min(4, rows)):
            print(f"row {r} v2 : {o[r, 0, :8].tolist()}")
            print(f"row {r} ref: {ref[r, 0, :8].tolist()}")
        ratio = (o / ref.clamp_min(1e-6).sign() / ref.abs().clamp_min(1e-6))
        fin = torch.isfinite(ratio)
        if fin.any():
            print(f"out/ref ratio: median={ratio[fin].median().item():.4f} "
                  f"mean={ratio[fin].mean().item():.4f} "
                  f"min={ratio[fin].min().item():.4f} "
                  f"max={ratio[fin].max().item():.4f}")
        print(f"|out| mean={o.abs().mean().item():.4f} "
              f"|ref| mean={ref.abs().mean().item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
