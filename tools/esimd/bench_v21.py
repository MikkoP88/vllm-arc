#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P6.21 v2.2 ablation bench — where do the U32k ms go?

Level = ablate & 7 (pipeline depth; numerics valid only at 4):
  0 staging only (KV loads + dequant + SLM stores + barriers + epilogue)
  1 + acc RMW SLM traffic
  2 + QK dpas chains (+ scale/mask)
  3 + softmax (folds, rescale, exp, sums)
  4 + PV dpas (== full, must match the perf-window numbers)
Variant bits (compose with level 0 — r92_v22 law: the bits used to
satisfy the old >= guards and silently ran the full pipeline):
  +8   zero dequant results (dequant proven free at r92_v23)
  +32  skip V SLM stores
  +64  skip K SLM stores (loads+dequant still run)
  +128 skip the 8 global loads (load-issue/latency share of the floor)
  +96/+224 = combos: loads+dequant only / pure barriers+q+epilogue
"""
from __future__ import annotations

import json
import os
import statistics
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
PAGE = 832


def make_case(rows, kv_len, seed):
    g = torch.Generator(device="xpu").manual_seed(seed)
    q = torch.randn(rows, QH, HD, device="xpu", dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + PAGE - 1) // PAGE
    pad = max(1, npages // 8)
    k = (torch.randn(npages + pad, PAGE, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(npages + pad, PAGE, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    table = torch.arange(npages, dtype=torch.int32,
                         device="xpu").unsqueeze(0)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
    seqused = torch.tensor([kv_len], dtype=torch.int32, device="xpu")
    ks = torch.full((1, KVH), 1.3, device="xpu")
    vs = torch.full((1, KVH), 1.7, device="xpu")
    return q, k, v, cu_q, seqused, table, ks, vs


def timed(fn, iters=7, warmup=2):
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


def main() -> int:
    torch.xpu.set_device("xpu:0")
    # numerics sanity FIRST (level 4 must stay correct in every build
    # whose perf numbers we read): fro vs stock on a mid case.
    q, k, v, cu_q, seqused, table, ks, vs = make_case(512, 1664, seed=7)
    out = torch.empty_like(q)
    fa_dense_v2_ops.fa_dense_varlen_fwd_v2(
        q, k, v, cu_q, seqused, table, out, ks, vs, SCALE, 4)
    torch.xpu.synchronize()
    dummy = torch.zeros_like(cu_q)
    ref, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        512, 1664, 0.0, ks, vs, SCALE, None, False, True, -1, -1, 0.0,
        False, None, 0, False, None, None)
    fro = ((out.float() - ref.float()).norm() /
           ref.float().norm()).item()
    print(f"sanity fro(level4 vs stock, 512x1664): {fro:.2e} "
          f"({'OK' if fro < 8e-3 else 'BROKEN — do not trust rates'})")
    del q, k, v, out, ref
    torch.xpu.empty_cache()

    rows, kv = 8192, 32768
    q, k, v, cu_q, seqused, table, ks, vs = make_case(rows, kv, seed=41)
    out = torch.empty_like(q)
    names = {0: "staging", 1: "+accRMW", 2: "+QK", 3: "+softmax",
             4: "+PV(full)",
             128: "L0 noLoads", 96: "L0 noKVstore",
             224: "L0 noLoad+noStore"}
    levels = [4, 0, 1, 2, 3, 224, 128, 96]
    # thermally fair: 3 interleaved rounds, min per level (the GPU
    # downclocks progressively within a window — a plain ladder
    # attributes the drift to whatever runs last).
    best = {}
    for rnd in range(3):
        for lvl in levels:
            t = timed(lambda: fa_dense_v2_ops.fa_dense_varlen_fwd_v2(
                q, k, v, cu_q, seqused, table, out, ks, vs, SCALE, lvl),
                iters=5, warmup=1)
            best[lvl] = min(best.get(lvl, 1e9), t)
        print(f"round {rnd}: " +
              " ".join(f"L{l}={best[l]:.1f}" for l in levels))
    res = [{"level": l, "name": names[l], "ms_min": round(best[l], 2)}
           for l in levels]
    for r in res:
        print(f"L{r['level']:<2d} {r['name']:20s}: {r['ms_min']:8.2f} ms")
    with open("/work/bench_v21.json", "w") as f:
        json.dump({"sanity_fro": round(fro, 6), "rows": rows, "kv": kv,
                   "levels": res}, f, indent=1)
    print("wrote /work/bench_v21.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
