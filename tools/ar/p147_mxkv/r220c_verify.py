#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""r220c_verify.py -- arc7 Phase-8 r220c: VALUE verifier for the
ARC_MXKV_P64NATIVE multi-SG route.

The r220a finding (2026-11-22): p204's native "canary" is graph-vs-
eager of the SAME kernel -- self-referential. A deterministic value
bug corrupts both sides identically and canaries at exactly 0.0; it
only surfaces nondeterminism or unwritten rows differing between two
empty buffers. The 1K "clean" cells therefore proved nothing about
the multi-SG band laws, and the observed NaNs are reinterpreted as
candidates for unwritten outputs or races.

This verifier closes the gap with four eager diffs per format:
  d_16_64 : |native _64 - native _16|max   -- the true policy A/A
            (the _16 route on this wheel is the certified path)
  d_det   : |_64 run1 - _64 run2|max       -- determinism/race check
  d_64_v2 : |_64 - triton v2|max           -- cross-impl band
  d_16_v2 : |_16 - triton v2|max           -- the historical band
            (certifies the cell itself is comparable)
All native outputs are NaN-prefilled: nan16/nan64 count NaNs AFTER
the call -- nonzero = unwritten rows (or written-NaN; rowcounts vs
the request count distinguish coverage from spots).

Cell construction is byte-identical to p204_graphtime.py's (same
seeds, same builders, same env geometry knobs -- see r220a_val.sh:
D=256 HREP=6 BLOCK=64 MAXSK=cap, no BTWIDTH pad). Usage:
  ARC_MXKV_DIRECTBF16=1 python r220c_verify.py OUT_DIR
Exit 0 always -- the JSON is the datum; the gate lives in the
wrapping r<leg> script.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import torch

_MXKV_DIR = (
    Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
)

D = int(os.environ.get("P204_D", "256"))
HK = 2
HREP = int(os.environ.get("P204_HREP", "6"))
HQ = HK * HREP
BLOCK = int(os.environ.get("P204_BLOCK", "64"))
DEV = "xpu"
FMTS = tuple(
    f
    for f in os.environ.get("P204_FMTS", "mxfp8,mxfp4,mxk8v4").split(",")
    if f.strip()
)
CTX = int(os.environ.get("P204_CTX", "1024"))
SCALE = 1.0 / math.sqrt(D)


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
mx_store = mxkv.mx_store
mx_decode = mxkv.mx_decode
slot_bytes = mxkv.slot_bytes

from vllm_xpu_kernels.mx_attn_interface import _FORMATS  # noqa: E402


def build_paged(total_tokens, seed=7):
    n_blocks = total_tokens // BLOCK + 2
    perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed))
    order = perm.tolist()
    lens = (
        [total_tokens]
        if total_tokens < 2
        else [max(1, total_tokens // 3), total_tokens - max(1, total_tokens // 3)]
    )
    tables, tok_block, tok_pos = [], [], []
    cur = 0
    for ln in lens:
        ntb = (ln + BLOCK - 1) // BLOCK
        tab = order[cur : cur + ntb]
        tables.append(torch.tensor(tab, dtype=torch.int32, device=DEV))
        for t in range(ln):
            tok_block.append(tab[t // BLOCK])
            tok_pos.append(t % BLOCK)
        cur += ntb
    return (
        lens,
        tables,
        torch.tensor(tok_block, dtype=torch.int32, device=DEV),
        torch.tensor(tok_pos, dtype=torch.int32, device=DEV),
        n_blocks,
    )


def flat_tables(lens, tables):
    mb = max(t.shape[0] for t in tables)
    bt = torch.zeros(len(lens), mb, dtype=torch.int32, device=DEV)
    for i, t in enumerate(tables):
        bt[i, : t.shape[0]] = t
    return bt


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"r220c_verify -- device={DEV}, ctx={CTX}, D={D}, HQ={HQ}, "
        f"HK={HK}, block={BLOCK}, torch={torch.__version__}"
    )
    op = torch.ops._vllm_fa2_C.mx_paged_decode
    R = {}
    for fmt in FMTS:
        gen = torch.Generator().manual_seed(CTX)
        xk = torch.randn(CTX, HK, D, generator=gen).to(DEV)
        xv = torch.randn(CTX, HK, D, generator=gen).to(DEV)
        lens, tables, tb, tp, n_blocks = build_paged(CTX, seed=CTX % 97)
        q = (
            torch.randn(len(lens), HQ, D, generator=gen)
            .to(torch.bfloat16)
            .to(DEV)
        )
        qf = q.float()
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        bt = flat_tables(lens, tables)
        torch.xpu.synchronize()
        cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
        sl_dev = torch.tensor(lens, dtype=torch.int32, device=DEV)
        fmt_id = _FORMATS[fmt]
        # r220a cells ran MAXSK=cap with the table at its real width
        # (no BTWIDTH pad): mk = mb * BLOCK.
        mk = bt.shape[1] * BLOCK

        def run_native(q=q, cache=cache, fmt_id=fmt_id, cu_q=cu_q,
                       sl_dev=sl_dev, bt=bt, mk=mk, op=op):
            o = torch.empty_like(q)
            o.fill_(float("nan"))
            op(q, cache, fmt_id, cu_q, sl_dev, bt, 1, mk,
               SCALE, -1, None, None, o)
            return o

        out16 = run_native()  # env route unset = the certified _16 path
        os.environ["ARC_MXKV_P64NATIVE"] = "1"
        try:
            out64a = run_native()
            out64b = run_native()
        finally:
            del os.environ["ARC_MXKV_P64NATIVE"]
        v2 = mx_decode(qf, cache, bt, sl_dev, SCALE, block_size=BLOCK)
        torch.xpu.synchronize()

        def dmax(a, b):
            return (a.float() - b.float()).abs().max().item()

        f16, f64a, f64b, fv2 = (t.float().cpu() for t in (out16, out64a, out64b, v2))
        cell = {
            "d_16_64": round(dmax(f64a, f16), 6),
            "d_det": round(dmax(f64a, f64b), 6),
            "d_64_v2": round(dmax(f64a, fv2), 6),
            "d_16_v2": round(dmax(f16, fv2), 6),
            "nan16": int(torch.isnan(f16).sum()),
            "nan64": int(torch.isnan(f64a).sum()),
            "rows": int(f16.numel()),
        }
        # r220c2 pattern probe: WHICH half is NaN / where _64 deviates.
        # Shape (nreq, HQ, D): per-seq, per-head, and d-half counts.
        n64 = torch.isnan(f64a)
        if n64.any():
            per_seq = n64.flatten(1).sum(1).tolist()
            per_head = n64.sum((0, 2)).tolist()
            per_dh = [int(n64[..., : D // 2].sum()), int(n64[..., D // 2 :].sum())]
            cell["nan_pat"] = {
                "per_seq": per_seq,
                "per_head": per_head,
                "d_lo_hi": per_dh,
            }
        dev = (f64a - f16).abs() > 2e-3
        if dev.any() and (~torch.isnan(f16)).all():
            cell["dev_pat"] = {
                "per_seq": dev.flatten(1).sum(1).tolist(),
                "per_head": dev.sum((0, 2)).tolist(),
                "d_lo_hi": [int(dev[..., : D // 2].sum()), int(dev[..., D // 2 :].sum())],
            }
        R[fmt] = cell
        print(f"  {fmt}: {json.dumps(cell)}")
    path = out / f"r220c_verify_{CTX}.json"
    with open(path, "w", newline="\n") as f:
        json.dump(R, f, indent=1)
    print(f"P220C_DONE {path}")


if __name__ == "__main__":
    main()
