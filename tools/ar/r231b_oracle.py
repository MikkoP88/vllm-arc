#!/usr/bin/env python3
"""r231b_oracle.py -- build25-vs-build27 bitwise oracle for the r231b
Q pre-stage (behavior-preservation gate: the SLM-staged A operand must
not change a single output bit), plus output banking.

Replay contract mirrors the r229 scorer cell exactly (banked bt_head
geometry, qB = q.to(bf16), max_seqlen_k = 4114*64, the payload's own
fmt field wins -- the payload-fmt law).

Modes:
  save <outdir> <payload.pt> [payload2.pt ...]
        one native call per payload; saves [:r] cpu output as
        <outdir>/<basename> (no .pt suffix games: basename kept)
  cmp <dirA> <dirB>
        bitwise compare every same-named pair; PASS only if every
        tensor is torch.equal
"""

import os
import sys

import torch

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

D = 256
ENGINE_BT_WIDTH = 4114


def rebuild_pool(p, dev, block_stride):
    """Reconstruct the pool view at the banked geometry (r225a law)."""
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"])
    bt_head = p["bt_head"].to(torch.long)
    win = p["kv_win"].to(torch.uint8)
    storage = torch.zeros(
        off + shape[0] * block_stride, dtype=torch.uint8, device=dev)
    for i, b in enumerate(bt_head.tolist()):
        storage[
            off + b * block_stride: off + (b + 1) * block_stride
        ] = win[i].reshape(-1)
    return torch.as_strided(storage, shape, stride, off), storage


def native_out(path, dev):
    p = torch.load(path, map_location="cpu", weights_only=False)
    fmt = str(p["fmt"])  # payload fmt wins (r231a5 law)
    r = int(p["q"].shape[0])
    block_stride = int(p["kv_stride"][0])
    qB = p["q"].to(torch.bfloat16).to(dev)
    sl = p["sl"][:r].to(torch.int32).to(dev)
    scale = float(p["scale"])
    bt_dtype = p["bt"].dtype
    bt_head = p["bt_head"].to(torch.long)
    nbt = int(bt_head.numel())
    bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=bt_dtype, device=dev)
    bt[:, :nbt] = (
        bt_head.unsqueeze(0).repeat(r, 1).to(dev).to(bt_dtype))
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)
    pool, storage = rebuild_pool(p, dev, block_stride)
    try:
        out = mx_paged_decode(
            qB, pool, fmt, cu_q, sl, bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale,
        )[:r].cpu()
    finally:
        del pool, storage
        torch.xpu.empty_cache()
    return out


def save(outdir, paths):
    os.makedirs(outdir, exist_ok=True)
    for path in paths:
        out = native_out(path, torch.device("xpu"))
        name = os.path.basename(path)
        torch.save(out, os.path.join(outdir, name))
        print(f"R231B_SAVED {name} shape={tuple(out.shape)}")


def cmp(dira, dirb):
    names = sorted(
        n for n in os.listdir(dira) if not n.startswith("."))
    if not names:
        print("R231B_CMP FAIL (no saved outputs)")
        return 1
    bad = 0
    for n in names:
        pa = os.path.join(dira, n)
        pb = os.path.join(dirb, n)
        if not os.path.exists(pb):
            print(f"R231B_PAIR {n} MISSING_B")
            bad += 1
            continue
        a = torch.load(pa, map_location="cpu", weights_only=False)
        b = torch.load(pb, map_location="cpu", weights_only=False)
        if torch.equal(a, b):
            print(f"R231B_PAIR {n} BITWISE_EQUAL")
        else:
            d = (a.float() - b.float()).abs()
            n_mism = int((a != b).sum())
            print(
                f"R231B_PAIR {n} DIFF n={n_mism}/{a.numel()} "
                f"dmax={float(d.max()):.3e}")
            bad += 1
    print(f"R231B_ORACLE {'PASS' if bad == 0 else 'FAIL'} "
          f"pairs={len(names)} bad={bad}")
    return 0 if bad == 0 else 1


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    if sys.argv[1] == "save":
        save(sys.argv[2], sys.argv[3:])
        return 0
    if sys.argv[1] == "cmp":
        return cmp(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
