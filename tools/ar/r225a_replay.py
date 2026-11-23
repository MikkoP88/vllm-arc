#!/usr/bin/env python3
"""r225a_replay.py — r224c-class GPU replay with a FACTOR MATRIX on the
banked r224a payloads (runs INSIDE an arc-image container with /dev/dri;
the wheel and the tree's Triton kernels are installed there).

r224 established: the mxk8v4 native output is wrong by 156-245x vs the
float64 ground truth while the Triton lane sits at 4.8e-4..3.6e-3, the
divergence is deterministic and magnitude-dependent, and pool geometry
is exonerated (bit-exact replay of both kernels). Static reading of the
wheel source (r225) checked every format-specific site — host slot
offsets, SLM staging scatter, scale-line walks, reduce guards — and
each individual dequant leg is e2e-certified in its SYMMETRIC format
(mxfp8: e4m3 K + e4m3 V; mxfp4: e2m1 K + e2m1 V). mxk8v4 is the only
MIXED instantiation (e4m3 K + e2m1 V), and the remaining suspects are
runtime-selectable. This script replays ONE factor per invocation so
each cell is a clean measurement:

  base   : shipping config — direct_bf16 decode ON, C++ auto splits,
           certified _16 sub-tile frame (re-baseline; must reproduce
           the banked native output bit-exactly, r224c law)
  dbf0   : caller sets ARC_MXKV_DIRECTBF16=0 — float-assembly decode on
           both legs (isolates the direct-bf16 exponent fold)
  nsplit1: num_splits_kv=1 — single split, no ReduceSplitK combine, the
           epilogue normalizes in-kernel (isolates the split/combine
           machinery + stale-guard interplay)
  p64    : caller sets ARC_MXKV_P64NATIVE=1 — the r220a multi-SG _64
           policy frame (isolates the _16 sub-tile frame)

Per payload: rebuild the exact pool view from the banked window bytes
(storage[kv_off + block * 51200], recorded shape/strides/offset), replay
the Triton oracle and the native kernel under the factor at the engine's
true table width (4114 / max_seqlen_k=263296 — the autotune-key law),
report dmax vs banked outputs, and SAVE the factor's native output for
float64 arbitration (r225b_arbitrate.py). The factor whose repair
collapses dmax(native, oracle) to the Triton lane's error level is the
root cause.

Usage: r225a_replay.py <base|dbf0|nsplit1|p64> <out.pt> <payload.pt> [...]
"""

import os
import sys

import torch

from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import mx_decode
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

BLOCK_STRIDE = 51200  # 64 tokens * 2 kv-heads * 400 B slot (observed pool)
# The engine's essay-class calls run the full-capacity table; any other
# width changes the triton autotune key and the C++ auto-split input
# (r224c: the first bisect attempt stalled 20+ min in a fresh sweep).
ENGINE_BT_WIDTH = 4114
FACTORS = ("base", "dbf0", "nsplit1", "p64")


def dmax(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def rebuild_pool(p, dev):
    """Reconstruct the pool view with the recorded geometry (r224c law)."""
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"])
    bt_head = p["bt_head"].to(torch.long)
    win = p["kv_win"].to(torch.uint8)
    storage = torch.zeros(off + shape[0] * BLOCK_STRIDE,
                          dtype=torch.uint8, device=dev)
    for i, b in enumerate(bt_head.tolist()):
        storage[off + b * BLOCK_STRIDE: off + (b + 1) * BLOCK_STRIDE] = \
            win[i].reshape(-1)
    return torch.as_strided(storage, shape, stride, off), storage


def replay_one(path: str, dev, factor: str):
    p = torch.load(path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]  # banked rows (payload caps q/outs at 8)
    q16 = p["q"].to(dev)
    sl = p["sl"][:r].to(dev)
    out_triton = p["out_triton"][:r].to(dev)
    out_native = p["out_native"][:r].to(dev)
    scale = float(p["scale"])
    fmt = p["fmt"]
    bt_dtype = p["bt"].dtype  # the engine's table dtype, preserved
    bt_head = p["bt_head"].to(torch.long)
    nbt = int(bt_head.numel())
    if nbt * 64 < int(sl.max()):
        print(f"   !! WINDOW TRUNCATED: banked {nbt} blocks cover "
              f"{nbt * 64} tokens < sl_max {int(sl.max())} — skip")
        return None
    bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=bt_dtype, device=dev)
    bt[:, :nbt] = (bt_head.unsqueeze(0).repeat(r, 1)
                   .to(dev).to(bt_dtype))
    sl32 = sl.to(torch.int32)
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)
    nsplit = 1 if factor == "nsplit1" else 0

    print(f"== {os.path.basename(path)} [{factor}] sl={sl[:4].tolist()}")
    pool, storage = rebuild_pool(p, dev)
    try:
        t_rep = mx_decode(q16, pool, bt, sl, scale)
        n_rep = mx_paged_decode(
            q16.to(torch.bfloat16), pool, fmt, cu_q, sl32,
            bt, max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale, num_splits_kv=nsplit,
        )
        print(f"   REPLAY triton-vs-banked       dmax="
              f"{dmax(t_rep, out_triton):.3e}")
        print(f"   REPLAY native[{factor}]-vs-banked dmax="
              f"{dmax(n_rep, out_native):.3e}")
        print(f"   REPLAY native[{factor}]-vs-oracle dmax="
              f"{dmax(n_rep, t_rep):.3e}  (banked native-vs-triton "
              f"{p['dmax']:.3e})")
        return n_rep[:r].cpu()
    finally:
        del pool, storage
        torch.xpu.empty_cache()


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] not in FACTORS:
        print(__doc__)
        print(f"usage: {sys.argv[0]} <{'|'.join(FACTORS)}> <out.pt> "
              f"<payload.pt> [...]")
        return 2
    factor, out_path, payloads = sys.argv[1], sys.argv[2], sys.argv[3:]
    dev = torch.device("xpu")
    saved = {}
    for path in payloads:
        try:
            out = replay_one(path, dev, factor)
            if out is not None:
                saved[os.path.basename(path)] = out
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"== {path}: R225A_FAILED {exc!r}")
            torch.xpu.empty_cache()
    torch.save(saved, out_path)
    print(f"R225A_DONE factor={factor} saved={len(saved)} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
