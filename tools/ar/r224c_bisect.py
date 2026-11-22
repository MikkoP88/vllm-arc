#!/usr/bin/env python3
"""r224c_bisect.py — offline GPU replay + factor bisection of an r224a
A/B payload (runs INSIDE an arc-image container with /dev/dri; the wheel
and the tree's Triton kernels are installed there).

Given mxkv_ab.*.pt payloads (real traffic: q rows, block table, seq
lens, the pool window bytes WITH their true shape/strides/storage-offset,
and BOTH outputs — the native kernel's and the certified Triton lane's),
this script, per payload:

  1. RECONSTRUCTS the exact pool tensor the engine handed the kernels:
     a fresh uint8 storage with the window bytes placed at
     storage[kv_off + block_id * stride0 ...], viewed with the recorded
     (shape, stride, storage_offset). Only bt-referenced blocks carry
     real bytes; the kernels never walk the rest.
  2. REPLAYS both decoders on the reconstruction:
       triton = mx_decode(q16, pool, bt, sl, scale)         -> vs out_triton
       native = mx_paged_decode(q.bf16, pool, fmt, ...)     -> vs out_native
     Agreement with the banked outputs validates the reconstruction
     before any conclusion is drawn from a mutation (and disagreement
     with BOTH is itself a finding: the payload is missing a factor).
  3. BISECTS one factor at a time — each variant reruns the NATIVE
     kernel against the TRITON replay (the oracle), both at the
     engine's true table width (4114, max_seqlen_k=263296):
       offset0   : same bytes, view storage_offset=0 — isolates the
                   per-layer 435.2 MB offsets of the hybrid pool views
                   (the one geometric fact an offline synthetic pool
                   never had)
       bf16floor : the Triton oracle itself run with bf16 q — the
                   fp16-vs-bf16 round-trip noise floor the certified
                   lane already accepts
     Payloads whose banked window truncates the walk (sl_max beyond
     banked blocks x 64) are refused — their oracle would read zeros.
  4. Prints the verdict matrix; the factor whose repair collapses
     dmax(native, oracle) to the noise floor is the root cause.
     Diagnostic tool, not a gate (exit 0 always; a payload that fails
     — e.g. a late-layer pool OOM on replay — reports and yields to
     the next).

Usage: r224c_bisect.py <payload.pt> [payload2.pt ...]
"""

import sys

import torch

from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import mx_decode
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

BLOCK_STRIDE = 51200  # 64 tokens * 2 kv-heads * 400 B slot (observed pool)
# The engine's essay-class calls: "MXKV native call ... max_seqlen_k=263296
# bt_width=4114" (the full-capacity table). Replaying at any other width
# changes the triton autotune key AND the C++ auto-split input — the
# first bisect attempt stalled 20+ min in a fresh autotune sweep on a
# 96-wide table. Content beyond ceil(sl/64) columns is never walked.
ENGINE_BT_WIDTH = 4114


def dmax(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def rebuild_pool(p, dev, off_override=None):
    """Reconstruct the pool view with recorded geometry (or offset 0)."""
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"]) if off_override is None else int(off_override)
    bt_head = p["bt_head"].to(torch.long)
    win = p["kv_win"].to(torch.uint8)
    storage = torch.zeros(off + shape[0] * BLOCK_STRIDE,
                          dtype=torch.uint8, device=dev)
    for i, b in enumerate(bt_head.tolist()):
        storage[off + b * BLOCK_STRIDE: off + (b + 1) * BLOCK_STRIDE] = \
            win[i].reshape(-1)
    return torch.as_strided(storage, shape, stride, off), storage


def bisect_one(path: str, dev) -> None:
    p = torch.load(path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]  # banked rows (payload caps q/outs at 8)
    q16 = p["q"].to(dev)
    sl = p["sl"][:r].to(dev)
    out_triton = p["out_triton"][:r].to(dev)
    out_native = p["out_native"][:r].to(dev)
    scale = float(p["scale"])
    fmt = p["fmt"]
    bt_dtype = p["bt"].dtype  # the engine's table dtype, preserved
    # every MTP-verify row of the banked group shares one request's
    # table; assert that before rebuilding rows from row 0 alone
    bt8 = p["bt"][:r]
    same = all(bool((bt8[i] == bt8[0]).all()) for i in range(r))
    bt_head = p["bt_head"].to(torch.long)
    nbt = int(bt_head.numel())
    if nbt * 64 < int(sl.max()):
        print(f"   !! WINDOW TRUNCATED: banked {nbt} blocks cover "
              f"{nbt * 64} tokens < sl_max {int(sl.max())} — oracle would "
              f"read zeros; skip this payload")
        return
    bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=bt_dtype, device=dev)
    bt[:, :nbt] = (bt_head.unsqueeze(0).repeat(r, 1)
                   .to(dev).to(bt_dtype))
    sl32 = sl.to(torch.int32)
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)

    print(f"== {path}")
    print(f"   fmt={fmt} banked_rows={r} scale={scale} "
          f"sl={sl[:4].tolist()} off={p['kv_off']} "
          f"shape={tuple(p['kv_shape'])} stride={tuple(p['kv_stride'])}")
    print(f"   bt: dtype={bt_dtype} head={bt_head[:6].tolist()}... "
          f"width={nbt} rows_identical={same}")
    print(f"   banked: dmax={p['dmax']:.3e} dmean={p['dmean']:.3e} "
          f"nan={p['nan_native']}/{p['nan_triton']}")

    pool, storage = rebuild_pool(p, dev)
    try:
        t_rep = mx_decode(q16, pool, bt, sl, scale)
        n_rep = mx_paged_decode(
            q16.to(torch.bfloat16), pool, fmt, cu_q, sl32,
            bt, max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale, num_splits_kv=0,
        )
        print(f"   REPLAY triton-vs-banked  dmax="
              f"{dmax(t_rep, out_triton):.3e}")
        print(f"   REPLAY native-vs-banked  dmax="
              f"{dmax(n_rep, out_native):.3e}")
        print(f"   REPLAY native-vs-oracle  dmax="
              f"{dmax(n_rep, t_rep):.3e}  (banked {p['dmax']:.3e})")

        # ---- factor bisection (one factor per variant) ----
        pool0, storage0 = rebuild_pool(p, dev, off_override=0)
        try:
            v = mx_paged_decode(
                q16.to(torch.bfloat16), pool0, fmt, cu_q, sl32,
                bt, max_seqlen_k=ENGINE_BT_WIDTH * 64,
                softmax_scale=scale, num_splits_kv=0,
            )
            print(f"   VARIANT offset0   native-vs-oracle dmax="
                  f"{dmax(v, t_rep):.3e}")
        finally:
            del pool0, storage0
            torch.xpu.empty_cache()
        try:
            t_bf = mx_decode(q16.to(torch.bfloat16), pool, bt, sl, scale)
            print(f"   VARIANT bf16floor triton(bf16)-vs-oracle dmax="
                  f"{dmax(t_bf, t_rep):.3e}  (noise floor)")
        except Exception as exc:  # noqa: BLE001 — dtype probe
            print(f"   VARIANT bf16floor rejected: {exc!r}")
    finally:
        del pool, storage
        torch.xpu.empty_cache()
    print("R224C_DONE")


def main() -> int:
    dev = torch.device("xpu")
    for path in sys.argv[1:]:
        try:
            bisect_one(path, dev)
        except Exception as exc:  # noqa: BLE001 — report, keep bisecting
            print(f"== {path}: R224C_PAYLOAD_FAILED {exc!r}")
            torch.xpu.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
