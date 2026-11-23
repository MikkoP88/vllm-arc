#!/usr/bin/env python3
"""r225i_synth.py — synthetic multi-token sweep of the post-fix
residual (runs INSIDE an arc-image container with /dev/dri; the r225
wheel installed).

Every prior numerical gate for this kernel ran at seq_len==1, where
softmax P == 1.0 is exactly representable in bf16 — the P-precision
machinery (ladder, reorder into PV-A fragments) was never numerically
exercised.  r225d-g/h exonerated: decode bit-assembly (fixed, base ==
dbf0 bitwise), group-scale mapping (no column structure), split
combine (nsplit1 identical), P rungs (base equidistant from
exact/P1/P2/P3), sequence extent (no rank-1 signature), output
rounding (265-4252 ulp vs triton's 0.25).

This script builds RANDOM-BAND synthetic windows (payload bytes drawn
so every decode band is populated: e4m3 without NaN patterns, e8m0
codes in [120,134], e2m1 nibbles uniform), replays the native op at a
seqlen sweep that crosses the machinery boundaries —

  16/17   : single-tile vs tile+1 (rescale/first-block path)
  64/65   : block boundary
  512+    : multi-block, deep rescale chains
  16384+  : auto num_splits > 1 (ReduceSplitK combine live)

— and scores native against the fp64 exact model ON THE SAME BYTES
(decode_kv ground truth).  The comparison runs at OUTPUT-GRID
resolution (the output is bf16): a bitwise match against
round_bf16(exact) at EVERY element means the kernel is exact-to-grid
(the Triton lane's 0.25-ulp class); the first seqlen where mismatches
appear localizes the machinery that injects the 2^-9-class residual.
P rungs (single/pair/triple bf16 P) are printed for context.

Geometry (pool shape/stride/offset, bt dtype, table width) is copied
from a banked r224a payload so the call matches the engine's.

Usage: r225i_synth.py <template_payload.pt> <out_json>
"""

import json
import sys

import torch

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

BLOCK = 64
D = 256
GROUP = 6
BLOCK_STRIDE = 51200
ENGINE_BT_WIDTH = 4114
SEQLENS = (1, 2, 15, 16, 17, 32, 33, 64, 65, 128, 512, 4096, 16384,
           65536)


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float64)


def model_attention(q, K, V, sl_val, scale, rungs):
    outs = {n: torch.empty(q.shape[0], q.shape[1], D,
                           dtype=torch.float64) for n in rungs}
    nb_i = (sl_val + BLOCK - 1) // BLOCK
    last = sl_val - (nb_i - 1) * BLOCK
    k_seq = torch.cat([K[:nb_i - 1].reshape(-1, 2, D), K[nb_i - 1][:last]],
                      dim=0)
    v_seq = torch.cat([V[:nb_i - 1].reshape(-1, 2, D), V[nb_i - 1][:last]],
                      dim=0)
    for h in range(q.shape[1]):
        kv_h = h // GROUP
        s = (q[0, h] @ k_seq[:, kv_h].T) * scale
        p_attn = torch.softmax(s.double(), dim=-1)
        for n in rungs:
            if n == 0:
                p_eff = p_attn
            else:
                p_eff = bf16(p_attn)
                rem = p_attn - p_eff
                for _ in range(n - 1):
                    term = bf16(rem)
                    p_eff = p_eff + term
                    rem = rem - term
            outs[n][0, h] = p_eff @ v_seq[:, kv_h]
    return outs


def synth_window(nb: int, seed: int, off):
    g = torch.Generator().manual_seed(seed)
    win = torch.zeros(nb, BLOCK, 2, 400, dtype=torch.uint8)
    # K e4m3 bytes: avoid the two NaN encodings (0x7F / 0xFF)
    kb = torch.randint(0, 256, (nb, BLOCK, 2, D), dtype=torch.uint8,
                       generator=g)
    kb[kb == 0x7F] = 0x7E
    kb[kb == 0xFF] = 0xFE
    win[..., : off["ks"]] = kb
    # V e2m1 packed nibbles: uniform bytes (all 16 nibble values live)
    win[..., off["v"]:off["v"] + D // 2] = torch.randint(
        0, 256, (nb, BLOCK, 2, D // 2), dtype=torch.uint8, generator=g)
    # Ks / Vs e8m0 codes in [120, 134] — inside the fast band, moderate
    for key, base in (("ks", off["ks"]), ("vs", off["vs"])):
        win[..., base:base + 8] = torch.randint(
            120, 135, (nb, BLOCK, 2, 8), dtype=torch.uint8, generator=g)
    return win


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def grid_stats(native_bf16, model_f64):
    """Bitwise match of the native bf16 output against round_bf16(model)
    plus the max ulp deviation (sub-grid resolution)."""
    m = model_f64.to(torch.bfloat16)
    mism = int((native_bf16.view(torch.int16)
                != m.view(torch.int16)).sum())
    ref = model_f64.abs()
    diff = (native_bf16.double() - model_f64).abs()
    big = ref > torch.finfo(torch.bfloat16).tiny
    if big.any():
        e = torch.floor(torch.log2(ref[big].clamp_min(1e-30)))
        ulp = torch.pow(2.0, e - 7)
        max_ulp = float((diff[big] / ulp).max())
    else:
        max_ulp = 0.0
    return mism, max_ulp


def main() -> int:
    template_path, out_json = sys.argv[1], sys.argv[2]
    tp = torch.load(template_path, map_location="cpu", weights_only=False)
    shape = tuple(tp["kv_shape"])
    stride = tuple(tp["kv_stride"])
    off_i = int(tp["kv_off"])
    bt_dtype = tp["bt"].dtype
    off = _mx_off(D, "mxk8v4")
    scale = 1.0 / (D ** 0.5)
    dev = torch.device("xpu")
    rungs = (0, 1, 2, 3)
    names = {0: "exact", 1: "P1", 2: "P2", 3: "P3"}

    results = []
    for sl_val in SEQLENS:
        nb = (sl_val + BLOCK - 1) // BLOCK
        win = synth_window(nb, seed=sl_val, off=off)
        # CPU ground truth on the SAME bytes
        K, V = decode_kv(win[..., : off["v"]], win[..., off["v"]:],
                         "mxk8v4", D)
        K, V = K.double(), V.double()
        q = torch.randn(1, 12, D, generator=torch.Generator().manual_seed(7)
                        ).to(torch.bfloat16)
        models = model_attention(q.double(), K, V, sl_val, scale, rungs)

        # GPU replay (engine geometry from the template)
        storage = torch.zeros(off_i + shape[0] * BLOCK_STRIDE,
                              dtype=torch.uint8, device=dev)
        for b in range(nb):
            storage[off_i + b * BLOCK_STRIDE:
                    off_i + (b + 1) * BLOCK_STRIDE] = \
                win[b].reshape(-1).to(dev)
        pool = torch.as_strided(storage, shape, stride, off_i)
        bt = torch.zeros(1, ENGINE_BT_WIDTH, dtype=bt_dtype, device=dev)
        bt[0, :nb] = torch.arange(nb, dtype=bt_dtype, device=dev)
        sl32 = torch.tensor([sl_val], dtype=torch.int32, device=dev)
        cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        n_out = mx_paged_decode(
            q.to(dev), pool, "mxk8v4", cu_q, sl32, bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale, num_splits_kv=0,
        )[:1].cpu()

        mism, max_ulp = grid_stats(n_out[0], models[0][0])
        row = {"sl": sl_val, "mism": mism, "max_ulp": max_ulp}
        for n in rungs:
            row[names[n]] = dmax(n_out, models[n].to(torch.bfloat16))
        results.append(row)
        print(f"sl={sl_val:>6}  mism={mism:>5}/{12 * D}  "
              f"max_ulp={max_ulp:7.2f}  " + "  ".join(
                  f"{names[n]}={row[names[n]]:.3e}" for n in rungs))
        del pool, storage
        torch.xpu.empty_cache()

    with open(out_json, "w") as f:
        json.dump(results, f, indent=1)
    print(f"R225I_SYNTH_DONE -> {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
