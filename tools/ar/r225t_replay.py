#!/usr/bin/env python3
"""r225t_replay.py — q-cast-matched validation cell (r225 final gate).

r225q+r225r closed the "fourth source": the build24+ native kernel is
byte-identical across every harness state (md5 51736f834358 on
2447.30) and sits at the fp64 floor for its OWN input (dmax 1.556e-2,
mism 1/3072, max_ulp 0.54). The gate3/gate4 FAILs measured it against
a Triton oracle that consumes a DIFFERENT input: the banked q is
torch.float16, the oracle eats fp16 q verbatim while the native op
eats q.to(bfloat16) — and the fp64 walk of the cast difference alone
reproduces the entire gate ladder (r225r: 2.316e-3 / 4.939e-2 /
3.327e-3 vs observed 7.812e-3 / 5.078e-2 / 5.859e-3).

This cell applies the corrected criterion — each lane vs the fp64
walk of its own input — plus a q-matched cross-lane check:

  t_rep : mx_decode(q16, ...)          fp16 q (engine lane) — must be
                                       BIT-EXACT vs the banked engine
                                       triton output (r224c law)
  qBf   : q16.to(bf16).to(fp16)        the native's bf16 values in an
                                       fp16 container (exact: bf16
                                       mantissa fits fp16; same
                                       triton autotune key — no sweep)
  t_B   : mx_decode(qBf, ...)          q-matched oracle
  n     : mx_paged_decode(qB, ...)     native at <factor> splits

PASS = triton bit-exact AND native bitwise-stable in-process AND
native-vs-walkBF16 at the floor band (dmax <= 2.5e-2, mism <= 16/3072,
max_ulp <= 16.0 with the pre-registered tie-outlier allowance — floor
0.54-1.51 ulp, tie tail 7.51/10.51 at 1/3072) AND t_B-vs-walkBF16 at
the triton band (dmax <= 1e-2) AND native-vs-t_B <= 5e-2 AND no NaN.

Usage: r225t_replay.py <base|nsplit1|nsplit2|nsplit8> <payload.pt>
"""

import os
import sys

import torch

sys.path.insert(0, "/tmp")
from r225j_iso import dmax, grid_stats, model_row  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import mx_decode  # noqa: E402
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

BLOCK_STRIDE = 51200  # 64 tokens * 2 kv-heads * 400 B slot (observed pool)
ENGINE_BT_WIDTH = 4114
D = 256
FACTORS = {"base": 0, "nsplit1": 1, "nsplit2": 2, "nsplit8": 8}

BAND_DMAX = 2.5e-2  # native-vs-walk floor band (floor: 1.556e-2)
BAND_MISM = 16  # grid points off (floor: 1/3072)
# max-ulp carries the PRE-REGISTERED tie-outlier allowance (banked
# VERDICT gate section, written before this gate ran from build23's
# r225l measurements): fp32-vs-fp64 near-tie elements flip up to
# 10.51 ulp at mism 1/3072 (frac_gt2 = 3.3e-4 <= 1e-3) on 2447.18 at
# S=1/S=2. Allowance: max_ulp <= 16 AND mism <= 16 — any broader
# signature (many elements, dmax above band) still fails.
BAND_ULP = 16.0  # floor 0.54-1.51; tie tail 7.51/10.51 at 1/3072
BAND_TRITON = 1e-2  # t_B-vs-walk band (r224 triton floor: ~5e-3)
BAND_CROSS = 5e-2  # native-vs-t_B (both on the native's input)


def rebuild_pool(p, dev):
    """Reconstruct the pool view with the recorded geometry (r225a law)."""
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"])
    bt_head = p["bt_head"].to(torch.long)
    win = p["kv_win"].to(torch.uint8)
    storage = torch.zeros(off + shape[0] * BLOCK_STRIDE, dtype=torch.uint8, device=dev)
    for i, b in enumerate(bt_head.tolist()):
        storage[off + b * BLOCK_STRIDE : off + (b + 1) * BLOCK_STRIDE] = win[i].reshape(-1)
    return torch.as_strided(storage, shape, stride, off), storage


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in FACTORS:
        print(__doc__)
        print(f"usage: {sys.argv[0]} <{'|'.join(FACTORS)}> <payload.pt>")
        return 2
    factor, path = sys.argv[1], sys.argv[2]
    dev = torch.device("xpu")
    off = _mx_off(D, "mxk8v4")

    p = torch.load(path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    q16 = p["q"].to(dev)
    sl = p["sl"][:r].to(dev)
    out_triton = p["out_triton"][:r].to(dev)
    scale = float(p["scale"])
    fmt = p["fmt"]
    bt_dtype = p["bt"].dtype
    bt_head = p["bt_head"].to(torch.long)
    nbt = int(bt_head.numel())
    bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=bt_dtype, device=dev)
    bt[:, :nbt] = bt_head.unsqueeze(0).repeat(r, 1).to(dev).to(bt_dtype)
    sl32 = sl.to(torch.int32)
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)
    qB = q16.to(torch.bfloat16)
    qBf = qB.to(torch.float16)

    name = os.path.basename(path)
    print(f"== {name} [{factor}] fmt={fmt!r} q={p['q'].dtype} sl={sl[:4].tolist()}")
    pool, storage = rebuild_pool(p, dev)
    try:
        t_rep = mx_decode(q16, pool, bt, sl, scale)
        bitexact = dmax(t_rep, out_triton)
        t_B = mx_decode(qBf, pool, bt, sl, scale)
        n1 = mx_paged_decode(
            qB,
            pool,
            fmt,
            cu_q,
            sl32,
            bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale,
            num_splits_kv=FACTORS[factor],
        )[:r].cpu()
        n2 = mx_paged_decode(
            qB,
            pool,
            fmt,
            cu_q,
            sl32,
            bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale,
            num_splits_kv=FACTORS[factor],
        )[:r].cpu()
    finally:
        del pool, storage
        torch.xpu.empty_cache()

    stable = all(torch.equal(n1[i], n2[i]) for i in range(r))
    nan_ct = int(torch.isnan(n1).sum())

    K, V = decode_kv(p["kv_win"][..., : off["v"]], p["kv_win"][..., off["v"] :], "mxk8v4", D)
    K, V = K.double(), V.double()
    walk = torch.stack([model_row(qB[i].double().cpu(), K, V, int(p["sl"][i]), scale) for i in range(r)])
    tB_cpu = t_B[:r].cpu()

    worst = (0, 0.0)
    dm_n = 0.0
    dm_t = 0.0
    for i in range(r):
        mism, mul = grid_stats(n1[i], walk[i])
        worst = (max(worst[0], mism), max(worst[1], mul))
        dm_n = max(dm_n, dmax(n1[i], walk[i]))
        dm_t = max(dm_t, dmax(tB_cpu[i], walk[i]))
    dm_cross = max(dmax(n1[i], tB_cpu[i].to(n1.dtype)) for i in range(r))

    ok = (
        bitexact == 0.0
        and stable
        and nan_ct == 0
        and dm_n <= BAND_DMAX
        and worst[0] <= BAND_MISM
        and worst[1] <= BAND_ULP
        and dm_t <= BAND_TRITON
        and dm_cross <= BAND_CROSS
    )
    print(f"   TRITON_BITEXACT dmax={bitexact:.3e}")
    print(f"   NATIVE_STABLE bitwise={stable} nan={nan_ct}")
    print(f"   NATIVE_VS_WALK dmax={dm_n:.3e} mism={worst[0]}/{12 * D} max_ulp={worst[1]:.2f}")
    print(f"   QBORACLE_VS_WALK dmax={dm_t:.3e}")
    print(f"   NATIVE_VS_QMATCHED_ORACLE dmax={dm_cross:.3e}")
    print(f"R225T_CELL {factor} {name} {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
