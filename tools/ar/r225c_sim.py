#!/usr/bin/env python3
"""r225c_sim.py — nibble-level simulation of the r225 root-cause
candidate on the banked r224a payloads (CPU-only).

r225a factored the mxk8v4 native error to the e2m1 V leg's direct-bf16
fast path (ARC_MXKV_DIRECTBF16=0 collapses the error 8-25x; nsplit1 and
p64 change nothing; the e4m3 K leg has no direct path at all since the
r212m3 revert). Reading the assembly:

    e2m1_scaled_bf16_u32: bits = s | (base_e + ee) << 7 | m << 6

is exact for e2m1 NORMALS (value 2^(ee-1) x (1 + m/2)) but wrong for
the subnormal nibble (ee=0, m=1 -> value 0.5): the formula applies the
implicit leading 1 that subnormals do not have, giving 1.5 x 2^(k-1)
instead of 1.0 x 2^(k-1) — every V element quantized to +-0.5 decodes
1.5x too large on the fast path. The slow float path (e2m1_to_float:
sub = m * 0.5) is correct.

This script simulates that corruption in float64 — decode V, multiply
exactly the (ee=0, m=1)-nibble elements by 1.5, recompute attention —
and tests three signatures against the banked native output:

  1. dmax(O_sim, O_ref) ~= the banked native-vs-ref error
  2. dmax(O_sim, out_native) ~ the bf16-noise class (the simulation
     explains the native output)
  3. the affected-element fraction (deep layers should carry more
     +-0.5 nibbles -> the depth trend r224d measured)

Nibble identification works from the payload's own bytes (nib & 6 == 0
and nib & 1 -> ee == 0, m == 1), independent of the decode path.

Usage: r225c_sim.py <payload.pt> [payload2.pt ...]
"""

import sys

import torch

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv

BLOCK = 64
D = 256
GROUP = 6  # GQA: Hq=12 per worker / Hk=2


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def affected_mask(win):
    """(nb, 64, 2, D) bool: V elements whose e2m1 nibble is (ee=0,m=1).

    V bytes start at slot offset off["v"]; element e of head h lives in
    byte v_bytes[..., e>>1], nibble e&1 (element 2j = LOW nibble)."""
    off = _mx_off(D, "mxk8v4")
    vb = win[..., off["v"]:off["v"] + D // 2]     # (nb, 64, 2, D/2) u8
    nb_els = D // 2
    idx = torch.arange(nb_els, dtype=torch.long)
    nib_lo = (vb & 0xF).long()                    # element 2j
    nib_hi = (vb >> 4).long()                     # element 2j+1
    aff = torch.zeros(win.shape[0], win.shape[1], win.shape[2], D,
                      dtype=torch.bool)
    # ee = (nib >> 1) & 3 == 0 and m = nib & 1 == 1  ->  nib in {1, 9}
    is_aff = lambda n: (n == 1) | (n == 9)  # noqa: E731
    aff[..., 0::2] = is_aff(nib_lo)
    aff[..., 1::2] = is_aff(nib_hi)
    return aff


def attention(q, K, V, sl, scale):
    """float64 attention; K/V (nb, 64, 2, D) already decoded."""
    r = q.shape[0]
    o = torch.empty(r, q.shape[1], D, dtype=torch.float64)
    for i in range(r):
        nb_i = int((sl[i] + BLOCK - 1) // BLOCK)
        last = int(sl[i] - (nb_i - 1) * BLOCK)
        k_seq = torch.cat(
            [K[:nb_i - 1].reshape(-1, 2, D), K[nb_i - 1][:last]], dim=0)
        v_seq = torch.cat(
            [V[:nb_i - 1].reshape(-1, 2, D), V[nb_i - 1][:last]], dim=0)
        for h in range(q.shape[1]):
            kv_h = h // GROUP
            s = (q[i, h] @ k_seq[:, kv_h].T) * scale
            p_attn = torch.softmax(s, dim=-1)
            o[i, h] = p_attn @ v_seq[:, kv_h]
    return o


def sim_one(path: str) -> None:
    p = torch.load(path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    q = p["q"][:r].double()
    sl = p["sl"][:r].to(torch.long)
    win = p["kv_win"]
    scale = float(p["scale"])
    out_native = p["out_native"][:r]
    off = _mx_off(D, "mxk8v4")

    k_bytes = win[..., : off["v"]]
    v_bytes = win[..., off["v"]:]
    K, V = decode_kv(k_bytes, v_bytes, "mxk8v4", D)
    K, V = K.double(), V.double()

    aff = affected_mask(win)
    frac = float(aff.float().mean())

    o_ref = attention(q, K, V, sl, scale)
    V_cor = V.clone()
    V_cor[aff] *= 1.5
    o_sim = attention(q, K, V_cor, sl, scale)

    print(f"== {path}")
    print(f"   affected V nibbles (ee=0,m=1): {frac * 100:.1f}%")
    print(f"   banked native-vs-ref dmax     = "
          f"{dmax(out_native, o_ref):.3e}")
    print(f"   SIM    o_sim-vs-ref   dmax    = {dmax(o_sim, o_ref):.3e}")
    print(f"   SIM    o_sim-vs-native dmax   = "
          f"{dmax(o_sim, out_native):.3e}  <- noise class if THE bug")
    # the same simulation with the corruption REMOVED must be ~0
    o_chk = attention(q, K, V, sl, scale)
    print(f"   sanity o_ref reproducibility  = {dmax(o_chk, o_ref):.3e}")


def main() -> int:
    for path in sys.argv[1:]:
        try:
            sim_one(path)
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"== {path}: R225C_FAILED {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
