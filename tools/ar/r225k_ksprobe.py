#!/usr/bin/env python3
"""r225k_ksprobe.py — CPU probe of the Ks-carrier finding (r225j):
real Ks scale bytes on a synthetic window reproduce the residual
(40% element mismatches); synthetic codes [120,134] are exact.

1. Histogram the REAL Ks/Vs e8m0 code values per payload (what band
   do deep-layer codes actually live in?).
2. Unit-check the wheel's float decode laws against the overlay's
   decode_kv for EVERY byte value (e4m3: all 256; e2m1: all 16
   nibbles) at a sweep of e8m0 codes — the r212m3-reverted K lane has
   no direct-bf16, so its correctness rests on e4m3_to_float and the
   scale multiply; subnormal payload bytes were 1/16 of the synthetic
   distribution but dominate real deep-layer bytes.

Usage: r225k_ksprobe.py <payload.pt> [payload2.pt ...]
"""

import sys

import torch

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv

D = 256


def e4m3_to_float_ref(b: int) -> float:
    """OCP e4m3 (ee=15&m=7 is NaN; no inf)."""
    s = -1.0 if b & 0x80 else 1.0
    ee = (b >> 3) & 0xF
    m = b & 7
    if ee == 0:
        val = m * 2.0 ** -9      # subnormal: 2^(1-7) * m/8
    elif ee == 15 and m == 7:
        return float("nan")
    else:
        val = 2.0 ** (ee - 7) * (1 + m / 8)
    return s * val


def e2m1_to_float_ref(nib: int) -> float:
    s = -1.0 if nib & 8 else 1.0
    ee = (nib >> 1) & 3
    m = nib & 1
    if ee == 0:
        return s * m * 0.5
    return s * 2.0 ** (ee - 1) * (1 + 0.5 * m)


def e8m0_to_scale_ref(code: int) -> float:
    return 2.0 ** (code - 127)


def main() -> int:
    off = _mx_off(D, "mxk8v4")

    # ---- 1: real code histograms ----
    for path in sys.argv[1:]:
        p = torch.load(path, map_location="cpu", weights_only=False)
        win = p["kv_win"].to(torch.uint8)
        for name, base, width in (("Ks", off["ks"], 8), ("Vs", off["vs"], 8)):
            codes = win[..., base:base + width].reshape(-1)
            hist = torch.bincount(codes.long(), minlength=256)
            nz = hist.nonzero().flatten()
            lo = int(nz.min())
            hi = int(nz.max())
            med = float(codes.float().median())
            n_under = int((codes < 120).sum())
            n_over = int((codes > 134).sum())
            print(f"== {path.split('/')[-1]} {name}: codes "
                  f"[{lo}..{hi}] median={med:.0f} "
                  f"(<120: {n_under}  >134: {n_over}  "
                  f"total {codes.numel()})")
            # the 8 most common codes
            top = torch.topk(hist, 8)
            print("   top codes: " + " ".join(
                f"{int(c)}x{int(n)}" for c, n in
                zip(top.indices.tolist(), top.values.tolist()) if n))

    # ---- 2: decode-law unit check over all bytes x code sweep ----
    print("== decode law unit check (vs decode_kv ground truth)")
    for c_lo, c_hi in ((127, 127), (120, 134), (60, 90), (100, 119),
                       (135, 160), (0, 30)):
        # build a 1-block window holding EVERY e4m3 byte (256 values fit
        # in one head's 256 K columns) and EVERY e2m1 nibble pair in V
        win = torch.zeros(1, 64, 2, 400, dtype=torch.uint8)
        kb = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 1, 256)
        win[..., :256] = kb.repeat(1, 64, 2, 1)
        # 128 V bytes whose lo/hi nibbles each cover all 16 values
        vb = ((torch.arange(128) % 16) |
              (((torch.arange(128) // 8) % 16) << 4)).to(torch.uint8) \
            .reshape(1, 1, 1, 128)
        win[..., off["v"]:off["v"] + 128] = vb.repeat(1, 64, 2, 1)
        cs = torch.randint(c_lo, c_hi + 1, (1, 64, 2, 8),
                           dtype=torch.uint8,
                           generator=torch.Generator().manual_seed(3))
        win[..., off["ks"]:off["ks"] + 8] = cs
        win[..., off["vs"]:off["vs"] + 8] = cs
        K, V = decode_kv(win[..., : off["v"]], win[..., off["v"]:],
                         "mxk8v4", D)
        # expected: ref decode laws, bf16-rounded like the kernel
        scale = e8m0_to_scale_ref(int(cs[0, 0, 0, 0]))
        k_exp = torch.tensor([e4m3_to_float_ref(b) for b in range(256)])
        # cs varies per group; check group 0's columns (its code = cs[0])
        K_g0 = K[0, 0, 0, :32].double()
        e_g0 = k_exp[:32].double() * scale
        dk = float((K_g0.to(torch.bfloat16).double()
                    - e_g0.to(torch.bfloat16).double()).abs().max())
        v_exp = torch.tensor([e2m1_to_float_ref(n) for n in range(16)])
        vb0 = win[0, 0, 0, off["v"]:off["v"] + 16]
        nibs = torch.stack([vb0 & 0xF, vb0 >> 4], dim=1).reshape(-1)
        V_g0 = V[0, 0, 0, :32].double()
        e_v = v_exp[nibs.long()].double() * scale
        dv = float((V_g0.to(torch.bfloat16).double()
                    - e_v.to(torch.bfloat16).double()).abs().max())
        print(f"   codes[{c_lo}..{c_hi}]: K-law dmax={dk:.3e} "
              f"V-law dmax={dv:.3e} (scale={scale:.3e})")

    print("R225K_KSPROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
