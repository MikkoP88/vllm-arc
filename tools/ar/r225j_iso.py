#!/usr/bin/env python3
"""r225j_iso.py — isolation bisect of the real-payload residual
(r225i proved the kernel bitwise-exact on synthetic windows at every
seqlen; the residual must live in what synthetics did not cover).

Two axes, one GPU session:

  A. PER-ROW ISOLATION — replay each banked row ALONE (q[i:i+1],
     sl[i], shared window).  Exact rows + erroneous batch => the
     multi-row batch geometry injects the error; erroneous rows =>
     the payload bytes carry it.

  B. SURFACE SWAP — a synthetic window (proven exact) with ONE byte
     surface replaced by the real payload's surface (K / Ks / V /
     Vs), single row at the payload's own seqlen.  The first surface
     that turns the output erroneous carries the defect class; the
     model always decodes the HYBRID bytes via decode_kv.

Usage: r225j_iso.py <payload.pt> [payload2.pt ...]
"""

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
SURFACES = ("K", "Ks", "V", "Vs")


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def grid_stats(native_bf16, model_f64):
    m = model_f64.to(torch.bfloat16)
    mism = int((native_bf16.view(torch.int16)
                != m.view(torch.int16)).sum())
    ref = model_f64.abs()
    diff = (native_bf16.double() - model_f64).abs()
    big = ref > torch.finfo(torch.bfloat16).tiny
    if big.any():
        e = torch.floor(torch.log2(ref[big].clamp_min(1e-30)))
        max_ulp = float((diff[big] / torch.pow(2.0, e - 7)).max())
    else:
        max_ulp = 0.0
    return mism, max_ulp


def model_row(q_row, K, V, sl_val, scale):
    """fp64 attention for ONE row on decoded K/V."""
    nb_i = (sl_val + BLOCK - 1) // BLOCK
    last = sl_val - (nb_i - 1) * BLOCK
    k_seq = torch.cat([K[:nb_i - 1].reshape(-1, 2, D), K[nb_i - 1][:last]],
                      dim=0)
    v_seq = torch.cat([V[:nb_i - 1].reshape(-1, 2, D), V[nb_i - 1][:last]],
                      dim=0)
    o = torch.empty(q_row.shape[0], D, dtype=torch.float64)
    for h in range(q_row.shape[0]):
        kv_h = h // GROUP
        s = (q_row[h] @ k_seq[:, kv_h].T) * scale
        p_attn = torch.softmax(s.double(), dim=-1)
        o[h] = p_attn @ v_seq[:, kv_h]
    return o


def synth_window(nb, seed, off):
    g = torch.Generator().manual_seed(seed)
    win = torch.zeros(nb, BLOCK, 2, 400, dtype=torch.uint8)
    kb = torch.randint(0, 256, (nb, BLOCK, 2, D), dtype=torch.uint8,
                       generator=g)
    kb[kb == 0x7F] = 0x7E
    kb[kb == 0xFF] = 0xFE
    win[..., : off["ks"]] = kb
    win[..., off["v"]:off["v"] + D // 2] = torch.randint(
        0, 256, (nb, BLOCK, 2, D // 2), dtype=torch.uint8, generator=g)
    for key in ("ks", "vs"):
        base = off[key]
        win[..., base:base + 8] = torch.randint(
            120, 135, (nb, BLOCK, 2, 8), dtype=torch.uint8, generator=g)
    return win


def surface_slice(win, name, off):
    if name == "K":
        return slice(0, off["ks"])
    if name == "Ks":
        return slice(off["ks"], off["ks"] + 8)
    if name == "V":
        return slice(off["v"], off["v"] + D // 2)
    return slice(off["vs"], off["vs"] + 8)


class Replay:
    """Engine-geometry GPU replay on an arbitrary window."""

    def __init__(self, template, dev):
        self.shape = tuple(template["kv_shape"])
        self.stride = tuple(template["kv_stride"])
        self.off_i = int(template["kv_off"])
        self.bt_dtype = template["bt"].dtype
        self.dev = dev

    def __call__(self, q_rows, win, sl_vals, scale):
        nb = win.shape[0]
        r = len(sl_vals)
        storage = torch.zeros(self.off_i + self.shape[0] * BLOCK_STRIDE,
                              dtype=torch.uint8, device=self.dev)
        for b in range(nb):
            storage[self.off_i + b * BLOCK_STRIDE:
                    self.off_i + (b + 1) * BLOCK_STRIDE] = \
                win[b].reshape(-1).to(self.dev)
        pool = torch.as_strided(storage, self.shape, self.stride,
                                self.off_i)
        bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=self.bt_dtype,
                         device=self.dev)
        bt[:, :nb] = torch.arange(nb, dtype=self.bt_dtype, device=self.dev)
        sl32 = torch.tensor(sl_vals, dtype=torch.int32, device=self.dev)
        cu_q = torch.arange(r + 1, dtype=torch.int32, device=self.dev)
        out = mx_paged_decode(
            q_rows.to(self.dev), pool, "mxk8v4", cu_q, sl32, bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale, num_splits_kv=0,
        )[:r].cpu()
        del pool, storage
        torch.xpu.empty_cache()
        return out


def main() -> int:
    dev = torch.device("xpu")
    off = _mx_off(D, "mxk8v4")

    for path in sys.argv[1:]:
        p = torch.load(path, map_location="cpu", weights_only=False)
        r = p["q"].shape[0]
        sl = p["sl"][:r].to(torch.long).tolist()
        win = p["kv_win"].to(torch.uint8)
        scale = float(p["scale"])
        # the native op requires bf16 q; the model decodes the SAME
        # bf16 values so both sides see identical operands
        q = p["q"].to(torch.bfloat16)
        rep = Replay(p, dev)
        K, V = decode_kv(win[..., : off["v"]], win[..., off["v"]:],
                         "mxk8v4", D)
        K, V = K.double(), V.double()
        name = path.split("/")[-1]
        print(f"== {name}  rows={r} sl={sl}")

        # ---- A: per-row isolation ----
        for i in range(r):
            q1 = q[i:i + 1]
            n_out = rep(q1, win, [sl[i]], scale)[0]
            model = model_row(q1[0].double(), K, V, sl[i], scale)
            mism, mul = grid_stats(n_out, model)
            print(f"   row {i} alone : dmax={dmax(n_out, model):.3e}"
                  f"  mism={mism}/{12 * D}  max_ulp={mul:.2f}")

        # ---- B: surface swaps (row 0, synthetic base) ----
        nb = win.shape[0]
        base_win = synth_window(nb, seed=11, off=off)
        for sname in SURFACES:
            hyb = base_win.clone()
            hyb[..., surface_slice(hyb, sname, off)] = \
                win[..., surface_slice(win, sname, off)]
            hK, hV = decode_kv(hyb[..., : off["v"]],
                               hyb[..., off["v"]:], "mxk8v4", D)
            model = model_row(q[0].double(), hK.double(), hV.double(),
                              sl[0], scale)
            n_out = rep(q[0:1], hyb, [sl[0]], scale)[0]
            mism, mul = grid_stats(n_out, model)
            print(f"   swap {sname:<2}     : dmax={dmax(n_out, model):.3e}"
                  f"  mism={mism}/{12 * D}  max_ulp={mul:.2f}")

        # control: synthetic base alone at the same geometry
        bK, bV = decode_kv(base_win[..., : off["v"]],
                           base_win[..., off["v"]:], "mxk8v4", D)
        model = model_row(q[0].double(), bK.double(), bV.double(),
                          sl[0], scale)
        n_out = rep(q[0:1], base_win, [sl[0]], scale)[0]
        mism, mul = grid_stats(n_out, model)
        print(f"   synth base    : dmax={dmax(n_out, model):.3e}"
              f"  mism={mism}/{12 * D}  max_ulp={mul:.2f}")

    print("R225J_ISO_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
