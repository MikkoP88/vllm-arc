#!/usr/bin/env python3
"""r225m_splitsim.py — CPU simulation of the split-KV combine contract
(r225 gate3 follow-up: fixes 1+2+3 are live in the gated bytes, the
outputs MOVED, yet a same-class residual persists — a FOURTH source).

Question this probe answers: is the post-fix-3 CONTRACT (fp32 partials,
fp32 exp_sums/max_logits, fp32 combine with precise exp2, one final
bf16 cast) mathematically sufficient in the flat deep-layer softmax
regime — or does the kernel's split path carry a real walk/partition
defect beyond precision?

Levels (same fp64 logits/inputs for all; only the split-path fidelity
varies; partition mirrors the kernel: 16-token tiles, uniform ceil-div
nbps = ceil(kv_tiles/S), split s owns tiles [s*nbps, min((s+1)*nbps,
kv_tiles))):

  L0  everything fp64, regrouped by split — regrouping is exact; any
      error here is simulation noise (expect ~1e-15).
  L1  the post-fix-3 kernel contract: per-split m_s/E_s/O_s in fp32,
      partial O stored fp32, combine in fp32 (precise exp), final
      round to bf16. If the contract suffices: ~1e-6.
  L2  pre-fix-3: partial O stored bf16 (temp_out was bf16) — must
      reproduce the historical ~5e-2 class, validating the simulator.

Compare every level (and the gated wheel output) against the fp64
reference; if L1 is ~1e-6 while the wheel's base sits at ~5e-2, the
residual is a real kernel split-path defect (tile walk / partition /
partial write), not precision — and the hunt moves there.

Usage: r225m_splitsim.py <payload.pt> <wheel_base.pt> [S ...]
  (default S ladder: 2 8; the wheel base is r225d_base.pt keyed by
  payload stem)
"""

import sys

import torch

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv

BLOCK = 64
D = 256
GROUP = 6
TILE = 16  # get<1>(TileShapeQK): KV tokens per mainloop tile (page-16)


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def logits_for_payload(p):
    """fp64 logits + values per (row, head): mirrors r225b's reference
    walk (window position-ordered, GQA gather), returns lists indexed
    [row][head] of (logits (n,), v (n, D))."""
    r = p["q"].shape[0]
    q = p["q"][:r].double()
    sl = p["sl"][:r].to(torch.long)
    win = p["kv_win"]
    scale = float(p["scale"])
    off = _mx_off(D, "mxk8v4")
    K, V = decode_kv(win[..., : off["v"]], win[..., off["v"]:], "mxk8v4", D)
    K, V = K.double(), V.double()
    per_row = []
    for i in range(r):
        nb_i = int((sl[i] + BLOCK - 1) // BLOCK)
        last = int(sl[i] - (nb_i - 1) * BLOCK)
        k_seq = torch.cat(
            [K[:nb_i][:-1].reshape(-1, 2, D), K[:nb_i][-1][:last]], dim=0)
        v_seq = torch.cat(
            [V[:nb_i][:-1].reshape(-1, 2, D), V[:nb_i][-1][:last]], dim=0)
        row = []
        for h in range(q.shape[1]):
            kv_h = h // GROUP
            s = (q[i, h] @ k_seq[:, kv_h].T) * scale
            row.append((s, v_seq[:, kv_h]))
        per_row.append(row)
    return per_row


def split_bounds(n, s_count):
    """Kernel partition: tile ranges from uniform ceil-div."""
    tiles = (n + TILE - 1) // TILE
    nbps = (tiles + s_count - 1) // s_count
    bounds = []
    for s in range(s_count):
        t0 = s * nbps
        t1 = min((s + 1) * nbps, tiles)
        if t0 >= t1:
            break
        bounds.append((min(t0 * TILE, n), min(t1 * TILE, n)))
    return bounds


def combine_level(logits, v, bounds, dtype_partial, dtype_comb):
    """One (row, head) split-path evaluation at a fidelity level.

    Per split: m_s = max(logits_s) cast to dtype_comb; E_s = sum of
    exp(l - m_s) accumulated in dtype_comb; O_s = p_s @ v accumulated
    in dtype_comb then stored (rounded) to dtype_partial. Combine in
    dtype_comb: acc = sum O_s * exp(m_s - m_g), den = sum E_s * exp,
    out = acc / den.
    """
    parts = []
    for lo, hi in bounds:
        ls = logits[lo:hi].to(dtype_comb)
        vs = v[lo:hi].to(dtype_comb)
        m = ls.max()
        ps = torch.exp(ls - m)
        e = ps.sum()
        o = ps @ vs
        parts.append((m, e, o.to(dtype_partial).to(dtype_comb)))
    m_g = torch.stack([m for m, _, _ in parts]).max()
    acc = torch.zeros(v.shape[1], dtype=dtype_comb)
    den = torch.zeros((), dtype=dtype_comb)
    for m, e, o in parts:
        rs = torch.exp(m - m_g)
        acc += o * rs
        den += e * rs
    return (acc / den)


def main() -> int:
    payload_path, base_path = sys.argv[1], sys.argv[2]
    s_ladder = [int(x) for x in sys.argv[3:]] or [2, 8]
    stem = payload_path.split("/")[-1][:-3]

    p = torch.load(payload_path, map_location="cpu", weights_only=False)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]

    per_row = logits_for_payload(p)

    # fp64 single-shot reference (the arbitration ground truth)
    o_ref = torch.empty(r, p["q"].shape[1], D, dtype=torch.float64)
    for i in range(r):
        for h in range(p["q"].shape[1]):
            s, v = per_row[i][h]
            o_ref[i, h] = torch.softmax(s, dim=-1) @ v

    wb = base[stem + ".pt"][:r].to(torch.bfloat16).double() \
        if stem + ".pt" in base else None
    if wb is None:
        print(f"WARN: {stem} not in wheel base bank; wheel compare off")

    print(f"== {stem}  n={int(p['sl'][0])} rows={r} "
          f"wheel-base-vs-ref dmax="
          f"{dmax(wb, o_ref) if wb is not None else float('nan'):.3e}")
    for nsplit in s_ladder:
        for lvl, dpart, dcomb in (("L0", torch.float64, torch.float64),
                                  ("L1", torch.float32, torch.float32),
                                  ("L2", torch.bfloat16, torch.float32)):
            out = torch.empty(r, p["q"].shape[1], D, dtype=torch.float64)
            for i in range(r):
                n = int(p["sl"][i])
                bounds = split_bounds(n, nsplit)
                for h in range(p["q"].shape[1]):
                    s, v = per_row[i][h]
                    out[i, h] = combine_level(s, v, bounds,
                                              dpart, dcomb).double()
            out_b = out.to(torch.bfloat16).double()
            line = (f"   S={nsplit} {lvl}: dmax-vs-ref="
                    f"{dmax(out_b, o_ref):.3e}")
            if wb is not None:
                line += f"  vs-wheel={dmax(out_b, wb):.3e}"
            print(line)
    print("R225M_SPLITSIM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
