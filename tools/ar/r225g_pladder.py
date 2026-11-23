#!/usr/bin/env python3
"""r225g_pladder.py — P-representation ladder test of the post-fix
residual.

r225f exonerated the group-scale mapping (no per-column structure);
the residual's shape (err/|O| p50 ~ 1.5e-3 = 2^-9.4 on sharp rows,
huge ulp multiples only at deep-cancellation elements) is the
signature of the softmax probabilities entering the PV product as a
SINGLE bf16 term — the exact floor the r198 mini-ledger describes for
a pair-only ladder, which the in-kernel TRIPLE (tArP + tArP2 + tArP3,
mainloop lines ~1340-1418) was supposed to eliminate.

The triple was validated by (a) the r198/r200f offline ledger and (b)
the bitwise faces gate at seq_len==1 — where softmax P == 1.0 exactly,
so (b) never exercised P precision at all.  This script reproduces the
hypothesis the direct way: compute the float64 attention with P
truncated to 1 / 2 / 3 bf16 terms (V, K exact; fp64 accumulation) and
ask which rung the kernel's actual output sits on:

  base ~= round_bf16(refP1)  -> triple DEAD, single-bf16 P live
  base ~= round_bf16(refP2)  -> pair live, third term dead
  base ~= round_bf16(refP3)  -> triple live; residual is other noise

Usage: r225g_pladder.py <bank_dir> <out_dir> <stem> [stem2 ...]
"""

import sys

import torch

BLOCK = 64
D = 256
GROUP = 6  # GQA: Hq=12 per worker / Hk=2


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float64)


def attention(q, K, V, sl, scale, p_rungs):
    """float64 attention; p_rungs = number of bf16 terms for P (0=exact).
    Returns {rungs: O}."""
    r = q.shape[0]
    outs = {n: torch.empty(r, q.shape[1], D, dtype=torch.float64)
            for n in p_rungs}
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
            for n in p_rungs:
                if n == 0:
                    p_eff = p_attn
                else:
                    p_eff = bf16(p_attn)
                    rem = p_attn - p_eff
                    for _ in range(n - 1):
                        term = bf16(rem)
                        p_eff = p_eff + term
                        rem = rem - term
                outs[n][i, h] = p_eff @ v_seq[:, kv_h]
    return outs


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def main() -> int:
    bank_dir, out_dir = sys.argv[1], sys.argv[2]
    stems = sys.argv[3:]
    if not stems:
        print("R225G_PLADDER_VERDICT: FAIL (no stems)")
        return 1

    base = torch.load(f"{out_dir}/r225d_base.pt", map_location="cpu",
                      weights_only=False)
    sys.path.insert(0, "/tmp")
    from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
    from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402

    rungs = (0, 1, 2, 3)
    names = {0: "exact", 1: "P1 single-bf16", 2: "P2 pair",
             3: "P3 triple"}
    for stem in stems:
        key = stem + ".pt"
        p = torch.load(f"{bank_dir}/{key}", map_location="cpu",
                       weights_only=False)
        r = p["q"].shape[0]
        q = p["q"][:r].double()
        sl = p["sl"][:r].to(torch.long)
        win = p["kv_win"]
        scale = float(p["scale"])
        off = _mx_off(D, "mxk8v4")
        K, V = decode_kv(win[..., : off["v"]], win[..., off["v"]:],
                         "mxk8v4", D)
        K, V = K.double(), V.double()

        outs = attention(q, K, V, sl, scale, rungs)
        o_nat = base[key][:r]
        o_nat_r = o_nat.to(torch.bfloat16).double()

        print(f"== {stem}")
        for n in rungs:
            # kernel output vs the rung (both as bf16-rounded grids)
            d_raw = dmax(outs[n], o_nat)
            d_r = dmax(outs[n].to(torch.bfloat16).double(), o_nat_r)
            print(f"   {names[n]:<15} vs base: dmax={d_raw:.3e}"
                  f"  bf16-rounded={d_r:.3e}")
        # calibration: exact rung vs the banked triton lane
        d_tri = dmax(outs[0], p["out_triton"][:r])
        print(f"   (calibration: exact vs banked triton = {d_tri:.3e})")

    print("R225G_PLADDER_VERDICT: PROFILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
