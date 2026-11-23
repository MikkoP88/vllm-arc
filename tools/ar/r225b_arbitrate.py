#!/usr/bin/env python3
"""r225b_arbitrate.py — float64 arbitration of the r225a factor outputs
(CPU-only; runs in any container with the tree's vllm_gaudi on path).

Loads the banked r224a payloads, computes the float64 ground-truth
attention ONCE per payload (the proven r224d math: mx_quant.decode_kv
is the byte-layout ground truth, GQA G=6, D=256, window
position-indexed), then scores every r225a factor's saved native output
against it. The factor that collapses to the Triton lane's error level
is the root cause; factors that stay at the banked native error level
are exonerated.

Verdict rule: a factor is AT_TRITON_LEVEL when its dmax vs the float64
reference is <= 2x the banked Triton lane's dmax vs the same reference
(the Triton lane itself varies 4.8e-4..3.6e-3 across payloads).

Usage: r225b_arbitrate.py <bank_dir> <out_dir> <stem> [<stem2> ...]
  bank_dir: directory holding mxkv_ab.<pid>.<n>.pt payloads
  out_dir : directory holding r225a_<factor>.pt output banks
  stem    : payload filename stems to arbitrate (e.g. mxkv_ab.2447.18)
"""

import os
import sys

import torch

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv

BLOCK = 64
D = 256
GROUP = 6  # GQA: Hq=12 per worker / Hk=2
FACTORS = ("base", "dbf0", "nsplit1", "p64")


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def ref_from_payload(p):
    """The proven r224d reference: float64 attention over the payload's
    own pool bytes (decode_kv is the byte-layout ground truth)."""
    r = p["q"].shape[0]
    q = p["q"][:r].double()                       # (r, 12, 256)
    sl = p["sl"][:r].to(torch.long)
    win = p["kv_win"]                              # (nb, 64, 2, 400)
    scale = float(p["scale"])
    off = _mx_off(D, "mxk8v4")

    k_bytes = win[..., : off["v"]]
    v_bytes = win[..., off["v"]:]
    K, V = decode_kv(k_bytes, v_bytes, "mxk8v4", D)
    K, V = K.double(), V.double()

    o_ref = torch.empty(r, q.shape[1], D, dtype=torch.float64)
    for i in range(r):
        nb_i = int((sl[i] + BLOCK - 1) // BLOCK)
        # kv_win is gathered in bt_head order — dim0 IS the position in
        # that order, NOT the pool's block id (r224d law).
        last = int(sl[i] - (nb_i - 1) * BLOCK)
        k_seq = K[:nb_i]
        v_seq = V[:nb_i]
        k_seq = torch.cat(
            [k_seq[:-1].reshape(-1, 2, D), k_seq[-1][:last]], dim=0)
        v_seq = torch.cat(
            [v_seq[:-1].reshape(-1, 2, D), v_seq[-1][:last]], dim=0)
        for h in range(q.shape[1]):
            kv_h = h // GROUP
            s = (q[i, h] @ k_seq[:, kv_h].T) * scale
            p_attn = torch.softmax(s, dim=-1)
            o_ref[i, h] = p_attn @ v_seq[:, kv_h]
    return o_ref


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    bank_dir, out_dir = sys.argv[1], sys.argv[2]
    stems = sys.argv[3:]

    factor_banks = {}
    for f in FACTORS:
        path = os.path.join(out_dir, f"r225a_{f}.pt")
        if os.path.isfile(path):
            factor_banks[f] = torch.load(path, map_location="cpu",
                                         weights_only=False)
        else:
            print(f"   (no output bank for factor {f})")

    repaired = []
    for stem in stems:
        ppath = os.path.join(bank_dir, f"{stem}.pt")
        p = torch.load(ppath, map_location="cpu", weights_only=False)
        r = p["q"].shape[0]
        o_ref = ref_from_payload(p)
        dt = dmax(p["out_triton"][:r], o_ref)
        dn = dmax(p["out_native"][:r], o_ref)
        print(f"== {stem}  banked: triton-vs-ref={dt:.3e} "
              f"native-vs-ref={dn:.3e}")
        for f, bank in factor_banks.items():
            if stem + ".pt" not in bank:
                print(f"   {f:<8} (missing)")
                continue
            df = dmax(bank[stem + ".pt"], o_ref)
            level = "AT_TRITON_LEVEL" if df <= 2.0 * dt else "STILL_WRONG"
            if level == "AT_TRITON_LEVEL":
                repaired.append(f)
            print(f"   {f:<8} dmax={df:.3e}  {level}")
    ok = sorted(set(repaired))
    print(f"R225B_VERDICT: "
          f"{'+'.join(ok) if ok else 'NONE'} at triton level")
    return 0


if __name__ == "__main__":
    sys.exit(main())
