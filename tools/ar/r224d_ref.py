#!/usr/bin/env python3
"""r224d_ref.py — float64 ground-truth attention over an r224a payload
(CPU-only; runs in any container with the tree's vllm_gaudi on path).

With the r224c bisection having excluded pool geometry (bit-exact
replay of BOTH kernels; offset0 a no-op) and pinned the mxk8v4
divergence as deterministic, data-dependent, and growing with layer
depth, this script computes the attention in float64 from the payload's
own pool bytes — the byte-layout ground truth is mx_quant.decode_kv,
the same reference the Triton kernels were built to reproduce — and
arbitrates which banked output is correct:

    O_ref = softmax(q @ K^T * scale) @ V      per q-head over its GQA
    kv-head (G = Hq / Hk = 12 / 2 = 6), causal to the row's sl.

Reports dmax/dmean of out_triton and out_native against O_ref, the
per-head error profile of the wrong side, and the score-range
correlation (softmax/exp-path defects scale with logit magnitude;
dequant defects do not). Diagnostic, not a gate.

Usage: r224d_ref.py <payload.pt> [payload2.pt ...]
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


def ref_one(path: str) -> None:
    p = torch.load(path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    q = p["q"][:r].double()                       # (r, 12, 256)
    sl = p["sl"][:r].to(torch.long)
    win = p["kv_win"]                              # (nb, 64, 2, 400)
    bt_head = p["bt_head"].to(torch.long)
    scale = float(p["scale"])
    out_triton = p["out_triton"][:r]
    out_native = p["out_native"][:r]
    off = _mx_off(D, "mxk8v4")

    # dequant the whole window once: (nb, 64, 2, 256) float32
    k_bytes = win[..., : off["v"]]
    v_bytes = win[..., off["v"]:]
    K, V = decode_kv(k_bytes, v_bytes, "mxk8v4", D)
    K, V = K.double(), V.double()

    print(f"== {path}")
    print(f"   fmt={p['fmt']} rows={r} sl={sl.tolist()} off={p['kv_off']} "
          f"bt_head={bt_head[:4].tolist()}... banked dmax={p['dmax']:.3e}")

    o_ref = torch.empty(r, q.shape[1], D, dtype=torch.float64)
    score_max = torch.empty(r, q.shape[1])
    for i in range(r):
        nb_i = int((sl[i] + BLOCK - 1) // BLOCK)
        # kv_win was index_select'd in bt_head order — its dim0 IS the
        # position in that order, NOT the pool's block id (K[bt_head]
        # indexes by id and overruns the 96-slot window).
        last = int(sl[i] - (nb_i - 1) * BLOCK)
        k_seq = K[:nb_i]                           # (nb_i, 64, 2, D)
        v_seq = V[:nb_i]
        # flatten tokens, trimming the final block beyond the causal limit
        k_seq = torch.cat(
            [k_seq[:-1].reshape(-1, 2, D), k_seq[-1][:last]], dim=0)
        v_seq = torch.cat(
            [v_seq[:-1].reshape(-1, 2, D), v_seq[-1][:last]], dim=0)
        for h in range(q.shape[1]):
            kv_h = h // GROUP
            s = (q[i, h] @ k_seq[:, kv_h].T) * scale
            score_max[i, h] = s.abs().max()
            p_attn = torch.softmax(s, dim=-1)
            o_ref[i, h] = p_attn @ v_seq[:, kv_h]

    dt = dmax(out_triton, o_ref)
    dn = dmax(out_native, o_ref)
    print(f"   REF dmax triton-vs-ref = {dt:.3e}")
    print(f"   REF dmax native-vs-ref = {dn:.3e}")
    verdict = "TRITON_WRONG" if dt > dn else (
        "NATIVE_WRONG" if dn > dt else "AMBIGUOUS")
    print(f"   R224D_VERDICT: {verdict} "
          f"(ratio={max(dt, dn) / max(min(dt, dn), 1e-12):.1f})")
    # score-range correlation: softmax/exp defects scale with |logit|
    err = (out_native.double() - o_ref).amax(dim=(0, 2))  # per q-head
    sm = score_max.amax(dim=0)
    if float(sm.std()) > 0:
        corr = float(torch.corrcoef(torch.stack([sm, err]))[0, 1])
        print(f"   per-head err={torch.round(err, decimals=3).tolist()}")
        print(f"   per-head score_max={torch.round(sm, decimals=1).tolist()}")
        print(f"   corr(|score_max|, err) = {corr:.3f}")


def main() -> int:
    for path in sys.argv[1:]:
        try:
            ref_one(path)
        except Exception as exc:  # noqa: BLE001 — report, keep going
            print(f"== {path}: R224D_FAILED {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
