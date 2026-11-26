#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p251_group64_oracle.py — P6.47 P-I r251: the group-64 E8M0 offline
band oracle (cycle 2 of the P-I improve phase, 2026-11-26).

QUESTION (docs/MX-RESEARCH-arc.md §5.1 after the r250 correction):
widening the OCP MX group from 32 to 64 halves the E8M0 scale bytes
— a 2.0–2.9 % slot-byte saving on the KV lanes (16→8 scale B of
400/272 at D=256). The weights side is vendor-locked: the PWAL
layout is [k/32, n] e8m0, the fp4_gemm_w4a16/w4a8
weight-decomposition route reads that layout, and OCP MX defines
the 32-element block — so the lever reduces to the KV side, where
the coarser shared scales land on the acceptance-convicted K
operand (r238 law; r250 disposition).

This oracle prices the QUALITY leg with zero engine wiring: the
certified mx_quant encoders run with the module GROUP switched to
64, and the round-trip error band is measured against group-32 on
IDENTICAL inputs, per operand element-format, per population — the
oracle-input-precision-match law (each variant quantizes its own
input copy; the comparison is variant-vs-variant on one source).

Populations (rows [N, D], blocks of 32 along D):
  gauss    iid N(0,1) — the neutral reference
  heavy    per-block amplitudes |N(0,1)|^3 over an N(0,1) base —
           heavy-tailed adjacent-group amax ratios (the realistic
           heterogeneous case; real post-RoPE K / projected V rows
           carry per-block structure)
  adverse  adjacent 32-block amplitudes alternate 1 / 4 — the legal
           worst case for scale sharing (one 64-scale must cover a
           4x range)

Metrics per (fmt, pop, D): rmse and max-abs error vs the fp32
source, the g64/g32 ratio of each, the fraction of 32-blocks whose
group-64 scale code differs from their own group-32 code (the
coarsening frequency), and the worst-block rmse ratio. D in
{128, 256} (the family head sizes; the target runs 256).

NOT measured here (so the disposition cannot overread): slot
layout/packing at group-64 (the wiring surface — mx_layout
offsets, the Triton store/decode scale-offset law, engine pool
arithmetic, the native kernel — all assume the 32-group law), and
real-distribution acceptance (the rig acceptance probe is the
arbiter IF the band passes). The oracle REPORTS; the VERDICT
decides. Exit 0 always (measurement leg, not a gate).

Output: <out>/p251_group64.json + a human table to stdout. Every
write site uses newline="\\n" (p139 law). Runs on CPU inside a
cert-family container (bit-exact element math: exp2 on integer-
valued fp32, e4m3/e2m1 casts — device-independent); no XPU device
is requested, so the W14 incumbent is untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

# Load mx_quant by FILE PATH under a synthetic top-level name (the
# p147_mxkv_offline law: a synthetic dotless name keeps resolution
# off the installed package machinery entirely — the certified
# reference bytes run, nothing else). PRIMARY source = the installed
# WHEEL inside the cert container (the exact mx_quant that serves);
# dev fallback = a source-tree layout relative to this file.
_WHEEL_MX_QUANT = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv/mx_quant.py"
)


def _load_mx_quant() -> object:
    src = _WHEEL_MX_QUANT if _WHEEL_MX_QUANT.is_file() else None
    if src is None:  # dev fallback: repo layout (tools/ar/p147_mxkv/<me>)
        here = Path(__file__).resolve()
        try:
            cand = here.parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv" / "mx_quant.py"
        except IndexError:  # staged flat on a rig host — no repo tree above
            cand = None
        if cand is not None and cand.is_file():
            src = cand
    if src is None:
        raise SystemExit("p251: mx_quant.py not found (wheel path or source tree)")
    spec = importlib.util.spec_from_file_location("arc_mx_quant_p251", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _pop_gauss(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(n, d, generator=gen)


def _pop_heavy(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    base = torch.randn(n, d, generator=gen)
    amp = torch.randn(n, d // 32, generator=gen).abs().pow(3.0)
    return base * amp.repeat_interleave(32, dim=-1)


def _pop_adverse(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
    base = torch.randn(n, d, generator=gen)
    amp = torch.ones(d // 32)
    amp[0::2] = 4.0
    return base * amp.repeat_interleave(32).expand(n, d)


def _roundtrip(mxq: object, x: torch.Tensor, fmt: str) -> torch.Tensor:
    """Certified encode + inline dequant of the VALUES (no slot
    packing — the band question is value fidelity; packing is the
    wiring surface, dispositioned separately)."""
    code, scale = mxq.e8m0_scales(x, fmt)
    g = x.shape[-1] // code.shape[-1]
    q = (x.float().reshape(*x.shape[:-1], -1, g) / scale[..., None]).reshape(*x.shape)
    if fmt == "mxfp8":
        return q.to(torch.float8_e4m3fn).float() * scale.repeat_interleave(g, dim=-1)
    sign = (q < 0).to(torch.uint8)
    idx = mxq._e2m1_index(q.abs()).to(torch.uint8)
    nib = (sign << 3) | idx
    vals = mxq._decode_e2m1(nib)
    return vals.reshape(*x.shape) * scale.repeat_interleave(g, dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/p251_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(251)
    mxq = _load_mx_quant()

    pops = {"gauss": _pop_gauss, "heavy": _pop_heavy, "adverse": _pop_adverse}
    rows: list[dict[str, object]] = []
    for pi, (pname, pfn) in enumerate(pops.items()):
        for d in (128, 256):
            for fmt in ("mxfp8", "mxfp4"):
                gen = torch.Generator().manual_seed(2510 + d + 17 * pi)
                x = pfn(4096, d, gen)
                rec: dict[str, object] = {"d": d, "pop": pname, "fmt": fmt}
                for g in (32, 64):
                    mxq.GROUP = g  # the certified encoders read it at call time
                    xr = _roundtrip(mxq, x, fmt)
                    err = (xr - x).abs()
                    rec[f"rmse{g}"] = float(err.pow(2).mean().sqrt())
                    rec[f"max{g}"] = float(err.max())
                    if g == 32:
                        code32, _ = mxq.e8m0_scales(x, fmt)
                rec["rmse_ratio"] = rec["rmse64"] / max(rec["rmse32"], 1e-30)
                rec["max_ratio"] = rec["max64"] / max(rec["max32"], 1e-30)
                # coarsening frequency: 32-blocks whose g64 scale code differs
                # from their own g32 code (the shared scale moved for them).
                # code64[..., j] covers the member pair (2j, 2j+1) of code32.
                mxq.GROUP = 64
                code64, _ = mxq.e8m0_scales(x, fmt)
                code32_pairs = code32.reshape(*code64.shape[:-1], -1, 2)
                rec["coarsened_frac"] = float(
                    (code32_pairs != code64[..., None]).float().mean()
                )
                rows.append(rec)
                print(
                    f"D={d} {pname:7s} {fmt:6s} "
                    f"rmse32={rec['rmse32']:.6f} rmse64={rec['rmse64']:.6f} "
                    f"ratio={rec['rmse_ratio']:.3f} max_ratio={rec['max_ratio']:.3f} "
                    f"coarsened={rec['coarsened_frac'] * 100:.1f}%",
                    flush=True,
                )
    doc = {
        "leg": "r251",
        "purpose": "group-64 E8M0 offline band oracle",
        "torch": torch.__version__,
        "seed": 251,
        "rows": rows,
    }
    with open(out / "p251_group64.json", "w", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print("P251_ORACLE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
