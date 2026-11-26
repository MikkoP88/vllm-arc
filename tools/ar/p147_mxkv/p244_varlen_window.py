#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p244_varlen_window.py — P6.47 P-C2 (r244): offline XPU validation of
the TQ-LANE PREFILL sliding-window fix (the python half of P-C2).

P-C2's engine edit routes sliding-window models' prefill through the
stock fa_utils varlen lane with window_size=(W-1, 0) —
engine.py ``_flash_attn_varlen`` — against the STOCK cert28 image venv
(no arc7mx wheel needed: the varlen prefill op is untouched upstream
surface; only the MX paged-decode kernel needed the P-C2 twin). This
probe is that fix's offline authority: it imports the op by the
engine's exact chain and proves the window actually MASKS.

Cells (prefill varlen geometries, GQA G=4, bf16):
  d128  D=128 HQ=8 HK=2
  d256  D=256 HQ=8 HK=2

Faces: stock (windowless causal varlen vs causal fp64 — the
non-sliding regression guard; the engine's windowless route must stay
exact) / exact (W=64 binding, lens [200, 350] in one varlen call, vs
window-masked fp64; d_blind vs the causal-only ref is the
window-blind discriminator — a blind op fails the masked gate AND the
explicit blindness check) / nonbinding (W=1024, lens [50, 200]:
window cannot bind — windowed must gate vs the UNMASKED ref and agree
with the windowless twin) / boundary (W=64 lens 63/64/65;
W=1024 lens 1023/1024/1025 — the window and length edges) /
telemetry (ctx 8192 D=256: windowed W=1024 vs windowless walk cost —
informational).

Gates: the r198 face law ADAPTED to the stock lane's precision
envelope — the stock varlen op emits bf16 from an fp32-accum pipeline,
so its windowless face itself measures a ~1e-2 / ~2-ulp floor against
the fp64 reference (the 2e-3 constant was calibrated on OUR MX decode
kernel, not this op). Each cell's stock face runs FIRST and its
measured maxdiff becomes that cell's calibrated floor: windowed faces
gate at d <= floor + 2e-3 (the triangle law with the stock face as
the calibrated alternative) or bf16-rounded 1-ulp, plus cos>=0.99999,
plus the explicit blindness gate (W < n => d_blind >= 1e-3 — observed
~1.0, 100x the floor). Exit 0 only if every gate passes AND the
fa_utils import chain (the engine's exact import) succeeded. Output:
<out>/p244_varlen.json. Every write site uses newline="\\n" (p139 law).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

DEV = "xpu"
CELLS = {
    "d128": {"D": 128, "HQ": 8, "HK": 2},
    "d256": {"D": 256, "HQ": 8, "HK": 2},
}
R = {"_meta": {}}
FAILED = []
# Per-cell stock-lane precision floor, measured by face_stock (runs
# first for every cell): the windowed faces gate at floor + 2e-3.
STOCK_D = {}
# The stock face IS the floor source, so its own gate is the absolute
# breakage guard only: d <= 5e-2 (~6 bf16 ulps at the data scale — a
# real defect: wrong mask, wrong GQA map, wrong scale, shows d ~
# O(0.1-1) or a cos collapse; the precision detail is carried by the
# floor it measures for the windowed faces).
STOCK_ABS = 5e-2

# --- the engine's exact import (P-C2 engine.py try/except ImportError) ---
# If fa_utils fails to import, the P-C2 sliding-prefill route is dead at
# the source (engine raises NotImplementedError for sliding models) — a
# fallback to the shallower xpu_ops shim is recorded but FAILS the probe.
try:
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func as _fa_varlen,
    )

    IMPORT_PATH = "fa_utils"
except ImportError as _e:  # noqa: BLE001
    print(f"  FAIL import/fa_utils: {_e}")
    FAILED.append(f"import/fa_utils: {_e}")
    try:
        from vllm._xpu_ops import xpu_ops as _xops

        _fa_varlen = _xops.flash_attn_varlen_func
        IMPORT_PATH = "xpu_ops_fallback"
    except ImportError as _e2:  # noqa: BLE001
        print(f"  FAIL import/xpu_ops: {_e2}")
        FAILED.append(f"import/xpu_ops: {_e2}")
        _fa_varlen = None
        IMPORT_PATH = "none"
R["_meta"]["import_path"] = IMPORT_PATH
R["_meta"]["torch"] = torch.__version__


def _rec(face, cell, **kv):
    R.setdefault(face, {}).setdefault(cell, {}).update(kv)


def _fail(face, cell, msg):
    FAILED.append(f"{face}/{cell}: {msg}")
    print(f"  FAIL {face}/{cell}: {msg}")


def _ok(face, cell, msg):
    print(f"  ok   {face}/{cell}: {msg}")


def cos_sim(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def cell_name(c):
    return f"d{c['D']}"


def _prep(c, lens, seed):
    """Seeded bf16 q/k/v varlen batch: (total, H, D) on XPU."""
    gen = torch.Generator().manual_seed(seed)
    total = sum(lens)
    q = torch.randn(total, c["HQ"], c["D"], generator=gen).to(torch.bfloat16)
    k = torch.randn(total, c["HK"], c["D"], generator=gen).to(torch.bfloat16)
    v = torch.randn(total, c["HK"], c["D"], generator=gen).to(torch.bfloat16)
    return q.to(DEV), k.to(DEV), v.to(DEV)


def varlen_call(c, q, k, v, lens, window=None):
    """The engine's exact call form (engine.py _flash_attn_varlen)."""
    cu = [0]
    for ln in lens:
        cu.append(cu[-1] + ln)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=DEV)
    return _fa_varlen(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_t,
        cu_seqlens_k=cu_t,
        max_seqlen_q=max(lens),
        max_seqlen_k=max(lens),
        softmax_scale=1.0 / math.sqrt(c["D"]),
        causal=True,
        window_size=None if window is None else (window - 1, 0),
    )


def _ref_prefill(c, q, k, v, lens, window=None):
    """fp64 CPU reference: causal (+ optional (W-1, 0) window) per seq."""
    scale = 1.0 / math.sqrt(c["D"])
    gqa = c["HQ"] // c["HK"]
    outs = []
    t0 = 0
    for ln in lens:
        qs = q[t0 : t0 + ln].double().cpu()
        # GQA head map: q head h -> kv head h // gqa (repeat_interleave
        # is the exact inverse mapping the kernel applies).
        ks = k[t0 : t0 + ln].double().cpu().repeat_interleave(gqa, dim=1)
        vs = v[t0 : t0 + ln].double().cpu().repeat_interleave(gqa, dim=1)
        qi = torch.arange(ln).unsqueeze(1)
        kj = torch.arange(ln).unsqueeze(0)
        mask = kj > qi
        if window is not None:
            mask = mask | (kj < qi - (window - 1))
        s = torch.einsum("ihd,jhd->hij", qs, ks) * scale
        s = s.masked_fill(mask.unsqueeze(0), float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hij,jhd->ihd", p, vs))
        t0 += ln
    return torch.cat(outs, 0)


def _gated(face, cell, tag, out, ref_same, other=None, extra="", floor=0.0):
    """Common gate: bf16-output-aware exact comparison (r198 face law).

    floor: the cell's stock-lane precision envelope (measured by
    face_stock) — the stock op emits bf16 from fp32 accum, so its
    native error vs fp64 is ~1e-2, not the MX-kernel 2e-3.
    """
    outf = out.float().cpu()
    ref_same = ref_same.cpu()
    if not torch.isfinite(outf).all():
        _fail(face, cell, f"{tag}: non-finite output")
        return None
    d_exact = (outf - ref_same).abs().max().item()
    c_same = cos_sim(outf, ref_same)
    ref_r = ref_same.bfloat16().float()
    diff_r = (outf - ref_r).abs()
    d_rounded = diff_r.max().item()
    mag = torch.maximum(outf.abs(), ref_same.abs())
    expo = torch.floor(torch.log2(mag.clamp_min(1e-38)))
    ulp_elem = torch.exp2(torch.where(mag > 0, expo, torch.full_like(expo, -126.0)) - 7.0)
    ok_exact = d_exact <= 2e-3 + floor
    ok_rounded = bool((diff_r <= ulp_elem).all())
    if other is not None:
        of = other.float().cpu()
        d_o = (of - ref_same).abs().max().item()
        ok_exact = ok_exact or d_exact <= 2e-3 + d_o
        extra = f"{extra} d_other={d_o:.2e}"
        diff_o = (outf - of.bfloat16().float()).abs()
        mag_o = torch.maximum(outf.abs(), of.abs())
        expo_o = torch.floor(torch.log2(mag_o.clamp_min(1e-38)))
        ulp_o = torch.exp2(
            torch.where(mag_o > 0, expo_o, torch.full_like(expo_o, -126.0)) - 7.0
        )
        ok_rounded = ok_rounded or bool((diff_o <= ulp_o).all())
    gated = (ok_exact or ok_rounded) and c_same >= 0.99999
    if not gated:
        _fail(
            face,
            cell,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} "
            f"cos={c_same:.7f}{extra}",
        )
    else:
        _ok(
            face,
            cell,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} "
            f"cos={c_same:.7f}{extra}",
        )
    return dict(maxdiff=d_exact, maxdiff_rounded=d_rounded, cos_same=c_same, gated=gated)


def face_stock(c, seed=5):
    """Windowless baseline guard: stock causal varlen vs causal fp64."""
    name = cell_name(c)
    print(f"[face: stock/{name}] windowless causal varlen vs causal fp64")
    lens = [200, 350]
    q, k, v = _prep(c, lens, seed)
    out = varlen_call(c, q, k, v, lens)
    torch.xpu.synchronize()
    ref = _ref_prefill(c, q, k, v, lens)
    rec = _gated("stock", name, "causal", out, ref, floor=STOCK_ABS - 2e-3)
    if rec:
        # Record the measured stock-lane floor for this cell — the
        # windowed faces gate against it (triangle law).
        STOCK_D[name] = rec["maxdiff"]
        rec["is_stock_floor"] = True
        _rec("stock", name, causal=rec)


def face_exact(c, T_lens=(200, 350), W=64, seed=99):
    """Windowed varlen vs window-masked fp64 (the window BINDS: W << n)."""
    name = cell_name(c)
    print(f"[face: exact/{name}] windowed varlen (W={W}) vs window-masked fp64")
    lens = list(T_lens)
    q, k, v = _prep(c, lens, seed)
    out = varlen_call(c, q, k, v, lens, window=W)
    torch.xpu.synchronize()
    ref_win = _ref_prefill(c, q, k, v, lens, window=W)
    ref_full = _ref_prefill(c, q, k, v, lens)
    d_blind = (out.float().cpu() - ref_full).abs().max().item()
    rec = _gated(
        "exact", name, f"win{W}", out, ref_win,
        extra=f" d_blind={d_blind:.2e}", floor=STOCK_D.get(name, 0.0),
    )
    if rec is not None and d_blind < 1e-3:
        _fail("exact", name, f"win{W}: WINDOW-BLIND (d_blind={d_blind:.2e})")
        rec = None
    if rec is not None:
        rec["d_window_blind"] = d_blind
        _rec("exact", name, **{f"win{W}": rec})


def face_nonbinding(c, lens=(50, 200), W=1024, seed=1234):
    """n <= W: the window cannot bind — windowed must gate vs the
    UNMASKED ref and agree with the windowless twin."""
    name = cell_name(c)
    print(f"[face: nonbinding/{name}] lens <= W={W}: windowed vs windowless twin")
    ls = list(lens)
    q, k, v = _prep(c, ls, seed)
    out_w = varlen_call(c, q, k, v, ls, window=W)
    out_0 = varlen_call(c, q, k, v, ls)
    torch.xpu.synchronize()
    ref = _ref_prefill(c, q, k, v, ls)
    rec = _gated("nonbinding", name, f"win{W}", out_w, ref, other=out_0)
    if rec:
        _rec("nonbinding", name, **{f"win{W}": rec})


def face_boundary(c, seed=77):
    """Window/length edges: W=64 at 63/64/65; W=1024 at 1023/1024/1025."""
    name = cell_name(c)
    print(f"[face: boundary/{name}] window-edge lens")
    for W, lens in ((64, [63, 64, 65]), (1024, [1023, 1024, 1025])):
        q, k, v = _prep(c, lens, seed + W)
        out = varlen_call(c, q, k, v, lens, window=W)
        torch.xpu.synchronize()
        ref = _ref_prefill(c, q, k, v, lens, window=W)
        rec = _gated(
            "boundary", name, f"win{W}", out, ref,
            floor=STOCK_D.get(name, 0.0),
        )
        if rec:
            _rec("boundary", name, **{f"win{W}": rec})


def face_telemetry(n=8192, W=1024, iters=20):
    """Informational: windowed vs windowless varlen cost at ctx 8192."""
    c = CELLS["d256"]
    name = cell_name(c)
    print(f"[face: telemetry/{name}] ctx {n}: windowed W={W} vs windowless")
    lens = [n]
    q, k, v = _prep(c, lens, 4242)

    def _bench(window):
        for _ in range(5):
            varlen_call(c, q, k, v, lens, window=window)
        torch.xpu.synchronize()
        times = []
        for _ in range(iters):
            s = torch.xpu.Event(enable_timing=True)
            e = torch.xpu.Event(enable_timing=True)
            s.record()
            varlen_call(c, q, k, v, lens, window=window)
            e.record()
            torch.xpu.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]

    t_w = _bench(W)
    t_0 = _bench(None)
    print(f"telemetry/{name}: win{W}={t_w:.3f}ms windowless={t_0:.3f}ms")
    _rec("telemetry", name, ctx=n, W=W, win_ms=t_w, windowless_ms=t_0)


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if _fa_varlen is None:
        print("VERDICT FAIL (no varlen op available)")
        R["_meta"]["verdict"] = "FAIL"
        with open(outdir / "p244_varlen.json", "w", newline="\n") as f:
            json.dump(R, f, indent=1)
        sys.exit(1)
    for c in CELLS.values():
        face_stock(c)
        face_exact(c)
        face_nonbinding(c)
        face_boundary(c)
    face_telemetry()
    ok = not FAILED
    verdict = "PASS" if ok else "FAIL"
    print(f"VERDICT {verdict} ({len(FAILED)} failures)")
    R["_meta"]["verdict"] = verdict
    R["_meta"]["failures"] = FAILED
    with open(outdir / "p244_varlen.json", "w", newline="\n") as f:
        json.dump(R, f, indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
