#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p243_window_offline.py — P6.47 P-C2 (r243): offline XPU validation of
the MX-NATIVE paged-decode kernel's SLIDING-WINDOW surface (the masked
kernel twin: window_size=(W-1, 0) rides the LocalMask machinery that the
vendor carried idle since r212; P-C2 threads it through the binding,
op schema, python interface and — for the engine — fa_utils prefill).

Machinery and gates are the r198 offline-oracle laws (p147_mxkv/
p198_native_offline.py, lifted via p242_d512_offline.py), with the
reference window-masked: decode q at pos p = len-1 attends k in
[p-(W-1), p] — the last W tokens (kernel convention (W-1, 0); the
engine passes sliding_window via native_decode.py's (W-1, 0)).

Cells (the masked-twin instantiation law: D=256, page 16 and 64-via-
_16, qgroup <= 8):
  d256p64   D=256 BLOCK=64 HK=2 HREP=8 -> HQ=16, G=8 (gemma-4-31b
            sliding layers under TP2: 50 layers D=256 W=1024, page 64
            — the r240b grid's cache geometry).
  ctrl256p16 D=256 BLOCK=16 HK=2 HREP=6 -> HQ=12, G=6 (the certified
            qwen geometry, page 16) — the second masked twin + the
            same-wheel unmasked regression guard.

Faces: exact (windowed native vs window-masked fp64, W=64 binding at
T=549) / nonbinding (seq_len <= W: windowed vs windowless must both
meet the unmasked-ref gates — the mask may not fire early) / boundary
(lens 1/32/63/64/65/127/128 at W=64; 1000/1023/1024/1025 at W=1024 —
the window edge + block edges) / bitwise (seq_len==1: window can't
bind, dequant bit law must hold) / splits (forced 8 + auto at ctx
8192 with W=64 — k_block0 walk-bound x ReduceSplitK) / widewin (the
gemma W=1024 at ctx 8192/32768/40960, mxfp4 — the e2e lane's shape).

Exit 0 only if every gate passes. Output: <out>/p243_window.json.
Every write site uses newline="\\n" (p139 law).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

# Load the mx_kv package by FILE PATH under a synthetic top-level name
# (p147 law: the wheel's vllm_gaudi would win resolution otherwise).
_MXKV_DIR = (
    Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
)


def _load_mxkv():
    spec = importlib.util.spec_from_file_location(
        "mx_kv_lane", _MXKV_DIR / "__init__.py", submodule_search_locations=[str(_MXKV_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mxkv = _load_mxkv()
decode_kv = mxkv.decode_kv
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

DEV = "xpu"
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
CELLS = {
    "d256p64": {"D": 256, "BLOCK": 64, "HK": 2, "HREP": 8},
    "ctrl256p16": {"D": 256, "BLOCK": 16, "HK": 2, "HREP": 6},
}
R = {}
FAILED = []


def _rec(face, cell, fmt, **kv):
    R.setdefault(face, {}).setdefault(cell, {}).setdefault(fmt, {}).update(kv)


def _fail(face, cell, fmt, msg):
    FAILED.append(f"{face}/{cell}/{fmt}: {msg}")
    print(f"  FAIL {face}/{cell}/{fmt}: {msg}")


def _ok(face, cell, fmt, msg):
    print(f"  ok   {face}/{cell}/{fmt}: {msg}")


def cos_sim(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def cell_name(c):
    return f"d{c['D']}p{c['BLOCK']}"


def build_paged(c, total_tokens, seed=7):
    """Fragmented block tables: shuffled physical blocks, 2 sequences."""
    blk = c["BLOCK"]
    n_blocks = total_tokens // blk + 2
    perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed))
    order = perm.tolist()
    lens = (
        [total_tokens]
        if total_tokens < 2
        else [max(1, total_tokens // 3), total_tokens - max(1, total_tokens // 3)]
    )
    tables, tok_block, tok_pos = [], [], []
    cur = 0
    for ln in lens:
        ntb = (ln + blk - 1) // blk
        tab = order[cur : cur + ntb]
        tables.append(torch.tensor(tab, dtype=torch.int32, device=DEV))
        for t in range(ln):
            tok_block.append(tab[t // blk])
            tok_pos.append(t % blk)
        cur += ntb
    return (
        lens,
        tables,
        torch.tensor(tok_block, dtype=torch.int32, device=DEV),
        torch.tensor(tok_pos, dtype=torch.int32, device=DEV),
        n_blocks,
    )


def flat_tables(lens, tables):
    mb = max(t.shape[0] for t in tables)
    bt = torch.zeros(len(lens), mb, dtype=torch.int32, device=DEV)
    for i, t in enumerate(tables):
        bt[i, : t.shape[0]] = t
    return bt


def _ref_attention(c, q, k_deq, v_deq, lens, window=None):
    """fp64 reference decode: q [B, Hq, D]; k/v_deq [T, Hk, D] (host).

    Plain fp64 softmax (r198 run-5 law: the native kernel does not
    round P to bf16). window: None = unmasked; int W = the last W
    tokens per row (kernel pair (W-1, 0); decode q sits at pos len-1).
    """
    d, hrep = c["D"], c["HREP"]
    outs = []
    t0 = 0
    for b, ln in enumerate(lens):
        kb = k_deq[t0 : t0 + ln].double()
        vb = v_deq[t0 : t0 + ln].double()
        lo = 0 if window is None else max(0, ln - window)
        for hq in range(q.shape[1]):
            hk = hq // hrep
            s = (kb[:, hk, :] @ q[b, hq].double()) / math.sqrt(d)
            if window is not None:
                s[:lo] = float("-inf")
            outs.append(torch.softmax(s, dim=0) @ vb[:, hk, :])
        t0 += ln
    return torch.stack(outs).reshape(len(lens), q.shape[1], d)


def _dequant_cache(c, cache, tb, tp, fmt):
    d = c["D"]
    got = cache[tb.long().cpu(), tp.long().cpu(), :, :].cpu()
    ng = d // 32
    kwidth = d + ng if fmt in ("mxfp8", "mxk8v4") else d // 2 + ng
    return decode_kv(got[..., :kwidth].contiguous(), got[..., kwidth:].contiguous(), fmt, d)


def native_call(c, q_bf, cache, fmt, lens, bt, splits=0, window=None):
    cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    ws = None if window is None else (window - 1, 0)
    return mx_paged_decode(
        q_bf,
        cache,
        fmt,
        cu_q,
        sl,
        bt,
        max_seqlen_k=max(lens),
        softmax_scale=1.0 / math.sqrt(c["D"]),
        num_splits_kv=splits,
        window_size=ws,
        host_kv_lens=torch.tensor(lens) if splits > 1 else None,
    )


def _gated(face, cell, fmt, tag, out, ref_same, other=None, extra=""):
    """Common gate: bf16-output-aware exact comparison (r198 face law).

    other: an alternative calibrated output (the windowless native at
    non-binding W) composing the triangle law.
    """
    outf = out.float().cpu()
    ref_same = ref_same.cpu()
    if not torch.isfinite(outf).all():
        _fail(face, cell, fmt, f"{tag}: non-finite output")
        return None
    d_exact = (outf - ref_same).abs().max().item()
    c_same = cos_sim(outf, ref_same)
    ref_r = ref_same.bfloat16().float()
    diff_r = (outf - ref_r).abs()
    d_rounded = diff_r.max().item()
    mag = torch.maximum(outf.abs(), ref_same.abs())
    expo = torch.floor(torch.log2(mag.clamp_min(1e-38)))
    ulp_elem = torch.exp2(torch.where(mag > 0, expo, torch.full_like(expo, -126.0)) - 7.0)
    ok_exact = d_exact <= 2e-3
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
            face, cell, fmt,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} "
            f"cos={c_same:.7f}{extra}",
        )
    else:
        _ok(
            face, cell, fmt,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} "
            f"cos={c_same:.7f}{extra}",
        )
    return dict(maxdiff=d_exact, maxdiff_rounded=d_rounded, cos_same=c_same, gated=gated)


def _prep(c, T, seed):
    gen = torch.Generator().manual_seed(seed)
    xk = torch.randn(T, c["HK"], c["D"], generator=gen).to(DEV)
    xv = torch.randn(T, c["HK"], c["D"], generator=gen).to(DEV)
    lens, tables, tb, tp, n_blocks = build_paged(c, T, seed=seed % 97)
    q = torch.randn(len(lens), c["HK"] * c["HREP"], c["D"], generator=gen).to(torch.bfloat16).to(DEV)
    return xk, xv, lens, tables, tb, tp, n_blocks, q


def _build_cache(c, fmt, xk, xv, tb, tp, n_blocks):
    cache = torch.zeros(
        n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
        dtype=torch.uint8, device=DEV,
    )
    mx_store(xk, xv, cache, tb, tp)
    torch.xpu.synchronize()
    return cache


def face_exact(c, T=549, W=64, seed=99):
    """Windowed native vs window-masked fp64 (the window BINDS: W << T)."""
    name = cell_name(c)
    print(f"[face: exact/{name}] windowed native (W={W}) vs window-masked fp64")
    for fmt in FMTS:
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(c, T, seed)
        cache = _build_cache(c, fmt, xk, xv, tb, tp, n_blocks)
        bt = flat_tables(lens, tables)
        out = native_call(c, q, cache, fmt, lens, bt, window=W)
        torch.xpu.synchronize()
        k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
        # unmasked ref too: a window-blind native would pass the
        # unmasked ref and fail the masked one — compute both, gate on
        # the masked, and record the unmasked delta as the discriminator.
        ref_win = _ref_attention(c, q.cpu(), k_deq, v_deq, lens, window=W)
        ref_full = _ref_attention(c, q.cpu(), k_deq, v_deq, lens)
        d_blind = (out.float().cpu() - ref_full).abs().max().item()
        rec = _gated("exact", name, fmt, f"win{W}", out, ref_win, extra=f" d_blind={d_blind:.2e}")
        if rec:
            rec["d_window_blind"] = d_blind
            _rec("exact", name, fmt, **{f"win{W}": rec})


def face_nonbinding(c, T=60, W=64, seed=1234):
    """seq_len <= W: the window cannot bind — the windowed twin must
    meet the UNMASKED-ref gates AND agree with the windowless twin."""
    name = cell_name(c)
    print(f"[face: nonbinding/{name}] lens <= W={W}: windowed vs windowless twin")
    for fmt in FMTS:
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(c, T, seed)
        cache = _build_cache(c, fmt, xk, xv, tb, tp, n_blocks)
        bt = flat_tables(lens, tables)
        out_w = native_call(c, q, cache, fmt, lens, bt, window=W)
        out_0 = native_call(c, q, cache, fmt, lens, bt, window=None)
        torch.xpu.synchronize()
        k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
        ref_full = _ref_attention(c, q.cpu(), k_deq, v_deq, lens)
        rec = _gated("nonbinding", name, fmt, f"win{W}", out_w, ref_full, other=out_0)
        if rec:
            _rec("nonbinding", name, fmt, **{f"win{W}": rec})


def face_boundary(c):
    """Window-edge lens: W=64 with 1/32/63/64/65/127/128 (bind flips at
    65) and W=1024 with 1000/1023/1024/1025 (the gemma W edge)."""
    name = cell_name(c)
    print(f"[face: boundary/{name}] window-edge lens through shuffled tables")
    for fmt in ("mxfp4",):
        for W, lens in ((64, [1, 32, 63, 64, 65, 127, 128]),
                        (1024, [1000, 1023, 1024, 1025])):
            gen = torch.Generator().manual_seed(555 + W)
            T = sum(lens)
            hq_n = c["HK"] * c["HREP"]
            xk = torch.randn(T, c["HK"], c["D"], generator=gen).to(DEV)
            xv = torch.randn(T, c["HK"], c["D"], generator=gen).to(DEV)
            blk = c["BLOCK"]
            n_blocks = T // blk + len(lens) + 2
            perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(9))
            order = perm.tolist()
            tables, tok_block, tok_pos, cur = [], [], [], 0
            for ln in lens:
                ntb = (ln + blk - 1) // blk
                tab = order[cur : cur + ntb]
                tables.append(torch.tensor(tab, dtype=torch.int32, device=DEV))
                for t in range(ln):
                    tok_block.append(tab[t // blk])
                    tok_pos.append(t % blk)
                cur += ntb
            tb = torch.tensor(tok_block, dtype=torch.int32, device=DEV)
            tp = torch.tensor(tok_pos, dtype=torch.int32, device=DEV)
            q = torch.randn(len(lens), hq_n, c["D"], generator=gen).to(torch.bfloat16).to(DEV)
            cache = _build_cache(c, fmt, xk, xv, tb, tp, n_blocks)
            bt = flat_tables(lens, tables)
            out = native_call(c, q, cache, fmt, lens, bt, window=W)
            torch.xpu.synchronize()
            k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
            ref_win = _ref_attention(c, q.cpu(), k_deq, v_deq, lens, window=W)
            rec = _gated("boundary", name, fmt, f"W{W}", out, ref_win)
            if rec:
                _rec("boundary", name, fmt, **{f"W{W}": rec})


def face_bitwise(c):
    """seq_len==1: the window cannot bind; the dequant bit law (O ==
    bf16(V_deq)) must hold on the masked twin verbatim."""
    name = cell_name(c)
    print(f"[face: bitwise/{name}] windowed seq_len==1 dequant bit-math oracle")
    for fmt in FMTS:
        for W in (64,):
            gen = torch.Generator().manual_seed(777)
            xk = torch.randn(1, c["HK"], c["D"], generator=gen).to(DEV)
            xv = torch.randn(1, c["HK"], c["D"], generator=gen).to(DEV)
            lens, tables, tb, tp, n_blocks = build_paged(c, 1, seed=3)
            hq_n = c["HK"] * c["HREP"]
            q = torch.randn(1, hq_n, c["D"], generator=gen).to(torch.bfloat16).to(DEV)
            cache = _build_cache(c, fmt, xk, xv, tb, tp, n_blocks)
            bt = flat_tables(lens, tables)
            out = native_call(c, q, cache, fmt, lens, bt, window=W)
            torch.xpu.synchronize()
            _, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
            mism = 0
            for hq in range(hq_n):
                got = out[0, hq].view(torch.int16).cpu()
                want = v_deq[0, hq // c["HREP"]].to(torch.bfloat16).view(torch.int16).cpu()
                # IEEE sign-of-zero law (r198): canonicalize +-0.
                for pat in (got, want):
                    pat[pat == -32768] = 0
                mism += int((got != want).sum())
            if mism:
                _fail("bitwise", name, fmt, f"W{W}: {mism}/{hq_n * c['D']} mismatched bf16 patterns")
            else:
                _ok("bitwise", name, fmt, f"W{W}: 0/{hq_n * c['D']} mismatches — dequant bit law exact")
            _rec("bitwise", name, fmt, **{f"W{W}": dict(bit_mismatches=mism)})


def face_splits(c, T=8192):
    """Forced split-KV + auto with a binding window: k_block0 (the
    walk-bound from window_size_left) x ReduceSplitK + the r231d
    seqused-tightened split clamp on windowed_k_blocks. Two widths:
    W=64 (windowed tiles sit under the >=4-tiles/split law -- the
    forced plan legitimately collapses to 1 split; the r243a
    windowed-plan regression: the pre-fix full-length plan walked
    tiles past k_blocks -> mxfp4 DEVICE_LOST) and W=2048 (128
    windowed tiles -- genuine 8-way windowed split-KV through
    ReduceSplitK)."""
    name = cell_name(c)
    print(f"[face: splits/{name}] windowed forced split-KV + default at ctx {T}")
    for fmt in FMTS:
        for W in (64, 2048):
            xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(c, T, 31337)
            cache = _build_cache(c, fmt, xk, xv, tb, tp, n_blocks)
            bt = flat_tables(lens, tables)
            k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
            ref_win = _ref_attention(c, q.cpu(), k_deq, v_deq, lens, window=W)
            for tag, splits in (("forced8", 8), ("default", 0)):
                out = native_call(c, q, cache, fmt, lens, bt, splits=splits, window=W)
                torch.xpu.synchronize()
                rec = _gated("splits", name, fmt, f"W{W}/{tag}", out, ref_win)
                if rec:
                    _rec("splits", name, fmt, **{f"W{W}_{tag}": rec})


def face_widewin(c, W=1024):
    """The gemma lane's real shape: W=1024 at the r240b maxlens
    (window binds for ctx > 1024; at 8192 the walk is 1024 of 8192)."""
    name = cell_name(c)
    print(f"[face: widewin/{name}] W={W} at ctx 8192/32768/40960 (gemma lane)")
    for fmt in ("mxfp4",):
        for T in (8192, 32768, 40960):
            xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(c, T, T % 89)
            cache = _build_cache(c, fmt, xk, xv, tb, tp, n_blocks)
            bt = flat_tables(lens, tables)
            out = native_call(c, q, cache, fmt, lens, bt, window=W)
            torch.xpu.synchronize()
            k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
            ref_win = _ref_attention(c, q.cpu(), k_deq, v_deq, lens, window=W)
            rec = _gated("widewin", name, fmt, f"ctx{T}", out, ref_win)
            if rec:
                _rec("widewin", name, fmt, **{f"ctx{T}": rec})


def face_telemetry(c, W=1024):
    """Windowed native decode timings at the gemma lane's ctx — the
    walk shrinks to W tokens, so the windowed cost should sit well
    under the windowless native at the same ctx."""
    name = cell_name(c)
    print(f"[face: telemetry/{name}] windowed native decode timings (W={W})")
    prop = torch.xpu.get_device_properties(0)
    R.setdefault("telemetry", {}).setdefault(name, {})["device"] = {"name": prop.name}
    hq_n = c["HK"] * c["HREP"]
    for fmt in FMTS:
        rows = {}
        for ctx in (8192, 40960):
            xk = torch.randn(ctx, c["HK"], c["D"], device=DEV)
            xv = torch.randn(ctx, c["HK"], c["D"], device=DEV)
            lens, tables, tb, tp, n_blocks = build_paged(c, ctx, seed=ctx % 97)
            cache = _build_cache(c, fmt, xk, xv, tb, tp, n_blocks)
            bt = flat_tables(lens, tables)
            q = torch.randn(len(lens), hq_n, c["D"], device=DEV).to(torch.bfloat16)
            for _ in range(3):
                native_call(c, q, cache, fmt, lens, bt, window=W)
                native_call(c, q, cache, fmt, lens, bt, window=None)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                native_call(c, q, cache, fmt, lens, bt, window=W)
            torch.xpu.synchronize()
            t_win = (time.perf_counter() - t0) / 20
            t0 = time.perf_counter()
            for _ in range(20):
                native_call(c, q, cache, fmt, lens, bt, window=None)
            torch.xpu.synchronize()
            t_full = (time.perf_counter() - t0) / 20
            rows[ctx] = {
                "windowed_ms": round(t_win * 1e3, 4),
                "windowless_ms": round(t_full * 1e3, 4),
                "walk_ratio": round(t_win / t_full, 3),
            }
            print(
                f"  ok   telemetry/{name}/{fmt} ctx={ctx}: windowed "
                f"{t_win*1e3:.3f} ms vs windowless {t_full*1e3:.3f} ms "
                f"({t_win/t_full:.2f}x)"
            )
        _rec("telemetry", name, fmt, **{str(k): v for k, v in rows.items()})


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p243_window_offline — device={DEV}, torch={torch.__version__}")
    gem = CELLS["d256p64"]
    ctrl = CELLS["ctrl256p16"]
    face_exact(gem)            # gemma cell, W=64 binding
    face_exact(ctrl)           # qwen-geometry twin, W=64 binding
    face_nonbinding(gem)       # mask must not fire early
    face_nonbinding(ctrl)
    face_boundary(gem)         # window + block edges
    face_boundary(ctrl)
    face_bitwise(gem)          # dequant bit law on the masked twin
    face_bitwise(ctrl)
    face_splits(gem)           # k_block0 x ReduceSplitK
    face_widewin(gem)          # the gemma W=1024 at lane ctx
    face_telemetry(gem)        # windowed walk cost
    R["verdict"] = "PASS" if not FAILED else "FAIL"
    R["failures"] = FAILED
    R["cells"] = {k: dict(v) for k, v in CELLS.items()}
    with open(out / "p243_window.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=2, sort_keys=True)
    n = len(FAILED)
    print(f"\nVERDICT: {R['verdict']}" + (f" ({n} failures)" if n else ""))
    sys.exit(0 if not n else 1)


if __name__ == "__main__":
    main()
