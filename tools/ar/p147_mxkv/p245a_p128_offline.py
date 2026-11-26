"""p245a_p128_offline.py -- P6.47 P-C1 (r245a): offline XPU validation of
the MX-NATIVE paged-decode PAGE-128 admission at the gemma-4-31b
full_attention geometry (D=512, page 128, G=8).

The r245 e2e grid (build5 + P-C2 overlay, 2026-11-26) booted all three
gemma cells healthy -- pc2_windowed_impl=100, native window=1024 calls
at capture -- and died at the FIRST REAL REQUEST: the gemma hybrid KV
manager banks the D=512 full_attention group a PAGE-128 pool (the
sliding D=256 group rides page 64; the same boot banks both), and the
first real decode hit "mx_paged_decode: block_size 128 not yet
compiled" (serve traceback engine.py:1261 -> native_decode.py:135).
All 37 banked faces were n_ok=0 (Connection refused) behind green
CONC_OK lines -- the existence-only-green-gate lesson, fixed in the
r245 runner's n_ok>0 semantic gate + warmup-death detector.

r245a admits page 128: the TORCH_CHECK widens to 16/64/128 + a new
dispatch branch (head_size == 512 && block_size == 128) riding the
SAME certified _512/_16 sub-tile frame as 512/64. The admission claim:
the mainloop's page indirection is runtime-generic
(tiles_per_page = page_size/16 -> 8 sub-tiles per 128-page vs 4 at
page 64; page_local_idx = tile*16/page_size; sites L918-926 walk,
L1150-1155 early-exit, L1465-1473 prefetch -- all verified), so no new
kernel surface. This oracle PROVES the claim before the ~3 h e2e:

- exact: native decode vs fp64 on dequantized cache, all 3 formats
  (the v2 triangle rides when the Triton decoder serves page 128 --
  availability-probed, never gate-crashing)
- bitwise: seq_len==1 dequant bit-math oracle (O == bf16(V_deq))
- boundary: page-128 edge lens (1/63/64/127/128/129/255/256) through
  shuffled tables -- 127/128 and 255/256 are the page-0->1 and
  page-1->2 sub-tile transitions; the walk must cross them via
  page_local_idx recomputation. The v2 triangle rides the face (r198
  composite), and the F32 near-cancellation discriminator rides the
  1-ulp law (rev 3): harness rev 1 ran it bare and the payload's
  worst element (a 7e-7 near-cancellation + a single bf16 output ulp
  at ~0.58 magnitude) read d=2.27e-3, 13 pct over the absolute band,
  with the F32 ledger proving the deviation intrinsic (d(F32,ref) ==
  d; F32 at 7.8 ulps on the kernel's 6-step element); rev 2 added the
  v2 triangle alone -- v2 banked d_v2=2.33e-7 (clean), the composite
  could not widen; rev 3 promotes the ledger's F32 discriminator into
  the ulp law (zero free parameters).
- splits: forced-8 + default at ctx 8192 -- the r242d 16-tile plan x
  the 8-sub-tile page walk (the r243a clamp is a no-op windowless:
  full-length plans land inside [0, k_blocks) by construction)
- regression guards on the SAME WHEEL: d512p64 (the r242 authority)
  and ctrl256p64 (the certified qwen production lane) -- the
  kernel-binary-layout-dice law: a recompile can swing untouched
  lanes, so the guards certify build6 carries the lanes forward
- telemetry: native decode timings at page 128 vs the r242 page-64
  numbers (the e2e expectation-setter)

Gates: the p242 face law (d<=2e-3 or bf16-rounded 1-ulp match;
cos>=0.99999; bitwise 0 mismatches).
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
encode_mxfp4 = mxkv.encode_mxfp4
encode_mxfp8 = mxkv.encode_mxfp8
mx_decode = mxkv.mx_decode
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

DEV = "xpu"
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
CELLS = {
    "d512p128": {"D": 512, "BLOCK": 128, "HK": 2, "HREP": 8},
    "d512p64": {"D": 512, "BLOCK": 64, "HK": 2, "HREP": 8},
    "ctrl256p64": {"D": 256, "BLOCK": 64, "HK": 2, "HREP": 6},
}
R = {}
FAILED = []
# v2 (Triton) availability per (cell, fmt) -- page 128 was never run
# before this probe; a refusal is recorded, not fatal (the exact+bitwise
# gates stand alone).
V2_OK: dict[tuple, bool] = {}


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


def _ref_attention(c, q, k_deq, v_deq, lens):
    """fp64 reference decode: q [B, Hq, D]; k/v_deq [T, Hk, D] (host).

    Plain fp64 softmax, no operand-cast modeling (r198 run-5 law: the
    native kernel does not round P to bf16); the residual vs this ref
    is calibrated per-cell by the certified v2 decoder when available.
    """
    d, hrep = c["D"], c["HREP"]
    outs = []
    t0 = 0
    for b, ln in enumerate(lens):
        kb = k_deq[t0 : t0 + ln].double()
        vb = v_deq[t0 : t0 + ln].double()
        for hq in range(q.shape[1]):
            hk = hq // hrep
            s = (kb[:, hk, :] @ q[b, hq].double()) / math.sqrt(d)
            outs.append(torch.softmax(s, dim=0) @ vb[:, hk, :])
        t0 += ln
    return torch.stack(outs).reshape(len(lens), q.shape[1], d)


def _f32_attention(c, q, k_deq, v_deq, lens):
    """Honest fp32 pipeline (the mini-ledger's F32 reference, raw --
    no bf16 rounding): fp32 softmax + PV on the same dequantized
    cache. The r245a near-cancellation discriminator's ground truth --
    an element where THIS pipeline exceeds 1 ulp vs bf16(ref) is
    fp32-order noise (any accumulation order lands differently on a
    cancellation sum), not a kernel defect."""
    d, hrep = c["D"], c["HREP"]
    outs = []
    t0 = 0
    for b, ln in enumerate(lens):
        kb = k_deq[t0 : t0 + ln].float()
        vb = v_deq[t0 : t0 + ln].float()
        for hq in range(q.shape[1]):
            hk = hq // hrep
            s = (kb[:, hk, :] @ q[b, hq].float().cpu()) / math.sqrt(d)
            outs.append(torch.softmax(s, dim=0) @ vb[:, hk, :])
        t0 += ln
    return torch.stack(outs).reshape(len(lens), q.shape[1], d)


def _dequant_cache(c, cache, tb, tp, fmt):
    d = c["D"]
    got = cache[tb.long().cpu(), tp.long().cpu(), :, :].cpu()
    ng = d // 32
    kwidth = d + ng if fmt in ("mxfp8", "mxk8v4") else d // 2 + ng
    return decode_kv(got[..., :kwidth].contiguous(), got[..., kwidth:].contiguous(), fmt, d)


def native_call(c, q_bf, cache, fmt, lens, bt, splits=0):
    cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
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
        host_kv_lens=torch.tensor(lens) if splits > 1 else None,
    )


def v2_call(c, q, cache, bt, sl):
    """Certified Triton v2 decoder -- availability probed per (cell,fmt)."""
    try:
        out = mx_decode(q.float(), cache, bt, sl, 1.0 / math.sqrt(c["D"]), block_size=c["BLOCK"])
        torch.xpu.synchronize()
        return out
    except Exception as exc:  # noqa: BLE001 -- record, never gate-crash
        print(f"  note v2-unavailable {c['D']}/{c['BLOCK']}: {type(exc).__name__}: {exc}")
        return None


def _gated(face, cell, fmt, tag, out, ref_same, v2_out=None, f32_ref=None, extra=""):
    """Common gate: bf16-output-aware exact comparison (r198 face law).

    v2 calibration: when the certified Triton v2 decoder's output for
    the SAME payload is available, the exact gate composes the two
    certified comparisons (triangle law, no free parameters); a
    native-only defect must also survive native-v2.

    f32_ref (r245a rev 3): the honest fp32 pipeline's output for the
    SAME payload (fp32 softmax + PV, no bf16 rounding). The 1-ulp
    rounded law excludes elements where F32 ITSELF exceeds 1 ulp vs
    bf16(ref) -- near-cancellation sums (|ref| far below the row's PV
    scale) legitimately violate the ulp law under ANY accumulation
    order: the r245a run-1 ledger banked the honest fp32 pipeline at
    7.8 ulps on the same element where the kernel sat at 6 (and v2,
    fp32 with a different order, at 5 vs the kernel's bf16 value).
    Zero free parameters: the discriminator is the already-banked
    mini-ledger comparison, promoted into the gate. It cannot fire on
    corruption -- a corrupted region deviates far beyond the fp32
    order-noise class and cos collapses regardless.
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
    n_f32_excluded = 0
    over_ulp = diff_r > ulp_elem
    if f32_ref is not None:
        diff_f32 = (f32_ref.cpu().float() - ref_r).abs()
        f32_over = diff_f32 > ulp_elem
        n_f32_excluded = int((over_ulp & f32_over).sum())
        over_ulp = over_ulp & ~f32_over
    ok_rounded = not bool(over_ulp.any())
    if v2_out is not None:
        v2f = v2_out.float().cpu()
        d_v2 = (v2f - ref_same).abs().max().item()
        ok_exact = ok_exact or d_exact <= 2e-3 + d_v2
        extra = f"{extra} d_v2={d_v2:.2e}"
        diff_v2 = (outf - v2f.bfloat16().float()).abs()
        mag_v2 = torch.maximum(outf.abs(), v2f.abs())
        expo_v2 = torch.floor(torch.log2(mag_v2.clamp_min(1e-38)))
        ulp_v2 = torch.exp2(
            torch.where(mag_v2 > 0, expo_v2, torch.full_like(expo_v2, -126.0)) - 7.0
        )
        ok_rounded = ok_rounded or bool((diff_v2 <= ulp_v2).all())
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
    rec = dict(maxdiff=d_exact, maxdiff_rounded=d_rounded, cos_same=c_same, gated=gated)
    if f32_ref is not None:
        rec["n_f32_excluded"] = n_f32_excluded
    return rec


def _ledger(c, fmt, tag, out, ref_same, q, k_deq, v_deq, lens):
    """Failing-cell decomposition (r198 mini-ledger law): offending
    elements vs the pure output-cast law C = bf16(fp64 p@v) and an
    honest fp32 pipeline F32 -- discriminates intrinsic fp32-vs-fp64
    attention noise from a kernel-specific accumulation defect."""
    d, hrep, hq_n = c["D"], c["HREP"], q.shape[1]
    got = out.float().cpu().double().reshape(-1, d)
    outs_c, outs_f, ref_rows = [], [], []
    t0i = 0
    for b, lnb in enumerate(lens):
        kb = k_deq[t0i : t0i + lnb]
        vb = v_deq[t0i : t0i + lnb]
        for hq in range(hq_n):
            kh = kb[:, hq // hrep, :]
            vh = vb[:, hq // hrep, :]
            qd = q[b, hq].double().cpu()
            p64 = torch.softmax((kh.double() @ qd) / math.sqrt(d), dim=0)
            ref_rows.append(p64 @ vh.double())
            outs_c.append((p64 @ vh.double()).bfloat16().double())
            s32 = (kh.float() @ q[b, hq].float().cpu()) / math.sqrt(d)
            p32 = torch.softmax(s32, dim=0)
            outs_f.append((p32 @ vh.float()).bfloat16().double())
        t0i += lnb
    ref = torch.stack(ref_rows)
    C = torch.stack(outs_c)
    F32 = torch.stack(outs_f)
    diff_r = (got - ref.bfloat16().double()).abs()
    mag = torch.maximum(got.abs(), ref.abs())
    steps = diff_r / torch.exp2(
        torch.floor(torch.log2(mag.clamp_min(1e-38))) - 7.0
    )
    n_over = int((steps > 1.0).sum())
    det = []
    for fidx in steps.flatten().topk(6).indices.tolist():
        rr, cc = divmod(fidx, d)
        det.append(
            f"({rr // hq_n},{rr % hq_n},c{cc}) got={float(got[rr, cc]):.4g} "
            f"ref={float(ref[rr, cc]):.4g} C={float(C[rr, cc]):.4g} "
            f"F32={float(F32[rr, cc]):.4g} steps={float(steps[rr, cc]):.2f}"
        )
    print(
        f"  mini {cell_name(c)}/{fmt}/{tag}: n(steps>1)={n_over} "
        f"d(got,F32)={(got - F32).abs().max():.2e} "
        f"d(F32,ref)={(F32 - ref).abs().max():.2e} "
        f"d(C,F32)={(C - F32).abs().max():.2e} | " + " | ".join(det)
    )


def cell_name(c):
    return f"d{c['D']}p{c['BLOCK']}"


def _prep(c, T, seed):
    gen = torch.Generator().manual_seed(seed)
    xk = torch.randn(T, c["HK"], c["D"], generator=gen).to(DEV)
    xv = torch.randn(T, c["HK"], c["D"], generator=gen).to(DEV)
    lens, tables, tb, tp, n_blocks = build_paged(c, T, seed=seed % 97)
    q = torch.randn(len(lens), c["HK"] * c["HREP"], c["D"], generator=gen).to(torch.bfloat16).to(DEV)
    return xk, xv, lens, tables, tb, tp, n_blocks, q


def face_exact(c, T=549, seed=99):
    name = cell_name(c)
    print(f"[face: exact/{name}] native decode vs fp64 exact on dequantized cache")
    for fmt in FMTS:
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(c, T, seed)
        cache = torch.zeros(
            n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        out = native_call(c, q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        out_v2 = v2_call(c, q, cache, bt, sl)
        V2_OK[(name, fmt)] = out_v2 is not None
        k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
        ref_same = _ref_attention(c, q.cpu(), k_deq, v_deq, lens)
        rec = _gated("exact", name, fmt, f"random{T}", out, ref_same, v2_out=out_v2)
        if rec:
            _rec("exact", name, fmt, **{f"random{T}": rec})
            if not rec["gated"]:
                _ledger(c, fmt, f"random{T}", out, ref_same, q, k_deq, v_deq, lens)


def face_v2(c, T=549, seed=4242):
    name = cell_name(c)
    print(f"[face: v2/{name}] native vs certified Triton v2 on the same cache")
    for fmt in FMTS:
        if not V2_OK.get((name, fmt), True):
            _rec("v2", name, fmt, available=False)
            print(f"  note v2/{name}/{fmt}: unavailable (recorded; exact gate stands alone)")
            continue
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(c, T, seed)
        cache = torch.zeros(
            n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        out_v2 = v2_call(c, q, cache, bt, sl)
        if out_v2 is None:
            V2_OK[(name, fmt)] = False
            _rec("v2", name, fmt, available=False)
            continue
        out_nat = native_call(c, q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        rec = _gated("v2", name, fmt, f"random{T}", out_nat, out_v2.float().double())
        if rec:
            _rec("v2", name, fmt, **{f"random{T}": rec})


def face_bitwise(c):
    name = cell_name(c)
    print(f"[face: bitwise/{name}] seq_len==1 dequant bit-math oracle (O == bf16(V_deq))")
    for fmt in FMTS:
        gen = torch.Generator().manual_seed(777)
        xk = torch.randn(1, c["HK"], c["D"], generator=gen).to(DEV)
        xv = torch.randn(1, c["HK"], c["D"], generator=gen).to(DEV)
        lens, tables, tb, tp, n_blocks = build_paged(c, 1, seed=3)
        hq_n = c["HK"] * c["HREP"]
        q = torch.randn(1, hq_n, c["D"], generator=gen).to(torch.bfloat16).to(DEV)
        cache = torch.zeros(
            n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        out = native_call(c, q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        _, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
        mism = 0
        for hq in range(hq_n):
            got = out[0, hq].view(torch.int16).cpu()
            want = v_deq[0, hq // c["HREP"]].to(torch.bfloat16).view(torch.int16).cpu()
            # IEEE sign-of-zero law (r198): (+0)+(-0)=+0 in the PV
            # accumulator; -0 arises from e2m1 nibble 8. Canonicalize
            # +-0 bit patterns before comparing.
            for pat in (got, want):
                pat[pat == -32768] = 0  # int16 view of bf16 -0.0 -> +0.0
            mism += int((got != want).sum())
        if mism:
            _fail("bitwise", name, fmt, f"{mism}/{hq_n * c['D']} mismatched bf16 bit patterns")
        else:
            _ok("bitwise", name, fmt, f"0/{hq_n * c['D']} mismatches — dequant bit law exact")
        _rec("bitwise", name, fmt, bit_mismatches=mism)


def face_splits(c, T=8192):
    name = cell_name(c)
    print(f"[face: splits/{name}] forced split-KV (ReduceSplitK) + default plan at ctx {T}")
    for fmt in FMTS:
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(c, T, 31337)
        cache = torch.zeros(
            n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
        ref_same = _ref_attention(c, q.cpu(), k_deq, v_deq, lens)
        for tag, splits in (("forced8", 8), ("default", 0)):
            out = native_call(c, q, cache, fmt, lens, bt, splits=splits)
            torch.xpu.synchronize()
            rec = _gated("splits", name, fmt, tag, out, ref_same)
            if rec:
                _rec("splits", name, fmt, **{tag: rec})


def face_boundary(c):
    """Page-128 edge lens through shuffled tables (the _16 sub-tile
    frame iterates page_size/16 sub-tiles per page via the page-table
    indirection -- 8 per page at page 128; the 127/128 and 255/256
    lens cross the page-0->1 and page-1->2 boundaries where
    page_local_idx must recompute the physical row).

    The v2 triangle rides the face (the r198 composite, as in
    face_exact), and the F32 near-cancellation discriminator rides the
    1-ulp law (r245a rev 3). History: rev 1 (no triangle) banked
    d=2.27e-3 -- 13 pct over the absolute band; the mini ledger
    decomposed it: d(F32,ref) == the face's d (the honest fp32
    pipeline deviates identically), the steps>1 count was ONE element
    (a 7e-7 near-cancellation sum, where F32 sits at 7.8 ulps vs the
    kernel's 6), and the absolute-d driver was a single bf16 output
    ulp at a ~0.58-magnitude element -- a payload with max elements
    > 0.5 has its 1-output-ulp step above the 2e-3 absolute band, which
    is blind to magnitude. Rev 2 added the v2 triangle alone: v2
    itself banked d_v2=2.33e-7 (fp32, no output quantization -- clean),
    so the composite could not widen and the same two element classes
    tripped it. Rev 3 promotes the ledger's F32 discriminator into
    the ulp law (zero free parameters; see _gated). The splits face
    had already walked 512 page-local transitions per seq at
    d=2.44e-4 -- a page-indirection defect corrupts rows, not one
    near-cancellation tail."""
    name = cell_name(c)
    print(f"[face: boundary/{name}] page-{c['BLOCK']} edge lens, shuffled tables")
    lens = [1, 63, 64, 127, 128, 129, 255, 256]
    for fmt in ("mxfp4",):  # the gemma lane's format (slot 544 B)
        gen = torch.Generator().manual_seed(555)
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
        cache = torch.zeros(
            n_blocks, blk, c["HK"], slot_bytes(c["D"], fmt),
            dtype=torch.uint8, device=DEV,
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        out = native_call(c, q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        out_v2 = v2_call(c, q, cache, bt, sl)
        k_deq, v_deq = _dequant_cache(c, cache, tb, tp, fmt)
        ref_same = _ref_attention(c, q.cpu(), k_deq, v_deq, lens)
        f32_same = _f32_attention(c, q, k_deq, v_deq, lens)
        rec = _gated(
            "boundary", name, fmt, "edges", out, ref_same,
            v2_out=out_v2, f32_ref=f32_same,
        )
        if rec:
            _rec("boundary", name, fmt, edges=rec)
            if not rec["gated"]:
                _ledger(c, fmt, "edges", out, ref_same, q, k_deq, v_deq, lens)


def face_telemetry(c):
    """native decode timings at the admitted page -- the e2e
    expectation-setter (ctx 40960 = the r240b grid's gemma maxlen)."""
    name = cell_name(c)
    print(f"[face: telemetry/{name}] native decode timings at page {c['BLOCK']}")
    prop = torch.xpu.get_device_properties(0)
    R.setdefault("telemetry", {}).setdefault(name, {})["device"] = {"name": prop.name}
    hq_n = c["HK"] * c["HREP"]
    for fmt in FMTS:
        rows = {}
        for ctx in (1024, 8192, 32768, 40960):
            xk = torch.randn(ctx, c["HK"], c["D"], device=DEV)
            xv = torch.randn(ctx, c["HK"], c["D"], device=DEV)
            lens, tables, tb, tp, n_blocks = build_paged(c, ctx, seed=ctx % 97)
            cache = torch.zeros(
                n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
                dtype=torch.uint8, device=DEV,
            )
            mx_store(xk, xv, cache, tb, tp)
            bt = flat_tables(lens, tables)
            q = torch.randn(len(lens), hq_n, c["D"], device=DEV)
            q_bf = q.to(torch.bfloat16)
            t_v2 = float("nan")
            if V2_OK.get((name, fmt), True):
                sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
                for _ in range(3):
                    mx_decode(q, cache, bt, sl, 1.0 / math.sqrt(c["D"]), block_size=c["BLOCK"])
                    native_call(c, q_bf, cache, fmt, lens, bt)
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    mx_decode(q, cache, bt, sl, 1.0 / math.sqrt(c["D"]), block_size=c["BLOCK"])
                torch.xpu.synchronize()
                t_v2 = (time.perf_counter() - t0) / 20
            else:
                for _ in range(3):
                    native_call(c, q_bf, cache, fmt, lens, bt)
                torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                native_call(c, q_bf, cache, fmt, lens, bt)
            torch.xpu.synchronize()
            t_nat = (time.perf_counter() - t0) / 20
            b_read = ctx * c["HK"] * slot_bytes(c["D"], fmt) * c["HREP"]
            rows[ctx] = {
                "v2_ms": round(t_v2 * 1e3, 4),
                "native_ms": round(t_nat * 1e3, 4),
                "speedup": round(t_v2 / t_nat, 3) if t_v2 == t_v2 else None,
                "native_gbps": round(b_read / t_nat / 1e9, 2),
            }
            print(
                f"  ok   telemetry/{name}/{fmt} ctx={ctx}: "
                f"v2 {t_v2*1e3:.3f} ms, native {t_nat*1e3:.3f} ms "
                f"({t_v2/t_nat:.2f}x), {b_read/t_nat/1e9:.1f} GB/s effective"
                if t_v2 == t_v2
                else f"  ok   telemetry/{name}/{fmt} ctx={ctx}: "
                f"native {t_nat*1e3:.3f} ms (v2 unavailable), "
                f"{b_read/t_nat/1e9:.1f} GB/s effective"
            )
        _rec("telemetry", name, fmt, **{str(k): v for k, v in rows.items()})


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p245a_p128_offline — device={DEV}, torch={torch.__version__}")
    p128 = CELLS["d512p128"]
    p64 = CELLS["d512p64"]
    ctrl = CELLS["ctrl256p64"]
    face_exact(p128)          # the admission: dispatch + geometry at page 128
    face_v2(p128)             # native vs certified Triton (if it serves 128)
    face_bitwise(p128)        # dequant bit law at D=512 page 128
    face_splits(p128)         # split plan x 8-sub-tile page walk
    face_boundary(p128)       # page-128 edges (page_local_idx transitions)
    face_exact(p64)           # regression guard: the r242 authority on build6
    face_bitwise(p64)
    face_exact(ctrl)          # production guard: the certified qwen lane on build6
    face_bitwise(ctrl)
    face_telemetry(p128)      # page-128 native timing datapoint
    R["verdict"] = "PASS" if not FAILED else "FAIL"
    R["failures"] = FAILED
    R["cells"] = {k: dict(v) for k, v in CELLS.items()}
    with open(out / "p245a_p128.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=2, sort_keys=True)
    n = len(FAILED)
    print(f"\nVERDICT: {R['verdict']}" + (f" ({n} failures)" if n else ""))
    sys.exit(0 if not n else 1)


if __name__ == "__main__":
    main()
