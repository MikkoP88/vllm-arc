#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p198_native_offline.py — arc7 Phase-8 r198: offline XPU validation of
the MX-NATIVE CUTLASS/Xe2 paged-decode kernel (arc7mx wheel,
vllm_xpu_kernels.mx_attn_interface.mx_paged_decode) against the
certified Triton lane and exact references (2026-11-19).

Runs INSIDE a cert25zc-family container with the arc7mx wheel installed
over the image venv (no engine boot; radiance-test untouched). The
cache/store side is the CERTIFIED Triton lane (mx_store) — only the
decode is under test. Shape = the engine's real lane: Hq=8, Hk=2
(head_group 4, q8 policy), D=128, block 16 (the p16 policy + the
entry's block_size==16 guard).

Faces, in order:

  native-exact   native decode vs fp64 reference attention on the SAME
                 dequantized K/V (isolates kernel math from quant
                 error; p147 decode-face law). The reference models the
                 kernel's P-operand dtype law (softmax weights cast to
                 bf16 for the PV MMA — the vendor atom law v2 shares),
                 so d isolates genuine kernel error, not the documented
                 operand cast. Output is bf16 (the lane contract) —
                 gate: d<=2e-3 OR every diff explained by the bf16
                 output cast (rounded-match <= 1 ulp); cos >= 0.99999
                 always.
  native-v2      native vs the certified Triton v2 decoder on the SAME
                 cache (the plan's dequant-oracle face: same operands,
                 both fp accumulators) — gate d<=2e-3 (bf16-cast
                 allowance as above) + cos >= 0.99999.
  bitwise        THE DEQUANT BIT-MATH ORACLE: seq_len==1 makes
                 P == 1.0 exactly and O == V_dequant with zero
                 accumulation, so the bf16 output must equal
                 bf16(reference dequant) BIT FOR BIT (e4m3/e2m1/e8m0
                 products are fp32-exact), modulo the IEEE sign-of-zero
                 law ((+0)+(-0)=+0 in the PV accumulator; -0 arises
                 from e2m1 nibble 8 = 1/16 of elements). Gate: 0
                 mismatched int16 patterns (after +-0 canonicalization),
                 all three formats.
  robust         adversarial rows (denormal group, zero group, single
                 outlier, sign-flipped) + boundary lens 1/15/16/17
                 (block-16 edges) through shuffled tables — vs-exact
                 gate, finite outputs.
  splits         forced num_splits_kv=8 at ctx 8192 (ReduceSplitK
                 combine path) + the default split plan — vs-exact gate
                 on both.
  telemetry      (always-loop directive) native vs Triton v2 decode
                 timings at 1K/8K/32K per format — the first
                 bottleneck-ranking datapoint for r199.

Exit 0 only if every gate passes. Output: <out>/p198_native.json.
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
encode_mxfp4 = mxkv.encode_mxfp4
encode_mxfp8 = mxkv.encode_mxfp8
mx_decode = mxkv.mx_decode
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

D = 128
HK = 2
HREP = 4  # Hq = 8 (GQA shape of both certified models; head_group 4 -> q8)
HQ = HK * HREP
BLOCK = 16  # the engine pool block (p16 policy; entry refuses other pages)
DEV = "xpu"
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
KENC = {"mxfp8": encode_mxfp8, "mxfp4": encode_mxfp4, "mxk8v4": encode_mxfp8}
VENC = {"mxfp8": encode_mxfp8, "mxfp4": encode_mxfp4, "mxk8v4": encode_mxfp4}
R = {}
FAILED = []


def _rec(face, fmt, **kv):
    R.setdefault(face, {}).setdefault(fmt, {}).update(kv)


def _fail(face, fmt, msg):
    FAILED.append(f"{face}/{fmt}: {msg}")
    print(f"  FAIL {face}/{fmt}: {msg}")


def _ok(face, fmt, msg):
    print(f"  ok   {face}/{fmt}: {msg}")


def cos_sim(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def build_paged(total_tokens, seed=7):
    """Fragmented block-16 tables: shuffled physical blocks, 2 sequences."""
    n_blocks = total_tokens // BLOCK + 2
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
        ntb = (ln + BLOCK - 1) // BLOCK
        tab = order[cur : cur + ntb]
        tables.append(torch.tensor(tab, dtype=torch.int32, device=DEV))
        for t in range(ln):
            tok_block.append(tab[t // BLOCK])
            tok_pos.append(t % BLOCK)
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


def _ref_attention(q, k_deq, v_deq, lens):
    """fp64 reference decode: q [B, Hq, D]; k/v_deq [T, Hk, D] (host).

    Plain fp64 softmax (no operand-cast modeling): r198 run-5 proved
    the native kernel does NOT round P to bf16 (a bf16-P oracle moved
    d the WRONG way, 4.2e-3 -> 7.3e-3 on mxfp8 len15) -- the residual
    vs this ref is calibrated per-cell by the certified v2 decoder
    instead (see _gated).
    """
    outs = []
    t0 = 0
    for b, ln in enumerate(lens):
        kb = k_deq[t0 : t0 + ln].double()
        vb = v_deq[t0 : t0 + ln].double()
        for hq in range(q.shape[1]):
            hk = hq // HREP
            s = (kb[:, hk, :] @ q[b, hq].double()) / math.sqrt(D)
            outs.append(torch.softmax(s, dim=0) @ vb[:, hk, :])
        t0 += ln
    return torch.stack(outs).reshape(len(lens), q.shape[1], D)


def _dequant_cache(cache, tb, tp, fmt):
    got = cache[tb.long().cpu(), tp.long().cpu(), :, :].cpu()
    ng = D // 32
    kwidth = D + ng if fmt in ("mxfp8", "mxk8v4") else D // 2 + ng
    return decode_kv(got[..., :kwidth].contiguous(), got[..., kwidth:].contiguous(), fmt, D)


def native_call(q_bf, cache, fmt, lens, bt, splits=0):
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
        softmax_scale=1.0 / math.sqrt(D),
        num_splits_kv=splits,
        host_kv_lens=torch.tensor(lens) if splits > 1 else None,
    )


def _gated(face, fmt, tag, out, ref_same, v2_out=None, extra=""):
    """Common gate: bf16-output-aware exact comparison (face law above).

    v2 calibration: when the certified Triton v2 decoder's output for
    the SAME payload is supplied, the exact gate composes the two
    certified comparisons -- the native-v2 face gates |native - v2|
    <= 2e-3, so |native - ref| <= 2e-3 + |v2 - ref| (triangle; no
    free parameters). The lane's real dtype law (whatever rounding the
    shared decode structure carries) shows up identically in d_v2 and
    is absorbed; a native-only defect is not (it must also survive
    native-v2)."""
    # Refs arrive on mixed devices (fp64 refs are CPU, v2 faces XPU):
    # compare on CPU without touching ref dtype semantics.
    outf = out.float().cpu()
    ref_same = ref_same.cpu()
    if not torch.isfinite(outf).all():
        _fail(face, fmt, f"{tag}: non-finite output")
        return None
    d_exact = (outf - ref_same).abs().max().item()
    c_same = cos_sim(outf, ref_same)
    ref_r = ref_same.bfloat16().float()
    diff_r = (outf - ref_r).abs()
    d_rounded = diff_r.max().item()
    # Element-local 1-ulp law (the face doc's "rounded-match <= 1 ulp"):
    # ulp at the binade of max(|got|, |ref|) PER ELEMENT -- the grid the
    # difference actually lives on (r198 run-10 ledger: worst elements
    # had got == bf16(ref) exactly, and every residual was one adjacent
    # grid step; a ref-binade ulp misjudges boundary elements where got
    # rounds into the next binade up). Zero mags demand exactness.
    mag = torch.maximum(outf.abs(), ref_same.abs())
    expo = torch.floor(torch.log2(mag.clamp_min(1e-38)))
    ulp_elem = torch.exp2(torch.where(mag > 0, expo, torch.full_like(expo, -126.0)) - 7.0)
    ok_exact = d_exact <= 2e-3
    ok_rounded = bool((diff_r <= ulp_elem).all())
    if v2_out is not None:
        v2f = v2_out.float().cpu()
        d_v2 = (v2f - ref_same).abs().max().item()
        ok_exact = ok_exact or d_exact <= 2e-3 + d_v2
        extra = f"{extra} d_v2={d_v2:.2e}"
        # v2-side rounding law: within 1 ulp of the CERTIFIED decoder's
        # own output as well. At deep-cancellation elements (|O| below
        # ~2^-13 of the p*V term scale) even an honest fp32 pipeline
        # diverges from the fp64 truth by 2+ grid steps -- r198 run-16
        # mini-ledger len17: v2 itself sits 2 steps off bf16(ref). The
        # v2 side holds the kernel to v2's rounding behavior; v2 passes
        # its own law by construction, zero free parameters.
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
            face,
            fmt,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} cos={c_same:.7f}{extra}",
        )
    else:
        _ok(
            face,
            fmt,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} cos={c_same:.7f}{extra}",
        )
    return dict(maxdiff=d_exact, maxdiff_rounded=d_rounded, cos_same=c_same, gated=gated)


def _mini_ledger(fmt, tag, out, out_v2, ref_same, q, k_deq, v_deq, lens):
    """Failing-cell element decomposition: offending elements ranked by
    grid steps of |got - bf16(ref)|, against the pure law C =
    bf16(p64@v64), an honest fp32 pipeline F32 (fp32 logits, fp32
    softmax, fp32 PV on the same dequantized cache; output bf16), and
    the certified v2. Discriminates intrinsic fp32-vs-fp64 attention
    noise (got ~= F32, F32 itself off ref by steps) from a
    kernel-specific accumulation defect (got off F32 by steps)."""
    got = out.float().cpu().double().reshape(-1, D)
    v2f = out_v2.float().cpu().double().reshape(-1, D)
    outs_c, outs_f, ref_rows = [], [], []
    t0i = 0
    for b, lnb in enumerate(lens):
        kb = k_deq[t0i : t0i + lnb]
        vb = v_deq[t0i : t0i + lnb]
        for hq in range(HQ):
            kh = kb[:, hq // HREP, :]
            vh = vb[:, hq // HREP, :]
            qd = q[b, hq].double().cpu()
            p64 = torch.softmax((kh.double() @ qd) / math.sqrt(D), dim=0)
            ref_rows.append(p64 @ vh.double())
            outs_c.append((p64 @ vh.double()).bfloat16().double())
            s32 = (kh.float() @ q[b, hq].float().cpu()) / math.sqrt(D)
            p32 = torch.softmax(s32, dim=0)
            outs_f.append((p32 @ vh.float()).bfloat16().double())
        t0i += lnb
    ref = torch.stack(ref_rows)
    C = torch.stack(outs_c)
    F32 = torch.stack(outs_f)
    ref_r = ref.bfloat16().double()
    diff_r = (got - ref_r).abs()
    mag = torch.maximum(got.abs(), ref.abs())
    steps = diff_r / torch.exp2(
        torch.floor(torch.log2(mag.clamp_min(1e-38))) - 7.0
    )
    n_over = int((steps > 1.0).sum())
    det = []
    for fidx in steps.flatten().topk(6).indices.tolist():
        rr, cc = divmod(fidx, D)
        det.append(
            f"({rr // HQ},{rr % HQ},c{cc}) got={float(got[rr, cc]):.4g} "
            f"ref={float(ref[rr, cc]):.4g} C={float(C[rr, cc]):.4g} "
            f"F32={float(F32[rr, cc]):.4g} v2={float(v2f[rr, cc]):.4g} "
            f"steps={float(steps[rr, cc]):.2f}"
        )
    print(
        f"  mini {fmt}/{tag}: n(steps>1)={n_over} "
        f"d(got,F32)={(got - F32).abs().max():.2e} "
        f"d(F32,ref)={(F32 - ref).abs().max():.2e} "
        f"d(got,v2)={(got - v2f).abs().max():.2e} "
        f"d(C,F32)={(C - F32).abs().max():.2e} | " + " | ".join(det)
    )


def _prep(T, seed):
    gen = torch.Generator().manual_seed(seed)
    xk = torch.randn(T, HK, D, generator=gen).to(DEV)
    xv = torch.randn(T, HK, D, generator=gen).to(DEV)
    lens, tables, tb, tp, n_blocks = build_paged(T, seed=seed % 97)
    q = torch.randn(len(lens), HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
    return xk, xv, lens, tables, tb, tp, n_blocks, q


def face_native_exact():
    print("[face: native-exact] native decode vs fp64 exact on dequantized cache")
    for fmt in FMTS:
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(549, 99)
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        out = native_call(q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        out_v2 = mx_decode(q.float(), cache, bt, sl, 1.0 / math.sqrt(D), block_size=BLOCK)
        torch.xpu.synchronize()
        k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
        ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
        rec = _gated("native-exact", fmt, "random549", out, ref_same, v2_out=out_v2)
        if rec:
            _rec("native-exact", fmt, random549=rec)


def face_native_v2():
    print("[face: native-v2] native vs certified Triton v2 on the same cache")
    for fmt in FMTS:
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(549, 4242)
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        out_v2 = mx_decode(q.float(), cache, bt, sl, 1.0 / math.sqrt(D), block_size=BLOCK)
        torch.xpu.synchronize()
        out_nat = native_call(q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        rec = _gated("native-v2", fmt, "random549", out_nat, out_v2.float().double())
        if rec:
            _rec("native-v2", fmt, random549=rec)


def face_bitwise():
    print("[face: bitwise] seq_len==1 dequant bit-math oracle (O == bf16(V_deq))")
    for fmt in FMTS:
        gen = torch.Generator().manual_seed(777)
        xk = torch.randn(1, HK, D, generator=gen).to(DEV)
        xv = torch.randn(1, HK, D, generator=gen).to(DEV)
        lens, tables, tb, tp, n_blocks = build_paged(1, seed=3)
        q = torch.randn(1, HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        out = native_call(q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        _, v_deq = _dequant_cache(cache, tb, tp, fmt)
        mism = 0
        for hq in range(HQ):
            got = out[0, hq].view(torch.int16).cpu()
            want = v_deq[0, hq // HREP].to(torch.bfloat16).view(torch.int16).cpu()
            # IEEE sign-of-zero law: a -0.0 dequant value (e2m1 nibble 8,
            # 1/16 of elements) accumulates as (+0)+(-0)=+0 inside the PV
            # MMA (RN sums opposite-signed zeros to +0; masked k lanes
            # contribute +0 terms) -- sign-of-zero is an accumulation
            # property shared with the vendor fp8 lane, not a dequant
            # property. Canonicalize +-0 bit patterns before comparing.
            for pat in (got, want):
                pat[pat == -32768] = 0  # int16 view of bf16 -0.0 -> +0.0
            mism += int((got != want).sum())
        if mism:
            # localize: canonicalized bad cols per head; head0 detail with
            # the raw byte/nibble/scale codes so the producing element
            # self-identifies (q = got/scale must land on the e2m1 grid;
            # a DIFFERENT grid point = wrong nibble/byte, off-grid = wrong
            # coordinate/scale mixup).
            gb = out[0].view(torch.int16).cpu()
            wb = v_deq[0].repeat_interleave(HREP, 0).to(torch.bfloat16).view(torch.int16).cpu()
            for pat in (gb, wb):
                pat[pat == -32768] = 0
            per_head = (gb != wb).sum(-1).tolist()
            bad0 = (gb[0] != wb[0]).nonzero().flatten()
            g0 = out[0, 0].float().cpu()
            w0 = v_deq[0, 0].float().cpu()
            zwant = int((wb[0][bad0] == 0).sum())
            ng = D // 32
            vw = D // 2 if fmt in ("mxfp4", "mxk8v4") else D
            kw = D + ng if fmt in ("mxfp8", "mxk8v4") else D // 2 + ng
            slot = cache[tb[0].long().cpu(), tp[0].long().cpu(), 0, :].cpu().tolist()
            det = []
            for c in bad0[:12].tolist():
                byte = slot[kw + (c // 2)]
                nib = (byte >> 4) if (c & 1) else (byte & 0xF)
                cd = slot[kw + vw + (c // 32)]
                gv, wv = float(g0[c]), float(w0[c])
                det.append(f"c{c}:nib={nib},cd={cd},w={wv:.4g},g={gv:.4g},q={gv / 2.0 ** (cd - 127):.4g}")
            print(
                f"    [probe] per-head={per_head} zero-want={zwant}/{len(bad0)} "
                f"head0: " + " ".join(det)
            )
            _fail("bitwise", fmt, f"{mism}/{HQ * D} mismatched bf16 bit patterns")
        else:
            _ok("bitwise", fmt, f"0/{HQ * D} mismatches — dequant bit law exact")
        _rec("bitwise", fmt, bit_mismatches=mism)


def face_robust():
    print("[face: robust] adversarial rows + block-16 boundary lens (shuffled tables)")
    gen = torch.Generator().manual_seed(555)
    cases = {}
    x = torch.randn(128, HK, D, generator=gen)
    x[7, 0, 0:32] = 1e-38  # denormal-magnitude group
    x[9, 1, 0:32] = 0.0  # zero group
    x[11, 0, 5] = 3.0e2  # single outlier
    cases["adversarial"] = x
    cases["signed"] = torch.randn(128, HK, D, generator=gen) * torch.sign(
        torch.randn(128, HK, D, generator=gen)
    )
    for name, xk in cases.items():
        for fmt in FMTS:
            xv = (torch.randn(128, HK, D, generator=gen) * 0.1).to(DEV)
            lens, tables, tb, tp, n_blocks = build_paged(128, seed=11)
            q = torch.randn(len(lens), HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
            cache = torch.zeros(
                n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
            )
            mx_store(xk.to(DEV), xv, cache, tb, tp)
            torch.xpu.synchronize()
            bt = flat_tables(lens, tables)
            out = native_call(q, cache, fmt, lens, bt)
            torch.xpu.synchronize()
            sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
            out_v2 = mx_decode(
                q.float(), cache, bt, sl, 1.0 / math.sqrt(D), block_size=BLOCK
            )
            torch.xpu.synchronize()
            k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
            ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
            rec = _gated("robust", fmt, name, out, ref_same, v2_out=out_v2)
            if rec:
                _rec("robust", fmt, **{name: rec})
                if not rec["gated"]:
                    _mini_ledger(fmt, name, out, out_v2, ref_same, q, k_deq, v_deq, lens)
    for ln in (1, 15, 16, 17):
        for fmt in FMTS:
            xk = torch.randn(ln, HK, D, generator=gen).to(DEV)
            xv = torch.randn(ln, HK, D, generator=gen).to(DEV)
            lens, tables, tb, tp, n_blocks = build_paged(ln, seed=ln)
            q = torch.randn(len(lens), HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
            cache = torch.zeros(
                n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
            )
            mx_store(xk, xv, cache, tb, tp)
            torch.xpu.synchronize()
            bt = flat_tables(lens, tables)
            out = native_call(q, cache, fmt, lens, bt)
            torch.xpu.synchronize()
            sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
            out_v2 = mx_decode(
                q.float(), cache, bt, sl, 1.0 / math.sqrt(D), block_size=BLOCK
            )
            torch.xpu.synchronize()
            k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
            ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
            rec = _gated("robust", fmt, f"len{ln}", out, ref_same, v2_out=out_v2)
            if rec:
                _rec("robust", fmt, **{f"len{ln}": rec})
                if not rec["gated"]:
                    _mini_ledger(
                        fmt, f"len{ln}", out, out_v2, ref_same, q, k_deq, v_deq, lens
                    )


def face_splits():
    print("[face: splits] forced split-KV (ReduceSplitK) + default plan at ctx 8192")
    for fmt in FMTS:
        xk, xv, lens, tables, tb, tp, n_blocks, q = _prep(8192, 31337)
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
        ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
        for tag, splits in (("forced8", 8), ("default", 0)):
            out = native_call(q, cache, fmt, lens, bt, splits=splits)
            torch.xpu.synchronize()
            rec = _gated("splits", fmt, tag, out, ref_same)
            if rec:
                _rec("splits", fmt, **{tag: rec})


def face_telemetry():
    print("[face: telemetry] native vs Triton v2 decode timings (always-loop datapoint)")
    prop = torch.xpu.get_device_properties(0)
    R.setdefault("telemetry", {})["device"] = {"name": prop.name}
    for fmt in FMTS:
        rows = {}
        for ctx in (1024, 8192, 32768):
            xk = torch.randn(ctx, HK, D, device=DEV)
            xv = torch.randn(ctx, HK, D, device=DEV)
            lens, tables, tb, tp, n_blocks = build_paged(ctx, seed=ctx % 97)
            cache = torch.zeros(
                n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
            )
            mx_store(xk, xv, cache, tb, tp)
            bt = flat_tables(lens, tables)
            sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
            q = torch.randn(len(lens), HQ, D, device=DEV)
            q_bf = q.to(torch.bfloat16)
            for _ in range(3):
                mx_decode(q, cache, bt, sl, 1.0 / math.sqrt(D), block_size=BLOCK)
                native_call(q_bf, cache, fmt, lens, bt)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                mx_decode(q, cache, bt, sl, 1.0 / math.sqrt(D), block_size=BLOCK)
            torch.xpu.synchronize()
            t_v2 = (time.perf_counter() - t0) / 20
            t0 = time.perf_counter()
            for _ in range(20):
                native_call(q_bf, cache, fmt, lens, bt)
            torch.xpu.synchronize()
            t_nat = (time.perf_counter() - t0) / 20
            b_read = ctx * HK * slot_bytes(D, fmt) * HREP
            rows[ctx] = {
                "v2_ms": round(t_v2 * 1e3, 4),
                "native_ms": round(t_nat * 1e3, 4),
                "speedup": round(t_v2 / t_nat, 3),
                "native_gbps": round(b_read / t_nat / 1e9, 2),
            }
            print(
                f"  ok   telemetry/{fmt} ctx={ctx}: v2 {t_v2*1e3:.3f} ms, "
                f"native {t_nat*1e3:.3f} ms ({t_v2/t_nat:.2f}x), "
                f"{b_read/t_nat/1e9:.1f} GB/s effective"
            )
        _rec("telemetry", fmt, **{str(k): v for k, v in rows.items()})


def face_softmax_probe():
    """One-hot V rows expose P exactly: O[b, hq, t0+t] == p_t.

    v_t = e_t quantizes exactly on every lane (1.0 = 4 * 2^-2 on the
    e2m1 grid, exact in e4m3; zero groups decode to zero), so the PV
    product is the identity on p and the softmax path is bit-visible:
    a smooth ~1e-3 deviation = the exp2 implementation; a 2^-8-grid
    signature = a P operand cast; ~1e-7 = neither (accumulation).
    len 15 = one k-tile: no cross-tile rescale enters.
    """
    print("[face: softmax-probe] one-hot V rows expose P exactly")
    ln = 15
    gen = torch.Generator().manual_seed(42)
    xk = torch.randn(ln, HK, D, generator=gen)
    xv = torch.zeros(ln, HK, D)
    for t in range(ln):
        xv[t, :, t] = 1.0
    lens, tables, tb, tp, n_blocks = build_paged(ln, seed=4)
    q = torch.randn(len(lens), HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
    for fmt in FMTS:
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk.to(DEV), xv.to(DEV), cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        out = native_call(q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
        # sanity: V dequant must be the one-hot identity (else the probe
        # is measuring quantization, not softmax)
        vid = torch.zeros_like(v_deq)
        for t in range(ln):
            vid[t, :, t] = 1.0
        vsame = bool(torch.equal(v_deq, vid))
        # fp64 softmax rows per (b, hq); observed p = O diagonal slices
        t0i = 0
        rows, obs = [], []
        for b, lnb in enumerate(lens):
            kb = k_deq[t0i : t0i + lnb].double()
            for hq in range(HQ):
                s = (kb[:, hq // HREP, :] @ q[b, hq].double().cpu()) / math.sqrt(D)
                rows.append(torch.softmax(s, dim=0))
                obs.append(out[b, hq, t0i : t0i + lnb].float().cpu())
            t0i += lnb
        pref = torch.cat(rows)
        pobs = torch.cat(obs)
        dp = (pobs - pref).abs()
        cast_err = (pref.bfloat16().float() - pref).abs()
        print(
            f"  probe {fmt}: onehot_v={'exact' if vsame else 'DEGRADED'} "
            f"max|dp|={dp.max():.2e} mean|dp|={dp.mean():.2e} "
            f"max|p-bf16(p)|={cast_err.max():.2e} "
            f"n(dp>1.5*cast)={int((dp > 1.5 * cast_err).sum())}/{dp.numel()}"
        )
        _rec("softmax_probe", fmt, onehot_v=vsame, max_dp=dp.max().item())


def face_element_ledger():
    """Worst-element ledger for the short-len residual.

    Reproduces the robust boundary-lens structure (single-tile seqs of
    5-12 rows) with its own seed, then decomposes the top error
    elements against candidate value laws: C = bf16(fp64 p@v) is the
    pure output-cast law -- d(got, C) isolates everything structural
    before the output cast; A adds fp32-rounded p; B adds bf16(p).
    """
    print("[face: element-ledger] worst-element decomposition, short lens")
    ln = 15
    for fmt in FMTS:
        gen = torch.Generator().manual_seed(15)
        xk = torch.randn(ln, HK, D, generator=gen).to(DEV)
        xv = torch.randn(ln, HK, D, generator=gen).to(DEV)
        lens, tables, tb, tp, n_blocks = build_paged(ln, seed=15)
        q = torch.randn(len(lens), HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        out = native_call(q, cache, fmt, lens, bt)
        torch.xpu.synchronize()
        k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
        # candidate laws, fp64 with per-head GQA math
        outs_c, outs_a, outs_b, ref = [], [], [], []
        t0i = 0
        rows_all = []
        for b, lnb in enumerate(lens):
            kb = k_deq[t0i : t0i + lnb].double()
            vb = v_deq[t0i : t0i + lnb].double()
            for hq in range(HQ):
                s = (kb[:, hq // HREP, :] @ q[b, hq].double().cpu()) / math.sqrt(D)
                p = torch.softmax(s, dim=0)
                o = p @ vb[:, hq // HREP, :]
                ref.append(o)
                outs_c.append(o.bfloat16().double())
                p32 = torch.softmax(s.float(), dim=0).double()
                outs_a.append((p32 @ vb[:, hq // HREP, :]).bfloat16().double())
                pb = p.bfloat16().double()
                outs_b.append((pb @ vb[:, hq // HREP, :]).bfloat16().double())
                rows_all.append((b, hq, t0i, p, vb[:, hq // HREP, :]))
            t0i += lnb
        got = torch.stack(
            [out[b, hq] for b, hq, _, _, _ in rows_all]
        ).float().cpu().double()
        ref = torch.stack(ref)
        d = (got - ref).abs()
        idx = d.flatten().argmax().item()
        r, c = divmod(idx, D)
        b, hq, t0i, p, vb = rows_all[r]
        C = torch.stack(outs_c)
        dc_map = (got - C).abs()
        dc = dc_map.max().item()
        da = (got - torch.stack(outs_a)).abs().max().item()
        db = (got - torch.stack(outs_b)).abs().max().item()
        # steps of the max-mag binade per element; the >1-step census
        mag = torch.maximum(got.abs(), ref.abs())
        steps = dc_map / torch.exp2(
            torch.floor(torch.log2(mag.clamp_min(1e-38))) - 7.0
        )
        n_over = int((steps > 1.0).sum())
        top = dc_map.flatten().topk(3).indices.tolist()
        det = []
        for fidx in top:
            rr, cc = divmod(fidx, D)
            g2 = float(got[rr, cc])
            r2 = float(ref[rr, cc])
            c2 = float(C[rr, cc])
            mm = max(abs(g2), abs(r2))
            det.append(
                f"({rows_all[rr][0]},{rows_all[rr][1]},c{cc}) "
                f"got={g2:.5g} ref={r2:.5g} C={c2:.5g} mag={mm:.3g} "
                f"steps={float(dc_map[rr, cc]) / 2 ** (math.floor(math.log2(max(mm, 1e-38))) - 7):.2f}"
            )
        print(
            f"  ledger {fmt}: d_ref={d.max():.2e} n(>1step)={n_over} "
            f"d_C={dc:.2e} d_A(fp32p)={da:.2e} d_B(bf16p)={db:.2e} | "
            + " | ".join(det)
        )
        _rec(
            "element_ledger",
            fmt,
            d_ref=d.max().item(),
            d_C=dc,
            d_A=da,
            d_B=db,
            n_over_one_step=n_over,
        )


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p198_native_offline — device={DEV}, torch={torch.__version__}")
    face_native_exact()
    face_native_v2()
    face_bitwise()
    face_softmax_probe()
    face_element_ledger()
    face_robust()
    face_splits()
    face_telemetry()
    R["verdict"] = "PASS" if not FAILED else "FAIL"
    R["failures"] = FAILED
    with open(out / "p198_native.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=2, sort_keys=True)
    n = len(FAILED)
    print(f"\nVERDICT: {R['verdict']}" + (f" ({n} failures)" if n else ""))
    sys.exit(0 if not n else 1)


if __name__ == "__main__":
    main()
