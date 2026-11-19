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
                 error; p147 decode-face law). Output is bf16 (the
                 lane contract) — gate: d<=2e-3 OR every diff explained
                 by the bf16 output cast (rounded-match <= 1 ulp);
                 cos >= 0.99999 always.
  native-v2      native vs the certified Triton v2 decoder on the SAME
                 cache (the plan's dequant-oracle face: same operands,
                 both fp accumulators) — gate d<=2e-3 (bf16-cast
                 allowance as above) + cos >= 0.99999.
  bitwise        THE DEQUANT BIT-MATH ORACLE: seq_len==1 makes
                 P == 1.0 exactly and O == V_dequant with zero
                 accumulation, so the bf16 output must equal
                 bf16(reference dequant) BIT FOR BIT (e4m3/e2m1/e8m0
                 products are fp32-exact). Gate: 0 mismatched int16
                 patterns, all three formats.
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
    """fp64 reference decode: q [B, Hq, D]; k/v_deq [T, Hk, D] (host)."""
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


def _gated(face, fmt, tag, out, ref_same, extra=""):
    """Common gate: bf16-output-aware exact comparison (face law above)."""
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
    d_rounded = (outf - ref_r).abs().max().item()
    ulp = (2.0 ** -8) * ref_same.abs().max().item()
    ok_exact = d_exact <= 2e-3
    ok_rounded = d_rounded <= ulp
    gated = (ok_exact or ok_rounded) and c_same >= 0.99999
    if not gated:
        _fail(
            face,
            fmt,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} (ulp~{ulp:.1e}) "
            f"cos={c_same:.7f}{extra}",
        )
    else:
        _ok(
            face,
            fmt,
            f"{tag}: d={d_exact:.2e} d_rounded={d_rounded:.2e} cos={c_same:.7f}{extra}",
        )
    return dict(maxdiff=d_exact, maxdiff_rounded=d_rounded, cos_same=c_same)


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
        k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
        ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
        rec = _gated("native-exact", fmt, "random549", out, ref_same)
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
            mism += int((got != want).sum())
        if mism:
            # localize the corruption structure: per-head uniformity
            # (stride/head mapping), bad-column alignment (scale-group
            # boundaries at multiples of 32), got/want ratio (wrong
            # scale code = constant per group; wrong payload byte =
            # random).
            per_head = [
                int(
                    (
                        out[0, h].view(torch.int16).cpu()
                        != v_deq[0, h // HREP]
                        .to(torch.bfloat16)
                        .view(torch.int16)
                        .cpu()
                    ).sum()
                )
                for h in range(HQ)
            ]
            g0 = out[0, 0].float().cpu()
            w0 = v_deq[0, 0].float().cpu()
            bad0 = (
                g0.to(torch.bfloat16).view(torch.int16)
                != w0.to(torch.bfloat16).view(torch.int16)
            ).nonzero().flatten()
            ratios = [
                f"{float(g0[i] / w0[i]):.3g}" if w0[i] != 0 else "nan"
                for i in bad0[:8].tolist()
            ]
            print(
                f"    [probe] per-head={per_head} "
                f"head0 bad-cols[:24]={bad0[:24].tolist()} "
                f"ratios[:8]={ratios}"
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
            k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
            ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
            rec = _gated("robust", fmt, name, out, ref_same)
            if rec:
                _rec("robust", fmt, **{name: rec})
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
            k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
            ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
            rec = _gated("robust", fmt, f"len{ln}", out, ref_same)
            if rec:
                _rec("robust", fmt, **{f"len{ln}": rec})


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


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p198_native_offline — device={DEV}, torch={torch.__version__}")
    face_native_exact()
    face_native_v2()
    face_bitwise()
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
