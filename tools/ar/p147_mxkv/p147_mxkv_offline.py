#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p147_mxkv_offline.py — P6.47 leg A: offline XPU validation of the
MXFP4/MXFP8 KV-cache lane (P6.47 goal, 2026-11-11).

Runs INSIDE a cert23-family container with an XPU device visible (no
engine boot; radiance-test untouched). Gates, in order:

  ref      reference self-test (CPU, exact math): encode->decode
           round-trip error bounds on random tensors; zero rows
           encode to zero payload; ceil_log2_exact equals the naive
           torch ceil(log2) on a 1M-point logspace sweep.
  bytes    store-kernel BYTE LAW: triton store bytes == mx_quant
           reference bytes for every (token, head, slot) — the bit
           contract the whole lane stands on (adversarial rows
           included: denormal group, zero group, single outlier).
  decode   decode-vs-exact: kernel attention output vs fp64 reference
           attention on the SAME dequantized K/V (isolates kernel
           correctness from quantization error).
  accuracy decode-vs-original: vs fp64 attention on the UNQUANTIZED
           K/V — the end-to-end fidelity metric (cos-sim floor).
  robust   adversarial store+decode: denormal-magnitude rows, single
           outlier, all-zero, sign-flipped, and boundary seq lens
           (1/63/64/65) through a fragmented (shuffled) block table.
           No NaN/Inf anywhere; vs-exact gate on every case.
  telemetry (always-loop directive) store/decode timings + effective
           bytes/s at 1K/8K/32K context — the first bottleneck-ranking
           datapoint for the optimization loop.

Slot contract under test (mx_layout): a slot is [ K payload ][ Ks
codes ][ V payload ][ Vs codes ]; kslot/vslot tensors handed to
decode_kv are each [payload + codes] of their own block.

Exit 0 only if every gate passes. Output: <out>/p147_offline.json +
human summary to stdout. Every write site uses newline="\\n" (p139 law).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

# Load the mx_kv package by FILE PATH under a synthetic top-level name.
# Inside the cert container the vllm_gaudi wheel is installed, and a
# source-tree /work/vllm_gaudi without its parent __init__ files is only
# a namespace portion — the wheel's regular package would win resolution
# (proven on the first leg-A attempt). A dotless synthetic name makes
# the package's relative imports (from .mx_layout import ...) resolve
# without touching vllm_gaudi at all.
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
capacity_mult = mxkv.capacity_mult
ceil_log2_exact = mxkv.ceil_log2_exact
decode_kv = mxkv.decode_kv
encode_mxfp4 = mxkv.encode_mxfp4
encode_mxfp8 = mxkv.encode_mxfp8
mx_decode = mxkv.mx_decode
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes
offsets = mxkv.offsets

D = 128
HK = 2
HREP = 4  # Hq = HK * HREP = 8 (GQA shape of both certified models)
BLOCK = 64
DEV = "xpu"
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
# Role-aware element encoders (mxk8v4 = e4m3 K + e2m1 V — the mixed slot):
# K and V use the uniform format's single encoder; mxk8v4 composes the
# per-leg element classes, exactly encode_mxk8v4's math (mx_quant.py).
KENC = {"mxfp8": encode_mxfp8, "mxfp4": encode_mxfp4, "mxk8v4": encode_mxfp8}
VENC = {"mxfp8": encode_mxfp8, "mxfp4": encode_mxfp4, "mxk8v4": encode_mxfp4}
# Format-fidelity sanity floors (vs unquantized fp64 attention), empirically
# calibrated on leg A: unit-scale random K/V at D=128 is the hardest face —
# mxfp8 lands ~0.9993, mxfp4 ~0.9883 (round-trip faces of the REFERENCE
# itself land 0.9997/0.9933). mxk8v4 floors = the 4-bit-V leg (binding
# side; the 8-bit-K leg is strictly tighter and its content defects are
# caught exactly by the bytes face).
COS_FLOOR_ORIG = {"mxfp8": 0.99850, "mxfp4": 0.98500, "mxk8v4": 0.98500}
# round-trip sanity floors for the REFERENCE itself: max-rel is the wrong
# metric on mxfp4 (values just under a grid midpoint quantize to 0 —
# rel ~1.0 is inherent 4-bit behavior, not a defect); cos-sim + relative
# RMSE capture the actual fidelity.
COS_FLOOR_REF = {"mxfp8": 0.99950, "mxfp4": 0.99000, "mxk8v4": 0.99000}
RMSE_REL_CEIL = {"mxfp8": 0.07, "mxfp4": 0.20, "mxk8v4": 0.20}
R = {}  # results tree
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


def kslot(x, fmt):
    """[K payload | Ks codes] for a [..., D] tensor (mx_layout block)."""
    enc, codes = KENC[fmt](x)
    return torch.cat([enc, codes], dim=-1)


def vslot(x, fmt):
    """[V payload | Vs codes] — V-role elements (e2m1 on mxk8v4)."""
    enc, codes = VENC[fmt](x)
    return torch.cat([enc, codes], dim=-1)


def build_paged(total_tokens, seed=7):
    """Fragmented block tables: shuffled physical blocks, 2 sequences
    (1 sequence when total_tokens < 2 — zero-length sequences are never
    scheduled by the engine and are not a harness case)."""
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


def face_ref():
    print("[face: ref] reference self-test (CPU, exact math)")
    gen = torch.Generator().manual_seed(20261111)
    for fmt in FMTS:
        x = torch.randn(64, HK, D, generator=gen)
        # both roles through their own element class (mxk8v4: e4m3 K /
        # e2m1 V) — the true mixed-slot round-trip
        k, v = decode_kv(kslot(x, fmt), vslot(x, fmt), fmt, D)
        c = min(cos_sim(k, x), cos_sim(v, x))
        rmse_rel = max(
            (k - x).pow(2).mean().sqrt().item(), (v - x).pow(2).mean().sqrt().item()
        ) / x.pow(2).mean().sqrt().item()
        if c < COS_FLOOR_REF[fmt] or rmse_rel > RMSE_REL_CEIL[fmt]:
            _fail("ref", fmt, f"round-trip cos {c:.6f} rmse_rel {rmse_rel:.4f}")
        else:
            _ok("ref", fmt, f"round-trip cos {c:.6f}, rel-RMSE {rmse_rel:.4f}")
        _rec("ref", fmt, round_trip_cos=c, round_trip_rmse_rel=rmse_rel)
        z = torch.zeros(4, HK, D)
        zeros_ok = True
        for enc_fn in (KENC[fmt], VENC[fmt]):
            encz, codesz = enc_fn(z)
            if encz.abs().sum().item() != 0.0 or codesz.max().item() != 0:
                zeros_ok = False
        if not zeros_ok:
            _fail("ref", fmt, "zero rows produced nonzero payload/codes")
        else:
            _ok("ref", fmt, "zero rows encode to zero payload + code 0")
    # E8M0 exponent oracle: the DEFINING PROPERTY, not naive-agreement.
    # (Leg-A finding #2, 2026-11-11: torch ceil(log2) crosses an ulp edge
    # on ~1/1M logspace points; the bit-derived exact version is right —
    # the property 2^e >= r > 2^(e-1) settles it objectively.)
    r = torch.logspace(-6, 6, 1_000_000).float()
    e = ceil_log2_exact(r)
    hi_ok = torch.exp2(e.to(torch.float32)) >= r
    lo_ok = (e <= -127) | (torch.exp2((e - 1).to(torch.float32)) < r)
    bad = int((~(hi_ok & lo_ok)).sum())
    naive_mism = int((e != torch.ceil(torch.log2(r))).sum())
    if bad:
        _fail("ref", "-", f"ceil_log2_exact violates its property on {bad}/1M points")
    else:
        _ok(
            "ref",
            "-",
            f"ceil_log2_exact satisfies 2^e >= r > 2^(e-1) on 1M points "
            f"(naive disagrees on {naive_mism} — ulp edge, informational)",
        )
    _rec("ref", "-", property_violations=bad, naive_vs_exact_mismatch=naive_mism)


def face_bytes():
    print("[face: bytes] store-kernel byte law (XPU)")
    gen = torch.Generator().manual_seed(4242)
    kx = torch.randn(256, HK, D, generator=gen)
    kx[7, 0, 0:32] = 1e-38  # denormal-magnitude group
    kx[9, 1, 0:32] = 0.0  # zero group
    kx[11, 0, 5] = 3.0e2  # single outlier in one group
    vx = torch.randn(256, HK, D, generator=gen) * 0.5
    for fmt in FMTS:
        _, _, tb, tp, n_blocks = build_paged(256)
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(kx.to(DEV), vx.to(DEV), cache, tb, tp)
        torch.xpu.synchronize()
        got = cache[tb.long().cpu(), tp.long().cpu(), :, :].cpu()
        off = offsets(D, fmt)
        mismatch = 0
        for name, src in (("k", kx), ("v", vx)):
            ref = kslot(src, fmt) if name == "k" else vslot(src, fmt)
            got_blk = got[..., off[name] : off[name] + ref.shape[-1]]
            mismatch += int((got_blk != ref).sum())
        if mismatch:
            _fail("bytes", fmt, f"{mismatch} byte mismatches kernel vs reference")
        else:
            _ok("bytes", fmt, f"0 byte mismatches over {256 * HK} slots")
        _rec("bytes", fmt, byte_mismatches=mismatch)


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
    # K block width: e4m3 bytes (mxfp8, mxk8v4) or e2m1 nibbles (mxfp4)
    kwidth = D + ng if fmt in ("mxfp8", "mxk8v4") else D // 2 + ng
    return decode_kv(got[..., :kwidth].contiguous(), got[..., kwidth:].contiguous(), fmt, D)


def _run_decode_face(face, xk, xv, q, lens, tables, tb, tp, n_blocks, tag):
    for fmt in FMTS:
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        out = mx_decode(q, cache, bt, sl, sm_scale=1.0 / math.sqrt(D))
        torch.xpu.synchronize()
        out = out.cpu()
        if not torch.isfinite(out).all():
            _fail(face, fmt, f"non-finite output ({tag})")
            continue
        k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
        ref_same = _ref_attention(q.cpu(), k_deq, v_deq, lens)
        d_exact = (out - ref_same).abs().max().item()
        c_same = cos_sim(out, ref_same)
        ref_orig = _ref_attention(q.cpu(), xk.cpu(), xv.cpu(), lens)
        c_orig = cos_sim(out, ref_orig)
        rmse = out.sub(ref_orig).pow(2).mean().sqrt().item()
        gated = d_exact <= 2e-3 and c_same >= 0.99999
        if not gated:
            _fail(face, fmt, f"{tag}: vs-exact maxdiff {d_exact:.2e} cos {c_same:.7f}")
        elif face == "accuracy" and c_orig < COS_FLOOR_ORIG[fmt]:
            _fail(face, fmt, f"{tag}: vs-original cos {c_orig:.6f} < {COS_FLOOR_ORIG[fmt]}")
        else:
            _ok(
                face,
                fmt,
                f"{tag}: vs-exact d={d_exact:.2e} c={c_same:.7f}; "
                f"vs-orig cos={c_orig:.6f} rmse={rmse:.4f}",
            )
        _rec(
            face,
            fmt,
            **{
                tag: dict(
                    maxdiff=d_exact, cos_same=c_same, cos_orig=c_orig, rmse=rmse
                )
            },
        )


def face_decode():
    print("[face: decode] kernel decode vs fp64 exact on dequantized cache (XPU)")
    gen = torch.Generator().manual_seed(99)
    T = 549  # build_paged splits [T//3, T-T//3]: crosses block boundaries
    xk = torch.randn(T, HK, D, generator=gen).to(DEV)
    xv = torch.randn(T, HK, D, generator=gen).to(DEV)
    lens, tables, tb, tp, n_blocks = build_paged(T)
    q = torch.randn(len(lens), HK * HREP, D, generator=gen).to(DEV)
    _run_decode_face("decode", xk, xv, q, lens, tables, tb, tp, n_blocks, "random")


def face_accuracy():
    print("[face: accuracy] kernel decode vs fp64 attention on unquantized K/V (XPU)")
    gen = torch.Generator().manual_seed(1234)
    xk = torch.randn(600, HK, D, generator=gen).to(DEV)
    xv = torch.randn(600, HK, D, generator=gen).to(DEV)
    q = torch.randn(2, HK * HREP, D, generator=gen).to(DEV)
    lens, tables, tb, tp, n_blocks = build_paged(600)
    _run_decode_face("accuracy", xk, xv, q, lens, tables, tb, tp, n_blocks, "random600")


def face_robust():
    print("[face: robust] adversarial tensors + boundary seq lens (XPU)")
    gen = torch.Generator().manual_seed(555)
    cases = {}
    cases["denormal"] = torch.full((128, HK, D), 1e-38) * torch.randn(1, generator=gen)
    cases["outlier"] = torch.randn(128, HK, D, generator=gen) * 1e-3
    cases["outlier"][3, 0, 10] = 4e2
    cases["zeroed"] = torch.zeros(128, HK, D)
    cases["signed"] = torch.randn(128, HK, D, generator=gen) * torch.sign(
        torch.randn(128, HK, D, generator=gen)
    )
    for name, xk in cases.items():
        lens, tables, tb, tp, n_blocks = build_paged(128, seed=11)
        xv = (torch.randn(128, HK, D, generator=gen) * 0.1).to(DEV)
        q = torch.randn(len(lens), HK * HREP, D, generator=gen).to(DEV)
        _run_decode_face("robust", xk.to(DEV), xv, q, lens, tables, tb, tp, n_blocks, name)
    for ln in (1, 63, 64, 65):
        xk = torch.randn(ln, HK, D, generator=gen).to(DEV)
        xv = torch.randn(ln, HK, D, generator=gen).to(DEV)
        lens, tables, tb, tp, n_blocks = build_paged(ln, seed=ln)
        q = torch.randn(len(lens), HK * HREP, D, generator=gen).to(DEV)
        _run_decode_face("robust", xk, xv, q, lens, tables, tb, tp, n_blocks, f"len{ln}")


def face_telemetry():
    print("[face: telemetry] store/decode timings + effective bytes/s (XPU)")
    prop = torch.xpu.get_device_properties(0)
    R.setdefault("telemetry", {})["device"] = {
        "name": prop.name,
        "count": torch.xpu.device_count(),
    }
    for fmt in FMTS:
        rows = {}
        for ctx in (1024, 8192, 32768):
            xk = torch.randn(ctx, HK, D, device=DEV)
            xv = torch.randn(ctx, HK, D, device=DEV)
            lens, tables, tb, tp, n_blocks = build_paged(ctx, seed=ctx % 97)
            cache = torch.zeros(
                n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
            )
            bt = flat_tables(lens, tables)
            sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
            q = torch.randn(len(lens), HK * HREP, D, device=DEV)
            for _ in range(3):  # warmup + compile
                mx_store(xk, xv, cache, tb, tp)
                mx_decode(q, cache, bt, sl, 1.0 / math.sqrt(D))
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                mx_store(xk, xv, cache, tb, tp)
            torch.xpu.synchronize()
            t_store = (time.perf_counter() - t0) / 20
            t0 = time.perf_counter()
            for _ in range(20):
                mx_decode(q, cache, bt, sl, 1.0 / math.sqrt(D))
            torch.xpu.synchronize()
            t_dec = (time.perf_counter() - t0) / 20
            b_read = ctx * HK * slot_bytes(D, fmt) * HREP  # K+V payload re-read per q-head
            rows[ctx] = {
                "store_ms": round(t_store * 1e3, 4),
                "decode_ms": round(t_dec * 1e3, 4),
                "decode_gbps": round(b_read / t_dec / 1e9, 2),
                "slot_bytes": slot_bytes(D, fmt),
                "cap_mult_vs_fp16": round(capacity_mult(fmt), 3),
            }
            print(
                f"  ok   telemetry/{fmt} ctx={ctx}: store {t_store*1e3:.3f} ms, "
                f"decode {t_dec*1e3:.3f} ms, {b_read/t_dec/1e9:.1f} GB/s effective"
            )
        _rec("telemetry", fmt, **{str(k): v for k, v in rows.items()})


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p147_mxkv_offline — device={DEV}, torch={torch.__version__}")
    face_ref()
    face_bytes()
    face_decode()
    face_accuracy()
    face_robust()
    face_telemetry()
    R["verdict"] = "PASS" if not FAILED else "FAIL"
    R["failures"] = FAILED
    with open(out / "p147_offline.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=2, sort_keys=True)
    n = len(FAILED)
    print(f"\nVERDICT: {R['verdict']}" + (f" ({n} failures)" if n else ""))
    sys.exit(0 if not n else 1)


if __name__ == "__main__":
    main()
