#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p267_nan_canary.py — P6.47 P-D (r267): offline NaN-canary of the r266
engine MX-KV decode onset (c3/c4/c5: token 1 correct, 'our' attractor
from token 2, 324 x 400-nan; prefill clean; zero DEVICE_LOST).

Runs inside a cert29-family container against the image's OWN arc7mx
wheel (same bytes the engine legs ran — no pip reinstall): if the
canary is clean where the engine is broken, the defect is engine
wiring (metadata values / XPU-graph capture-replay), not the
kernel/store math.  Cells are the TWO gemma layer classes at engine geometry plus the
certified control:

    slide  D=256 page 64  Hk=8 G=2  W=1024   (the 50 sliding layers:
                                               16 q heads / 8 kv heads
                                               per rank -- num_attention_
                                               heads 32 / num_key_value_
                                               heads 16 at TP2, config
                                               truth 2026-11-28)
    glob   D=512 page 64  Hk=2 G=8  windowless (the 10 full layers:
                                               16 q / 2 kv per rank --
                                               num_global_key_value_
                                               heads 4 at TP2; PAGE 64:
                                               serve args block_size=64,
                                               the engine header's
                                               page-size-unification note,
                                               and p242's certified
                                               d512p64 cell -- the
                                               earlier "p128" census
                                               claim was an unverified
                                               inference, corrected
                                               2026-11-28)
    ctrl   D=256 page 64  Hk=2 G=6             (the certified qwen lane,
                                               r242's ctrl256p64)

ENGINE-DISPATCH LAW (the r267 fidelity fix): the engine's
_decode_attention (engine.py 776-830) routes by layer class --
sliding-window layers ride the WHEEL's mx_paged_decode (windowed),
windowless/global layers ride the FORK's Triton mx_decode called
WITHOUT block_size (engine.py ~823) -- the wrapper default is 64 and
BOTH engine pools are page 64 (serve args block_size=64, the engine
header's page-size-unification note, p242's certified d512p64; the
earlier "global pool p128" census claim was an unverified inference,
corrected 2026-11-28), so the default equals the true page and D-B is
EXONERATED as the corruption cause -- it stays a latent robustness
defect (a p128 pool would be walked HALF-page at seq_len>64), proven
on the synthetic cell of face pagewidth.  Every gated face runs the
engine's TRUE lane at the engine's TRUE pages (glob via Triton
default == pool page; slide via wheel windowed native).

FACES (each banked to the JSON; FAIL list -> nonzero exit):
  stepwise    THE onset face, never run before: bulk-store T prefill
              rows, then S incremental decode steps — each step
              stores ONE token row (the engine's decode-step store)
              and immediately decodes the same step's q over ALL rows
              incl. the one just written (bt is pre-built full
              length, the engine's admission-time block allocation).
              Gate: non-finite or the r198 exact law (d<=2e-3 or
              bf16 1-ulp; cos>=0.99999) vs fp64 over the dequantized
              cache.  The engine's token-1-clean/token-2-broken onset
              maps to step-0 clean (prefill rows only) and steps 1..
              corrupt.
  store-audit per incremental store: dequantize the pointed-to row
              and compare vs the mx_quant bit-exact reference of the
              source row, K AND V separately (the kv-store-separate-
              strides law), plus 4 sentinel rows elsewhere must stay
              byte-identical.  The v source arrives with DIVERGENT
              token+head strides ([HK,N,D].transpose(0,1): stride(2)=1
              so the wrapper's normalization never fires, while the
              token/head strides differ from k's) — re-audited here at
              the gemma cells (the banked audits used matching-stride
              sources).
  widewin     G1: the W=1024 long-ctx cell the r243 authority ran as
              mxfp4 ONLY — mxfp8 + mxk8v4 (mxfp4 as regression) at ctx
              8192/32768/40960 (engine ARC_MAXLEN; 40960/page64 gives
              the engine's bt width 640), windowed native vs
              window-masked fp64, with the d_blind discriminator.
  autosplit   multi-request value check at the engine's interleaved-
              request shape: 6 requests x ~1600 tokens in one shuffled
              table, auto plan (splits=0) vs forced-8 vs 1, all vs
              fp64 per-request refs.  (Hole/padded bt columns are
              never read: the walk is seq-len bounded.)
  uninit      EVIDENCE face (no gate): decode over blocks never
              written — (a) a 0xFF-filled pool (scale byte 0xFF ->
              2^128 inf dequant -> NaN logits) and (b) a fresh
              torch.empty pool — a census of non-finite outputs that
              characterizes what an engine WRONG-ROW read produces
              (the stale-slot-reuse signature's NaN source).
  pagewidth   D-B hardening probe (evidence, not gated): the engine's
              windowless call sites omit block_size and the wrapper
              defaults 64 — EXONERATED as the gemma corruption cause
              (both engine pools are page 64: serve args block_size
              64, page-size unification, p242's d512p64; the earlier
              p128 census claim was corrected 2026-11-28) but still a
              LATENT robustness defect.  A SYNTHETIC p128 D=512 cell
              ladders ln=64..256 default vs pagefix: default d blows
              up past 64 while pagefix holds the r198 law == the
              kernel property banked as hardening evidence for the
              wrapper fix (None -> cache.shape[1]).
  ctrl        the certified qwen-cell regression guard (bulk exact
              over a shuffled p64 table): the wheel's native decode
              AND the engine-true Triton lane (default 64 == pool page
              here, so both arms must pass — the r242 ctrl256p64 lane
              at the engine's actual call shape).

Exit 0 = canary clean (kernel/store math green at the engine's true
lanes and pages; the defect is engine wiring — metadata values, or
D-B recorded by pagewidth); nonzero = a gated face reproduced offline
(every FAIL line names face/cell/fmt + numbers).
Output: argv[1] = out dir (p267_canary.json; console captured by the
runner).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
import os
import site
import sys
from pathlib import Path

import torch


def _mxkv_dir() -> Path:
    """THE image's installed mx_kv (the bytes the engine legs ran — the
    r266 legs set no ARC_MXKV_OVERLAY, so site-packages IS the engine
    posture).  Order: explicit env -> cert-image site path -> any
    site-packages -> mounted-tree fallback (r242 harness shape)."""
    env = os.environ.get("P267_MXKV_DIR")
    if env:
        return Path(env)
    rel = Path("vllm_gaudi") / "xpu" / "patches" / "mx_kv"
    cands = [Path("/opt/venv/lib/python3.12/site-packages") / rel]
    with contextlib.suppress(Exception):
        cands += [Path(sp) / rel for sp in site.getsitepackages()]
    for cand in cands:
        if (cand / "mx_quant.py").is_file():
            return cand
    return Path(__file__).resolve().parents[3] / rel


_MXKV_DIR = _mxkv_dir()
_CELL_SEED = {"slide": 11, "glob": 23, "synthp128": 47}


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


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
encode_mxk8v4 = mxkv.encode_mxk8v4
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes
triton_decode = mxkv.mx_decode  # the ENGINE's windowless-decode lane (engine.py:823)

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

DEV = "xpu"
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
CELLS = {
    "slide": {"D": 256, "BLOCK": 64, "HK": 8, "HREP": 2, "W": 1024},
    "glob": {"D": 512, "BLOCK": 64, "HK": 2, "HREP": 8, "W": None},
    "ctrl": {"D": 256, "BLOCK": 64, "HK": 2, "HREP": 6, "W": None},
}
R: dict = {}
FAILED: list[str] = []


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


# ------------------------------------------------------------------ #
# pool + tables                                                       #
# ------------------------------------------------------------------ #

def build_paged(c, total_tokens, seed=7, n_seq=1):
    """Shuffled physical blocks; per-token (block, pos) destinations."""
    blk = c["BLOCK"]
    n_blocks = total_tokens // blk + 2
    perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(seed))
    order = perm.tolist()
    if n_seq == 1:
        lens = [total_tokens]
    else:
        lens = [max(1, total_tokens // n_seq + i) for i in range(n_seq)]
        lens[-1] = total_tokens - sum(lens[:-1])
    tables, tok_block, tok_pos = [], [], []
    cur = 0
    for ln in lens:
        ntb = (ln + blk - 1) // blk
        tab = order[cur: cur + ntb]
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


# ------------------------------------------------------------------ #
# reference + decode helpers                                           #
# ------------------------------------------------------------------ #

def dequant_rows(c, cache, tb, tp, rows, fmt):
    """Dequantize token rows (by block,pos) -> (k, v) [rows, Hk, D]
    host fp32.  Reads EXACTLY the bytes the walk reads."""
    d = c["D"]
    ng = d // 32
    kwidth = d + ng if fmt in ("mxfp8", "mxk8v4") else d // 2 + ng
    got = cache[tb[rows].long().cpu(), tp[rows].long().cpu(), :, :].cpu()
    return decode_kv(got[..., :kwidth].contiguous(), got[..., kwidth:].contiguous(), fmt, d)


def ref_decode_rows(c, q, k_deq, v_deq, ln, window=None):
    """fp64 windowed decode of ONE row: q [Hq, D] over k/v_deq
    [ln, Hk, D] (host, contiguous from position 0).  window=None ->
    full attention; W -> the last W positions (r243 mask law)."""
    d, hrep = c["D"], c["HREP"]
    lo = 0 if window is None else max(0, ln - window)
    outs = []
    for hq in range(q.shape[0]):
        hk = hq // hrep
        kb = k_deq[lo:ln, hk, :].double()
        vb = v_deq[lo:ln, hk, :].double()
        s = (kb @ q[hq].double()) / math.sqrt(d)
        outs.append(torch.softmax(s, dim=0) @ vb)
    return torch.stack(outs)


def native_call(c, q_bf, cache, fmt, lens, bt, splits=0, window=None):
    cu_q = torch.arange(len(lens) + 1, dtype=torch.int32, device=DEV)
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    ws = None if window is None else (window - 1, 0)
    return mx_paged_decode(
        q_bf, cache, fmt, cu_q, sl, bt,
        max_seqlen_k=max(lens),
        softmax_scale=1.0 / math.sqrt(c["D"]),
        num_splits_kv=splits,
        window_size=ws,
        host_kv_lens=torch.tensor(lens) if splits > 1 else None,
    )


def engine_call(c, q_bf, cache, fmt, lens, bt, window=None, pagefix=False):
    """The ENGINE's decode dispatch, engine.py _decode_attention 1:1:
    sliding-window layers -> the wheel's windowed native decode;
    windowless/global layers -> the fork's Triton mx_decode, which the
    engine calls WITHOUT block_size (wrapper default 64 == both engine
    pools' true page -- D-B latent-hardening, exonerated as the
    corruption cause).  pagefix=True passes the pool's true page
    explicitly (identical on p64 pools; the isolation arm for any
    non-p64 pool).  q
    passes as bf16 exactly as the engine does (the A9 law: Triton
    promotes the q*K products to fp32 in-kernel, bit-equal to the old
    .float() upcast); out follows q's dtype via empty_like."""
    if window is not None:
        return native_call(c, q_bf, cache, fmt, lens, bt, splits=0, window=window)
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    scale = 1.0 / math.sqrt(c["D"])
    if pagefix:
        return triton_decode(q_bf, cache, bt, sl, scale, block_size=c["BLOCK"])
    return triton_decode(q_bf, cache, bt, sl, scale)


def gate(face, cell, fmt, tag, out, ref):
    """r198 face law + the NaN canary.  Returns metrics dict."""
    outf = out.float().cpu()
    if not torch.isfinite(outf).all():
        nf = int((~torch.isfinite(outf)).sum())
        _fail(face, cell, fmt, f"{tag}: NON-FINITE output ({nf} elems)")
        return dict(nonfinite=nf)
    d_exact = (outf - ref).abs().max().item()
    c_same = cos_sim(outf, ref)
    ref_r = ref.bfloat16().float()
    diff_r = (outf - ref_r).abs()
    mag = torch.maximum(outf.abs(), ref.abs())
    expo = torch.floor(torch.log2(mag.clamp_min(1e-38)))
    ulp = torch.exp2(torch.where(mag > 0, expo, torch.full_like(expo, -126.0)) - 7.0)
    ok_rounded = bool((diff_r <= ulp).all())
    ok_exact = d_exact <= 2e-3
    if not ((ok_exact or ok_rounded) and c_same >= 0.99999):
        _fail(face, cell, fmt, f"{tag}: d={d_exact:.2e} rounded_ok={ok_rounded} cos={c_same:.7f}")
        return dict(maxdiff=d_exact, cos_same=c_same, gated=False)
    _ok(face, cell, fmt, f"{tag}: d={d_exact:.2e} cos={c_same:.7f}")
    return dict(maxdiff=d_exact, cos_same=c_same, gated=True)


# ------------------------------------------------------------------ #
# sources: K contiguous, V with DIVERGENT token+head strides          #
# (the kv-store-separate-strides law, re-audit at the gemma cells)    #
# ------------------------------------------------------------------ #

def make_sources(c, n, seed, diverge=True):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n, c["HK"], c["D"], generator=g).to(torch.float32)
    if diverge:
        wide = torch.randn(c["HK"], n, c["D"], generator=g).to(torch.float32)
        v = wide.transpose(0, 1)  # [n, HK, D]; strides (D, n*D, 1)
    else:
        v = torch.randn(n, c["HK"], c["D"], generator=g).to(torch.float32)
    return k.to(DEV), v.to(DEV)


def ref_encode_row(c, fmt, k_row, v_row):
    """mx_quant bit-exact reference of ONE (k, v) row [HK, D] ->
    (k_ref, v_ref) [HK, D] fp32 (the encode/decode round-trip law)."""
    d = c["D"]
    if fmt == "mxk8v4":
        kb, kc, vb, vc = encode_mxk8v4(k_row.unsqueeze(0), v_row.unsqueeze(0))
    else:
        enc = encode_mxfp8 if fmt == "mxfp8" else encode_mxfp4
        kb, kc = enc(k_row.unsqueeze(0))
        vb, vc = enc(v_row.unsqueeze(0))
    k, v = decode_kv(torch.cat([kb, kc], -1), torch.cat([vb, vc], -1), fmt, d)
    return k.squeeze(0), v.squeeze(0)


# ------------------------------------------------------------------ #
# FACE: stepwise (the onset) + store-audit                            #
# ------------------------------------------------------------------ #

def face_stepwise(c, cell, fmt, T=1024, S=24):
    d, hk, hrep = c["D"], c["HK"], c["HREP"]
    hq = hk * hrep
    lens, tables, tb, tp, n_blocks = build_paged(c, T + S, seed=_CELL_SEED[cell] + 5)
    bt = flat_tables(lens, tables)  # full-length table upfront (engine law)
    cache = torch.zeros(n_blocks, c["BLOCK"], hk, slot_bytes(d, fmt),
                        dtype=torch.uint8, device=DEV)
    k, v = make_sources(c, T + S, seed=4242 + _CELL_SEED[cell])
    q_all = torch.randn(S + 1, hq, d, generator=torch.Generator().manual_seed(77)).to(DEV)
    mx_store(k[:T], v[:T], cache, tb[:T], tp[:T])

    # step 0: decode over prefill rows only (the engine's clean case)
    out = engine_call(c, q_all[0:1].to(torch.bfloat16), cache, fmt, [T], bt,
                      window=c["W"], pagefix=True)
    kq, vq = dequant_rows(c, cache, tb, tp, list(range(T)), fmt)
    ref = ref_decode_rows(c, q_all[0].cpu(), kq, vq, T, window=c["W"])
    m = gate("stepwise", cell, fmt, f"step0 T={T}", out.squeeze(0), ref)
    _rec("stepwise", cell, fmt, step0=m)

    # sentinel rows for the mutation audit (mid-prefill, untouched)
    sent = [7, T // 3, T // 2, T - 13]
    sent_before = cache[tb[sent].long().cpu(), tp[sent].long().cpu()].clone()

    audit_fails = 0
    for s in range(1, S + 1):
        row = T + s - 1
        # the engine's decode-step store: ONE token row
        mx_store(k[row: row + 1], v[row: row + 1], cache, tb[row: row + 1], tp[row: row + 1])
        # store-audit: pointed row vs bit-exact reference, K AND V
        kr, vr = ref_encode_row(c, fmt, k[row].cpu(), v[row].cpu())
        kg, vg = dequant_rows(c, cache, tb, tp, [row], fmt)
        # both sides decode the SAME source through the SAME deterministic
        # decode_kv: identical stored bytes -> identical values, so any
        # diff > 1e-6 is a byte-level store defect (scale or payload)
        e_k = (kg[0] - kr).abs().max().item()
        e_v = (vg[0] - vr).abs().max().item()
        if max(e_k, e_v) > 1e-6 or not (
            torch.isfinite(kg).all() and torch.isfinite(vg).all()
        ):
            _fail("store-audit", cell, fmt,
                  f"row={row} e_k={e_k:.2e} e_v={e_v:.2e} (bit-law breach)")
            audit_fails += 1
        # sentinel mutation check
        safter = cache[tb[sent].long().cpu(), tp[sent].long().cpu()]
        if not torch.equal(safter, sent_before):
            _fail("store-audit", cell, fmt, f"row={row} SENTINEL MUTATED")
            audit_fails += 1
            sent_before = safter.clone()
        # decode this step's q over ALL rows (incl. the one just stored)
        ln = T + s
        out = engine_call(c, q_all[s: s + 1].to(torch.bfloat16), cache, fmt, [ln],
                          bt, window=c["W"], pagefix=True)
        kq, vq = dequant_rows(c, cache, tb, tp, list(range(ln)), fmt)
        ref = ref_decode_rows(c, q_all[s].cpu(), kq, vq, ln, window=c["W"])
        m = gate("stepwise", cell, fmt, f"step{s} ln={ln}", out.squeeze(0), ref)
        if not m.get("gated", False):
            _rec("stepwise", cell, fmt, first_bad_step=s, metrics=m)
            break
    else:
        _rec("stepwise", cell, fmt, first_bad_step=None, audit_fails=audit_fails)
        if audit_fails == 0:
            _ok("store-audit", cell, fmt, f"{S} incremental stores bit-clean, sentinels intact")
        else:
            _fail("store-audit", cell, fmt, f"{audit_fails} audit breaches in {S} stores")


# ------------------------------------------------------------------ #
# FACE: widewin (G1 — mxfp8/mxk8v4 at W=1024 long ctx)                #
# ------------------------------------------------------------------ #

def face_widewin(c, cell, fmt):
    for ctx in (8192, 32768, 40960):
        lens, tables, tb, tp, n_blocks = build_paged(c, ctx, seed=ctx % 97)
        bt = flat_tables(lens, tables)
        cache = torch.zeros(n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
                            dtype=torch.uint8, device=DEV)
        k, v = make_sources(c, ctx, seed=ctx % 97)
        mx_store(k, v, cache, tb, tp)
        q = torch.randn(1, c["HK"] * c["HREP"], c["D"],
                        generator=torch.Generator().manual_seed(ctx)).to(DEV).to(torch.bfloat16)
        # engine-true lane per class: sliding -> windowed native;
        # windowless -> Triton at the pool's true page (pagefix)
        out = engine_call(c, q, cache, fmt, [ctx], bt, window=c["W"], pagefix=True)
        kq, vq = dequant_rows(c, cache, tb, tp, list(range(ctx)), fmt)
        ref = ref_decode_rows(c, q.squeeze(0).float().cpu(), kq, vq, ctx, window=c["W"])
        m = gate("widewin", cell, fmt, f"ctx={ctx} W={c['W']}", out.squeeze(0), ref)
        if c["W"]:
            # d_blind: windowed output vs window-BLIND ref (the mask must bind)
            ref_b = ref_decode_rows(c, q.squeeze(0).float().cpu(), kq, vq, ctx, window=None)
            dbl = (out.squeeze(0).float().cpu() - ref_b).abs().max().item()
            print(f"  note widewin/{cell}/{fmt}: ctx={ctx} d_blind={dbl:.2e} (must be >> d)")
        else:
            dbl = None
        _rec("widewin", cell, fmt, ctx=ctx, d_blind=dbl, **m)


# ------------------------------------------------------------------ #
# FACE: autosplit (interleaved multi-request tables)                  #
# ------------------------------------------------------------------ #

def face_autosplit(c, cell, fmt, n_req=6, per_req=1600):
    total = n_req * per_req
    lens, tables, tb, tp, n_blocks = build_paged(c, total, seed=607, n_seq=n_req)
    bt = flat_tables(lens, tables)
    cache = torch.zeros(n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
                        dtype=torch.uint8, device=DEV)
    k, v = make_sources(c, total, seed=99)
    mx_store(k, v, cache, tb, tp)
    q = torch.randn(n_req, c["HK"] * c["HREP"], c["D"],
                    generator=torch.Generator().manual_seed(3)).to(DEV).to(torch.bfloat16)
    outs = {}
    if c["W"]:
        for tag, sp in (("auto", 0), ("forced8", 8), ("s1", 1)):
            outs[tag] = native_call(c, q, cache, fmt, lens, bt, splits=sp,
                                    window=c["W"]).float().cpu()
    else:
        # engine-true windowless lane: Triton at the pool's true page
        # (pagefix — the default-64 walk defect is face_pagewidth's).
        # splits are a wheel-internal concept; the Triton path owns its
        # own split plan (ARC_MXKV_NSPLIT / auto).
        outs["triton"] = engine_call(c, q, cache, fmt, lens, bt, window=None,
                                     pagefix=True).float().cpu()
    refs = []
    t0 = 0
    for b, ln in enumerate(lens):
        kq, vq = dequant_rows(c, cache, tb, tp, list(range(t0, t0 + ln)), fmt)
        refs.append(ref_decode_rows(c, q[b].float().cpu(), kq, vq, ln, window=c["W"]))
        t0 += ln
    ref = torch.stack(refs)
    for tag in outs:
        m = gate("autosplit", cell, fmt, f"{tag} n_req={n_req}", outs[tag], ref)
        _rec("autosplit", cell, fmt, tag=tag, **m)


# ------------------------------------------------------------------ #
# FACE: pagewidth (D-B — the engine's missing block_size, evidence)    #
# ------------------------------------------------------------------ #

def face_pagewidth(fmt, T=1024):
    """D-B hardening probe, EVIDENCE (not gated into FAILED): the
    engine's five windowless call sites omit block_size and the
    wrapper defaults 64 (triton_mx_decode.py) into the kernel's
    BLOCK_SIZE -- EXONERATED as the gemma corruption cause (both
    engine pools are page 64; default == true page on every pool the
    engine builds) but latent: a p128 pool would be walked HALF-page
    at seq_len>64.  This face proves the kernel property on a
    SYNTHETIC p128 D=512 cell (engine-true glob geometry except the
    page): for ln<=64 the default walk is accidentally correct (row j
    -> bt[j//64]=bt[0], pos j); from ln=65 j//64 indexes the
    128-token table pitch and pos wraps -- wrong rows everywhere.
    Ladder kept <= 256 so the walk stays inside the table width
    (T=1024 -> 8 blocks at pool page): wrong-ROW corruption isolated
    from out-of-bounds.  Default arm vs pagefix (true page)."""
    c = {"D": 512, "BLOCK": 128, "HK": 2, "HREP": 8, "W": None}
    cell = "synthp128"
    d, hk, hrep = c["D"], c["HK"], c["HREP"]
    lens, tables, tb, tp, n_blocks = build_paged(c, T, seed=_CELL_SEED[cell] + 9)
    bt = flat_tables(lens, tables)
    cache = torch.zeros(n_blocks, c["BLOCK"], hk, slot_bytes(d, fmt),
                        dtype=torch.uint8, device=DEV)
    k, v = make_sources(c, T, seed=808 + _CELL_SEED[cell])
    mx_store(k, v, cache, tb, tp)
    q = torch.randn(1, hk * hrep, d,
                    generator=torch.Generator().manual_seed(21)).to(DEV).to(torch.bfloat16)
    kq, vq = dequant_rows(c, cache, tb, tp, list(range(T)), fmt)
    scale = 1.0 / math.sqrt(d)
    ladder = {}
    for ln in (64, 65, 96, 128, 192, 256):
        ref = ref_decode_rows(c, q.squeeze(0).float().cpu(), kq[:ln], vq[:ln], ln)
        sl = torch.tensor([ln], dtype=torch.int32, device=DEV)
        o_def = triton_decode(q, cache, bt, sl, scale)  # engine.py ~823 verbatim
        o_fix = triton_decode(q, cache, bt, sl, scale, block_size=c["BLOCK"])
        arm = {}
        for tag, o in (("default", o_def), ("pagefix", o_fix)):
            of = o.squeeze(0).float().cpu()
            nf = int((~torch.isfinite(of)).sum())
            arm[tag] = dict(
                nonfinite=nf,
                dmax=float("nan") if nf else (of - ref).abs().max().item(),
            )
        ladder[str(ln)] = arm
        print(f"  note pagewidth/{cell}/{fmt}: ln={ln} "
              f"d_default={arm['default']['dmax']:.2e} "
              f"d_pagefix={arm['pagefix']['dmax']:.2e} "
              f"nf={arm['default']['nonfinite']}/{arm['pagefix']['nonfinite']}")
    d_b = all(
        ladder[str(ln)]["pagefix"]["dmax"] <= 2e-3
        and ladder[str(ln)]["pagefix"]["nonfinite"] == 0
        for ln in (64, 65, 96, 128, 192, 256)
    ) and any(
        ladder[str(ln)]["default"]["dmax"] > 2e-3 or ladder[str(ln)]["default"]["nonfinite"]
        for ln in (64, 65, 96, 128, 192, 256)
    )
    _rec("pagewidth", cell, fmt, ladder=ladder, d_b_reproduced=d_b)
    print(f"  note pagewidth/{cell}/{fmt}: D_B_REPRODUCED={d_b} "
          "(default-64 walk on a synthetic p128 pool corrupts past ln=64 "
          "while the true-page walk holds — latent-hardening evidence for "
          "the wrapper fix block_size None -> cache.shape[1]")


# ------------------------------------------------------------------ #
# FACE: uninit (evidence — what a WRONG-ROW read yields)               #
# ------------------------------------------------------------------ #

def face_uninit(c, cell, fmt, T=512):
    shape = (T // c["BLOCK"] + 2, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt))
    for kind in ("ff", "empty"):
        if kind == "ff":
            cache = torch.full(shape, 0xFF, dtype=torch.uint8, device=DEV)
        else:
            cache = torch.empty(shape, dtype=torch.uint8, device=DEV)
        lens, tables, tb, tp, _ = build_paged(c, T, seed=13)
        bt = flat_tables(lens, tables)
        q = torch.randn(1, c["HK"] * c["HREP"], c["D"]).to(DEV).to(torch.bfloat16)
        try:
            out = native_call(c, q, cache, fmt, [T], bt, window=c["W"])
            outf = out.float().cpu()
            nf = int((~torch.isfinite(outf)).sum())
            bad = [h for h in range(outf.shape[1]) if not torch.isfinite(outf[0, h]).all()]
            print(f"  note uninit/{cell}/{fmt}/{kind}: non_finite={nf}/{outf.numel()} "
                  f"bad_heads={bad[:4]}")
            _rec("uninit", cell, fmt, kind=kind, nonfinite=nf, numel=outf.numel(),
                 bad_heads=bad[:8])
        except Exception as exc:  # noqa: BLE001 — characterize, never crash
            print(f"  note uninit/{cell}/{fmt}/{kind}: EXC {type(exc).__name__}: {exc}")
            _rec("uninit", cell, fmt, kind=kind, exc=f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ #
# FACE: ctrl (the certified qwen-cell regression guard)               #
# ------------------------------------------------------------------ #

def face_ctrl(c, fmt, T=549):
    lens, tables, tb, tp, n_blocks = build_paged(c, T, seed=99)
    bt = flat_tables(lens, tables)
    cache = torch.zeros(n_blocks, c["BLOCK"], c["HK"], slot_bytes(c["D"], fmt),
                        dtype=torch.uint8, device=DEV)
    k, v = make_sources(c, T, seed=4242)
    mx_store(k, v, cache, tb, tp)
    q = torch.randn(1, c["HK"] * c["HREP"], c["D"],
                    generator=torch.Generator().manual_seed(5)).to(DEV).to(torch.bfloat16)
    kq, vq = dequant_rows(c, cache, tb, tp, list(range(T)), fmt)
    ref = ref_decode_rows(c, q.squeeze(0).float().cpu(), kq, vq, T)
    # the wheel's native windowless decode (the r242 authority lane)
    out = native_call(c, q, cache, fmt, lens, bt, window=None)
    m_nat = gate("ctrl", "ctrl", fmt, f"native T={T} d256p64", out.squeeze(0), ref)
    # the ENGINE-TRUE lane for every qwen windowless cell: Triton with
    # the wrapper default -- 64 == this pool's page, so the default
    # call is CORRECT here (qwen's immunity to D-B, at the engine's
    # exact call shape)
    sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
    out_t = triton_decode(q, cache, bt, sl, 1.0 / math.sqrt(c["D"]))
    m_tri = gate("ctrl", "ctrl", fmt, f"triton-default T={T} d256p64",
                 out_t.squeeze(0), ref)
    _rec("ctrl", "ctrl", fmt, native=m_nat, triton=m_tri)


# ------------------------------------------------------------------ #

def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[p267] canary start: cells={list(CELLS)} fmts={FMTS}")
    print(f"[p267] torch={torch.__version__} dev={torch.xpu.get_device_name(0)}")
    print(f"[p267] provenance: mxkv_dir={_MXKV_DIR}")
    for f in ("mx_quant.py", "triton_mx_store.py", "triton_mx_decode.py"):
        print(f"[p267]   {f} md5={_md5(_MXKV_DIR / f)}")
    import vllm_xpu_kernels.mx_attn_interface as _mxi

    print(f"[p267]   mx_attn_interface={_mxi.__file__} md5={_md5(Path(_mxi.__file__))}")
    torch.manual_seed(7)

    for fmt in FMTS:
        print(f"[face] ctrl {fmt}")
        face_ctrl(CELLS["ctrl"], fmt)
    for cell in ("slide", "glob"):
        c = CELLS[cell]
        for fmt in FMTS:
            print(f"[face] stepwise {cell} {fmt}")
            face_stepwise(c, cell, fmt)
    for cell in ("slide", "glob"):
        for fmt in FMTS:
            print(f"[face] widewin {cell} {fmt}")
            face_widewin(CELLS[cell], cell, fmt)
    for fmt in FMTS:
        print(f"[face] pagewidth synthp128 {fmt}")
        face_pagewidth(fmt)
    for cell in ("slide", "glob"):
        for fmt in ("mxfp8", "mxk8v4"):
            print(f"[face] autosplit {cell} {fmt}")
            face_autosplit(CELLS[cell], cell, fmt)
    for cell in ("slide", "glob"):
        face_uninit(CELLS[cell], cell, "mxfp8")

    d_b = any(
        R.get("pagewidth", {}).get(cell, {}).get(fmt, {}).get("d_b_reproduced")
        for cell in R.get("pagewidth", {})
        for fmt in R["pagewidth"][cell]
    )
    if not FAILED:
        verdict = (
            "PASS with D_B proven — kernel/store math green at the engine's "
            "true lanes and pages; the synthetic p128 cell shows the wrapper "
            "default-64 walk corrupts past ln=64 while the true-page walk "
            "holds (latent-hardening evidence: land the wrapper fix, "
            "block_size None -> cache.shape[1])"
            if d_b
            else "PASS — canary clean: the engine defect is metadata-value "
            "wiring (G3), not kernel/store math; D-B was not reproduced on "
            "the synthetic cell (investigate the page assumption)"
        )
    else:
        verdict = f"FAIL — {len(FAILED)} offline reproductions"
    print(f"\n[p267] VERDICT: {verdict}")
    for f in FAILED:
        print(f"[p267]   {f}")
    (outdir / "p267_canary.json").write_text(
        json.dumps({"faces": R, "failed": FAILED, "verdict": verdict}, indent=1))
    print(f"[p267] wrote {outdir / 'p267_canary.json'}")
    sys.exit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
