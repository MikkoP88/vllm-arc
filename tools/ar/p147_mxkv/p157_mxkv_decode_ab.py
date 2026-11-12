#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p157_mxkv_decode_ab.py — P6.47 leg A4: offline XPU A/B of the split-KV
decode v2 (ARC_MXKV_DECODER=v2, triton_mx_decode.py) against the certified
v1 (decode-perf research results/2026-11-12-p647-decode-perf-research/).

Runs INSIDE a cert24x-family container with an XPU device visible (no engine
boot). Gates, in order:

  exact    v2 (join + shift unpack flavors) vs fp64 reference attention on
           the SAME dequantized K/V — the p147 decode-face gates
           (max|d| <= 2e-3, cos >= 0.99999), on random / adversarial /
           boundary-length fragmented tables.
  v1delta  v2 vs v1 on the SAME cache — split reassociation bound
           (max|d| <= 2e-3): v1 is bit-frozen, so this isolates the
           scheduling change from everything else.
  telemetry  decode timings fmt x ctx {1K, 8K, 32K, 131072} at the qwen-like
           head config (HK=8, Hq=32): decode ms, unique-bytes GB/s, speedup
           vs v1 — the L1 (split-K) / L2 (shift unpack) evidence bank.

Exit 0 only if every gate passes. Output: <out>/p157_decode_ab.json +
human summary. Every write site uses newline="\\n" (p139 law).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

# mx_kv package by FILE PATH under a synthetic top-level name (p147 law:
# the installed wheel's regular package would win resolution otherwise).
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
offsets = mxkv.offsets
# v2 reads the unpack flavor from the module global at call time — the
# harness flips it per leg (no reload needed).
tmx = sys.modules["mx_kv_lane.triton_mx_decode"]
mxd = importlib.import_module("mx_kv_lane.mx_dequant")
mx_dequant_cache = mxd.mx_dequant_cache

D = 128
HK = 2
HREP = 4  # correctness faces: Hq = 8 (p147 shape)
HK_T = 8  # telemetry face: qwen-like Hq = 32
BLOCK = 64
DEV = "xpu"
SCALE = 1.0 / math.sqrt(D)
FMTS = ("mxfp8", "mxfp4", "mxk8v4")
KENC = {"mxfp8": encode_mxfp8, "mxfp4": encode_mxfp4, "mxk8v4": encode_mxfp8}
VENC = {"mxfp8": encode_mxfp8, "mxfp4": encode_mxfp4, "mxk8v4": encode_mxfp4}
VARIANTS = ("v1", "v2join", "v2shift")
R = {}  # results tree
FAILED = []


def _rec(face, key, **kv):
    R.setdefault(face, {}).setdefault(key, {}).update(kv)


def _fail(face, key, msg):
    FAILED.append(f"{face}/{key}: {msg}")
    print(f"  FAIL {face}/{key}: {msg}")


def _ok(face, key, msg):
    print(f"  ok   {face}/{key}: {msg}")


def cos_sim(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def kslot(x, fmt):
    enc, codes = KENC[fmt](x)
    return torch.cat([enc, codes], dim=-1)


def vslot(x, fmt):
    enc, codes = VENC[fmt](x)
    return torch.cat([enc, codes], dim=-1)


def run_variant(variant, q, cache, bt, sl, block_n=None, num_warps=None, num_stages=None):
    if variant == "v1":
        return mx_decode(q, cache, bt, sl, SCALE, decoder="v1")
    tmx._SHIFT = variant == "v2shift"
    kw = {}
    if block_n is not None:
        kw["block_n"] = block_n
    if num_warps is not None:
        kw["num_warps"] = num_warps
    if num_stages is not None:
        kw["num_stages"] = num_stages
    return mx_decode(q, cache, bt, sl, SCALE, decoder="v2", **kw)


def build_paged(total_tokens, seed=7):
    """Fragmented block tables (p147 law: shuffled physical blocks)."""
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
    hq_n = q.shape[1]
    hrep = hq_n // k_deq.shape[1]
    t0 = 0
    for b, ln in enumerate(lens):
        kb = k_deq[t0 : t0 + ln].double()
        vb = v_deq[t0 : t0 + ln].double()
        for hq in range(hq_n):
            hk = hq // hrep
            s = (kb[:, hk, :] @ q[b, hq].double()) * SCALE
            outs.append(torch.softmax(s, dim=0) @ vb[:, hk, :])
        t0 += ln
    return torch.stack(outs).reshape(len(lens), hq_n, D)


def _dequant_cache(cache, tb, tp, fmt):
    got = cache[tb.long().cpu(), tp.long().cpu(), :, :].cpu()
    ng = D // 32
    kwidth = D + ng if fmt in ("mxfp8", "mxk8v4") else D // 2 + ng
    return decode_kv(got[..., :kwidth].contiguous(), got[..., kwidth:].contiguous(), fmt, D)


def face_exact(tag, xk, xv, q, lens, tables, tb, tp, n_blocks):
    """exact + v1delta gates for every (fmt, v2 variant) on one case."""
    for fmt in FMTS:
        cache = torch.zeros(
            n_blocks, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables(lens, tables)
        sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
        outs = {}
        for variant in VARIANTS:
            out = run_variant(variant, q, cache, bt, sl)
            torch.xpu.synchronize()
            outs[variant] = out.cpu()
        k_deq, v_deq = _dequant_cache(cache, tb, tp, fmt)
        ref = _ref_attention(q.cpu(), k_deq, v_deq, lens)
        for variant in VARIANTS:
            if not torch.isfinite(outs[variant]).all():
                _fail("exact", f"{fmt}/{variant}", f"non-finite output ({tag})")
                continue
            d_exact = (outs[variant] - ref).abs().max().item()
            c_same = cos_sim(outs[variant], ref)
            gated = d_exact <= 2e-3 and c_same >= 0.99999
            if variant != "v1":
                d_v1 = (outs[variant] - outs["v1"]).abs().max().item()
                gated = gated and d_v1 <= 2e-3
            else:
                d_v1 = 0.0
            if not gated:
                _fail(
                    "exact",
                    f"{fmt}/{variant}",
                    f"{tag}: vs-exact {d_exact:.2e} cos {c_same:.7f} vs-v1 {d_v1:.2e}",
                )
            else:
                _ok(
                    "exact",
                    f"{fmt}/{variant}",
                    f"{tag}: d={d_exact:.2e} c={c_same:.7f} v1delta={d_v1:.2e}",
                )
            _rec(
                "exact",
                f"{fmt}/{variant}",
                **{
                    tag: dict(
                        maxdiff=round(d_exact, 8),
                        cos_same=round(c_same, 8),
                        v1_delta=round(d_v1, 8),
                    )
                },
            )


def faces_exact():
    print("[face: exact] v2 join+shift vs fp64 exact + vs v1 (XPU)")
    gen = torch.Generator().manual_seed(99)
    # random: 549 tokens in 2 seqs crossing block boundaries (p147 shape)
    xk = torch.randn(549, HK, D, generator=gen).to(DEV)
    xv = torch.randn(549, HK, D, generator=gen).to(DEV)
    lens, tables, tb, tp, n_blocks = build_paged(549)
    q = torch.randn(len(lens), HK * HREP, D, generator=gen).to(DEV)
    face_exact("random549", xk, xv, q, lens, tables, tb, tp, n_blocks)
    # adversarial rows (p147 robust subset: the zero and outlier extremes).
    # xv stays nonzero-random (p147 law): zero V makes every output exactly
    # zero and cos_sim(0,0) undefined — the A4 first-run artifact (d=0.0,
    # cos=nan on ALL variants incl. v1: case bug, not kernel bug).
    xk = torch.randn(128, HK, D, generator=gen) * 1e-3
    xk[3, 0, 10] = 4e2
    xk[7, 1, :32] = 0.0
    xv = torch.randn(128, HK, D, generator=gen) * 0.1
    lens, tables, tb, tp, n_blocks = build_paged(128, seed=11)
    q = torch.randn(len(lens), HK * HREP, D, generator=gen).to(DEV)
    face_exact("adversarial", xk.to(DEV), xv.to(DEV), q, lens, tables, tb, tp, n_blocks)
    # boundary lengths: 1 (single token, empty-split parks), 63/64/65
    # (block-edge), all through fragmented tables
    for ln in (1, 63, 64, 65):
        xk = torch.randn(ln, HK, D, generator=gen).to(DEV)
        xv = torch.randn(ln, HK, D, generator=gen).to(DEV)
        lens, tables, tb, tp, n_blocks = build_paged(ln, seed=ln)
        q = torch.randn(len(lens), HK * HREP, D, generator=gen).to(DEV)
        face_exact(f"len{ln}", xk, xv, q, lens, tables, tb, tp, n_blocks)


def face_telemetry():
    print("[face: telemetry] decode A/B fmt x ctx at HK=8 Hq=32 (XPU)")
    prop = torch.xpu.get_device_properties(0)
    R.setdefault("telemetry", {})["device"] = {
        "name": prop.name,
        "count": torch.xpu.device_count(),
    }
    for fmt in FMTS:
        for ctx in (1024, 8192, 32768, 131072):
            xk = torch.randn(ctx, HK_T, D, device=DEV)
            xv = torch.randn(ctx, HK_T, D, device=DEV)
            lens = [ctx]  # B=1: the occupancy-starved solo regime v2 targets
            tables, tok_block, tok_pos = [], [], []
            n_blocks = ctx // BLOCK + 2
            tab = list(range(n_blocks))
            tables.append(torch.tensor(tab[: (ctx + BLOCK - 1) // BLOCK], dtype=torch.int32, device=DEV))
            for t in range(ctx):
                tok_block.append(tab[t // BLOCK])
                tok_pos.append(t % BLOCK)
            tb = torch.tensor(tok_block, dtype=torch.int32, device=DEV)
            tp = torch.tensor(tok_pos, dtype=torch.int32, device=DEV)
            cache = torch.zeros(
                n_blocks, BLOCK, HK_T, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
            )
            mx_store(xk, xv, cache, tb, tp)
            torch.xpu.synchronize()
            bt = flat_tables(lens, tables)
            sl = torch.tensor(lens, dtype=torch.int32, device=DEV)
            q = torch.randn(1, HK_T * HREP, D, device=DEV)
            timings = {}
            outs = {}
            for variant in VARIANTS:
                for _ in range(3):  # warmup + compile
                    run_variant(variant, q, cache, bt, sl)
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    run_variant(variant, q, cache, bt, sl)
                torch.xpu.synchronize()
                timings[variant] = (time.perf_counter() - t0) / 20
                outs[variant] = run_variant(variant, q, cache, bt, sl)
            b_unique = ctx * HK_T * slot_bytes(D, fmt)
            row = {
                "decode_ms": {v: round(timings[v] * 1e3, 4) for v in VARIANTS},
                "gbps_unique": {
                    v: round(b_unique / timings[v] / 1e9, 2) for v in VARIANTS
                },
                "speedup_vs_v1": {
                    v: round(timings["v1"] / timings[v], 3)
                    for v in VARIANTS
                    if v != "v1"
                },
                # cross-decoder agreement at full ctx (device-side; the
                # fp64 CPU ref is impractical at 131072)
                "v1_delta_max": {
                    v: round(
                        float(
                            (outs[v] - outs["v1"]).abs().amax().item()
                        ),
                        8,
                    )
                    for v in VARIANTS
                    if v != "v1"
                },
                "slot_bytes": slot_bytes(D, fmt),
            }
            bad_delta = max(row["v1_delta_max"].values()) > 2e-3
            if bad_delta:
                _fail("telemetry", f"{fmt}", f"ctx={ctx}: v2 deviates from v1 > 2e-3")
            else:
                _ok(
                    "telemetry",
                    f"{fmt}",
                    f"ctx={ctx}: v1 {row['decode_ms']['v1']:.3f} ms, "
                    f"v2join {row['decode_ms']['v2join']:.3f} ms "
                    f"(x{row['speedup_vs_v1']['v2join']}), "
                    f"v2shift {row['decode_ms']['v2shift']:.3f} ms "
                    f"(x{row['speedup_vs_v1']['v2shift']}), "
                    f"v1delta<={max(row['v1_delta_max'].values()):.2e}",
                )
            _rec("telemetry", fmt, **{str(ctx): row})


def face_dequant():
    """Prefill-axis face: bulk-dequant (mx_dequant_cache) timings fmt x ctx
    for both unpack flavors — the large-continuation prefill cost that the
    superiority-axes law (2026-11-12) puts in scope. Content sanity at 8K
    vs the reference decode (loose gate; the strict content gates are the
    p147/engine xchecks)."""
    print("[face: dequant] bulk-dequant A/B fmt x ctx (XPU)")
    for fmt in FMTS:
        for ctx in (8192, 32768, 131072):
            n_blocks = ctx // BLOCK + 2
            tab = list(range(n_blocks))
            tables = [torch.tensor(tab[: (ctx + BLOCK - 1) // BLOCK], dtype=torch.int32, device=DEV)]
            tb = torch.tensor(
                [tab[t // BLOCK] for t in range(ctx)], dtype=torch.int32, device=DEV
            )
            tp = torch.tensor([t % BLOCK for t in range(ctx)], dtype=torch.int32, device=DEV)
            cache = torch.zeros(
                n_blocks, BLOCK, HK_T, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
            )
            xk = torch.randn(ctx, HK_T, D, device=DEV)
            xv = torch.randn(ctx, HK_T, D, device=DEV)
            mx_store(xk, xv, cache, tb, tp)
            torch.xpu.synchronize()
            bt = flat_tables([ctx], tables)
            alloc = (ctx + BLOCK - 1) // BLOCK * BLOCK
            kb = torch.empty(1, HK_T, alloc, D, dtype=torch.float16, device=DEV)
            vb = torch.empty(1, HK_T, alloc, D, dtype=torch.float16, device=DEV)
            timings = {}
            for flavor in ("join", "shift"):
                tmx._SHIFT = flavor == "shift"
                for _ in range(3):
                    mx_dequant_cache(cache, bt, kb, vb, ctx)
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    mx_dequant_cache(cache, bt, kb, vb, ctx)
                torch.xpu.synchronize()
                timings[flavor] = (time.perf_counter() - t0) / 20
            tmx._SHIFT = False
            # content sanity at 8K: first 64 tokens vs reference decode
            d_content = -1.0
            if ctx == 8192:
                got = cache[tb.long().cpu(), tp.long().cpu(), :, :].cpu()
                ng = D // 32
                kw = D + ng if fmt in ("mxfp8", "mxk8v4") else D // 2 + ng
                rk, rv = decode_kv(
                    got[:64][..., :kw].contiguous(), got[:64][..., kw:].contiguous(), fmt, D
                )
                d_content = max(
                    float((kb[0, :, :64].float().cpu() - rk.permute(1, 0, 2)).abs().max()),
                    float((vb[0, :, :64].float().cpu() - rv.permute(1, 0, 2)).abs().max()),
                )
                if d_content > 0.01:
                    _fail("dequant", fmt, f"ctx={ctx}: content maxdiff {d_content:.4f}")
            b_read = ctx * HK_T * slot_bytes(D, fmt)
            b_write = ctx * HK_T * 2 * D * 2
            row = {
                "ms": {f: round(timings[f] * 1e3, 4) for f in timings},
                "gbps_rw": {
                    f: round((b_read + b_write) / timings[f] / 1e9, 2) for f in timings
                },
                "speedup_shift": round(timings["join"] / timings["shift"], 3),
                "content_maxdiff_8k": d_content,
            }
            _ok(
                "dequant",
                fmt,
                f"ctx={ctx}: join {row['ms']['join']:.3f} ms, "
                f"shift {row['ms']['shift']:.3f} ms "
                f"(x{row['speedup_shift']}), "
                + (f"content d={d_content:.4f}" if d_content >= 0 else "content n/a"),
            )
            _rec("dequant", fmt, **{str(ctx): row})


def face_tune():
    """Tuning sweep: v2join (BLOCK_N, num_warps, num_stages) per fmt at
    32K — two independent levers. Register pressure: the double-e2m1
    mxfp4 leg regressed to 0.87x at (128,4) in A4. Latency: A4 mxfp8
    @131K ran at ~4% ALU / <=36% DRAM of peak, i.e. the tile loop's
    dependent load chain (bt -> slot -> payload) is the bound — SLM
    double-buffered prefetch (num_stages>1) is the pipelining fix.
    Winner combo gets a correctness spot vs the fp64 reference."""
    print("[face: tune] v2 (BLOCK_N, num_warps, num_stages) sweep @ ctx=32768 (XPU)")
    combos = tuple(
        (bn, nw, ns)
        for bn, nw in ((64, 4), (64, 8), (128, 4), (128, 8))
        for ns in (1, 2, 3)
    )
    ctx = 32768
    for fmt in FMTS:
        n_blocks = ctx // BLOCK + 2
        tab = list(range(n_blocks))
        tables = [torch.tensor(tab[: (ctx + BLOCK - 1) // BLOCK], dtype=torch.int32, device=DEV)]
        tb = torch.tensor(
            [tab[t // BLOCK] for t in range(ctx)], dtype=torch.int32, device=DEV
        )
        tp = torch.tensor([t % BLOCK for t in range(ctx)], dtype=torch.int32, device=DEV)
        cache = torch.zeros(
            n_blocks, BLOCK, HK_T, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        xk = torch.randn(ctx, HK_T, D, device=DEV)
        xv = torch.randn(ctx, HK_T, D, device=DEV)
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()
        bt = flat_tables([ctx], tables)
        sl = torch.tensor([ctx], dtype=torch.int32, device=DEV)
        q = torch.randn(1, HK_T * HREP, D, device=DEV)
        best = None
        rows = {}
        for bn, nw, ns in combos:
            try:
                for _ in range(3):
                    run_variant(
                        "v2join", q, cache, bt, sl, block_n=bn, num_warps=nw,
                        num_stages=ns,
                    )
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    run_variant(
                        "v2join", q, cache, bt, sl, block_n=bn, num_warps=nw,
                        num_stages=ns,
                    )
                torch.xpu.synchronize()
                ms = (time.perf_counter() - t0) / 20 * 1e3
            except Exception as exc:  # noqa: BLE001 — record, keep the sweep alive
                rows[f"n{bn}w{nw}s{ns}"] = f"err:{type(exc).__name__}"
                continue
            rows[f"n{bn}w{nw}s{ns}"] = round(ms, 4)
            if best is None or ms < best[3]:
                best = (bn, nw, ns, ms)
        if best is None:
            _fail("tune", fmt, f"all combos errored: {rows}")
            _rec("tune", fmt, sweep_ms=rows, winner=None)
            continue
        # correctness spot at the winner combo (tuning changes only
        # scheduling; the gate proves it: vs fp64 exact on a 549-token
        # fragmented case)
        T = 549
        gen = torch.Generator().manual_seed(31)
        xk2 = torch.randn(T, HK, D, generator=gen).to(DEV)
        xv2 = torch.randn(T, HK, D, generator=gen).to(DEV)
        lens, tables2, tb2, tp2, nb2 = build_paged(T, seed=31)
        cache2 = torch.zeros(
            nb2, BLOCK, HK, slot_bytes(D, fmt), dtype=torch.uint8, device=DEV
        )
        mx_store(xk2, xv2, cache2, tb2, tp2)
        torch.xpu.synchronize()
        q2 = torch.randn(len(lens), HK * HREP, D, generator=gen).to(DEV)
        out = run_variant(
            "v2join", q2, cache2, flat_tables(lens, tables2),
            torch.tensor(lens, dtype=torch.int32, device=DEV),
            block_n=best[0], num_warps=best[1], num_stages=best[2],
        ).cpu()
        got = cache2[tb2.long().cpu(), tp2.long().cpu(), :, :].cpu()
        ng = D // 32
        kw = D + ng if fmt in ("mxfp8", "mxk8v4") else D // 2 + ng
        rk, rv = decode_kv(got[..., :kw].contiguous(), got[..., kw:].contiguous(), fmt, D)
        ref = _ref_attention(q2.cpu(), rk, rv, lens)
        d_exact = (out - ref).abs().max().item()
        if d_exact > 2e-3 or not torch.isfinite(out).all():
            _fail(
                "tune", fmt,
                f"winner n{best[0]}w{best[1]}s{best[2]}: vs-exact {d_exact:.2e}",
            )
        else:
            _ok(
                "tune",
                fmt,
                f"winner BLOCK_N={best[0]} warps={best[1]} stages={best[2]} "
                f"{best[3]:.3f} ms (sweep {rows}); correctness d={d_exact:.2e}",
            )
        _rec(
            "tune", fmt, sweep_ms=rows, winner=[best[0], best[1], best[2]],
            maxdiff=round(d_exact, 8),
        )


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    print(f"p157_mxkv_decode_ab — device={DEV}, torch={torch.__version__}")
    faces_exact()
    face_telemetry()
    face_tune()
    face_dequant()
    R["verdict"] = "PASS" if not FAILED else "FAIL"
    R["failures"] = FAILED
    with open(out / "p157_decode_ab.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(R, f, indent=2, sort_keys=True)
    n = len(FAILED)
    print(f"\nVERDICT: {R['verdict']}" + (f" ({n} failures)" if n else ""))
    sys.exit(0 if not n else 1)


if __name__ == "__main__":
    main()
