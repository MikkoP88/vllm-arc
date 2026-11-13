# SPDX-License-Identifier: Apache-2.0
"""triton_mx_decode.py — MXFP4/MXFP8/MXK8V4 paged attention decode kernel (P6.47,
vllm_gaudi/xpu/patches/mx_kv/).

PURPOSE: single-query-token flash decode from the paged uint8 MX KV
cache — the MX analogue of the wheel's triton_turboquant_decode.py on
the same AoS slot geometry (mx_layout.py). Per (batch, q-head)
program: walk the block table in BLOCK_N token tiles, load packed
slots, dequantize K inline (E8M0 group scales + e4m3/e2m1 elements),
score q.k, run the online-softmax accumulator, dequantize V the same
way, and write the attention row out.

DEQUANT PATHS (bit-law: element math mirrors mx_quant._decode_e2m1
and the fp8 hardware cast):
  mxfp8: bytes bitcast e4nv->f32, x scale where scale =
         exp2(code - 127) broadcast 1-per-32-lanes.
  mxfp4: byte -> (lo, hi) nibbles (even element low); idx = nib & 7,
         e = (nib & 6) >> 1, m = nib & 1,
         mag = (e == 0) ? m * 0.5 : 2^(e-1) * (1 + 0.5 * m);
         sign = nib >= 8; lanes interleaved via tl.join + reshape.
  The interleave matches the store pack (even element low nibble) and
  the mx_quant reference byte layout.

EFFICIENCY (always-loop directive): scale metadata is 1 B per 32
elements per row (TQ pays 4 B fp16 scale+zero per row) and e2m1
decode is pure shift/compare arithmetic — no tables, no extra
gathers; masked lanes contribute exp(-inf) = 0 so no special casing.

V2 SPLIT-KV DECODER (decode-perf research 2026-11-12,
results/2026-11-12-p647-decode-perf-research/): v1's grid (bsz, Hq)
walks seq_len/BLOCK_N tiles serially on one program per q-head —
measured 6.5-7.2 GB/s effective at 32K ctx (mxfp8), an order of
magnitude under DRAM: latency-bound, not bandwidth-bound. The wheel's
own TQ decode (the incumbent fp8 baseline path) is split-KV scheduled
all along. V2 ports that structure onto the MX geometry:
  stage 1 `_mx_decode_split` grid (B, Hq, 32): each program walks ONE
  contiguous split with v1's per-row math verbatim (fused fp32
  reductions, no tl.dot — no precision-mode questions) and stores the
  partial row [:D]=acc/l, [D]=lse into a cached mid buffer; the
  per-batch EFFECTIVE split count s_eff = clamp(ceil(tiles/8), 1, 32)
  is derived IN-KERNEL from the device-resident seq_len — short
  sequences collapse to few splits without any host read, and surplus
  programs park an explicit empty marker (lse=-inf; the wheel's
  early-return leaves stale mid bytes — ours cannot leak stale state).
  stage 2 `_mx_decode_combine`: standard log-sum-exp weighted reduce
  over splits, all-empty guard preserving v1's zero-seq semantics.
The e2m1 unpack has two flavors on the shared dequant helper: "join"
(v1's join/reshape interleave — bit-frozen) and "shift"
(duplicated-load + dynamic nibble shift, pure vector ALU, no
cross-lane op) — the A/B leg measures before anything defaults.
Decoder selection: ARC_MXKV_DECODER=v1|v2|v3 (default v2 since the A9
true-shape rebake — BLOCK_N=32 per-q-head split wins every serve cell at
D=256/HK=4/Hq=24; v3, the GQA-batched split, keeps the solo-short-ctx
niche and stays the A/B face; v1 stays the bit-frozen escape hatch every
pre-A4 banked verdict rode); ARC_MXKV_UNPACK=join|shift overrides the
baked per-format flavor (v2/v3 only).

V3 GQA-BATCHED SPLIT (leg A6 evidence): num_stages pipelining is a
no-op on the v2 tile loop (s1/s2/s3 within noise on every combo) —
the walk is bound by the data-dependent pointer chase (block-table
load -> slot address), which the compiler cannot prefetch across.
`_mx_decode_split_gqa` grid (B, Hk, 32): one program serves all
N_REP q-heads of a kv-head, dequanting K/V once per tile (the chase
amortizes N_REP-fold instead of re-walking the same slice per q-head)
with N_REP unrolled rows of (q, m, l, acc) state under
tl.static_range — no tl.dot, no 3D broadcast materialization, v1's
per-row math verbatim. A5/A6 tune-table winners (the (BLOCK_N,
num_warps, num_stages) factorial, results/.../legA5 + legA6):
e4m3-payload mxfp8 takes (128, 4); the sub-byte formats spill at
BLOCK_N=128 (a 17-20 ms cliff vs 2.4-6.0 ms winners) and take (64,
8) — the table is baked into `_MX_TUNE` and applied per format
unless the caller overrides.

GRAPH SAFETY: seq lens and block tables are read from device buffers
inside the kernel (the TQ decode pattern) — no host reads, no
data-dependent host control flow; XPU-graph safe. The mid buffer is
allocated once per shape on the eager warmup and reused (allocation
can never land inside a capture).

Provenance: from-scratch in-house lane (owner directive 2026-11-11);
kernel structure follows the wheel's sub-byte KV decode template
(triton_turboquant_decode.py — survey
results/2026-11-11-p647-mxkv-research/wheelref/).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .mx_layout import slot_bytes

# Decoder selection (A9 true-shape rebake, 2026-11-12): v2 (per-q-head
# split-KV) with the BLOCK_N=32 table is the serve default — at the real
# qwen geometry (D=256, HK=4, Hq=24) v2 beats the v3 GQA walk at every
# B>1 cell and all long-ctx cells; v3 keeps the solo-short-ctx niche
# (B=1 @1K, ~2x) — reachable via ARC_MXKV_DECODER=v3, as are the v1
# bit-frozen escape hatch and v3's A/B face. Read once at import — the
# engine's env law.
_DECODER = os.environ.get("ARC_MXKV_DECODER", "v2")
# e2m1 unpack-flavor OVERRIDE for the split decoders ("join" = v1's
# interleave, bit-frozen; "shift" = duplicated-load + dynamic nibble
# shift). None (no env) -> per-format _MX_TUNE flavor; the A/B harness
# flips this module attr per variant (call-time read propagates).
_ENV_UNPACK = os.environ.get("ARC_MXKV_UNPACK", "")
_SHIFT = (_ENV_UNPACK == "shift") if _ENV_UNPACK else None


@triton.jit
def _mx_dequant_e2m1(nib):
    # nib uint8 -> f32 value on the e2m1 grid (mx_quant._decode_e2m1 mirror).
    e = (nib & 0x6).to(tl.float32) / 2.0
    m = (nib & 0x1).to(tl.float32)
    mag = tl.where(e == 0.0, m * 0.5, tl.exp2(e - 1.0) * (1.0 + 0.5 * m))
    sgn = 1.0 - 2.0 * (nib >= 8).to(tl.float32)
    return mag * sgn


@triton.jit
def _mx_dequant_block(
    cache_ptr,
    slot,
    payload_off,  # 0 for K, v_off for V
    mask,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    N_GROUPS: tl.constexpr,
    SUB4: tl.constexpr,  # payload is e2m1 nibbles (else e4m3 bytes)
    SHIFT: tl.constexpr = 0,  # 1 = shift-unpack (v2 A/B flavor; v1 stays join)
):
    # slot: [BLOCK_N] int64 slot byte bases; mask: [BLOCK_N] valid lanes.
    # Returns f32 [BLOCK_N, D] dequantized K-or-V rows.
    g = tl.arange(0, N_GROUPS)
    gmask = mask[:, None]
    if SUB4:
        codes = tl.load(cache_ptr + slot[:, None] + payload_off + D // 2 + g[None, :], mask=gmask, other=0)
        scale = tl.exp2(codes.to(tl.float32) - 127.0)
        if SHIFT:
            # duplicated-load interleave: element d's byte is d//2, its
            # nibble shift is (d%2)*4 — pure vector ALU, no join/reshape
            # cross-lane op (semantics identical to the join path).
            dd = tl.arange(0, D)
            raw = tl.load(
                cache_ptr + slot[:, None] + payload_off + (dd // 2)[None, :],
                mask=gmask,
                other=0,
            ).to(tl.int32)
            nib = (raw >> ((dd % 2) * 4)[None, :]) & 0xF
            elems = _mx_dequant_e2m1(nib)
        else:
            raw = tl.load(
                cache_ptr + slot[:, None] + payload_off + tl.arange(0, D // 2)[None, :],
                mask=gmask,
                other=0,
            )
            lo = _mx_dequant_e2m1(raw & 0xF)
            hi = _mx_dequant_e2m1(raw >> 4)
            elems = tl.reshape(tl.join(lo, hi), (BLOCK_N, D))
        sc = tl.reshape(tl.broadcast_to(scale[:, :, None], (BLOCK_N, N_GROUPS, 32)), (BLOCK_N, D))
        return elems * sc
    else:
        codes = tl.load(cache_ptr + slot[:, None] + payload_off + D + g[None, :], mask=gmask, other=0)
        scale = tl.exp2(codes.to(tl.float32) - 127.0)
        raw = tl.load(
            cache_ptr + slot[:, None] + payload_off + tl.arange(0, D)[None, :],
            mask=gmask,
            other=0,
        )
        kf = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        sc = tl.reshape(tl.broadcast_to(scale[:, :, None], (BLOCK_N, N_GROUPS, 32)), (BLOCK_N, D))
        return kf * sc


@triton.jit
def _mx_decode(
    q_ptr,
    cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    out_ptr,
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_o_b,
    stride_o_h,
    stride_bt_b,
    stride_c_b,
    stride_c_t,
    stride_c_h,
    D: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K4: tl.constexpr,
    V4: tl.constexpr,
    N_REP: tl.constexpr,  # GQA: Hq // Hk
    BLOCK_SIZE: tl.constexpr,  # cache block (tokens per block)
    BLOCK_N: tl.constexpr,  # decode tile
):
    b = tl.program_id(0)
    hq = tl.program_id(1)
    hk = hq // N_REP

    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + b * stride_q_b + hq * stride_q_h + offs_d)

    seq_len = tl.load(seq_lens_ptr + b)
    # K block size decides V's offset (mx_layout law; mxk8v4 = e4m3 K +
    # e2m1 V, the turboquant_k8v4 quality doctrine).
    v_off = (D // 2 if K4 else D) + N_GROUPS

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < seq_len
        blk = tl.load(
            block_tables_ptr + b * stride_bt_b + offs_n // BLOCK_SIZE,
            mask=nmask,
            other=0,
        )
        slot = (
            blk.to(tl.int64) * stride_c_b
            + (offs_n % BLOCK_SIZE).to(tl.int64) * stride_c_t
            + hk.to(tl.int64) * stride_c_h
        )
        k = _mx_dequant_block(cache_ptr, slot, 0, nmask, BLOCK_N, D, N_GROUPS, K4)
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
        scores = tl.where(nmask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        v = _mx_dequant_block(cache_ptr, slot, v_off, nmask, BLOCK_N, D, N_GROUPS, V4)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # zero-length sequences (never scheduled by the engine, but the
    # offline harness probes them): 0/0 -> defined zeros, no NaN.
    out = tl.where(l_i == 0.0, tl.zeros((D,), dtype=tl.float32), acc / l_i)
    tl.store(out_ptr + b * stride_o_b + hq * stride_o_h + offs_d, out)


@triton.jit
def _mx_decode_split(
    q_ptr,
    cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    mid_ptr,  # [B, Hq, NUM_SPLITS, D+1] f32: rows [:D]=acc/l, [D]=lse
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_bt_b,
    stride_c_b,
    stride_c_t,
    stride_c_h,
    D: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K4: tl.constexpr,
    V4: tl.constexpr,
    N_REP: tl.constexpr,  # GQA: Hq // Hk
    BLOCK_SIZE: tl.constexpr,  # cache block (tokens per block)
    BLOCK_N: tl.constexpr,  # decode tile
    NUM_SPLITS: tl.constexpr,  # grid z; effective count adapts per batch
    SHIFT: tl.constexpr,  # e2m1 unpack flavor (v2 A/B)
):
    # Stage 1 of the split-KV decode (the wheel's proven scheduling on
    # MX geometry): one program per (batch, q-head, split), v1's per-row
    # math verbatim inside the split range. Programs beyond the batch's
    # effective split count park the empty marker — mid ALWAYS gets a
    # written row, so stale bytes can never reach the combine.
    b = tl.program_id(0)
    hq = tl.program_id(1)
    sid = tl.program_id(2)
    hk = hq // N_REP

    seq_len = tl.load(seq_lens_ptr + b)
    # Per-batch effective splits from the DEVICE seq_len (no host read):
    # aim for >= 2 tiles of serial work per split (splitLen >= BLOCK_N is
    # not required — masked lanes handle the tail), bounded by the grid.
    # 1K ctx -> 4 splits, 8K+ -> 32: occupancy where v1 starves.
    n_tiles = (seq_len + BLOCK_N - 1) // BLOCK_N
    s_eff = tl.minimum(NUM_SPLITS, tl.maximum(1, (n_tiles + 1) // 2))
    split_len = (seq_len + s_eff - 1) // s_eff
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    offs_d = tl.arange(0, D)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    if split_start < split_end:
        q = tl.load(q_ptr + b * stride_q_b + hq * stride_q_h + offs_d)
        # K block size decides V's offset (mx_layout law; mxk8v4 = e4m3 K +
        # e2m1 V, the turboquant_k8v4 quality doctrine).
        v_off = (D // 2 if K4 else D) + N_GROUPS
        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < split_end
            blk = tl.load(
                block_tables_ptr + b * stride_bt_b + offs_n // BLOCK_SIZE,
                mask=nmask,
                other=0,
            )
            slot = (
                blk.to(tl.int64) * stride_c_b
                + (offs_n % BLOCK_SIZE).to(tl.int64) * stride_c_t
                + hk.to(tl.int64) * stride_c_h
            )
            k = _mx_dequant_block(cache_ptr, slot, 0, nmask, BLOCK_N, D, N_GROUPS, K4, SHIFT)
            scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
            scores = tl.where(nmask, scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, 0))
            p = tl.exp(scores - m_new)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 0)
            v = _mx_dequant_block(cache_ptr, slot, v_off, nmask, BLOCK_N, D, N_GROUPS, V4, SHIFT)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

    base = b * stride_mid_b + hq * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    tl.store(mid_ptr + base + offs_d, acc / safe_l)
    lse = tl.where(l_i > 0.0, m_i + tl.log(safe_l), -float("inf"))
    tl.store(mid_ptr + base + D, lse)


@triton.jit
def _mx_decode_split_gqa(
    q_ptr,
    cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    mid_ptr,  # [B, Hq, NUM_SPLITS, D+1] f32: rows [:D]=acc/l, [D]=lse
    sm_scale,
    stride_q_b,
    stride_q_h,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_bt_b,
    stride_c_b,
    stride_c_t,
    stride_c_h,
    D: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K4: tl.constexpr,
    V4: tl.constexpr,
    N_REP: tl.constexpr,  # GQA: Hq // Hk
    BLOCK_SIZE: tl.constexpr,  # cache block (tokens per block)
    BLOCK_N: tl.constexpr,  # decode tile
    NUM_SPLITS: tl.constexpr,  # grid z; effective count adapts per batch
    SHIFT: tl.constexpr,  # e2m1 unpack flavor (v2/v3 A/B)
):
    # v3 stage 1: same split math as v2, but one program per (batch,
    # kv-head, split) serving all N_REP q-heads of that kv-head. A6
    # proved the tile walk is bound by the data-dependent pointer chase
    # (bt -> slot: num_stages pipelining is a no-op across it), and v2
    # pays that chase N_REP times per tile through N_REP co-resident
    # programs re-walking the same slice. v3 dequants K/V once per tile
    # and keeps N_REP unrolled rows of (q, m, l, acc) state — trace-time
    # tuple-carried unrolling under tl.static_range, ~3 KB of extra
    # registers vs the spill cliff A5 mapped at BLOCK_N=128 sub-byte
    # payloads. mid layout and the combine stage are identical to v2.
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sid = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + b)
    n_tiles = (seq_len + BLOCK_N - 1) // BLOCK_N
    # v3 occupancy floor: ONE tile per split (not v2's >=2). The GQA
    # batch already cuts the grid 4x (Hk not Hq), so the >=2-tile floor
    # starves short ctx — A8: v3 @1K ran 0.531 ms vs v2join's 0.210
    # (64 vs 256 programs); 1-tile splits restore the program count and
    # with it the short-ctx win, at unchanged >=8K behavior (both cap
    # at NUM_SPLITS once n_tiles >= 32). Empty splits park lse=-inf.
    s_eff = tl.minimum(NUM_SPLITS, tl.maximum(1, n_tiles))
    split_len = (seq_len + s_eff - 1) // s_eff
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    offs_d = tl.arange(0, D)
    # Per-r unrolled state as TUPLES (the tracer coerces loop-captured
    # lists to tuples — append is illegal under static_range); tuple
    # concat + constexpr-index reads are the supported aggregate form.
    qs = ()
    mis = ()
    lis = ()
    accs = ()
    for r in tl.static_range(N_REP):
        qs = qs + (
            tl.load(
                q_ptr
                + b * stride_q_b
                + (hk * N_REP + r) * stride_q_h
                + offs_d
            ),
        )
        mis = mis + (-float("inf"),)
        lis = lis + (0.0,)
        accs = accs + (tl.zeros((D,), dtype=tl.float32),)

    if split_start < split_end:
        v_off = (D // 2 if K4 else D) + N_GROUPS
        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            nmask = offs_n < split_end
            blk = tl.load(
                block_tables_ptr + b * stride_bt_b + offs_n // BLOCK_SIZE,
                mask=nmask,
                other=0,
            )
            slot = (
                blk.to(tl.int64) * stride_c_b
                + (offs_n % BLOCK_SIZE).to(tl.int64) * stride_c_t
                + hk.to(tl.int64) * stride_c_h
            )
            k = _mx_dequant_block(
                cache_ptr, slot, 0, nmask, BLOCK_N, D, N_GROUPS, K4, SHIFT
            )
            v = _mx_dequant_block(
                cache_ptr, slot, v_off, nmask, BLOCK_N, D, N_GROUPS, V4, SHIFT
            )
            for r in tl.static_range(N_REP):
                scores = tl.sum(qs[r][None, :] * k, axis=1) * sm_scale
                scores = tl.where(nmask, scores, -float("inf"))
                m_new = tl.maximum(mis[r], tl.max(scores, 0))
                p = tl.exp(scores - m_new)
                alpha = tl.exp(mis[r] - m_new)
                lis = lis[:r] + (lis[r] * alpha + tl.sum(p, 0),) + lis[r + 1 :]
                accs = (
                    accs[:r]
                    + (accs[r] * alpha + tl.sum(p[:, None] * v, axis=0),)
                    + accs[r + 1 :]
                )
                mis = mis[:r] + (m_new,) + mis[r + 1 :]

    for r in tl.static_range(N_REP):
        hq = hk * N_REP + r
        base = b * stride_mid_b + hq * stride_mid_h + sid * stride_mid_s
        safe_l = tl.where(lis[r] > 0.0, lis[r], 1.0)
        tl.store(mid_ptr + base + offs_d, accs[r] / safe_l)
        lse = tl.where(lis[r] > 0.0, mis[r] + tl.log(safe_l), -float("inf"))
        tl.store(mid_ptr + base + D, lse)


@triton.jit
def _mx_decode_combine(
    mid_ptr,
    out_ptr,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    stride_o_b,
    stride_o_h,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,  # next_power_of_2(NUM_SPLITS)
):
    # Stage 2: log-sum-exp weighted combine of the split partials.
    # out = sum_s w_s * (acc_s/l_s) / sum_s w_s with w_s = exp(lse_s -
    # max_s lse_s) — the standard flash-decoding split reduction; empty
    # splits carry lse=-inf -> w=0; an all-empty sequence (l_sum=0)
    # reduces to defined zeros (v1 zero-seq semantics).
    b = tl.program_id(0)
    hq = tl.program_id(1)
    offs_s = tl.arange(0, BLOCK_S)
    smask = offs_s < NUM_SPLITS
    offs_d = tl.arange(0, D)
    base = b * stride_mid_b + hq * stride_mid_h
    m = tl.load(
        mid_ptr + base + offs_s * stride_mid_s + D, mask=smask, other=-float("inf")
    )
    m_star = tl.max(m, 0)
    # exp(-inf - (-inf)) = nan: clamp the anchor when every split is empty.
    safe_star = tl.where(m_star == -float("inf"), 0.0, m_star)
    w = tl.exp(m - safe_star)
    accs = tl.load(
        mid_ptr + base + offs_s[:, None] * stride_mid_s + offs_d[None, :],
        mask=smask[:, None],
        other=0.0,
    )
    l_sum = tl.sum(w, 0)
    out = tl.sum(w[:, None] * accs, axis=0) / tl.where(l_sum > 0.0, l_sum, 1.0)
    tl.store(out_ptr + b * stride_o_b + hq * stride_o_h + offs_d, out)


_MX_SPLIT_MAX = 32  # wheel parity (max_num_kv_splits)
_mid_cache: dict = {}

# Per-format (BLOCK_N, num_warps, num_stages, shift_flavor) — A9
# true-shape rebake (D=256, HK=4, Hq=24 — the REAL qwen3.8-27b-fp8
# geometry; results/2026-11-12-p647-decode-perf-research/legA9): the
# cert25x table was factorial winners at HK=8/Hq=32/D=128 where v3's
# shared walk + BLOCK_N=64 won; at the true shape BLOCK_N=32 wins every
# serve cell (B=8@2048: 1.41/1.17/1.52 ms; B=1@4K/32K) and v2's
# per-q-head grid beats the v3 GQA walk at B>1 (v3 B=8 2.2-6.1 ms —
# the N_REP=6 unroll spills at BLOCK_N=64/D=256). v3 keeps ONE niche
# (B=1 @1K: 0.33 vs v2's 0.74 ms) — reachable via ARC_MXKV_DECODER=v3.
# Flavor law unchanged (mxk8v4 shift, mxfp4 join, mxfp8 flat). Warps:
# mxfp8 4 (e4m3 K, lighter V leg), mxfp4 8 (double e2m1).
_MX_TUNE = {
    "mxfp8": (32, 4, 2, False),
    "mxfp4": (32, 8, 2, False),
    "mxk8v4": (32, 8, 2, True),
}


def _split_launch(
    stage1,
    grid_heads,
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int,
    block_n: int,
    shift: bool,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Shared split-KV launcher (stage 1 + stage 2). The mid buffer is
    cached per (B, Hq, splits, D, device) — allocated once per shape on
    the eager warmup and reused (graph law: no allocation inside a
    capture). Stage 1 runs at (B, grid_heads, _MX_SPLIT_MAX) — Hq for
    v2, Hk for the GQA-batched v3 — and the combine always at (B, Hq);
    the mid layout is identical for both."""
    bsz, hq, d = q.shape
    hk = cache.shape[2]
    sb = cache.shape[-1]
    want = {
        f: slot_bytes(d, f) for f in ("mxfp8", "mxfp4", "mxk8v4")
    }
    fmt = next(f for f, w in want.items() if w == sb)
    out = torch.empty_like(q)
    key = (bsz, hq, _MX_SPLIT_MAX, d, str(q.device))
    mid = _mid_cache.get(key)
    if mid is None:
        mid = torch.empty(
            bsz, hq, _MX_SPLIT_MAX, d + 1, dtype=torch.float32, device=q.device
        )
        _mid_cache[key] = mid
    stage1[(bsz, grid_heads, _MX_SPLIT_MAX)](
        q,
        cache,
        block_tables,
        seq_lens,
        mid,
        sm_scale,
        q.stride(0),
        q.stride(1),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        block_tables.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        D=d,
        N_GROUPS=d // 32,
        K4=fmt == "mxfp4",
        V4=fmt in ("mxfp4", "mxk8v4"),
        N_REP=hq // hk,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        NUM_SPLITS=_MX_SPLIT_MAX,
        SHIFT=shift,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _mx_decode_combine[(bsz, hq)](
        mid,
        out,
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        out.stride(0),
        out.stride(1),
        D=d,
        NUM_SPLITS=_MX_SPLIT_MAX,
        BLOCK_S=triton.next_power_of_2(_MX_SPLIT_MAX),
        num_warps=1,
    )
    return out


def _mx_decode_v2(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int,
    block_n: int,
    shift: bool,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Split-KV decode (per-q-head stage 1). Tuning surface and A4-A6
    evidence: see the module header and _MX_TUNE."""
    hq = q.shape[1]
    return _split_launch(
        _mx_decode_split, hq, q, cache, block_tables, seq_lens, sm_scale,
        block_size, block_n, shift, num_warps, num_stages,
    )


def _mx_decode_v3(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int,
    block_n: int,
    shift: bool,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """GQA-batched split-KV decode (per-kv-head stage 1 serving all
    N_REP q-heads): the pointer chase amortizes N_REP-fold. A7 is its
    A/B leg; see the module header for the A6 pointer-chase evidence."""
    hk = cache.shape[2]
    return _split_launch(
        _mx_decode_split_gqa, hk, q, cache, block_tables, seq_lens, sm_scale,
        block_size, block_n, shift, num_warps, num_stages,
    )


def mx_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int = 64,
    block_n: int | None = None,
    decoder: str | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Host wrapper. q [B, Hq, D] fp32 device; cache uint8 paged
    [num_blocks, block_size, Hk, slot_bytes(fmt)] (format inferred from
    slot width, mx_layout law); block_tables int32 [B, max_blocks];
    seq_lens int32 [B]. Returns [B, Hq, D] fp32.

    decoder: None -> ARC_MXKV_DECODER -> "v2" (the A9 true-shape serve
    winner: per-q-head split stage 1); "v3" = the GQA-batched split
    (solo-short-ctx niche); "v1" = the bit-frozen certified path (see
    the module header). block_n/num_warps/num_stages default to the
    per-format _MX_TUNE winners for v2/v3 (None) and to v1's frozen
    (128, 4); explicit values override — the A/B harness's tuning face.
    Unpack flavor: ARC_MXKV_UNPACK/_SHIFT override the table's
    per-format flavor.
    """
    dec = decoder or _DECODER
    if dec != "v1":
        sb = cache.shape[-1]
        d = q.shape[-1]
        want = {
            f: slot_bytes(d, f) for f in ("mxfp8", "mxfp4", "mxk8v4")
        }
        fmt = next(f for f, w in want.items() if w == sb)
        bn_t, nw_t, ns_t, sh_t = _MX_TUNE[fmt]
        block_n = bn_t if block_n is None else block_n
        num_warps = nw_t if num_warps is None else num_warps
        num_stages = ns_t if num_stages is None else num_stages
        # flavor: env/harness override (_SHIFT set) > per-format table
        shift = sh_t if _SHIFT is None else _SHIFT
        launch = _mx_decode_v2 if dec == "v2" else _mx_decode_v3
        return launch(
            q, cache, block_tables, seq_lens, sm_scale, block_size, block_n,
            shift, num_warps, num_stages,
        )
    # v1 is bit-frozen: fixed launch config (BLOCK_N=block_n — default
    # 128 — and its hardcoded num_warps=4); tuning params are v2/v3's.
    block_n = 128 if block_n is None else block_n
    bsz, hq, d = q.shape
    hk = cache.shape[2]
    sb = cache.shape[-1]
    want = {
        f: slot_bytes(d, f) for f in ("mxfp8", "mxfp4", "mxk8v4")
    }
    fmt = next(f for f, w in want.items() if w == sb)
    out = torch.empty_like(q)
    _mx_decode[(bsz, hq)](
        q,
        cache,
        block_tables,
        seq_lens,
        out,
        sm_scale,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        block_tables.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        D=d,
        N_GROUPS=d // 32,
        K4=fmt == "mxfp4",
        V4=fmt in ("mxfp4", "mxk8v4"),
        N_REP=hq // hk,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out
