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
Decoder selection: ARC_MXKV_DECODER=v1|v2 (default v1 — the certified
path and every banked quality verdict stay tied to v1 until v2's own
gates pass); ARC_MXKV_UNPACK=join|shift (v2 only, default join).

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

# Decoder selection (decode-perf research, 2026-11-12): v1 stays the
# default/certified path; v2 (split-KV) is opt-in until its offline A4
# + engine A/B legs pass. Read once at import — the engine's env law.
_DECODER = os.environ.get("ARC_MXKV_DECODER", "v1")
# e2m1 unpack flavor for v2 ONLY ("join" = v1's interleave, bit-frozen;
# "shift" = duplicated-load + dynamic nibble shift). A/B decides.
_SHIFT = os.environ.get("ARC_MXKV_UNPACK", "join") == "shift"


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


def _mx_decode_v2(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int,
    block_n: int,
    shift: bool,
) -> torch.Tensor:
    """Split-KV decode launch (stage 1 + stage 2). The mid buffer is
    cached per (B, Hq, splits, D, device) — allocated once per shape on
    the eager warmup and reused (graph law: no allocation inside a
    capture)."""
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
    _mx_decode_split[(bsz, hq, _MX_SPLIT_MAX)](
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
        num_warps=4,
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


def mx_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int = 64,
    block_n: int = 128,
    decoder: str | None = None,
) -> torch.Tensor:
    """Host wrapper. q [B, Hq, D] fp32 device; cache uint8 paged
    [num_blocks, block_size, Hk, slot_bytes(fmt)] (format inferred from
    slot width, mx_layout law); block_tables int32 [B, max_blocks];
    seq_lens int32 [B]. Returns [B, Hq, D] fp32.

    decoder: None -> ARC_MXKV_DECODER -> "v1" (the certified path);
    "v2" = split-KV stage1+stage2 (opt-in until its validation legs
    pass — see the module header).
    """
    dec = decoder or _DECODER
    if dec == "v2":
        return _mx_decode_v2(
            q, cache, block_tables, seq_lens, sm_scale, block_size, block_n, _SHIFT
        )
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
