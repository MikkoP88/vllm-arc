# SPDX-License-Identifier: Apache-2.0
"""triton_mx_decode.py — MXFP4/MXFP8 paged attention decode kernel (P6.47,
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

GRAPH SAFETY: seq lens and block tables are read from device buffers
inside the kernel (the TQ decode pattern) — no host reads, no
data-dependent host control flow; XPU-graph safe.

Provenance: from-scratch in-house lane (owner directive 2026-11-11);
kernel structure follows the wheel's sub-byte KV decode template
(triton_turboquant_decode.py — survey
results/2026-11-11-p647-mxkv-research/wheelref/).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .mx_layout import slot_bytes


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
    MXFP4: tl.constexpr,
):
    # slot: [BLOCK_N] int64 slot byte bases; mask: [BLOCK_N] valid lanes.
    # Returns f32 [BLOCK_N, D] dequantized K-or-V rows.
    g = tl.arange(0, N_GROUPS)
    gmask = mask[:, None]
    if MXFP4:
        codes = tl.load(cache_ptr + slot[:, None] + payload_off + D // 2 + g[None, :], mask=gmask, other=0)
        scale = tl.exp2(codes.to(tl.float32) - 127.0)
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
    MXFP4: tl.constexpr,
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
    v_off = (D // 2 + N_GROUPS) if MXFP4 else (D + N_GROUPS)

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
        k = _mx_dequant_block(cache_ptr, slot, 0, nmask, BLOCK_N, D, N_GROUPS, MXFP4)
        scores = tl.sum(q[None, :] * k, axis=1) * sm_scale
        scores = tl.where(nmask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 0)
        v = _mx_dequant_block(cache_ptr, slot, v_off, nmask, BLOCK_N, D, N_GROUPS, MXFP4)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # zero-length sequences (never scheduled by the engine, but the
    # offline harness probes them): 0/0 -> defined zeros, no NaN.
    out = tl.where(l_i == 0.0, tl.zeros((D,), dtype=tl.float32), acc / l_i)
    tl.store(out_ptr + b * stride_o_b + hq * stride_o_h + offs_d, out)


def mx_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    block_size: int = 64,
    block_n: int = 128,
) -> torch.Tensor:
    """Host wrapper. q [B, Hq, D] fp32 device; cache uint8 paged
    [num_blocks, block_size, Hk, slot_bytes(fmt)] (format inferred from
    slot width, mx_layout law); block_tables int32 [B, max_blocks];
    seq_lens int32 [B]. Returns [B, Hq, D] fp32.
    """
    bsz, hq, d = q.shape
    hk = cache.shape[2]
    sb = cache.shape[-1]
    want = {"mxfp8": slot_bytes(d, "mxfp8"), "mxfp4": slot_bytes(d, "mxfp4")}
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
        MXFP4=fmt == "mxfp4",
        N_REP=hq // hk,
        BLOCK_SIZE=block_size,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return out
