# SPDX-License-Identifier: Apache-2.0
"""mx_dequant.py — MXFP4/MXFP8/MXK8V4 whole-cache dequant kernel (P6.47,
vllm_gaudi/xpu/patches/mx_kv/).

PURPOSE: bulk-dequantize the cached prefix out of the paged uint8 MX KV
cache into dense fp16 K/V buffers — the MX analogue of the wheel's
``_tq_full_dequant_kv`` (triton_turboquant_decode.py), used by the
engine's large-continuation prefill path (chunks with q_len above
``_CONTINUATION_DECODE_THRESHOLD`` dequant once and run flash-attn,
instead of rescanning the compressed context per query row).

NUMERICS LAW: element decode mirrors triton_mx_decode._mx_dequant_block
(the mx_quant.py reference) — E8M0 group scales at 1 B per 32 elements,
e4m3 hardware cast (mxfp8) or the arithmetic e2m1 nibble chain (mxfp4),
even element in the low nibble.

SAFETY: one program per (token, head); tokens at or past ``cached_len``
write zeros instead of gathering uninitialized cache bytes (the block
table has no entry for them — a wild gather could fault). Pure device
reads, no host syncs: eager-path only by construction (the engine never
graph-captures chunked-prefill continuations of this size), but the
kernel is shape-static and replay-safe regardless.

Provenance: from-scratch in-house lane (owner directive 2026-11-11);
kernel structure follows the wheel's sub-byte KV dequant template
(triton_turboquant_decode._tq_full_dequant_kv — survey
results/2026-11-11-p647-mxkv-research/wheelref/).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .mx_layout import slot_bytes
from .triton_mx_decode import _mx_dequant_e2m1


@triton.jit
def _mx_full_dequant(
    cache_ptr,
    block_table_ptr,
    k_out_ptr,
    v_out_ptr,
    cached_len,
    stride_bt_row,
    stride_c_b,
    stride_c_t,
    stride_c_h,
    stride_k_h,
    stride_k_t,
    D: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K4: tl.constexpr,
    V4: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    if tok < cached_len:
        blk = tl.load(block_table_ptr + tok // BLOCK_SIZE).to(tl.int64)
        slot = (
            blk * stride_c_b
            + (tok % BLOCK_SIZE).to(tl.int64) * stride_c_t
            + head.to(tl.int64) * stride_c_h
        )
        g = tl.arange(0, N_GROUPS)
        # K block size decides V's offset (mx_layout law); each leg picks
        # its element path independently (mxk8v4 = e4m3 K + e2m1 V).
        v_off = (D // 2 if K4 else D) + N_GROUPS
        # ---- K ----
        if K4:
            kcode = tl.load(cache_ptr + slot + D // 2 + g)
            ksc = tl.exp2(kcode.to(tl.float32) - 127.0)
            kraw = tl.load(cache_ptr + slot + tl.arange(0, D // 2))
            klo = _mx_dequant_e2m1(kraw & 0xF)
            khi = _mx_dequant_e2m1(kraw >> 4)
            kelems = tl.reshape(tl.join(klo, khi), (D,))
            ksc_flat = tl.reshape(
                tl.broadcast_to(ksc[:, None], (N_GROUPS, 32)), (D,)
            )
            krow = kelems * ksc_flat
        else:
            kcode = tl.load(cache_ptr + slot + D + g)
            ksc = tl.exp2(kcode.to(tl.float32) - 127.0)
            kraw = tl.load(cache_ptr + slot + tl.arange(0, D))
            kf = kraw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            ksc_flat = tl.reshape(
                tl.broadcast_to(ksc[:, None], (N_GROUPS, 32)), (D,)
            )
            krow = kf * ksc_flat
        # ---- V ----
        if V4:
            vcode = tl.load(cache_ptr + slot + v_off + D // 2 + g)
            vsc = tl.exp2(vcode.to(tl.float32) - 127.0)
            vraw = tl.load(cache_ptr + slot + v_off + tl.arange(0, D // 2))
            vlo = _mx_dequant_e2m1(vraw & 0xF)
            vhi = _mx_dequant_e2m1(vraw >> 4)
            velems = tl.reshape(tl.join(vlo, vhi), (D,))
            vsc_flat = tl.reshape(
                tl.broadcast_to(vsc[:, None], (N_GROUPS, 32)), (D,)
            )
            vrow = velems * vsc_flat
        else:
            vcode = tl.load(cache_ptr + slot + v_off + D + g)
            vsc = tl.exp2(vcode.to(tl.float32) - 127.0)
            vraw = tl.load(cache_ptr + slot + v_off + tl.arange(0, D))
            vf = vraw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            vsc_flat = tl.reshape(
                tl.broadcast_to(vsc[:, None], (N_GROUPS, 32)), (D,)
            )
            vrow = vf * vsc_flat
    else:
        # Beyond the cached prefix: zero-fill (no block-table entry, no
        # gather of uninitialized cache bytes).
        krow = tl.zeros((D,), dtype=tl.float32)
        vrow = tl.zeros((D,), dtype=tl.float32)

    out_off = head.to(tl.int64) * stride_k_h + tok.to(tl.int64) * stride_k_t
    tl.store(k_out_ptr + out_off + tl.arange(0, D), krow.to(tl.float16))
    tl.store(v_out_ptr + out_off + tl.arange(0, D), vrow.to(tl.float16))


def mx_dequant_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    cached_len: int,
) -> str:
    """Host wrapper. cache uint8 paged [num_blocks, block_size, Hk,
    slot_bytes(fmt)]; block_table int32 [1, max_blocks]; k_out/v_out fp16
    [1, Hk, T, D] dense dequant targets (T = k_out.shape[2], block-padded
    up from cached_len); cached_len the valid prefix length. Format is
    inferred from the cache's slot width (mx_layout law). Returns fmt.
    """
    d = k_out.shape[-1]
    t_alloc = k_out.shape[2]
    hk = cache.shape[2]
    block_size = cache.shape[1]
    sb = cache.shape[-1]
    want = {
        f: slot_bytes(d, f) for f in ("mxfp8", "mxfp4", "mxk8v4")
    }
    fmt = next(f for f, b in want.items() if b == sb)
    _mx_full_dequant[(t_alloc, hk)](
        cache,
        block_table,
        k_out,
        v_out,
        cached_len,
        block_table.stride(0),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        k_out.stride(1),
        k_out.stride(2),
        D=d,
        N_GROUPS=d // 32,
        K4=fmt == "mxfp4",
        V4=fmt in ("mxfp4", "mxk8v4"),
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    return fmt
