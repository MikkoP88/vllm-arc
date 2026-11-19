# SPDX-License-Identifier: Apache-2.0
"""mx_attn_interface.py — MX-native paged decode (arc7, P6.47 Phase 8).

Python surface for the arc7 MX-native decode kernel: the mirror of
flash_attn_interface's decode path (per-seq split plan on host, no D2H
sync) for MX KV caches stored in the TQ-AoS slot layout (one uint8 pool
[num_blocks, block_size, num_kv_heads, slot_bytes]; per slot
[K bytes][Ks e8m0 x D/32][V bytes][Vs e8m0 x D/32]).

The slot geometry mirrors vllm_gaudi/xpu/patches/mx_kv/mx_layout.py
(the python side owns the law; these formulas must not drift):

    mxfp8 : slot = 2*D + 2*(D/32)
    mxfp4 : slot = D + 2*(D/32)
    mxk8v4: slot = 3*D/2 + 2*(D/32)

Decode only: Q is [total_q, num_heads_q, D] with cu_seqlens_q (each
seq contributes q_len == 1 rows — the packed GQA/MTP verify case rides
the same kernel the vendor's spec-decode fast path uses). Prefill stays
on the caller's existing path.
"""

from __future__ import annotations

from typing import Optional

import torch

from .flash_attn_interface import (
    _as_int32_device_tensor,
    _infer_num_xe_cores,
    _kv_tile_from_block_size,
    build_decode_split_plan,
)

# Format ids shared with the C++ entry (mx_attn_interface.h).
MXFMT_MXFP8 = 0
MXFMT_MXFP4 = 1
MXFMT_MXK8V4 = 2

_FORMATS = {
    "mxfp8": MXFMT_MXFP8,
    "mxfp4": MXFMT_MXFP4,
    "mxk8v4": MXFMT_MXK8V4,
}

_GROUP = 32  # elements per e8m0 scale group (MX law)


def _slot_bytes(d: int, fmt: int) -> int:
    n_groups = d // _GROUP
    if fmt == MXFMT_MXFP8:
        return 2 * d + 2 * n_groups
    if fmt == MXFMT_MXFP4:
        return d + 2 * n_groups
    if fmt == MXFMT_MXK8V4:
        return 3 * d // 2 + 2 * n_groups
    raise ValueError(f"unknown MX format id: {fmt}")


def mx_paged_decode(
    q: torch.Tensor,
    cache: torch.Tensor,
    fmt: str,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    num_splits_kv: int = 0,
    host_kv_lens: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MX-native paged decode.

    Args:
        q: [total_q, num_heads_q, D] bf16; each sequence contributes
           exactly one query row (decode / packed-verify).
        cache: the uint8 MX KV pool [num_blocks, block_size,
           num_kv_heads, slot_bytes] (contiguous; strides are read from
           the tensor — the kernel consumes them verbatim).
        fmt: "mxfp8" | "mxfp4" | "mxk8v4".
        cu_seqlens_q: cumulative query lengths (int32).
        seqused_k: per-seq used KV length (int32).
        block_table: [batch, max_blocks_per_seq] int32 page table.
        max_seqlen_k: max KV length in the batch.
        softmax_scale: defaults to D**-0.5.
        num_splits_kv: split-KV cap; 0 lets the kernel default.
        host_kv_lens: optional host-side per-seq KV lengths (feeds the
           per-seq split plan without a device sync — same contract as
           flash_attn_varlen_func).
        out: optional preallocated [total_q, num_heads_q, D] output.
    """
    if fmt not in _FORMATS:
        raise ValueError(f"unknown MX KV format: {fmt} (known: {sorted(_FORMATS)})")
    fmt_id = _FORMATS[fmt]

    if cache.dtype != torch.uint8 or cache.dim() != 4:
        raise ValueError(
            f"cache must be the uint8 MX pool [num_blocks, block_size, "
            f"num_kv_heads, slot_bytes], got {tuple(cache.shape)} "
            f"{cache.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [total_q, heads, D], got {tuple(q.shape)}")

    batch = cu_seqlens_q.numel() - 1
    max_seqlen_q = 1
    head_size = q.size(2)
    block_size = cache.size(1)

    expected_slot = _slot_bytes(head_size, fmt_id)
    if cache.size(3) < expected_slot:
        raise ValueError(
            f"slot width {cache.size(3)} < {expected_slot} required for "
            f"{fmt} at D={head_size}")

    if softmax_scale is None:
        softmax_scale = head_size ** (-0.5)
    if host_kv_lens is not None:
        seqused_k = _as_int32_device_tensor(host_kv_lens, q.device)

    # Per-seq split plan on host (mirror of the stock decode path).
    splits_per_seq_dev = None
    work_list_dev = None
    if (host_kv_lens is not None and num_splits_kv is not None
            and num_splits_kv > 1):
        kv_tile = _kv_tile_from_block_size(block_size)
        num_xe_cores = _infer_num_xe_cores(q.device)
        num_heads_kv = cache.size(2)
        splits_cpu, work_list_cpu = build_decode_split_plan(
            host_kv_lens,
            kv_tile=kv_tile,
            num_kv_splits=num_splits_kv,
            num_xe_cores=num_xe_cores,
            num_heads_kv=num_heads_kv,
        )
        if work_list_cpu.numel() > 0:
            splits_per_seq_dev = splits_cpu.to(
                device=q.device, non_blocking=True)
            work_list_dev = work_list_cpu.to(
                device=q.device, non_blocking=True)

    if out is None:
        out = torch.empty_like(q)

    # The decode kernel reads per-seq KV lengths from the cu_seqlens_k
    # slot (the vendor decode flow passes seqused_k there — see
    # flash_api.cpp's cutlass_paged_decode_interface call).
    torch.ops._vllm_fa2_C.mx_paged_decode(
        q,
        cache,
        fmt_id,
        cu_seqlens_q,
        seqused_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        num_splits_kv if num_splits_kv > 0 else -1,
        splits_per_seq_dev,
        work_list_dev,
        out,
    )
    return out
