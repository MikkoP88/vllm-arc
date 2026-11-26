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

import contextlib
import os

import torch

with contextlib.suppress(ImportError):
    from . import _vllm_fa2_C  # noqa: F401  (dlopens the extension — the
    # package __init__ does NOT; arc4 law: a bare package import leaves
    # torch.ops._vllm_fa2_C empty)

from .flash_attn_interface import (
    _as_int32_device_tensor,
    _infer_num_xe_cores,
    _min_blocks_for_split,
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


def _mx_build_decode_split_plan(
    kv_lens,
    kv_tile: int,
    num_kv_splits: int,
    num_xe_cores: int,
    num_heads_kv: int,
):
    """Vendor build_decode_split_plan semantics with the Xe-core
    oversubscription factor knobbed (ARC_MXKV_WGTARGET; 2 = vendor
    parity, higher pushes per-seq splits past the auto policy cap —
    r207 splits-headroom leg). Same guarantees as the vendor contract:
    splits_per_seq[i] <= num_kv_splits and the work items partition
    [0, kv_tiles) exactly once per seq."""
    oversub = max(1, int(os.environ.get("ARC_MXKV_WGTARGET", "2") or "2"))
    if isinstance(kv_lens, torch.Tensor):
        kv_lens_list = kv_lens.to(dtype=torch.int32, device="cpu").tolist()
    else:
        kv_lens_list = [int(v) for v in kv_lens]

    tiles_per_seq = [
        max(1, (kv + kv_tile - 1) // kv_tile) for kv in kv_lens_list]
    total_tiles = sum(tiles_per_seq)

    min_wgs = max(1, num_xe_cores * oversub // max(1, num_heads_kv))
    target_tiles_per_wg = max(4, total_tiles // min_wgs)

    min_blocks_for_split = _min_blocks_for_split(kv_tile)

    splits_per_seq = []
    work_items = []
    for i, n_tiles in enumerate(tiles_per_seq):
        if (n_tiles <= target_tiles_per_wg
                or n_tiles < min_blocks_for_split
                or num_kv_splits <= 1):
            n_splits = 1
        else:
            n_splits = ((n_tiles + target_tiles_per_wg - 1)
                        // target_tiles_per_wg)
            n_splits = min(n_splits, num_kv_splits, n_tiles)
        splits_per_seq.append(n_splits)

        base, rem = divmod(n_tiles, n_splits)
        start = 0
        for s in range(n_splits):
            count = base + (1 if s < rem else 0)
            work_items.append([i, start, count, s])
            start += count

    return (
        torch.tensor(splits_per_seq, dtype=torch.int32),
        torch.tensor(work_items, dtype=torch.int32),
    )


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
    softmax_scale: float | None = None,
    num_splits_kv: int = 0,
    host_kv_lens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    window_size: tuple[int, int] | None = None,
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
        window_size: optional (left, right) sliding-window bounds —
           the flash-attn convention (wheel-law: sliding_window_val =
           1 + window_size[0]). A W-token window rides (W-1, 0) = the
           last W tokens per row; None keeps the windowless decode.
           The masked kernel twin is compiled for D=256 at page 16/64
           (qgroup <= 8); other geometries refuse loudly, never
           serve unmasked (P-C2, 2026-11-25).
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

    max_seqlen_q = 1
    head_size = q.size(2)

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
        # r242d (2026-11-26): the plan MUST price tiles in the KERNEL's
        # k-walk units. Every geometry this fork dispatches walks 16-row
        # k-tiles (the r212c single-SG law: page-64 pools ride the _16
        # policy via 16-row sub-tiles), so the stock block-size mirror
        # (_kv_tile_from_block_size -> 64 at page 64) under-covered the
        # work list 4x: forced-split work items walked only the FIRST
        # QUARTER of each seq's K blocks (r242 forced8 cos~0.51, D=512
        # page 64, all three formats; the legacy/auto path was always
        # correct -- mx_get_num_splits prices 16). Page 16 is unchanged
        # (the mirror already returned 16 there).
        kv_tile = 16
        # r243a (2026-11-26): with a window riding the call, the plan
        # must price the WINDOWED extent. The kernel adds k_block0 (=
        # max(L-1-wl, 0) // kv_tile, the walk bound from
        # window_size_left) to every work item's tile start, so a plan
        # built from FULL lens walks past k_blocks -- past the
        # block-table row: at W=64 / ctx 8192 forced-8, the split-0
        # item walked tiles 508..571 of 512 (r243 mxfp4 DEVICE_LOST;
        # the mxfp8 cell survived on mapped garbage with every OOB
        # column masked -- correct output, undefined reads). Price each
        # seq at the device's windowed tile count ceil(L/kv_tile) -
        # k_block0, padded to a tile multiple so the builder's ceil-div
        # reproduces it exactly.
        plan_lens = host_kv_lens
        if window_size is not None:
            wl = max(int(window_size[0]), 0)
            if isinstance(host_kv_lens, torch.Tensor):
                full_lens = host_kv_lens.to(
                    dtype=torch.int64, device="cpu").tolist()
            else:
                full_lens = [int(v) for v in host_kv_lens]
            plan_lens = [
                ((L + kv_tile - 1) // kv_tile
                 - max(L - 1 - wl, 0) // kv_tile) * kv_tile
                for L in full_lens
            ]
        num_xe_cores = _infer_num_xe_cores(q.device)
        num_heads_kv = cache.size(2)
        splits_cpu, work_list_cpu = _mx_build_decode_split_plan(
            plan_lens,
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

    # P-C2 (2026-11-25): the flash window pair rides the schema after
    # softmax_scale; None = the historical windowless decode (-1/-1).
    wl, wr = (-1, -1) if window_size is None else (
        int(window_size[0]), int(window_size[1]))

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
        wl,
        wr,
        num_splits_kv if num_splits_kv > 0 else -1,
        splits_per_seq_dev,
        work_list_dev,
        out,
    )
    return out
