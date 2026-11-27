"""native_decode.py — arc7 MX-native decode bridge (Phase 8).

Routes the MTP verify-batch rows to the wheel-side MX-native CUTLASS/Xe2
kernel (`vllm_xpu_kernels.mx_attn_interface.mx_paged_decode`, built by
the arc7mx wheel section) instead of the Triton split-dot lanes.

The engine's split-row plan already holds exactly the kernel's contract:
one row per (request, draft-token) with its own causal limit and block
table. The vendor decode kernel models each row as its own one-query
"sequence": cu_seqlens_q = arange(R+1), seqused_k = the per-row limits,
block_table = the per-row gathered tables, at a contiguous row pitch (a
stride-0 expand VIEW is materialized here -- r262 stride law; the vendor
flow passes seqused_k in the cu_seqlens_k slot). All ops here are device ops — the
bridge runs inside XPU-graph capture with the plan (capture law: no
pageable H2D, no host reads of device tensors; max_seqlen_k comes from
the metadata's host-known field).

DORMANT until certified: the engine gates this behind ARC_MXKV_NATIVE=1
(the Triton v2/v5 lanes stay the certified default). A missing/old wheel
fails LOUD — a silent fallback would hide the misconfiguration in A/B
legs.
"""

from __future__ import annotations

import logging
import os

import torch

_checked = False
_mx_paged_decode = None
# r212b capture telemetry (2026-11-21): the launch4 A/B showed the
# e2e native cost ~100x the offline per-call map at matched walked
# rows — the frozen walk extent is the suspect. One line per
# (fmt, rows, max_seqlen_k, bt width) class, logged at CAPTURE time
# (decode replays never re-run this python), so the serve log carries
# exactly the bucket classes the engine froze and their walk bounds.
_telemetry_logged: set[tuple] = set()
# r262 stride law (2026-11-27): one line per (row pitch, pool) class
# whenever a caller hands in a block-table VIEW the wheel cannot address.
_bt_law_logged: set[tuple[int, int]] = set()


def _load():
    """Import the wheel surface once; cache and re-raise loudly on absence."""
    global _checked, _mx_paged_decode
    if _checked:
        return _mx_paged_decode
    _checked = True
    try:
        from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode
    except ImportError as e:
        raise ImportError(
            "ARC_MXKV_NATIVE=1 requires the arc7mx wheel "
            f"(vllm_xpu_kernels mx_attn_interface): {e}"
        ) from e
    _mx_paged_decode = mx_paged_decode
    return _mx_paged_decode


def mx_decode_native(
    q_rows: torch.Tensor,  # (R, Hq, D) model dtype — one row per (req, draft token)
    kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot) uint8 pool
    bt_rows: torch.Tensor,  # (R, max_blocks) int32 — per-row block tables
    sl_rows: torch.Tensor,  # (R,) int32 — per-row causal KV limits
    sm_scale: float,
    max_seq_len: int,  # metadata field — SUPERSEDED (see r212c-e2e law below)
    fmt: str,  # "mxfp8" | "mxfp4" | "mxk8v4"
    sliding_window: int | None = None,
) -> torch.Tensor:
    """One native-kernel call for the whole verify-batch step.

    sliding_window (P-C2, 2026-11-25): the impl's per-layer window
    width. Rides the kernel as the flash pair (W-1, 0) = the last W
    tokens per row — None keeps the windowless decode (qwen target;
    gemma-4-31b full_attention layers). The masked kernel twin exists
    for D=256 page 16/64 at qgroup<=8; anything else refuses loudly
    rather than serving window-blind.
    """
    mx_paged_decode = _load()
    n_rows = q_rows.shape[0]
    # r262 stride law (2026-11-27, P6.47 gemma DEVICE_LOST; e2e A/B
    # results/2026-11-27-p647-r262-expandview-ab): the wheel takes the
    # block table as data_ptr + size(1) and addresses row r at
    # r * max_pages_per_seq (mx_paged_decode_xe2.cpp,
    # mx_paged_decode_mainloop.hpp) -- a CONTIGUOUS pitch. The engine's
    # windowed continuation-prefill synth path passes
    # bt_row.expand(q_len, -1): stride (0, 1). Unmaterialized, rows 1..63
    # read the persistent table's zeroed rows (block 0: windowed attention
    # silently off on those rows) and rows >= 64 read PAST the table as
    # block ids -- the 118-row multi rehit hung the GPU at every suite
    # attempt (DEVICE_LOST, r255-r260; fatal or not depends on the bytes
    # the over-read lands in). The Triton lanes are stride-aware, so only
    # this lane needs the normalization. is_contiguous() is a metadata
    # check and the copy a device op (capture-safe); a size-1 row dim is
    # contiguous by definition and the decode/split-row tables are
    # gathered contiguous, so only a multi-row expand view ever copies.
    if not bt_rows.is_contiguous():
        _pitch = int(bt_rows.stride(0))
        bt_rows = bt_rows.contiguous()
        _lkey = (_pitch, int(kv_cache.data_ptr()))
        if _lkey not in _bt_law_logged:
            _bt_law_logged.add(_lkey)
            logging.getLogger("vllm.mx_native").info(
                "MXKV native bt materialized: rows=%d bt_width=%d stride0=%d "
                "pool=0x%x (r262 stride law: the wheel addresses rows at a "
                "contiguous pitch)",
                n_rows,
                int(bt_rows.shape[1]),
                _pitch,
                int(kv_cache.data_ptr()),
            )
    # each row is its own one-query "sequence" (device fill — capture-safe)
    cu_q = torch.arange(n_rows + 1, device=q_rows.device, dtype=torch.int32)
    # the certified engine boots fp16 activations (qwen3.8-27b-fp8 posture,
    # dtype=torch.float16); the kernel is bf16-specialized. Device cast —
    # capture-safe, µs-class under replay, and the round-trip error
    # (<=2^-9 relative, 3 mantissa bits) rides inside the fp16-dot class
    # the MX lanes already accept (r212a fail1 2026-11-21: TORCH_CHECK
    # "query must be [total_q, heads, D] bf16" killed all three cells at
    # warmup capture; the caller casts the output back).
    q_in = (
        q_rows
        if q_rows.dtype == torch.bfloat16
        else q_rows.to(torch.bfloat16)
    )
    # r212f walk-extent fix (2026-11-21, no-op-falsification audit):
    # max_seqlen_k FREEZES per captured graph, and the C++ auto-split
    # heuristic prices the split count from it (kv_tiles =
    # ceil(max_seqlen_k/16)). The metadata's max_seq_len is the capture
    # DUMMY's diluted value (observed 9/13/18/5 across buckets, r212b
    # telemetry) -> splits freeze at 1 and every replayed step walks
    # its full real KV on ONE workgroup per (row, kv-head). The first
    # r212c "fix" (45eac0fe) was a NO-OP: it computed the capacity and
    # printed it in the telemetry line but still passed
    # int(max_seq_len) to the kernel — the "falsified" byte-identical
    # ITLs disproved nothing, and the r212d cells (ctx/cap) never
    # tested the diluted value itself. Price it at the bucket's walk
    # CAPACITY (bt width x block_size): the kernel's per-row
    # device-side limits bound the true walk (r212b accuracy passed at
    # max_seqlen_k=9 with 32K ctx — the arg never truncates), so the
    # frozen split count simply covers the worst-case replay class;
    # short rows early-exit their splits. Validated offline by the
    # r212f mseq ladder before any e2e.
    _max_sk = int(bt_rows.shape[1]) * int(kv_cache.shape[1])
    # r216/r219: engine-side split-depth pin (env ARC_MXKV_NSPLIT;
    # unset = 0 = the C++ auto heuristic). The r209/r210 "40 optimal"
    # verdict was probe-priced pre-template-split; the sign-inversion
    # law demanded the e2e re-price (r216/216b/217/217b: 128 banked).
    # Defined BEFORE the telemetry block so splits= can log it.
    _ns = int(os.environ.get("ARC_MXKV_NSPLIT", "0") or 0)
    _w = int(sliding_window) if sliding_window else -1
    _tkey = (fmt, n_rows, _max_sk, int(bt_rows.shape[1]), _w)
    if _tkey not in _telemetry_logged:
        _telemetry_logged.add(_tkey)
        logging.getLogger("vllm.mx_native").info(
            "MXKV native call fmt=%s rows=%d max_seqlen_k=%d "
            "bt_width=%d stride=%d splits=%s window=%d (r212f walk; "
            "r219 splits logged: int pin or auto; P-C2 window rides "
            "the masked twin)",
            fmt,
            n_rows,
            _max_sk,
            int(bt_rows.shape[1]),
            int(kv_cache.stride(1)) if kv_cache.dim() == 4 else -1,
            _ns if _ns else "auto",
            _w,
        )
    return mx_paged_decode(
        q_in,
        kv_cache,
        fmt,
        cu_q,
        sl_rows.to(torch.int32),
        bt_rows,
        max_seqlen_k=_max_sk,
        softmax_scale=sm_scale,
        num_splits_kv=_ns,
        window_size=(sliding_window - 1, 0) if sliding_window else None,
    )
