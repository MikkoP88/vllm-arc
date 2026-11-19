"""native_decode.py — arc7 MX-native decode bridge (Phase 8).

Routes the MTP verify-batch rows to the wheel-side MX-native CUTLASS/Xe2
kernel (`vllm_xpu_kernels.mx_attn_interface.mx_paged_decode`, built by
the arc7mx wheel section) instead of the Triton split-dot lanes.

The engine's split-row plan already holds exactly the kernel's contract:
one row per (request, draft-token) with its own causal limit and block
table. The vendor decode kernel models each row as its own one-query
"sequence": cu_seqlens_q = arange(R+1), seqused_k = the per-row limits,
block_table = the per-row gathered tables (the vendor flow passes
seqused_k in the cu_seqlens_k slot). All ops here are device ops — the
bridge runs inside XPU-graph capture with the plan (capture law: no
pageable H2D, no host reads of device tensors; max_seqlen_k comes from
the metadata's host-known field).

DORMANT until certified: the engine gates this behind ARC_MXKV_NATIVE=1
(the Triton v2/v5 lanes stay the certified default). A missing/old wheel
fails LOUD — a silent fallback would hide the misconfiguration in A/B
legs.
"""

from __future__ import annotations

import torch

_checked = False
_mx_paged_decode = None


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
    q_rows: torch.Tensor,  # (R, Hq, D) bf16 — one row per (req, draft token)
    kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot) uint8 pool
    bt_rows: torch.Tensor,  # (R, max_blocks) int32 — per-row block tables
    sl_rows: torch.Tensor,  # (R,) int32 — per-row causal KV limits
    sm_scale: float,
    max_seq_len: int,  # host-known metadata field (capture law)
    fmt: str,  # "mxfp8" | "mxfp4" | "mxk8v4"
) -> torch.Tensor:
    """One native-kernel call for the whole verify-batch step."""
    mx_paged_decode = _load()
    n_rows = q_rows.shape[0]
    # each row is its own one-query "sequence" (device fill — capture-safe)
    cu_q = torch.arange(n_rows + 1, device=q_rows.device, dtype=torch.int32)
    return mx_paged_decode(
        q_rows,
        kv_cache,
        fmt,
        cu_q,
        sl_rows.to(torch.int32),
        bt_rows,
        max_seqlen_k=int(max_seq_len),
        softmax_scale=sm_scale,
    )
