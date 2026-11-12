# SPDX-License-Identifier: Apache-2.0
"""mx_layout.py — MXFP4/MXFP8 KV-cache slot layout + pool arithmetic (P6.47,
vllm_gaudi/xpu/patches/mx_kv/).

PURPOSE: single source of truth for the byte geometry every layer of
the MX KV lane agrees on — the Triton store/decode kernels, the engine
cache-shape integration (cache slot sizing), the offline harness, and
the capacity tables in the performance reports. A layout constant
lives HERE, never duplicated at a call site.

CACHE TENSOR (TQ-AoS compatible, the wheel's paged layout):
    cache = torch.empty(num_blocks, block_size, Hk, slot_bytes, uint8)
One slot holds ONE (token, head-of-Hk) pair — K block then V block,
each immediately followed by its own E8M0 scale bytes so a block walk
touches contiguous bytes (no separate scale tensor, no gather):

    mxfp8:  [ K e4m3 bytes x D ][ Ks x D/32 ][ V e4m3 bytes x D ][ Vs x D/32 ]
    mxfp4:  [ K e2m1 nibbles D/2 ][ Ks x D/32 ][ V e2m1 nibbles D/2 ][ Vs x D/32 ]
    mxk8v4: [ K e4m3 bytes x D ][ Ks x D/32 ][ V e2m1 nibbles D/2 ][ Vs x D/32 ]
             (the quality-doctrine mix: 8-bit keys keep retrieval precision,
             4-bit values ride the byte savings — the wheel's certified
             turboquant_k8v4 "quality-safest preset" analogue; P6.47 gate
             run1: mxfp4's 4-bit K degrades long-return under churn,
             essay REP-LOOP post-transition, code/multi hash drift)

D=128 (both certified models' GDN/attn head sizes on this lane): mxfp8
= 264 B, mxk8v4 = 200 B, mxfp4 = 136 B per (token, head) — vs 512 B
fp16 KV and 256 B certified fp8_e4m3 KV (mxk8v4 = 0.78x fp8 bytes). GQA value heads share nothing here: the store
kernel writes per-Hk-head slots exactly like the wheel's TQ kernels
(MQ heads recompute K per group at decode).

POOL ARITHMETIC (report capacity columns use the same functions):
    slot_bytes(D, fmt)   bytes per (token, head)
    pool_gib(...)        GiB a (n_tokens, Hk) cache needs
    capacity_mult(fmt)   KV-token capacity multiplier vs fp16 KV

Provenance: from-scratch in-house lane (owner directive 2026-11-11);
byte-compatibility contract mirrors the wheel's turboquant slot
convention (survey: results/2026-11-11-p647-mxkv-research/).
"""

from __future__ import annotations

from .mx_quant import GROUP

D_DEFAULT = 128  # certified models' head size on this lane
BYTES_FP16_KV = 2 * 2 * D_DEFAULT  # K + V, 2 B each, per (token, head)
BYTES_FP8_KV = 2 * D_DEFAULT  # certified fp8_e4m3 KV baseline


def slot_bytes(d: int, fmt: str) -> int:
    """Bytes per (token, head) slot; fmt in {"mxfp4", "mxfp8", "mxk8v4"}.

    Mirrors mx_quant.slot_bytes (kept here so engine/report layers can
    import layout without importing the quant reference).
    """
    n_groups = d // GROUP
    if fmt == "mxfp8":
        return 2 * d + 2 * n_groups
    if fmt == "mxfp4":
        return d + 2 * n_groups
    if fmt == "mxk8v4":
        return 3 * d // 2 + 2 * n_groups
    raise ValueError(f"unknown MX KV format: {fmt}")


def offsets(d: int, fmt: str) -> dict[str, int]:
    """Byte offsets of each block inside one slot."""
    n_groups = d // GROUP
    if fmt == "mxfp8":
        return {"k": 0, "ks": d, "v": d + n_groups, "vs": 2 * d + n_groups}
    if fmt == "mxfp4":
        return {
            "k": 0,
            "ks": d // 2,
            "v": d // 2 + n_groups,
            "vs": d + n_groups,
        }
    if fmt == "mxk8v4":
        return {
            "k": 0,
            "ks": d,
            "v": d + n_groups,
            "vs": 3 * d // 2 + n_groups,
        }
    raise ValueError(f"unknown MX KV format: {fmt}")


def pool_bytes(n_tokens: int, hk: int, d: int, fmt: str) -> int:
    """Total bytes a (n_tokens, hk) MX KV cache occupies (block-padded rows
    still cost slot_bytes per token — block_size divides token count)."""
    return n_tokens * hk * slot_bytes(d, fmt)


def pool_gib(n_tokens: int, hk: int, d: int, fmt: str) -> float:
    return pool_bytes(n_tokens, hk, d, fmt) / (1024**3)


def capacity_mult(fmt: str, d: int = D_DEFAULT) -> float:
    """KV-token capacity multiplier vs fp16 KV at the same byte budget."""
    return BYTES_FP16_KV / slot_bytes(d, fmt)


def stride_bytes_per_token(hk: int, d: int, fmt: str) -> int:
    """cache[..., head, :].stride at the head axis == slot_bytes."""
    return slot_bytes(d, fmt)
