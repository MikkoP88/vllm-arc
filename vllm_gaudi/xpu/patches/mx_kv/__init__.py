# SPDX-License-Identifier: Apache-2.0
"""mx_kv — MXFP4/MXFP8/MXK8V4 KV-cache lane (P6.47, in-house,
2026-11-11).

Layered entry point for the MX (OCP microscaling) KV-cache-dtype
support on --kv-cache-dtype {mxfp4, mxfp8, mxk8v4} (mxk8v4 = e4m3 K +
e2m1 V, the turboquant_k8v4 quality doctrine):

- mx_layout:  slot byte geometry + pool arithmetic (single source of
  truth shared by kernels, engine integration, reports).
- mx_quant:   bit-exact PyTorch reference encode/decode (the numerics
  law the Triton kernels must round-trip).
- triton_mx_store:  fused quantize+pack+scatter into the paged cache.
- triton_mx_decode: paged flash decode with inline MX dequant.

Validation: tools/ar/p147_mxkv/p147_mxkv_offline.py (reference-vs-
kernel round-trip, decode-vs-exact-attention, XPU telemetry) — leg A
gate before any engine integration. Survey/provenance:
results/2026-11-11-p647-mxkv-research/.
"""

from .mx_layout import capacity_mult, offsets, pool_bytes, pool_gib, slot_bytes
from .mx_quant import (
    E2M1_MIDS,
    EMAX,
    GROUP,
    ceil_log2_exact,
    decode_kv,
    encode_mxfp4,
    encode_mxfp8,
    encode_mxk8v4,
)
from .triton_mx_decode import mx_decode, mx_decode_rows, mx_decode_rows_dot
from .triton_mx_store import mx_store

__all__ = [
    "E2M1_MIDS",
    "EMAX",
    "GROUP",
    "capacity_mult",
    "ceil_log2_exact",
    "decode_kv",
    "encode_mxfp4",
    "encode_mxfp8",
    "encode_mxk8v4",
    "mx_decode",
    "mx_decode_rows",
    "mx_decode_rows_dot",
    "mx_store",
    "offsets",
    "pool_bytes",
    "pool_gib",
    "slot_bytes",
]
