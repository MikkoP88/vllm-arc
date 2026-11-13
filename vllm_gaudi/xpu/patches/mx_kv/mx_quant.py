# SPDX-License-Identifier: Apache-2.0
"""mx_quant.py — MXFP4/MXFP8 KV-cache quantization reference (P6.47,
vllm_gaudi/xpu/patches/mx_kv/).

PURPOSE: bit-exact PyTorch reference for the MX (OCP microscaling) KV
cache lane directed 2026-11-11: every Triton kernel in this family
(triton_mx_store / triton_mx_decode) must round-trip THIS module's
semantics exactly; the offline harness (tools/ar/p147_mxkv/
p147_mxkv_offline.py) enforces it on XPU before any engine
integration leg.

FORMAT (OCP MX, per 32-element group, per (token, head) row):
- E8M0 group scale: scale = 2^ceil(log2(amax / EMAX)) with EMAX the
  element-format max (448 for e4m3, 6 for e2m1); amax == 0 encodes
  code 0 (2^-127); codes clamp to [0, 254] (255 is the E8M0 NaN).
- MXFP8 elements: e4m3 (e4m3fn — no inf, NaN only at 0x7F), cast RN
  with saturation after division by the group scale.
- MXFP4 elements: e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} x sign, RN
  thresholds at the midpoints (.25/.75/1.25/1.75/2.5/3.5/5.0); nibble
  = (sign << 3) | index, two nibbles per byte (even element low).

SLOT LAYOUT (TQ-AoS compatible, per (token, head) in the uint8 paged
cache; blocks interleaved [K][Ks][V][Vs] — see mx_layout, the single
layout source of truth this module mirrors):
  MXFP8:  K e4m3 D + Ks | V e4m3 D + Vs          (D=128 -> 264 B)
  MXFP4:  K e2m1 D/2 + Ks | V e2m1 D/2 + Vs      (D=128 -> 136 B)
  MXK8V4: K e4m3 D + Ks | V e2m1 D/2 + Vs        (D=128 -> 200 B)
Capacity vs fp16 KV (2*2*D): MXFP8 = 1.94x, MXK8V4 = 2.56x, MXFP4 =
3.76x; vs the certified fp8_e4m3 KV (2*D): MXK8V4 = 1.28x, MXFP4 = 1.88x.
MXK8V4 is the quality-doctrine mix (8-bit keys, 4-bit values — the
wheel's turboquant_k8v4 precedent): mxfp4's 4-bit keys degrade
long-return retrieval under KV churn (P6.47 cert24 gate run1 essay
REP-LOOP; banked results/2026-11-12-p647-mxkv-vstride-fix + gate).

Provenance: from-scratch in-house lane (owner directive 2026-11-11) —
no donor carries MX KV support (survey banked
results/2026-11-11-p647-mxkv-research/); kernel structure follows the
wheel's own sub-byte KV template (triton_turboquant_{store,decode},
int4_per_token_head — wheelref/ in the survey bank).
"""

from __future__ import annotations

import torch

GROUP = 32  # OCP MX group width (elements per E8M0 scale)
EMAX = {"mxfp8": 448.0, "mxfp4": 6.0}  # element-format maxima
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)  # RN thresholds


def slot_bytes(d: int, fmt: str) -> int:
    """Byte size of one (token, head) KV slot for an MX format."""
    n_groups = d // GROUP
    if fmt == "mxfp8":
        return 2 * d + 2 * n_groups
    if fmt == "mxfp4":
        return d + 2 * n_groups
    if fmt == "mxk8v4":
        return 3 * d // 2 + 2 * n_groups
    raise ValueError(fmt)


def ceil_log2_exact(r: torch.Tensor) -> torch.Tensor:
    """ceil(log2(r)) computed EXACTLY from r's IEEE-754 bits (r >= 0 fp32).

    exp = (biased_e - 127) + (mantissa != 0); zero/subnormal -> -127.
    This is the cross-implementation bit-exactness law: the Triton
    kernels mirror the same integer scheme on the same fp32 divide
    result, so no log2 ulp divergence can desync a scale byte.
    """
    bits = r.view(torch.int32) & 0x7FFFFFFF
    e = ((bits >> 23) & 0xFF) - 127
    m_nz = (bits & 0x7FFFFF) != 0
    exp = torch.where(e >= -127, e + m_nz.to(torch.int32), torch.full_like(e, -127))
    return exp


def e8m0_scales(x: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """E8M0 group scales for the last-dim groups of x [..., d].

    Returns (codes uint8 [..., d/GROUP], scales float32 same shape):
    scale = 2^(code-127) with code = clamp(127 + ceil_log2_exact(amax/EMAX)).
    Denormal amax (< EMAX * 2^-126) clamps to code 0 and its payload
    saturates — documented corner, cos-sim irrelevant.
    """
    d = x.shape[-1]
    if d % GROUP:
        raise ValueError(f"head dim {d} not a multiple of MX group {GROUP}")
    amax = x.float().abs().reshape(*x.shape[:-1], d // GROUP, GROUP).amax(-1)
    exp = ceil_log2_exact(amax / EMAX[fmt])
    code = (127 + exp).clamp(0, 254).to(torch.uint8)
    return code, torch.exp2(code.float() - 127.0)


def encode_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [..., d] to (fp8 bytes uint8-viewable [..., d], scale codes)."""
    code, scale = e8m0_scales(x, "mxfp8")
    q = (x.float().reshape(*x.shape[:-1], -1, GROUP) / scale[..., None]).reshape(
        *x.shape
    )
    return q.to(torch.float8_e4m3fn).view(torch.uint8), code


def _e2m1_index(q: torch.Tensor) -> torch.Tensor:
    """RN quantizer index [0, 7] for non-negative q on the e2m1 grid."""
    idx = torch.zeros_like(q)
    for i, m in enumerate(E2M1_MIDS):
        idx = idx + (q >= m).to(q.dtype)
    return idx


def encode_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode [..., d] to (packed nibble bytes [..., d/2], scale codes)."""
    code, scale = e8m0_scales(x, "mxfp4")
    q = (x.float().reshape(*x.shape[:-1], -1, GROUP) / scale[..., None]).reshape(
        *x.shape
    )
    sign = (q < 0).to(torch.uint8)
    idx = _e2m1_index(q.abs()).to(torch.uint8)
    nib = (sign << 3) | idx  # [..., d]
    d = x.shape[-1]
    pairs = nib.reshape(*nib.shape[:-1], d // 2, 2)
    return (pairs[..., 0] | (pairs[..., 1] << 4)), code


def encode_mxk8v4(
    k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixed-slot encode: (k e4m3 bytes, k scale codes, v e2m1 nibble
    bytes, v scale codes). Element math is exactly encode_mxfp8 on K and
    encode_mxfp4 on V (each leg's own EMAX); the Triton store kernel
    mirrors the same pair of paths (K4/V4 constexpr split).
    """
    kq, kcode = encode_mxfp8(k)
    vbytes, vcode = encode_mxfp4(v)
    return kq, kcode, vbytes, vcode


def _decode_e2m1(nib: torch.Tensor) -> torch.Tensor:
    """Arithmetic e2m1 decode: idx -> grid value (Triton mirrors this)."""
    idx = (nib & 0x7).float()
    e = torch.floor(idx / 2.0)
    m = idx - 2.0 * e
    mag = torch.where(e == 0, m * 0.5, torch.exp2(e - 1.0) * (1.0 + 0.5 * m))
    return torch.where(nib >= 8, -mag, mag)


def decode_kv(
    k_bytes: torch.Tensor, v_bytes: torch.Tensor, fmt: str, d: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode packed slot tensors [..., slot_bytes] to (K, V) float32 [..., d].

    This is the byte-layout ground truth: the Triton kernels must
    reproduce it from the same packed bytes.
    """
    n_groups = d // GROUP
    if fmt == "mxfp8":
        ks = k_bytes[..., d : d + n_groups]
        vs = v_bytes[..., d : d + n_groups]
        k = k_bytes[..., :d].view(torch.float8_e4m3fn).float() * torch.exp2(
            ks.float() - 127.0
        ).repeat_interleave(GROUP, dim=-1)
        v = v_bytes[..., :d].view(torch.float8_e4m3fn).float() * torch.exp2(
            vs.float() - 127.0
        ).repeat_interleave(GROUP, dim=-1)
        return k, v
    if fmt == "mxfp4":
        # K block lives at [0 : d/2 + n_groups) of k_bytes, V likewise of
        # v_bytes (each tensor carries its own block; see mx_layout).
        kl = k_bytes[..., : d // 2]
        ks = k_bytes[..., d // 2 : d // 2 + n_groups]
        lo = (kl & 0xF).to(torch.uint8)
        hi = (kl >> 4).to(torch.uint8)
        k = torch.stack(
            [_decode_e2m1(lo), _decode_e2m1(hi)], dim=-1
        ).reshape(*kl.shape[:-1], d) * torch.exp2(ks.float() - 127.0).repeat_interleave(
            GROUP, dim=-1
        )
        vl = v_bytes[..., : d // 2]
        vs = v_bytes[..., d // 2 : d // 2 + n_groups]
        lo = (vl & 0xF).to(torch.uint8)
        hi = (vl >> 4).to(torch.uint8)
        v = torch.stack(
            [_decode_e2m1(lo), _decode_e2m1(hi)], dim=-1
        ).reshape(*vl.shape[:-1], d) * torch.exp2(vs.float() - 127.0).repeat_interleave(
            GROUP, dim=-1
        )
        return k, v
    if fmt == "mxk8v4":
        # K block = e4m3 (mxfp8 leg), V block = e2m1 (mxfp4 leg) — the
        # slices arrive per-block ([K][Ks] | [V][Vs], mx_layout law).
        ks = k_bytes[..., d : d + n_groups]
        k = k_bytes[..., :d].view(torch.float8_e4m3fn).float() * torch.exp2(
            ks.float() - 127.0
        ).repeat_interleave(GROUP, dim=-1)
        vl = v_bytes[..., : d // 2]
        vs = v_bytes[..., d // 2 : d // 2 + n_groups]
        lo = (vl & 0xF).to(torch.uint8)
        hi = (vl >> 4).to(torch.uint8)
        v = torch.stack(
            [_decode_e2m1(lo), _decode_e2m1(hi)], dim=-1
        ).reshape(*vl.shape[:-1], d) * torch.exp2(vs.float() - 127.0).repeat_interleave(
            GROUP, dim=-1
        )
        return k, v
    raise ValueError(fmt)
