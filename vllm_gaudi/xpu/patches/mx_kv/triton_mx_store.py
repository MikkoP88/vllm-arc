# SPDX-License-Identifier: Apache-2.0
"""triton_mx_store.py — MXFP4/MXFP8 KV-cache fused store kernel (P6.47,
vllm_gaudi/xpu/patches/mx_kv/).

PURPOSE: quantize+pack+scatter one token's K and V rows into the paged
uint8 MX cache in a single kernel — the MX analogue of the wheel's
`_tq_fused_store_fp8` (triton_turboquant_store.py), on the identical
AoS slot geometry defined in mx_layout.py:

    mxfp8: [ K e4m3 x D ][ Ks x D/32 ][ V e4m3 x D ][ Vs x D/32 ]
    mxfp4: [ K e2m1 D/2 ][ Ks x D/32 ][ V e2m1 D/2 ][ Vs x D/32 ]

NUMERICS LAW (bit-exact vs mx_quant.py reference, enforced by the
offline harness): key and value are loaded through SEPARATE
token+head strides (upstream store law — the wheel's kernels take
stride_val_tok/stride_val_head; the two tensors arrive with
independent layouts); the E8M0 exponent is derived EXACTLY from the fp32
divide result's bits —
    r = amax / EMAX ; exp = (biased_e - 127) + (mantissa != 0)
    r zero/subnormal -> exp = -127 ; code = clamp(127 + exp, 0, 254)
so no log2/ceil ulp divergence can desync a scale byte between torch
and Triton (mx_quant.ceil_log2_exact is the mirrored reference).
Element quantization: mxfp8 = hardware RN cast to e4m3 after
group-scale division (q <= EMAX by construction); mxfp4 = the
7-midpoint compare chain of E2M1_MIDS, sign bit packed at nibble bit
3, even element in the low nibble.

EFFICIENCY (always-loop directive): one program per (token, head)
writes its whole slot contiguously — K, Ks, V, Vs — no separate scale
tensors, no gather, one E8M0 integer chain per block (codes and scale
from the SAME pass); E8M0 scales cost 1 B per 32 elements (vs the TQ
fp16 scale+zero 4 B per row), so the decode-side re-read is 8x
lighter on the metadata path.

GRAPH SAFETY: no host reads; the only branch is the device-side
negative-slot guard (uniform per program, upstream store law — see
the kernel comment); every access is unmasked (full D rows); safe
under XPU-graph capture once integrated (the engine leg wraps calls
exactly like the wheel's TQ store).

Provenance: from-scratch in-house lane (P6.47, 2026-11-11);
kernel structure follows the wheel's sub-byte KV template
(triton_turboquant_store.py — survey
results/2026-11-11-p647-mxkv-research/wheelref/).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .mx_layout import slot_bytes


@triton.jit
def _mx_e8m0(amax, EMAX: tl.constexpr):
    """(code uint8, scale f32) for one amax vector — the exact integer
    chain mirrored from mx_quant.ceil_log2_exact (see module docstring).
    EMAX arrives as a call-site literal (448.0 mxfp8 / 6.0 mxfp4):
    annotation-form constexpr globals are rejected by this Triton
    (leg-A finding #1, 2026-11-11)."""
    r = amax / EMAX
    rbits = r.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    e = ((rbits >> 23) & 0xFF) - 127
    mnz = (rbits & 0x7FFFFF) != 0
    exp = tl.where(e >= -127, e + mnz.to(tl.int32), -127)
    code = tl.minimum(tl.maximum(127 + exp, 0), 254).to(tl.uint8)
    return code, tl.exp2(code.to(tl.float32) - 127.0)


@triton.jit
def _mx_e2m1_nibbles(q):
    # q in [-6, 6] after scale division; 7-midpoint RN chain (E2M1_MIDS),
    # index 0..7, sign at nibble bit 3.
    a = tl.abs(q)
    idx = (
        (a >= 0.25).to(tl.uint8)
        + (a >= 0.75).to(tl.uint8)
        + (a >= 1.25).to(tl.uint8)
        + (a >= 1.75).to(tl.uint8)
        + (a >= 2.5).to(tl.uint8)
        + (a >= 3.5).to(tl.uint8)
        + (a >= 5.0).to(tl.uint8)
    )
    return ((q < 0).to(tl.uint8) << 3) | idx


@triton.jit
def _mx_pack_nibbles(nib2, D: tl.constexpr):
    # [G, 32] nibbles -> [D/2] bytes: even element low nibble.
    # Weights [1, 16] (shift semantics 2^[0,4] — arange*16 gives [0,16]
    # and annihilates the low nibble: leg-A finding #3, 2026-11-11).
    pairs = tl.reshape(nib2, (D // 2, 2)).to(tl.int32)
    w = (tl.arange(0, 2) * 15 + 1)[None, :]
    return tl.sum(pairs * w, axis=1).to(tl.uint8)


@triton.jit
def _mx_fused_store(
    k_ptr,
    v_ptr,
    cache_ptr,
    tok_block_ptr,
    tok_pos_ptr,
    stride_k_n,
    stride_k_h,
    stride_v_n,
    stride_v_h,
    stride_c_b,
    stride_c_t,
    stride_c_h,
    D: tl.constexpr,
    N_GROUPS: tl.constexpr,
    K4: tl.constexpr,
    V4: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)

    # NEGATIVE-SLOT GUARD (upstream store law — triton_turboquant_store.py
    # guards `if slot < 0` in both kernel variants): rows whose slot id is
    # negative are SKIPPED, never written. The engine's dummy/warmup
    # batches map every row to slot -1 (rig-proven: the 8192-row warmup
    # prefill and the [-1]*8 pad batches, trace9) — without the guard each
    # such row stores 2x slot_bytes BEFORE the pool base (blk=-1 ->
    # -stride_c_b + 63*stride_c_t = -1056/-528 B at D=256), an
    # out-of-bounds write into the tensor allocated immediately before
    # the pool (P6.47 spec-text-corruption root cause, 2026-11-12).
    # Device-side uniform branch — resolves per program at replay, so
    # XPU-graph capture stays safe (same law as the upstream guard).
    blk = tl.load(tok_block_ptr + tok)
    if blk < 0:
        return
    pos = tl.load(tok_pos_ptr + tok)

    # K/V rows as [N_GROUPS, 32] so the group amax is one tl.max(axis=1).
    # SEPARATE k/v token+head strides (upstream law — the wheel's store
    # kernels take stride_val_tok/stride_val_head distinct from the key
    # strides: key and value arrive at the store call with INDEPENDENT
    # layouts). Sharing keys' strides for the v load read wrong addresses
    # whenever the layouts differ — stored-V rel-L2 42-80% with K clean
    # at 2-3% (trace24 chunk_snr, P6.47 root cause #2, 2026-11-12); the
    # offline harness's matching-stride tensors could never see it.
    g = tl.arange(0, N_GROUPS)
    lane = tl.arange(0, 32)
    offs2 = g[:, None] * 32 + lane[None, :]
    k2 = tl.load(
        k_ptr + tok.to(tl.int64) * stride_k_n + head * stride_k_h + offs2
    ).to(tl.float32)
    v2 = tl.load(
        v_ptr + tok.to(tl.int64) * stride_v_n + head * stride_v_h + offs2
    ).to(tl.float32)

    slot = (
        blk.to(tl.int64) * stride_c_b
        + pos.to(tl.int64) * stride_c_t
        + head * stride_c_h
    )
    flat = tl.reshape(offs2, (D,))

    # K block size decides where V starts (mx_layout law); each leg picks
    # its element path independently — mxfp8 (0,0), mxfp4 (1,1), mxk8v4
    # (0,1): 8-bit keys keep retrieval precision, 4-bit values ride the
    # byte savings (turboquant_k8v4 quality doctrine; P6.47 gate run1:
    # 4-bit K degrades long-return under churn).
    v_off = (D // 2 if K4 else D) + N_GROUPS
    # ---- K ----
    if K4:
        kcode, kscale = _mx_e8m0(tl.max(tl.abs(k2), axis=1), 6.0)
        knib = _mx_e2m1_nibbles(k2 / kscale[:, None])
        tl.store(cache_ptr + slot + tl.arange(0, D // 2), _mx_pack_nibbles(knib, D))
        tl.store(cache_ptr + slot + D // 2 + g, kcode)
    else:
        kcode, kscale = _mx_e8m0(tl.max(tl.abs(k2), axis=1), 448.0)
        kq = (k2 / kscale[:, None]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(cache_ptr + slot + flat, tl.reshape(kq, (D,)))
        tl.store(cache_ptr + slot + D + g, kcode)
    # ---- V ----
    if V4:
        vcode, vscale = _mx_e8m0(tl.max(tl.abs(v2), axis=1), 6.0)
        vnib = _mx_e2m1_nibbles(v2 / vscale[:, None])
        tl.store(cache_ptr + slot + v_off + tl.arange(0, D // 2), _mx_pack_nibbles(vnib, D))
        tl.store(cache_ptr + slot + v_off + D // 2 + g, vcode)
    else:
        vcode, vscale = _mx_e8m0(tl.max(tl.abs(v2), axis=1), 448.0)
        vq = (v2 / vscale[:, None]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        tl.store(cache_ptr + slot + v_off + flat, tl.reshape(vq, (D,)))
        tl.store(cache_ptr + slot + v_off + D + g, vcode)


def mx_store(
    keys: torch.Tensor,
    values: torch.Tensor,
    cache: torch.Tensor,
    tok_block: torch.Tensor,
    tok_pos: torch.Tensor,
) -> str:
    """Host wrapper. keys/values [N, Hk, D] (fp32/fp16/bf16, device);
    cache uint8 [num_blocks, block_size, Hk, slot_bytes(fmt)]; tok_block
    / tok_pos int32 [N] the paged destination of each token. Format is
    inferred from the cache's slot width (mx_layout law). Returns fmt.
    """
    n, hk, d = keys.shape
    sb = cache.shape[-1]
    want = {
        f: slot_bytes(d, f) for f in ("mxfp8", "mxfp4", "mxk8v4")
    }
    fmt = next(f for f, b in want.items() if b == sb)
    # The kernels index rows assuming element-contiguous D (upstream law);
    # normalize the never-happens non-contiguous tail case defensively.
    if keys.stride(2) != 1:
        keys = keys.contiguous()
    if values.stride(2) != 1:
        values = values.contiguous()
    _mx_fused_store[(n, hk)](
        keys,
        values,
        cache,
        tok_block,
        tok_pos,
        keys.stride(0),
        keys.stride(1),
        values.stride(0),
        values.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        D=d,
        N_GROUPS=d // 32,
        K4=fmt in ("mxfp4",),
        V4=fmt in ("mxfp4", "mxk8v4"),
        num_warps=1,
    )
    return fmt
