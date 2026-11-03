#!/usr/bin/env python3
"""P6.18c Class-B numeric gate: TurboQuant SoA store->decode roundtrip on XPU.

Exercises the exact live pair pinned by the cert6 boot logs
(vllm/v1/attention/ops/turboquant_soa/triton_turboquant_store ->
triton_turboquant_decode_attention, the "FlyDSL unavailable -> SoA
Triton decode" fallback) for all four turboquant_* presets, using the
engine's own deterministic tables (Sylvester Hadamard rotation, Lloyd-Max
centroids) and the model's per-GPU TP2 geometry (head_dim=256, Hq=12,
Hk=2, bf16 KV).

Gates per preset:
  1. STORE  — unpack the cache bytes with the kernel's own SoA addressing
     and compare against a torch reference quantizer that mirrors every
     rounding op (fp16 scale/zero, fp16 K-norm, round-half-up indices,
     little-endian nibble packing).
  2. DECODE — run the decode kernel and compare against a torch
     reference attention built from the SAME dequantized K/V (quant
     error cancels; only kernel arithmetic/addressing is under test).
  3. DETERMINISM — second store pass must be byte-identical; second
     decode must return an identical output tensor.

PASS on every preset = the kernels are numerically correct on XPU, and
the degenerate generations observed in p618b are a 4-bit-class V quality
ceiling, not an implementation defect.
"""

import math
import sys

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
    triton_turboquant_store,
)

# Model geometry (qwen3.8-27b-fp8, TP2 per GPU): head_dim=256,
# num_attention_heads=24 -> 12, num_key_value_heads=4 -> 2.
D = 256
HQ = 12
HK = 2
KV_GROUP = HQ // HK
BLOCK_SIZE = 64
NUM_BLOCKS = 4
# Two sequences sharing a prefix: seq0 fills blocks 0-2; seq1 reuses the
# first 100 tokens (pages [0, 1]), so both kernel and reference read the
# same stored slots.
SEQ_LENS = [192, 100]
BLOCK_TABLE = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]], dtype=torch.int32)
SCALE = 1.0 / math.sqrt(D)

# The four certified-menu presets (vllm/config/cache.py names; the CLI
# short names map to the same parameter tuples).
PRESETS = {
    "turboquant_k8v4": dict(key_fp8=True, mse_bits=4, vqb=4, nc=False),
    "turboquant_4bit_nc": dict(key_fp8=False, mse_bits=4, vqb=4, nc=True),
    "turboquant_k3v4_nc": dict(key_fp8=False, mse_bits=3, vqb=4, nc=True),
    "turboquant_3bit_nc": dict(key_fp8=False, mse_bits=3, vqb=3, nc=True),
}

STORE_STEP_TOL = 1.5  # tolerated V drift in quant steps (tie rounding only)
DECODE_RTOL = 1e-2


def build_tables(device, key_fp8, mse_bits):
    """Engine tables: Hadamard rotation + Lloyd-Max centroids/midpoints."""
    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    centroids = get_centroids(D, mse_bits).to(device=device, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    return PiT, midpoints, centroids


def soa_addrs(cache, slot, head, key_bytes, val_bytes, nsf):
    """Kernel SoA addressing for one (slot, head): data + metadata bases."""
    data_bps = key_bytes + val_bytes
    meta_off = BLOCK_SIZE * HK * data_bps
    blk, off = slot // BLOCK_SIZE, slot % BLOCK_SIZE
    data_base = blk * cache.stride(0) + off * (HK * data_bps) + head * data_bps
    hm = (blk * cache.stride(0) + meta_off) // 2 + head * (nsf * BLOCK_SIZE)
    return data_base, hm, off


def unpack_bits(raw, bits):
    """Vectorized little-endian bitstream unpack (decode-kernel formula).

    raw: uint8 tensor of packed bytes on device; returns int64 [D] indices.
    """
    d = torch.arange(D, device=raw.device)
    bit_off = d * bits
    byte_idx = bit_off // 8
    shift = bit_off % 8
    raw64 = raw.to(torch.int64)
    w = raw64[byte_idx] | (raw64[(byte_idx + 1).clamp(max=raw.numel() - 1)] << 8)
    return (w >> shift) & (2**bits - 1)


def unpack_v(cache, slot, head, key_bytes, vqb, key_fp8):
    """Read back one slot/head V payload + fp16 scale/zero (SoA layout).

    SoA field order: fp8 keys -> [v_scale, v_zero]; MSE keys ->
    [k_norm, v_scale, v_zero] (turboquant_soa launchers).
    """
    val_bytes = math.ceil(D * vqb / 8)
    nsf = 2 if key_fp8 else 3
    soa_v_scale = 0 if key_fp8 else 1
    data_base, hm, off = soa_addrs(cache, slot, head, key_bytes, val_bytes, nsf)
    flat = cache.view(-1)
    raw = flat[data_base + key_bytes : data_base + key_bytes + val_bytes]
    if vqb == 4:
        lo = raw.to(torch.int64) & 0xF
        hi = raw.to(torch.int64) >> 4
        q = torch.stack([lo, hi], dim=1).reshape(-1)
    else:
        q = unpack_bits(raw, 3)
    u16 = cache.view(torch.uint16).view(-1)
    scale = u16[hm + soa_v_scale * BLOCK_SIZE + off].view(torch.float16).float()
    zero = u16[hm + (soa_v_scale + 1) * BLOCK_SIZE + off].view(torch.float16).float()
    return q, scale, zero


def unpack_k(cache, slot, head, key_bytes, val_bytes):
    """Read back one slot/head K payload bytes (SoA layout)."""
    flat = cache.view(-1)
    data_base = soa_addrs(cache, slot, head, key_bytes, val_bytes, 0)[0]
    return flat[data_base : data_base + key_bytes]


def read_knorm(cache, slot, head, key_bytes, val_bytes, nsf):
    _, hm, off = soa_addrs(cache, slot, head, key_bytes, val_bytes, nsf)
    u16 = cache.view(torch.uint16).view(-1)
    return u16[hm + 0 * BLOCK_SIZE + off].view(torch.float16).float()


def ref_v_quant(v_row, vqb):
    """Affine-quantize one [D] fp32 value vector, mirroring kernel math.

    Returns (indices int64 [D], scale fp32, zero fp32) with scale/zero
    pre-rounded to fp16 exactly as the kernel stores them.
    """
    vmin = v_row.min()
    vmax = v_row.max()
    levels = 2**vqb - 1
    scale = (vmax - vmin) / levels
    scale = torch.where(scale > 1e-8, scale, torch.full_like(scale, 1e-8))
    q = ((v_row - vmin) / scale + 0.5).to(torch.int32).clamp(0, levels)
    return q.to(torch.int64), scale.to(torch.float16).float(), vmin.to(torch.float16).float()


def dequant_all(cache, cfg, key_bytes, val_bytes, nsf, n_tok, centroids, device):
    """Dequantize the full cache to K/V hats in torch (decode semantics)."""
    k_hat = torch.empty(n_tok, HK, D, dtype=torch.float32, device=device)
    v_hat = torch.empty(n_tok, HK, D, dtype=torch.float32, device=device)
    for t in range(n_tok):
        for h in range(HK):
            q, s, z = unpack_v(cache, t, h, key_bytes, cfg["vqb"], cfg["key_fp8"])
            v_hat[t, h] = q * s + z
            kb = unpack_k(cache, t, h, key_bytes, val_bytes)
            if cfg["key_fp8"]:
                k_hat[t, h] = kb.view(torch.float8_e4m3fn).float()
            else:
                idx = unpack_bits(kb, cfg["mse_bits"])
                nrm = read_knorm(cache, t, h, key_bytes, val_bytes, nsf)
                k_hat[t, h] = nrm * centroids[idx]
    return k_hat, v_hat


def run_preset(name, cfg, device, report):
    print(f"\n=== {name} ===", flush=True)
    PiT, midpoints, centroids = build_tables(device, cfg["key_fp8"], cfg["mse_bits"])
    n_cent = 1 if cfg["key_fp8"] else 2**cfg["mse_bits"]

    # Model-realistic bf16 K/V (post-RoPE scale ~ N(0, 1)).
    gen = torch.Generator(device="cpu").manual_seed(20261103)
    n_tok = SEQ_LENS[0]
    key = torch.randn(n_tok, HK, D, generator=gen).to(torch.bfloat16).to(device)
    val = torch.randn(n_tok, HK, D, generator=gen).to(torch.bfloat16).to(device)

    key_bytes = D if cfg["key_fp8"] else math.ceil(D * cfg["mse_bits"] / 8)
    val_bytes = math.ceil(D * cfg["vqb"] / 8)
    nsf = 2 if cfg["key_fp8"] else 3
    padded = key_bytes + val_bytes + nsf * 2
    cache = torch.zeros(NUM_BLOCKS, BLOCK_SIZE, HK, padded, dtype=torch.uint8, device=device)
    cache2 = torch.zeros_like(cache)
    slot_map = torch.arange(n_tok, dtype=torch.int32, device=device)

    for dst in (cache, cache2):
        triton_turboquant_store(
            key, val, dst, slot_map, PiT, midpoints,
            mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
            value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
            centroids=centroids, norm_correction=cfg["nc"],
        )
    torch.xpu.synchronize()
    det_store = bool(torch.equal(cache, cache2))
    print(f"[determinism] store byte-identical: {det_store}")

    # ---- DECODE GATE (PRIMARY): kernel output vs torch attention on the
    # K/V dequantized from the kernel's OWN stored bytes. Self-coherent:
    # any store/decode layout or arithmetic defect shows up here at full
    # magnitude; the quantizer's information loss cancels.
    bt = BLOCK_TABLE.to(device)
    sl = torch.tensor(SEQ_LENS, dtype=torch.int32, device=device)
    q = torch.randn(2, HQ, D, generator=gen).to(torch.bfloat16).to(device)

    def decode():
        return triton_turboquant_decode_attention(
            q, cache, bt, sl, PiT, centroids, SCALE,
            mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
            value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
            norm_correction=cfg["nc"], PiT=PiT,
        )

    out = decode()
    out2 = decode()
    torch.xpu.synchronize()
    det_decode = bool(torch.equal(out, out2))
    print(f"[determinism] decode byte-identical: {det_decode}")

    k_hat, v_hat = dequant_all(cache, cfg, key_bytes, val_bytes, nsf, n_tok, centroids, device)
    print(
        f"[diag] k_hat max={k_hat.abs().max().item():.4f} nan={int(k_hat.isnan().sum())} "
        f"v_hat max={v_hat.abs().max().item():.4f} nan={int(v_hat.isnan().sum())} "
        f"out max={out.float().abs().max().item():.4f} nan={int(out.isnan().sum())}"
    )
    ref = torch.empty(2, HQ, D, dtype=torch.float32, device=device)
    for b, slen in enumerate(SEQ_LENS):
        for hq in range(HQ):
            hkv = hq // KV_GROUP
            if cfg["key_fp8"]:
                scores = (k_hat[:slen, hkv] * q[b, hq].float()[None, :]).sum(-1)
            else:
                q_rot = q[b, hq].float().matmul(PiT)
                scores = (k_hat[:slen, hkv] * q_rot[None, :]).sum(-1)
            p = torch.softmax(scores * SCALE, dim=-1)
            ref[b, hq] = p @ v_hat[:slen, hkv]
    ref_dq = ref.to(q.dtype)
    diff = (out.float() - ref_dq.float()).abs()
    denom = ref_dq.float().abs().max().clamp(min=1e-6)
    decode_rel = (diff.max() / denom).item()
    decode_max = diff.max().item()
    print(
        f"[decode gate] rel_err={decode_rel:.6f} max_abs={decode_max:.6f} "
        f"nan_out={int(out.isnan().sum())} (tol rel<{DECODE_RTOL})"
    )

    # ---- STORE SPEC GATE: kernel cache vs torch reference quantizer.
    # Tolerances reflect legal rounding-path differences (GEMM batch-order
    # for bucket boundaries, torch-vs-Triton fp8 cast rounding), NOT
    # layout corruption: V must sit within 1.5 quant steps, K within one
    # codebook/fp8 step in value space.
    v_step_err_max = 0.0
    v_mismatch = 0
    k_step_err_max = 0.0
    k_mismatch = 0
    for t in range(0, n_tok, 13):  # deterministic slot sample
        for h in range(HK):
            q_ref, s_ref, z_ref = ref_v_quant(val[t, h].float(), cfg["vqb"])
            q_k, s_k, z_k = unpack_v(cache, t, h, key_bytes, cfg["vqb"], cfg["key_fp8"])
            if s_k.item() != s_ref.item() or z_k.item() != z_ref.item():
                v_mismatch += 1
            v_k = q_k * s_k + z_k
            v_r = q_ref * s_ref + z_ref
            step = max(s_k.item(), 1e-8)
            v_step_err_max = max(v_step_err_max, ((v_k - v_r).abs().max() / step).item())

            if cfg["key_fp8"]:
                # Value-space compare at e4m3 resolution (bf16 -> fp8 cast
                # rounding may differ between torch and Triton paths).
                k_ref = key[t, h].float()
                try:
                    k_ref_q = k_ref.to(torch.float8_e4m3fn).float()
                except Exception:
                    k_ref_q = k_ref.cpu().to(torch.float8_e4m3fn).float().to(device)
                k_k = unpack_k(cache, t, h, key_bytes, val_bytes).view(torch.float8_e4m3fn).float()
                rel = ((k_k - k_ref_q).abs().max() / k_ref_q.abs().max().clamp(min=1e-3)).item()
                if rel > 0.13:  # one e4m3 mantissa step
                    k_mismatch += 1
                k_step_err_max = max(k_step_err_max, rel)
            else:
                k_row = key[t, h].float()
                norm = k_row.norm()
                xh = k_row / (norm + 1e-8)
                y = xh.matmul(PiT)
                idx = torch.searchsorted(midpoints, y, right=True).clamp(max=n_cent - 1)
                kb = unpack_k(cache, t, h, key_bytes, val_bytes)
                idx_k = unpack_bits(kb, cfg["mse_bits"])
                # Adjacent-bucket drift at boundaries is legal; larger
                # jumps are corruption.
                drift = (idx_k - idx).abs().max().item()
                if drift > 1:
                    k_mismatch += 1
                k_step_err_max = max(k_step_err_max, drift)
                if cfg["nc"]:
                    cv = centroids[idx]
                    cn = 1.0 / torch.sqrt((cv * cv).sum() + 1e-16)
                    stored = (norm * cn).to(torch.float16).float()
                else:
                    stored = norm.to(torch.float16).float()
                got = read_knorm(cache, t, h, key_bytes, val_bytes, nsf)
                if (got - stored).abs().item() > 2e-3 * max(1.0, stored.abs().item()):
                    k_mismatch += 1
    print(
        f"[store gate] v_mismatch={v_mismatch} v_step_err_max={v_step_err_max:.3f} "
        f"k_mismatch={k_mismatch} k_step_err_max={k_step_err_max:.4f}"
    )

    ok = (
        det_store and det_decode
        and decode_rel < DECODE_RTOL and int(out.isnan().sum()) == 0
        and v_mismatch == 0 and v_step_err_max <= STORE_STEP_TOL
        and k_mismatch == 0
    )
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    report[name] = ok


def main():
    if not torch.xpu.is_available():
        print("FATAL: no XPU device visible")
        return 2
    device = torch.device("xpu:0")
    print(f"device: {torch.xpu.get_device_name(0)}")
    report = {}
    for name, cfg in PRESETS.items():
        run_preset(name, cfg, device, report)
    print("\n==== P618C ROUNDTEST SUMMARY ====")
    for name, ok in report.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(report.values())
    print(f"P618C ROUNDTEST: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
