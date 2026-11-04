#!/usr/bin/env python3
"""P6.19d MQ-kernel gate: TurboQuant multi-query verify attention (SoA) on XPU.

Validates the tq_mq_soa kernel pair (the donor-v19 MQ structure adapted
to the stock Opt#3 SoA layout) offline against the same live store
kernel the engine runs, before any serve leg (t8) uses it. Pattern and
helpers from the banked p618c roundtest (results/2026-11-03-p618c-
roundtest/p618c_tq_roundtrip.py).

Gates per preset (all four turboquant_* presets):
  1. MQ vs REFERENCE — torch per-row causal attention on the K/V
     dequantized from the kernel's own stored bytes (quant error
     cancels; kernel arithmetic/addressing under test). Row r attends
     positions [0, q0+r) — the split-mode semantics.
  2. MQ vs SPLIT — the deployed fix's dispatch (q_len single-row
     triton_turboquant_decode_attention calls with per-row length
     q0+r, leg t5-proven clean under graphs). Same cache, same tables:
     only accumulation order may differ.
  3. DETERMINISM — second MQ call byte-identical.
Shapes: Q_LEN 4 (engine k3 verify), 5 (Q_BLOCK padding path), 2 (edge);
seq 192 (crosses 32 splits: split_len 6) and short seq 8.
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
from vllm_gaudi.xpu.patches.tq_mq_soa import (
    triton_turboquant_mq_decode_attention_soa,
)

D = 256
HQ = 12
HK = 2
KV_GROUP = HQ // HK
BLOCK_SIZE = 64
NUM_BLOCKS = 4
SEQ_LENS = [192, 100]
BLOCK_TABLE = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]], dtype=torch.int32)
SCALE = 1.0 / math.sqrt(D)

PRESETS = {
    "turboquant_k8v4": dict(key_fp8=True, mse_bits=4, vqb=4, nc=False),
    "turboquant_4bit_nc": dict(key_fp8=False, mse_bits=4, vqb=4, nc=True),
    "turboquant_k3v4_nc": dict(key_fp8=False, mse_bits=3, vqb=4, nc=True),
    "turboquant_3bit_nc": dict(key_fp8=False, mse_bits=3, vqb=3, nc=True),
}

REF_RTOL = 1e-2     # roundtest decode-gate tolerance (quant cancels)
SPLIT_RTOL = 1e-2   # same math, different accumulation order


def build_tables(device, key_fp8, mse_bits):
    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    centroids = get_centroids(D, mse_bits).to(device=device, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
    return PiT, midpoints, centroids


def soa_addrs(cache, slot, head, key_bytes, val_bytes, nsf):
    data_bps = key_bytes + val_bytes
    meta_off = BLOCK_SIZE * HK * data_bps
    blk, off = slot // BLOCK_SIZE, slot % BLOCK_SIZE
    data_base = blk * cache.stride(0) + off * (HK * data_bps) + head * data_bps
    hm = (blk * cache.stride(0) + meta_off) // 2 + head * (nsf * BLOCK_SIZE)
    return data_base, hm, off


def unpack_bits(raw, bits):
    d = torch.arange(D, device=raw.device)
    bit_off = d * bits
    byte_idx = bit_off // 8
    shift = bit_off % 8
    raw64 = raw.to(torch.int64)
    w = raw64[byte_idx] | (raw64[(byte_idx + 1).clamp(max=raw.numel() - 1)] << 8)
    return (w >> shift) & (2**bits - 1)


def unpack_v(cache, slot, head, key_bytes, vqb, key_fp8):
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
    flat = cache.view(-1)
    data_base = soa_addrs(cache, slot, head, key_bytes, val_bytes, 0)[0]
    return flat[data_base : data_base + key_bytes]


def read_knorm(cache, slot, head, key_bytes, val_bytes, nsf):
    _, hm, off = soa_addrs(cache, slot, head, key_bytes, val_bytes, nsf)
    u16 = cache.view(torch.uint16).view(-1)
    return u16[hm + 0 * BLOCK_SIZE + off].view(torch.float16).float()


def dequant_all(cache, cfg, key_bytes, val_bytes, nsf, n_tok, centroids, device):
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


def run_case(name, cfg, device, q_len, seq_len, report):
    """One (preset, Q_LEN, seq_len) case: store fresh, run the 3 gates."""
    PiT, midpoints, centroids = build_tables(device, cfg["key_fp8"], cfg["mse_bits"])
    gen = torch.Generator(device="cpu").manual_seed(20261104)
    n_tok = seq_len
    key = torch.randn(n_tok, HK, D, generator=gen).to(torch.bfloat16).to(device)
    val = torch.randn(n_tok, HK, D, generator=gen).to(torch.bfloat16).to(device)

    key_bytes = D if cfg["key_fp8"] else math.ceil(D * cfg["mse_bits"] / 8)
    val_bytes = math.ceil(D * cfg["vqb"] / 8)
    nsf = 2 if cfg["key_fp8"] else 3
    padded = key_bytes + val_bytes + nsf * 2
    n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache = torch.zeros(n_blocks, BLOCK_SIZE, HK, padded, dtype=torch.uint8, device=device)
    slot_map = torch.arange(n_tok, dtype=torch.int32, device=device)
    triton_turboquant_store(
        key, val, cache, slot_map, PiT, midpoints,
        mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
        value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
        centroids=centroids, norm_correction=cfg["nc"],
    )
    torch.xpu.synchronize()

    # Verify-shaped query: [Q_LEN, Hq, D]; block table = one request
    # walking blocks 0..n_blocks-1; q0 = row-0 limit = seq - (q_len - 1).
    q = torch.randn(q_len, HQ, D, generator=gen).to(torch.bfloat16).to(device)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=device).view(1, -1)
    q0 = seq_len - (q_len - 1)

    mq = triton_turboquant_mq_decode_attention_soa(
        q, cache, bt, torch.tensor([q0], dtype=torch.int32, device=device),
        PiT, centroids, SCALE,
        mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
        value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
        norm_correction=cfg["nc"], PiT=PiT,
    )
    mq2 = triton_turboquant_mq_decode_attention_soa(
        q, cache, bt, torch.tensor([q0], dtype=torch.int32, device=device),
        PiT, centroids, SCALE,
        mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
        value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
        norm_correction=cfg["nc"], PiT=PiT,
    )
    torch.xpu.synchronize()
    det = bool(torch.equal(mq, mq2))

    # Gate 2: split-mode dispatch (t5 fix) per row.
    split = torch.empty_like(mq)
    for r in range(q_len):
        out_r = triton_turboquant_decode_attention(
            q[r : r + 1], cache, bt,
            torch.tensor([q0 + r], dtype=torch.int32, device=device),
            PiT, centroids, SCALE,
            mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
            value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
            norm_correction=cfg["nc"], PiT=PiT,
        )
        split[r : r + 1] = out_r.to(mq.dtype)
    torch.xpu.synchronize()

    # Gate 1: torch reference on dequantized cache (quant cancels).
    k_hat, v_hat = dequant_all(cache, cfg, key_bytes, val_bytes, nsf, n_tok, centroids, device)
    ref = torch.empty(q_len, HQ, D, dtype=torch.float32, device=device)
    for r in range(q_len):
        lim = q0 + r  # row r attends [0, lim)
        for hq in range(HQ):
            hkv = hq // KV_GROUP
            if cfg["key_fp8"]:
                scores = (k_hat[:lim, hkv] * q[r, hq].float()[None, :]).sum(-1)
            else:
                q_rot = q[r, hq].float().matmul(PiT)
                scores = (k_hat[:lim, hkv] * q_rot[None, :]).sum(-1)
            p = torch.softmax(scores * SCALE, dim=-1)
            ref[r, hq] = p @ v_hat[:lim, hkv]

    mref = ((mq.float() - ref).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
    msplit = (
        (mq.float() - split.float()).abs().max()
        / split.float().abs().max().clamp(min=1e-6)
    ).item()
    nan_ct = int(mq.isnan().sum())
    ok = det and mref < REF_RTOL and msplit < SPLIT_RTOL and nan_ct == 0
    tag = f"{name} q{q_len} seq{seq_len}"
    print(
        f"[{tag}] {'PASS' if ok else 'FAIL'} rel_ref={mref:.6f} "
        f"rel_split={msplit:.6f} det={det} nan={nan_ct}",
        flush=True,
    )
    report[tag] = ok


def main():
    if not torch.xpu.is_available():
        print("FATAL: no XPU device visible")
        return 2
    device = torch.device("xpu:0")
    print(f"device: {torch.xpu.get_device_name(0)}")
    report = {}
    for name, cfg in PRESETS.items():
        for q_len, seq_len in ((4, 192), (5, 192), (2, 192), (4, 8)):
            run_case(name, cfg, device, q_len, seq_len, report)
    print("\n==== P619D MQVAL SUMMARY ====")
    for tag, ok in report.items():
        print(f"  {tag}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(report.values())
    print(f"P619D MQVAL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
