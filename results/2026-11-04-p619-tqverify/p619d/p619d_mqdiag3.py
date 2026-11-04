#!/usr/bin/env python3
"""P6.19d MQ DIAG 3 — stage-1 partials wrong (torch-combine == stage2 ==
same wrong). With split_len=1 each split holds ONE token, so partial s is
exactly: o_s = v_hat[s] (weight 1), lse_s = (q.row3 . k_hat[s]) * SCALE.
This leg dumps those analytically (roundtest dequant) vs the kernel's
mid_o for head0/row3, splits 0-7. k8v4 q4 seq8.
"""

import math

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_store import (
    triton_turboquant_store,
)
from vllm_gaudi.xpu.patches import tq_mq_soa as mq

D = 256
HQ = 12
HK = 2
BLOCK_SIZE = 64
SCALE = 1.0 / math.sqrt(D)
SPLITS = 32


def soa_addrs(cache, slot, head, key_bytes, val_bytes, nsf):
    data_bps = key_bytes + val_bytes
    meta_off = BLOCK_SIZE * HK * data_bps
    blk, off = slot // BLOCK_SIZE, slot % BLOCK_SIZE
    data_base = blk * cache.stride(0) + off * (HK * data_bps) + head * data_bps
    hm = (blk * cache.stride(0) + meta_off) // 2 + head * (nsf * BLOCK_SIZE)
    return data_base, hm, off


def main():
    device = torch.device("xpu:0")
    key_fp8, mse_bits, vqb = True, 4, 4
    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    centroids = get_centroids(D, mse_bits).to(device=device, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    seq_len, q_len = 8, 4
    gen = torch.Generator(device="cpu").manual_seed(20261104)
    key = torch.randn(seq_len, HK, D, generator=gen).to(torch.bfloat16).to(device)
    val = torch.randn(seq_len, HK, D, generator=gen).to(torch.bfloat16).to(device)

    key_bytes, val_bytes, nsf = D, math.ceil(D * vqb / 8), 2
    padded = key_bytes + val_bytes + nsf * 2
    cache = torch.zeros(1, BLOCK_SIZE, HK, padded, dtype=torch.uint8, device=device)
    slot_map = torch.arange(seq_len, dtype=torch.int32, device=device)
    triton_turboquant_store(
        key, val, cache, slot_map, PiT, midpoints,
        mse_bits=mse_bits, key_packed_size=key_bytes,
        value_quant_bits=vqb, key_fp8=key_fp8,
        centroids=centroids, norm_correction=False,
    )
    q = torch.randn(q_len, HQ, D, generator=gen).to(torch.bfloat16).to(device)
    bt = torch.arange(1, dtype=torch.int32, device=device).view(1, -1)
    q0_t = torch.tensor([seq_len - (q_len - 1)], dtype=torch.int32, device=device)

    q_rot = q.contiguous()
    mid_o = torch.full((HQ, SPLITS, 4, D + 1), float("nan"), dtype=torch.float32, device=device)
    mq._tq_mq_soa_stage1[(1, HQ, SPLITS)](
        q_rot, cache, cache.view(torch.uint16), bt, q0_t, centroids, mid_o,
        q_rot.stride(0), q_rot.stride(1), cache.stride(0),
        mid_o.stride(0), mid_o.stride(1),
        NUM_KV_HEADS=HK, HEAD_DIM=D, BLOCK_SIZE=BLOCK_SIZE,
        NUM_KV_SPLITS=SPLITS, KV_GROUP_SIZE=HQ // HK,
        Q_LEN=q_len, Q_BLOCK=4, MSE_BITS=mse_bits, VQB=vqb,
        VAL_DATA_BYTES=val_bytes, KEY_DATA_BYTES=key_bytes,
        META_REGION_OFFSET=BLOCK_SIZE * HK * (key_bytes + val_bytes),
        NUM_SOA_FIELDS=nsf, SOA_K_NORM=0, SOA_V_SCALE=0, SOA_V_ZERO=1,
        ATTN_SCALE=SCALE, BLOCK_D=256, BLOCK_KV=4,
        KEY_FP8=1, FP8_E4B15=0,
        num_warps=1, num_stages=1,
    )
    torch.xpu.synchronize()

    flat = cache.view(-1)
    u16 = cache.view(torch.uint16).view(-1)
    print("token | lse_mine  lse_ref  | o_mine[0:3] vs v_hat[0:3]")
    for s in range(seq_len):
        # dequant K token s head 0 (fp8) and V (4-bit affine)
        db, hm, off = soa_addrs(cache, s, 0, key_bytes, val_bytes, nsf)
        k_hat = flat[db : db + D].view(torch.float8_e4m3fn).float()
        raw = flat[db + D : db + D + val_bytes].to(torch.int64)
        lo = raw & 0xF
        hi = raw >> 4
        vidx = torch.stack([lo, hi], dim=1).reshape(-1)
        vs = u16[hm + 0 * BLOCK_SIZE + off].view(torch.float16).float()
        vz = u16[hm + 1 * BLOCK_SIZE + off].view(torch.float16).float()
        v_hat = vidx * vs + vz

        lse_ref = ((q[3, 0].float() * k_hat).sum() * SCALE).item()
        o_mine = mid_o[0, s, 3, :D]
        d_o = (o_mine - v_hat).abs().max().item()
        lse_mine = mid_o[0, s, 3, D].item()
        print(f"  {s}   | {lse_mine:8.4f} {lse_ref:8.4f} | "
              f"maxdiff_o={d_o:.6f}  v_hat0={v_hat[0].item():.4f} "
              f"o0={o_mine[0].item():.4f}")


if __name__ == "__main__":
    main()
