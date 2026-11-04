#!/usr/bin/env python3
"""P6.19d MQ DIAG 5 — permutation search. Single-token splits: o_mine[s]
must EQUAL v_hat[t, h] exactly for some (t, h); lse_mine[s] must EQUAL
(q[r, hq] . k_hat[t, h]) * SCALE for some (r, hq, t, h). Search the full
grids; report exact matches. k8v4 q4 seq8 head0 dump.
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
    mse_bits, vqb = 4, 4
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
        value_quant_bits=vqb, key_fp8=True,
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
    k_hat = torch.empty(seq_len, HK, D, dtype=torch.float32, device=device)
    v_hat = torch.empty(seq_len, HK, D, dtype=torch.float32, device=device)
    for t in range(seq_len):
        for h in range(HK):
            db, hm, off = soa_addrs(cache, t, h, key_bytes, val_bytes, nsf)
            k_hat[t, h] = flat[db : db + D].view(torch.float8_e4m3fn).float()
            raw = flat[db + D : db + D + val_bytes].to(torch.int64)
            vidx = torch.stack([raw & 0xF, raw >> 4], dim=1).reshape(-1)
            vs = u16[hm + 0 * BLOCK_SIZE + off].view(torch.float16).float()
            vz = u16[hm + 1 * BLOCK_SIZE + off].view(torch.float16).float()
            v_hat[t, h] = vidx * vs + vz

    print("split | o_match (exact)                 | lse_match (tol 0.01)")
    for s in range(seq_len):
        o_mine = mid_o[0, s, 3, :D]
        lse_mine = mid_o[0, s, 3, D].item()
        o_hit = "NONE"
        for t in range(seq_len):
            for h in range(HK):
                if torch.equal(o_mine, v_hat[t, h]):
                    o_hit = f"v_hat[t={t},h={h}]"
                    break
            if o_hit != "NONE":
                break
        lse_hit = "NONE"
        for r in range(q_len):
            for hq in range(HQ):
                dots = (k_hat.float() * q[r, hq].float()[None, None, :]).sum(-1) * SCALE
                for t in range(seq_len):
                    for h in range(HK):
                        if abs(dots[t, h].item() - lse_mine) < 0.01:
                            lse_hit = f"q[r={r},hq={hq}] . k[t={t},h={h}] = {dots[t, h].item():.4f}"
                            break
                    if lse_hit != "NONE":
                        break
                if lse_hit != "NONE":
                    break
            if lse_hit != "NONE":
                break
        print(f"  {s}   | {o_hit:32s} | {lse_hit}")


if __name__ == "__main__":
    main()
