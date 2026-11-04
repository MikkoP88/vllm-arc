#!/usr/bin/env python3
"""P6.19d MQ DIAG 4 — reverse-engineer what each split actually computed.
For split s (single token by construction) the kernel produced lse_mine[s]
and o_mine[s]. Search over: token windows [a,b), query rows 0-3, heads
(kv heads via q heads 0/2? head0 only), score scale on/off — find the
combination whose lse matches lse_mine[s]. k8v4 q4 seq8, head0.
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

    # dequant k_hat/v_hat head0 via stock-formula unpack (proven)
    flat = cache.view(-1)
    u16 = cache.view(torch.uint16).view(-1)
    k_hat = torch.empty(seq_len, D, dtype=torch.float32, device=device)
    v_hat = torch.empty(seq_len, D, dtype=torch.float32, device=device)
    for s in range(seq_len):
        db, hm, off = soa_addrs(cache, s, 0, key_bytes, val_bytes, nsf)
        k_hat[s] = flat[db : db + D].view(torch.float8_e4m3fn).float()
        raw = flat[db + D : db + D + val_bytes].to(torch.int64)
        vidx = torch.stack([raw & 0xF, raw >> 4], dim=1).reshape(-1)
        vs = u16[hm + 0 * BLOCK_SIZE + off].view(torch.float16).float()
        vz = u16[hm + 1 * BLOCK_SIZE + off].view(torch.float16).float()
        v_hat[s] = vidx * vs + vz

    # candidate scores per (q_row, kv_head): rows 0-3, kv heads 0-1 (q head
    # 0 maps kv 0, but test both in case hid mapping is off)
    cands = {}
    for qr in range(q_len):
        for kvh in range(HK):
            sc = (k_hat.float() * q[qr, 0].float()[None, :]).sum(-1) * SCALE
            cands[(qr, kvh)] = sc  # [seq_len]
    # also: scores with NO scale, and DOUBLE scale, kv head 0, row 3
    base = (k_hat.float() * q[3, 0].float()[None, :]).sum(-1)
    cands[("r3 noscale", 0)] = base
    cands[("r3 2scale", 0)] = base * (2 * SCALE)

    print("split | lse_mine  | best match (tol 0.02)")
    for s in range(seq_len):
        lse_mine = mid_o[0, s, 3, D].item()
        o_mine = mid_o[0, s, 3, :D]
        hit = "NONE"
        for (tag, _), sc in cands.items():
            for a in range(seq_len):
                for b in range(a + 1, seq_len + 1):
                    m = sc[a:b].max()
                    l = torch.exp(sc[a:b] - m).sum()
                    lse = (m + torch.log(l)).item()
                    if abs(lse - lse_mine) < 0.02:
                        o_ref = (
                            torch.softmax(sc[a:b], dim=-1)[:, None] * v_hat[a:b]
                        ).sum(0)
                        d = (o_ref - o_mine).abs().max().item()
                        hit = f"{tag} window[{a},{b}) lse={lse:.4f} o_diff={d:.5f}"
                        break
                if hit != "NONE":
                    break
            if hit != "NONE":
                break
        print(f"  {s}   | {lse_mine:8.4f} | {hit}")


if __name__ == "__main__":
    main()
