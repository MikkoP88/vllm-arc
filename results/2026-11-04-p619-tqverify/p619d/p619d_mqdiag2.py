#!/usr/bin/env python3
"""P6.19d MQ DIAG 2 — splits=1 clean, splits=32 wrong on all rows => defect
is in cross-split partials or the combine. This leg launches STAGE 1 ALONE
into a manual mid_o, then:
  A) combines the same mid_o in torch (exact stage-2 math)   -> if CORRECT,
     the stage-2 KERNEL is the bug;
  B) launches the stage-2 kernel on the same mid_o            -> compare;
  C) dumps the lse column for head 0, all splits, row 3 (full-attention
     row) vs a torch per-split reference  -> which partials are wrong.
Case: k8v4 q4 seq8 (splits=32, split_len=1, empties 8..31).
"""

import math

import torch
import triton

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
from vllm_gaudi.xpu.patches import tq_mq_soa as mq

D = 256
HQ = 12
HK = 2
BLOCK_SIZE = 64
SCALE = 1.0 / math.sqrt(D)
SPLITS = 32


def main():
    device = torch.device("xpu:0")
    print(f"device: {torch.xpu.get_device_name(0)}")
    key_fp8, mse_bits, vqb = True, 4, 4
    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    centroids = get_centroids(D, mse_bits).to(device=device, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    seq_len, q_len = 8, 4
    gen = torch.Generator(device="cpu").manual_seed(20261104)
    key = torch.randn(seq_len, HK, D, generator=gen).to(torch.bfloat16).to(device)
    val = torch.randn(seq_len, HK, D, generator=gen).to(torch.bfloat16).to(device)

    key_bytes = D
    val_bytes = math.ceil(D * vqb / 8)
    nsf = 2
    padded = key_bytes + val_bytes + nsf * 2
    n_blocks = 1
    cache = torch.zeros(n_blocks, BLOCK_SIZE, HK, padded, dtype=torch.uint8, device=device)
    slot_map = torch.arange(seq_len, dtype=torch.int32, device=device)
    triton_turboquant_store(
        key, val, cache, slot_map, PiT, midpoints,
        mse_bits=mse_bits, key_packed_size=key_bytes,
        value_quant_bits=vqb, key_fp8=key_fp8,
        centroids=centroids, norm_correction=False,
    )
    q = torch.randn(q_len, HQ, D, generator=gen).to(torch.bfloat16).to(device)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=device).view(1, -1)
    q0 = seq_len - (q_len - 1)
    q0_t = torch.tensor([q0], dtype=torch.int32, device=device)

    # split reference (proven clean)
    sp = torch.empty(q_len, HQ, D, dtype=torch.bfloat16, device=device)
    for r in range(q_len):
        out_r = triton_turboquant_decode_attention(
            q[r : r + 1], cache, bt,
            torch.tensor([q0 + r], dtype=torch.int32, device=device),
            PiT, centroids, SCALE,
            mse_bits=mse_bits, key_packed_size=key_bytes,
            value_quant_bits=vqb, key_fp8=key_fp8,
            norm_correction=False, PiT=PiT,
        )
        sp[r : r + 1] = out_r.to(torch.bfloat16)

    # manual stage1 launch (wrapper parity)
    q_rot = q.contiguous()
    Q_BLOCK = 4
    mid_o = torch.empty(HQ, SPLITS, Q_BLOCK, D + 1, dtype=torch.float32, device=device)
    output = torch.empty(Q_BLOCK, HQ, D, dtype=q.dtype, device=device)
    grid = (1, HQ, SPLITS)
    mq._tq_mq_soa_stage1[grid](
        q_rot, cache, cache.view(torch.uint16), bt, q0_t, centroids, mid_o,
        q_rot.stride(0), q_rot.stride(1), cache.stride(0),
        mid_o.stride(0), mid_o.stride(1),
        NUM_KV_HEADS=HK, HEAD_DIM=D, BLOCK_SIZE=BLOCK_SIZE,
        NUM_KV_SPLITS=SPLITS, KV_GROUP_SIZE=HQ // HK,
        Q_LEN=q_len, Q_BLOCK=Q_BLOCK, MSE_BITS=mse_bits, VQB=vqb,
        VAL_DATA_BYTES=val_bytes, KEY_DATA_BYTES=key_bytes,
        META_REGION_OFFSET=BLOCK_SIZE * HK * (key_bytes + val_bytes),
        NUM_SOA_FIELDS=nsf, SOA_K_NORM=0, SOA_V_SCALE=0, SOA_V_ZERO=1,
        ATTN_SCALE=SCALE, BLOCK_D=256, BLOCK_KV=4,
        KEY_FP8=1, FP8_E4B15=0,
        num_warps=1, num_stages=1,
    )
    mq._tq_mq_soa_stage2[(1, HQ)](
        mid_o, output,
        mid_o.stride(0), mid_o.stride(1),
        output.stride(0), output.stride(1),
        Q_LEN=q_len, Q_BLOCK=Q_BLOCK, NUM_KV_SPLITS=SPLITS,
        BLOCK_DV=256, Lv=D,
        num_warps=4, num_stages=2,
    )
    torch.xpu.synchronize()

    # A) torch combine of the SAME mid_o (exact stage-2 math)
    lse = mid_o[:, :, :, D].clone()          # [Hq, splits, Q_BLOCK]
    ov = mid_o[:, :, :, :D].clone()          # [Hq, splits, Q_BLOCK, D]
    M = lse.max(dim=1, keepdim=True).values  # [Hq, 1, Q_BLOCK]
    w = torch.exp(lse - M)
    w = torch.where(torch.isinf(lse), torch.zeros_like(w), w)
    tc = (w.unsqueeze(-1) * ov).sum(1) / w.sum(1).unsqueeze(-1).clamp(min=1e-30)
    # -> [Hq, Q_BLOCK, D]

    print("\n-- per-row max-abs diffs vs split ref --")
    for r in range(q_len):
        d_s2 = (output[r].float() - sp[r].float()).abs().max().item()
        d_tc = (tc[:, r].float() - sp[r].float().abs().max().item() * 0).abs()
        d_tc = (tc[:, r] - sp[r].float()).abs().max().item()
        print(f"  row{r}: stage2_kernel={d_s2:.6f} torch_combine={d_tc:.6f}")

    print("\n-- lse dump head0, row3 (attends all 8 tokens) --")
    l0 = mid_o[0, :, 3, D].tolist()
    print("  mine:", [f"{x:.3f}" for x in l0])

    # torch per-split reference lse for head0, row3 (q head 0 -> kv head 0)
    hq = 0
    scores_all = (q[3, hq].float()[None, :] * key[:seq_len, 0].float()).sum(-1) * SCALE
    per = []
    split_len = -(-seq_len // SPLITS)
    for s in range(SPLITS):
        lo, hi = split_len * s, min(split_len * (s + 1), seq_len)
        if lo >= hi:
            per.append(float("-inf"))
            continue
        sc = scores_all[lo:hi]
        m = sc.max()
        l = torch.exp(sc - m).sum()
        per.append((m + torch.log(l)).item())
    print("  ref :", [f"{x:.3f}" for x in per])

    print("\n-- o dump head0, split0, row3 first6 --")
    print("  mine:", mid_o[0, 0, 3, :6].tolist())


if __name__ == "__main__":
    main()
