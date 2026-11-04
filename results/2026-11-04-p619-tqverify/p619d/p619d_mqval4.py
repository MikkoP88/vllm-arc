#!/usr/bin/env python3
"""P619d MQVAL4 — engine-exact geometry reproduction (BLOCK_SIZE=128).

t8f facts the prior offline gates never modeled:
  block_size 128 (not 64), q fp16, seq_lens int32[38] sliced at i,
  q0 = seq_lens[i:i+1] - 3 (recorded subtract), bt row 2052 wide with
  pages [351, 352, ...] and ZERO-FILLED tail, k8v4, Hq12/Hk2, D256.
If MQ (or split) breaks HERE, the bug reproduces offline -> bisect.
If both are correct, the engine zero (mq_out=[0,0,0,0] in t8f) is
engine-context and the next step is an in-engine mid_o/lse probe.
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
BS = 128          # ENGINE block size (t8f: cache (4563, 128, 2, 388))
N_REQS = 38       # engine batch width (seq_lens[38])
I = 5             # our request's index in the batch
Q_LEN = 4
SEQ = 9           # seq_lens[I]
BT_W = 2052       # engine bt row width
SCALE = 1.0 / math.sqrt(D)


def main():
    device = torch.device("xpu:0")
    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    centroids = get_centroids(D, 4).to(device=device, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    gen = torch.Generator(device="cpu").manual_seed(20261104)
    key = torch.randn(SEQ, HK, D, generator=gen).to(torch.float16).to(device)
    val = torch.randn(SEQ, HK, D, generator=gen).to(torch.float16).to(device)

    key_bytes, val_bytes, nsf = D, math.ceil(D * 4 / 8), 2
    padded = key_bytes + val_bytes + nsf * 2
    n_pages_total = 460
    pages = torch.arange(100, 100 + math.ceil(SEQ / BS), dtype=torch.int32)
    cache = torch.zeros(n_pages_total, BS, HK, padded, dtype=torch.uint8,
                        device=device)
    tok_page = torch.repeat_interleave(pages, BS)[:SEQ]
    slot_map = (tok_page * BS + torch.arange(SEQ) % BS).to(torch.int32).to(device)
    triton_turboquant_store(
        key, val, cache, slot_map, PiT, midpoints,
        mse_bits=4, key_packed_size=key_bytes,
        value_quant_bits=4, key_fp8=True,
        centroids=centroids, norm_correction=False,
    )
    torch.xpu.synchronize()

    q = torch.randn(Q_LEN, HQ, D, generator=gen).to(torch.float16).to(device)

    # engine-shaped metadata: 38-request seq_lens, i=5; 2052-wide bt rows
    seq_lens = torch.full((N_REQS,), 3, dtype=torch.int32, device=device)
    seq_lens[I] = SEQ
    bt = torch.zeros(N_REQS, BT_W, dtype=torch.int32, device=device)
    bt[I, :pages.numel()] = pages.to(device)

    q0 = seq_lens[I:I + 1] - (Q_LEN - 1)   # the exact recorded-subtract form
    print("q0 =", q0.tolist())

    mq = triton_turboquant_mq_decode_attention_soa(
        q, cache, bt[I:I + 1], q0, PiT, centroids, SCALE,
        mse_bits=0, key_packed_size=256, value_quant_bits=4,
        key_fp8=True, norm_correction=False, PiT=PiT,
    ).float()

    stock = torch.empty(Q_LEN, HQ, D, dtype=torch.float32, device=device)
    for r in range(Q_LEN):
        sl_r = seq_lens[I:I + 1] - (Q_LEN - 1 - r)
        stock[r] = triton_turboquant_decode_attention(
            q[r:r + 1], cache, bt[I:I + 1], sl_r, PiT, centroids, SCALE,
            mse_bits=0, key_packed_size=256, value_quant_bits=4,
            key_fp8=True, norm_correction=False, PiT=PiT,
        ).float()
    torch.xpu.synchronize()

    print("mq norms   :", [round(float(r.norm()), 4) for r in mq])
    print("stock norms:", [round(float(r.norm()), 4) for r in stock])
    for r in range(Q_LEN):
        rel = ((mq[r] - stock[r]).norm() / stock[r].norm().clamp(min=1e-9)).item()
        print(f"row {r}: rel={rel:.6f}")


if __name__ == "__main__":
    main()
