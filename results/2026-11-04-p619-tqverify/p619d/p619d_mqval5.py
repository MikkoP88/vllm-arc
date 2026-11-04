#!/usr/bin/env python3
"""P619d MQVAL5 — ported MQ kernel vs the ENGINE's TQ ops pair (AoS).

The first MQ draft (tq_mq_soa) was validated against the FlyDSL-gated
SoA tree — self-consistent but the wrong world; in-engine it read the
AoS cache as zeros (t8/t8e/t8f). This gate validates tq_mq_aos against
the pair the engine actually runs on XPU (FlyDSL off):
  store : vllm.v1.attention.ops.triton_turboquant_store        (AoS)
  decode: vllm.v1.attention.ops.triton_turboquant_decode       (AoS)
Engine-exact geometry (t8f): BLOCK_SIZE=128, q fp16, seq_lens int32[38]
sliced at i=5, q0 = seq_lens[I:I+1] - 3 (recorded-subtract form), bt row
2052 wide with zero tail, Hq12/Hk2, D256. KPS per engine config.py:
fp8 -> head_dim; MSE -> ceil(D*bits/8)+2 (norm fp16 inline).
PASS: every preset, every row rel(mq, stock) <= 1e-4, det 2/2 identical.
"""
import math
import sys

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import (
    triton_turboquant_store,
)
from vllm_gaudi.xpu.patches.tq_mq_aos import (
    triton_turboquant_mq_decode_attention,
)

D = 256
HQ = 12
HK = 2
BS = 128
N_REQS = 38
I = 5
Q_LEN = 4
SEQ = 9
BT_W = 2052
SCALE = 1.0 / math.sqrt(D)

# (name, key_fp8, key_mse_bits, vqb, nc) — engine preset semantics
PRESETS = [
    ("turboquant_k8v4", True, 0, 4, False),
    ("turboquant_4bit_nc", False, 4, 4, True),
    ("turboquant_k3v4_nc", False, 3, 4, True),
    ("turboquant_3bit_nc", False, 3, 3, True),
]


def main():
    device = torch.device("xpu:0")
    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    gen = torch.Generator(device="cpu").manual_seed(20261104)
    key = torch.randn(SEQ, HK, D, generator=gen).to(torch.float16).to(device)
    val = torch.randn(SEQ, HK, D, generator=gen).to(torch.float16).to(device)
    q = torch.randn(Q_LEN, HQ, D, generator=gen).to(torch.float16).to(device)

    fails = 0
    for name, key_fp8, kmb, vqb, nc in PRESETS:
        kps = D if key_fp8 else math.ceil(D * kmb / 8) + 2
        val_bytes = math.ceil(D * vqb / 8)
        padded = kps + val_bytes + 4
        padded += padded % 2
        cbits = vqb if key_fp8 else kmb  # engine centroid_bits
        centroids = get_centroids(D, cbits).to(
            device=device, dtype=torch.float32)
        c_sorted, _ = centroids.sort()
        midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

        n_pages_total = 460
        pages = torch.arange(100, 100 + math.ceil(SEQ / BS), dtype=torch.int32)
        cache = torch.zeros(
            n_pages_total, BS, HK, padded, dtype=torch.uint8, device=device)
        tok_page = torch.repeat_interleave(pages, BS)[:SEQ]
        slot_map = (tok_page * BS + torch.arange(SEQ) % BS)
        slot_map = slot_map.to(torch.int32).to(device)
        triton_turboquant_store(
            key, val, cache, slot_map, PiT, midpoints,
            mse_bits=kmb, key_packed_size=kps,
            value_quant_bits=vqb, key_fp8=key_fp8,
        )
        torch.xpu.synchronize()

        seq_lens = torch.full((N_REQS,), 3, dtype=torch.int32, device=device)
        seq_lens[I] = SEQ
        bt = torch.zeros(N_REQS, BT_W, dtype=torch.int32, device=device)
        bt[I, :pages.numel()] = pages.to(device)
        q0 = seq_lens[I:I + 1] - (Q_LEN - 1)

        runs = []
        for _ in range(2):
            mq = triton_turboquant_mq_decode_attention(
                q, cache, bt[I:I + 1], q0, PiT, centroids, SCALE,
                mse_bits=kmb, key_packed_size=kps, value_quant_bits=vqb,
                key_fp8=key_fp8, norm_correction=nc, PiT=PiT,
            ).float()
            runs.append(mq)
        det = bool(torch.equal(runs[0], runs[1]))

        stock = torch.empty(Q_LEN, HQ, D, dtype=torch.float32, device=device)
        for r in range(Q_LEN):
            sl_r = seq_lens[I:I + 1] - (Q_LEN - 1 - r)
            stock[r] = triton_turboquant_decode_attention(
                q[r:r + 1], cache, bt[I:I + 1], sl_r, PiT, centroids, SCALE,
                mse_bits=kmb, key_packed_size=kps, value_quant_bits=vqb,
                key_fp8=key_fp8, norm_correction=nc, PiT=PiT,
            ).float()
        torch.xpu.synchronize()

        rels = [
            ((runs[0][r] - stock[r]).norm() / stock[r].norm()
             .clamp(min=1e-9)).item()
            for r in range(Q_LEN)
        ]
        ok = all(x <= 1e-4 for x in rels) and det
        fails += 0 if ok else 1
        print(f"{name:22s} slot={padded:3d} det={det} "
              f"max_rel={max(rels):.2e} {'PASS' if ok else 'FAIL'} "
              f"norms={[round(float(r0.norm()), 3) for r0 in runs[0]]}")

    print("MQVAL5:", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
