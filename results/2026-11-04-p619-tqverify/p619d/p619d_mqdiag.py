#!/usr/bin/env python3
"""P6.19d MQ kernel DIAG — bisect the offline-gate failure (all rel~0.15-0.5,
rel_ref==rel_split => MQ alone is wrong). Prints per-row MQ-vs-split diffs at
splits=32 and splits=1 (splits=1 removes the multi-partial combine path;
if splits=1 is clean the defect is in the split/combine structure, else in
the core loop). Preset k8v4 (fp8 key, simplest path) + k3v4_nc (MSE path).
"""

import math

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
SCALE = 1.0 / math.sqrt(D)


def run(name, cfg, device, q_len, seq_len):
    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    centroids = get_centroids(D, cfg["mse_bits"]).to(device=device, dtype=torch.float32)
    c_sorted, _ = centroids.sort()
    midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

    gen = torch.Generator(device="cpu").manual_seed(20261104)
    key = torch.randn(seq_len, HK, D, generator=gen).to(torch.bfloat16).to(device)
    val = torch.randn(seq_len, HK, D, generator=gen).to(torch.bfloat16).to(device)

    key_bytes = D if cfg["key_fp8"] else math.ceil(D * cfg["mse_bits"] / 8)
    val_bytes = math.ceil(D * cfg["vqb"] / 8)
    nsf = 2 if cfg["key_fp8"] else 3
    padded = key_bytes + val_bytes + nsf * 2
    n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    cache = torch.zeros(n_blocks, BLOCK_SIZE, HK, padded, dtype=torch.uint8, device=device)
    slot_map = torch.arange(seq_len, dtype=torch.int32, device=device)
    triton_turboquant_store(
        key, val, cache, slot_map, PiT, midpoints,
        mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
        value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
        centroids=centroids, norm_correction=cfg["nc"],
    )
    q = torch.randn(q_len, HQ, D, generator=gen).to(torch.bfloat16).to(device)
    bt = torch.arange(n_blocks, dtype=torch.int32, device=device).view(1, -1)
    q0 = seq_len - (q_len - 1)

    def split_ref():
        out = torch.empty(q_len, HQ, D, dtype=torch.bfloat16, device=device)
        for r in range(q_len):
            out_r = triton_turboquant_decode_attention(
                q[r : r + 1], cache, bt,
                torch.tensor([q0 + r], dtype=torch.int32, device=device),
                PiT, centroids, SCALE,
                mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
                value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
                norm_correction=cfg["nc"], PiT=PiT,
            )
            out[r : r + 1] = out_r.to(torch.bfloat16)
        return out

    sp = split_ref()
    print(f"\n=== {name} q{q_len} seq{seq_len} ===", flush=True)
    for splits in (32, 1):
        mq = triton_turboquant_mq_decode_attention_soa(
            q, cache, bt, torch.tensor([q0], dtype=torch.int32, device=device),
            PiT, centroids, SCALE,
            mse_bits=cfg["mse_bits"], key_packed_size=key_bytes,
            value_quant_bits=cfg["vqb"], key_fp8=cfg["key_fp8"],
            norm_correction=cfg["nc"], PiT=PiT,
            max_num_kv_splits=splits,
        )
        torch.xpu.synchronize()
        for r in range(q_len):
            d = (mq[r].float() - sp[r].float()).abs().max().item()
            m = sp[r].float().abs().max().item()
            print(f"  splits={splits:2d} row{r}: max_abs_diff={d:.6f} "
                  f"scale={m:.4f} rel={d / max(m, 1e-6):.6f}")


def main():
    device = torch.device("xpu:0")
    print(f"device: {torch.xpu.get_device_name(0)}")
    run("k8v4", dict(key_fp8=True, mse_bits=4, vqb=4, nc=False), device, 4, 8)
    run("k8v4", dict(key_fp8=True, mse_bits=4, vqb=4, nc=False), device, 4, 192)
    run("k3v4_nc", dict(key_fp8=False, mse_bits=3, vqb=4, nc=True), device, 4, 8)


if __name__ == "__main__":
    main()
