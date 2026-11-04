#!/usr/bin/env python3
"""P619d MQVAL6 — THE decisive replay: the ported AoS MQ kernel on the
actual t8f ENGINE dump (the exact bytes the SoA-pair kernels decoded to
exact zero). PASS requires:
  - offline AoS stock row0 == engine ref0 (replay end-to-end valid)
  - ported MQ row0 ≈ engine ref0 (rel <= 1e-4) and all rows nonzero
"""
import sys

import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm_gaudi.xpu.patches.tq_mq_aos import (
    triton_turboquant_mq_decode_attention,
)

D = 256

d = torch.load("/root/p619f_dump.pt", map_location="cpu", weights_only=False)
device = torch.device("xpu:0")
tq = d["tq"]

q = d["q"].to(device)
cache = d["cache_pages"].to(device)
n_pages = cache.shape[0]
bt = torch.arange(n_pages, dtype=torch.int32, device=device).view(1, -1)
q0 = d["q0"].to(device).to(torch.int32)
q0_val = int(q0.reshape(-1)[0])
q_len = q.shape[0]

# engine parity: fp8-key centroids use value bits (config.centroid_bits)
PiT = _build_hadamard(D, str(device)).to(torch.float32)
centroids = get_centroids(D, tq["vqb"]).to(device=device, dtype=torch.float32)

ref0 = d["ref0"].float()[0].to(device)
ref0_n = float(ref0.norm())

mq = triton_turboquant_mq_decode_attention(
    q, cache, bt, q0, PiT, centroids, d["scale"],
    mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
    value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
    norm_correction=tq["nc"], PiT=PiT,
).float()
mq2 = triton_turboquant_mq_decode_attention(
    q, cache, bt, q0, PiT, centroids, d["scale"],
    mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
    value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
    norm_correction=tq["nc"], PiT=PiT,
).float()

st0 = triton_turboquant_decode_attention(
    q[0:1], cache, bt, q0.clone(), PiT, centroids, d["scale"],
    mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
    value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
    norm_correction=tq["nc"], PiT=PiT,
).float()[0]
torch.xpu.synchronize()

mq_n = [round(float(r.norm()), 4) for r in mq]
rel_mq0 = float((mq[0] - ref0).norm() / ref0.norm().clamp(min=1e-9))
rel_st0 = float((st0 - ref0).norm() / ref0.norm().clamp(min=1e-9))
det = bool(torch.equal(mq, mq2))
nonzero = all(float(r.norm()) > 1.0 for r in mq)

print(f"engine ref0 norm    : {ref0_n:.4f}")
print(f"offline AoS stock0  : norm={float(st0.norm()):.4f} "
      f"rel_vs_engine={rel_st0:.6f}")
print(f"ported MQ rows      : norms={mq_n} det={det}")
print(f"MQ row0 vs engine   : rel={rel_mq0:.6f}")
ok = (rel_st0 <= 1e-5 and rel_mq0 <= 1e-4 and det and nonzero
      and float(st0.norm()) > 1.0)
print("MQVAL6:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
