#!/usr/bin/env python3
"""P619d MQVAL3b — same replay as MQVAL3 but PRINTS norms + raw bytes so
the zero can be located instead of inferred."""
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm_gaudi.xpu.patches.tq_mq_soa import (
    triton_turboquant_mq_decode_attention_soa,
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
q_len = q.shape[0]
q0_val = int(q0.reshape(-1)[0])

print("gathered[0] slot0 head0 bytes[:16]:",
      d["cache_pages"][0, 0, 0, :16].tolist())
print("gathered[0] slot0 head1 bytes[:8] :",
      d["cache_pages"][0, 0, 1, :8].tolist())
print("gathered[0] meta u16 tail (v_scale,v_zero) slot0 h0:",
      d["cache_pages"][0, 0, 0, 384:].tolist())

PiT = _build_hadamard(D, str(device)).to(torch.float32)
centroids = get_centroids(D, tq["mse_bits"]).to(
    device=device, dtype=torch.float32)

mq = triton_turboquant_mq_decode_attention_soa(
    q, cache, bt, q0, PiT, centroids, d["scale"],
    mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
    value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
    norm_correction=tq["nc"], PiT=PiT,
).float()
print("offline mq norms:", [round(float(r.norm()), 4) for r in mq])

sl0 = torch.tensor([q0_val], dtype=torch.int32, device=device)
st0 = triton_turboquant_decode_attention(
    q[0:1], cache, bt, sl0, PiT, centroids, d["scale"],
    mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
    value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
    norm_correction=tq["nc"], PiT=PiT,
).float()
print("offline stock0 norm:", round(float(st0[0].norm()), 4))
print("eng ref0 norm     :", round(float(d["ref0"].float()[0].norm()), 4))
if float(st0[0].norm()) > 0 and float(d["ref0"].float()[0].norm()) > 0:
    rel = ((st0[0] - d["ref0"].float()[0].to(device)).norm()
           / d["ref0"].float()[0].to(device).norm()).item()
    print("offline-stock0 vs eng-ref0 rel:", round(rel, 6))
