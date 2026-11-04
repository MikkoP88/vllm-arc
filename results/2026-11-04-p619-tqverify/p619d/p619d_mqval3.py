#!/usr/bin/env python3
"""P619d MQVAL3 — offline replay of the t8f IN-ENGINE dump.

t8f dumped the exact MQ-call inputs (q rows, q0, block-table row, the
referenced cache pages, the in-engine MQ output and the in-engine stock
row-0 reference). This script re-runs BOTH kernels offline on those
tensors and answers two questions at once:

  A. offline-MQ vs offline-STOCK (per row)   -> kernel math on engine
     inputs: a real bug reproduces here, bisectable with the diag
     ladder (stage-1 partials vs analytic per split).
  B. offline-MQ vs in-engine MQ output       -> engine-context drift
     (timing/stream): differs only if something outside the kernels
     changed the result.

The gathered ``cache_pages[k]`` equals the original cache at block id
``bt_row[0, k]``, so an arange block table over the gathered cache
addresses identical physical slots.
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
from vllm_gaudi.xpu.patches.tq_mq_soa import (
    triton_turboquant_mq_decode_attention_soa,
)

D = 256


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/p619f_dump.pt"
    d = torch.load(path, map_location="cpu", weights_only=False)
    device = torch.device("xpu:0")
    tq = d["tq"]
    print(f"dump: q {tuple(d['q'].shape)} {d['q_dtype']} | q0 {d['q0'].tolist()}"
          f" | bt {tuple(d['bt_row'].shape)} {d['bt_row'].dtype}"
          f" | pages {tuple(d['cache_pages'].shape)}"
          f" | cache_shape {d['cache_shape']} | tq={tq} scale={d['scale']}")
    print(f"seq_lens {tuple(d['seq_lens'].shape)} {d['seq_lens'].dtype} "
          f"head={d['seq_lens'][:8].tolist()}")

    q = d["q"].to(device)
    cache = d["cache_pages"].to(device)
    n_pages = cache.shape[0]
    bt = torch.arange(n_pages, dtype=torch.int32, device=device).view(1, -1)
    q0 = d["q0"].to(device).to(torch.int32)
    q_len = q.shape[0]
    q0_val = int(q0.reshape(-1)[0])

    PiT = _build_hadamard(D, str(device)).to(torch.float32)
    centroids = get_centroids(D, tq["mse_bits"]).to(
        device=device, dtype=torch.float32)

    mq = triton_turboquant_mq_decode_attention_soa(
        q, cache, bt, q0, PiT, centroids, d["scale"],
        mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
        value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
        norm_correction=tq["nc"], PiT=PiT,
    ).float()

    stock = torch.empty(q_len, *q.shape[1:], dtype=torch.float32, device=device)
    for r in range(q_len):
        sl_r = torch.tensor([q0_val + r], dtype=torch.int32, device=device)
        stock[r] = triton_turboquant_decode_attention(
            q[r:r + 1], cache, bt, sl_r, PiT, centroids, d["scale"],
            mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
            value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
            norm_correction=tq["nc"], PiT=PiT,
        ).float()

    eng_mq = d["mq_out"].to(device).float()
    eng_ref0 = d["ref0"].to(device).float()

    print("\nrow | mq-vs-stock | mq-vs-engineMQ | stock-vs-engineRef0")
    for r in range(q_len):
        a = ((mq[r] - stock[r]).norm() / stock[r].norm().clamp(min=1e-9)).item()
        b = ((mq[r] - eng_mq[r]).norm() / eng_mq[r].norm().clamp(min=1e-9)).item()
        print(f"  {r} | {a:.6f}     | {b:.6f}        | -")
    s0 = ((stock[0] - eng_ref0[0]).norm()
          / eng_ref0[0].norm().clamp(min=1e-9)).item()
    e_mq_ref = ((eng_mq[0] - eng_ref0[0]).norm()
                / eng_ref0[0].norm().clamp(min=1e-9)).item()
    print(f"\nstock0-vs-engineRef0 rel = {s0:.6f}  (engine A/B sanity)")
    print(f"engineMQ0-vs-engineRef0 rel = {e_mq_ref:.6f}  (the in-engine diff)")
    print(f"offlineMQ0-vs-stock0 rel = "
          f"{((mq[0] - stock[0]).norm() / stock[0].norm().clamp(min=1e-9)).item():.6f}")


if __name__ == "__main__":
    main()
