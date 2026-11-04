#!/usr/bin/env python3
"""P619d MQVAL3c — swap matrix on the t8f dump.

Facts so far: engine MQ = 0, offline MQ = 0, offline stock = 0 on the
dumped bytes; the SAME wrappers on a fresh-stored cache (MQVAL4) are
correct. Swap one input at a time to find which dumped input carries
the zero:
  A. stock, kwargs byte-identical to v5i's _aosa_decode call
  B. MQ baseline on the dump (expect 0)
  C. KILLER: store fresh random K/V into an EMPTY page of the dumped
     cache tensor itself (page 1), bt row -> page 1, same wrapper path.
     Nonzero => wrapper+kernel fine; the dumped page-0 BYTES are
     semantically different from stock-store output (engine store
     writes something my decode pair doesn't understand).
     Zero     => wrapper broken on this tensor/call shape.
  D. dumped cache page 0, random q (rule q values out).
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
from vllm_gaudi.xpu.patches.tq_mq_soa import (
    triton_turboquant_mq_decode_attention_soa,
)
from vllm.v1.attention.ops.turboquant_soa.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)

D = 256
SEQ = 9
Q_LEN = 4
SCALE = 1.0 / math.sqrt(D)

d = torch.load("/root/p619f_dump.pt", map_location="cpu", weights_only=False)
device = torch.device("xpu:0")
tq = d["tq"]

q = d["q"].to(device)
cache = d["cache_pages"].to(device)          # (2052,128,2,388) uint8
n_pages = cache.shape[0]
q0 = d["q0"].to(device).to(torch.int32)
q0_val = int(q0.reshape(-1)[0])

PiT = _build_hadamard(D, str(device)).to(torch.float32)
centroids = get_centroids(D, tq["mse_bits"]).to(
    device=device, dtype=torch.float32)
c_sorted, _ = centroids.sort()
midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2

KW = dict(mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
          value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
          norm_correction=tq["nc"], PiT=PiT)

def mq(bt_row, qq, qq0, cc=None):
    return triton_turboquant_mq_decode_attention_soa(
        qq, cc if cc is not None else cache, bt_row, qq0, PiT, centroids,
        d["scale"], **KW).float()

def stock0(bt_row, qq, sl, cc=None):
    return triton_turboquant_decode_attention(
        qq[0:1], cc if cc is not None else cache, bt_row, sl, PiT,
        centroids, d["scale"], **KW).float()

def norms(t):
    return [round(float(r.norm()), 4) for r in t]

# --- A. stock, v5i-exact kwargs, dump page 0 -------------------------
bt_ar = torch.arange(n_pages, dtype=torch.int32, device=device).view(1, -1)
sl0 = q0.clone()
a = stock0(bt_ar, q, sl0)
print("A stock dump-p0 norms:", norms(a))

# --- B. MQ baseline on dump ------------------------------------------
b = mq(bt_ar, q, q0)
print("B mq    dump-p0 norms:", norms(b))

# --- C. KILLER: fresh store into the dumped tensor's empty page 1 ----
gen = torch.Generator(device="cpu").manual_seed(20261104)
key = torch.randn(SEQ, 2, D, generator=gen).to(torch.float16).to(device)
val = torch.randn(SEQ, 2, D, generator=gen).to(torch.float16).to(device)
slot_map = (1 * 128 + torch.arange(SEQ)).to(torch.int32).to(device)  # page 1 slots 0..8
triton_turboquant_store(
    key, val, cache, slot_map, PiT, midpoints,
    mse_bits=4, key_packed_size=D,
    value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
    centroids=centroids, norm_correction=tq["nc"],
)
torch.xpu.synchronize()
nz = int((cache[1] != 0).sum())
print("C stored page1 nonzero:", nz)
bt_c = torch.zeros(1, n_pages, dtype=torch.int32, device=device)
bt_c[0, 0] = 1
c = mq(bt_c, q, q0)
print("C mq    fresh-p1 norms:", norms(c))
c2 = stock0(bt_c, q, sl0)
print("C stock fresh-p1 norms:", norms(c2))

# --- D. dump page 0, random q ----------------------------------------
qr = torch.randn(Q_LEN, 12, D, generator=gen).to(torch.float16).to(device)
dd = mq(bt_ar, qr, q0)
print("D mq    dump-p0 randq norms:", norms(dd))
print("D stock dump-p0 randq  norm:", norms(stock0(bt_ar, qr, sl0)))
