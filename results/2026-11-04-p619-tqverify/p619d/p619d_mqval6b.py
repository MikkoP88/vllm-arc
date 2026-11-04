#!/usr/bin/env python3
"""P619d MQVAL6b — why offline(dump)=44.00 while engine ref0=30.28?

Two suspects:
  1. The t8f dump's gather `kv_cache[cpu_long_pages]` (XPU advanced
     indexing with a CPU index tensor) did NOT copy the live bytes.
     -> test: known-pattern cache, gather the same way, compare.
  2. A decode-semantic delta: sweep {seq 6 vs 9} x {e4nv vs e4b15}
     x {nibble order} x {scale/zero swap} against engine ref0.
"""
import torch

from vllm.model_executor.layers.quantization.turboquant.centroids import (
    get_centroids,
)
from vllm.v1.attention.backends.turboquant_attn import _build_hadamard
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)

D = 256

device = torch.device("xpu:0")

# --- 1. XPU advanced-index fidelity ---------------------------------
torch.manual_seed(0)
big = torch.arange(100 * 128 * 2 * 8, dtype=torch.uint8, device=device)
big = big.view(100, 128, 2, 8)
pages_cpu = torch.tensor([3, 17, 5, 99], dtype=torch.long)  # CPU index
g = big[pages_cpu]                     # the instrument's gather form
g_ref = big[pages_cpu.to(device)]      # device-index reference
print("gather cpu-idx == dev-idx:", torch.equal(g.cpu(), g_ref.cpu()),
      "| spot:", big[3, 0, 0, :4].cpu().tolist(), g[0, 0, 0, :4].cpu().tolist())

# --- 2. semantics sweep on the real dump ----------------------------
d = torch.load("/root/p619f_dump.pt", map_location="cpu", weights_only=False)
tq = d["tq"]
q = d["q"].to(device)
cache = d["cache_pages"].to(device)
n_pages = cache.shape[0]
bt = torch.arange(n_pages, dtype=torch.int32, device=device).view(1, -1)
q0_val = int(d["q0"].reshape(-1)[0])
ref0 = d["ref0"].float()[0].to(device)
PiT = _build_hadamard(D, str(device)).to(torch.float32)
centroids = get_centroids(D, tq["vqb"]).to(device=device, dtype=torch.float32)

results = []
for sl_val in (q0_val, q0_val + 3):  # 6 (recorded) vs 9 (full seq)
    sl = torch.tensor([sl_val], dtype=torch.int32, device=device)
    out = triton_turboquant_decode_attention(
        q[0:1], cache, bt, sl, PiT, centroids, d["scale"],
        mse_bits=tq["mse_bits"], key_packed_size=tq["kps"],
        value_quant_bits=tq["vqb"], key_fp8=tq["key_fp8"],
        norm_correction=tq["nc"], PiT=PiT,
    ).float()[0]
    rel = float((out - ref0).norm() / ref0.norm())
    results.append((f"sl={sl_val} e4nv", float(out.norm()), rel))

# byte-level reinterpretations of the SAME dumped record (hand decode):
# slot0 h0: K fp8 e4nv vs e4b15 dot q0-row; V nibbles lo/hi vs hi/lo;
# scale/zero std vs swapped — quick torch emulation on ONE record set.
q0r = q[0].float()                      # [Hq, D]
import struct

bt0 = d["bt_row"][0]
cp = d["cache_pages"][0]                # the request's page (slots 0..8)
k_bytes = cp[:q0_val + 3, 0, :D]        # all 9 tokens, head 0
v_bytes = cp[:q0_val + 3, 0, D:D + 128]
meta = cp[:q0_val + 3, 0, D + 128:D + 132]


def fp8_e4nv(b):
    b = b.int()
    sign = (b & 0x80).float() * -2 + 1
    exp = ((b >> 4) & 0x7).float()
    man = (b & 0xF).float()
    val = torch.where(exp > 0, (2 ** (exp - 1)) * (1 + man / 16),
                      (man / 16) * 2 ** -2)
    return sign * val


def fp8_e4b15(b):
    b = b.int()
    sign = (b & 0x80).float() * -2 + 1
    biased = (b & 0x7F).float()
    return sign * 2.0 ** (biased - 15.0)


def u16le(lo, hi):
    return (lo.int() | (hi.int() << 8))


def fp16_bits(u16):
    return u16.to(torch.uint16).view(torch.float16).float()


def emu(tok, bc, nib_swap, sc_swap, e4b):
    kb = k_bytes[tok]
    k = fp8_e4b15(kb) if e4b else fp8_e4nv(kb)
    s = float((q0r[0] * k.to(q0r.device)).sum() * d["scale"])
    vb = v_bytes[tok]
    lo = (vb & 0xF).float()
    hi = (vb >> 4).float()
    idx = torch.stack([hi, lo], dim=-1).reshape(-1) if nib_swap \
        else torch.stack([lo, hi], dim=-1).reshape(-1)
    sc = fp16_bits(u16le(meta[tok, 0], meta[tok, 1]))
    zz = fp16_bits(u16le(meta[tok, 2], meta[tok, 3]))
    a, b_ = (zz, sc) if sc_swap else (sc, zz)
    return s, idx * a + b_


for e4b in (False, True):
    for nib in (False, True):
        for sw in (False, True):
            sv = [emu(t, 0, nib, sw, e4b) for t in range(6)]
            m = max(s for s, _ in sv)
            acc = torch.zeros(D)
            Z = 0.0
            for s, v in sv:
                e = 2.718281828 ** (s - m)
                acc += e * v
                Z += e
            out = acc / Z
            rel = float((out - ref0[0].cpu()).norm()
                        / ref0[0].norm())
            results.append((f"emu e4b15={e4b} nib_swap={nib} sc_swap={sw}",
                            float(out.norm()), rel))

for name, n, rel in results:
    print(f"{name:38s} norm={n:9.4f} rel_vs_ref0={rel:.6f}")
print(f"engine ref0 norm: {float(ref0.norm()):.4f}")
