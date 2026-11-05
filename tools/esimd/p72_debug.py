#!/usr/bin/env python3
"""p72 debug: three-way compare — pure-torch fp32 reference vs stock
_vllm_fa2_C.varlen_fwd vs the ESIMD kernel — on minimal shapes, to
localize the probe-1 garbage (fro 7e4). No gates here, just truth."""
import os
import sys

import torch

try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401
except ImportError:
    from vllm import _xpu_ops  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fa_dense_ops  # noqa: E402


def build(bs, q_lens, kv_lens, q_heads, kv_heads, page, seed, descale=False):
    g = torch.Generator(device="xpu").manual_seed(seed)
    rows = sum(q_lens)
    q = torch.randn(rows, q_heads, 256, device="xpu", dtype=torch.float16,
                    generator=g) * 0.5
    pages = sum((l + page - 1) // page for l in kv_lens)
    k = (torch.randn(pages, page, kv_heads, 256, device="xpu", generator=g)
         * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(pages, page, kv_heads, 256, device="xpu", generator=g)
         * 0.5).to(torch.float8_e4m3fn)
    table = torch.arange(pages, dtype=torch.int32, device="xpu").reshape(
        bs, -1)  # identity mapping, no straddle tricks in debug
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)),
                        dtype=torch.int32, device="xpu")
    seqused = torch.tensor(kv_lens, dtype=torch.int32, device="xpu")
    ks = vs = None
    if descale:
        # broadcast per-seq (serve-faithful; per-head random trips the
        # stock col-0 quirk and breaks the 3-way comparability)
        ks = (torch.rand(bs, 1, device="xpu", generator=g) + 0.5
              ).float().expand(-1, kv_heads).contiguous()
        vs = (torch.rand(bs, 1, device="xpu", generator=g) + 0.5
              ).float().expand(-1, kv_heads).contiguous()
    return q, k, v, cu_q, seqused, table, ks, vs


def ref_torch(q, k, v, cu_q, seqused, table, ks, vs):
    """fp32 causal bottom-right reference walking the SAME paged cache."""
    qf = q.float()
    out = torch.empty_like(qf)
    bs = cu_q.shape[0] - 1
    ksh = ks if ks is not None else torch.ones(bs, k.shape[2], device="xpu")
    vsh = vs if vs is not None else torch.ones(bs, k.shape[2], device="xpu")
    for s in range(bs):
        qb, qe = int(cu_q[s]), int(cu_q[s + 1])
        ql = qe - qb
        kl = int(seqused[s])
        kf = k[s].reshape(-1, k.shape[2], 256).float()[:kl]  # identity table
        vf = v[s].reshape(-1, v.shape[2], 256).float()[:kl]
        for h in range(q.shape[1]):
            kvh = h // (q.shape[1] // k.shape[2])
            kd = float(ksh[s, kvh])
            vd = float(vsh[s, kvh])
            sc = (qf[qb:qe, h] @ kf[:, kvh].T) * (256 ** -0.5) * kd
            for i in range(ql):
                lim = i + (kl - ql)  # bottom-right causal
                p = torch.softmax(sc[i, : lim + 1], 0)
                out[qb + i, h] = (p @ vf[: lim + 1, kvh]) * vd
    return out


def stock(q, k, v, cu_q, seqused, table, ks=None, vs=None):
    dummy = torch.zeros_like(cu_q)
    ones = torch.ones(cu_q.shape[0] - 1, v.shape[2], device="xpu",
                      dtype=torch.float32)  # fp8 path derefs descales
    r = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0,
        ks if ks is not None else ones, vs if vs is not None else ones,
        256 ** -0.5, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return r[0].float()


def kern(q, k, v, cu_q, seqused, table, ks=None, vs=None):
    out = torch.empty_like(q)
    fa_dense_ops.fa_dense_varlen_fwd(
        q, k, v, cu_q, seqused, table, out, ks, vs, 256 ** -0.5)
    return out.float()


def fro(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-9)).item()


cases = [
    ("E probe-case4: bs2 [128,128]/[1024,640] 12/2 desc",
     dict(bs=2, q_lens=[128, 128], kv_lens=[1024, 640], q_heads=12,
          kv_heads=2, page=1664, seed=4, descale=True)),
    ("F same, NO descale",
     dict(bs=2, q_lens=[128, 128], kv_lens=[1024, 640], q_heads=12,
          kv_heads=2, page=1664, seed=4)),
    ("G bs2 EQUAL kv lens",
     dict(bs=2, q_lens=[128, 128], kv_lens=[1024, 1024], q_heads=12,
          kv_heads=2, page=1664, seed=4, descale=True)),
    ("H bs2 tiny [4,4]/[32,20]",
     dict(bs=2, q_lens=[4, 4], kv_lens=[32, 20], q_heads=2, kv_heads=1,
          page=1664, seed=5, descale=True)),
]
for name, c in cases:
    q, k, v, cu_q, seqused, table, ks, vs = build(**c)
    ref = ref_torch(q, k, v, cu_q, seqused, table, ks, vs)
    st = stock(q, k, v, cu_q, seqused, table, ks, vs)
    ke = kern(q, k, v, cu_q, seqused, table, ks, vs)
    print(f"== {name}")
    print(f"   ref-vs-stock fro={fro(st, ref):.3e}   ref-vs-kernel fro={fro(ke, ref):.3e}")
    r0, s0, k0 = ref[0, 0, :6], st[0, 0, :6], ke[0, 0, :6]
    print(f"   row0/h0 ref  {r0.tolist()}")
    print(f"   row0/h0 stock {s0.tolist()}")
    print(f"   row0/h0 kern  {k0.tolist()}")
    print(f"   kern rownorms {[f'{ke[i].norm().item():.2f}' for i in range(0, 16, 4)]}")
    print(f"   ref  rownorms {[f'{ref[i].norm().item():.2f}' for i in range(0, 16, 4)]}")
    # per-seq fro (multi-seq localizer)
    offs = [int(x) for x in cu_q.tolist()]
    for s_ in range(len(offs) - 1):
        k_s = ke[offs[s_]:offs[s_ + 1]]
        r_s = ref[offs[s_]:offs[s_ + 1]]
        st_s = st[offs[s_]:offs[s_ + 1]]
        print(f"   seq{s_}: kern-vs-ref {fro(k_s, r_s):.3e}  stock-vs-ref {fro(st_s, r_s):.3e}")
