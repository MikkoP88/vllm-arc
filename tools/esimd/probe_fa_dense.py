#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p72 offline numerics probe: ESIMD dense-FA kernel vs stock
``_vllm_fa2_C.varlen_fwd`` on serve-shaped randoms.

Probe discipline (W14 banked laws):
  * UNIFORM-TOKEN BLINDNESS — vary slot contents AND page placement; a
    probe whose lanes all read the same cache bytes cannot catch lane
    permutation bugs. Every case uses distinct random pages per sequence.
  * Gate = Frobenius + cosine (elementwise rel is ill-posed: softmax
    averages cross zero). Banked class: fro ~2e-3, cos_min ~0.999998.
  * Ladder: grow one dimension at a time (dims -> GQA -> multi-seq ->
    causal boundary -> non-pow2 page geometry -> serve shape).

Run (rig esimd container, GPUs):
  /opt/venv/bin/python probe_fa_dense.py [--quick]
Sizing note: PAGE=1664 keeps the cache tensor huge; numPages is kept at the
minimum each case needs (the stock op only touches listed pages).
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

# register the stock op before the first torch.ops._vllm_fa2_C call (the
# serve chain registers it via vllm._xpu_ops; standalone containers may
# only have the vllm_xpu_kernels package — either import works)
try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401,E402
except ImportError:
    from vllm import _xpu_ops  # noqa: F401,E402

# kernel .so: esimd build container (/src/esimd-fa/python) or a cert4
# container cap-stage copy (/root/fa_dense_ops.so) — ARC_DENSE_FA_ESIMD_SO wins
_so = os.environ.get("ARC_DENSE_FA_ESIMD_SO", "")
if _so:
    sys.path.insert(0, os.path.dirname(os.path.abspath(_so)))
elif os.path.isdir("/src/esimd-fa/python"):
    sys.path.insert(0, "/src/esimd-fa/python")  # esimd build container
else:
    sys.path.insert(0, "/root")  # cert4 container: .so at /root
import fa_dense_ops  # noqa: E402  (loader law: name == basename)

PAGE = 1664  # serve geometry (hybrid KV manager pads to mamba page size)


def make_case(bs, q_lens, kv_lens, q_heads, kv_heads, seed, page=PAGE,
              descale=True):
    g = torch.Generator(device="xpu").manual_seed(seed)
    rows = sum(q_lens)
    q = torch.randn(rows, q_heads, 256, device="xpu", dtype=torch.float16,
                    generator=g) * 0.5
    # distinct random pages per sequence (blindness law)
    pages_per_seq = [(l + page - 1) // page for l in kv_lens]
    total_pages = sum(pages_per_seq)
    k = (torch.randn(total_pages, page, kv_heads, 256, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    v = (torch.randn(total_pages, page, kv_heads, 256, device="xpu",
                     generator=g) * 0.5).to(torch.float8_e4m3fn)
    # block table: disjoint page ranges, shuffled order within a sequence
    table, base = [], 0
    for s, np_ in enumerate(pages_per_seq):
        ids = list(range(base, base + np_))
        g2 = torch.Generator().manual_seed(seed * 100 + s)
        ids = [ids[i] for i in torch.randperm(len(ids), generator=g2).tolist()]
        table.append(torch.tensor(ids + [0] * (32 - len(ids)),
                                  dtype=torch.int32))
        base += np_
    block_table = torch.stack(table).to("xpu")
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0)),
                        dtype=torch.int32, device="xpu")
    seqused = torch.tensor(kv_lens, dtype=torch.int32, device="xpu")
    ks = vs = None
    if descale == "per_head":
        ks = (torch.rand(bs, kv_heads, device="xpu", generator=g) + 0.5
              ).float()
        vs = (torch.rand(bs, kv_heads, device="xpu", generator=g) + 0.5
              ).float()
    elif descale == "per_seq":
        ks = (torch.rand(bs, 1, device="xpu", generator=g) + 0.5
              ).float().expand(-1, kv_heads).contiguous()
        vs = (torch.rand(bs, 1, device="xpu", generator=g) + 0.5
              ).float().expand(-1, kv_heads).contiguous()
    elif descale:
        # SERVE-FAITHFUL (the only stock-comparable form): the checkpoint
        # ships UNIFORM scales (kv_cache_scales.json: all 128 entries 0.1)
        # and the backend expands a scalar _k_scale — ONE value for all
        # seqs and heads. The stock op mis-indexes descales for seq >= 1
        # and ignores head columns >= 1 (p72 experiments, 2026-11-01) —
        # both invisible under a uniform value, and both mean per-seq /
        # per-head configs can only be validated against torch_ref.
        kv = (torch.rand(1, device="xpu", generator=g) + 0.5).item()
        ks = torch.full((bs, kv_heads), kv, device="xpu")
        vv = (torch.rand(1, device="xpu", generator=g) + 0.5).item()
        vs = torch.full((bs, kv_heads), vv, device="xpu")
    return q, k, v, cu_q, seqused, block_table, ks, vs


def stock_ref(q, k, v, cu_q, seqused, table, ks, vs):
    dummy = torch.zeros_like(cu_q)
    # the fp8 path derefs the descale optionals ("bad optional access"
    # otherwise — p72 debug lesson); None == 1.0 mathematically
    if ks is None:
        ks = torch.ones(cu_q.shape[0] - 1, k.shape[2], device="xpu",
                        dtype=torch.float32)
    if vs is None:
        vs = torch.ones(cu_q.shape[0] - 1, k.shape[2], device="xpu",
                        dtype=torch.float32)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        256 ** -0.5, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


def run_case(name, ref_fn=stock_ref, **kw):
    q, k, v, cu_q, seqused, table, ks, vs = make_case(**kw)
    ref = ref_fn(q, k, v, cu_q, seqused, table, ks, vs).float()
    out = torch.empty_like(q)
    fa_dense_ops.fa_dense_varlen_fwd(
        q, k, v, cu_q, seqused, table, out, ks, vs, 256 ** -0.5)
    o = out.float()
    fro = (o - ref).norm() / ref.norm().clamp_min(1e-9)
    cos = torch.nn.functional.cosine_similarity(
        o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    ok = fro < 5e-3 and cos > 0.99999
    print(f"[{'PASS' if ok else 'FAIL'}] {name:34s} fro={fro:.2e} cos={cos:.8f}")
    return ok


def torch_ref(q, k, v, cu_q, seqused, table, ks, vs):
    """fp32 causal bottom-right reference (identity-table cases only) —
    used for the per-head-descale case where the stock op provably applies
    only column 0 (p72 descale-semantics experiment) and cannot serve as
    the reference."""
    qf = q.float()
    out = torch.empty_like(qf)
    bs, kvh = cu_q.shape[0] - 1, k.shape[2]
    grp = q.shape[1] // kvh
    ksh = ks if ks is not None else torch.ones(bs, kvh, device="xpu")
    vsh = vs if vs is not None else torch.ones(bs, kvh, device="xpu")
    for s in range(bs):
        qb, qe = int(cu_q[s]), int(cu_q[s + 1])
        ql, kl = qe - qb, int(seqused[s])
        kf = k[table[s, 0].item()].reshape(-1, kvh, 256).float()[:kl]
        vf = v[table[s, 0].item()].reshape(-1, kvh, 256).float()[:kl]
        for h in range(q.shape[1]):
            kvh_ = h // grp
            sc = (qf[qb:qe, h] @ kf[:, kvh_].T) * (256 ** -0.5
                                                   ) * float(ksh[s, kvh_])
            for i in range(ql):
                lim = i + (kl - ql)
                p = torch.softmax(sc[i, : lim + 1], 0)
                out[qb + i, h] = (p @ vf[: lim + 1, kvh_]) * float(vsh[s, kvh_])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()

    cases = [
        ("1seq small causal", dict(bs=1, q_lens=[16], kv_lens=[64],
                                   q_heads=12, kv_heads=2, seed=1)),
        ("1seq full-attend (q==kv)", dict(bs=1, q_lens=[256], kv_lens=[256],
                                          q_heads=12, kv_heads=2, seed=2)),
        ("boundary q<k (chunk)", dict(bs=1, q_lens=[512], kv_lens=[2048],
                                      q_heads=12, kv_heads=2, seed=3)),
        ("GQA group 6, 2 kv", dict(bs=2, q_lens=[128, 128],
                                   kv_lens=[1024, 640], q_heads=12,
                                   kv_heads=2, seed=4)),
        ("page-straddle seq", dict(bs=1, q_lens=[64], kv_lens=[1690],
                                   q_heads=12, kv_heads=2, seed=5)),
        ("no-descale", dict(bs=2, q_lens=[64, 64], kv_lens=[512, 333],
                            q_heads=12, kv_heads=2, seed=6, descale=False)),
        ("per-head descale (torch ref)", dict(
            bs=1, q_lens=[16], kv_lens=[64], q_heads=12, kv_heads=2,
            seed=9, descale="per_head"), torch_ref),
        ("per-seq descale (torch ref)", dict(
            bs=2, q_lens=[64, 64], kv_lens=[512, 333], q_heads=12,
            kv_heads=2, seed=10, descale="per_seq"), torch_ref),
        # dense-layer serve page (leg-F ROUTED capture: k=(512,832,2,256));
        # PAGE=1664 above is the mamba-padded hybrid page
        ("dense-serve page 832", dict(bs=2, q_lens=[512, 512],
                                      kv_lens=[1664, 832], q_heads=12,
                                      kv_heads=2, seed=11, page=832)),
    ]
    if not a.quick:
        cases.append(("serve-ish U8k chunk", dict(bs=2, q_lens=[4096, 4096],
                                                  kv_lens=[8192, 5000],
                                                  q_heads=12, kv_heads=2,
                                                  seed=7)))
    allok = all(
        run_case(e[0], ref_fn=(e[2] if len(e) > 2 else stock_ref), **e[1])
        for e in cases)

    # determinism (integrity law): two runs byte-identical
    q, k, v, cu_q, seqused, table, ks, vs = make_case(
        bs=2, q_lens=[256, 128], kv_lens=[2048, 1500], q_heads=12,
        kv_heads=2, seed=8)
    o1 = torch.empty_like(q)
    o2 = torch.empty_like(q)
    fa_dense_ops.fa_dense_varlen_fwd(q, k, v, cu_q, seqused, table, o1, ks,
                                     vs, 256 ** -0.5)
    fa_dense_ops.fa_dense_varlen_fwd(q, k, v, cu_q, seqused, table, o2, ks,
                                     vs, 256 ** -0.5)
    det = torch.equal(o1, o2)
    print(f"[{'PASS' if det else 'FAIL'}] determinism repeat")
    print("ALL_PASS" if allok and det else "SOME_FAIL")
    sys.exit(0 if allok and det else 1)


if __name__ == "__main__":
    main()
