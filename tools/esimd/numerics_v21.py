#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P6.21 v2.1 numerics ladder (post N=8->N=16 QK' rewrite).

Gates (plan P6.21 r88 vehicle):
  A  identity attention: out[m]/vd one-hot at dim m, all rows.
  L1 fro/cos vs chunked fp32 torch-exact on small/mid legal shapes
     (rows <= kv), permuted page tables, PER-KV-HEAD descales:
     fro < 5e-3 AND cos > 0.99999.
  L2 U32k slice (8192, 32768) vs the stock vxk op (fro < 8e-3,
     cos > 0.99999) — stock was torch-validated in r87/r88 legs.
  L3 OOB canary pool around exact-size q/k/v/out at the U32k shape
     (v1 corrupted neighbors there) + a mid shape.
"""
from __future__ import annotations

import json
import os
import sys

import torch

try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401,E402
except ImportError:
    from vllm import _xpu_ops  # noqa: E402

_so = os.environ.get("ARC_DENSE_FA_ESIMD_SO", "")
if _so:
    sys.path.insert(0, os.path.dirname(os.path.abspath(_so)))
import fa_dense_v2_ops  # noqa: E402

QH, KVH, HD = 12, 2, 256
SCALE = HD ** -0.5
PAGE = 832
E4 = torch.float8_e4m3fn
KS = [1.3, 0.7]   # per-kvHead descales (catches kvHead indexing)
VS = [1.7, 1.1]


def make_case(rows, kv_len, seed):
    g = torch.Generator(device="xpu").manual_seed(seed)
    q = torch.randn(rows, QH, HD, device="xpu", dtype=torch.float16,
                    generator=g) * 0.5
    npages = (kv_len + PAGE - 1) // PAGE
    pad = max(1, npages // 8)
    k = (torch.randn(npages + pad, PAGE, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(E4)
    v = (torch.randn(npages + pad, PAGE, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(E4)
    ids = list(range(npages))
    gcpu = torch.Generator().manual_seed(seed * 7 + 1)
    perm = torch.randperm(len(ids), generator=gcpu).tolist()
    table = torch.tensor([ids[i] for i in perm] + [0] * pad,
                         dtype=torch.int32, device="xpu").unsqueeze(0)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
    seqused = torch.tensor([kv_len], dtype=torch.int32, device="xpu")
    ks = torch.tensor([KS], device="xpu")
    vs = torch.tensor([VS], device="xpu")
    return q, k, v, cu_q, seqused, table, ks, vs


def run_v2(q, k, v, cu_q, seqused, table, ks, vs):
    out = torch.empty_like(q)
    fa_dense_v2_ops.fa_dense_varlen_fwd_v2(
        q, k, v, cu_q, seqused, table, out, ks, vs, SCALE)
    torch.xpu.synchronize()
    return out


def stock_ref(q, k, v, cu_q, seqused, table, ks, vs):
    dummy = torch.zeros_like(cu_q)
    out, _ = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, dummy, seqused, None, table, None,
        int(cu_q[1].item()), int(seqused.max().item()), 0.0, ks, vs,
        SCALE, None, False, True, -1, -1, 0.0, False, None, 0, False,
        None, None)
    return out


def exact_head(qh, kt, vt, kv, kv_off, kd, vd, chunk=1024):
    """Exact fp32 attention for one head, chunked over rows with a
    vectorized bottom-right causal mask."""
    rows = qh.shape[0]
    kt = kt * kd
    toks = torch.arange(kv, device="xpu")
    out = torch.empty(rows, HD, device="xpu", dtype=torch.float32)
    for r0 in range(0, rows, chunk):
        r1 = min(r0 + chunk, rows)
        s = (qh[r0:r1].float() @ kt.T) * SCALE
        mask = toks.unsqueeze(0) > (
            torch.arange(r0, r1, device="xpu").unsqueeze(1) + kv_off)
        s = s.masked_fill(mask, float("-inf"))
        out[r0:r1] = torch.softmax(s, dim=-1) @ vt
    return out * vd


def exact_ref(q, k, v, table, rows, kv):
    tab = table[0].tolist()
    npg = (kv + PAGE - 1) // PAGE
    kv_off = kv - rows
    ref = torch.empty(rows, QH, HD, device="xpu", dtype=torch.float32)
    for h in range(QH):
        kvh = h // (QH // KVH)
        kt = torch.cat([k[tab[pg], :, kvh, :].float() for pg in
                        range(npg)], 0)[:kv]
        vt = torch.cat([v[tab[pg], :, kvh, :].float() for pg in
                        range(npg)], 0)[:kv]
        ref[:, h, :] = exact_head(q[:, h, :], kt, vt, kv, kv_off,
                                  KS[kvh], VS[kvh])
        del kt, vt
    return ref


def metrics(o, r):
    o, r = o.float().flatten(), r.float().flatten()
    fro = ((o - r).norm() / r.norm().clamp_min(1e-9)).item()
    cos = (torch.dot(o, r) / (o.norm() * r.norm()).clamp_min(1e-9)).item()
    return fro, cos


def case_a(rows, kv, res):
    # NOTE the expectation: q[m]=e_m, K[t]=e_t => logits are EXACTLY 0
    # for t != m (not -inf), so out[m]/vd = softmax(c*delta(t==m)) over
    # the causal window t <= m + (kv - rows) — one-hot ONLY when the
    # window is the single token m. p_m = e^c/(e^c + edge).
    q = torch.zeros(rows, QH, HD, device="xpu", dtype=torch.float16)
    for m in range(rows):
        q[m, :, m] = 1.0
    k = torch.zeros(2, 832, KVH, HD, device="xpu", dtype=E4)
    v = torch.zeros(2, 832, KVH, HD, device="xpu", dtype=E4)
    for t in range(min(kv, 256)):
        k[0, t, :, t] = 1.0
        v[0, t, :, t] = 1.0
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
    seqused = torch.tensor([kv], dtype=torch.int32, device="xpu")
    table = torch.zeros(1, 4, dtype=torch.int32, device="xpu")
    ks = torch.tensor([KS], device="xpu")
    vs = torch.tensor([VS], device="xpu")
    out = run_v2(q, k, v, cu_q, seqused, table, ks, vs)
    c = SCALE * KS[0]
    ec = float(torch.exp(torch.tensor(c)))
    kv_off = kv - rows
    worst = 0.0
    bad = []
    for r in range(rows):
        n = r + kv_off + 1
        p1 = ec / (ec + (n - 1))
        row = (out[r, 0, :].float() / VS[0]).cpu()
        got = row[:n]
        exp_vec = torch.full((n,), (1.0 - p1) / (n - 1)) if n > 1 else \
            torch.ones((1,))
        exp_vec[r] = p1
        rf = ((got - exp_vec).norm() /
              exp_vec.norm().clamp_min(1e-9)).item()
        tail = row[n:].abs().max().item() if n < HD else 0.0
        if rf > 0.02 or tail > 1e-3:
            bad.append((r, round(rf, 4), round(tail, 5)))
        worst = max(worst, rf)
    ok = not bad
    res.append(("A", f"{rows}x{kv}", ok,
                f"worst row-fro={worst:.4f}" if ok else f"bad={bad[:4]}"))
    print(f"A ({rows}x{kv}): {'PASS' if ok else f'FAIL {bad[:4]}'} "
          f"(worst row-fro={worst:.4f})")


def l1(rows, kv, seed, res):
    q, k, v, cu_q, seqused, table, ks, vs = make_case(rows, kv, seed)
    out = run_v2(q, k, v, cu_q, seqused, table, ks, vs)
    ref = exact_ref(q, k, v, table, rows, kv)
    fro, cos = metrics(out, ref.to(torch.float16))
    ok = fro < 5e-3 and cos > 0.99999
    res.append(("L1", f"{rows}x{kv}", ok, f"fro={fro:.2e} cos={cos:.7f}"))
    print(f"L1 ({rows}x{kv}) vs torch-exact: fro={fro:.2e} "
          f"cos={cos:.7f} -> {'PASS' if ok else 'FAIL'}")


def l2(res):
    # (a) U32k vs stock under UNIFORM descales — stock's descale
    # handling for PER-KV-HEAD-varying values is NOT the h//group
    # semantic (p72_descale_semantics evidence; serve only ever sends
    # scalar-broadcast), so a uniform ks/vs isolates kernel math.
    rows, kv = 8192, 32768
    q, k, v, cu_q, seqused, table, ks, vs = make_case(rows, kv, seed=99)
    ks = torch.full((1, KVH), KS[0], device="xpu")
    vs = torch.full((1, KVH), VS[0], device="xpu")
    out = run_v2(q, k, v, cu_q, seqused, table, ks, vs)
    ref = stock_ref(q, k, v, cu_q, seqused, table, ks, vs)
    fro, cos = metrics(out, ref)
    ok = fro < 8e-3 and cos > 0.99999
    res.append(("L2a", f"{rows}x{kv}", ok,
                f"vs-stock fro={fro:.2e} cos={cos:.7f}"))
    print(f"L2a U32k vs stock (uniform ds): fro={fro:.2e} "
          f"cos={cos:.7f} -> {'PASS' if ok else 'FAIL'}")
    # (b) U16k-class slice vs torch-exact WITH per-kv-head descales.
    rows, kv = 4096, 16384
    q, k, v, cu_q, seqused, table, ks, vs = make_case(rows, kv, seed=98)
    out = run_v2(q, k, v, cu_q, seqused, table, ks, vs)
    ref = exact_ref(q, k, v, table, rows, kv)
    fro, cos = metrics(out, ref.to(torch.float16))
    ok = fro < 5e-3 and cos > 0.99999
    res.append(("L2b", f"{rows}x{kv}", ok,
                f"vs-exact fro={fro:.2e} cos={cos:.7f}"))
    print(f"L2b ({rows}x{kv}) vs torch-exact: fro={fro:.2e} "
          f"cos={cos:.7f} -> {'PASS' if ok else 'FAIL'}")


def l3(shape, seed, res):
    """OOB canary pool: exact-size q/k/v/out carved from one flat u8
    pool with A5-filled canaries between and around them."""
    rows, kv = shape
    qn = rows * QH * HD * 2
    npages = (kv + PAGE - 1) // PAGE
    pad = max(1, npages // 8)
    kn = (npages + pad) * PAGE * KVH * HD
    can = 65536

    def align(x):
        return (x + 255) & ~255

    offs, cur = [], 0
    for sz in (can, qn, can, kn, can, kn, can, qn, can):
        cur = align(cur)
        offs.append(cur)
        cur += sz
    pool = torch.full((cur,), 165, dtype=torch.uint8, device="xpu")
    qo, ko, vo, oo = offs[1], offs[3], offs[5], offs[7]
    canaries = [offs[0], offs[2], offs[4], offs[6], offs[8]]
    g = torch.Generator(device="xpu").manual_seed(seed)
    q = torch.randn(rows, QH, HD, device="xpu", dtype=torch.float16,
                    generator=g) * 0.5
    pool[qo:qo + qn].view(torch.float16).copy_(q.flatten())
    k = (torch.randn(npages + pad, PAGE, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(E4)
    v = (torch.randn(npages + pad, PAGE, KVH, HD, device="xpu",
                     generator=g) * 0.5).to(E4)
    pool[ko:ko + kn].view(E4).copy_(k.flatten())
    pool[vo:vo + kn].view(E4).copy_(v.flatten())
    qv = pool[qo:qo + qn].view(torch.float16).view(rows, QH, HD)
    kvv = pool[ko:ko + kn].view(E4).view(npages + pad, PAGE, KVH, HD)
    vv = pool[vo:vo + kn].view(E4).view(npages + pad, PAGE, KVH, HD)
    ov = pool[oo:oo + qn].view(torch.float16).view(rows, QH, HD)
    ids = list(range(npages))
    gcpu = torch.Generator().manual_seed(seed * 7 + 1)
    perm = torch.randperm(len(ids), generator=gcpu).tolist()
    table = torch.tensor([ids[i] for i in perm] + [0] * pad,
                         dtype=torch.int32, device="xpu").unsqueeze(0)
    cu_q = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
    seqused = torch.tensor([kv], dtype=torch.int32, device="xpu")
    ks = torch.tensor([KS], device="xpu")
    vs = torch.tensor([VS], device="xpu")
    fa_dense_v2_ops.fa_dense_varlen_fwd_v2(
        qv, kvv, vv, cu_q, seqused, table, ov, ks, vs, SCALE)
    torch.xpu.synchronize()
    bad = sum(int((pool[c:c + 4096] != 165).sum().item()) for c in canaries)
    ok = bad == 0
    res.append(("L3", f"{rows}x{kv}", ok, f"canary bytes corrupted={bad}"))
    print(f"L3 OOB ({rows}x{kv}): canary corrupted={bad} "
          f"-> {'PASS' if ok else 'FAIL'}")


def main() -> int:
    torch.xpu.set_device("xpu:0")
    res = []
    case_a(8, 8, res)
    case_a(8, 64, res)
    case_a(64, 64, res)
    l1(8, 832, seed=11, res=res)
    l1(64, 832, seed=12, res=res)
    l1(512, 832, seed=13, res=res)
    l1(512, 1664, seed=14, res=res)
    l1(100, 833, seed=15, res=res)     # non-mult-64 rows, partial page
    l1(2048, 8192, seed=16, res=res)
    l2(res)
    l3((8192, 32768), seed=17, res=res)
    l3((512, 832), seed=18, res=res)
    allok = all(r[2] for r in res)
    with open("/work/numerics_v21.json", "w") as f:
        json.dump({"all_pass": allok, "cases": [list(r) for r in res]},
                  f, indent=1)
    print("=== LADDER", "PASS" if allok else "FAIL", "===")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
