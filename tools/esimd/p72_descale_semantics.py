#!/usr/bin/env python3
"""p72: what does stock varlen_fwd do with PER-HEAD-varying [bs, kvHeads]
descales? (serve never sends them — _k_scale is scalar-broadcast — but the
probe tripped over it: kernel/math say group h//6, stock does something
else). ks=[1,2], vs=1: compare stock per-q-head vs H1 (group h//G, math) /
H2 (alternating h%kvH) / H3 (first col only)."""
import sys

import torch

try:
    import vllm_xpu_kernels.flash_attn_interface  # noqa: F401
except ImportError:
    from vllm import _xpu_ops  # noqa: F401

sys.path.insert(0, "/root")
import fa_dense_ops  # noqa: E402

g = torch.Generator(device="xpu").manual_seed(3)
qh, kvh = 12, 2
q = torch.randn(16, qh, 256, device="xpu", dtype=torch.float16, generator=g) * 0.5
k = (torch.randn(1, 1664, kvh, 256, device="xpu", generator=g) * 0.5).to(torch.float8_e4m3fn)
v = (torch.randn(1, 1664, kvh, 256, device="xpu", generator=g) * 0.5).to(torch.float8_e4m3fn)
cu_q = torch.tensor([0, 16], dtype=torch.int32, device="xpu")
seqused = torch.tensor([64], dtype=torch.int32, device="xpu")
table = torch.zeros(1, 32, dtype=torch.int32, device="xpu")
ks = torch.tensor([[1.0, 2.0]], device="xpu")
vs = torch.ones(1, kvh, device="xpu")


def stock(ks):
    r = torch.ops._vllm_fa2_C.varlen_fwd(
        q, k, v, None, cu_q, torch.zeros_like(cu_q), seqused, None, table,
        None, 16, 64, 0.0, ks, vs, 256 ** -0.5, None, False, True, -1, -1,
        0.0, False, None, 0, False, None, None)
    return r[0].float()


def ref(kd_per_head):
    qf = q.float()
    kf = k[0, :64].float()
    vf = v[0, :64].float()
    out = torch.empty_like(qf)
    for h in range(qh):
        sc = (qf[:, h] @ kf[:, h // 6].T) * (256 ** -0.5) * kd_per_head[h]
        for i in range(16):
            lim = i + 48
            p = torch.softmax(sc[i, : lim + 1], 0)
            out[i, h] = p @ vf[: lim + 1, h // 6]
    return out


st = stock(ks)
h1 = ref([1.0 if h < 6 else 2.0 for h in range(qh)])   # group (math, kernel)
h2 = ref([1.0 if h % 2 == 0 else 2.0 for h in range(qh)])  # alternating
h3 = ref([1.0] * qh)                                    # ignores col 1
def phfro(a, b, h):
    return ((a[:, h] - b[:, h]).norm() / b[:, h].norm().clamp_min(1e-9)).item()


for nm, r in (("H1 group h//6 (kernel)", h1), ("H2 alternating h%2", h2),
              ("H3 first-col-only", h3)):
    print(f"{nm:26s} fro={((st - r).norm() / r.norm()).item():.3e} "
          f"head0={phfro(st, r, 0):.2e} head1={phfro(st, r, 1):.2e} "
          f"head6={phfro(st, r, 6):.2e} head11={phfro(st, r, 11):.2e}")
# kernel side (H1 expected)
out = torch.empty_like(q)
fa_dense_ops.fa_dense_varlen_fwd(q, k, v, cu_q, seqused, table, out, ks, vs,
                                 256 ** -0.5)
ke = out.float()
print(f"kernel vs H1             fro={((ke - h1).norm() / h1.norm()).item():.3e}")
