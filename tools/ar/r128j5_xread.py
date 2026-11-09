# r128j5_xread.py -- P6.34 arc7 JM g16 x-read discriminator.
# Laws so far: gcell A/B/C FAIL deterministically; out values are EXACT
# multiples of 0.5 -> impossible for x~randn dot products -> the failing
# outputs carry NO x dependence (kernel sees x~=ones?). mixmatch F / P
# PASS deterministically, poison-immune. gcell vs mixmatch differ ONLY
# in RNG stream + allocation history.
# This probe: exact gcell-A construction, then
#   (1) fingerprint x/w4/ws4 device-side vs host-side (aliasing check)
#   (2) out(x) vs out(2x) vs out(ones) -- x-dependence of the KERNEL
#   (3) ref via 3 paths: x.double()@dev / x.cpu().double() / einsum-CPU
#   (4) ref(ones): does it reproduce the -836.0 family EXACTLY?
import sys
import torch

sys.path.insert(0, "/tmp/arc7wheel")
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402

ops = torch.ops._xpu_C
dev = torch.device("xpu")
LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64)
K, N, M = 1024, 1003, 1
npad = ((N + 15) // 16) * 16


def mk_fullrange_f8(n, k):
    base = torch.arange(256, dtype=torch.uint8, device=dev)
    base[127] = 126
    base[255] = 254
    return base.repeat((n * k + 255) // 256)[: n * k].view(n, k)


def nib_vals(w4):
    lo, hi = w4 & 0xF, w4 >> 4
    return torch.stack((lo, hi), dim=2).reshape(
        w4.shape[0], w4.shape[1] * 2)


def _pad_n(t, npad_, dim):
    if t.shape[dim] == npad_:
        return t
    pad = torch.zeros(
        [npad_ - t.shape[dim] if i == dim else s
         for i, s in enumerate(t.shape)],
        dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def repack_f4(w4v):
    n, k = w4v.shape
    nt, kt = (n + 15) // 16, k // 16
    w4v = _pad_n(w4v, nt * 16, 0)
    v = (w4v.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


# ---- exact gcell cell-A sequence ----
torch.manual_seed(0)
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w4_u8 = mk_fullrange_f8(N, K // 2)
ws4 = torch.randint(123, 131, (K // 32, N), dtype=torch.uint8, device=dev)
rp4 = repack_f4(nib_vals(w4_u8))
sc = _pad_n(ws4, npad, 1).view(torch.float8_e8m0fnu)

# (1) fingerprints: device-side read vs host copy of the SAME tensor
xs_dev = x.float().sum().item()
xs_host = x.cpu().float().sum().item()
ws_dev = int(ws4.long().sum().item())
ws_host = int(ws4.cpu().long().sum().item())
print(f"x  sum dev={xs_dev:.6f} host={xs_host:.6f} "
      f"{'SAME' if abs(xs_dev - xs_host) < 1e-2 else 'DIVERGE'}")
print(f"ws sum dev={ws_dev} host={ws_host} "
      f"{'SAME' if ws_dev == ws_host else 'DIVERGE'}")
print(f"x  first8 host={x.cpu().reshape(-1)[:8].tolist()}")

# (2) kernel x-dependence
out1 = ops.jm_fp4_gemm_w4a16(x, rp4, sc, torch.bfloat16, None)[..., :N]
out2 = ops.jm_fp4_gemm_w4a16(2 * x, rp4, sc, torch.bfloat16, None)[..., :N]
ones = torch.ones_like(x)
outO = ops.jm_fp4_gemm_w4a16(ones, rp4, sc, torch.bfloat16, None)[..., :N]
o1 = out1.float().cpu().double()
o2 = out2.float().cpu().double()
oO = outO.float().cpu().double()
print(f"out(x)[0,:4]   = {o1[0, :4].tolist()}")
print(f"out(2x)[0,:4]  = {o2[0, :4].tolist()}")
print(f"out(ones)[0,:4]= {oO[0, :4].tolist()}")
r21 = (o2 / o1.clamp_min(1e-9)).abs().median().item()
print(f"median |out(2x)/out(x)| = {r21:.6g} (2 => x-dependent)")
d1O = (o1 - oO).abs().max().item()
print(f"max |out(x) - out(ones)| = {d1O:.6g} "
      f"({'x-dependent' if d1O > 1 else 'KERNEL SEES ONES/CONST'})")

# (3) three refs
w4c = w4_u8.cpu()
wsc = ws4.cpu()
lo = LUT[(w4c & 0xF).long()]
hi = LUT[(w4c >> 4).long()]
w = torch.stack((lo, hi), dim=1).reshape(-1, K)
g = torch.pow(2.0, wsc.float() - 127.0).double()
wg = w.t() * torch.repeat_interleave(g, 32, dim=0)
refA = x.double().cpu() @ wg              # cast on DEV then copy (gcell)
refB = x.cpu().double() @ wg              # copy then cast (mixmatch)
refC = torch.einsum(
    "k,nk->n", x.cpu().double(),
    w * g.repeat_interleave(32, 1).t())   # pure CPU einsum
for nm, r in (("refA cast-on-dev", refA), ("refB cast-on-cpu", refB),
              ("refC einsum-cpu", refC)):
    rel = ((o1[0] - r[0]).abs() / r[0].abs().clamp_min(1.0)).max().item()
    print(f"{nm}: [0,:4]={r[0, :4].tolist()} rel-vs-out={rel:.4g}")
print(f"refA vs refB maxabs={(refA - refB).abs().max().item():.3g} "
      f"refA vs refC maxabs={(refA - refC).abs().max().item():.3g}")

# (4) does ref(ones) reproduce the out family?
refO = torch.einsum(
    "k,nk->n", torch.ones(K, dtype=torch.float64),
    w * g.repeat_interleave(32, 1).t())
print(f"ref(ones)[0,:4]={refO[:4].tolist()}")
print(f"out==ref(ones) maxabs={(o1[0] - refO).abs().max().item():.6g}")
print("XREAD_DONE")
