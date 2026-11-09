# r128j7_addr.py -- P6.34 arc7 JM g16 address-vs-value discriminator.
# Laws: gcell-A construction FAILS (rel 1.758, deterministic, out = exact
# multiples of 0.5); the SAME VALUES in a different allocation order PASS
# (r128j6: rel 0.0077, one-hots exact, blocks clean). => layout-dependent
# kernel defect, not value-dependent. Surgery on the FAILING context:
#   (1) reproduce the FAIL verbatim
#   (2) same VALUES via .clone() (fresh addresses) -> verdict
#   (3) original tensors, repeated calls -> still fails?
#   (4) one-hot k=2 in-context (was exact in the passing context)
#   (5) out(ones) vs CPU ref(ones) in-context
#   (6) data_ptrs of every operand + out, before and after
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

torch.manual_seed(0)
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
base = torch.arange(256, dtype=torch.uint8, device=dev)
base[127] = 126
base[255] = 254
w4_u8 = base.repeat((N * (K // 2) + 255) // 256)[
    : N * (K // 2)].view(N, K // 2)
ws4 = torch.randint(123, 131, (K // 32, N), dtype=torch.uint8, device=dev)
lo, hi = w4_u8 & 0xF, w4_u8 >> 4
w4n = torch.stack((lo, hi), dim=2).reshape(N, K)
w4n = torch.cat([w4n, torch.zeros(
    npad - N, K, dtype=torch.uint8, device=dev)], 0)
v = (w4n.view(npad // 16, 16, K // 16, 16).permute(0, 2, 1, 3)
     .contiguous().view(npad // 16, K // 16, 16, 8, 2))
rp4 = (v[..., 0] | (v[..., 1] << 4)).view(npad // 16, K // 16, 128)
sc = torch.cat([ws4, torch.zeros(
    (K // 32, npad - N), dtype=torch.uint8, device=dev)], 1
).contiguous().view(torch.float8_e8m0fnu)

# CPU truth (bytes copied before any op call)
w4c = torch.stack((w4_u8.cpu() & 0xF, w4_u8.cpu() >> 4), dim=2).reshape(
    N, K)
W = LUT[w4c.long()].double()
G = torch.pow(2.0, ws4.cpu().float() - 127.0).double()
WGK = (W * G.t().repeat_interleave(32, 1)).t().contiguous()


def rel_of(out):
    ref = x.cpu().double() @ WGK
    return ((out - ref).abs() / ref.abs().clamp_min(1.0)).max().item()


def ptrs(tag):
    print(f"  ptrs[{tag}] x={x.data_ptr():#x} rp4={rp4.data_ptr():#x} "
          f"sc={sc.data_ptr():#x}", flush=True)


def call(xv, rpv, scv):
    o = ops.jm_fp4_gemm_w4a16(xv, rpv, scv, torch.bfloat16, None)
    return o[..., :N].float().cpu().double()[0]


ptrs("built")
o1 = call(x, rp4, sc)
r1 = rel_of(o1)
print(f"1 original tensors        rel={r1:.4g} "
      f"{'OK' if r1 < 0.02 else 'FAIL'} out[:4]={o1[:4].tolist()}")

o2 = call(x, rp4, sc)
print(f"3 repeat call #2          rel={rel_of(o2):.4g} "
      f"{'OK' if rel_of(o2) < 0.02 else 'FAIL'}")
o3 = call(x, rp4, sc)
print(f"3 repeat call #3          rel={rel_of(o3):.4g} "
      f"{'OK' if rel_of(o3) < 0.02 else 'FAIL'}")

xc, rc, scc = x.clone(), rp4.clone(), sc.clone()
print(f"  clone ptrs x={xc.data_ptr():#x} rp4={rc.data_ptr():#x} "
      f"sc={scc.data_ptr():#x}")
o4 = call(xc, rc, scc)
# ref must use the clone bytes -- identical values, but verify anyway
assert torch.equal(xc.cpu(), x.cpu())
r4 = rel_of(o4)
print(f"2 CLONED values           rel={r4:.4g} "
      f"{'OK' if r4 < 0.02 else 'FAIL'} out[:4]={o4[:4].tolist()}")

e = torch.zeros_like(x)
e[0, 2] = 1.0
a2 = call(e, rp4, sc)
print(f"4 one-hot k=2 in-context  maxabs-vs-true="
      f"{(a2 - WGK[2]).abs().max().item():.4g} "
      f"got[:3]={a2[:3].tolist()} true[:3]={WGK[2][:3].tolist()}")

ones = torch.ones_like(x)
oO = call(ones, rp4, sc)
refO = torch.ones(K, dtype=torch.float64) @ WGK
print(f"5 out(ones) vs CPU ref(ones) maxabs="
      f"{(oO - refO).abs().max().item():.4g} "
      f"got[:4]={oO[:4].tolist()} true[:4]={refO[:4].tolist()}")

# 6) empty_cache then rebuild same values in a shaken heap
del o1, o2, o3, o4, a2, oO
torch.xpu.empty_cache()
ptrs("after empty_cache (originals alive)")
o5 = call(x, rp4, sc)
r5 = rel_of(o5)
print(f"6 after empty_cache       rel={r5:.4g} "
      f"{'OK' if r5 < 0.02 else 'FAIL'}")
print("ADDR_DONE")
