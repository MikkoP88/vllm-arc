# r128j6_coef.py -- P6.34 arc7 JM g16 A-side coefficient telemetry.
# Laws: kernel linear in x (out(2x)=2*out(x) EXACT); tensors clean
# dev==host; yet out(x) is an exact multiple of 0.5 with fractional
# bf16 x -> the A-side coefficients the kernel actually applies must be
# integral (x seen quantized) or wrong. Extract them directly:
#   (1) one-hot e_k probes k=0..15 -> kernel coefficient per (col, k),
#       compare vs true LUT[nib]*scale
#   (2) block sweep: x masked to each 128-k block -> per-block output
#       vs true block partial sums (localize wrong k ranges)
#   (3) triple CPU ref paths on the verbatim cell
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


# ---- verbatim gcell-A tensors (deterministic FAIL law) ----
torch.manual_seed(0)
x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w4_u8 = mk_fullrange_f8(N, K // 2)
ws4 = torch.randint(123, 131, (K // 32, N), dtype=torch.uint8, device=dev)
rp4 = repack_f4(nib_vals(w4_u8))
sc = _pad_n(ws4, npad, 1).view(torch.float8_e8m0fnu)

w4c = nib_vals(w4_u8).cpu()                      # [N, K] nibble values
wsc = ws4.cpu()
W = LUT[w4c.long()].double()                     # [N, K] true weights
G = torch.pow(2.0, wsc.float() - 127.0).double()  # [K/32, N]
WG = W * G.t().repeat_interleave(32, 1)          # [N, K] scaled
WGK = WG.t().contiguous()                        # [K, N]


def kern(xv):
    return ops.jm_fp4_gemm_w4a16(
        xv, rp4, sc, torch.bfloat16, None)[..., :N].float().cpu().double()


# (3) triple ref first
o1 = kern(x)[0]
refB = x.cpu().double() @ WGK
refC = torch.einsum("k,kn->n", x.cpu().double().reshape(-1), WGK)
print(f"refB vs refC maxabs={(refB - refC).abs().max().item():.3g}")
rel = ((o1 - refB).abs() / refB.abs().clamp_min(1.0)).max().item()
print(f"cell verbatim rel-vs-refB = {rel:.4g}")

# (1) one-hot coefficient extraction, k = 0..15
print("-- one-hot coefficients (col 0..3 shown; true = WG[k, col]) --")
bad_ks = []
for kk in range(16):
    e = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
    e[0, kk] = 1.0
    a = kern(e)[0]
    truth = WGK[kk]
    d = (a - truth).abs().max().item()
    ok = d < 0.05
    if not ok:
        bad_ks.append(kk)
    if kk < 8 or not ok:
        print(f"  k={kk:2d}: got {a[:3].tolist()} true "
              f"{truth[:3].tolist()} maxabs={d:.4g} "
              f"{'OK' if ok else 'BAD'}")
print(f"bad one-hot ks: {bad_ks}")

# (2) block sweep: masked x per 128-k block
print("-- block sweep (128-k blocks, rel vs true partial) --")
for b0 in range(0, K, 128):
    xm = torch.zeros_like(x)
    xm[0, b0:b0 + 128] = x[0, b0:b0 + 128]
    ob = kern(xm)[0]
    rb = xm.cpu().double() @ WGK
    relb = ((ob - rb).abs() / rb.abs().clamp_min(1.0)).max().item()
    print(f"  block k={b0:4d}..{b0 + 127:4d}: rel={relb:.4g} "
          f"{'OK' if relb < 0.05 else 'BAD'}", flush=True)
print("COEF_DONE")
