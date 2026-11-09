# r128i2_f4probe.py -- P6.34 arc7 JM f4 (g16 lane) pinpoint.
# GATE cut 6: m816 (e4m3 + group32) PASSES; g16 (e2m1 + group32) fails
# huge -> the f4-specific path. Sweep ALL 16 e2m1 codes as uniform
# nibbles + even/odd split + one-hot nibble position tests.
# M=1, K=32 (one k-tile, one group), N=16, uniform scale 127.
# Weights are NIBBLE VALUES [N, K] (gate repack_f4 contract).
import sys
import torch

sys.path.insert(0, "/tmp/arc7wheel")
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402

ops = torch.ops._xpu_C
dev = torch.device("xpu")

LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

K, N = 32, 16
x1 = torch.ones(1, 1, K, dtype=torch.bfloat16, device=dev)
sc = torch.full((K // 32, N), 127, dtype=torch.uint8, device=dev
                ).view(torch.float8_e8m0fnu)


def pad_n(t, rows):
    pad = torch.zeros((rows - t.shape[0], t.shape[1]), dtype=t.dtype)
    return torch.cat([t, pad], 0)


def repack_f4(w4v):  # nibble values [N, K] -> tiles [N/16, K/16, 128]
    n, k = w4v.shape
    nt, kt = (n + 15) // 16, k // 16
    w4v = pad_n(w4v, nt * 16)
    v = (w4v.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


def run(name, w4v, expect):
    rp = repack_f4(w4v).to(dev)
    out = ops.jm_fp4_gemm_w4a16(x1, rp, sc, torch.bfloat16, None)
    got = out[..., :N].float().reshape(-1).cpu()
    if not torch.is_tensor(expect):
        expect = torch.full((N,), float(expect))
    bad = (got.double() - expect.double()).abs().max().item()
    print(f"{name}: jm {got[:6].tolist()} "
          f"maxabs={bad:.4g} {'OK' if bad < 1e-3 else 'WRONG'}")


# T1 all 16 codes uniform
for c in range(16):
    w4v = torch.full((N, K), c, dtype=torch.uint8)
    run(f"T1 nibble={c:2d}   ", w4v, 32.0 * LUT[c])

# T2 even k = code 2 (1.0), odd k = code 4 (2.0) -> 16*1 + 16*2 = 48
w4v = torch.zeros(N, K, dtype=torch.uint8)
w4v[:, 0::2] = 2
w4v[:, 1::2] = 4
run("T2 even1 odd2  ", w4v, 48.0)

# T3 one-hot: column 2, k=0 (even -> lo nibble) = code 4 (2.0)
w4v = torch.zeros(N, K, dtype=torch.uint8)
w4v[2, 0] = 4
e3 = torch.zeros(N)
e3[2] = 2.0
run("T3 c2 k0 =2    ", w4v, e3)

# T4 one-hot odd k: column n, k = 2*(n%16)+1 = code 2 (1.0)
w4v = torch.zeros(N, K, dtype=torch.uint8)
for n_ in range(N):
    w4v[n_, 2 * (n_ % 16) + 1] = 2
run("T4 odd k onehot", w4v, torch.ones(N))

# T5 scale interplay: scale byte 129 (4x) on group 0, weights code 2
sc2 = torch.full((K // 32, N), 127, dtype=torch.uint8, device=dev)
sc2[0, :] = 129
w4v = torch.full((N, K), 2, dtype=torch.uint8)
rp = repack_f4(w4v).to(dev)
out = ops.jm_fp4_gemm_w4a16(x1, rp, sc2.view(torch.float8_e8m0fnu),
                            torch.bfloat16, None)[..., :N].float().cpu()
bad = (out.reshape(-1).double() - 80.0).abs().max().item()
print(f"T5 g0 scale 4x  : jm {out.reshape(-1)[:4].tolist()} "
      f"maxabs={bad:.4g} {'OK' if bad < 1e-3 else 'WRONG'}")
print("F4PROBE_DONE")
