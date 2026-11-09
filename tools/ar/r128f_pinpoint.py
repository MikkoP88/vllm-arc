# r128f_pinpoint.py -- P6.34 arc7 JM GATE-1 cut-5 pinpoint.
# Correctness failed on every cell (rel 0.4 .. 1e4) while the r128e3
# primitive-chain probe passes; this isolates the wheel op on ONE tile
# (M=1, K=32, N=16; tiles=1, splits=1, no swizzle effect) with exact
# e4m3 values through the w16pt lane:
#   run A: all weights code 0x38 (=1.0)      -> every out == 32.0
#   run B: w[0][5] = 2.0 (code 0x40)         -> out[0] == 33.0, rest 32
#   run C: x[0][7] = 3.0                      -> every out == 34.0 (run A)
#   run D: w[4][*] col scale probe via one-hot k: w[2][k]=1 only at k=11
#          (code 0x38 at k=11, 0 elsewhere)  -> out[2] == 1.0, rest 0
# Any layout/lane/k-offset defect shows up as a wrong element position.
import sys
import torch

sys.path.insert(0, "/tmp/arc7wheel")
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402

ops = torch.ops._xpu_C
dev = torch.device("xpu")


def repack_f8(w8):  # [N, K] u8 -> column tiles [N/16, K/16, 256]
    n, k = w8.shape
    nt, kt = (n + 15) // 16, k // 16
    if n < nt * 16:
        w8 = torch.cat(
            [w8, torch.zeros(nt * 16 - n, k, dtype=torch.uint8)], 0)
    return (w8.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
            .contiguous().view(nt, kt, 256))


def run(name, w8, x):
    rp = repack_f8(w8).to(dev)
    xd = x.to(dev)
    out = ops.jm_fp8_gemm_w8a16(xd, rp, None, None)[..., :16].float().cpu()
    ref = (x.float() @ w8.view(torch.float8_e4m3fn).float().t())
    print(f"{name}: jm  {out[0, :6].tolist()}")
    print(f"{name}: ref {ref[0, :6].tolist()}")
    bad = (out - ref).abs().max().item()
    print(f"{name}: maxabs={bad:.4g} {'OK' if bad < 1e-3 else 'WRONG'}")


K, N = 32, 16
C1, C2, C3 = 0x38, 0x40, 0x42  # e4m3: 1.0, 2.0, 3.0
x1 = torch.ones(1, 1, K, dtype=torch.bfloat16)

w = torch.full((N, K), C1, dtype=torch.uint8)
run("A all-1.0     ", w, x1)

w = torch.full((N, K), C1, dtype=torch.uint8)
w[0, 5] = C2
run("B w[0][5]=2   ", w, x1)

x3 = torch.ones(1, 1, K, dtype=torch.bfloat16)
x3[0, 0, 7] = 3.0
w = torch.full((N, K), C1, dtype=torch.uint8)
run("C x[7]=3      ", w, x3)

w = torch.zeros((N, K), dtype=torch.uint8)
w[2, 11] = C1
run("D onehot(2,11)", w, x1)

w = torch.zeros((N, K), dtype=torch.uint8)
for n in range(N):
    w[n, n] = C1  # diagonal: k = n (k < 16 only)
run("E diag k=n    ", w, x1)

# F: dead-k SET via distinct A values -- x[k]=k+1, w all 1.0.
#    out[0] = sum of (k+1) over live k.  Full = 528; dead {3,7,..,31}
#    (=144) -> 384; dead {3,7,11,15} only -> 488.
xf = torch.arange(1, K + 1, dtype=torch.bfloat16).view(1, 1, K)
w = torch.full((N, K), C1, dtype=torch.uint8)
run("F x=k+1 sum   ", w, xf)

# G: dead-k set on B side via distinct w[0][k]=k+1, x all 1.0
#    (confounded with A-dead k: term survives only if BOTH live).
w = torch.full((N, K), C1, dtype=torch.uint8)
wg = torch.arange(1, K + 1, dtype=torch.uint8).clamp(max=0x7E)
w[0] = wg
run("G w[0]=k+1    ", w, x1)

# H: C-store / B-column lane aliveness -- k=0 (known live) one-hots on
#    columns n=3 and n=5.  H1 out[3]==1 => epilogue store from lane 3
#    AND B column 3 both alive; H2 (k=5) same for a k in the second 8.
for (nn, kk) in ((3, 0), (3, 5), (5, 3), (0, 3)):
    w = torch.zeros((N, K), dtype=torch.uint8)
    w[nn, kk] = C1
    run(f"H w[{nn}][{kk}]=1  ", w, x1)
print("PINPOINT_DONE")
