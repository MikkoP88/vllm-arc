# r128i_mxprobe.py -- P6.34 arc7 JM GROUP32 (mxfp8 g16) scale-path probe.
# GATE cut 6: f8 lanes all PASS, g16 fails huge (rel 1e2..1e3) -> the
# e8m0 group-scale path (resolve_group_strides / kernel read / harness
# repack) has its own defect. This probe isolates with exact powers of
# two: M=1, K=64 (2 groups), N=16, w all e4m3 1.0.
#   T1 uniform sc=127 (2^0):      out == 64 everywhere
#   T2 sc[0, g0]=128 (2x):        out[0] == 96, rest 64
#   T3 sc[:, g1]=128:             every out == 96 (k 32..63 doubled)
#   T4 transposed [K/32, N] pass: same as T3 (second resolver branch)
#   T5 random scales vs exact ref (per-group pow2 fold in fp64)
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


K, N = 64, 16
C1 = 0x38  # e4m3 1.0
x1 = torch.ones(1, 1, K, dtype=torch.bfloat16)
w = torch.full((N, K), C1, dtype=torch.uint8)
rp = repack_f8(w).to(dev)
xd = x1.to(dev)


def e8m0(n_rows, n_cols, fill=127):
    return torch.full((n_rows, n_cols), fill, dtype=torch.uint8
                      ).view(torch.float8_e8m0fnu)


def run(name, sc_u8, expect_note):
    sc = sc_u8.to(dev)
    out = ops.jm_fp8_gemm_w8a16_mxfp8(xd, rp, sc)[..., :N].float().cpu()
    ref = None
    if sc_u8.shape[0] == N:  # [N, KG] reference
        wf = w.view(torch.float8_e4m3fn).float().double()
        pw = torch.pow(
            2.0, sc_u8.double() - 127.0)  # uint8 -> pow2 (e8m0 decode)
        ref = torch.einsum("ng,nkg->n", pw, wf.view(N, K // 32, 32).sum(-1))
    got = out.reshape(-1)
    msg = f"{name}: jm  {got[:4].tolist()}"
    if ref is not None:
        r = ref.tolist()
        bad = (got.double() - ref).abs().max().item()
        msg = (f"{name}: jm {got[:4].tolist()} ref {r[:4]} "
               f"maxabs={bad:.4g} {'OK' if bad < 1e-3 else 'WRONG'}")
    print(msg, "|", expect_note)


# T1 uniform
run("T1 uniform 127 ", e8m0(N, K // 32), "expect 64")
# T2 one column, one group
sc = e8m0(N, K // 32)
sc[0, 0] = 128
run("T2 sc[0,g0]=2x", sc, "expect out0=96 rest 64")
# T3 whole group doubled
sc = e8m0(N, K // 32)
sc[:, 1] = 128
run("T3 g1 x2      ", sc, "expect 96 all")
# T4 transposed layout [KG, N]
sc = e8m0(N, K // 32)
sc[:, 1] = 128
run("T4 transposed ", sc.t().contiguous(), "expect 96 all (branch 2)")
# T5 random
torch.manual_seed(7)
sc = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8
                   ).view(torch.float8_e8m0fnu)
run("T5 random     ", sc, "expect ref match")
print("MXPROBE_DONE")
