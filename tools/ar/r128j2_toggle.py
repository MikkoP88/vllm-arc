# r128j2_toggle.py -- P6.34 arc7 JM g16 pass-vs-fail bisector.
# r128i3 G2 (K=1024 N=1003 M=1) PASSES rel 0.0077 on wheel build6;
# gate cell / r128j cell C (same shape, same distributions) FAIL ~1.1.
# Both weight tensors are uniform over nibble codes -> not pattern
# dependent. Remaining axes = CONSTRUCTION: CPU-built tensors moved to
# dev (r128i3) vs dev-built (gate). Toggle one axis at a time from the
# passing baseline.
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


def repack_f4(w4v):
    n, k = w4v.shape
    nt, kt = (n + 15) // 16, k // 16
    if n < nt * 16:
        w4v = torch.cat([w4v, torch.zeros(
            nt * 16 - n, k, dtype=w4v.dtype, device=w4v.device)], 0)
    v = (w4v.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


def nib_vals(w4):
    lo, hi = w4 & 0xF, w4 >> 4
    return torch.stack((lo, hi), dim=2).reshape(
        w4.shape[0], w4.shape[1] * 2)


def case(tag, K=1024, N=1003, M=1, w_dev=False, sc_dev=False, x_dev=False,
         packed=False):
    torch.manual_seed(3)
    x = torch.randn(M, K, dtype=torch.bfloat16)
    w4v = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    w8 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
    scb = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8)
    npad = ((N + 15) // 16) * 16
    if packed:  # weights from packed bytes (gate source) on same device
        src = w8.to(dev) if w_dev else w8
        w4n = nib_vals(src)
    else:
        w4n = w4v.to(dev) if w_dev else w4v
    rp = repack_f4(w4n)
    if not w_dev:
        rp = rp.to(dev)
    if sc_dev:  # gate path: pad dim 1 on dev
        sc = torch.cat(
            [scb.to(dev), torch.zeros(
                (K // 32, npad - N), dtype=torch.uint8, device=dev)], 1
        ).contiguous().view(torch.float8_e8m0fnu)
    else:  # r128i3 path: transpose-pad-transpose on CPU, then move
        sc = torch.cat([scb.t().contiguous(), torch.zeros(
            (npad - N, K // 32), dtype=torch.uint8)], 0
        ).t().contiguous().to(dev).view(torch.float8_e8m0fnu)
    xd = x.to(dev) if not x_dev else torch.randn(
        M, K, dtype=torch.bfloat16, device=dev)
    out = ops.jm_fp4_gemm_w4a16(
        xd, rp, sc, torch.bfloat16, None)[..., :N].float().cpu().double()
    w = LUT[(w4v if not packed else
             nib_vals(w8)).long()].double()
    g = torch.pow(2.0, scb.double() - 127.0)
    ref = x.double() @ (w.t() * torch.repeat_interleave(g, 32, dim=0))
    bad = ((out - ref).abs() / ref.abs().clamp_min(1.0)).max().item()
    print(f"{tag:44s} rel={bad:.4g} {'OK' if bad < 2e-2 else 'WRONG'}",
          flush=True)
    return bad


case("T0 r128i3 baseline (all CPU-built)")
case("T1 repack built ON DEV (nibble randint)")
case("T2 scales gate-path (pad dim1 on dev)")
case("T3 x born on dev")
case("T4 weights from PACKED bytes, CPU-built")
case("T5 weights packed + repack ON DEV")
case("T6 packed + dev repack + dev scales (gate)")
print("TOGGLE_DONE")
