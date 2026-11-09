# r128j_gcell.py -- P6.34 arc7 JM g16 gate-cell discriminator.
# Facts banked: f64_ref g16 on-device == CPU math EXACTLY (r128j mirror,
# maxabs=0 at every intermediate incl. mk_fullrange weights). Kernel exact
# at N=16 fullrange (r128i2) and at gate shapes with RANDOM weights
# (r128i3). Gate cell (fullrange + randint scales + PADDED ws4) fails
# rel 1e2..1e3; uniform-scale variant fails 1.316 (r128i4).
# => isolate which axis flips it: weights x scales x pad, plus error
# structure (which columns, sign, ratio) on the failing cell.
# Gate functions copied VERBATIM from r128b_jm_gate.py (import would
# re-run the whole gate -- no __main__ guard).
import sys
import torch

sys.path.insert(0, "/tmp/arc7wheel")
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402

ops = torch.ops._xpu_C
dev = "xpu"
torch.manual_seed(0)


def mk_fullrange_f8(n, k):  # verbatim from r128b_jm_gate.py:45
    base = torch.arange(256, dtype=torch.uint8, device=dev)
    base[127] = 126
    base[255] = 254
    return base.repeat((n * k + 255) // 256)[: n * k].view(n, k)


def nib_vals(w4):  # verbatim :76
    lo, hi = w4 & 0xF, w4 >> 4
    return torch.stack((lo, hi), dim=2).reshape(
        w4.shape[0], w4.shape[1] * 2)


def _pad_n(t, npad, dim):  # verbatim :56
    if t.shape[dim] == npad:
        return t
    pad = torch.zeros(
        [npad - t.shape[dim] if i == dim else s
         for i, s in enumerate(t.shape)],
        dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def repack_f4(w4v):  # verbatim :81
    n, k = w4v.shape
    nt, kt = (n + 15) // 16, k // 16
    w4v = _pad_n(w4v, nt * 16, 0)
    v = (w4v.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).double()


def ref_g16(x, w4_u8, ws4):  # f64_ref g16 branch; decode on CPU (r128j
    # mirror: device chain == CPU math exactly, CPU is the trusted side)
    xf = x.double().cpu()
    k = x.shape[-1]
    w4c = w4_u8.cpu()
    lo = LUT[(w4c & 0xF).long()]
    hi = LUT[(w4c >> 4).long()]
    w = torch.stack((lo, hi), dim=1).reshape(-1, k)
    g = torch.pow(2.0, ws4.cpu().view(torch.uint8).float() - 127.0).double()
    wg = w.t() * torch.repeat_interleave(g, 32, dim=0)
    return xf @ wg


def cell(tag, k, n, m, weights, scales, pad, dump=False):
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
    if weights == "fullrange":
        w4_u8 = mk_fullrange_f8(n, k // 2)
    else:
        w4_u8 = torch.randint(
            0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    npad = ((n + 15) // 16) * 16
    if scales == "gate":
        ws4 = torch.randint(
            123, 131, (k // 32, n), dtype=torch.uint8, device=dev)
    else:
        ws4 = torch.full((k // 32, n), 127, dtype=torch.uint8, device=dev)
    rp4 = repack_f4(nib_vals(w4_u8))
    sc = (_pad_n(ws4, npad, 1) if pad else ws4).view(
        torch.float8_e8m0fnu)
    out = ops.jm_fp4_gemm_w4a16(
        x, rp4, sc, torch.bfloat16, None)[..., :n].float()
    ref = ref_g16(x, w4_u8, ws4)
    rel = ((out.double().cpu() - ref).abs().max()
           / ref.abs().max().clamp_min(1e-9)).item()
    verdict = "OK  " if rel < 0.02 else "FAIL"
    print(f"{tag:34s} {verdict} rel={rel:.4g}", flush=True)
    if dump and rel >= 0.02:
        o = out[0].double().cpu()
        r = ref[0].double().cpu()
        relcol = (o - r).abs() / r.abs().clamp_min(1e-9)
        nz = relcol > 0.05
        print(f"  cols>5% wrong: {nz.sum().item()}/{n}")
        if nz.any():
            rat = (o / r.clamp_min(1e-9))[nz]
            print(f"  ratio med {rat.median().item():.6g} "
                  f"min {rat.min().item():.6g} max {rat.max().item():.6g}")
            idx = nz.nonzero()[:8].flatten().tolist()
            for i in idx:
                print(f"  col {i}: out={o[i].item():+12.5f} "
                      f"ref={r[i].item():+12.5f}")
    return rel


K, N, M = 1024, 1003, 1
cell("A fullrange gate-sc PAD (gate verbatim)", K, N, M,
     "fullrange", "gate", True, dump=True)
cell("B fullrange unif-127 PAD", K, N, M, "fullrange", "unif", True)
cell("C random    gate-sc PAD", K, N, M, "random", "gate", True)
cell("D fullrange gate-sc NOPAD", K, N, M, "fullrange", "gate", False)
cell("E fullrange unif-127 NOPAD", K, N, M, "fullrange", "unif", False)
cell("F random    unif-127 PAD", K, N, M, "random", "unif", True)
cell("G random    gate-sc NOPAD", K, N, M, "random", "gate", False)
print("GCELL_DONE")
