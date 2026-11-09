# r128j3_mixmatch.py -- P6.34 arc7 JM g16 object-level bisector.
# T6 (gate construction, seed 3, scales 120..136) PASSES; r128j cell C
# (gate construction, seed 0, scales 123..131, all drawn on dev) FAILS.
# Value coverage rules out value-sensitivity -> mix tensor OBJECTS
# between a confirmed-pass set and a confirmed-fail set in ONE process:
# whichever substitution flips the verdict owns the defect. Then dump
# ptr/stride/contiguity/values of the culprit.
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


def ref_of(xc, w4n_cpu, scb_cpu):
    w = LUT[w4n_cpu.long()].double()
    g = torch.pow(2.0, scb_cpu.double() - 127.0)
    return xc.double() @ (w.t() * torch.repeat_interleave(g, 32, dim=0))


def run(tag, x, rp, sc, ref):
    out = ops.jm_fp4_gemm_w4a16(
        x, rp, sc, torch.bfloat16, None)[..., :N].float().cpu().double()
    bad = ((out - ref).abs() / ref.abs().clamp_min(1.0)).max().item()
    print(f"{tag:40s} rel={bad:.4g} {'OK' if bad < 2e-2 else 'WRONG'}",
          flush=True)
    return bad


# ---- PASS set (T6): CPU draws seed 3, moved / built on dev ----
torch.manual_seed(3)
x_p = torch.randn(M, K, dtype=torch.bfloat16).to(dev)
w4v_p = torch.randint(0, 16, (N, K), dtype=torch.uint8)
scb_p = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8)
rp_p = repack_f4(w4v_p.to(dev))
sc_p = torch.cat([scb_p.to(dev), torch.zeros(
    (K // 32, npad - N), dtype=torch.uint8, device=dev)], 1
).contiguous().view(torch.float8_e8m0fnu)
ref_p = ref_of(x_p.cpu(), w4v_p, scb_p)

# ---- FAIL set (gate cell C): dev draws seed 0 ----
torch.manual_seed(0)
x_f = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
w8_f = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
ws4_f = torch.randint(123, 131, (K // 32, N), dtype=torch.uint8,
                      device=dev)
rp_f = repack_f4(nib_vals(w8_f))
sc_f = torch.cat([ws4_f, torch.zeros(
    (K // 32, npad - N), dtype=torch.uint8, device=dev)], 1
).contiguous().view(torch.float8_e8m0fnu)
w4n_f = nib_vals(w8_f).cpu()
ref_f = ref_of(x_f.cpu(), w4n_f, ws4_f.cpu())

run("P set verbatim (x_p rp_p sc_p)", x_p, rp_p, sc_p, ref_p)
run("F set verbatim (x_f rp_f sc_f)", x_f, rp_f, sc_f, ref_f)
run("P x + P rp + F sc", x_p, rp_p, sc_f, ref_f)
run("P x + F rp + P sc", x_p, rp_f, sc_p, ref_p)
run("F x + P rp + P sc", x_f, rp_p, sc_p, ref_p)
run("P x + F rp + F sc", x_p, rp_f, sc_f, ref_f)
run("F x + P rp + F sc", x_f, rp_p, sc_f, ref_f)
run("F x + F rp + P sc", x_f, rp_f, sc_p, ref_f)

for nm, t in (("x_p", x_p), ("x_f", x_f), ("rp_p", rp_p), ("rp_f", rp_f)):
    print(f"{nm}: ptr={t.data_ptr():#x} shape={tuple(t.shape)} "
          f"stride={t.stride()} contig={t.is_contiguous()}")
for nm, t in (("sc_p", sc_p), ("sc_f", sc_f)):
    u8 = t.view(torch.uint8)
    print(f"{nm}: ptr={u8.data_ptr():#x} shape={tuple(u8.shape)} "
          f"stride={u8.stride()} contig={u8.is_contiguous()} "
          f"uniq={u8.cpu().unique().tolist()}")
# cross-check: fail-set scales substituted with PASS VALUES (same object
# construction as sc_f) -> values vs object decoupling
sc_fv = torch.cat([scb_p.to(dev), torch.zeros(
    (K // 32, npad - N), dtype=torch.uint8, device=dev)], 1
).contiguous().view(torch.float8_e8m0fnu)
run("F x + F rp + sc_f-but-P-values", x_f, rp_f, sc_fv, ref_f)
print("MIXMATCH_DONE")
