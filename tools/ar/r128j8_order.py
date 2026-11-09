# r128j8_order.py -- P6.34 arc7 JM g16 ref-order discriminator.
# Laws: kernel out == r128j7 CPU ref (rel 0.0077, out = -836 family);
# gcell (identical math/bytes, ref computed AFTER the op call, no CPU
# copies made before it) reports rel 1.758 with ref col0 = -61.2. Both
# refs are pure-CPU f64 on supposedly identical bytes. A/B the only
# order difference: op-call-first (gcell) vs copies-first (r128j7).
# Byte-hash every D2H copy in both arms.
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


def build():
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    base = torch.arange(256, dtype=torch.uint8, device=dev)
    base[127] = 126
    base[255] = 254
    w4_u8 = base.repeat((N * (K // 2) + 255) // 256)[
        : N * (K // 2)].view(N, K // 2)
    ws4 = torch.randint(123, 131, (K // 32, N),
                        dtype=torch.uint8, device=dev)
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
    return x, w4_u8, ws4, rp4, sc


def ref_from_bytes(xb, w4b, wsb):
    lo = LUT[(w4b & 0xF).long()]
    hi = LUT[(w4b >> 4).long()]
    w = torch.stack((lo, hi), dim=1).reshape(-1, xb.shape[-1])
    g = torch.pow(2.0, wsb.float() - 127.0).double()
    wg = w.t() * torch.repeat_interleave(g, 32, dim=0)
    return xb.double() @ wg


def h(t):
    return hashlib_used(t.cpu().contiguous())


def hashlib_used(t):
    import hashlib
    return hashlib.md5(t.numpy().tobytes()).hexdigest()[:12]


def kern(xv, rpv, scv):
    return ops.jm_fp4_gemm_w4a16(
        xv, rpv, scv, torch.bfloat16, None)[..., :N].float().cpu().double()


def rel(out, ref):
    return ((out - ref).abs() / ref.abs().clamp_min(1.0)).max().item()


# ---- ARM 1: gcell order (op FIRST, then copy bytes for the ref) ----
x, w4_u8, ws4, rp4, sc = build()
out1 = kern(x, rp4, sc)
xb1, w4b1, wsb1 = x.cpu(), w4_u8.cpu(), ws4.cpu()
ref1 = ref_from_bytes(xb1, w4b1, wsb1)
print(f"ARM1 op-first : rel={rel(out1, ref1):.4g} "
      f"out[:3]={out1[0, :3].tolist()} ref[:3]={ref1[0, :3].tolist()}")

# ---- ARM 2: copies FIRST, then op ----
x, w4_u8, ws4, rp4, sc = build()
xb2, w4b2, wsb2 = x.cpu(), w4_u8.cpu(), ws4.cpu()
ref2 = ref_from_bytes(xb2, w4b2, wsb2)
out2 = kern(x, rp4, sc)
print(f"ARM2 copy-first: rel={rel(out2, ref2):.4g} "
      f"out[:3]={out2[0, :3].tolist()} ref[:3]={ref2[0, :3].tolist()}")

# ---- byte identity across arms and across REPEAT copies ----
x3, w4_u83, ws43, _, _ = build()
xb3, w4b3, wsb3 = x3.cpu(), w4_u83.cpu(), ws43.cpu()
print(f"hashes x : {h(x)} {h(x3)} | w4: {h(w4_u8)} {h(w4_u83)} "
      f"| ws4: {h(ws4)} {h(ws43)}")
print(f"arm byte-equal x={torch.equal(xb1, xb2)}/{torch.equal(xb1, xb3)} "
      f"w4={torch.equal(w4b1, w4b2)}/{torch.equal(w4b1, w4b3)} "
      f"ws={torch.equal(wsb1, wsb2)}/{torch.equal(wsb1, wsb3)}")
# re-copy AFTER more GPU work: same bytes?
out3 = kern(x, rp4, sc)
xb4, w4b4, wsb4 = x.cpu(), w4_u8.cpu(), ws4.cpu()
print(f"re-copy after ops equal: x={torch.equal(xb4, xb1)} "
      f"w4={torch.equal(w4b4, w4b1)} ws={torch.equal(wsb4, wsb1)}")
ref4 = ref_from_bytes(xb4, w4b4, wsb4)
print(f"ARM1 re-ref rel vs out1 = {rel(out1, ref4):.4g}")
print("ORDER_DONE")
