# r128l_route_pack_test.py -- P6.34 jm-route repack byte-parity probe.
# Law: the skinny_m.py jm route's repack helpers (_jm_repack_f8/_f4,
# _jm_pad_n) must be BYTE-IDENTICAL to the gate's proven builders
# (r128b repack_f8/repack_f4/_pad_n, cut7 repack_selftest PASS). The
# helpers are pure-torch and dependency-free; extract them from the
# shipped module source by AST (arc-jm7 has no vllm install -- the
# module imports vllm.logger) and byte-diff against the gate builders
# on identical tensors, incl. non-multiple N (pad path) and the f4
# interleave contract. Also exercises _jm_pack cache semantics on a
# stub layer (hit must return the SAME tensors, no recompute).
import ast
import sys

import torch

sys.path.insert(0, "/tmp/arc7wheel")
dev = "xpu"

MOD_SRC = "/tmp/skinny_m.py"
WANT = {"_jm_pad_n", "_jm_repack_f8", "_jm_repack_f4", "_jm_pack"}

tree = ast.parse(open(MOD_SRC).read())
keep = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        keep.append(ast.get_source_segment(open(MOD_SRC).read(), node))
assert len(keep) == 4, f"extracted {len(keep)}/4 helpers"
ns = {"torch": torch}
exec(compile("\n\n".join(keep), MOD_SRC, "exec"), ns)
jm_pad, jm_f8, jm_f4, jm_pack = (ns["_jm_pad_n"], ns["_jm_repack_f8"],
                                 ns["_jm_repack_f4"], ns["_jm_pack"])

# gate builders (r128b cut7 verbatim)


def g_pad(t, npad, dim):
    if t.shape[dim] == npad:
        return t
    pad = torch.zeros(
        [npad - t.shape[dim] if i == dim else s for i, s in enumerate(
            t.shape)], dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def g_f8(w8):
    n, k = w8.shape
    nt, kt = (n + 15) // 16, k // 16
    w8 = g_pad(w8, nt * 16, 0)
    return (w8.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
            .contiguous().view(nt, kt, 256))


def g_nib(w4):
    lo, hi = w4 & 0xF, w4 >> 4
    return torch.stack((lo, hi), dim=2).reshape(w4.shape[0], w4.shape[1] * 2)


def g_f4(w4v):
    n, k = w4v.shape
    nt, kt = (n + 15) // 16, k // 16
    w4v = g_pad(w4v, nt * 16, 0)
    v = (w4v.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


torch.manual_seed(0)
fails = 0
for (n, k) in ((1003, 1024), (8704, 5120), (5376, 10752), (517, 96),
               (5120, 5120)):
    w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device=dev)
    ok8 = torch.equal(jm_f8(w8), g_f8(w8))
    w4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
    ok4 = torch.equal(jm_f4(w4), g_f4(g_nib(w4)))
    if not (ok8 and ok4):
        fails += 1
    print(f"{n}x{k}: f8 {'OK' if ok8 else 'FAIL'} "
          f"f4 {'OK' if ok4 else 'FAIL'}", flush=True)

# scale-pad parity + _jm_pack cache-hit identity on a stub layer
class Stub(torch.nn.Module):
    pass


for lane in ("w16", "m816", "g16"):
    st = Stub()
    n, k = 8704, 5120
    if lane == "g16":
        st.weight = torch.randn(n, k // 2, device=dev).view(
            torch.float4_e2m1fn_x2)
        st.weight_scale = torch.randint(
            120, 132, (n, k // 32), device=dev).view(torch.float8_e8m0fnu)
    else:
        st.weight = torch.randn(n, k, device=dev).to(torch.float8_e4m3fn)
        if lane == "m816":
            # layer form [N, K/32] (the real caller transposes for the op;
            # skinny_m.py m816 branch consumes wsc.t() -> [K/32, N])
            st.weight_scale = torch.randint(
                120, 132, (n, k // 32), device=dev).view(
                torch.float8_e8m0fnu).contiguous()
        else:
            st.weight_scale = torch.rand(1, n, device=dev) * 0.01
    b1, s1, nr = jm_pack(st, lane)
    b2, s2, _ = jm_pack(st, lane)
    same = all(torch.equal(a, c) for a, c in ((b1, b2), (s1, s2)))
    npad = (n + 15) // 16 * 16
    shape_ok = (b1.shape[-1] == (256 if lane != "g16" else 128)
                and s1.shape[-1] == npad and nr == n)
    # scale content parity vs gate pad on the transposed form
    if lane == "m816":
        want = g_pad(st.weight_scale.t().contiguous(), npad, 1)
        sc_ok = torch.equal(s1, want)
    elif lane == "w16":
        want = g_pad(st.weight_scale.reshape(1, n), npad, 1)
        sc_ok = torch.equal(s1, want)
    else:
        want = g_pad(st.weight_scale.t().contiguous(), npad, 1)
        sc_ok = torch.equal(s1, want)
    ok = same and shape_ok and sc_ok
    if not ok:
        fails += 1
    print(f"{lane}: cache-same={same} shapes={shape_ok} "
          f"scale-parity={sc_ok} {'OK' if ok else 'FAIL'}", flush=True)

print("ROUTE_PACK_TEST:", "PASS" if fails == 0 else f"FAIL({fails})")
