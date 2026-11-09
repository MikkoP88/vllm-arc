# r128m_route_through.py -- P6.34 jm ROUTE-through correctness probe.
# Law: r128l proved the route's repack/pad/cache glue byte-parity vs the
# gate builders; this probe proves the FULL serving-side dispatch --
# _try_jm on stub layers (the exact layer forms skinny_m.py wraps:
# weight [N,K] f8 / [N,K/2] f4x2, weight_scale [1]/[1,N]/[N,K/32]) --
# against f64 dequant refs, per lane, incl. bias and the [M, N_pad] ->
# [:, :N] slice. Runs against the WHEEL UNDER TEST via PYTHONPATH
# (build7 = ping-pong kernel); M in {1,2,4,8} exercises the lane caps.
import ast
import sys

import torch

dev = "xpu"
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402

ops = torch.ops._xpu_C
MOD_SRC = "/tmp/skinny_m.py"
WANT = {"_jm_pad_n", "_jm_repack_f8", "_jm_repack_f4", "_jm_pack",
        "_try_jm", "_flatten_m", "_JM_LANE_M_MAX", "_state"}

# _try_jm needs `logger` in its namespace -- provide the probe's shim
src = open(MOD_SRC).read()
tree = ast.parse(src)
keep = []


def _name_of(node):
    if isinstance(node, ast.FunctionDef):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(
            node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name):
        return node.target.id
    return None


for node in tree.body:
    if _name_of(node) in WANT:
        keep.append(ast.get_source_segment(src, node))
assert len(keep) == len(WANT), f"extracted {len(keep)}/{len(WANT)}"


class _Log:
    def info(self, *a, **k):
        print("[route]", *a, **k)

    def warning(self, *a, **k):
        print("[route WARN]", *a, **k)


ns = {"torch": torch, "logger": _Log()}
exec(compile("\n\n".join(keep), MOD_SRC, "exec"), ns)
try_jm = ns["_try_jm"]
M_MAX = ns["_JM_LANE_M_MAX"]

LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64, device=dev)


class Stub(torch.nn.Module):
    pass


def mk_fullrange_f8(n, k):
    base = torch.arange(256, dtype=torch.uint8, device=dev)
    base[127] = 126
    base[255] = 254
    return base.repeat((n * k + 255) // 256)[: n * k].view(n, k)


CASES = [(1024, 1003), (5376, 10752), (5120, 8704), (96, 517)]
MS = [1, 2, 4, 8]
fails = 0
cells = 0
for (k, n) in CASES:
    torch.manual_seed(0)
    x = torch.randn(8, k, dtype=torch.bfloat16, device=dev)
    w8_u8 = mk_fullrange_f8(n, k)
    w4_u8 = mk_fullrange_f8(n, k // 2)
    ws = torch.randint(123, 131, (n, k // 32), dtype=torch.uint8,
                       device=dev).view(torch.float8_e8m0fnu)
    w_pc = torch.rand(1, n, device=dev) * 0.01 + 0.005
    w_pt = torch.rand(1, device=dev) * 0.01 + 0.005
    bias = torch.randn(n, dtype=torch.float32, device=dev) * 0.1

    # f64 refs (route-lane semantics, bias added)
    w8f = w8_u8.view(torch.float8_e4m3fn).double()
    g = torch.pow(2.0, ws.view(torch.uint8).float() - 127.0).double()
    lo = LUT[(w4_u8 & 0xF).long()]
    hi = LUT[(w4_u8 >> 4).long()]
    w4k = torch.stack((lo, hi), dim=2).reshape(n, k).double()
    wg16 = w4k.t() * torch.repeat_interleave(g.t(), 32, dim=0)
    refs = {
        "w16pt": (lambda xf: xf @ w8f.t() * w_pt.double()),
        "w16pc": (lambda xf: xf @ w8f.t() * w_pc.double()),
        "m816": (lambda xf: xf @ (
            w8f.t() * torch.repeat_interleave(g.t(), 32, dim=0))),
        "g16": (lambda xf: xf @ wg16),
    }

    for m in MS:
        xm = x[:m]
        xf = xm.double()
        for lane in ("w16pt", "w16pc", "m816", "g16"):
            st = Stub()
            if lane == "g16":
                st.weight = w4_u8.view(torch.float4_e2m1fn_x2)
                st.weight_scale = ws
            elif lane == "m816":
                st.weight = w8_u8.view(torch.float8_e4m3fn)
                st.weight_scale = ws
            elif lane == "w16pc":
                st.weight = w8_u8.view(torch.float8_e4m3fn)
                st.weight_scale = w_pc
            else:
                st.weight = w8_u8.view(torch.float8_e4m3fn)
                st.weight_scale = w_pt
            exp_lane = ("w16" if lane.startswith("w16") else
                        "m816" if lane == "m816" else "g16")
            cap_ok = m <= M_MAX[exp_lane]
            out = try_jm(st, xm, bias)
            ref = refs[lane](xf) + bias.double()
            if not cap_ok:
                # over-cap M must return None (defer to wrapped route)
                ok = out is None
                print(f"{k}x{n} M={m} {lane}: over-cap defer "
                      f"{'OK' if ok else 'FAIL(got tensor)'}")
                if not ok:
                    fails += 1
                continue
            cells += 1
            if out is None:
                print(f"FAIL {k}x{n} M={m} {lane}: None (route refused)")
                fails += 1
                continue
            err = (out.float().double() - ref).abs()
            rel = (err / ref.abs().clamp_min(1.0)).max().item()
            ok = (out.shape == (m, n) and rel < 3e-2
                  and torch.isfinite(out.float()).all())
            if not ok:
                fails += 1
            print(f"{k}x{n} M={m} {lane}: rel={rel:.3g} "
                  f"shape={tuple(out.shape)} {'ok' if ok else 'FAIL'}",
                  flush=True)

print(f"cells={cells} fails={fails}")
print("ROUTE_THROUGH:", "PASS" if fails == 0 else "FAIL")
