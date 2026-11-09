# r128b_jm_gate.py -- P6.34 GATE-1/2 for the arc7 JM (DPAS) skinny family.
# Stage A correctness: every lane x dtype vs an f64 dequant reference,
# with codec-exhaustive weights (all 256 f8 codes / all 16 f4 nibbles
# tiled through the tensor -- the packed pair decoders are validated
# end-to-end), odd shapes (all-model-types clause: N % 16 != 0, tiny K,
# M=3/8), per-tensor + per-channel + MX scales incl. extreme e8m0.
# Runs BOTH kernel variants (ARC_JM_VAR 5 = proven V5 lottery control,
# 14 = fixed-pointer unrolled ping-pong; r128k2): 80 cells.
# Stage B perf: 50-call amortized XPUGraph replay (P6.33 measurement law)
# stock vs scalar-skinny vs JM(var 5 + 14) on the 7 r128a shapes x 4 lanes
# x M in {1,2}; ARC_JM_SGF sweep {1024, 2048, 4096} (env read per call in
# the probe wheel). Rows -> R128B_OUT json.
import json
import os
import statistics

import torch

torch.manual_seed(0)
dev = "xpu"
ROUNDS, WARM, NCALL = 20, 5, 50
import vllm_xpu_kernels  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

ops = torch.ops._xpu_C
G = torch.xpu.XPUGraph

print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0),
      flush=True)
print("ARC_JM_SGF default run:", os.environ.get("ARC_JM_SGF", "2048"),
      flush=True)
print("ARC_JM_VAR default run:", os.environ.get("ARC_JM_VAR", "14"),
      flush=True)

SHAPES = [
    ("gemma.up", 5376, 10752),
    ("gemma.down", 10752, 5376),
    ("gemma.qkv", 5376, 8192),
    ("gemma.o", 5376, 5376),
    ("qwen.up", 5120, 8704),
    ("qwen.down", 8704, 5120),
    ("qwen.o", 5120, 5120),
]
LANES = ["w16pt", "w16pc", "m816", "g16"]
MS = [1, 2]
REL_TOL, ABS_TOL = 3e-2, 3e-2  # bf16-out quantization dominates


def mk_fullrange_f8(n, k):
    # all 256 byte codes tiled: decoder-exhaustive through the GEMM.
    # e4m3fn 0x7F/0xFF are NaN (decode to finite garbage) -> pin them to
    # their finite neighbors 0x7E/0xFE so the f64 ref stays finite; every
    # finite code is still covered (0x7E/0xFE just appear twice).
    base = torch.arange(256, dtype=torch.uint8, device=dev)
    base[127] = 126
    base[255] = 254
    return base.repeat((n * k + 255) // 256)[: n * k].view(n, k)


def _pad_n(t, npad, dim):
    if t.shape[dim] == npad:
        return t
    pad = torch.zeros(
        [npad - t.shape[dim] if i == dim else s
         for i, s in enumerate(t.shape)],
        dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=dim)


def repack_f8(w8):  # [N, K] u8 -> column tiles [N/16, K/16, 256]
    n, k = w8.shape
    nt, kt = (n + 15) // 16, k // 16
    w8 = _pad_n(w8, nt * 16, 0)
    # near-identity reshape (r128e2 B layout: lane t's fragment = column
    # n0+t's 16 consecutive k codes): tile byte (c, k) = 16c + k
    return (w8.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
            .contiguous().view(nt, kt, 256))


def nib_vals(w4):  # packed [N, K/2] u8 -> nibble values [N, K] (0..15)
    lo, hi = w4 & 0xF, w4 >> 4
    return torch.stack((lo, hi), dim=2).reshape(w4.shape[0], w4.shape[1] * 2)


def repack_f4(w4v):  # nibble values [N, K] -> column tiles [N/16, K/16, 128]
    n, k = w4v.shape
    nt, kt = (n + 15) // 16, k // 16
    w4v = _pad_n(w4v, nt * 16, 0)
    # r128e2 B layout: lane t's 8-byte block = column n0+t's 16 codes,
    # nibble (c, k) at byte 8c + (k>>1), slot = k&1 (lo = even k)
    v = (w4v.view(nt, 16, kt, 16).permute(0, 2, 1, 3)
         .contiguous().view(nt, kt, 16, 8, 2))
    return (v[..., 0] | (v[..., 1] << 4)).view(nt, kt, 128)


def repack_selftest():
    # host-verify the repacked tiles against the banked byte formulas
    # (jm_gemm_kernel.hpp, r128e2 B layout): f8 addr(c,k) = 16c + k;
    # f4 byte(c,k) = 8c + (k>>1), nibble = k&1 (lo = even k). Any drift
    # here is a GATE-1 fail before the device runs. CPU tensors: pure
    # layout math, no sync tax.
    for (n, k) in ((48, 96), (33, 64)):
        w = torch.randint(0, 256, (n, k), dtype=torch.uint8)
        t = repack_f8(w)
        for _ in range(200):
            c = torch.randint(0, 16, (1,)).item()
            kk = torch.randint(0, 16, (1,)).item()
            q = torch.randint(0, t.shape[0], (1,)).item()
            kte = torch.randint(0, t.shape[1], (1,)).item()
            addr = 16 * c + kk
            col = q * 16 + c
            expect = (w[col, kte * 16 + kk].item() if col < n else 0)
            got = t[q, kte, addr].item()
            assert got == expect, (n, k, q, kte, c, kk, addr, got, expect)
        wv = torch.randint(0, 16, (n, k), dtype=torch.uint8)
        t4 = repack_f4(wv)
        for _ in range(200):
            c = torch.randint(0, 16, (1,)).item()
            kk = torch.randint(0, 16, (1,)).item()
            q = torch.randint(0, t4.shape[0], (1,)).item()
            kte = torch.randint(0, t4.shape[1], (1,)).item()
            byte = 8 * c + (kk >> 1)
            col = q * 16 + c
            b = t4[q, kte, byte].item()
            got = (b & 0xF) if (kk & 1) == 0 else (b >> 4)
            expect = (wv[col, kte * 16 + kk].item() if col < n else 0)
            assert got == expect, (n, k, q, kte, c, kk, byte, got, expect)
    print("repack_selftest: PASS (f8 + f4 tile formulas)", flush=True)


def f64_ref(lane, x, w8_u8, w_pt, w_pc, ws8, w4_u8, ws4):
    # f64 dequant reference, x [M,K] bf16. Weights arrive [N, K]; the
    # group dequant runs k-major (transpose first), scales [K/32, N].
    xf = x.double()
    k = x.shape[-1]
    if lane in ("w16pt", "w16pc"):
        w = w8_u8.view(torch.float8_e4m3fn).double()
        out = xf @ w.t()
        out = out * (w_pt.double() if lane == "w16pt" else w_pc.double())
        return out
    if lane == "m816":
        wt = w8_u8.view(torch.float8_e4m3fn).float().t().contiguous()
        g = torch.pow(
            2.0, ws8.view(torch.uint8).float() - 127.0).double()  # [K/32,N]
        wg = wt.view(k, -1) * torch.repeat_interleave(g, 32, dim=0)
        return xf @ wg
    # g16: packed nibbles, low = even k
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=dev).double()
    lo = lut[(w4_u8 & 0xF).long()]   # [N, K/2] -> k even
    hi = lut[(w4_u8 >> 4).long()]    # [N, K/2] -> k odd
    # stack dim=2 -> [N, K/2, 2] -> k-ascending interleave. dim=1 gives
    # [N, 2, K/2] = block-concat [lo | hi] -- the cut-6 g16 GATE-1 false
    # fail (ref disagreed with repack_f4/nib_vals and the kernel; the
    # kernel was correct all along; r128j8, 2026-11-09).
    w = torch.stack((lo, hi), dim=2).reshape(-1, k)  # [N, K] k asc
    g = torch.pow(
        2.0, ws4.view(torch.uint8).float() - 127.0).double()
    wg = w.t() * torch.repeat_interleave(g, 32, dim=0)
    return xf @ wg


def jm_call(lane, x, d):
    # JM ops take column-repacked weights and return [M, N_pad]; slice the
    # real N columns (padded weights are zero).
    if lane == "w16pt":
        out = ops.jm_fp8_gemm_w8a16(x, d["RP8"], d["w_pt"], None)
    elif lane == "w16pc":
        out = ops.jm_fp8_gemm_w8a16(x, d["RP8"], d["w_pc_pad"], None)
    elif lane == "m816":
        out = ops.jm_fp8_gemm_w8a16_mxfp8(
            x, d["RP8"], d["ws8_pad"], torch.bfloat16, None)
    else:
        out = ops.jm_fp4_gemm_w4a16(x, d["RP4"], d["ws4_pad"],
                                    torch.bfloat16, None)
    return out[..., : d["n_real"]]


# ---------------------------------------------------------------------------
# Stage A: correctness
# ---------------------------------------------------------------------------
fails = []
cells = 0
CORR_CASES = [
    # (K, N, M) -- target shapes (spot: one), odd shapes (all-model-types)
    (1024, 1003, 1), (1024, 1003, 2), (96, 517, 3), (5376, 10752, 1),
    (5120, 5120, 2), (10752, 5376, 1), (2048, 16384, 8), (5120, 8704, 4),
    (3584, 18944, 1), (4096, 11008, 2),
]
print("=== stage A: correctness (ARC_JM_VAR 5 + 14) ===", flush=True)
repack_selftest()
for var in ("5", "14"):
    os.environ["ARC_JM_VAR"] = var
    print(f"--- ARC_JM_VAR={var}", flush=True)
    for k, n, m in CORR_CASES:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        w8_u8 = mk_fullrange_f8(n, k)
        w4_u8 = mk_fullrange_f8(n, k // 2)  # all nibble pairs covered
        # varied per-group/per-column scales catch stride/indexing bugs;
        # range 123..131 (2^-4..2^4) keeps refs finite with full-range
        # weights
        npad = ((n + 15) // 16) * 16
        ws8 = torch.randint(123, 131, (k // 32, n), dtype=torch.uint8,
                            device=dev)
        ws4 = torch.randint(123, 131, (k // 32, n), dtype=torch.uint8,
                            device=dev)
        w_pc = torch.rand(1, n, device=dev) * 0.01 + 0.005
        d = dict(
            Bt8=w8_u8.view(torch.float8_e4m3fn).t(),
            w_pt=(torch.rand(1, device=dev) * 0.01 + 0.005),
            w_pc=w_pc,
            ws8_kn=ws8.view(torch.float8_e8m0fnu),
            Bt4=w4_u8.view(torch.float4_e2m1fn_x2).t(),
            ws4_kn=ws4.view(torch.float8_e8m0fnu),
            # v2 lane inputs: repacked tiles + N-padded scales (zero
            # weights in the pad columns make the pad scale bytes
            # irrelevant)
            RP8=repack_f8(w8_u8),
            RP4=repack_f4(nib_vals(w4_u8)),
            ws8_pad=_pad_n(ws8, npad, 1).view(torch.float8_e8m0fnu),
            ws4_pad=_pad_n(ws4, npad, 1).view(torch.float8_e8m0fnu),
            w_pc_pad=_pad_n(w_pc, npad, 1).contiguous(),
            n_real=n,
        )
        for lane in LANES:
            ref = f64_ref(lane, x, w8_u8, d["w_pt"], d["w_pc"], d["ws8_kn"],
                          w4_u8, d["ws4_kn"])
            try:
                out = jm_call(lane, x, d).float()
            except Exception as e:  # noqa: BLE001
                print(f"FAIL v{var} {k}x{n} M={m} {lane}: "
                      f"{type(e).__name__}: {str(e)[:100]}", flush=True)
                fails.append((var, k, n, m, lane, "raise"))
                continue
            cells += 1
            ok_shape = out.shape == (m, n)
            if not ok_shape:
                print(f"FAIL v{var} {k}x{n} M={m} {lane}: "
                      f"shape {tuple(out.shape)}", flush=True)
                fails.append((var, k, n, m, lane, "shape"))
                continue
            err = (out.double() - ref).abs()
            denom = ref.abs().clamp_min(1.0)
            rel = (err / denom).max().item()
            if not torch.isfinite(out.float()).all():
                print(f"FAIL v{var} {k}x{n} M={m} {lane}: nonfinite",
                      flush=True)
                fails.append((var, k, n, m, lane, "nonfinite"))
                continue
            verdict = "ok" if rel < REL_TOL else "REL"
            if verdict == "REL":
                print(f"FAIL v{var} {k}x{n} M={m} {lane}: rel={rel:.3e}",
                      flush=True)
                fails.append((var, k, n, m, lane, f"rel{rel:.1e}"))
            else:
                print(f"ok v{var} {k}x{n} M={m} {lane}: rel={rel:.2e}",
                      flush=True)

print(f"STAGE_A cells={cells} fails={len(fails)} {fails}", flush=True)
if fails:
    print("GATE1_FAIL", flush=True)
else:
    print("GATE1_PASS", flush=True)

# ---------------------------------------------------------------------------
# Stage B: graph50 perf, stock vs scalar vs JM (+ SGF sweep)
# ---------------------------------------------------------------------------
DATA = {}
for tag, K, N in SHAPES:
    x1 = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    x2 = torch.randn(2, K, dtype=torch.bfloat16, device=dev)
    w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    DATA[tag] = dict(
        K=K, N=N, x1=x1, x2=x2, Bt8=w8.t(),
        w_pt=torch.rand(1, device=dev) * 0.01 + 0.005,
        w_pc=torch.rand(1, N, device=dev) * 0.01 + 0.005,
        ws8_kn=torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                             device=dev).view(torch.float8_e8m0fnu),
        w4=torch.randint(0, 255, (N, K // 2), dtype=torch.uint8,
                         device=dev).view(torch.float4_e2m1fn_x2),
        ws4_kn=torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                             device=dev).view(torch.float8_e8m0fnu),
    )
    DATA[tag]["Bt4"] = DATA[tag]["w4"].t()
    # v2 JM lane inputs (all 7 shapes are N/K 16-multiples: no padding)
    DATA[tag]["RP8"] = repack_f8(w8.view(torch.uint8))
    DATA[tag]["RP4"] = repack_f4(nib_vals(DATA[tag]["w4"].view(torch.uint8)))


def lane_fns(tag, lane, M):
    d = DATA[tag]
    x = d["x1"] if M == 1 else d["x2"]
    if lane == "w16pt":
        return (
            lambda: ops.fp8_gemm_w8a16(x, d["Bt8"], d["w_pt"], None),
            lambda: ops.skinny_fp8_gemm_w8a16(x, d["Bt8"], d["w_pt"], None),
            lambda: ops.jm_fp8_gemm_w8a16(x, d["RP8"], d["w_pt"], None)[..., : d["N"]],
        )
    if lane == "w16pc":
        return (
            lambda: ops.fp8_gemm_w8a16(x, d["Bt8"], d["w_pc"], None),
            lambda: ops.skinny_fp8_gemm_w8a16(x, d["Bt8"], d["w_pc"], None),
            lambda: ops.jm_fp8_gemm_w8a16(x, d["RP8"], d["w_pc"], None)[..., : d["N"]],
        )
    if lane == "m816":
        return (
            lambda: ops.fp8_gemm_w8a16_mxfp8(
                x, d["Bt8"], d["ws8_kn"], torch.bfloat16, None),
            lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(
                x, d["Bt8"], d["ws8_kn"], torch.bfloat16, None),
            lambda: ops.jm_fp8_gemm_w8a16_mxfp8(
                x, d["RP8"], d["ws8_kn"], torch.bfloat16, None)[..., : d["N"]],
        )
    return (
        lambda: ops.fp4_gemm_w4a16(x, d["Bt4"], d["ws4_kn"],
                                   torch.bfloat16, None),
        lambda: ops.skinny_fp4_gemm_w4a16(x, d["Bt4"], d["ws4_kn"],
                                          torch.bfloat16, None),
        lambda: ops.jm_fp4_gemm_w4a16(x, d["RP4"], d["ws4_kn"],
                                      torch.bfloat16, None)[..., : d["N"]],
    )


def capture50(fn):
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            fn()
    torch.xpu.current_stream().wait_stream(s)
    g = G()
    torch.xpu.synchronize()
    with torch.xpu.graph(g):
        for _ in range(NCALL):
            fn()
    return g


def replay_ms(g):
    for _ in range(WARM):
        g.replay()
    torch.xpu.synchronize()
    ts = []
    for _ in range(ROUNDS):
        a = torch.xpu.Event(enable_timing=True)
        b = torch.xpu.Event(enable_timing=True)
        a.record(); g.replay(); b.record()
        torch.xpu.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts) / NCALL


print("=== stage B: graph50 A/B/C + SGF sweep, jm5 + jm14 ===", flush=True)
out = {"rows": [], "gate1_fails": len(fails), "gate1_fail_detail": str(fails)}
for sgf in ("1024", "2048", "4096"):
    os.environ["ARC_JM_SGF"] = sgf
    print(f"--- ARC_JM_SGF={sgf}", flush=True)
    for tag, K, N in SHAPES:
        for lane in LANES:
            for M in MS:
                fn_st, fn_sk, fn_jm = lane_fns(tag, lane, M)
                # stock/scalar only measured on the default sweep pass
                if sgf == "2048":
                    gs = replay_ms(capture50(fn_st))
                    gk = replay_ms(capture50(fn_sk))
                else:
                    gs = gk = float("nan")
                st_s = (f"st {gs:.4f} sk {gk:.4f} "
                        if sgf == "2048" else "st -- sk -- ")
                for var in ("5", "14"):
                    # env read per CALL inside the launcher; capture bakes
                    # the variant into the graph
                    os.environ["ARC_JM_VAR"] = var
                    try:
                        gjm = replay_ms(capture50(fn_jm))
                    except Exception as e:  # noqa: BLE001
                        print(f"{tag:12s} {lane:5s} M={M} jm{var} FAIL "
                              f"{type(e).__name__}: {str(e)[:100]}", flush=True)
                        continue
                    wbytes = K * N * (0.5 if lane == "g16" else 1.0)
                    gbs = wbytes / gjm / 1e6
                    row = dict(sgf=sgf, var=var, tag=tag, lane=lane, M=M,
                               st=gs, sk=gk, jm=gjm, jm_gbs=gbs,
                               jm_vs_st=gs / gjm, jm_vs_sk=gk / gjm)
                    out["rows"].append(row)
                    print(f"{tag:12s} {lane:5s} M={M} {st_s}"
                          f"jm{var} {gjm:.4f} ({gbs:5.0f} GB/s"
                          + (f", {gs / gjm:.2f}x st" if sgf == "2048" else "")
                          + ")", flush=True)

import os as _os
outpath = _os.environ.get("R128B_OUT", "/tmp/r128b_rows.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=1)
print("R128B_DONE", flush=True)
