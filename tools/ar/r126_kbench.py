# r126_kbench.py -- P6.33 arc6 skinny-M kernel gate + W4A8 dispatch answer.
#
# Runs INSIDE the arc6test image (arc6 wheel loaded), GPU legs (the
# window script stops W14 first). For the six real layer shapes x
# M in {1, 2, 4, 8} (the skinny regime; the python route sends exactly
# these to the skinny kernels) and {4096, 32768} (large-M stock
# controls):
#
#   skinny_w16pt / skinny_w16pc -- NEW skinny_fp8_gemm_w8a16 on the exact
#       tensors the stock champion call uses (Bt8 [K, N] Fortran view,
#       fp32 scalar / [1, N] scale)
#   skinny_m816 -- NEW skinny_fp8_gemm_w8a16_mxfp8 on the exact p631
#       reroute tensors (Bt8, ws8_kn [K/32, N] contiguous e8m0)
#   skinny_g16 -- NEW skinny_fp4_gemm_w4a16 on the exact p630 tensors
#       (Bt packed f4x2 [K/2, N] Fortran view, ws_kn [K/32, N] e8m0)
#   stock controls -- fp8_gemm_w8a16 (w16pt/w16pc), fp8_gemm_w8a16_mxfp8,
#       fp4_gemm_w4a16 on the SAME tensors (identical-input A/B)
#   w4a8 probe -- fp4_gemm_w4a8 with the arc6 dispatch arms: does the
#       (e4m3, f4) joint now REACH oneDNN (md accept vs wrapper throw)?
#       The r123 "oneDNN refusal" was our own dispatch default (P6.32
#       item 3 correction) -- this cell banks the arc6 answer.
#
# v2 methodology (window7): INTERLEAVED A/B timing. The window5/6 runs
# benched each stock block then each skinny block sequentially, and the
# stock medians swung 30-40% BETWEEN RUNS (gemma.up m816 0.2955 -> 0.1873,
# qwen.o w16pt 0.3014 -> 0.1753) -- slow clock/thermal drift across the
# ~10-min bench decided GATE-2 cells, not the kernels. Now every skinny
# cell times stock and skinny back-to-back per ITERATION (same in-order
# queue, adjacent thermal state) and the gate reads the paired medians.
#
# Correctness: every skinny cell vs the same torch dequant reference the
# r123 lanes used (e8m0 depow, e2m1 LUT, f8 -> f32 bit decode); rel-err
# gate 2e-2 (bf16 out rounding dominates; expected ~1e-3).
import json
import statistics
import sys

import torch

torch.manual_seed(0)
dev = "xpu"
ITERS, WARM = 50, 15

SHAPES = [
    ("gemma.up", 5376, 10752),
    ("gemma.down", 10752, 5376),
    ("gemma.o", 5376, 5376),
    ("qwen.up", 5120, 8704),
    ("qwen.down", 8704, 5120),
    ("qwen.o", 5120, 5120),
]
MS_SKINNY = [1, 2, 4, 8]
MS_BIG = [4096, 32768]

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2, dtype=torch.float32
)
_E2M1[8:] *= -1.0

E4M3_MAX = 448.0
REL_GATE = 2e-2
GB_BAR = 197.0  # directive bar: stock oneDNN reference level (qwen.up w16pc)


def bench(fn):
    for _ in range(WARM):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(ITERS):
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.xpu.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def ab_bench(fn_a, fn_b):
    """Interleaved paired timing: a and b alternate per ITERATION on the
    same in-order queue, so slow drift (clocks/thermal) hits both sides of
    every pair. Returns (median_a, median_b)."""
    for _ in range(WARM):
        fn_a()
        fn_b()
    torch.xpu.synchronize()
    ta, tb = [], []
    for _ in range(ITERS):
        s1 = torch.xpu.Event(enable_timing=True)
        e1 = torch.xpu.Event(enable_timing=True)
        s2 = torch.xpu.Event(enable_timing=True)
        e2 = torch.xpu.Event(enable_timing=True)
        s1.record()
        fn_a()
        e1.record()
        s2.record()
        fn_b()
        e2.record()
        torch.xpu.synchronize()
        ta.append(s1.elapsed_time(e1))
        tb.append(s2.elapsed_time(e2))
    return statistics.median(ta), statistics.median(tb)


def e8m0_depow(sc_u8):
    """e8m0 scale bytes -> f32 powers of two."""
    return torch.pow(2.0, sc_u8.view(torch.uint8).to(torch.float32) - 127.0)


def qtok(x):
    """per-token (row) e4m3 dynamic quant."""
    a = x.detach().abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-6)
    a = a / E4M3_MAX
    return (
        (x.float() / a).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn),
        a,
    )


def _rel(y, ref):
    return float((y.float() - ref).norm() / ref.norm())


def nbytes(lane, K, N):
    return K * N if lane != "g16" else K * N // 2


print("=== [0] env ===")
print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0))
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402  (dlopen law, r120b)

ops = torch.ops._xpu_C
for op_name in (
    "skinny_fp8_gemm_w8a16",
    "skinny_fp8_gemm_w8a16_mxfp8",
    "skinny_fp4_gemm_w4a16",
    "fp8_gemm_w8a16",
    "fp8_gemm_w8a16_mxfp8",
    "fp4_gemm_w4a16",
    "fp4_gemm_w4a8",
):
    if not hasattr(ops, op_name):
        print(f"FATAL: _xpu_C.{op_name} absent from the loaded wheel")
        sys.exit(1)
print("[gate] arc6 skinny ops + stock set PRESENT", flush=True)

rows = []
failures = []
w4a8_gate_open = None

for tag, K, N in SHAPES:
    for M in MS_SKINNY + MS_BIG:
        skinny_regime = M in MS_SKINNY
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
        Bt8 = w8.t()  # nt view [K, N]
        w_pt = torch.rand(1, device=dev, dtype=torch.float32) * 0.01 + 0.005
        w_pc = (
            torch.rand(1, N, device=dev, dtype=torch.float32) * 0.01 + 0.005
        )
        ws8_kn = torch.randint(
            120, 136, (K // 32, N), dtype=torch.uint8, device=dev
        ).view(torch.float8_e8m0fnu)

        w = torch.randint(
            0, 255, (N, K // 2), dtype=torch.uint8, device=dev
        ).view(torch.float4_e2m1fn_x2)
        ws = torch.randint(
            120, 136, (N, K // 32), dtype=torch.uint8, device=dev
        ).view(torch.float8_e8m0fnu)
        ws_kn = ws.t().contiguous()
        Bt = w.t()

        row = {"tag": tag, "M": M, "K": K, "N": N}

        # references (fp32 dequant ground truth)
        with torch.no_grad():
            ref_pt = x.float() @ (w8.float().t() * w_pt)
            ref_pc = x.float() @ (w8.float().t() * w_pc)
            ref_m816 = x.float() @ (
                w8.float().t() * e8m0_depow(ws8_kn).repeat_interleave(
                    32, dim=0))
            b = w.view(torch.uint8).to(torch.int16)
            lo = (b & 0xF).to(torch.int64)
            hi = ((b >> 4) & 0xF).to(torch.int64)
            lut = _E2M1.to(dev)
            wdq = torch.empty(N, K, dtype=torch.float32, device=dev)
            wdq[:, 0::2] = lut[lo]
            wdq[:, 1::2] = lut[hi]
            wdq *= e8m0_depow(ws).repeat_interleave(32, dim=1)
            ref_g16 = x.float() @ wdq.t()

        # lane -> (stock fn, skinny fn, reference) in bench order
        lanes = {
            "w16pt": (lambda: ops.fp8_gemm_w8a16(x, Bt8, w_pt, None),
                      lambda: ops.skinny_fp8_gemm_w8a16(x, Bt8, w_pt, None),
                      ref_pt),
            "w16pc": (lambda: ops.fp8_gemm_w8a16(x, Bt8, w_pc, None),
                      lambda: ops.skinny_fp8_gemm_w8a16(x, Bt8, w_pc, None),
                      ref_pc),
            "m816": (lambda: ops.fp8_gemm_w8a16_mxfp8(
                        x, Bt8, ws8_kn, torch.bfloat16, None),
                     lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(
                        x, Bt8, ws8_kn, torch.bfloat16, None),
                     ref_m816),
            "g16": (lambda: ops.fp4_gemm_w4a16(
                        x, Bt, ws_kn, torch.bfloat16, None),
                    lambda: ops.skinny_fp4_gemm_w4a16(
                        x, Bt, ws_kn, torch.bfloat16, None),
                    ref_g16),
        }

        for lane, (fn_st, fn_sk, ref) in lanes.items():
            if skinny_regime:
                # interleaved paired A/B (drift-immune, window7+)
                st_ms, sk_ms = ab_bench(fn_st, fn_sk)
                row[lane] = st_ms
                row["skinny_" + lane] = sk_ms
                row["ratio_" + lane] = sk_ms / st_ms
                with torch.no_grad():
                    rel = _rel(fn_sk(), ref)
                row["rel_skinny_" + lane] = rel
                if rel > REL_GATE:
                    failures.append((tag, M, "skinny_" + lane, rel))
                # stock correctness at M<=8 too (same gate as window5/6)
                with torch.no_grad():
                    rel_st = _rel(fn_st(), ref)
                row["rel_" + lane] = rel_st
                if rel_st > REL_GATE:
                    failures.append((tag, M, lane, rel_st))
            else:
                # large-M stock control: plain sequential bench, no skinny
                try:
                    row[lane] = bench(fn_st)
                except Exception as ex:  # noqa: BLE001
                    print(f"[{tag} M={M} {lane}] FAILED: "
                          f"{type(ex).__name__}: {ex}")
                    row[lane] = None

        # ---- w4a8 probe (once, at the first M=1 row): arc6 dispatch arms
        # -> does oneDNN accept the (e4m3, f4) joint?
        if M == 1 and w4a8_gate_open is None:
            try:
                x8, asc = qtok(x)
                y = ops.fp4_gemm_w4a8(
                    x8, asc.contiguous(), Bt, ws_kn,
                    torch.bfloat16, None)
                w4a8_gate_open = True
                row["w4a8_probe"] = "ACCEPTED"
                print("[gate] (e4m3, f4) joint ACCEPTED via arc6 arms — "
                      "W4A8 lane confirmed OPEN (r123 refusal was our "
                      "dispatch default, P6.32 item 3)")
            except Exception as ex:  # noqa: BLE001
                w4a8_gate_open = False
                row["w4a8_probe"] = f"REJECTED: {ex}"[:200]
                print(f"[gate] (e4m3, f4) joint still rejected: {ex}")

        rows.append(row)
        print(json.dumps(row), flush=True)

print("\n=== r126 summary (interleaved paired medians, ms) ===")
print("    stock->skinny (ratio) | skinny eff GB/s  [197 bar: ! below]")
for M in MS_SKINNY:
    print(f"-- M={M}")
    for tag, K, N in SHAPES:
        sel = next(r for r in rows if r["tag"] == tag and r["M"] == M)
        parts = []
        for lane in ("w16pt", "w16pc", "m816", "g16"):
            sv, sp, rt = (sel.get("skinny_" + lane), sel.get(lane),
                          sel.get("ratio_" + lane))
            if sv is not None and sp:
                bw = nbytes(lane, K, N) / sv / 1e6
                flag = "!" if bw < GB_BAR else ""
                parts.append(
                    f"{lane} {sp:.4f}->{sv:.4f}({rt:.2f}x,{bw:.0f}{flag})")
        print(f"  {tag:12s} " + " ".join(parts))

print("\n=== gates ===")
# attempted gate cells (stock+skinny rel per lane at M<=8); printed as
# failed/total so a clean pass cannot read as "0 cells ran"
gate_total = sum(
    2 for r in rows if r["M"] in MS_SKINNY
    for k in ("w16pt", "w16pc", "m816", "g16") if k in r)
print(f"GATE-1 correctness (rel < {REL_GATE}): "
      f"{'PASS' if not failures else 'FAIL'} "
      f"({len(failures)}/{gate_total} gate cells failed)")
for f in failures:
    print("   ", f)
sp1 = [(r["tag"], round(r["ratio_w16pc"], 3)) for r in rows
       if r["M"] == 1 and r.get("ratio_w16pc")]
print(f"GATE-2 skinny_w16pc M=1 vs stock (paired): "
      f"{('PASS' if all(s < 1.0 for _, s in sp1) else 'REPORT') if sp1 else 'FAIL (no cells ran)'} {sp1}")
sub = [(r["tag"], l, round(r["ratio_" + l], 3), round(
        nbytes(l, r["K"], r["N"]) / r["skinny_" + l] / 1e6))
       for r in rows if r["M"] == 1
       for l in ("w16pt", "w16pc", "m816", "g16")
       if r.get("ratio_" + l) and r["ratio_" + l] >= 1.0]
print(f"GATE-2b all-lane M=1 losses (ratio>=1.0): {len(sub)}")
for t in sub:
    print("   ", t)
print(f"GATE-3 w4a8 dispatch (arc6 arms): {w4a8_gate_open}")

with open("/work/r126_rows.json", "w") as f:
    json.dump({"rows": rows, "w4a8_gate_open": w4a8_gate_open,
               "failures": [list(x) for x in failures]}, f, indent=1)
print("[ok] rows banked to /work/r126_rows.json (host: p633 dir)")
