# r126_kbench_ri.py -- P6.33 arc6 skinny-M kernel gate, ROUND-INTERLEAVED.
#
# The window7/8 single-pass kbench interleaves stock/skinny per ITERATION
# (kills slow clock drift between the pair) but benches each (shape, lane,
# M) cell as one contiguous block -- and the rig's bimodality flips on a
# seconds timescale, so cells benched at different points in the run see
# different modes: window8 read our w16pt 18% slower than our w16pc on
# gemma.o (identical kernel work, only the scale tensor shape differs),
# and stock gemma.up m816 doubled between windows 7 and 8.
#
# This variant runs the whole cell MATRIX as a round-robin: every
# (shape, lane, M) pair is benched once per ROUND, ROUNDS rounds, one
# queue sync per round -- every cell's median spans the same mode
# episodes. p10/p50/p90 per cell make residual bimodality visible
# instead of deciding the gate. GATE-1/2/2b/3 semantics unchanged from
# r126_kbench.py (same tensors, same refs, same 2e-2 rel gate, 197 GB/s
# directive bar); big-M stock controls stay sequential (stock-only).
import json
import statistics
import sys

import torch

torch.manual_seed(0)
dev = "xpu"
ROUNDS, WARM = 30, 12

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
GB_BAR = 197.0  # directive bar: stock oneDNN reference level


def bench(fn):
    for _ in range(WARM):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(ROUNDS):
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.xpu.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def e8m0_depow(sc_u8):
    return torch.pow(2.0, sc_u8.view(torch.uint8).to(torch.float32) - 127.0)


def qtok(x):
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


def pct(v, p):
    v = sorted(v)
    i = min(len(v) - 1, max(0, int(round(p / 100.0 * (len(v) - 1)))))
    return v[i]


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

failures = []
w4a8_gate_open = None
cells = []  # dicts: tag,M,K,N,lane,fn_st,fn_sk,ref,t_st=[],t_sk=[]
big_rows = []


def make_lanes(x, Bt8, w_pt, w_pc, ws8_kn, Bt, ws_kn):
    """Closures over PARAMETERS (early-bound): the round-robin defers every
    call past the tensor-setup loop, so plain lambdas over the loop
    variables would late-bind to the last iteration's tensors."""
    return {
        "w16pt": (lambda: ops.fp8_gemm_w8a16(x, Bt8, w_pt, None),
                  lambda: ops.skinny_fp8_gemm_w8a16(x, Bt8, w_pt, None)),
        "w16pc": (lambda: ops.fp8_gemm_w8a16(x, Bt8, w_pc, None),
                  lambda: ops.skinny_fp8_gemm_w8a16(x, Bt8, w_pc, None)),
        "m816": (lambda: ops.fp8_gemm_w8a16_mxfp8(
                    x, Bt8, ws8_kn, torch.bfloat16, None),
                 lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(
                    x, Bt8, ws8_kn, torch.bfloat16, None)),
        "g16": (lambda: ops.fp4_gemm_w4a16(
                    x, Bt, ws_kn, torch.bfloat16, None),
                lambda: ops.skinny_fp4_gemm_w4a16(
                    x, Bt, ws_kn, torch.bfloat16, None)),
    }


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

        lanes = make_lanes(x, Bt8, w_pt, w_pc, ws8_kn, Bt, ws_kn)
        refs = {"w16pt": ref_pt, "w16pc": ref_pc,
                "m816": ref_m816, "g16": ref_g16}

        if skinny_regime:
            for lane, (fn_st, fn_sk) in lanes.items():
                cells.append(dict(tag=tag, M=M, K=K, N=N, lane=lane,
                                  fn_st=fn_st, fn_sk=fn_sk, ref=refs[lane],
                                  t_st=[], t_sk=[]))
        else:
            for lane, (fn_st, _fn_sk) in lanes.items():
                try:
                    ms = bench(fn_st)
                except Exception as ex:  # noqa: BLE001
                    print(f"[{tag} M={M} {lane}] FAILED: "
                          f"{type(ex).__name__}: {ex}")
                    ms = None
                big_rows.append(dict(tag=tag, M=M, K=K, N=N, lane=lane,
                                     st=ms))

        # w4a8 dispatch probe (once, at the first M=1 shape)
        if M == 1 and w4a8_gate_open is None:
            try:
                x8, asc = qtok(x)
                ops.fp4_gemm_w4a8(x8, asc.contiguous(), Bt, ws_kn,
                                  torch.bfloat16, None)
                w4a8_gate_open = True
                print("[gate] (e4m3, f4) joint ACCEPTED via arc6 arms")
            except Exception as ex:  # noqa: BLE001
                w4a8_gate_open = False
                print(f"[gate] (e4m3, f4) joint still rejected: {ex}")

# ---- round-robin: one A/B pair per cell per round, one sync per round
print(f"=== [ri] {len(cells)} cells x {ROUNDS} rounds interleaved ===",
      flush=True)
for c in cells:
    for _ in range(WARM):
        c["fn_st"]()
        c["fn_sk"]()
torch.xpu.synchronize()
for r in range(ROUNDS):
    evs = []
    for c in cells:
        s1 = torch.xpu.Event(enable_timing=True)
        e1 = torch.xpu.Event(enable_timing=True)
        s2 = torch.xpu.Event(enable_timing=True)
        e2 = torch.xpu.Event(enable_timing=True)
        s1.record()
        c["fn_st"]()
        e1.record()
        s2.record()
        c["fn_sk"]()
        e2.record()
        evs.append((s1, e1, s2, e2))
    torch.xpu.synchronize()
    for c, (s1, e1, s2, e2) in zip(cells, evs):
        c["t_st"].append(s1.elapsed_time(e1))
        c["t_sk"].append(s2.elapsed_time(e2))
    if (r + 1) % 10 == 0:
        print(f"  round {r + 1}/{ROUNDS} done", flush=True)

# ---- correctness + rows
rows = []
for c in cells:
    with torch.no_grad():
        rel_sk = _rel(c["fn_sk"](), c["ref"])
        rel_st = _rel(c["fn_st"](), c["ref"])
    if rel_sk > REL_GATE:
        failures.append((c["tag"], c["M"], "skinny_" + c["lane"], rel_sk))
    if rel_st > REL_GATE:
        failures.append((c["tag"], c["M"], c["lane"], rel_st))
    p50s, p50k = pct(c["t_st"], 50), pct(c["t_sk"], 50)
    rows.append(dict(
        tag=c["tag"], M=c["M"], K=c["K"], N=c["N"], lane=c["lane"],
        st=p50s, sk=p50k, ratio=p50k / p50s,
        st_p10=pct(c["t_st"], 10), st_p90=pct(c["t_st"], 90),
        sk_p10=pct(c["t_sk"], 10), sk_p90=pct(c["t_sk"], 90),
        bw=nbytes(c["lane"], c["K"], c["N"]) / p50k / 1e6,
        rel_skinny=rel_sk, rel_stock=rel_st,
    ))

print("\n=== r126-ri summary (round-interleaved p50, ms) ===")
print("    stock->skinny (ratio) | skinny eff GB/s  [197 bar: ! below]")
for M in MS_SKINNY:
    print(f"-- M={M}")
    for tag, K, N in SHAPES:
        sel = [r for r in rows if r["tag"] == tag and r["M"] == M]
        parts = []
        for r in sel:
            flag = "!" if r["bw"] < GB_BAR else ""
            parts.append(
                f"{r['lane']} {r['st']:.4f}->{r['sk']:.4f}"
                f"({r['ratio']:.2f}x,{r['bw']:.0f}{flag})")
        print(f"  {tag:12s} " + " ".join(parts))

print("\n=== per-lane M=1 means (p50 ms / mean ratio / wins) ===")
for lane in ("w16pt", "w16pc", "m816", "g16"):
    sel = [r for r in rows if r["M"] == 1 and r["lane"] == lane]
    if sel:
        print(f"  {lane:6s} stock {statistics.mean(r['st'] for r in sel):.4f}"
              f" skinny {statistics.mean(r['sk'] for r in sel):.4f}"
              f" mean_ratio {statistics.mean(r['ratio'] for r in sel):.3f}"
              f" wins {sum(1 for r in sel if r['ratio'] < 1.0)}/{len(sel)}")

print("\n=== gates ===")
print(f"GATE-1 correctness (rel < {REL_GATE}): "
      f"{'PASS' if not failures else 'FAIL'} "
      f"({len(failures)}/{2 * len(cells)} gate cells failed)")
for f in failures:
    print("   ", f)
sp1 = [(r["tag"], round(r["ratio"], 3)) for r in rows
       if r["M"] == 1 and r["lane"] == "w16pc"]
print(f"GATE-2 skinny_w16pc M=1 vs stock (paired): "
      f"{('PASS' if all(s < 1.0 for _, s in sp1) else 'REPORT') if sp1 else 'FAIL (no cells ran)'} {sp1}")
sub = [(r["tag"], r["lane"], round(r["ratio"], 3), round(r["bw"]))
       for r in rows if r["M"] == 1 and r["ratio"] >= 1.0]
print(f"GATE-2b all-lane M=1 losses (ratio>=1.0): {len(sub)}")
for t in sub:
    print("   ", t)
allm = [(r["tag"], r["M"], r["lane"], round(r["ratio"], 3))
        for r in rows if r["ratio"] >= 1.0]
print(f"GATE-2c all-M losses (ratio>=1.0): {len(allm)} / {len(rows)}")
for t in allm:
    print("   ", t)
print(f"GATE-3 w4a8 dispatch (arc6 arms): {w4a8_gate_open}")

with open("/work/r126_rows_ri.json", "w") as f:
    json.dump({"rows": rows, "big": big_rows,
               "w4a8_gate_open": w4a8_gate_open,
               "failures": [list(x) for x in failures]}, f, indent=1)
print("[ok] rows banked to /work/r126_rows_ri.json (host: p633 dir)")
