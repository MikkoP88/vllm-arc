# r126_sweep.py -- P6.33 v2.4 probe knob sweep (arc6 probe wheel only).
#
# The probe wheel reads ARC_SKINNY_WI / ARC_SKINNY_SWIZZLE / ARC_SKINNY_KC_MIN
# per call (no caching), so ONE process can sweep launch-config combos. For
# every combo x shape x lane we run the window7 interleaved paired A/B
# (drift-immune) and a correctness rel check (knob bugs -- e.g. a broken
# swizzle bijection -- must fail loudly, not optimize a wrong kernel).
#
# Combos target the window7 M=1 residue: fast-stock w16pt/m816 cells
# (need 232-347 GB/s), the qwen.o/qwen.up absolute laggards (pow2 column
# stride suspect), and M>=4 load starvation. M=1 sweeps all combos; M=8
# spot-checks {base, KC=1024, SW=1}.
import json
import os
import statistics
import sys

import torch

torch.manual_seed(0)
dev = "xpu"
ITERS, WARM = 50, 15
REL_GATE = 2e-2

SHAPES = [
    ("gemma.up", 5376, 10752),
    ("gemma.down", 10752, 5376),
    ("gemma.o", 5376, 5376),
    ("qwen.up", 5120, 8704),
    ("qwen.down", 8704, 5120),
    ("qwen.o", 5120, 5120),
]
LANES = ("w16pt", "w16pc", "m816", "g16")

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2, dtype=torch.float32
)
_E2M1[8:] *= -1.0

COMBOS = [
    ("base", {}),
    ("sw", {"ARC_SKINNY_SWIZZLE": "1"}),
    ("wi16k", {"ARC_SKINNY_WI": "16384"}),
    ("wi16k+sw", {"ARC_SKINNY_WI": "16384", "ARC_SKINNY_SWIZZLE": "1"}),
    ("wi64k", {"ARC_SKINNY_WI": "65536"}),
    ("wi64k+sw", {"ARC_SKINNY_WI": "65536", "ARC_SKINNY_SWIZZLE": "1"}),
    ("kc1024", {"ARC_SKINNY_KC_MIN": "1024"}),
    ("kc1024+sw", {"ARC_SKINNY_KC_MIN": "1024", "ARC_SKINNY_SWIZZLE": "1"}),
    ("wi16k+kc1024+sw", {"ARC_SKINNY_WI": "16384",
                         "ARC_SKINNY_KC_MIN": "1024",
                         "ARC_SKINNY_SWIZZLE": "1"}),
    ("kc2048", {"ARC_SKINNY_KC_MIN": "2048"}),
]


def ab_bench(fn_a, fn_b):
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
    return torch.pow(2.0, sc_u8.view(torch.uint8).to(torch.float32) - 127.0)


def nbytes(lane, K, N):
    return K * N if lane != "g16" else K * N // 2


print("=== [0] env ===")
print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0))
import vllm_xpu_kernels  # noqa: F401,E402
import vllm_xpu_kernels._xpu_C  # noqa: F401,E402  (dlopen law, r120b)

ops = torch.ops._xpu_C
for op_name in ("skinny_fp8_gemm_w8a16", "skinny_fp8_gemm_w8a16_mxfp8",
                "skinny_fp4_gemm_w4a16"):
    if not hasattr(ops, op_name):
        print(f"FATAL: _xpu_C.{op_name} absent")
        sys.exit(1)
print("[ok] probe wheel ops present", flush=True)

# tensors + refs built ONCE (identical inputs across every combo)
DATA = {}
for tag, K, N in SHAPES:
    x = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    x8 = torch.randn(8, K, dtype=torch.bfloat16, device=dev)
    w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    Bt8 = w8.t()
    w_pt = torch.rand(1, device=dev, dtype=torch.float32) * 0.01 + 0.005
    w_pc = torch.rand(1, N, device=dev, dtype=torch.float32) * 0.01 + 0.005
    ws8_kn = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                           device=dev).view(torch.float8_e8m0fnu)
    w = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8,
                      device=dev).view(torch.float4_e2m1fn_x2)
    ws = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8,
                       device=dev).view(torch.float8_e8m0fnu)
    ws_kn = ws.t().contiguous()
    Bt = w.t()
    with torch.no_grad():
        def ref_f8(xr, sc):
            return xr.float() @ (w8.float().t() * sc)
        ref_m816 = x8.float() @ (
            w8.float().t() * e8m0_depow(ws8_kn).repeat_interleave(32, dim=0))
        b = w.view(torch.uint8).to(torch.int16)
        lo = (b & 0xF).to(torch.int64)
        hi = ((b >> 4) & 0xF).to(torch.int64)
        lut = _E2M1.to(dev)
        wdq = torch.empty(N, K, dtype=torch.float32, device=dev)
        wdq[:, 0::2] = lut[lo]
        wdq[:, 1::2] = lut[hi]
        wdq *= e8m0_depow(ws).repeat_interleave(32, dim=1)
        ref_g16 = x8.float() @ wdq.t()
    DATA[tag] = dict(K=K, N=N, x=x, x8=x8, Bt8=Bt8, w_pt=w_pt, w_pc=w_pc,
                     ws8_kn=ws8_kn, Bt=Bt, ws_kn=ws_kn,
                     ref_m816=ref_m816, ref_g16=ref_g16,
                     ref_pt=ref_f8(x, w_pt), ref_pc=ref_f8(x, w_pc))


def lane_fns(tag, lane, M):
    d = DATA[tag]
    x = d["x"] if M == 1 else d["x8"]
    if lane == "w16pt":
        return (lambda: ops.fp8_gemm_w8a16(x, d["Bt8"], d["w_pt"], None),
                lambda: ops.skinny_fp8_gemm_w8a16(x, d["Bt8"], d["w_pt"],
                                                  None))
    if lane == "w16pc":
        return (lambda: ops.fp8_gemm_w8a16(x, d["Bt8"], d["w_pc"], None),
                lambda: ops.skinny_fp8_gemm_w8a16(x, d["Bt8"], d["w_pc"],
                                                  None))
    if lane == "m816":
        return (lambda: ops.fp8_gemm_w8a16_mxfp8(
                    x, d["Bt8"], d["ws8_kn"], torch.bfloat16, None),
                lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(
                    x, d["Bt8"], d["ws8_kn"], torch.bfloat16, None))
    return (lambda: ops.fp4_gemm_w4a16(x, d["Bt"], d["ws_kn"],
                                       torch.bfloat16, None),
            lambda: ops.skinny_fp4_gemm_w4a16(x, d["Bt"], d["ws_kn"],
                                              torch.bfloat16, None))


def ref_for(tag, lane):
    d = DATA[tag]
    return {"w16pt": d["ref_pt"], "w16pc": d["ref_pc"],
            "m816": d["ref_m816"], "g16": d["ref_g16"]}[lane]


def set_env(kv):
    for k in ("ARC_SKINNY_WI", "ARC_SKINNY_SWIZZLE", "ARC_SKINNY_KC_MIN"):
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = v


rows = []
failures = []
for combo_name, kv in COMBOS:
    set_env(kv)
    print(f"=== combo {combo_name} {kv} (M=1) ===", flush=True)
    for tag, K, N in SHAPES:
        for lane in LANES:
            fn_st, fn_sk = lane_fns(tag, lane, 1)
            st_ms, sk_ms = ab_bench(fn_st, fn_sk)
            with torch.no_grad():
                rel = float((fn_sk().float() - ref_for(tag, lane)).norm()
                            / ref_for(tag, lane).norm())
            if rel > REL_GATE:
                failures.append((combo_name, tag, lane, rel))
            rows.append(dict(combo=combo_name, M=1, tag=tag, lane=lane,
                             st=st_ms, sk=sk_ms, ratio=sk_ms / st_ms,
                             bw=nbytes(lane, K, N) / sk_ms / 1e6, rel=rel))
    # M=8 spot on a fixed small set
    for combo_name8 in ("base", "sw", "kc1024"):
        if combo_name8 != combo_name:
            continue
        for tag, K, N in SHAPES:
            for lane in LANES:
                fn_st, fn_sk = lane_fns(tag, lane, 8)
                st_ms, sk_ms = ab_bench(fn_st, fn_sk)
                rows.append(dict(combo=combo_name + "/M8", M=8, tag=tag,
                                 lane=lane, st=st_ms, sk=sk_ms,
                                 ratio=sk_ms / st_ms,
                                 bw=nbytes(lane, K, N) / sk_ms / 1e6))
        print(f"=== combo {combo_name} (M=8 spot) done ===", flush=True)

print("\n=== sweep summary (M=1): best combo per shape/lane ===")
for tag, K, N in SHAPES:
    parts = []
    for lane in LANES:
        cells = [r for r in rows if r["M"] == 1 and r["tag"] == tag
                 and r["lane"] == lane]
        best = min(cells, key=lambda r: r["sk"])
        parts.append(f"{lane}:{best['combo']} {best['sk']:.3f}ms "
                     f"({best['bw']:.0f}GB/s,r{best['ratio']:.2f})")
    print(f"  {tag:11s} " + " ".join(parts))

print("\n=== per-combo M=1 mean skinny ms / mean ratio / wins-out-of-24 ===")
for combo_name, _ in COMBOS:
    cells = [r for r in rows if r["M"] == 1 and r["combo"] == combo_name]
    ms = statistics.mean(r["sk"] for r in cells)
    rt = statistics.mean(r["ratio"] for r in cells)
    wins = sum(1 for r in cells if r["ratio"] < 1.0)
    print(f"  {combo_name:17s} mean_sk={ms:.4f} mean_ratio={rt:.3f} "
          f"wins={wins}/24")

print(f"\ncorrectness: {'PASS' if not failures else 'FAIL'} "
      f"({len(failures)} cells over {REL_GATE})")
for f in failures:
    print("   ", f)

with open("/work/r126_sweep_rows.json", "w") as f:
    json.dump({"rows": rows, "failures": [list(x) for x in failures]},
              f, indent=1)
print("[ok] banked to /work/r126_sweep_rows.json")
