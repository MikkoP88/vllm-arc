# r126_refine.py -- P6.33 v2.4 combo-interleaved refine (probe wheel).
#
# The knob sweep showed two things: a mechanistic f4 win (KC_MIN=1024
# lands 64-aligned kc on all six shapes -- K is a multiple of 64
# everywhere -- killing the odd-32 single-chunk tail every stream paid on
# K=5376: gemma.up g16 -51%) and a measurement trap (qwen.o w16pc is
# BIMODAL across runs, 0.125 vs 0.28 ms; base vs wi64k differed 2.2x with
# IDENTICAL launch configs, so cross-combo absolute medians carry +/-30%
# drift).
#
# This refine interleaves the CANDIDATE COMBOS per iteration (env is read
# per call by the probe kernel, so one process round-robins combos; every
# combo sees the same allocator block, tensors and thermal state each
# iteration) and reports p10/p50/p90 per combo -- bimodality becomes
# visible instead of deciding the result.
import json
import os
import statistics
import sys

import torch

torch.manual_seed(0)
dev = "xpu"
ITERS, WARM = 40, 12

SHAPES = [
    ("gemma.up", 5376, 10752),
    ("gemma.down", 10752, 5376),
    ("gemma.o", 5376, 5376),
    ("qwen.up", 5120, 8704),
    ("qwen.down", 8704, 5120),
    ("qwen.o", 5120, 5120),
]
LANES = ("w16pt", "w16pc", "m816", "g16")

COMBOS = [
    ("base", {}),
    ("kc1024", {"ARC_SKINNY_KC_MIN": "1024"}),
    ("sw", {"ARC_SKINNY_SWIZZLE": "1"}),
    ("kc1024+sw", {"ARC_SKINNY_KC_MIN": "1024",
                   "ARC_SKINNY_SWIZZLE": "1"}),
    ("wi64k", {"ARC_SKINNY_WI": "65536"}),
    ("wi64k+sw", {"ARC_SKINNY_WI": "65536", "ARC_SKINNY_SWIZZLE": "1"}),
]
# Ms covered by the full combo interleave; later Ms get the top combos
MS_FULL = [1]
MS_TOP = [2, 4, 8]
TOP_NAMES = ["base", "kc1024", "sw"]

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2, dtype=torch.float32
)
_E2M1[8:] *= -1.0


def e8m0_depow(sc_u8):
    return torch.pow(2.0, sc_u8.view(torch.uint8).to(torch.float32) - 127.0)


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
for op_name in ("skinny_fp8_gemm_w8a16", "skinny_fp8_gemm_w8a16_mxfp8",
                "skinny_fp4_gemm_w4a16"):
    if not hasattr(ops, op_name):
        print(f"FATAL: _xpu_C.{op_name} absent")
        sys.exit(1)
print("[ok] probe wheel ops present", flush=True)

DATA = {}
for tag, K, N in SHAPES:
    xs = {M: torch.randn(M, K, dtype=torch.bfloat16, device=dev)
          for M in (1, 2, 4, 8)}
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
    DATA[tag] = dict(K=K, N=N, xs=xs, Bt8=Bt8, w_pt=w_pt, w_pc=w_pc,
                     ws8_kn=ws8_kn, Bt=Bt, ws_kn=ws_kn)


def sk_fn(tag, lane, M):
    d = DATA[tag]
    x = d["xs"][M]
    if lane == "w16pt":
        return lambda: ops.skinny_fp8_gemm_w8a16(x, d["Bt8"], d["w_pt"], None)
    if lane == "w16pc":
        return lambda: ops.skinny_fp8_gemm_w8a16(x, d["Bt8"], d["w_pc"], None)
    if lane == "m816":
        return lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(
            x, d["Bt8"], d["ws8_kn"], torch.bfloat16, None)
    return lambda: ops.skinny_fp4_gemm_w4a16(
        x, d["Bt"], d["ws_kn"], torch.bfloat16, None)


def set_env(kv):
    for k in ("ARC_SKINNY_WI", "ARC_SKINNY_SWIZZLE", "ARC_SKINNY_KC_MIN"):
        os.environ.pop(k, None)
    for k, v in kv.items():
        os.environ[k] = v


rows = []
for M in MS_FULL + MS_TOP:
    combos = COMBOS if M in MS_FULL else [c for c in COMBOS
                                          if c[0] in TOP_NAMES]
    print(f"=== M={M}: {len(combos)} combos interleaved ===", flush=True)
    for tag, K, N in SHAPES:
        for lane in LANES:
            fn = sk_fn(tag, lane, M)
            names = [c[0] for c in combos]
            times = {n: [] for n in names}
            for wn, kv in combos:  # warm every combo once per round
                set_env(kv)
                for _ in range(WARM):
                    fn()
            torch.xpu.synchronize()
            for _ in range(ITERS):
                evs = []
                for _, kv in combos:
                    set_env(kv)
                    s = torch.xpu.Event(enable_timing=True)
                    e = torch.xpu.Event(enable_timing=True)
                    s.record()
                    fn()
                    e.record()
                    evs.append((s, e))
                torch.xpu.synchronize()
                for n, (s, e) in zip(names, evs):
                    times[n].append(s.elapsed_time(e))
            for n in names:
                t = times[n]
                rows.append(dict(M=M, tag=tag, lane=lane, combo=n,
                                 p10=pct(t, 10), p50=pct(t, 50),
                                 p90=pct(t, 90),
                                 bw=nbytes(lane, K, N) / pct(t, 50) / 1e6,
                                 spread=pct(t, 90) / max(1e-9, pct(t, 10))))
            best = min(rows[-len(names):], key=lambda r: r["p50"])
            worst_spread = max(r["spread"] for r in rows[-len(names):])
            print(f"  M={M} {tag:11s} {lane:5s} " + " ".join(
                f"{n}:{r['p50']:.3f}({r['bw']:.0f})" for n, r in
                zip(names, rows[-len(names):])) +
                f" BEST={best['combo']} spread<={worst_spread:.1f}x",
                flush=True)

print("\n=== refine summary (p50 ms; best per cell) ===")
wins = {}
for M in MS_FULL + MS_TOP:
    print(f"-- M={M}")
    for tag, K, N in SHAPES:
        parts = []
        for lane in LANES:
            cells = [r for r in rows if r["M"] == M and r["tag"] == tag
                     and r["lane"] == lane]
            if not cells:
                continue
            best = min(cells, key=lambda r: r["p50"])
            wins[best["combo"]] = wins.get(best["combo"], 0) + 1
            parts.append(f"{lane}:{best['combo']}({best['p50']:.3f},"
                         f"{best['bw']:.0f}GB/s)")
        print(f"  {tag:11s} " + " ".join(parts))
print("\nbest-combo win counts:", dict(sorted(wins.items(),
                                              key=lambda kv: -kv[1])))

with open("/work/r126_refine_rows.json", "w") as f:
    json.dump({"rows": rows}, f, indent=1)
print("[ok] banked to /work/r126_refine_rows.json")
