# r128a_roofline_bars.py -- P6.34 step 1: the complete bar table + the
# device roofline, in the serve regime (50-call amortized XPUGraph
# replay; the P6.33 measurement law -- eager loops are host-rate
# illusions). Extends r127f: all six r126 shapes + gemma.qkv, all four
# lanes (w16pt/w16pc/m816/g16), M in {1,2}, stock vs scalar-skinny vs
# PURE-READ ceiling per shape. JM targets come off this table.
import json
import statistics
import sys

import torch

torch.manual_seed(0)
dev = "xpu"
ROUNDS, WARM, NCALL = 20, 5, 50
import vllm_xpu_kernels  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

ops = torch.ops._xpu_C
G = torch.xpu.XPUGraph

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

DATA = {}
for tag, K, N in SHAPES:
    x1 = torch.randn(1, K, dtype=torch.bfloat16, device=dev)
    x2 = torch.randn(2, K, dtype=torch.bfloat16, device=dev)
    w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    w_pt = torch.rand(1, device=dev, dtype=torch.float32) * 0.01 + 0.005
    w_pc = torch.rand(1, N, device=dev, dtype=torch.float32) * 0.01 + 0.005
    ws8_kn = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                           device=dev).view(torch.float8_e8m0fnu)
    w4 = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8,
                       device=dev).view(torch.float4_e2m1fn_x2)
    ws4_kn = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                           device=dev).view(torch.float8_e8m0fnu)
    DATA[tag] = dict(K=K, N=N, x1=x1, x2=x2, Bt8=w8.t(), w_pt=w_pt,
                     w_pc=w_pc, ws8_kn=ws8_kn, Bt4=w4.t(),
                     ws4_kn=ws4_kn, w_u8=w8.view(torch.uint8))


def lane_fns(tag, lane, M):
    d = DATA[tag]
    x = d["x1"] if M == 1 else d["x2"]
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
    return (lambda: ops.fp4_gemm_w4a16(x, d["Bt4"], d["ws4_kn"],
                                       torch.bfloat16, None),
            lambda: ops.skinny_fp4_gemm_w4a16(x, d["Bt4"], d["ws4_kn"],
                                              torch.bfloat16, None))


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


print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0),
      flush=True)
out = {"rows": []}
for tag, K, N in SHAPES:
    d = DATA[tag]
    # read ceiling: sum-reduce over the f8 weight bytes (read-only
    # traffic; our GEMM is weight-read-dominant)
    g_rd = capture50(lambda: d["w_u8"].sum(dtype=torch.int64))
    rd = replay_ms(g_rd)
    del g_rd
    gb_rd = K * N / rd / 1e6
    print(f"[ceiling] {tag:12s} read {rd:.4f} ms = {gb_rd:6.0f} GB/s",
          flush=True)
    for lane in LANES:
        for M in MS:
            fn_st, fn_sk = lane_fns(tag, lane, M)
            try:
                gs = replay_ms(capture50(fn_st))
                gk = replay_ms(capture50(fn_sk))
            except Exception as e:  # noqa: BLE001
                print(f"{tag:12s} {lane:5s} M={M} FAIL {type(e).__name__}: "
                      f"{str(e)[:120]}", flush=True)
                continue
            wbytes = K * N * (0.5 if lane == "g16" else 1.0)
            gbs_st = wbytes / gs / 1e6
            row = dict(tag=tag, lane=lane, M=M, st=gs, sk=gk,
                       ratio=gk / gs, gbs_st=gbs_st, rd_ms=rd,
                       gb_rd=gb_rd, pct_ceiling=100 * gbs_st / gb_rd)
            out["rows"].append(row)
            print(f"{tag:12s} {lane:5s} M={M} st {gs:.4f} ({gbs_st:5.0f} "
                  f"GB/s, {row['pct_ceiling']:4.1f}% ceil) "
                  f"sk {gk:.4f} ({gk/gs:.3f}x)", flush=True)
    sys.stdout.flush()
import os
outpath = os.environ.get("R128A_OUT", "/tmp/r128a_rows.json")
with open(outpath, "w") as f:
    json.dump(out, f, indent=1)
print("R128A_DONE", flush=True)
