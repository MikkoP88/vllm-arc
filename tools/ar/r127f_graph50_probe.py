# r127f_graph50_probe.py -- P6.33: per-call in-graph cost, amortized.
# r127e showed 1-call XPUGraph replay is ~0.25ms overhead-dominated, so it
# could not separate kernel cost from replay cost. This variant records a
# NCALL=50 loop of one op into a single graph; replay median / NCALL = the
# per-call in-graph time -- the serve regime (240 kernels per decode-step
# replay). eager = paired A/B loop (stock then skinny, same round) to dodge
# the rig bimodality; graph columns are also paired (capture st, time, then
# capture sk, time).
import statistics
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
]
MS = [2]


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
    return statistics.median(ts)


def eager_pair(fn_st, fn_sk):
    for _ in range(WARM):
        fn_st(); fn_sk()
    torch.xpu.synchronize()
    ts, tk = [], []
    for _ in range(ROUNDS):
        a = torch.xpu.Event(enable_timing=True)
        b = torch.xpu.Event(enable_timing=True)
        c = torch.xpu.Event(enable_timing=True)
        d = torch.xpu.Event(enable_timing=True)
        a.record(); fn_st(); b.record()
        c.record(); fn_sk(); d.record()
        torch.xpu.synchronize()
        ts.append(a.elapsed_time(b)); tk.append(c.elapsed_time(d))
    return statistics.median(ts), statistics.median(tk)


print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0))
for tag, K, N in SHAPES:
    for M in MS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
        Bt8 = w8.t()
        ws8_kn = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                               device=dev).view(torch.float8_e8m0fnu)
        fn_st = lambda: ops.fp8_gemm_w8a16_mxfp8(x, Bt8, ws8_kn, torch.bfloat16, None)
        fn_sk = lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(x, Bt8, ws8_kn, torch.bfloat16, None)
        e_st, e_sk = eager_pair(fn_st, fn_sk)
        g_st = capture50(fn_st)
        r_st = replay_ms(g_st) / NCALL
        g_sk = capture50(fn_sk)
        r_sk = replay_ms(g_sk) / NCALL
        del g_st, g_sk
        print(f"{tag:12s} M={M} eager st {e_st:.4f} sk {e_sk:.4f} ({e_sk/e_st:.3f}x)"
              f" | graph st {r_st:.4f} sk {r_sk:.4f} ({r_sk/r_st:.3f}x)", flush=True)
print("PROBE_DONE")
