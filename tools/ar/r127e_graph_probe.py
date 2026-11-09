# r127e_graph_probe.py -- P6.33: eager vs XPU-graph-replay per-call time for
# skinny m816 vs stock, at the gemma serve shapes (leg-D regression triage).
# r127 leg D: solo decode -38% with skinny recorded into the replayed decode
# graph, while the round-interleaved kbench (eager queue, paired) read the
# same cells neutral-to-winning. This probe times each op two ways:
#   eager  -- plain back-to-back loop (kbench regime)
#   replay -- one 1-call torch.xpu XPUGraph, replayed in a loop (graph regime)
# If replay/skinny >> 1.0 while replay/stock ~ 1.0, the cost is LZ-graph
# replay of our kernels, not the kernel bodies.
import statistics
import torch

torch.manual_seed(0)
dev = "xpu"
ROUNDS, WARM = 30, 10
import vllm_xpu_kernels  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

ops = torch.ops._xpu_C
G = torch.xpu.XPUGraph
print("graph cls:", G)

SHAPES = [
    ("gemma.up", 5376, 10752),
    ("gemma.down", 10752, 5376),
    ("gemma.qkv", 5376, 8192),
    ("gemma.o", 5376, 5376),
]
M = 2


def eager_ms(fn):
    for _ in range(WARM):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(ROUNDS):
        a = torch.xpu.Event(enable_timing=True)
        b = torch.xpu.Event(enable_timing=True)
        a.record(); fn(); b.record()
        torch.xpu.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def replay_ms(fn):
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        for _ in range(3):
            fn()
    torch.xpu.current_stream().wait_stream(s)
    g = G()
    torch.xpu.synchronize()
    with torch.xpu.graph(g):
        fn()
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


print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0))
for tag, K, N in SHAPES:
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
    Bt8 = w8.t()
    ws8_kn = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                           device=dev).view(torch.float8_e8m0fnu)
    fn_st = lambda: ops.fp8_gemm_w8a16_mxfp8(x, Bt8, ws8_kn, torch.bfloat16, None)
    fn_sk = lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(x, Bt8, ws8_kn, torch.bfloat16, None)
    e_st = eager_ms(fn_st)
    e_sk = eager_ms(fn_sk)
    r_st = replay_ms(fn_st)
    r_sk = replay_ms(fn_sk)
    print(f"{tag:12s} M={M} stock eager {e_st:.4f} replay {r_st:.4f} ({r_st/e_st:.3f}x)"
          f" | skinny eager {e_sk:.4f} replay {r_sk:.4f} ({r_sk/e_sk:.3f}x)"
          f" | replay sk/st {r_sk/r_st:.3f}x", flush=True)
print("PROBE_DONE")
