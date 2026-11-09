# r128a2_ceilings.py -- P6.34: true read rooflines per shape (the r128a
# sum(int64) probe materialized a cast -> 32 GB/s garbage). max() over
# the raw bytes is a single-pass native read reduction -- the same
# 50-call in-graph loop as every bar, so cache conditions match.
import statistics

import torch

dev = "xpu"
ROUNDS, WARM, NCALL = 20, 5, 50
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
for tag, K, N in SHAPES:
    for label, nbytes, mk in (
        ("f8", K * N, lambda: torch.randint(
            0, 255, (N, K), dtype=torch.uint8, device=dev)),
        ("f4", K * N // 2, lambda: torch.randint(
            0, 255, (N, K // 2), dtype=torch.uint8, device=dev)),
    ):
        buf = mk()
        g = capture50(lambda: buf.max())
        ms = replay_ms(g)
        del g, buf
        print(f"[ceiling] {tag:12s} {label} {nbytes/1e6:6.1f} MB "
              f"read-max {ms:.4f} ms = {nbytes/ms/1e6:6.0f} GB/s",
              flush=True)
# DRAM-vs-L2 discriminator: one big buffer (2x the largest shape) that
# cannot sit in L2 across the 50 calls
big = torch.randint(0, 255, (4 * 10752, 10752), dtype=torch.uint8,
                    device=dev)  # ~440 MB
g = capture50(lambda: big.max())
ms = replay_ms(g)
print(f"[ceiling] big-440MB     read-max {ms:.4f} ms = "
      f"{big.numel()/ms/1e6:6.0f} GB/s (DRAM path)", flush=True)
print("CEILINGS_DONE", flush=True)
