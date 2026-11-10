#!/usr/bin/env python3
"""p135b_replay_bench.py -- P6.35-B: isolate torch 2.13 XPU-graph replay host
cost vs node count (the serving trace shows ~29.5 ms device-idle per decode
step aligned with graphs[desc].replay(); device work is only ~18 ms/step).

Measures per-call HOST wall for replay() and eager enqueue across graph sizes
(1/10/50/135/400 nodes; 135 ~= kernels/step in the serving window), plus
device-side per-replay time via events for the 135-node graph.
"""
import time

import torch

dev = "xpu"
torch.xpu.set_device(0)
a = torch.randn(2048, 2048, device=dev, dtype=torch.half)
N = 200


def build(nodes):
    for _ in range(3):
        a.mul_(1.000001)
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    s = torch.xpu.Stream()
    s.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(s):
        with torch.xpu.graph(g):
            b = a
            for _ in range(nodes):
                b = b.mul(1.000001)
    torch.xpu.current_stream().wait_stream(s)
    torch.xpu.synchronize()
    return g, b


print("nodes  replay_host_us  eager_host_us  replay_dev_us")
for nodes in (1, 10, 50, 135, 400):
    g, _ = build(nodes)
    # enqueue-only host cost of replay
    for _ in range(5):
        g.replay()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        g.replay()
    t_rep = (time.perf_counter() - t0) / N * 1e6
    torch.xpu.synchronize()

    # eager enqueue of the same op count
    t0 = time.perf_counter()
    for _ in range(N):
        b = a
        for _ in range(nodes):
            b = b.mul(1.000001)
    t_eag = (time.perf_counter() - t0) / N * 1e6
    torch.xpu.synchronize()

    # device time per replay (135-node case also printed separately below)
    ev0, ev1 = torch.xpu.Event(True), torch.xpu.Event(True)
    ev0.record()
    for _ in range(N):
        g.replay()
    ev1.record()
    torch.xpu.synchronize()
    t_dev = ev0.elapsed_time(ev1) / N * 1e3
    print("%5d  %14.1f  %13.1f  %13.1f" % (nodes, t_rep, t_eag, t_dev))

# replay+full-sync-per-call (the worst-case blocking pattern)
g, _ = build(135)
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    g.replay()
    torch.xpu.synchronize()
t_sync = (time.perf_counter() - t0) / 50 * 1e6
print("replay+sync@135: %.1f us/call" % t_sync)
print("P135B_DONE")
