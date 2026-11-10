#!/usr/bin/env python3
"""p135d_instantiate_bench.py -- P6.35 D1 discriminator: is the linear
XPUGraph replay host cost (4.7-4.8 us/node, p135b) a torch-2.13
instantiation defect (=> keep_graph=True + explicit instantiate() fixes it)
or the driver-level launch itself (=> node-count reduction is the only lever)?

Cases per node count N:
  A keep_graph=False (default; capture_end instantiates): warmed replay host us
  B keep_graph=True, first replay() (auto-instantiate) wall ms
  B' keep_graph=True, warmed replay host us
  C keep_graph=True + explicit instantiate() BEFORE replay: inst wall ms,
     then warmed replay host us

Writes /root/p135d_bench.json. Probe scratch only; never ships.
"""
import json
import time

import torch


def build(n, keep_graph):
    g = torch.xpu.XPUGraph(keep_graph=keep_graph)
    x = torch.zeros(64, device="xpu")
    s = torch.xpu.Stream()
    with torch.xpu.graph(g, stream=s):
        for _ in range(n):
            x = x * 1.0001
    return g


def bench_replay(g, iters=30):
    for _ in range(3):
        g.replay()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    host = (time.perf_counter() - t0) / iters * 1e6
    torch.xpu.synchronize()
    return host


def main():
    assert torch.xpu.device_count() > 0, "no XPU device visible"
    out = {"torch": torch.__version__, "cases": []}
    for n in (1000, 6000):
        case = {"nodes": n}

        g_a = build(n, False)
        case["A_keepfalse_replay_host_us"] = round(bench_replay(g_a), 1)

        g_b = build(n, True)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        g_b.replay()  # auto-instantiates on first call
        case["B_first_replay_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        torch.xpu.synchronize()
        case["B_warmed_replay_host_us"] = round(bench_replay(g_b), 1)

        g_c = build(n, True)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        g_c.instantiate()
        case["C_instantiate_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        case["C_inst_then_replay_host_us"] = round(bench_replay(g_c), 1)

        out["cases"].append(case)
        print(case, flush=True)
        for g in (g_a, g_b, g_c):
            g.reset()
        torch.xpu.synchronize()

    with open("/root/p135d_bench.json", "w") as f:
        json.dump(out, f, indent=1)
    print("P135D_OK")


if __name__ == "__main__":
    main()
