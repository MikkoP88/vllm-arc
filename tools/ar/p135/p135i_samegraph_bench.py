#!/usr/bin/env python3
"""p135i_samegraph_bench.py -- P6.35 mechanism discriminator v4 (final cell).

Every prior bench falsified: queue depth (p135h B4), ccl-in-graph + peer
rendezvous (p135g), fixed per-node cost (p135e). All surviving evidence
(p135d tight same-graph loop ~= device drain; p135e isolated = 7.5 us;
p135h different-graphs = 8 us; serving ~= 24 ms) points at: replaying the
SAME exec while ITS OWN prior launch is still in flight blocks until that
launch retires. That is exactly serving's steady state (async scheduling
runs one step ahead; the ~6k-node verify graph replays onto itself).

  B7 same-graph re-launch, no sync: block ~= own drain => confirmed.
  B7b control: sync between => us.
  B8 two graphs sharing ONE memory pool, alternating: no block =>
     double-buffer capture is a viable vllm_gaudi-level fix.
  B8b two graphs, separate pools, alternating (control).

Writes /root/p135i_bench.json. Probe scratch only; never ships.
"""
import json
import time

import torch


def build_chain(base, n, pool=None):
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g, pool=pool):
        b = base
        for _ in range(n):
            b = b * 1.0001
    return g


def main():
    assert torch.xpu.device_count() > 0
    res = {}

    # ---- B7: same graph, back-to-back launches ----
    base = torch.zeros(256, device="xpu")
    g = build_chain(base, 20000)
    for _ in range(2):
        g.replay()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.xpu.synchronize()
    res["B7_drain_wall_us"] = round((time.perf_counter() - t0) * 1e6, 1)

    torch.xpu.synchronize()
    t0 = time.perf_counter()
    g.replay()
    res["B7b_empty_launch_us"] = round((time.perf_counter() - t0) * 1e6, 1)
    torch.xpu.synchronize()

    for trial in range(3):
        torch.xpu.synchronize()
        g.replay()  # enqueue, no sync
        t0 = time.perf_counter()
        g.replay()  # SAME exec while in flight
        res["B7_samegraph_inflight_us_t%d" % trial] = round(
            (time.perf_counter() - t0) * 1e6, 1)
        torch.xpu.synchronize()

    # ---- B8: alternating graphs, shared pool ----
    pool = torch.xpu.graph_pool_handle()
    b1 = torch.zeros(256, device="xpu")
    b2 = torch.zeros(256, device="xpu")
    ga = build_chain(b1, 20000, pool=pool)
    gb = build_chain(b2, 20000, pool=pool)
    for _ in range(2):
        ga.replay()
        gb.replay()
    torch.xpu.synchronize()
    res["B8_shared_pool_alt_us"] = []
    for trial in range(3):
        torch.xpu.synchronize()
        ga.replay()
        t0 = time.perf_counter()
        gb.replay()  # other exec, same pool, first still in flight
        res["B8_shared_pool_alt_us"].append(
            round((time.perf_counter() - t0) * 1e6, 1))
        torch.xpu.synchronize()

    # ---- B8b: alternating graphs, separate pools (p135h B4 re-check) ----
    b3 = torch.zeros(256, device="xpu")
    b4 = torch.zeros(256, device="xpu")
    gc = build_chain(b3, 20000)
    gd = build_chain(b4, 20000)
    torch.xpu.synchronize()
    res["B8b_separate_pool_alt_us"] = []
    for trial in range(3):
        torch.xpu.synchronize()
        gc.replay()
        t0 = time.perf_counter()
        gd.replay()
        res["B8b_separate_pool_alt_us"].append(
            round((time.perf_counter() - t0) * 1e6, 1))
        torch.xpu.synchronize()

    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open("/root/p135i_bench.json", "w"), indent=1)
    print("P135I_OK")


if __name__ == "__main__":
    main()
