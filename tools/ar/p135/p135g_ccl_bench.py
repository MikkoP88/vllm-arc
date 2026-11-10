#!/usr/bin/env python3
"""p135g_ccl_bench.py -- why does serving XPUGraph.replay block ~24 ms while
idle-device replay is 7.5 us (p135e)? Two hypotheses, one bench:

  (B) queue-acceptance throttling: enqueue a long-running graph WITHOUT
      sync, then time a TINY no-ccl graph replay. Block ~= backlog drain
      => launch throttled by queue depth.
  (A) ccl-in-graph sync: graph containing dist.all_reduce; time replay
      while the PEER rank is held back 200 ms. Block ~= 200 ms on the
      early rank => cross-rank sync inside the replay/launch path.

2 ranks, one container, ccl backend (TP2-sim). Writes /root/p135g_bench.json.
Probe scratch only; never ships.
"""
import json
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("xccl")
    r = dist.get_rank()
    torch.xpu.set_device(r)

    x = torch.ones(64, device="xpu") * (r + 1)

    # tiny no-ccl graph (1 mul)
    g_tiny = torch.xpu.XPUGraph()
    with torch.xpu.graph(g_tiny):
        x = x * 1.0001

    # ccl graph (1 all_reduce)
    g_ar = torch.xpu.XPUGraph()
    with torch.xpu.graph(g_ar):
        dist.all_reduce(x)

    # big no-ccl graph (~15 ms device): 20k tiny muls
    big = torch.zeros(256, device="xpu")
    g_big = torch.xpu.XPUGraph()
    with torch.xpu.graph(g_big):
        for _ in range(20000):
            big = big * 1.0001

    dist.barrier()
    torch.xpu.synchronize()
    res = {"rank0_view": r == 0}

    # B0: isolated tiny replay baseline (queue empty)
    for _ in range(3):
        g_tiny.replay()
    torch.xpu.synchronize()
    dist.barrier()
    vals = []
    for _ in range(10):
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        g_tiny.replay()
        vals.append((time.perf_counter() - t0) * 1e6)
    torch.xpu.synchronize()
    res["B0_tiny_isolated_us_r%d" % r] = round(sorted(vals)[5], 1)

    # B2 (before B1 so B1's backlog is clean): big graph full wall
    torch.xpu.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    g_big.replay()
    torch.xpu.synchronize()
    res["B2_big_full_wall_us_r%d" % r] = round((time.perf_counter() - t0) * 1e6, 1)

    # B1: enqueue big (no sync), then time tiny replay
    torch.xpu.synchronize()
    dist.barrier()
    g_big.replay()  # enqueue only; device starts draining
    t0 = time.perf_counter()
    g_tiny.replay()
    tiny_after_big = (time.perf_counter() - t0) * 1e6
    torch.xpu.synchronize()
    res["B1_tiny_after_big_enqueued_us_r%d" % r] = round(tiny_after_big, 1)

    # A0: ccl graph replay, peers in lockstep
    torch.xpu.synchronize()
    dist.barrier()
    vals = []
    for _ in range(10):
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        g_ar.replay()
        vals.append((time.perf_counter() - t0) * 1e6)
    torch.xpu.synchronize()
    dist.barrier()
    res["A0_ar_isolated_us_r%d" % r] = round(sorted(vals)[5], 1)

    # A2: ccl replay with peer held back 200 ms
    torch.xpu.synchronize()
    dist.barrier()
    if r == 1:
        time.sleep(0.2)
    t0 = time.perf_counter()
    g_ar.replay()
    a2 = (time.perf_counter() - t0) * 1e6
    torch.xpu.synchronize()
    dist.barrier()
    res["A2_ar_peer_held_200ms_us_r%d" % r] = round(a2, 1)

    # A3: control - no-ccl tiny replay with peer held back
    torch.xpu.synchronize()
    dist.barrier()
    if r == 1:
        time.sleep(0.2)
    t0 = time.perf_counter()
    g_tiny.replay()
    a3 = (time.perf_counter() - t0) * 1e6
    torch.xpu.synchronize()
    dist.barrier()
    res["A3_tiny_peer_held_200ms_us_r%d" % r] = round(a3, 1)

    json.dump(res, open("/root/p135g_bench_r%d.json" % r, "w"), indent=1)
    if r == 0:
        print(json.dumps(res, indent=1), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
