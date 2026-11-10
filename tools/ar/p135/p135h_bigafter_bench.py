#!/usr/bin/env python3
"""p135h_bigafter_bench.py -- P6.35 mechanism discriminator v3.

p135g falsified queue-throttle-for-tiny, ccl-in-graph, and peer-rendezvous.
The untested cell: launching a BIG graph while a BIG graph is still queued
(serving state: async scheduling runs one step ahead, so the ~6k-node verify
replay lands on a queue holding the previous step's ~6k-node graph).

  B5: control, empty queue -> us-scale launch.
  B4: enqueue big #1 (no sync), then launch big #2 into non-empty queue.
      ~0 => queue depth huge, mechanism elsewhere.
      ~first-graph-drain => launch waits for prior work to retire
      (host/device serialized through the replay call).
  B4b: same but 5 ms after the first enqueue (partially drained).
  B6: tiny graph onto big backlog (re-check with 20k-node backlog).

Single rank, no xccl. All graphs captured BEFORE any measurement
(torch.xpu.graph capture synchronizes + empties cache). Writes
/root/p135h_bench.json. Probe scratch only.
"""
import json
import time

import torch


def build_chain(base, n):
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        b = base
        for _ in range(n):
            b = b * 1.0001
    return g


def main():
    assert torch.xpu.device_count() > 0
    big1 = torch.zeros(256, device="xpu")
    big2 = torch.zeros(256, device="xpu")
    tiny = torch.zeros(64, device="xpu")

    g1 = build_chain(big1, 20000)
    g2 = build_chain(big2, 20000)
    gt = build_chain(tiny, 1)
    torch.xpu.synchronize()

    res = {}
    for tag, g in (("g1", g1), ("g2", g2)):
        for _ in range(2):
            g.replay()
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.xpu.synchronize()
        res["%s_full_wall_us" % tag] = round((time.perf_counter() - t0) * 1e6, 1)

    # B5: empty-queue launch (host enqueue only)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    g1.replay()
    res["B5_empty_queue_launch_us"] = round((time.perf_counter() - t0) * 1e6, 1)
    torch.xpu.synchronize()

    # B4: big onto non-empty queue
    for trial in range(3):
        torch.xpu.synchronize()
        g1.replay()  # enqueue only
        t0 = time.perf_counter()
        g2.replay()  # big onto non-empty queue
        b4 = (time.perf_counter() - t0) * 1e6
        torch.xpu.synchronize()
        res["B4_big_after_big_us_t%d" % trial] = round(b4, 1)

    # B4b: big onto a partially drained queue (5 ms after first enqueue)
    for trial in range(3):
        torch.xpu.synchronize()
        g1.replay()
        time.sleep(0.005)
        t0 = time.perf_counter()
        g2.replay()
        res["B4b_after_5ms_gap_us_t%d" % trial] = round(
            (time.perf_counter() - t0) * 1e6, 1)
        torch.xpu.synchronize()

    # B6: tiny onto big backlog
    torch.xpu.synchronize()
    g1.replay()
    t0 = time.perf_counter()
    gt.replay()
    res["B6_tiny_after_big_us"] = round((time.perf_counter() - t0) * 1e6, 1)
    torch.xpu.synchronize()

    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open("/root/p135h_bench.json", "w"), indent=1)
    print("P135H_OK")


if __name__ == "__main__":
    main()
