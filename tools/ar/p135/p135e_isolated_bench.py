#!/usr/bin/env python3
"""p135e_isolated_bench.py -- P6.35 D1 discriminator v2 (clean device).

p135b/p135d ran while W14 served on the same GPUs -> contention + possible
queue backpressure contaminated host-cost numbers. This bench measures, on an
IDLE device:

  - ISOLATED replay host cost: sync, then time replay() alone (no sync after,
    empty queue) -> pure host enqueue/launch cost per call, 20 reps median;
  - first replay() right after capture (re-instantiate-per-replay test);
  - full replay wall (replay+sync) -> host+device;
  - eager control: N individual mul enqueues (host) + device drain.

If isolated replay is O(1)-ish at 6000 nodes, torch 2.13 replay is fine and
the serving RFG wall (28.4 ms) must be re-attribute. If isolated replay is
~5 us/node * N every call, replay re-instantiates per call -> the fix is a
vllm_gaudi-level cached-exec/launch patch.

Writes /root/p135e_bench.json. Probe scratch only; never ships.
"""
import json
import statistics
import time

import torch


def build(n, keep_graph):
    g = torch.xpu.XPUGraph(keep_graph=keep_graph)
    x = torch.zeros(64, device="xpu")
    s = torch.xpu.Stream()
    with torch.xpu.graph(g, stream=s):
        for _ in range(n):
            x = x * 1.0001
    torch.xpu.synchronize()
    return g, x


def isolated_replay_us(g, reps=20):
    vals = []
    for _ in range(reps):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        g.replay()
        vals.append((time.perf_counter() - t0) * 1e6)
    torch.xpu.synchronize()
    return round(statistics.median(vals), 1)


def full_wall_us(g, reps=10):
    vals = []
    for _ in range(reps):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.xpu.synchronize()
        vals.append((time.perf_counter() - t0) * 1e6)
    return round(statistics.median(vals), 1)


def main():
    assert torch.xpu.device_count() > 0, "no XPU device visible"
    out = {"torch": torch.__version__, "cases": []}

    for n in (1000, 6000):
        case = {"nodes": n}

        # A: keep_graph=False (serving path via vllm shim), first replay
        # right after capture_end, then isolated/warmed.
        g_a, _ = build(n, False)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        g_a.replay()
        case["A_first_replay_after_capture_ms"] = round(
            (time.perf_counter() - t0) * 1e3, 2)
        torch.xpu.synchronize()
        case["A_isolated_replay_host_us"] = isolated_replay_us(g_a)
        case["A_full_wall_us"] = full_wall_us(g_a)

        # C: keep_graph=True + explicit instantiate BEFORE first replay.
        g_c, _ = build(n, True)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        g_c.instantiate()
        case["C_instantiate_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        case["C_isolated_replay_host_us"] = isolated_replay_us(g_c)
        case["C_full_wall_us"] = full_wall_us(g_c)

        # E: eager control on empty queue.
        x = torch.zeros(64, device="xpu")
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            x = x * 1.0001
        case["E_eager_enqueue_host_us"] = round(
            (time.perf_counter() - t0) * 1e6, 1)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            x = x * 1.0001
        torch.xpu.synchronize()
        case["E_eager_full_wall_us"] = round(
            (time.perf_counter() - t0) * 1e6, 1)

        out["cases"].append(case)
        print(case, flush=True)
        g_a.reset()
        g_c.reset()
        torch.xpu.synchronize()

    with open("/root/p135e_bench.json", "w") as f:
        json.dump(out, f, indent=1)
    print("P135E_OK")


if __name__ == "__main__":
    main()
