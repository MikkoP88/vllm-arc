"""r22a_evprobe.py — XPU Event.synchronize semantics micro-probe.
The product step shows a FLAT ~58 ms inside AsyncOutput.get_output's
copy_event.synchronize() (torch.cuda.Event(blocking=True) → torch.xpu.Event
under vllm's cuda→xpu aliasing) with an idle GPU. Discriminate:
  1. completed-event synchronize cost (idle GPU)
  2. query() spin cost on the same completed event
  3. waiting synchronize behind ~50 ms of enqueued matmuls
  4. Event(blocking=True) ctor acceptance + completed sync cost
Run inside arc-test on the product venv."""
import statistics
import time

import torch

dev = "xpu"
N = 30
a = torch.randn(1024, 1024, device=dev)
b = torch.randn(1024, 1024, device=dev)
(a @ b)
torch.xpu.synchronize()

print("== 1. completed-event synchronize (idle GPU) ==")
lat = []
for _ in range(N):
    e = torch.xpu.Event()
    (a @ b)
    torch.xpu.synchronize()
    e.record()
    t0 = time.perf_counter()
    e.synchronize()
    t1 = time.perf_counter()
    lat.append((t1 - t0) * 1e3)
print(f"n={N} p50={statistics.median(lat):.3f} ms min={min(lat):.3f} "
      f"p90={sorted(lat)[int(N*0.9)]:.3f} max={max(lat):.3f}")

print("== 2. query() spin on completed event ==")
lat = []
for _ in range(N):
    e = torch.xpu.Event()
    (a @ b)
    torch.xpu.synchronize()
    e.record()
    t0 = time.perf_counter()
    while not e.query():
        pass
    t1 = time.perf_counter()
    lat.append((t1 - t0) * 1e3)
print(f"n={N} p50={statistics.median(lat):.3f} ms min={min(lat):.3f} "
      f"max={max(lat):.3f}")

print("== 3. waiting synchronize behind ~50 ms enqueued matmuls ==")
big = torch.randn(4096, 4096, device=dev)
# calibrate
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(40):
    big @ big
torch.xpu.synchronize()
print(f"40x 4096^2 matmul wall = {(time.perf_counter()-t0)*1e3:.1f} ms")
lat = []
for _ in range(10):
    e = torch.xpu.Event()
    for _ in range(40):
        big @ big
    e.record()
    t0 = time.perf_counter()
    e.synchronize()
    t1 = time.perf_counter()
    lat.append((t1 - t0) * 1e3)
    torch.xpu.synchronize()
print(f"n=10 p50={statistics.median(lat):.3f} ms min={min(lat):.3f} "
      f"max={max(lat):.3f}")

print("== 4. Event(blocking=True) ctor + completed sync ==")
try:
    eb = torch.xpu.Event(blocking=True)
    (a @ b)
    torch.xpu.synchronize()
    eb.record()
    t0 = time.perf_counter()
    eb.synchronize()
    t1 = time.perf_counter()
    print(f"blocking ctor OK; completed sync = {(t1-t0)*1e3:.3f} ms")
    lat = []
    for _ in range(N):
        eb = torch.xpu.Event(blocking=True)
        (a @ b)
        torch.xpu.synchronize()
        eb.record()
        t0 = time.perf_counter()
        eb.synchronize()
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1e3)
    print(f"n={N} p50={statistics.median(lat):.3f} ms min={min(lat):.3f} "
          f"max={max(lat):.3f}")
except TypeError as ex:
    print("blocking kwarg REJECTED:", ex)

print("== 5. side-stream pattern (AsyncOutput replica) ==")
try:
    main = torch.xpu.Stream()
    copy = torch.xpu.Stream()
    lat = []
    for _ in range(N):
        with torch.xpu.stream(main):
            (a @ b)
        e = torch.xpu.Event()
        with torch.xpu.stream(copy):
            copy.wait_stream(main)
            e.record(copy)
        t0 = time.perf_counter()
        e.synchronize()
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1e3)
        torch.xpu.synchronize()
    print(f"n={N} p50={statistics.median(lat):.3f} ms min={min(lat):.3f} "
          f"max={max(lat):.3f}")
except Exception as ex:  # noqa: BLE001
    print("side-stream probe failed:", type(ex).__name__, ex)
print("EVPROBE_DONE")
