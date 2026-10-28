"""p10_probe.py — mechanism evidence for the GDN metadata fast path.

Verifies on the product torch/XPU stack, behind a busy in-order stream:
  A. `gpu_tensor[cpu_bool_mask, :k]` (core gdn_attn build's row select)
     blocks the CPU until the enqueued pipeline drains (implicit
     synchronous H2D of the mask).
  B. `mask_gpu.nonzero()` + `index_select` (the fast path's replacement)
     returns immediately — async launches only.
  C. Both produce identical rows.
Run inside arc-test."""
import time

import torch

dev = "xpu"
torch.manual_seed(0)
big = torch.randn(4096, 4096, device=dev)
bt = torch.randint(0, 1000, (64, 8), device=dev, dtype=torch.int32)
na = torch.randint(1, 5, (64,), device=dev, dtype=torch.int32)
mask_cpu = torch.zeros(64, dtype=torch.bool)
mask_cpu[:9] = True  # spec rows at the front, like the scheduler's layout
mask_gpu = mask_cpu.to(dev)


def busy():
    for _ in range(40):
        big @ big


# calibrate: pipeline length vs enqueue time
torch.xpu.synchronize()
t0 = time.perf_counter()
busy()
enq_ms = (time.perf_counter() - t0) * 1e3
torch.xpu.synchronize()
pipe_ms = (time.perf_counter() - t0) * 1e3
print(f"calibration: pipeline={pipe_ms:.0f}ms enqueue={enq_ms:.1f}ms", flush=True)

# A. CPU-mask advanced index behind the busy stream (core build's op)
torch.xpu.synchronize()
busy()
t0 = time.perf_counter()
sel_a = bt[mask_cpu, :5]
wa = (time.perf_counter() - t0) * 1e3
torch.xpu.synchronize()
print(f"A cpu-mask index wall={wa:.1f}ms  (blocked behind stream: "
      f"{wa > pipe_ms * 0.5})", flush=True)

# A2. same for num_accepted[mask_cpu]
torch.xpu.synchronize()
busy()
t0 = time.perf_counter()
na_a = na[mask_cpu]
wa2 = (time.perf_counter() - t0) * 1e3
torch.xpu.synchronize()
print(f"A2 cpu-mask index (1-D) wall={wa2:.1f}ms  "
      f"(blocked: {wa2 > pipe_ms * 0.5})", flush=True)

# B. fast-path replacement behind the same busy stream
torch.xpu.synchronize()
busy()
t0 = time.perf_counter()
rows = mask_gpu.nonzero(as_tuple=False).squeeze(1)
sel_b = bt.index_select(0, rows)[:, :5]
na_b = na.index_select(0, rows)
wb = (time.perf_counter() - t0) * 1e3
torch.xpu.synchronize()
print(f"B nonzero+index_select wall={wb:.2f}ms (async)", flush=True)

# C. equality
print("C equal(bt):", torch.equal(sel_a, sel_b),
      " equal(na):", torch.equal(na_a, na_b), flush=True)

# D. idle-GPU repeat of A (no enqueued work) — pure op cost
torch.xpu.synchronize()
t0 = time.perf_counter()
sel_a2 = bt[mask_cpu, :5]
torch.xpu.synchronize()
print(f"D idle cpu-mask index wall={(time.perf_counter()-t0)*1e3:.2f}ms",
      flush=True)
print("P10_PROBE_DONE", flush=True)
