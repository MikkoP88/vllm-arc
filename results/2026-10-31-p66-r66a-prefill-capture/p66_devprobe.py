#!/usr/bin/env python3
# p66_devprobe.py — P6.6 device-truth probe: what is the actual GEMM
# ceiling of the rig's B70s, and the model's true param count? Grounds the
# prefill MFU math (spec sheets disagree with the measured 2036 tok/s: at
# 2x27 GFLOP/token that is ~55 TFLOPS/card, above some published B70 FP16
# peaks — so either the card has more XMX than listed, or the params are
# fewer, or the tok/s figure carries contention error).
import json
import time

import torch

torch.xpu.set_device(0)
p = torch.xpu.get_device_properties(0)
print(f"DEVICE name={p.name} total_mem={p.total_memory/2**30:.1f} GiB "
      f"max_cus={getattr(p, 'max_compute_units', '?')} "
      f"eu_count={getattr(p, 'eu_count', '?')} "
      f"cores={getattr(p, 'num_cores', '?')}")
try:
    import subprocess
    out = subprocess.run(["xpu-smi", "discovery"], capture_output=True,
                         text=True, timeout=20).stdout
    print("XPUSMI_DISCOVERY_BEGIN")
    print(out[:2000])
    print("XPUSMI_DISCOVERY_END")
except Exception as e:  # noqa: BLE001
    print(f"xpu-smi unavailable: {e}")


def bench(m, n, k, dtype, iters=20, warm=5):
    a = torch.randn(m, k, dtype=dtype, device="xpu")
    b = torch.randn(k, n, dtype=dtype, device="xpu")
    for _ in range(warm):
        c = a @ b
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tf = 2.0 * m * n * k / dt / 1e12
    print(f"GEMM {dtype} {m}x{n}x{k}: {dt*1e3:.2f} ms -> {tf:.1f} TFLOPS "
          f"(per card; TP2 fleet peak = 2x for the model)")
    del a, b, c
    return tf


for dt_ in (torch.float16, torch.bfloat16):
    for shape in ((8192, 8192, 8192), (4096, 4096, 4096),
                  (8192, 4096, 16384)):
        bench(*shape, dt_)

# fp8 dequant->fp16 pattern (the W8A16 reroute): fp8 weight upcast cost is
# part of the engine path; measure a dequant+matmul combo for reference.
try:
    w8 = torch.randn(16384, 4096, dtype=torch.float8_e4m3fn, device="xpu")
    x = torch.randn(8192, 16384, dtype=torch.float16, device="xpu")
    for _ in range(3):
        c = x @ w8.to(torch.float16)
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = x @ w8.to(torch.float16)
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / 10
    print(f"W8A16-pattern dequant+matmul 8192x4096x16384: {dt*1e3:.2f} ms "
          f"-> {2.0*8192*4096*16384/dt/1e12:.1f} TFLOPS incl dequant")
except Exception as e:  # noqa: BLE001
    print(f"fp8 pattern probe failed: {e}")

cfg = json.load(open("/models/target/config.json"))
keep = {k: cfg[k] for k in cfg if not isinstance(cfg[k], (dict, list))}
print("MODEL_CONFIG_BEGIN")
print(json.dumps(keep, indent=0)[:2500])
print("MODEL_CONFIG_END")

h = cfg.get("hidden_size")
n_layers = (cfg.get("num_hidden_layers")
            or len(cfg.get("architectures") and cfg.get("text_config", {}))
            or cfg.get("text_config", {}).get("num_hidden_layers"))
inter = cfg.get("intermediate_size") or cfg.get("text_config", {}).get(
    "intermediate_size")
print(f"DERIVED hidden={h} layers={n_layers} intermediate={inter}")
print("P66_DEVPROBE: DONE")
