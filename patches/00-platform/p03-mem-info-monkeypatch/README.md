# p03 — torch.accelerator.get_memory_info monkeypatch (XPU/Arc)

## What

Module-level monkeypatch in `vllm_gaudi/xpu/platform.py` (the gaudi-idiom
spot — upstream `vllm_gaudi/platform.py:26-34` does the same for HPU):
`torch.accelerator.get_memory_info` (and `torch.accelerator.memory`'s copy)
routes XPU queries to `torch.xpu.mem_get_info`; non-XPU devices pass through
to the original symbol.

## Why

On Arc B70 the accelerator-level query flips to a static view after
`import vllm`: free reports exactly 0, total shrinks 31.89→30.3 GiB.
`MemorySnapshot.measure` reads that query → `request_memory` rejects an
idle GPU → `XPUWorker.init_device` aborts every EngineCore boot.
`torch.xpu.mem_get_info` keeps the live driver view in every context probed
(radiance evidence, plan Appendix A). Core's own wrapper
(`vllm/platforms/xpu.py`, installed at import via `_C_cache_ops
.getMemoryInfo`) suffers the same flip — ours is installed later (at
platform-resolution time) so it wins; core's is kept for passthrough.

## Provenance

Radiance `patches.py:patch_xpu_accelerator_memory_info` (donor HEAD c24eeed)
— newer fixed shared component (plan ledger #19), radiance-proven on the
0.30 dual-B70 stack; reshaped into the gaudi module-level platform idiom.

## Validation

results/2026-10-27-p1-platform/ — boot log must show EngineCore passing the
free-memory gate on an idle GPU (the failure this patch removes), plus a
direct probe: `import vllm; torch.accelerator.get_memory_info()` returns
nonzero free before weights load.
