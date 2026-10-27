# p01 — TEST gate

P0 gate (plan §10) — run inside the built image on the rig:

1. `python -c "import vllm_gaudi"` — package imports on a habana-less host.
2. `python -c "from vllm.platforms import current_platform; \
   print(type(current_platform).__module__)"` — resolves
   `vllm_gaudi.xpu.platform` (ArcXpuPlatform) on the dual-B70 rig.
3. Import vllm on the rig image WITHOUT Arc devices (CPU-only exec) — HPU
   surfaces dormant, platform falls back (UnspecifiedPlatform/cpu), no crash.
4. In-image serve boot (eager) of qwen3.8-27b-fp8 — P1 gate precondition.

PASS = all four green, evidence mirrored under `results/2026-10-27-p1-platform/`.
