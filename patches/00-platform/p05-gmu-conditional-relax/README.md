# p05 — GMU conditional relax (VLLM_ARC_GMU_RELAX)

## What

`ArcXpuPlatform.check_and_update_config`: when `VLLM_ARC_GMU_RELAX` is set
to a fraction (0,1] and `gpu_memory_utilization` still holds the engine
default (0.92), relax it to the env value with one INFO line. Default 0 =
disabled. An explicit `--gpu-memory-utilization` is always respected.

## Why

The D champion's certified B70 posture runs `--gpu-memory-utilization 0.8`
(serve flag `V1227_GMU=0.8` in `serve/boot_v1227_restore.sh`, plan Appendix
F); 0.8 on B70 is the sanctioned value (standing directive Appendix I #11).
The env keeps the programmatic path parity without forcing a flag.

## Validation

results/2026-10-27-p1-platform/ — serve posture line carries the relaxed
utilization when the env is set; unset env leaves 0.92 untouched.
