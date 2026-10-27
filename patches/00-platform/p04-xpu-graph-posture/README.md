# p04 — XPU graph posture

## What

`ArcXpuPlatform.check_and_update_config` logs the effective graph posture
(cudagraph_mode, capture_sizes, VLLM_XPU_ENABLE_XPU_GRAPH) once per phase.
Core's `XPUPlatform.check_and_update_config` already enforces the env law
(disables graphs when `VLLM_XPU_ENABLE_XPU_GRAPH!=1` / unsupported torch) —
inherited unchanged; this patch makes every serve log state its posture.

## Why

The whole P2 lane (graph+MTP defect pair, plan Appendix C) turns on graph
posture; legs must be auditable from the boot log alone (env/log
normalization law, plan §7). The standing rig law (VLLM_XPU_ENABLE_XPU_GRAPH=1
on all graph legs) is cited, not re-implemented.

## Scope note

FULL-pool sizing (54/64) and the capture-growth pre-size law (radiance,
plan Appendix D) land with the P2 attention/graph fixes (p12/p32) — they are
graph-content laws, not posture plumbing.

## Validation

results/2026-10-27-p1-platform/ — boot log carries the posture line with the
leg's actual mode (eager at P1).
