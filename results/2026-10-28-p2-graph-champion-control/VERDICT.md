# VERDICT — champion control: graphs + MTP SERVE on this rig today → the poison is our stack; root cause found in the capture-stream diff

## The control leg (lsv-test = vllm-iexperimental:v1.0.0, 2026-10-28 ~04:15)

Booted the D-champion script verbatim (serve_lmc_v133.sh): FULL_DECODE_ONLY
graphs via --compilation-config (64 sizes, captured in 26 s), MTP k4,
--async-scheduling, LMCache kv-transfer, TP2 — on vllm
0.26.1.dev0+g568afb3a1.d20261018.xpu, torch **2.12.0+xpu**, oneAPI 2025.3.2.

Readout: **five real /v1/completions all 200 (2.1-3.6 s each), health 200,
ZERO "recording to a command graph" errors, 175× 200 OK over the log** —
the exact posture that scores agg 274.41 serves on this hardware+driver
TODAY. The environment is unchanged; defect-b is a FORK-STACK artifact.

## The root cause (code diff, wheels side by side)

| | champion (0.26.1, torch 2.12) | ours (0.30, torch 2.13) |
|---|---|---|
| capture call | `torch.cuda.graph(graph, self.pool)` — NO stream arg | `torch.cuda.graph(graph, self.pool, stream=self._capture_stream(desc))` |
| effective capture stream | `torch.xpu.graph.default_capture_stream` (the SIDE stream) | `current_stream()` = the worker's device DEFAULT |

On torch 2.13.0+xpu the capture stream stays graph-associated after
capture_end ("poison enters during capture",
results/2026-10-27-p2-graph-defectb-answer: 27/27 boot captures finish
clean, nothing GPU-touching runs before the first request, yet the default
is recording-dead at the first real op). Capturing on the DEFAULT = the
serving queue is dead from boot; the first real op on it (AsyncOutput copy
at request completion) starts the rounds-1..7 death cascade. Replays were
always fine (k1k 200 in every poisoned round) — only the capture-stream
choice poisons. The prefix-off leg (flip with enable_prefix_caching=False)
is consistent: the trigger was never the prefix write.

## Fix (commit 94e39c3d, image building)

`CudaGraphManager._capture_stream` wrapped to return
`torch.xpu.graph.default_capture_stream` — captures ride the side stream,
champion discipline; the serving default never enters capture. p12b
hygiene layers stay installed as the integrity net (dormant when no flip
ever comes).

Stack deltas recorded for the final report: vllm 0.26.1→0.30.0 AND torch
2.12.0+xpu→2.13.0+xpu (oneAPI 2025.3→2026.0) both differ; the control
proves the lane is viable on this driver with the older pair, and the
capture-stream diff is the demonstrated mechanism on ours.

Raws: lsv_serve_full.log (this dir; rig container log docker-cp'd before
container stop).
