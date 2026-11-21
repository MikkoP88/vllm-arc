# r212f — mseq ladder: SPLIT STARVATION CONFIRMED at the kernel level

**Date**: 2026-11-21 · **Leg**: r212f (arc7 Phase-8) · **Stack**:
build21 wheel + `vllm-arc:v0.30.0-b70-cert25zc` · **Run**:
`r212f_launch1.log` 07:27-07:33 UTC · **Tool**:
`tools/ar/p147_mxkv/r212f_mseq.sh` (3 cells, c32768, mxfp4, engine
geometry D=256/HREP=6/BLOCK=64, eshape, splits=auto, graph 32×50;
`P204_MAXSK` int-literal support)

## Verdict

**PASS — the split-starvation hypothesis is CONFIRMED as the cause
of the 40× e2e native regression, and the fix value restores full
speed.**

| cell | max_seqlen_k | native_es | canary |
|---|---|---|---|
| ctx | 32768 (true ctx) | 1269.6 µs | 0 |
| m9 | 9 (the engine's DUMMY-diluted freeze) | **36326.9 µs** | 0 |
| cap | 262400 (the r212f bridge fix) | 1265.4 µs | 0 |

**RATIO m9/ctx = 28.61×.** One scalar kernel argument — the value
the engine freezes at capture — moves the same op at the same shape
from 1.27 ms to 36.3 ms per call. `native_auto` (the non-eshape
bridge path) blows up identically (36317 µs), confirming the C++
auto-split heuristic prices `kv_tiles = ceil(max_seqlen_k/16)` from
the arg: at 9 → 1 tile → `num_kv_splits = 1` frozen into the graph →
each (row, kv-head) walks its full real KV on ONE workgroup.

### The closed causal chain (r212a→r212f)

1. Capture dummy dilutes `metadata.max_seq_len` to 9/13/18/5
   (r212b telemetry).
2. The bridge passed it as `max_seqlen_k` (unchanged through the
   no-op r212c — see the r212f audit).
3. Splits freeze at 1 per captured graph (r212b law).
4. In-graph cost ≈ 28.6× the split-priced kernel (this leg) —
   ~36 ms/call at c32768, ~29 ms/layer class at engine buckets.
5. 16 MX layers ≈ the ~465 ms kineto-silent graph wall per 498 ms
   step (r212e's step timeline) — the e2e A/B's 6.5×/8.6×/1.4×.
6. Correctness was never at risk: per-row device-side limits
   (cu_seqlens_k) bound the true walk (r212b accuracy at 32K with
   max_seqlen_k=9; canaries 0 here at 28.6× slowdown).

The earlier "falsification" was a no-op (the telemetry printed the
capacity while the call passed the diluted value); the r212d
ctx/cap cells were blind to it by construction. Batch-independence
of the e2e cost (n8 = +12% over n1) is row-parallel saturation of
the un-split walk, as reopened-reading predicted.

### Fix

The bridge now passes `max_seqlen_k = bt_width × block_size`
(capacity, 262400) — the `cap` cell: 1265.4 µs, i.e. the fix value
is FREE at the kernel level (0.3% under ctx) and covers the
worst-case replay walk; short rows early-exit their splits.
E2E validation of the real fix: r212g (m4qa, the r212b faces).

## Artifacts

`p204_mseqf_{ctx,m9,cap}_c32768.json`, `r212f_{ctx,m9,cap}.log`,
`r212f_launch1.log`. Rig: `/root/build/p199/out/`, log
`/root/build/p199/logs/`.
