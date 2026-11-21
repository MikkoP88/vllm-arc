# r212d-diag — offline engine-shape knob ladder: knob-neutrality law

**Date**: 2026-11-21 · **Leg**: r212d-diag (arc7 Phase-8) · **Stack**:
build21 wheel + mx_kv overlay (`f34349e7…`) + `vllm-arc:v0.30.0-b70-
cert25zc` · **Run**: `r212d_launch1.log` 06:56-07:13 UTC · **Tool**:
`tools/ar/p147_mxkv/r212d_diag.sh` (6 cells, c32768, mxfp4, engine
geometry D=256/HREP=6/BLOCK=64, eshape, splits=auto, graph 32×50)

## Verdict

**PASS — every engine-vs-harness shape delta is performance-neutral
offline; the ~40× e2e gap is NOT in the op or its call shape.**

| cell | native_es | vs base | v2_es | canary |
|---|---|---|---|---|
| base | 1268.7 µs | 1.00× | 4086.9 µs | 0 |
| pool (POOLBLKS=11281) | 1272.2 µs | 1.00× | 4087.2 µs | 0 |
| btw0 (BTWIDTH=4100, zero pad) | 1276.6 µs | 1.01× | 4085.2 µs | 0 |
| btwS (BTWIDTH=4100, stale pad) | 1278.4 µs | 1.01× | 4085.1 µs | 0 |
| both (btwS+pool) | 1264.9 µs | 1.00× | 4087.4 µs | 0 |
| eng (both + MAXSK=cap) | 1271.9 µs | 1.00× | 4087.3 µs | 0 |

### KNOB-NEUTRALITY LAW

Reproducing the FULL engine call shape offline — pool over-allocated
to 11281 blocks (721994 tokens), block table padded to the
max_model_len-derived width 4100 with STALE random-valid ids,
capacity-priced max_seqlen_k 262400 — changes native's device time by
≤1% and dirties no canary. The kernel is fast at the engine's exact
shape; the stale-pad canary additionally proves the walk never reads
beyond-fill bt entries at this shape.

### The inversion that pins the gap on the invocation

At the same engine call shape the lanes INVERT offline vs e2e:

- offline: `native_es` 1271.9 µs beats `v2_es` 4087.3 µs — **3.2×
  faster**;
- e2e (r212b): native-frozen 497.8 ms/step vs all-Triton 76.1 ms —
  **6.5× slower** at c16384 n1.

The e2e native cost (~53 ms/layer vs 1.27 ms/call offline, ~40×)
cannot be the op: it survives split changes (r212c falsification),
max_seqlen_k changes, pool/bt-shape changes (this leg), and it
inverts the lane ordering the device itself produces at the same
shape. The 40× lives in HOW the engine invokes the captured op inside
the full-model XPU graph — around-kernel ops, graph interaction, or
invocation mechanics. Next instrument: in-situ kernel telemetry
(r212e torch-profiler bracket on the native posture, r67b recipe).

## Artifacts

`p204_diag_{base,pool,btw0,btwS,both,eng}_c32768.json`,
`r212d_{base,pool,btw0,btwS,both,eng}.log`, `r212d_launch1.log`.
Rig: `/root/build/p199/out/`, log `/root/build/p199/logs/`.
