# VERDICT — P6.47 Phase 7 attempt 1: v7 FA-shaped walk (r192)

**LEG**: 2026-11-19-p647-v7a-attempt1 (rig bank /root/build/p647/v7A26)
**IMAGE**: vllm-arc:v0.30.0-b70-cert25zb + in-container swap of the
v7-bearing triton_mx_decode.py (ephemeral --rm container; the image is
untouched — r192 mechanism)
**KERNEL**: `_mx_decode_split_dot2` (commit 90ba1725) — the v5 split-dot
with the tile loop restructured into raw-tile double buffering: next
tile's block ids prefetched at the loop top, raw packed bytes
prefetched under the softmax/p.V cover, dequant at consume. Behind
ARC_MXKV_V7=1 (call-time read; default off).

## Correctness — PERFECT

d = max|out7 − out5| = **0.000e+00 on all 15 cells** (3 formats × {G1
sub-tile 200, G1 1K, G1 8K, G2 mixed [4096,960], G8 32K}); all finite.
The restructure is BITWISE semantics-preserving: same op order per
tile, identity casts on the packed lanes. V7AB_PASS.

## Perf — REFUTED at every cell (ms/call, v5 → v7)

| cell | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| G1 1K | 0.563 → 0.738 (0.76x) | 0.612 → 0.646 (0.95x) | 0.620 → 0.682 (0.91x) |
| G1 8K | 1.856 → 2.231 (0.83x) | 2.492 → 2.654 (0.94x) | 2.612 → 2.810 (0.93x) |
| G1 32K | 4.871 → 7.472 (0.65x) | 7.620 → 9.613 (0.79x) | 7.627 → 10.116 (0.75x) |
| G8 8K | 10.460 → 15.722 (0.67x) | 15.841 → 20.046 (0.79x) | 16.058 → 21.124 (0.76x) |
| G8 32K | 35.609 → 58.035 (0.61x) | 58.270 → 75.051 (0.78x) | 58.432 → 79.153 (0.74x) |

Threshold was <= 1.0 ms at G1@32K; attempt 1 moves AWAY from it
(7.5-10.1 ms).

## Mechanism — the toolchain verdict

Manual software pipelining by program order DOES NOT WORK in
Triton-on-XPU: the compiler sinks gather loads to their use sites, so
the "prefetch" loads issued early in program order are re-scheduled to
the next iteration's dequant — nothing overlaps, and the double buffer
only adds live state across the dots (raw tiles + address chains
carried through the MMA block → register pressure, likely spills; the
uniform 0.61-0.95x regression across formats/shapes is a constant
per-tile overhead signature). Combined with A6/r177 (num_stages is a
no-op on the data-dependent bt→slot→gather chain), the in-language
latency-hiding toolbox for this chain is now EMPTY — both compiler
mechanisms (num_stages, manual restructure) are banked refutations.

## Disposition

- v7 attempt 1 CLOSED (refuted). The kernel stays in-tree behind
  ARC_MXKV_V7=1 (default off, bitwise-clean) as the attempt-1 record;
  certified defaults untouched.
- **Attempt 2 opened: deep-split re-sweep (TILE_FLOOR).** The v6 >=4
  serial-tile floor was tuned pre-K2/K3 when per-tile scalar-ALU
  dominated; with the packed lanes the serial chain latency is the
  larger residual, and the chain can instead be REMOVED by
  parallelism: TILE_FLOOR=1 at 32K = 1024 splits x 1 tile (grid
  4096 programs, zero serial chain; combine cost rises — the sweep
  finds the U-curve minimum). One constexpr on the certified v5
  (bit-frozen at default 4), env ARC_MXKV_V5_TFLOOR; split cap
  ARC_MXKV_V5_SPLITS already sweeps to 1024. r193 banks it.

**OUTCOME: correctness gate PERFECT (d=0), perf REFUTED 0.61-0.95x,
toolchain law banked (no in-language prefetch control on XPU),
attempt 2 (deep-split re-sweep) opened.**
