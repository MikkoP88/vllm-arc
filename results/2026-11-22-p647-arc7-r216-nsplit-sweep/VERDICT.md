# r216 — the engine split-depth sweep: ns64 = −11.45% — the biggest single e2e win of the arc7 MX line

**Date**: 2026-11-22 · **Leg**: r216 (arc7 Phase-8 e2e split sweep) ·
**Stack**: r212aa wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5 `9a01790b...`) +
`vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay bridge with the
r216 NSPLIT pin (`num_splits_kv` from `ARC_MXKV_NSPLIT`; unset = the
C++ auto heuristic; serve passthrough added — fork-owned overlay edit,
local mirror `vllm_gaudi/xpu/patches/mx_kv/native_decode.py`) ·
**Run**: `r216_nsplit.log` 03:08–03:27 UTC (three boots, c32768 n1)

## Verdict

**GATE_PASS — anchor validated (ns40 = 70.8, +0.85% vs the 70.2
reference: auto really picks ~40 and the knob is live end to end),
all cells clean (0 errors, rows=148).**

| cell | itl p50 (ms/step) | vs reference | reading |
|---|---|---|---|
| ns40 (anchor) | 70.8 | +0.85% | reproduces auto — plumbing proven |
| ns8 | 155.4 | **+121.3%** | parallelism-starved (4x the sequential walk per WG) |
| ns64 | **62.2** | **−11.45%** | deeper splits win big at engine shape |

## Reading

1. **The reduce cap 40 is NOT the engine optimum** — pinned 64
   improves mxfp4 32K decode by 11.45% (70.2 → 62.2 ms/tok-step), the
   largest single-lever e2e gain banked on this line. The r209/r210
   "40 optimal" verdict was probe-priced on pre-template-split wheels;
   this is a third instance of the sign-inversion class (probe-priced
   optimum inverting at engine shape), caught by the lawful e2e
   re-price.
2. **The mechanism is parallelism headroom**: at the capacity-frozen
   bucket (max_seqlen_k=262400, rows=148, 2 KV heads/rank), 40 splits
   = 11,840 WGs over ~130-ish slots ≈ 90 waves; 64 splits = 18,944 WGs
   ≈ 145 waves of FINER grain — shorter critical-path WGs hide
   launch/tail skew better. ns8's catastrophic +121% brackets the
   surface from below (latency-bound).
3. **Open questions carried to r216b** (in flight): the A/A
   confirmation on a fresh boot (the r213 law: single-boot deltas are
   provisional), the 8K/16K posture faces (the posture must not
   regress the 48.5/55.9 auto baselines — the r210 workload-bound
   warning about short-ctx over-splitting), and ns96 (the sweep
   stopped at the heuristic's 64 cap, not a kernel limit — the r208
   fork-owned reduce handles deep S).

**Not yet the shipping posture** — the env pin is the instrument; the
durable fix (C++ reduce-cap formula priced for the engine geometry)
rides the next wheel iteration. See r216b for the confirmation and the
posture decision.

**Reverted**: nothing — env-instrument leg; overlay edit is inert when
the env is unset.
