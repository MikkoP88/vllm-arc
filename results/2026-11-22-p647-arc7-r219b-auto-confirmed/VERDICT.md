# r219b — AUTO-AT-128 CONFIRMED: the C++ heuristic is the posture (env pin retired); mxfp4 decode now 1.34x fp8 at 32K with zero configuration

**Date**: 2026-11-22 · **Leg**: r219b (arc7 Phase-8 auto confirmation) ·
**Stack**: the r219b wheel (md5 `e9f51f3b9f6d5c86824b89d357133f70` —
r219 + ELSE-branch `policy_split_cap` 64→128, block16 aligned) +
`vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay · **Run**:
`r219b_auto.log` 09:12–09:26 UTC (two boots, faces c8192/c16384/c32768
n1) · **Closes**: the split-depth line (r216 → r219b)

## Verdict

**GATE_PASS — `mx_get_num_splits` now lands the validated optimum by
itself: auto = 48.5/51.2/56.8 (within 0.08% of the pinned ns128 path on
every face; the explicit ns128r repro is −0.05%). `ARC_MXKV_NSPLIT` is
retired as a posture requirement — it remains a probe instrument.**

| cell | 8192 | 16384 | 32768 | tps @32K | chunks |
|---|---|---|---|---|---|
| auto (unset) | 48.5 (+0.03%) | 51.2 (+0.08%) | **56.8 (−0.02%)** | 65.4 | 53–54 |
| ns128r (pin) | 48.5 (−0.07%) | 51.2 (+0.07%) | 56.8 (−0.05%) | 65.3 | 53–55 |

Telemetry prints `splits=auto` / `splits=128` as designed; 0 errors.

### The auto chain at the engine bucket (now observed twice)

`341 (bw) → 130 (reduce cap 13×W/H_q, W=120, H_q=12) → 128 (policy cap,
the ELSE branch at page 64)`. Both caps sit AT the optimum because the
r219 tail cells proved 192/256 regress — the heuristic cannot overshoot
into the bad region even at maxed geometry, and short ctx stays
workload-bounded (`kv_tiles/12`).

### State of the split-depth lever: CLOSED

Surface measured 8→256 across r216/r216b/r217/r217b/r219/r219b: optimum
128 (A/A-confirmed three times: 56.7/56.8/56.8 + auto's 56.8),
−19.2% vs the old auto-40, 1.34x fp8 at 32K (from 1.65x), tps +28.8%
with the long-ctx droop flattened. The remaining gap to fp8 (14.3
ms/step) is the r218 decomposition: floor 5.6 + union 8.7 — the
kv_tile-16→64 walk surgery (MMA N-tile, the r212c2 road, offline-first
in p204) is the next and last big lever of this line.

**Reverted**: nothing — the r219b wheel (e9f51f3b) is the candidate
posture wheel; overlay env pin inert when unset.
