# r217b — ns128 A/A-CONFIRMED as the mxfp4 engine posture (spread ≤0.13%); 32K gap to fp8 now 1.34x, tps droop flattened

**Date**: 2026-11-22 · **Leg**: r217b (arc7 Phase-8 posture
confirmation) · **Stack**: the r217 wheel (md5
`9ded4fb67e15ba092aac607d58db1b72`) +
`vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay with
`ARC_MXKV_NSPLIT=128` · **Run**: `r217b_aa.log` 06:44–06:52 UTC (one
fresh boot, faces c8192/c16384/c32768 n1) · **Confirms**: r217
(`../2026-11-22-p647-arc7-r217-nsplit-ceiling-lift/`)

## Verdict

**GATE_PASS — ns128 confirmed on a fresh boot: A/A spread −0.02% /
+0.08% / +0.13% (8K/16K/32K) vs the r217 single-boot cells, acceptance
healthy (chunks 52–57, tps 66.7 @32K), 0 errors. The r213 single-boot
law is satisfied; the posture banks.**

| ctx | r217 ns128 | r217b ns128b | spread | tps (b) | chunks |
|---|---|---|---|---|---|
| 8192 | 48.5 | 48.5 | −0.02% | 71.7 | 56 |
| 16384 | 51.2 | 51.2 | +0.08% | 67.1 | 57 |
| 32768 | 56.7 | 56.8 | +0.13% | **66.7** | 52 |

- **@32K: 56.8 vs the fp8 bar 42.5 = 1.34x** (auto was 1.65x, ns64
  1.46x); −19.1% vs auto; tps 66.7 vs auto 51.8 = **+28.8%**.
- tps curve 8K→32K: 71.7 → 67.1 → 66.7 — the long-ctx collapse is
  gone (auto drooped 75.8 → 51.8).

## Posture

- **Banked: `ARC_MXKV_NSPLIT=128`** (env pin on the r217 wheel;
  validating legs r217 + r217b; unset = C++ auto ≈ 40).
- Next (priority order): (a) C++ `mx_get_num_splits` re-priced for
  engine geometry so the pick is automatic (kill the env pin); (b)
  ceiling 192/256 probe (decelerating: −11.45 → −6.45 → −2.6% per
  octave); (c) kv_tile 16→64 walk policy — the r215 floor+union pools
  (~13 ms remaining at 32K after splits exhausted their headroom).

**Reverted**: nothing — env-instrument leg; pin inert when unset.
