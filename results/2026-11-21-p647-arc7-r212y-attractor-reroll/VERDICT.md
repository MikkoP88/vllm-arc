# r212y — attractor re-roll: HELD (4th wheel); the +34% basin is stable, k_lane band 595–606 becomes the cross-wheel currency

**Date**: 2026-11-21 · **Leg**: r212y (arc7 Phase-8 instrument) · **Stack**:
r212y wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`b64bd6602a92ca866a22944533fabdab`; r212x tree + `p212y_patch.py` —
ZERO semantic change: the K/V sever-gate comparison chains hoisted to
precomputed bools (`kp_sever`/`vp_sever`) before the K loop, both
gates read the bools; TU preflight clean, `r212y_build1.log`) +
`vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212y_leg1.log` 22:33–22:41
UTC (three same-wheel cells: m0 / m11 / m17 × 32768)

## Verdict

**THE ATTRACTOR HELD: m8 = 1769.2 (+34.0%) on a semantically-null
wheel — the fourth consecutive +34 (r212v +33.8, r212w +34.1, r212x
+34.5, r212y +34.0).** The bool-hoist re-rolls the GEMM-1 region's
codegen dice and lands in the same basin:

| cell | µs | r212x | r212u clean |
|---|---|---|---|
| m0 (K->tSrK + V real) | 1006.6 | 995.9 | 718.0 |
| m11 (K const + V light) | 414.0 | 417.6 | ~380 |
| m17 (K->tSrK + V light) | 1020.3 | 1017.6 | — |
| k_lane (m17−m11) | +606.3 | +600.1 | pool 338 |
| m8 oracle (mxfp8) | 1769.2 (+34.0%) | +34.5% | +0.2% |

Two working conclusions: (1) the attractor is a stable basin (m8
spread ±0.5% across four differently-edited wheels; k_lane spread
595.5/600.1/606.3 ≈ ±1%) — cross-wheel k_lane comparison UNDER the
attractor is defensible, and same-wheel deltas remain primary; (2)
the e4m3 lane carries a real ~+450 µs regression vs the r212q-era
region on every probe wheel — relevant to the mxfp8 endgame, queued
behind the K-lane work.

### The fork

**r212z: double-buffered K decode (depth-1 software pipeline).** The
K lane's ~600 µs is the tSrK-liveness chain (r212x) — per (K,D) step,
loads → staging → decode → operand → DPAS serialize (~73 ns/step ×
8192 steps at 32K). Decode (K,D+1) now issues inside iteration D,
before gemm(D), so the chain overlaps the DPAS pipe; two
B-fragment buffers alternate by D parity (DD=4 at d=256). Probe cells
and the e4m3 lane stay sequential. Register pressure is the risk
(two 32-reg K buffers + accumulators — the r212t scratch-wall
precedent); m17 vs the 595–606 band decides. The principled
attractor escape (template-split probes out of the shipping
instantiation) is queued for when cross-wheel absolutes turn
load-bearing (e2e certification faces).

## Artifacts

`offline/p212y_{m0,m11,m17}_32768.json`, `r212y_build1.log`,
`r212y_leg1.log`. Rig: `/root/build/p199/out/p212y_*`,
`/root/build/r212y_{build1,leg1}.log`. Tools: `p212y_patch.py`,
`r212y_reroll.sh`. (Note: the r212x wheel ee1724bb was not snapshotted
before the r212y build overwrote it — reconstructible by reverting
p212y and rebuilding; no evidence lost, all r212x artifacts banked.)
