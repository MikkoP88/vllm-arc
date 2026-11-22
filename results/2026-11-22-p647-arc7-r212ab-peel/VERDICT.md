# r212ab — static-parity peeled K pipeline: byte-correct but WASH-LOSS (+28); the GEMM-1 overlap thesis closes at whole-fragment granularity; reverted to r212aa

**Date**: 2026-11-22 · **Leg**: r212ab (arc7 Phase-8 instrument) ·
**Stack**: r212ab wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`6c80b54f24f6529ae1f950e09d2174d0`; r212aa tree + `p212ab_patch.py` —
the shipping e2m1 GEMM-1 D-loop peeled by 2 with STATIC buffer names
per textual half (even: cur tSrK, lookahead tSrK1; odd: cur tSrK1,
lookahead tSrK), prologue decodes (K,0), runtime tail guard (TU taught:
`size<4>(tKDgKD)` is RUNTIME — the gmem K tensor carries int extents,
so a constexpr tail is ill-formed; first TU failed 8× on exactly that,
fixed and re-flown, `r212ab_tu2.log` clean) + `vllm-arc:v0.30.0-b70-cert25zc`
· **Run**: `r212ab_leg1.log` 01:14–01:20 UTC (three same-wheel cells)

## Verdict

**Byte-correct, net-slower.** m0 = 999.7 (+28.1 vs r212aa's sequential
971.6), canary 0.0 — the peel's parity mapping is right and the
overlap structure is exactly r212z's mechanism minus the operand
demotion. The instrumented lane and the e4m3 lane are untouched:
m11 416.0 (r212aa 414.2), k_lane +606.8 (band 595–609), m8 1196.7
(+0.3% vs the r212aa baseline — the wrapper did not re-roll e4m3).

| cell | r212ab | r212aa | note |
|---|---|---|---|
| m0 (SHIPPING, pipelined e2m1) | 999.7 | 971.6 | +28.1 wash-loss |
| m11 (K const + V light, instr) | 416.0 | 414.2 | stable |
| m17 (K real + V light, instr) | 1022.8 | 1023.5 | stable |
| k_lane (m17−m11) | +606.8 | +609.3 | instrumented unchanged |
| m8 m0 oracle (mxfp8) | 1196.7 (+0.3%) | 1193.7 | undisturbed |

**Reading**: the second live 32-reg B fragment costs about what the
one-step decode overlap saves — with the sequential form already
scheduling decode and DPAS tightly, the peel adds pressure (likely
spill-class; within fresh-basin dice noise either way) without net
cover. Combined with r212z (runtime-parity variant: mechanism proven,
region collapsed), the GEMM-1 overlap thesis is CLOSED at whole-
fragment granularity: two structurally different implementations, both
net-negative. Recovering the K lane must now come from reducing the
decode ALU itself (SWAR 2-nibbles-per-op class, ~20-25% of the ~73
ns/step) or accepting 971.6 as the e2m1 shipping state and moving to
the e2e face.

**Reverted in full**: tree via `p212ab_revert.py` (mainloop back to
`5ce38305` = exact r212aa state — the loop bytes were preserved
verbatim inside the wrapper, so the revert is a drop), wheel restored
from `.r212aabak` (`9a01790b`). The r212ab wheel (`6c80b54f`) is
snapshotted as `.r212abbak`.

## The fork

**The e2e face is next.** The probe-harness dev_us is a proxy; the
standing goal is engine-level MXFP4/MXFP8 decode at FP8 level. The
converged shipping state is r212aa (m0 971.6, m8 1193.7 = new best,
canary-clean both lanes) — per the improvement-convergence law,
converge then certify: wheel into the engine image, full battery on
THAT, compare MXFP4/MXFP8 decode throughput/TTFT vs the FP8 baseline.
A further probe-harness iteration (SWAR decode) is queued behind the
e2e truth — if e2e already sits at FP8 parity on mxfp8's new-best
lane, the remaining e2m1 work re-prioritizes against real numbers.
