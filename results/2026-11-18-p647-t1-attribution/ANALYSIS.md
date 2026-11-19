# ANALYSIS — P6.47 Phase 6 T1 attribution probe (r191, cert25zb)

**LEG**: 2026-11-18-p647-t1-attribution (rig bank /root/build/p647/t1A26)
**IMAGE**: vllm-arc:v0.30.0-b70-cert25zb (K2/K3 default-on)
**PROBE**: p172_deepprobe faces A (kernel), B (aux), C (plan) — the
r172 campaign shape, re-run on the shipping K2/K3 image with
`python -u` (unbuffered). The original r191 wrapper run wedged: its
W14 quiesce at 22:09:32Z raced the r188-trap restore fired at
22:07:18Z (the boot script's exec/start hit the stopped container; the
stray W14 machinery contended the XPUs and the probe's first
`torch.xpu.synchronize` busy-waited — 100% CPU, frozen /proc io). The
unbuffered re-run completed rc=0 in ~3 min on the cleared GPUs; both
runs banked (`kbench/` wedged-empty, `kbench_diag/` complete).

## Kernel tables (adjacent-pair caveat: v2/v5 columns are NOT adjacent
## — v4 ran between them; r180 position-noise law applies to the
## closest deltas; e2e A/Bs remain the ground truth)

G1 (the n1 GOAL lane), per-call ms:

| fmt | v2 | v5 | ship | 1K | v2 | v5 | 8K | v2 | v5 | 32K |
|---|---|---|---|---|---|---|---|---|---|---|
| mxfp8 | 1.961 | 2.581 | v5 | | 3.597 | 1.528 | | 10.683 | 4.861 | |
| mxfp4 | 0.698 | 1.594 | v2 | | 2.475 | 2.273 | | 8.071 | 7.714 | |
| mxk8v4 | 0.828 | 1.845 | v5 | | 3.359 | 2.211 | | 11.099 | 7.659 | |

G8 (n8 lane), per-call ms @32K: m8 83.452 (v2) / 35.548 (v5);
m4 61.241 / 58.298; mkq 88.489 / 58.408. v4: catastrophic everywhere
(34.8–279.6 @32K) — refuted stands.

Aux (face B, kernel+aux minus kernel): 0.016–0.194 ms at G1 long ctx,
≤0.155 at G8 — the A1 aux-fold lever is dead for the GOAL lanes.
Plan (face C): 0.574 ms/step (G1) / 0.954 (G8) TOTAL (0.036–0.060
ms/layer amortized) — P1 low priority stands.

## Verdicts

1. **The kernel IS the n1 long-ctx binder.** Every non-kernel term is
   noise (<2% of step). The S1 ctx-proportional delta = the per-call
   kernel time scaling with ctx.
2. **Per-format defaults are already optimal at long ctx** (v5 for
   m8/mkq by 1.45–2.2x, m4 a kernel-level wash with the e2e B1 verdict
   keeping v2 at low G). Short-ctx G1 exceptions exist (all three
   formats: v2 wins @1K by 0.6–1.1 ms) but the crossover is
   ctx-dependent and seq_lens is device-side under capture — not
   host-lawfully dispatchable; absolute magnitudes are small and the
   short-ctx cells sit near e2e parity already.
3. **B1's e2e n1 m4 regression (+14 ms) exceeds the kernel-level
   delta (v5 −0.36 ms @32K)** — a non-kernel v5-path cost at G1
   (plan-shape replay). The G-adaptive rule (v2 below the knee, v5
   above; capture-shape-lawful) is the e2e-banked mitigation:
   designed, held for the v7 validation legs to carry.
4. **THE STRUCTURAL VERDICT**: best MX kernels at G1@32K run
   4.86 (m8) / 7.71 (m4) / 7.66 (mkq) ms per call vs the ctrl FA
   lane's ~0.2 ms/layer — a 20–35x structural gap. r182's ablation
   floor (3.228 ms dot+load+addr+reductions + 1.62 lawful f8->f16 cvt)
   leaves no removable component >1.7 ms INSIDE the current two-launch
   split-walk structure. FP8-level n1 long-ctx ITL requires ~20x less
   attention time — unreachable by tuning; it needs a different kernel
   structure. **Phase 7 (v7 FA-shaped MX decode) is the remaining
   lawful lever on the GOAL**: single kernel (no mid/combine),
   pipelined loads hiding the bt->slot gather (num_stages on the load
   stream), K2/K3-packed dequant feeding tl.dot operands directly,
   FA's tile discipline. Physics envelope: 0.17 ms/layer (bytes).

**Next**: v7 kernel design + offline correctness gate (d<=2e-3 class)
+ A/B probe at the T1 cells; G-adaptive dispatch rides the same legs;
convergence -> cert25zc -> full battery -> release.
