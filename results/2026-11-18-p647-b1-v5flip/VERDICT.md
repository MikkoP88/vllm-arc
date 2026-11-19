# VERDICT — P6.47 Phase 6 B1: mxfp4 verify-decoder v5 flip A/B (r188)

**LEG**: 2026-11-18-p647-b1-v5flip (rig bank /root/build/p647/flipA25)
**IMAGE**: vllm-arc:v0.30.0-b70-cert25zb (K2/K3 default-on)
**CELL**: m4qv5 = the r185b m4q chain cell-for-cell plus
`ARC_MXKV_VERIFY_DECODER=v5` (the only delta)
**GATES**: ACCEPT_KNEE_PASS required; acceptance must stay r185-m4q
level; ITL compared against the banked r185b m4q (v2) ladder.

## Result table (acceptance% / mean-len / ITL p50 / eff / ss, ms)

n1:

| ctx | m4q v2 (r185b) acc/len/itl/eff/ss | m4qv5 acc/len/itl/eff/ss | Δ ITL |
|---|---|---|---|
| 4096 | 94.7 / 3.84 / 49 / 12.8 / 49 | 94.0 / 3.82 / 58 / 15.2 / 57 | **+9** |
| 8192 | 83.6 / 3.51 / 58 / 16.6 / 58 | 88.1 / 3.64 / 66 / 18.0 / 65 | **+8** |
| 16384 | 92.8 / 3.78 / 76 / 20.1 / 76 | 92.8 / 3.78 / 85 / 22.5 / 83 | **+9** |
| 32768 | 91.5 / 3.75 / 112 / 29.9 / 111 | 91.5 / 3.75 / 126 / 33.6 / 122 | **+14** |

n8 (aggregate ITL p50 / steady-state ss):

| ctx | m4q v2 ITL / ss | m4qv5 ITL / ss | Δ ss |
|---|---|---|---|
| 4096 | 131 / 228 | 140 / 240 | +12 |
| 8192 | 201 / 457 | 220 / 536 | +79 |
| 16384 | 341 / 895 | 361 / **786** | **−109** |
| 32768 | 1551 / 1978 | 696 / **1703** | **−275 (−14%)** |

Quality: acceptance healthy at every cell (73.0–94.0%, KNEE=none);
ERRAUDIT ERROR=0 / TRACEBACK=0; LMCache auto-off posture confirmed
(boot NOTE + zero LMC lines in the serve log).

## VERDICT: blanket default flip REFUTED — G-adaptive lane opened

- **n1 (the GOAL deficit lane): v5 loses at EVERY ctx (+8..14 ms).**
  The kernel-level win (r179: v5+K3 7.618 vs v2 batch ~8.2–9.2 ms)
  does not survive to e2e at G=1. Mechanism (grid-shape attribution):
  the v2 batch launch parallelizes over (R=4 rows × Hq=24 q-heads ×
  splits) — ~24k programs at n1 — while v5 groups rows+heads into the
  MMA M dimension and parallelizes over (G=1 × HK=4 kv-heads × splits)
  — ~1k programs; at G=1 v2's program count wins, and v5's extra
  plan-build ops (bt_groups index_select) do not amortize.
- **n8 long-ctx: v5 wins decisively** — 32K ss 1978→1703 (−14%) and
  the aggregate ITL p50 halves (1551→696; the ramp-interleaved column
  also carries the shorter prefill stalls). 16K ss also improves
  (895→786). Below 16K v5 is neutral-to-worse (+12..79 ms ss).
- **Disposition**: `_V5_DEFAULT` stays ("mxfp8", "mxk8v4"); mxfp4
  remains v2. The evidence opens the **G-adaptive dispatch** lane:
  choose the verify decoder by split-batch request count at plan build
  (static per XPU-graph capture — shape-lawful), v2 below the knee,
  v5 above it. Knee placement from this bank: between 8 and 16
  observed concurrency steady-state cells; to be pinned by the T1
  probe's G-sweep faces (r191, in flight) before implementation.
- The engine gained a one-time verify-decoder dispatch log line
  (fmt+decoder+source per process) so every future leg's serve log
  carries the engagement marker this A/B lacked.

**OUTCOME: A/B banked, flip REFUTED, quality gates PASS, follow-up
lanes registered (G-adaptive dispatch; T1 attribution).**
