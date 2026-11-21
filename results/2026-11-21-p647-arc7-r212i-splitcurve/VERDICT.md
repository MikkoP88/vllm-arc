# r212i/r212i2/r212i3 — the split-cost curve, the plan path, and what prices the floor

**Date**: 2026-11-21 · **Leg**: r212i + r212i2 + r212i3 (arc7 Phase-8,
offline fast-cycle) · **Stack**: build21 wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, same sha as r212g/r212h) +
`vllm-arc:v0.30.0-b70-cert25zc`, harness `p204_graphtime.py` at
engine geometry (D=256 / HREP=6 / BLOCK=64, mxfp4, graph 32×50) ·
**Runs**: `r212i_launch1.log` 08:08-08:12, `r212i2_launch1.log`
08:16-08:21, `r212i3_launch2.log` 08:32-08:36 UTC (first r212i3
launch died to a script quoting bug, all cells rc=125 in 0 s —
`-e "$@"` binds `-e` to the first word only; fixed with an argv
array, relaunched clean)

## Verdict

**Three findings, one instrument law, one falsified hypothesis, and
the next kernel-iteration target priced.**

### 1. The forced-split instrument is broken in this wheel (r212i → r212i3)

Every forced-split cell (prebuilt plan via
`_mx_build_decode_split_plan`) canaries above 2e-3 at every ctx,
every mk, every plan lens — while `native_auto` / `native_es` are
clean everywhere. The r212i3 probes close the classification:

- `nan_left = 0` at all ctx → the op writes the **full** output
  (NaN-poison survived nothing) — never-written is falsified;
- rowmatch has no clean off-diagonal at any ctx → not a permutation;
- canary ≈ `ref_absmax` at every ctx (0.696 vs 0.676; 0.157 vs
  0.114; 0.0699 vs 0.0645) → the written values are uncorrelated
  with the reference at full magnitude: **wrong values**.

The plan path executes deterministically (costs vary with split
count and plan lens; outputs vary with plan lens) but computes the
wrong attention. The engine never passes a plan (auto only), so
shipping correctness is unaffected — this is an offline-instrument
defect: the python plan-builder and the kernel's plan consumer have
drifted (arc7 kernel iterations changed plan semantics; the builder
was never re-validated because the engine path ignores plans).

**Instrument law**: on the build21 wheel, p204 forced-split cells
mismeasure. Auto cells are the only valid instrument.

### 2. Coverage hypothesis falsified; capacity pricing is FREE (r212i2)

Hypothesis after r212i: plans must cover `max_seqlen_k`; the
cap-priced mk (bt_width × 64 > max real len) left plan-short tails
unassigned. r212i2 A falsifies it: forced plans fail **identically**
at `MAXSK=ctx` (mk = max real len = 21846, plan@real — the exact
r210 conditions that once passed, now at engine geometry D=256).

The mk ladder (auto cells, ctx=32768 real, canary 0 at every point
— the per-row law):

| mk | 21846 | 21888 | 80000 | 262400 |
|---|---|---|---|---|
| auto µs/call | 1040.2 | 1039.2 | 1039.4 | 1039.7 |

**Auto cost is flat across a 12× mk span (0.05% total delta).** The
capacity-priced `max_seqlen_k` posture (the r212g fix) costs zero at
op level versus exact-ctx pricing — the engine's bridge posture is
confirmed free, from the kernel side.

### 3. The "fixed floor" is the auto heuristic's own partitioning (r212i)

> **SUPERSEDED (same day, r212j/r212j2)**: the headroom table below
> is retracted — the forced-cell timings measured a 4× under-walk
> (tile-unit ABI defect, see
> `results/2026-11-21-p647-arc7-r212j-plan-abi/`). The honest curve
> shows auto BEATS every uniform forced split (1.67×/2.55× at
> 8K/32K); the floor is real work, not bad partitioning.

The post-r212g decomposition question was whether the ~600 µs fixed
per-call excess (native 1K ≈ 598 µs vs v2_es 222 µs) prices
split-plan machinery or launch/indirection. Answer: neither — it is
**the auto heuristic partitioning badly**. The (invalid-but-timed)
forced cells bound what a better partition would buy:

| ctx | auto µs | forced s16 µs | auto / s16 |
|---|---|---|---|
| 1024 | 598.4 | 301.6 | 1.98× |
| 8192 | 423.0 | 203.7 | 2.08× |
| 32768 | 1039.2 | 706.6 | 1.47× |

s16 == s64 to 0.03 µs at every ctx (parallelism saturation floor:
~301 µs at 1K, ~204 µs at 8K, ~706 µs at 32K). s1 prices the starved
walk (296.7 / 2290.4 / 9097.2 µs — the r212f pathology reproduced at
plain shape). **Caveat**: s16-class timings measure wrong-value
computation; they bound, not promise — the walk/partial-write/reduce
cost structure is presumably similar for correct math, but adopting
any forced split requires the plan path fixed first.

### Next (r212j, kernel iteration)

1. Audit the plan ABI in the live vxk tree
   (`_mx_build_decode_split_plan` vs the C++ plan consumer) — repair
   until forced cells canary 0; that rehabilitates split-tuning as
   an instrument.
2. Then tune the partition: auto leaves 1.5-2.1× on the table versus
   the s16-class floor at every ctx; wiring a tuned plan (or an
   improved in-kernel heuristic) into the bridge is worth up to
   ~300 µs/call at short ctx and ~330 µs at 32K → ≈ 16 layers ×
   0.3 ms ≈ 5 ms/step ≈ 7-9% ITL at the measured faces.
3. The E_orc oracle cell (default envs, post-edit harness) reproduced
   r212i_8192 (canaries identical, timings within 15%) — the
   PLANLEN/POISON/ROWPROBE env-gated edits are behavior-preserving.

## Gate note

`r212i3_launch2.log` gate prints spurious `AUTO_CANARY_FAIL` lines —
a falsy-or gate bug (`(canary or 1) > 2e-3` with canary 0.0), fixed
in the shipped script after the run; the banked JSONs show auto
canary 0.0 at every cell. r212i's gate FAIL is the real instrument
finding (Section 1), not a runner defect.

## Artifacts

`offline/p212i_{1024,8192,32768}.json` (r212i: forced canaries +
auto curve), `offline/p212i2_{A,B,C,D}_*.json` (mk ladder + coverage
cells), `offline/p212i3_{P1,P2,P3,E_orc}.json` (poison/rowprobe +
oracle), `r212i{,2,3}_launch*.log`, `r212i3_P{1,3}_*.log` (cell
stdout with rowmatch prints). Rig: `/root/build/p199/out/p212i*`,
logs `/root/build/p199/logs/`. Tools:
`r212i_splitcurve.sh`, `r212i2_mkcurve.sh`, `r212i3_poison.sh`;
harness gained env-gated `P204_PLANLEN` / `P204_POISON` /
`P204_ROWPROBE` knobs (default behavior oracle-verified unchanged).
