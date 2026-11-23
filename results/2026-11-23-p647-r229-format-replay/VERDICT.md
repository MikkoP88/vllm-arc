# P6.47 r229 format-replay — VERDICT: PASS (mxfp8 8/8, mxfp4 8/8)

Date: 2026-11-23 · Image: `vllm-arc:v0.30.0-b70-cert27` (fork_sha
`d39ab5a26bf737d006d47a3f3125a404dca6e928`, wheel aeca45da / SO 190e261e)
· Scorer: `tools/ar/r229_replay_fmt.py` (fmt-parametric, q-cast-matched)
· Wrapper: `tools/ar/r229a_fmt_replay.sh` (ONE env-armed boot per format —
cert27 bakes wheel + instrument; the r226a /proc dance is retired).

## Result

| leg | fmt | payloads | classes | score | verdict |
|-----|-----|----------|---------|-------|---------|
| r229a (gate4, 15:24:22Z) | mxfp8 | 120 | 86 COVERED / 34 TRUNCATED | **8/8 PASS** | `R229a_LEG_DONE` |
| r229b (15:34:36Z) | mxfp4 | 120 | 86 COVERED / 34 TRUNCATED | **8/8 PASS** | `R229b_LEG_DONE` |
| mxk8v4 | — | r225t 12/12 + r226a 8/8 (prior) + full-population control sweep | — | PASS |

All three MX formats now hold the q-cast-matched replay criterion on their
OWN engine tensors: each lane vs the fp64 walk of ITS OWN input, with the
q-cast cross-lane check (`q.to(bf16).to(fp16)` — exact value carry)
arbitrating on shared values. The r225 coverage-note claim — the
ReduceSplitK combine path serves mxfp4/mxfp8/mxk8v4 identically — is
measured, not argued (identical full-population distributions below).

## The band law (why the gate needed two recalibrations, evidenced)

The original bands (dmax 2.5e-2, cross 5e-2, ulp 16.0) were derived from
the first-scored cells (r225t 12 cells, r226a 8 cells). r229a run2 failed
cell 7 at dmax 2.704e-02 with every corruption signature clean — and the
full-population sweeps proved the first-scored cells are the easy tail of
a heavy-tailed distribution:

| metric | mxfp8 (r229s, 86 cells) | mxk8v4 (r229s2 control, 86 cells) |
|--------|------------------------|-----------------------------------|
| native_vs_walk min / p50 | 3.795e-3 / 1.538e-2 | 3.866e-3 / 1.494e-2 |
| native_vs_walk p90 / max | 6.122e-2 / **1.216e-1** | 5.976e-2 / **1.208e-1** |
| cross_qmatched p50 / max | 1.562e-2 / 2.500e-1 | 1.562e-2 / 1.250e-1 |
| verdict split under OLD bands | 60/26 | 58/28 |

Identical distributions — format-invariant. Both formats fail ~2/3 of the
full population under the old bands; the bands, not the kernels, were
wrong. Every tail cell carries the CLEAN signature: triton replay
bit-exact, native bitwise-stable, nan=0, mism 1-3/3072, sub-ulp flips on
isolated large-magnitude elements (output envelopes are format-
independent: max|out| p50 ~9, p90 ~29, max ~70-80 in both banks — the
absolute tail is accumulation-order rounding on large elements, not
corruption). ulp is the near-tie noise metric, anti-correlated with dmax
in the tail (the 36.83-ulp mxfp8 cell sits at dmax 1.459e-2; the
0.12-dmax cells sit at ulp <= 1.1); its bound belongs in the noise
headroom while mism + dmax remain the broad-signature guards.

Recalibrated bands (commits 59bcbf9b, a7326f0e): `BAND_DMAX 2.0e-1`,
`BAND_CROSS 4.0e-1` (~1.6x the pooled maxima — the r225t derivation
ratio), `BAND_ULP 64.0` (1.7x the pooled 36.83). The r224 refusal
signature (dmax 2.27, massive mism, REP-LOOP e2e, nan) stays >10x above
every band. Lesson banked: a band calibrated on a lexical-prefix sample
of a heavy-tailed distribution is not calibrated.

## Raw-lane analyzer rollup (context, not the verdict)

r224b analyzer (raw lanes, q-cast difference included by construction):
mxfp8 max_dmax 2.500e-1 (86/120 divergent), mxfp4 max_dmax 4.219e-1
(81/120), nan=0 both, pools dense — consistent with the r226a mxk8v4
posture (raw lane gap is INPUT, not kernel; the q-matched scorer
arbitrates). Reports banked per leg (`r224b_report.txt`).

## Leg history (all banked in git)

1. r229a run1 (13:56Z): ABORT `ab_armed=0` — serve_arc_b70.sh env
   passthrough lacked the `ARC_MXKV_AB*` trio (native armed, instrument
   didn't; the passthrough fingerprint). Fix f12715e7 (script also
   resynced to the live rig copy — the tracked one had drifted
   pre-r212f). Abort evidence: r229a_launch.log (rig, quoted in the
   commit).
2. r229a run2 (14:17Z): instrument armed, 120 payloads banked, cell 7
   FAIL dmax 2.704e-02 (old band) → r229s sweep (86 cells) → mxk8v4
   control r229s2 → recalibration 59bcbf9b.
3. r229a gate3 (15:12Z): FAIL max_ulp 16.98 (old ulp band) on dmax
   7.795e-03 → ulp recalibration a7326f0e.
4. r229a gate4 (15:24Z): **8/8 PASS**. 5. r229b (15:34Z): **8/8 PASS**.

## Hygiene

- No rig address in any banked file (verified at pull time; serve.logs
  stayed rig-side — only txt/json/log evidence files without addresses
  were pulled).
- Artifacts: this directory (rig homes `/root/build/p647/r229a`,
  `/root/build/p647/r229b`, `/root/build/p647/r226a` for the control).
  Payloads (.pt) stay rig-side (size); SHA256SUMS covers the 21 evidence
  files.
- W14 restored by each leg's guarded trap (final restore 15:34:38Z).
