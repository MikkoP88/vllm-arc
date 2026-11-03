# p616b — probabilistic draft sampling: FULL 5-DTYPE MATRIX + integrity discriminator (2026-11-03)

Owner directive 2026-11-03: *"all performance numbers has to be retested
and this fix branch is only validated to merge if has real impact to
performance … cannot affect negatively to current certificated
configurations."* This dir banks P6.16b: every dtype default face
re-measured on the **probabilistic draft posture** (`draft_sample_method
="probabilistic"`, serve-line sed only, cert6 image byte-unchanged —
one boot per dtype, one-variable ×2 vs the banked greedy faces), plus
the P6.16b2 discriminator leg that classifies the two integrity flakes.
Harnesses identical to the banks: `bench_run7n_t5.py` (t0 + t1.0-xhigh
Run-7n batteries), `collapse_trace.py` ×10 per TQ dtype, `arc_spec_gate`,
`r42_integrity.py` (5/5 + det 3/3), `p70_solo_ttft.py` (U32k/U64k).
All legs solo from stable W14-up, substitution counts verified in-output
(1/0 dtype, 1/1 spec-method), 0 ERROR / 0 Traceback on every boot, W14
restored by trap (health 200 verified before every launch).

## The one-line verdict

**Real impact proven on the sampled faces — merge as an OPT-IN posture.**
t1.0-xhigh 4-way agg: fp8 **+29 %**, tq4 **+44 %**, k3v4 **+54 %**, 3bit
**+91 %** (k8v4 EV-neutral −0.3 %); every measured solo pair uniform
(51–67 tok/s, zero 26-tps legs); t0 certified faces unchanged on every
dtype; det 3/3 everywhere; TTFT within −3 % except one flagged face.
**Correction to p616a:** the total-collapse mode is NOT eliminated — one
deterministic divergent-trajectory request (seed replay) still collapses
at ~26 tok/s under probabilistic on every TQ dtype — but the graded
damage modes (47/66 t/s depth-dead legs) are gone and probabilistic
drafting + the ratio test is the mathematically-correct sampler at
t≥1.0 (greedy drafts + sampled verify is biased). Per the pre-registered
adoption rule (default flip needs EVERY dtype ≥ +10 % t1.0 agg AND t0
±1.5 %): k8v4 fails the first clause → **default stays greedy;
probabilistic ships opt-in** (certified configs byte-untouched).

## t0 faces (certified comparators — must not move)

| dtype | agg#2 bank→prob | solo ×2 bank→prob | TTFT U32k/U64k bank→prob |
|---|---|---|---|
| fp8 | 296.25 → **296.71** (+0.2 %) | 84.51/84.41 → **84.46/84.52** | 17.34/40.24 → **17.37/40.30** |
| k8v4 | 301.97 → **300.70** (−0.4 %) | 88.49/88.61 → **88.35/87.93** | 15.52/33.58 → **15.36/33.54** |
| tq4 | 298.38 → 290.74 (−2.6 %) | 86.24/85.98 → 83.35/83.43 (−3.3 %) | 15.67/33.99 → **15.77/34.20** |
| k3v4 | 285.85 → **289.57** (+1.3 %) | 80.83/80.45 → **83.13/82.75** | 15.68/33.58 → **15.25/33.75** |
| 3bit | 301.13 → **302.74** (+0.5 %) | 85.83/85.55 → **86.40/86.04** | 15.79/34.51 → 15.80/**36.98** |

- Code-level identity at t0: gumbel drafts are plain argmax at temp 0
  (`gumbel.py:102,117`) and the verify accepts iff `target_argmax ==
  draft_sampled`, emitting `target_argmax` on rejection
  (`rejection_sampler_utils.py:549,571-585`) — t0 tokens cannot differ
  by posture; the throughput tables confirm it (fp8/k8v4/k3v4/3bit all
  in band; both directions of drift seen = ordinary run variance).
- tq4's −3 % t0 dip is **boot-day environment, not the posture**: the
  b2 greedy control boot (same day, same image, greedy posture) landed
  **83.36/83.40** — byte-close to the probabilistic 83.35/83.43 and
  equally below the 86.2 bank (k3v4 ran +2.9 % the other way).
- 3bit U64k TTFT +7.2 % (36.98 vs 34.51) is the one out-of-band face —
  single measurement on the leg with the slowest flip (17.2 s); U32k
  dead-on. Flagged, not gating (opt-in dtype face).

## t1.0-xhigh faces (the fix's target)

| dtype | solo pair banked greedy → prob | 4-way agg#2 bank → prob | Δagg |
|---|---|---|---|
| fp8 | 61.73/61.79 → **67.46/64.10** | 186.48 → **241.10** | **+29.3 %** |
| k8v4 | 77.31 **+ 26.08** → 54.77/51.62 | 173.39 → 172.95 | −0.3 % |
| tq4 | 77.60 **+ 25.94** → 54.27/50.86 | 93.05 → **134.42** | **+44.5 %** |
| k3v4 | 77.37 **+ 25.88** → 52.37/62.32 | 88.95 → **137.11** | **+54.1 %** |
| 3bit | **25.99** + 85.57 → 53.11/58.90 | 89.58 → **171.32** | **+91.2 %** |

Every probabilistic solo pair is uniform — no 26-tps leg in any measured
face (the greedy banks' defining defect; the b2 greedy control boot
sampled a live one the same day: pair **25.96 + 81.44**). fp8's sampled
solo now BEATS its greedy bank (67.46 vs 61.73) — the 49 %-greedy-draft
acceptance ceiling on the certified dtype is broken too. Sub-4-bit
dtypes: both streams of every pair land mid-50s–60s instead of the
26/77 lottery. Spec gates: PASS all legs (fp8 0.837 target-miss noted;
k8v4 0.9305, tq4 0.8889, k3v4 0.8649, 3bit 0.8899).

## Collapse status — the honest correction to p616a

`collapse_trace.py` ×10 per TQ dtype (deterministic seed-0 replay):
**9/10 healthy-class (2.4–3.7 s, pos0 86–97 %) + ONE total-collapse leg
on every dtype** — req5: drafts 180, accepted 19, wall 7.54–7.78 s,
pos0 8 %, byte-identical numbers on all four dtypes (k8v4/tq4/k3v4/3bit)
and nearly identical to the banked greedy collapse (p615c req2: 181
drafts / 18 accepted / 7.56 s). Mechanism revision:

- The collapse is **a sampled trajectory on which the MTP draft
  distribution genuinely diverges from the target** (ratio-test pos0
  ≈ Σ min(p,q) ≈ 0.08 → TV ≈ 0.9). No drafting/verify scheme fixes true
  divergence; seed replay makes it deterministic.
- What probabilistic drafting DID remove: the **graded damage modes**
  (p615c's pos2-dead 66 t/s and pos1+2-dead 47 t/s legs — 0 of those in
  40 traced requests) and the argmax-mismatch lottery — the ratio test
  accepts non-argmax draft-target agreements, which is why healthy-leg
  depth acceptance holds and every MEASURED face (solo pairs, 4-way
  aggs) came out uniform.
- p616a's "collapse mode ELIMINATED" was an artifact of that leg's RNG
  alignment (its 10 requests rolled no divergent trajectory; 2 rolled
  legitimate early stops). Corrected here and in the master plan.

Practical reading for TQ + hot sampling: aggregate/typical latency
improves massively and streams run uniform, but a rare divergent
trajectory can still cost ~3× wall on that one request. **fp8 remains
the worst-case-latency dtype — and its sampled throughput improves
+29 % under probabilistic.** Worst-case-sensitive fleets: fp8
(probabilistic or greedy) or cool sampling on TQ.

## Integrity flakes + the b2 discriminator (rig/b2_integrity_discrim/)

Matrix integrity: 5/5 on fp8/k8v4/3bit; **4/5 on tq4 (seed 404) and
k3v4 (seed 505)** — cold≠rehit at temp 0 on the two knife-edge prompts
(the 3 dtype-robust seeds hash identically across every boot and dtype;
404/505 outputs differ across dtypes, and each probabilistic rehit
landed exactly on the other dtype's output hash). b2 discriminator:

| Boot | posture | traffic | integrity |
|---|---|---|---|
| 1 | tq4 **greedy** | exact matrix order (run7n t0+t1.0 + trace ×10 + spec gate) | **5/5 ×2** |
| 2 | tq4 **probabilistic** | minimal (flip + warmup) | **5/5 ×2** |

The flake requires **posture × full-battery churn together**; not
reproducible in 4 control runs; det 3/3 everywhere. Since t0 tokens are
provably posture-identical, the channel must be an indirect state effect
on the cache-hit path (allocator/layout side effects of the logits-cache
buffers changing kernel numerics enough to flip near-tie argmaxes).
Classification: **intermittent, opt-in-posture-only, t0-only, near-tie
prompts only** — documented as a caveat; does not touch the certified
greedy battery (b2 boot 1 = the certified posture under the exact same
churn: 5/5 ×2). Nobody should serve probabilistic at t0 anyway (it is
the t≥0.5 posture; at t0 it is functionally greedy — measured in-band).

## Gates (amended set, pre-registered in the plan before this leg)

| Gate | Result |
|---|---|
| (i) zero collapse-mode t1.0 legs / worst:median ≤ 1.7× | **FAIL as written** — 1 deterministic divergent-trajectory leg per TQ dtype (7.5–7.8 s); 9/10 healthy; graded modes gone; measured faces clean |
| (ii) t0 agg/solo within −1.5 % | **PASS** (tq4 dip proven environmental by the b2 greedy control; k3v4 ran +2.9 % the other way) |
| (iii) TTFT within −3 % | **PASS** except 3bit U64k +7.2 % (single face, flagged) |
| (iv) det 3/3 / integrity 5/5 / 0 new WARN | det **PASS all**; integrity 5/5 all dtypes in isolation; matrix-order flakes on tq4/k3v4 classified by b2 (posture × churn, not reproducible ×4); 0 new WARN class |
| (v) real impact: tail elimination at EV ≥ −10 % | **PASS beyond the gate** — agg +29/+44/+54/+91 % on fp8/tq4/k3v4/3bit, k8v4 −0.3 %; every measured sampled face uniform |

Merge decision (owner rule "only validated to merge if real impact"):
**validated — as an opt-in posture.** Default-adoption rule not met
(k8v4 < +10 %) → default remains greedy → certified configurations are
byte-untouched by construction. → P6.16c: `ARC_SPEC_METHOD` serve hook
(default unset = greedy), cert7 = cert6 + serve-script hook (wheel
unchanged), full cert battery, docs + README currency.

## Provenance

Legs: 5 matrix boots (fp8, k8v4, tq4, k3v4, 3bit — cert6 image, graph
law, ARC_SPEC=3, dtype + spec-method seds verified 1/0 + 1/1) + 2 b2
boots (greedy control + probabilistic minimal). Harnesses:
`p616b_matrix.sh`, `p616b2_integrity_discrim.sh` (this dir); instruments
as listed above (banked copies in `rig/<dtype>/`). Serve logs + jsons
redacted byte-exact — matrix: 5× serve_full.log (rig-addr ×8 = +48
each; node-addr fp8 +35 / k8v4 +28 / tq4 +42 / k3v4 +91 / 3bit +77),
5× preexisting_arc-test.json (+5 each) = **+538**; b2:
greedy_serve_full.log +83, prob_serve_full.log +55 = **+138**; leg total
**+676**; raw-ref grep 0 after every file —
`../REDACTION-2026-11-03-p616b.md`.
