# VERDICT — P6.47 r238: cert28 per-lane n8 acceptance mined from the r237 bank — channel closed, K-precision law banked

**What**: the last open evidence channel from the r237 disposition
("a fresh bottleneck surface in the acceptance counters") closed by
mining the already-banked `metrics_before.txt` snapshots in
`results/2026-11-24-p647-r237-n8ss-cert28/bank/` — zero rig time
(r236 bank-mining precedent). Each boot banked a metrics snapshot
before each cell; the 16K-cell traffic delta (c32768_before −
c16384_before, warmup baselines 8–9 drafts) gives per-lane spec
acceptance under **n8 / 16K / MT=128 traffic** — a face the cert28
battery mining (agg8, ctok 2048 mix) did not cover.

## Table (~300 drafts per lane; acc/step = accepted_tokens/drafts, k=3 → max 3)

| lane | K / V | acc/step | acc/step ÷ ctrl | pos0 | pos1 | pos2 |
|---|---|---|---|---|---|---|
| ctrl | fp8 / fp8 | 2.306 | 1.000 | 0.810 | 0.765 | 0.732 |
| m4q | e2m1 / e2m1 | 2.170 | 0.941 | 0.833 | 0.701 | 0.636 |
| mkq | e4m3 / e2m1 | 2.399 | 1.040 | 0.841 | 0.791 | 0.767 |
| m8q | e4m3 / e4m3 | 2.430 | 1.053 | 0.899 | 0.846 | 0.685 |

## Reading

- **No fresh surface / no knee**: every lane healthy — pos0 ≥ 0.81,
  acc/step 2.17–2.43, no long-ctx acceptance degradation. S1 stays
  cured on cert28 under this traffic shape too.
- **The factorial is K-precision-dominated** (single-cell deltas,
  SE ≈ 0.04–0.05 per lane at ~300 drafts):
  - K effect at V=e2m1: m4q (e2m1 K) 2.170 vs mkq (e4m3 K) 2.399 —
    **e2m1 K costs ~0.23 acc/step (−10%, ~5 SE)**, concentrated at
    the deepest positions (pos2 0.636 vs 0.767).
  - V effect at K=e4m3: mkq (e2m1 V) 2.399 vs m8q (e4m3 V) 2.430 —
    **e2m1 V costs ~0.03 (sub-noise)**.
  - ctrl fp8 sits at 2.306; both e4m3-K lanes land above it
    (numerics-drift direction, within ~2 SE).
  - Mechanism: the draft (MTP, fp8) is lane-invariant; only the
    target verify rides the KV lane — deepest-position verification
    chains are where e2m1 **K** numerics diverge from the draft's
    expectations. e2m1 **V** numerics do not disturb acceptance.
- **Throughput consistency check**: acceptance does not dominate
  tps — m8q has the best acceptance (2.430) yet the worst n1 ITL
  (highest dequant cost): solo tps battery m4q 51.2 / m8q 47.7 /
  mkq 46.7. The m4q lane's aggregate position carries a ~6%
  acceptance tax on top of its ITL gap at n8; the mkq split
  (e4m3 K + e2m1 V) avoids it while keeping half the KV savings.
- **Caveats**: single 16K-cell deltas per lane (the final 32K cell
  of each boot has no metrics_after snapshot — one-snapshot design);
  ~300 drafts → lane-vs-ctrl gaps at ~2 SE except the K effect
  (~5 SE, the only claim promoted above noise).

## Verdict

**PASS** — channel closed, no fresh bottleneck. New record: first
per-lane n8/16K acceptance rows on cert28 + the **acceptance
K-precision law** (e2m1 K costs ~10% acc/step at deep positions;
e2m1 V is acceptance-free). This prices the last evidence channel
named in the r237 idle disposition — the MX hunt's idle stands on
all three channels now closed (engine-shape: no new lineage;
workload-shape: all face classes measured; counters: this leg).

## Hygiene

- Analysis-only: no new bank (sources = the already-scrubbed r237
  bank; this VERDICT cites them). No rig time; W14 untouched
  (health-200 verified at the r237 close-out and unchanged since —
  no arc containers ran in this window).
