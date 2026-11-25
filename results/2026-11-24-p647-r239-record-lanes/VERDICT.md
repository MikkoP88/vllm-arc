# VERDICT — P6.47 r239/r239b: era-stale record lanes re-discriminated on cert28 — every face beats its era anchor

**What**: the deep-sweep directive (2026-11-24: "re-run all analyses,
validations and performance tests on the latest certified image")
includes the record faces whose banked numbers predate cert28. r239 ran
the 4-cell record-lane matrix on cert28 (`vllm-arc:v0.30.0-b70-cert28`,
fork_sha 43f84cf9, .so pins asserted at refuse time); two cells
boot-failed for leg-local reasons and r239b re-ran them with the fixes.
Merged bank: `r239/` (gemma L1 + rhmxfp4 + qurk + aur) + `r239b/`
(gemma at ARC_MAXLEN=118784 + rhmxfp4 with the container-name fix).
Both legs' W14 EXIT traps restored radiance-test health 200 (r239
10:31:54, r239b 14:46:19).

## Cells

| cell | leg | outcome | evidence |
|---|---|---|---|
| gemma L1 record (maxlen 131072) | r239 | BOOT_FAIL — engine KV refusal | `r239/gemma/serve.log` |
| rhmxfp4 checkpoint lane | r239 | BOOT_FAIL — container-name cascade (script bug) | `r239/rhmxfp4/{boot.txt,serve.log}` |
| qurk (qwen ckpt-quark mxfp4xfp8, k3) | r239 | FACES_OK — spec 1.0, erraudit 0/0 | `r239/qurk/` |
| aur (qwen ckpt-quark mxfp8, k3) | r239 | FACES_OK — spec 0.9804, erraudit 0/0 | `r239/aur/` |
| gemma L1 record (maxlen 118784) | r239b | FACES_OK — full record + battery | `r239b/gemma/` |
| rhmxfp4 checkpoint lane | r239b | FACES_OK — spec 1.0, erraudit 0/0 | `r239b/rhmxfp4/` |

The r239 boot failures and their fixes:

- **gemma L1 at 131072**: the engine now refuses — `5.83 GiB KV cache
  is needed, which is larger than the available KV cache memory
  (5.64 GiB) ... estimated maximum model length is 120704`. The gemma
  bf16-weights lane no longer fits 131072 on cert28 (engine growth
  since cert19's 131K boots). Fix: `ARC_MAXLEN=118784` — the face class
  is preserved exactly (face120k needs 110028+16 = 110044 ≤ 118784 ≤
  120704 with margin).
- **rhmxfp4**: my boot-fail path did not `docker rm -f $CNAME`, so the
  dead gemma container held the name and the next boot died on
  `Conflict. The container name "/arc-r239" is already in use`. Fix
  (r239b): every boot-fail branch removes the container before
  returning. The rhmxfp4 posture itself was never at fault.

## The re-discriminated record (cert28 values vs era anchors)

Checkpoint-lane timeline faces (TTFT, seconds; anchors = the
cert11c-era r104/r106b phase-B records; one-sided band ×1.20 — the
gate fails only on regression):

| face | qurk cert28 | qurk anchor | ratio | aur cert28 | aur anchor | ratio | rhmxfp4 cert28 | rhmxfp4 anchor | ratio |
|---|---|---|---|---|---|---|---|---|---|
| U32k | 17.06 | 32.28 | 0.529 | 16.95 | 26.67 | 0.636 | 17.51 | 29.01 | 0.604 |
| C32k | 1.10 | 3.84 | 0.286 | 1.19 | 1.52 | 0.783 | 1.09 | 1.68 | 0.649 |
| U8k | 4.05 | 7.73 | 0.524 | 3.73 | 6.36 | 0.586 | 4.04 | 7.37 | 0.548 |
| U64k | 38.31 | 70.61 | 0.543 | 38.49 | 59.52 | 0.647 | 39.01 | 62.46 | 0.625 |

Every face is 1.5–3.5× FASTER than its cert11c-era anchor (r104-era
boots predate the prefill-lane closure work). Spec gates PASS (qurk
acceptance 1.0, rhmxfp4 1.0, aur 0.9804 — all ≥ 0.85 target); G1 3/3;
erraudit 0 ERROR / 0 Traceback on every cell; bystander p50 ITLs:
qurk 45.6 ms, aur 59.8 ms, rhmxfp4 47.6 ms.

Gemma L1 record (r239b, maxlen 118784; anchors = the cert19-era r111
record):

| face | cert28 | cert19 anchor | ratio |
|---|---|---|---|
| probe98 (98-tok gen probe) | 3.5 s, CLEAN | ~5 s OK | — |
| G1 factual/reason/code | 3/3 | 3/3 | — |
| face8k (8,028 ptok) | 4.2 s | 4.9 s | 0.86 |
| face32k (32,028 ptok) | 16.8 s | 17.2 s | 0.98 |
| face120k (110,028 ptok) | 108.4 s | 110.9 s | 0.98 |
| face32k_2 (prefix re-hit) | 0.9 s | 0.9 s | 1.00 |
| decode solo 1×256 ×3 | 29.0 tok/s ×3 | 28.4–28.5 | 1.02 |
| decode agg8 ×2048 ×3 | 361–474 tok/s aggregate | 194.5–195.5 aggregate | 1.85–2.4× |

**The agg8 "REGRESSION" flags are a units artifact, not a
regression.** The r239b gate compared the battery's per-user
`tps_mean` (45.1–59.2) against the era anchor's aggregate number
(195.0): the ratio is unit-mismatched. In consistent units, cert28's
gemma L1 agg8 aggregate is 8 × 45.1–59.2 = **361–474 tok/s — 1.85–2.4×
the cert19-era record**, measured under a harder protocol (2048
ctok/stream vs the era's 256: longer steady-state decode, less
prefill-mixing). The per-rep spread (48.5/59.2/45.1) is the known
gemma think-spam variance (REP-LOOP baseline law), not instability.
The solo battery is the stable reference: 29.0 tok/s on all three
reps (1.02× the era anchor).

## Gate lines (as banked, with the re-derivation)

- `R239_GATE: FAIL (record lanes re-discriminated on cert28; cross-era
  band x1.2)` — FAIL is solely the two boot-failed cells; every
  measured face passes (all ratios < 1.0).
- `R239B_GATE: FAIL (gemma+rhmxfp4 re-discriminated; cross-era band
  x1.2)` — FAIL is solely the agg8 unit-mismatch flags above; every
  same-units comparison passes.
- Cosmetic: the legs' report greps printed "spec-gate acceptance line
  not found (inspect spec_gate.txt)" — the grep pattern did not match
  the SPEC_GATE JSON line; the acceptance values are in
  `spec_gate.txt` and all three PASS (1.0 / 0.9804 / 1.0).

## Outcome

The record-lane drift class is closed: the era-stale record faces
(checkpoint-lane TTFTs, gemma L1 record + battery) now carry
cert28-measured values, and every one of them holds or beats its era
anchor. The banked record rows in PERFORMANCE-arc.md get these
numbers. Follow-on (this campaign): the r240 report-battery re-run
(solo faces → conc sweep → gemma 128K annex → docs/reports/
regeneration) is already in flight on the same image.

Banks: `results/2026-11-24-p647-r239-record-lanes/` (38 files,
SHA256SUMS, host addresses byte-redacted post-pull — law 8). Legs:
`tools/ar/r239_record_lanes_cert28.sh`, `tools/ar/r239b_gemma_rhmxfp4_fix.sh`,
pull `tools/ar/r239_pull.sh`.

## CORRECTION (2026-11-25) — the agg8 face revalidated: the "361–474
## tok/s aggregate" is retracted as a harness-numerator artifact

Owner revalidation request ("sounds too high") — confirmed. The r239b
gemma battery harness (inline in `r239b_gemma_rhmxfp4_fix.sh`) scored
each stream as `out[i] = mt / w_i` — the REQUESTED 2048-token budget
over the stream's own wall — with no `ignore_eos` and no finish-reason
check. On this face that numerator is invalid: the story-continuation
seeds stop early (finish=stop at ~460–760 of the 2048 requested
tokens; the engine's `Running:` counts drop 8→7→6→5 inside every
rep — visible in the banked `r239b/gemma/serve.log`), so a stream that
generated ~460 tokens in ~18 s was scored as 2048/18 ≈ 114 tok/s.
The banked `tps_mean` values (48.5 / 59.2 / 45.1) are therefore
inflated ×2.3–3.1, and the derivation "8 × tps_mean = 361–474 tok/s
aggregate" (this VERDICT's units-artifact re-derivation and the
1.85–2.4×-the-era-record claim above) is RETRACTED.

Honest values, recomputed from the banked raw artifacts
(`reval_agg8.py` in this bank; engine 10-s telemetry integrated over
the client-side rep windows — client algebra and engine integration
agree: rep1's two early-finished streams ≈ 460 tokens each,
13,204 ≈ 6×2048 + 916):

| rep | client tps_mean (retracted) | engine-integrated tokens | honest wall aggregate | full-8-way steady (first buckets) |
|---|---|---|---|---|
| 1 | 48.5 (→388) | 13,204 over 78.4 s | 168.4 tok/s | 196.5 |
| 2 | 59.2 (→474) | 11,875 over 77.4 s | 153.4 tok/s | 211.2 |
| 3 | 45.1 (→361) | 11,631 over 78.4 s | 148.4 tok/s | 208.0 |

**The corrected cert28 gemma agg8 record: 24.6–26.4 tok/s per stream
steady under 8-way (solo 29.0 — the expected batch tax), 196–211
tok/s aggregate at full 8-way concurrency.** Same-lane comparison on
the era protocol (8×256, actual-token numerator — the vmatrix
`gf8` battery, all streams complete): 195.9–196.6 aggregate —
**parity held, no improvement claim, no regression**. The rep-span
aggregates (148–168) sit below the 256-budget batteries only because
the 2048-budget streams stop early at different points (face-shape,
not engine-shape).

Every other r239/r239b face in this VERDICT is unaffected: the
checkpoint-lane TTFT faces, probe98/G1/G2 walls, and solo 29.0 ×3 use
`usage`-returned actual tokens or validate against engine-side rates
directly. The qwen agg8 numbers elsewhere in the docs come from
`r113_battery.py` (actual `usage.completion_tokens` numerator) and
`p139_conc.py` (`ignore_eos: true`, `agg_tps_wall = conc·mt/wall`) —
both audit-verified clean of this defect. Doc rows corrected in the
r240-docs6 PR; the raw artifacts and SHA256SUMS of this bank are
untouched (VERDICT addendum only, per the banking law).
