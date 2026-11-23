# P6.47 cert27 battery — VERDICT: PASS

Date: 2026-11-23 · Image: `vllm-arc:v0.30.0-b70-cert27` · fork_sha
`d39ab5a26bf737d006d47a3f3125a404dca6e928` (content commit) · wheel
`vllm_xpu_kernels-0.1.14.1+arc7mx` build25 md5 `aeca45daa740e9973a1c51bfc5ea2bc4`,
installed `libmqa_logits_kernels_xe_2.so` md5 `190e261edc79b3b497d8d8e801e65757`
(both asserted in-image by r227).

cert27 = cert26 + the r225-fixed arc7mx wheel + the r224a `ARC_MXKV_AB`
diagnostic instrument (default-off). Certified posture change vs cert26:
**mkq (mxk8v4 native) UN-REFUSED** — the r223 refusal amendment flips back.

## Gate result

`P647_CERT27_GATE_COMPLETE -- PASS` (r228, final line of
`battery_launch.log`). W14 restored by the wrapper trap at 13:51:50Z.

| leg | model / lane | LONGRETURN | T1_SEED | longprobes | mx_backend | native_lines | lmcache | tracebacks | XPU KV tokens |
|-----|--------------|-----------|---------|------------|-----------|--------------|---------|------------|---------------|
| ctrl | qwen fp8, no MX | PASS | PASS | CLEAN | 0 | 0 | 4 | 0 | — |
| m8q | qwen mxfp8 native | PASS | PASS | CLEAN (full+r1+r2+r3) | 1 | 108 | 0 | 0 | 386,484 |
| m4q | qwen mxfp4 native | PASS | PASS | CLEAN (full+r1+r2+r3) | 1 | 108 | 0 | 0 | 721,994 |
| mkq | qwen mxk8v4 native | PASS | PASS | CLEAN (full+r1+r2+r3) | 1 | 108 | 0 | 0 | 504,832 |
| m8g | gemma mxfp8 | FAIL (bank-only) | PASS | CORRUPTION_SIGNAL | 1 | 0 | 0 | 0 | — |
| m4g | gemma mxfp4 | FAIL (bank-only) | PASS | CORRUPTION_SIGNAL | 1 | 0 | 0 | 0 | — |
| mkg | gemma mxk8v4 | FAIL (bank-only) | PASS | CORRUPTION_SIGNAL | 1 | 0 | 0 | 0 | — |
| g0 | gemma fp8 control | FAIL (bank-only) | PASS | full=attractor, r1/r2 CLEAN | 0 | 0 | 0 | 0 | — |

GEMMA_COMPARE: **PASS** (no new modes / no amplification; known-class churn
banked for review).

## The headline: mkq native un-refused

cert26 mkq (mxk8v4, `ARC_MXKV_NATIVE=1`) failed with the REP-LOOP refusal
class across ALL prompt classes — longprobe CORRUPTION_SIGNAL on every pass,
garble non-empty, churn drifting. cert27 mkq on the same posture:

- LONGRETURN PASS: essay/code/multi/simple all single-hash across 3 cycles
  (`575444746e66` / `23d1fc474d91` / `41338d3d088a` / `84efc84cc142`),
  finish=stop, chars 4146/3042/529/1510, garble_empty, t1_seed_identical.
- All four longprobes CLEAN (full + r1 + r2 + r3 rehits deterministic).
- native_lines=108 with lmcache_lines=0 and tracebacks=0 — the wheel native
  path served the whole battery leg.

Root causes fixed by the wheel (r225 close-out, provenance
`tools/ar/r225_fix*_apply.py`): fix1 e2m1 subnormal mantissa (implicit bit +
OCP decode), fix2 `sycl::exp2` in the split-KV combine rescale (evicting
`native::exp2`'s ~1e-3 relative error), fix3 fp32 split partials
(`ElementOAccum float`; single bf16 quantization point at the final combine),
fix5 the S==1 sentinel-LSE predicate aligned to the O-normalize arm.

## Cross-image note (m8q/m4q output drift is expected)

cert26→cert27 essay hashes moved on the already-native lanes (m8q
`e717518d6875`→`575444746e66`, m4q `508ba7cd542d`→new): the r225 fixes
CHANGE native numerics — that is their purpose (r224 refusal lane-gap max
2.273e+00 → r226a p50 5.5e-2, nan 0/120). The certified gate is per-boot
self-consistency (3-cycle single-hash + T1_SEED + CLEAN rehits), not
cross-image hash stability; both hold.

## Gemma legs = documented native-KV baseline (bank-only, gated=0)

All three gemma MX legs reproduce exactly the g0 control's flag signature
(REP-LOOP on essay/code/multi/simple, topwin 0 everywhere = no
amplification) with churn destinations inside the documented known-class
set (e.g. `12c0e71e5be4`→`fc2dddcc7ce6`, the same destinations as the
cert25y/cert26 banks; m4g/simple's three-hash churn `4ef5307e6135`/
`815df9453920`/`af456639ad9d` also known-class). g0 full = first-cycle
attractor with r1/r2 rehits CLEAN — the established gemma think-spam
baseline (REP-LOOP baseline law). Verdict via gemma_compare vs g0: PASS.

Note the gemma MX legs show native_lines=0: gemma's decode buckets sit
below the capacity-priced `_NATIVE_MINLEN` thresholds, so gemma stays on
the Triton lane by construction — their CORRUPTION_SIGNAL is the model's
native-KV baseline, not an MX defect.

## Method

r228_cert27_gate.sh (= r223 structure + cert27 deltas): legs
ctrl/m8q/m4q/m8g/m4g/mkq/mkg/g0, each = boot (eager, spec-decode k4,
fp8 weights; MX legs `ARC_KVDTYPE` + `ARC_MXKV_NATIVE=1`) → garbage suite →
LONGRETURN 3-cycle hash gate → T1_SEED → longprobe full + 3 rehits → churn
900/1800/2700 → metrics + markers; gemma legs gated=0 (bank-only) with the
verdict delegated to gemma_compare vs g0; final W14 restore via guarded
trap (`w14_restore_p647c27.log`). Refuses: image manifest cert27_fork_sha,
in-image content markers, `libmqa_logits` .so md5 pin, stale-tree greps.

Evidence chain standing behind this image: r225t offline q-cast-matched
gate 12/12 on the r224a payloads; r226a e2e re-entry 8/8 fresh COVERED
payloads + engine A/B nan=0/120; build25 compile rc=0.

## Hygiene

- 8 serve.logs redacted rig-side (rig address → `REDACTED-RIG-ADDR`):
  176 occurrences → 0 bank-wide; post-redaction `SHA256SUMS` regenerated
  over 194 files (see `REDACTION-serve-logs.md`).
- Artifacts: this directory (rig home `/root/build/p647/cert27/`).
