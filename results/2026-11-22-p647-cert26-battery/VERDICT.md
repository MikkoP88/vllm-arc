# VERDICT — P6.47 r223 cert26 full battery (2026-11-22)

**PASS** — all legs under the amended gate. Image
`vllm-arc:v0.30.0-b70-cert26` (manifest
`cert26_fork_sha=cac6393159bdf1f6356e744159163fd679726221`,
refuse-gated at battery start). cert26 = cert25zc content + the arc7
MX-native decode line: the arc7mx wheel
`vllm_xpu_kernels-0.1.14.1+arc7mx` (md5 `62f234c8d0aa89c4aec3e8d2c6fa22ad`,
byte-pinned) BAKED + the fork mx_kv overlay native_verify bridge.

## Verdict of record

The battery instance ran the full 8-leg ladder with mkq-native first;
that cell FAILED its absolute LONGRETURN gate (REP-LOOP garble all 4
classes + rehits, deterministic — T1_SEED PASS) while the
uniform-dtype native lanes (m8q/m4q) are absolute-clean and cert25zc's
triton mkq was clean. Attribution: the native MIXED-fmt lane (e4m3 K +
e2m1 V) — per-tensor K/V layout divergence (kv-store separate-strides
law) that the r220c offline value gate's CONTIGUOUS synthetic pool
structurally could not see. **Posture amendment**: native mxk8v4
REFUSED at this gate; mkq serves the cert25zc triton lanes; the failed
cell is banked verbatim as `mkq_native_refused/` (the refusal
evidence); r224 defined as the reopen path (engine-stride-faithful
value gate + native-vs-triton mkq A/B). The compliant mkq cell was
rerun under the amended script — `P647_CERT26_GATE_COMPLETE -- PASS`.

Process note: the first battery instance's epilogue (SHA256SUMS +
completion lines) was killed by a mid-flight script edit (bash
re-read the replaced file at a stale byte offset — the wrapper-edit
hazard class); all 8 legs' evidence completed and banked before it.
The sums here are regenerated from the redacted bank per the
recompute law.

## Posture of record (cert26)

- **Native MX decode: m8q + m4q** (`ARC_KVDTYPE` +
  `ARC_MXKV_NATIVE=1`; NSPLIT + P64NATIVE unset = C++ auto at the
  validated optimum on the `_16` frame — r219b A/A within 0.08 pct of
  the pinned path; r220 arbiter closed the 16->64 tile lever).
- **mkq: cert25zc triton lanes** (native mixed-fmt refused, see above).
- **Gemma MX legs: cert25zc triton lanes** (the r216-r221 native
  evidence chain is qwen-only).
- LMC posture unchanged: ALL MX lanes auto-off, ctrl keeps LMC-on.

## Quality faces

| leg | mx | native | LMC | probe | P1 | garbage | longreturn | T1_SEED | erraudit |
|-----|----|--------|-----|-------|----|---------|-----------|---------|----------|
| ctrl | 0 | 0 | on(4) | 3/3 | Paris | - | - | - | ERROR=0 |
| m8q | 1 | 108 | auto-off | 3/3 | Paris | PASS | PASS | PASS | ERROR=0 |
| m4q | 1 | 108 | auto-off | 3/3 | Paris | PASS | PASS | PASS | ERROR=0 |
| mkq | 1 | 0 | auto-off | 3/3 | Paris | PASS | PASS | PASS | ERROR=0 |
| m8g | 1 | 0 | off | 3/3 shape | - | PASS | banked* | PASS | ERROR=0 |
| m4g | 1 | 0 | off | 3/3 shape | - | PASS | banked* | PASS | ERROR=0 |
| mkg | 1 | 0 | off | 3/3 shape | - | PASS | banked* | PASS | ERROR=0 |
| g0 | 0 | 0 | off | 3/3 | Paris | PASS | banked* | PASS | ERROR=0 |

*Gemma legs are comparative (reploop-classifier-baseline law): the
native g0 control itself fails LONGRETURN on the same REP-LOOP classes
(think-spam attractor is BASELINE on native KV — r164 disposition);
`GEMMA_COMPARE: PASS (no new modes / no amplification; known-class
churn banked for review)` is the gemma verdict of record
(`gemma_compare.txt`). Qwen lanes gate ABSOLUTE and are clean 3/3 —
including both native lanes.

`mkq_native_refused/`: LONGRETURN FAIL (REP-LOOP x9), T1_SEED PASS,
garbage PASS, erraudit ERROR=0 — banked as the refusal evidence.

## Decode smoke face (256-tok solo / 2048-tok agg8, tps medians)

| leg | solo | agg8 | agg8/ctrl | vs cert25zc agg8 |
|-----|------|------|-----------|------------------|
| ctrl | 60.7 | 303.4 | 1.00 | 304.1 |
| m8q | 48.0 | 254.0 | 0.84 | 236.7 (+7.3%) |
| m4q | 50.7 | 256.1 | 0.84 | 229.0 (+11.8%) |
| mkq | 50.2 | 235.3 | 0.78 | 226.6 (triton lane) |
| g0 | 27.7 | 192.5 | - | 192.5 |
| m8g | 25.2 | 144.8 | 0.75 (vs g0) | 144.5 |
| m4g | 26.6 | 142.6 | 0.74 (vs g0) | 142.2 |
| mkg | 26.1 | 138.2 | 0.72 (vs g0) | 138.0 |

The native line closes ~9 points of the agg8 gap on both native lanes
(m8q 0.78->0.84x, m4q 0.75->0.84x of ctrl) at this short smoke face.
The long-ctx faces carry the r219b evidence: mxfp4 itl 48.5/51.2/56.8
at 8K/16K/32K (auto splits at the validated 128 optimum, -19.2 pct vs
auto-40, tps 65.4 @32K, 1.34x fp8 walk-parallelism) — banked at
`results/2026-11-22-p647-r219b-*/` and carried in the image manifest
narrative. Open levers after cert26: the MX walk 14.3 ms/step vs fp8
~1.5 (r218 table: floor 5.6 + union 8.7), the r222 reduce-side
skip-empty-splits candidate (+1.6 ms/step at 1K-class ctx, r221d), the
r224 mixed-fmt stride diagnosis (mkq native reopen), the r221c
dual-GPU lockstep probe (tile-lever reopen instrument).
