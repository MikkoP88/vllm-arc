# r255 — the cert29 section-10 gate (P6.47)

**VERDICT: SPLIT — 9/10 legs structurally green + the P-C2 refusal face green
on the r255a rerun, but the gate is NOT cert-cutting evidence: all three
gemma MX legs died mid-suite at a deterministic probe point (DEVICE_LOST),
an OPEN defect that blocks the cert29 release.**

## What ran

`tools/ar/r255_cert29_gate.sh` (first pass md5 `bb671486edcf8add69680fe09bbb390d`,
launched 2026-11-26T22:09:03Z; the r255a gref-harness-fix rerun md5
`6a6f285720d96234c400621c2167d1bd`, launched 2026-11-27T01:11:11Z) on
`vllm-arc:v0.30.0-b70-cert29` (FORK_SHA `16effb497cd36cb25df7c19cb6f1938d2d45cdc5`).
First pass: `P647_CERT29_GATE_COMPLETE -- GATE_FAIL=1` (the gref harness
defect only). r255a single-leg rerun: `P647_CERT29_GATE_COMPLETE -- PASS`.

## The green record (first pass, 9 legs + compare)

| leg | posture | result |
| --- | --- | --- |
| ctrl | qwen fp8 baseline, LMC on | HEALTH 22:19:33, 3x PROMPT_OK (Paris), lmcache_lines=7, ERROR=0 |
| m8q | qwen + mxfp8 native | green; pool 386,484 tok (1.47x); full suite + rehits + churn 5.6-6.1 s/256tok |
| m4q | qwen + mxfp4 native | green; pool 721,994 tok (2.75x) |
| mkq | qwen + mxk8v4 native | green; pool 504,832 tok (1.93x) |
| lmc | qwen + mxfp4 + LMC interop (knob-free) | **G1-G5 all green**: packed kvshape, clean face, A/A re-hit 6.047->1.894 s (<50% wall), c32768 death sequence clean (no wedge), zero transfer errors; interop_live=3 lmcache_lines=7 |
| m8g | gemma + mxfp8 + native + window | boot green (pc2_windowed_impl=100, native=54, pool 54,426 @1.33x) — **then DEVICE_LOST mid-suite (below)** |
| m4g | gemma + mxfp4 + native + window | boot green (pool 105,669 @2.58x) — **then DEVICE_LOST mid-suite** |
| mkg | gemma + mxk8v4 + native + window | boot green (pool 71,856 @1.75x) — **then DEVICE_LOST mid-suite** |
| g0 | gemma native-KV control (fp8 weights) | **fully clean**: full ungated suite, ERROR=0 TRACEBACK=0, churn 9.4-9.6 s/256tok, mx_backend_lines=0 |
| gref | gemma + mxfp8 NO native — must refuse | **GREF_PC2_REFUSAL_OK refusal_lines=2** on the r255a rerun (first pass was a harness defect, see below) |
| compare | gemma comparative | GEMMA_COMPARE: PASS over the legs that produced samples |

## THE OPEN DEFECT — the gemma MX legs die deterministically at the
longprobe `multi` face's A/A resend

All three gemma MX-KV + native + window legs (m8g/m4g/mkg — every MX
dtype) crashed with the byte-identical signature:

- `RuntimeError: level_zero backend failed with error: 20
  (UR_RESULT_ERROR_DEVICE_LOST)` in the async model-runner output
  (`graph_safe_output.py` copy_event sync), EngineCore fatal, APIServer
  500s, full engine shutdown ~15 s before each leg's close
  (23:34:35 / 23:53:41 / 00:14:31).
- The faulting step is the SAME request in all three legs:
  `prompt_token_ids_len=3446`, `num_computed_tokens=3328`,
  `num_scheduled_tokens=118` — the p647_longprobe `multi` face
  (multilingual prompt; gemma-tokenized 3446 tok) on its SECOND sample:
  an A/A prefix-cache-hit resend (26 full page-128 blocks cached + a
  118-token tail chunk). The crash fires ~4 s into the resend, during
  the tail-PREFILL step.
- The probe's other faces survived their A/A resends on the same legs:
  `essay` (2,200 tok — tail 24) and `code` (4,400 tok — tail 48) both
  banked real data. The multi face's 118-token tail is the only one
  above the 64-token boundary (page-64: the only MULTI-BLOCK tail).
- dmesg (rig): `Engine reset: engine_class=ccs` fired on BOTH GPUs
  simultaneously (23:34:17 / 23:53:22 / 00:14:13, 13-18 s before the
  python error surfaced) + a bcs (copy-engine) reset ~11 s later — a
  deterministic faulting COMPUTE kernel in the tail-prefill step on
  both TP ranks. The devcoredumps were auto-deleted by the retention
  window (00:37:34) before capture — the r256 repro must grab
  `/sys/class/drm/*/device/devcoredump/data` immediately.
- The survivors bracket the delta: the same probe ran CLEAN on the qwen
  legs (m8q/m4q/mkq — same faces, same A/A resends, no window); g0
  (gemma, native-KV, standard lane) ran the full suite CLEAN; the r245
  e2e validated this exact gemma posture (cert28 + build34 wheel +
  P-C2 overlay) on 45 FRESH faces with zero tracebacks — but never
  exercised an A/A prefix-hit TAIL prefill. The window twin (D=256,
  W=1024, page 16/64) and/or the page-128 full-attention lane (D=512,
  build34's `_512/_16` sub-tile frame) under a prefix-hit tail prefill
  is the implicated path.

Consequences in the bank: the m8g/m4g/mkg legs' longprobe rehits
(r1/r2/r3), all three churns (`CHURN 4x256 done: []` — connection
refused against the dead port), xcheck (0) and erraudit (ERROR=65
TRACEBACK=4) are the death record, not serving evidence; the
GEMMA_COMPARE PASS is comparative over the surviving samples (g0 + the
legs' pre-crash faces) and does not clear the MX legs' e2e death
(the existence-only-green-gate law).

## The gref first-pass harness defect (r255a, fixed + rerun green)

The first-pass leg_gref loop broke at i=0 on "pgrep EngineCore absent" —
true of every healthy boot's first ~60 s — then docker-cp'd before the
in-container `serve_full.log` existed (`serve.log MISSING`), verdicted
FAIL on zero evidence, and pkill'd the 4-second-old boot before the
P-C2 refusal could fire at model init (~1.7 min in). The r255a fix
(sleeps first, copies the serve log every poll, breaks on the refusal
line, SEEN-latches process death, 90x10 s ceiling) reran gref alone:
`GREF_PC2_REFUSAL_OK refusal_lines=2` at 01:12:54 — the engine refused
the windowed MX impl off-native, boot-loud
(`MxAttentionImpl with sliding_window=1024 requires the native decode
lane (set ARC_MXKV_NATIVE=1)` from Worker_TP1), exactly the certified
guard. The P-C2 refusal face is GREEN.

## Banked files

Both launch logs (`launch.log` = the 10-leg battery, `gref_rerun_launch.log`
= the r255a gref rerun), per-leg dirs (boot/markers/battery/garbage/
longprobe/churn/xcheck/erraudit/metrics/native_telemetry/serve.log,
SHA256SUMS over the data files). Host addresses byte-redacted
(`10.20.3.REDACTED` / `10.100.8.REDACTED`, perl).

## Verdict

- QWEN side (the target model): CERTIFIED CLEAN — the fp8 baseline, the
  full MX-KV trio (mxfp8/mxfp4/mxk8v4 + native + auto-off LMC), and the
  P-H interop lane (G1-G5) all green on the first pass. The cert29
  image is superior for the qwen3.8-27b-fp8 target.
- GEMMA side: the P-C2 refusal face green; the gemma MX e2e lane has an
  OPEN deterministic DEVICE_LOST (prefix-hit tail prefill >64 tok). The
  g0 native-KV control is clean.
- **The cert29 tag/release does NOT cut until the DEVICE_LOST is
  root-caused and fixed and the gemma MX legs re-run clean** (the cert
  gate law: all issues fixed). Next rung: r256 — the minimal repro
  (boot gemma m8 posture once; multi prompt fresh -> resend; capture the
  devcoredump immediately; bisect tails 24/48/63/64/65/96/118/128) ->
  root cause in the windowed/page-128 prefill path -> fix -> re-leg.
