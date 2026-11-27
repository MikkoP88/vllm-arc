# P6.47 r258b — the synthetic fragmented-table replay: CLEAN (36/36 cells, 216/216 native calls OK)

**Date**: 2026-11-27 · **Leg**: r258b (plan §12 rung; r258a static audit →
r258b offline content repro) · **Image**: `vllm-arc:v0.30.0-b70-cert29`
(cert29_fork_sha=16effb49, the r255 crash image) · **Rung chain**:
r255 (3 banked crashes) → r256 (OFF-branch delta) → r257 (faithful replay
REPRODUCED + minimal pair clean → state delta isolated) → r258a (static
audit: every walk/plan/reduce derivation bounded for allocator-issued ids)
→ **r258b (this leg: the table-content + pool-bytes classes exonerated in
the direct eager form)**.

## What ran

`tools/ar/r258b_frag_leg.sh` booted a no-serve harness container on the
cert29 image and ran `tools/ar/r258b_frag_replay.py` (docker exec, the
r257 apparatus) directly against the arc7mx wheel's
`mx_paged_decode` — no engine, no serve. The matrix:

- **tables** (byte-exact literals from the banked crash dumps via
  `tools/p647_r258_crashtables.py`):
  - `m8g` — the r255 m8g crash leg's six per-pool rows (5 windowed
    page-64: 36 zero-pads + 18 real fragmented ids at 36..53; 1
    full-attn page-128: 27 real at the head),
  - `r257` — the r257 faithful replay's rows (same structure, entirely
    different id generation — the second banked crash),
  - `ctl` — the pristine-state control (consecutive fresh ids, the
    r256/r257-minimal free-list shape).
- **fills** (the never-zeroed fresh-block bytes — `new_block_ids_to_zero=null`
  in both crash dumps): `zero` (driver-fresh, the minimal-pair analog),
  `recycled` (e4m3 encodings of N(0,1)·0.5 + e8m0 scale codes 119..125 —
  the legit store band), `random` (whole-pool uint8 — ancient-garbage
  extreme incl. NaN codes), `nanband` (adversarial: every scale byte
  0xFF, every payload byte 0x7F).
- **seeds** 0/1/2 (q + fill vary). 3 × 4 × 3 = **36 cells**.

Per cell, the faithful step order — the six per-pool native calls of the
crash step, in the engine's exact form (native_decode.py + the banked
capture telemetry):

- windowed: q `[118, 16, 256]` bf16 (Hq 32/TP2), pool `[1600, 64, 8, 528]`
  uint8 (Hk 16/TP2 → stride(1)=4224 as banked), bt `[118, 640]` (the
  40960-capacity width; 36 head pads + ids at 36..53), `max_seqlen_k=40960`
  (the r212f walk-capacity law), `softmax_scale=256**-0.5`,
  `window_size=(1023, 0)` — the 256/64-via-_16 masked twin at qgroup 2
  (the _8 bucket).
- full-attn: q `[118, 16, 512]`, pool `[1600, 128, 2, 1056]` (Hk 4/TP2 →
  stride(1)=2112 as banked), bt `[118, 320]` (ids at the head 0..26),
  `max_seqlen_k=40960`, `softmax_scale=512**-0.5`, `window_size=None` —
  the 512/128-via-_16 windowless entry at qgroup 8. Both classes ride the
  NATIVE wheel on the ARC_MXKV_NATIVE=1 posture (the banked `window=-1`
  telemetry line).

sl = 3329..3446 int32 (computed=3328 scheduled=118, the exact r255/r257
geometry), cu_q = arange(119), splits=auto. Per-call 45 s SIGALRM
watchdog (the pool fills hoisted outside), UR_RESULT_ERROR_DEVICE_LOST →
DEVLOST, canary matmul after every cell, devcore watcher on the driver.

## Run-1: vacuous (banked as evidence, no claim)

The first harness shipped with positional `0` in the `fmt` slot — every
cell died at w0 with `ERR:ValueError: unknown MX KV format: 0`. 36/36
cells vacuous (run1/). The geometry audit that followed (model config +
banked telemetry + the dispatch ladder) is what produced the faithful
form above — Hk 8/2 (not 2), bt widths 640/320 (not 54/27),
max_seqlen_k 40960 (the capacity, not the request span), and BOTH classes
native (run-1 wrongly drove the full-attn twin through the triton lane).

## Verdict

**CLEAN — no repro.** `R258B_DONE hangs=0 devlosts=0 cells=36`; the
banked matrix.out carries **216/216 native calls OK, zero non-OK**
(36 cells × 6 calls; recount from the banked artifact, not the pull
script's census line — its regex only matches the call after the opening
`[`). No devcores (no GPU resets), canary clean throughout.

The two content classes the r258a audit left open are both exonerated
in the direct-call form:

1. **Fragmented block-id order** — the exact m8g and r257 crash-table
   literals (and their 640/320-wide capacity forms) walk clean.
2. **Never-zeroed fresh-block bytes** — recycled / random / nanband
   garbage in every walked block does not hang, devlost, or ERR.

## Where this leaves the hunt

The triangulation (all three on the cert29 image):

| leg | graph | state | result |
| --- | --- | --- | --- |
| r257 replay (the actual suite) | XPU-graph replay (rows=118 is a captured bucket) | suite-fragmented allocator | **DEVICE_LOST** |
| r257 minimal pair | XPU-graph replay | pristine | clean |
| r258b (this leg) | **eager direct calls** | the crash tables + garbage bytes | **clean** |

So the crash needs MORE than (table content) + (pool bytes) + (the
faithful call geometry). The remaining deltas, in decision order:

1. **The XPU-graph replay context** — the banked capture telemetry shows
   a `rows=118` native bucket (both workers), so the crash step was a
   REPLAY of a captured graph: frozen launch args, static cu_q/sl/bt
   buffers (the bt copy-in writes only the request's 54 entries — the
   640-wide static row's tail holds capture-dummy ids), and the
   command-graph execution (the whole 60-layer step as recorded
   commands; the hung kernel rode as job seqno 280 inside it).
2. **The engine-level cross-request state** — the suite's alloc/free
   churn, the GuC queue depth, the inter-request sequencing.

**Next rung (r259)**: the decisive discriminator is cheap — the r257
suite replay with `ARC_GRAPH=0` (eager engine, same posture otherwise).
Crash → the hang is engine-level state (graph-independent) → the surface
is the allocator/queue layer. Clean → the graph replay is a necessary
component → r260 = the offline graph-capture synthetic (capture a native
call under xpu_graph with a ctl table, then write the crash tables into
the static buffers and replay — the frozen-plan × stale-tail × replay
surface, all offline-reproducible per the pageable-H2D capture law).

## Bank

- `run2/` — the real matrix: matrix.out (verdict lines + per-call
  exceptions), results.json (structured), dmesg_final.txt (no resets),
  the harness copy (md5 c2eb579b361703c7e5fd2ddd025fd77b), SHA256SUMS.
- `run1/` — the vacuous fmt-slot run (36/36 ERR:ValueError), banked as
  harness-bug evidence.
- `launch.log` — the rig-side driver log (refuses, image identity, the
  W14 stop/restore cycle).
- No devcores (zero GPU resets — the watcher ran, nothing to capture).

W14 restored by the driver's health-aware trap (serve re-booted
post-leg; health-200 verified before session end).

## Correction addendum (r260 / r261, 2026-11-27)

The exoneration claimed above is RETRACTED (r260): `make_table(tail=True)`
placed the 18 windowed ids at columns 622..639 of the 640-wide table,
while the kernel's window walk reads columns 36..53 — every cell read
block 0 eighteen times and the real crash ids never entered the walk.
The 36/36 CLEAN verdict exonerates only the mis-placed form. The graph
replay is not a component either (r259: the eager suite crashes). The
corrected real-walk replay (r261, 60/60 clean) and the static audit
that followed — the synth path's stride-0 expand VIEW of one
block-table row is the defect — are in
`results/2026-11-27-p647-r261-realwalk-replay/VERDICT.md`.
