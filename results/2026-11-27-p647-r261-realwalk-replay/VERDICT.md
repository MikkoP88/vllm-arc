# P6.47 r261 — the real-walk table replay is ALL CLEAN (60/60): the materialized crash tables are not the surface; the static audit finds the expand-view block-table stride defect; the r260 V3 description is corrected

**Date**: 2026-11-27 · **Leg**: r261 (plan §12 rung; the r260
next-rung: the corrected real-walk replay) · **Image**:
`vllm-arc:v0.30.0-b70-cert29` (cert29_fork_sha=16effb49, the r255
crash image) · **Rung chain**: r255 (3 crashes) → r256 (OFF-branch
delta) → r257 (faithful replay REPRODUCED + minimal clean) → r258a
(static audit) → r258b (table placement defect) → r259 (eager crash:
graphs exonerated) → r260 (face bisect: every variant crashes at the
multi rehit) → **r261 (this leg: the real walk is clean offline; the
static audit finds a stride contract violation; r262 launched)**.

## What ran

`tools/ar/r261_realwalk_replay.py` inside a no-serve container on the
crash image (driver `tools/ar/r261_realwalk_leg.sh`): the crash step's
six native calls (five windowed pools, then full-attn) at the engine's
exact geometry — q [118,16,256] bf16 over a [1600,64,8,528] mxfp8
pool, window (1023, 0), max_seqlen_k 40960; q [118,16,512] over a
[1600,128,2,1056] pool, windowless — with the block-table rows
placed exactly as the engine dumps them: 36 zero-pads at columns
0..35 and the 18 real ids at columns 36..53 (the columns the window
walk actually reads). The matrix: five tables (m8g / r257 / the r260
V3 + V4 crash rows / the consecutive-fresh control) × four pool fills
(zero / recycled / random / nanband) × three seeds.

`R261_DONE hangs=0 devlosts=0 cells=60` — **60/60 cells clean**,
every call `OK` (f0 + w0..w4), zero devcoredumps, zero kernel resets.

## Verdict

**The kernel walking the real crash tables over any pool content does
not reproduce the DEVICE_LOST.** The offline oracle is now exact in
table placement, geometry and call order and still cannot reproduce,
so the crash input is not "the kernel + these tables + this content".
Together with the earlier rungs the closure is complete on the
offline side:

- the bytes in the fresh windowed blocks are not the trigger in any
  fill class (r258b for the misplaced form, this leg for the real
  walk);
- graph capture/replay is not required (r259: eager crashes);
- any prior traffic suffices, the debris class is irrelevant (r260).

The `ARC_MXKV_NATIVE=0` lane ablation proposed as r262 in the r260
VERDICT is **impossible**: `engine.py:620-629` raises at init when a
sliding-window layer runs without the native lane. That proposal is
superseded.

## The static audit: the synth path hands the wheel a stride-0 expand VIEW

Re-reading the whole 118-row path after the all-clean result exposed
a contract violation between the engine's synth path and the wheel —
code-proven below, e2e proof is the r262 leg:

1. **The engine** (`vllm_gaudi/xpu/patches/mx_kv/engine.py`, synth
   path, continuation prefill `8 < q_len <= 128`):
   `bt_row = attn_metadata.block_table[i : i + 1]` then
   `synth_bt = bt_row.expand(q_len, -1)` — shape [q_len, 640],
   **stride (0, 1)**: every row is the SAME persistent-table row.
2. **The bridge** (`native_decode.py`) passes `bt_rows` verbatim.
3. **The wheel** (`mx_paged_decode_xe2.cpp`) takes
   `block_table.data_ptr()` and `block_table.size(1)` only — no
   stride, no contiguity check on the block table — and the kernel
   (`mx_paged_decode_mainloop.hpp:922-925`) addresses row `idx_b` at
   `idx_b * max_pages_per_seq` plus the column: **a contiguous pitch**.
4. **The persistent table** is a zero-initialized [64, 640] int32
   buffer per KV-cache group (`vllm/v1/worker/block_table.py`;
   `ARC_MAXSEQS` 64), whose slot-mapping buffer is allocated right
   after it. Only row `i` of the live request is ever written under
   single-request traffic.

So the kernel sees, for a synth call of `q` rows:

| kernel row | what it dereferences | as block ids |
| --- | --- | --- |
| 0 | the request's real table row | valid |
| 1..63 | the table's following rows (never written) | 0 = the null block |
| 64..q-1 | device memory PAST the [64, 640] table | whatever the allocator placed there |

The Triton lanes are stride-aware (`triton_mx_decode.py` multiplies
the row index by `block_tables.stride(0)`), and the vendor flow passes
`block_table.stride(0)` to its kernel — the expand view is correct
for both. Only the native-wheel windowed synth path is broken. The
target qwen model rides the Triton lane and is unaffected.

**This explains every data point in the rung chain**:

- every cold in the r260 bank is ONE full forward (`mql=msl`, the
  first-chunk flash path — the synth path is never entered) → always
  clean;
- the essay/code rehits are 56/57-row synth rides (`n_tok=56
  seqs=[2104]` / `n_tok=57 seqs=[5433]`, native `rows=56` / `rows=57`)
  → rows 1..55 stay inside the zeroed table → always clean, in every
  rung;
- the multi rehit is a 118-row synth ride (`n_tok=118 seqs=[3446]`,
  native `rows=118`) → rows 64..117 read past the table → the crash
  site in every rung (r255, r257 suite, r259, all four r260 variants);
- the r257 minimal pair and this leg's 60 materialized cells are
  clean because no materialized table ever contains those bytes —
  the crash input is traffic-dependent memory OUTSIDE the tables;
- the devcoredump reason ("LR job cleanup") is the hung-batch class.

A second consequence is silent: for every continuation row but the
first, the five windowed layers have been attending over the null
block (zeros), so **all banked gemma rehit outputs were computed with
windowed attention effectively disabled on those rows**. Rehit
outputs are expected to change once the fix lands; that is the
defect leaving, not a regression.

**Still open (r262 decides)**: which bytes lie past the table and
why some 118-row rides survive (the r257 minimal pair) — the probe
phase dumps exactly those rows — and the e2e A/B of the fix.

## Correction to the r260 VERDICT: the V3 description was a misread

The r260 VERDICT table and its "V3 subtlety" section state that V3
ran five multi generations as a prior face and that V3's class-2 cold
"is itself a prefix rehit" (sha `886716bff6e5`) that "ran CLEAN" with
the crash geometry. **The bank shows otherwise.** `r260/v3_multi_multi/`
holds exactly two `R260_GEN` lines — the cold (`886716bff6e5`, 6131
chars, 142.9 s) and `phase=rehit ERROR HTTP 500` — then
`[multi] ERROR at phase=rehit` and `R260_DONE crashed=True
faces=multi+multi`; the probe's `main()` stops at the first error.
The serve log shows the cold as ONE full 3446-token forward
(`mql=3446 msl=3446`, the first-chunk flash path — no prefix hit) and
the 118-row synth forward at 05:21:18 as the crash step.

Corrected facts:

- **V3 = boot → warmup → one multi cold → crash at the FIRST multi
  rehit.** There was no "clean rehit with the crash geometry"; all
  four variants crashed at the first multi rehit.
- The cold-sha divergence (`886716bff6e5` vs the cross-boot attractor
  `5c80063c79a1`) is unexplained and is NOT evidence of a rehit; a
  first-long-request-after-boot effect is the candidate.
- The "free-list provenance of the fresh windowed tail blocks"
  inference drawn from it is retracted.

The r260 headline verdict (every variant crashes at the multi rehit;
V4 is the minimal repro) stands unchanged. The r258b bank likewise
still carries its retracted exoneration text; both banks now carry a
correction addendum.

## W14 boot hang after this leg (operational)

The leg's restore (05:59:13Z) booted W14 to "JIT kernel warmup
finished" (06:01:11Z) and then hung: `No available shared memory
broadcast block found in 60 seconds` every 60 s until 07:46Z, both
workers' main threads in a sleep-poll, zero GPU util, no kernel
errors, no stale device holders. A plain start + serve does not
clear it; **`docker restart radiance-test` + relaunch does**
(07:47:59Z → health 200 at 07:52:48Z). Evidence banked in
`w14_hang/` (thread states, dmesg, the container's serve log). The
r262 leg's restore is health-aware with one such retry.

## Next rung (r262, LAUNCHED)

`tools/ar/r262_expandview_leg.sh` — the e2e A/B on the same image
with an mx_kv overlay mounted by the wrapper's `ARC_MXKV_OVERLAY`
hook (built from the image's own bytes by `tools/ar/r262_patch.py`;
mode baked into the bytes, sha256-proven inside the container):

- **a** probe overlay (fix + a row dump of exactly what the wheel
  dereferences, once per pool): face `multi` → the bytes past the
  table, then the fixed ride;
- **c** fix overlay (the landing candidate: the bridge normalizes a
  non-contiguous `bt_rows` to contiguous): faces `essay code multi
  simple multi` → 25 gens;
- **b** control overlay (boot marker only): face `multi` → the crash
  must recur, proving the mount alone does not defuse it.

Fix proven → land it in the tree → re-leg m8g/m4g/mkg → re-gate
(r255-class) → the cert29 cut.

## Bank

- `r261/matrix.out`, `r261/results.json` — the 60 cells (per call
  OK, per-cell wall time: fills dominate — zero/nanband ~50 ms,
  recycled ~4.5 s, random ~12 s).
- `r261/r261_realwalk_replay.py` — the harness as run.
- `r261/dmesg_final.txt`, `r261/devcores/` (empty), `r261/SHA256SUMS`
  (rig-side).
- `launch.log` — the driver log (matrix + W14 restore line).
- `w14_hang/` — the W14 boot-hang evidence (threads.txt, dmesg.txt,
  serve_posture.log).
