# P6.47 r259 — the eager suite replay: CRASHED — the graph mechanics are exonerated, the surface is engine-level state

**Date**: 2026-11-27 · **Leg**: r259 (plan §12 rung; the r258b
triangulation's decisive discriminator) · **Image**:
`vllm-arc:v0.30.0-b70-cert29` (cert29_fork_sha=16effb49, the r255
crash image) · **Rung chain**: r255 (3 crashes) → r256 (OFF-branch
delta) → r257 (faithful replay REPRODUCED + minimal clean) → r258a
(static audit closed) → r258b (content classes exonerated in the
direct eager form) → **r259 (this leg: the crash reproduces WITHOUT
graphs)**.

## What ran

`tools/ar/r259_eager_leg.sh` — the r257 replay phase B EXACTLY (the
ACTUAL `p647_longprobe.py` v3 full suite, the identical binary probe,
md5 e452d0ce) on the m8g posture (mxfp8 KV + native + P-C2 window, the
enable_thinking OFF branch in every request) with ONE env delta:
**ARC_GRAPH=0** — the eager engine, the identical native calls running
python-side instead of as captured-graph replays. Devcore watcher live
(2 s cadence, the r257 apparatus).

Posture evidence (markers.txt): `mx_backend_lines=1`,
`native_lines=6`, `pc2_windowed_impl=100`, `lmcache_lines=0`,
`tracebacks=0` — and the DELTA evidence: **`xpu_graph_lines=0`** — a
clean eager boot, no graph capture activity anywhere in the serve log.

## Verdict

**CRASHED — the suite reproduces WITHOUT graphs.**

- The multi face's re-sample (temp 0.0, max_tokens 1536 — the rehit
  class; the request's `num_computed_tokens=3328` proves prefix-hit)
  hit the `UR_RESULT_ERROR_DEVICE_LOST` at 04:21:45, ~23 min into the
  eager suite (essay and code faces had completed clean before it).
- `DEVICE_LOST=3, TRACEBACK=4` in the serve log; engine dead
  (`HEALTH_DEAD` at capture).
- Both GPUs' engine resets at 04:21:27 (dmesg, `engine_class=ccs`,
  guc_id 22/28 — the same ccs compute-reset class as the r257 crash's
  02:45:59/02:46:19; the dmesg ring also still carries those earlier
  r257-era lines).
- **2 devcoredumps banked** (both cards, the eager crash's dumps).
- The probe's final line: `LONGPROBE p47r259 full: CORRUPTION_SIGNAL`
  (its classifier sees the post-crash 500s; the driver's
  `R259_EAGER_CRASHED` is the leg verdict).

## The crash dump — the geometry is IDENTICAL, the state is structural

The dump_input fired on the fatal path again (SchedulerOutput + stats):

- `prompt_token_ids_len=3446, num_computed_tokens=3328,
  num_scheduled_tokens=118` — the exact r255/r257 geometry; the crash
  ordinal rides the multi prompt's re-sample in every rung.
- `PrefixCacheStats(requests=1, queries=3446, hits=3328)` — the crash
  request is a **prefix rehit**: 3328 of 3446 tokens prefix-served;
  only the 118-token tail scheduled — the largest continuation class
  (the 118-row bucket).
- `num_common_prefix_blocks=[0, 0, 0, 0, 0, 27]` — the full-attn pool's
  27 blocks prefix-shared (27×128=3456 ≥ 3328); the five windowed pools
  share none (the sliding-window class is not prefix-cacheable — its
  KV rides freed-and-reissued blocks).
- The six per-pool block tables: the five windowed rows share their
  FIRST ~15 ids with the r257 crash tables (the LIFO free list
  deterministically reissues the cold sample's own window blocks — the
  rehit's window-region KV is the cold's content, which is why clean
  rehits produce correct output) and differ ONLY in the last ~2-3 ids
  (r259 g0 tail `[52, 236, 358]` vs r257's `[52, 177, 231]`) — the
  fresh blocks for the 118-token tail, drawn from the fragmented
  free list (essay/code-era debris ids). The full-attn row matches
  r257's in its first 26 ids (the prefix-retained blocks are
  allocator-stable across boots) and differs only in the 27th (the
  partially-filled fresh block).
- `new_block_ids_to_zero=null` — the windowed pools still never zero
  fresh blocks (the standing anomaly: the full-attn pool zeroes, the
  windowed pools do not).

## Where this leaves the hunt

The mode question is CLOSED. The triangulation:

| leg | engine | mode | state | result |
| --- | --- | --- | --- | --- |
| r257 minimal pair | engine | graph | pristine | clean |
| r258b | NO engine (direct native calls) | eager | the crash tables + garbage bytes | clean |
| r257 replay | engine | graph | suite state | DEVICE_LOST |
| **r259** | engine | **eager** | suite state | **DEVICE_LOST** |

**Engine + suite state is the necessary pair — graphs are NOT a
component.** The remaining engine-only deltas (the direct-call
synthetic does not model them): the same-step store kernels (the
118-token continuation writes the fresh blocks immediately before the
windowed native calls read them), the full 60-layer interleave (50
windowed + 10 full-attn native calls + stores per step), the real
pool tensors at the real VRAM pressure (GMU 0.8), and the GuC queue
depth.

**Next rung (r260, launched)**: the suite-face bisect — four
fresh-boot eager phases, each one prior face's full class then the
multi class: V1 essay→multi, V2 code→multi, V3 multi→multi (the same
class's own debris), V4 simple→multi (78-tok traffic). The r257
minimal pair already proved multi-ALONE clean on a pristine boot, so
the bisect finds the minimal prior-state prefix. Whichever variant
crashes becomes the cheap repro for the fix legs; a leading fix
candidate emerged from this leg's dump analysis: extend
`new_block_ids_to_zero` to the windowed pools (zero the fresh windowed
blocks at alloc — the full-attn pool already does; the windowed
omission is the anomaly, and the fix would eliminate the entire
recycled-content class from the fresh-block path).

## Bank

- `r259/` — boot.txt, markers.txt (the posture + xpu_graph_lines=0
  delta evidence), longprobe_full.out (the suite: essay/code clean,
  multi 500), serve.log (docker cp), erraudit.txt (DEVICE_LOST=3,
  TRACEBACK=4), req_lines.txt, devicelost_context.txt (the full crash
  dump incl. the SchedulerStats prefix line), dmesg_final.txt (the
  04:21:27 ccs resets + the older r257-era lines in the ring),
  devcores/ (2 dumps: both cards), SHA256SUMS.
- `launch.log` — the rig-side driver log (refuses, image identity,
  the W14 stop/restore cycle).
