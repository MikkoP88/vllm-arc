# P6.47 r260 — the suite-face bisect: ALL FOUR VARIANTS CRASH at the multi rehit; the r258b content exoneration is RETRACTED (a table-placement defect); the real-walk replay (r261) is launched

**Date**: 2026-11-27 · **Leg**: r260 (plan §12 rung; the r259
next-rung: the minimal prior-state prefix) · **Image**:
`vllm-arc:v0.30.0-b70-cert29` (cert29_fork_sha=16effb49, the r255
crash image) · **Rung chain**: r255 (3 crashes) → r256 (OFF-branch
delta) → r257 (faithful replay REPRODUCED + minimal clean) → r258a
(static audit) → r258b (content "exonerated" — **retracted this
leg**, see below) → r259 (eager crash: graphs exonerated) →
**r260 (this leg: any prior traffic suffices; V4 = the minimal
repro)**.

## What ran

`tools/ar/r260_bisect_leg.sh` — four fresh-boot EAGER phases, each
running one prior face's FULL class (the actual longprobe v3
semantics: cold → rehit → rehit2 at temp 0 / max_tokens 1536, then
t1a/t1b at temp 1 seed 1234 / 768; byte-identical prompts,
enable_thinking OFF) and then the multi class (the crash site in
every rung), via `tools/ar/r260_bisect_probe.py` (PROMPTS verified
byte-identical to the rig probe):

| variant | prior face | multi cold | multi rehit | verdict |
| --- | --- | --- | --- | --- |
| V1 | essay (2104 tok × 5 gens) | clean (141.7 s) | HTTP 500 | CRASHED |
| V2 | code (5433 tok × 5 gens) | clean (141.1 s) | HTTP 500 | CRASHED |
| V3 | multi (3446 tok × 5 gens) | clean (142.9 s) | HTTP 500 | CRASHED |
| V4 | simple (57 tok × 5 gens) | clean (141.1 s) | HTTP 500 | CRASHED |

`R260_SUMMARY v1_essay=2 v2_code=2 v3_multi=2 v4_simple=2` —
DEVICE_LOST=3 / TRACEBACK=4 in every phase's serve log, ccs engine
resets both GPUs at every crash, **8 devcoredumps banked** (2 per
variant).

## Verdict

**The minimal prior-state prefix is ANY prior traffic — five
57-token chat generations suffice.** The crash class is not a
function of the debris prompt class, its length, or its content: V3
(the same prompt's own debris) and V4 (trivial traffic) both crash.
The crash signature is invariant across all four variants:

- the multi face's **rehit** (temp 0, max_tokens 1536) —
  `num_computed_tokens=3328, num_scheduled_tokens=118`,
  `PrefixCacheStats(queries=3446, hits=3328)`, every rung;
- the full-attn pool: 26 cached prefix blocks + 1 fresh;
- every windowed pool: 16 reissued blocks (the cold's own window
  region — the LIFO reissue, content-correct) + 2 fresh blocks from
  the free list;
- `new_block_ids_to_zero=null` — the fresh windowed blocks are still
  never zeroed.

**V4 is the minimal cheap repro (~12 min)**: boot (6) + five tiny
gens (4) + multi cold (2.4) + the rehit that dies. Every fix-leg
iteration now has a fast, deterministic e2e oracle.

### The V3 subtlety: the rehit geometry itself is not sufficient

V3's class-2 cold is itself a prefix rehit — proven by the sha
divergence: temp-0 colds are byte-identical across boots and faces
(the universal think-spam attractor collapses essay/code/multi colds
to the same sha `5c80063c79a1`), while V3's class-2 cold produced
`886716bff6e5` — it took the prefix-hit path (computed=3328,
scheduled=118), the EXACT crash geometry — **and ran CLEAN**. The
next rehit of the same prompt, minutes later in the same boot,
CRASHED. The only state that varies between them is the free list
the two fresh-tail allocations drew from: after one more generation
of windowed churn, the fresh-tail blocks come from deeper
allocator history. The discriminating state is the PROVENANCE OF
THE FRESH WINDOWED BLOCKS — exactly the never-zeroed class the
windowed-zeroing fix candidate targets.

## The r258b retraction — a table-placement defect vacuated the exoneration

Auditing the r258b harness against the r260 dump rows exposed a
docstring-code disagreement in `make_table(tail=True)`: the
docstring says "36 head pads + the 18 ids at 36..53", but
`col = width - len(ids)` placed the 18 windowed ids at columns
**622..639** of the 640-wide table. The kernel's window walk reads
columns 36..53 (each row's window is `[sl[i]-1023, sl[i]]` → block
columns `floor((sl-1023)/64)..floor((sl-1)/64)` = 36..53 for
`sl` in 3329..3446) — so **every r258b cell read block 0 (zeros at
the walked columns) eighteen times; the real crash ids never
entered the walk**. The 36/36 CLEAN verdict only exonerates the
mis-placed form. The r258b "content classes exonerated" claim is
RETRACTED; the content hypothesis (garbage bytes in the fresh
windowed blocks) is OPEN again — and it is now the leading
candidate, consistent with r260's fresh-tail-provenance finding.

## Next rung (r261, LAUNCHED)

`tools/ar/r261_realwalk_replay.py` — the corrected direct replay:
the dump rows verbatim (36 zero-pads at 0..35, the 18 real ids at
36..53), crossed with FOUR banked crash tables (m8g / r257 / the
r260 V3 + V4 rows, literals verified byte-faithful against the
banks) + the consecutive-fresh control, × four pool fills (zero /
recycled / random / nanband) × 3 seeds = 60 cells. Outcomes:

- **any HANG/DEVLOST** → the offline repro is in hand (the kernel +
  the real walk is the surface); iterate fills/ids offline to pin
  the trigger bytes; the windowed-zeroing fix is then provable on
  the offline oracle first.
- **all 60 clean** → the remaining delta is the engine's same-step
  store/interleave context; r262 = the ARC_MXKV_NATIVE=0 lane
  ablation on the V4 e2e repro.

Then: the fix (leading candidate: extend the fresh-block zeroing to
the windowed pools), proven on the minimal repro → re-leg
m8g/m4g/mkg → re-gate (r255-class) → the cert29 cut.

## Bank

- `r260/v{1..4}_*/` — per phase: boot.txt, markers.txt (eager
  posture, `xpu_graph_lines=0`), probe.out (the R260_GEN ordinal
  lines; the crash is always at `face=multi phase=rehit`),
  longprobe_seq_p47r260N.json/.txt, serve.log, erraudit.txt
  (DEVICE_LOST=3, TRACEBACK=4), req_lines.txt,
  devicelost_context.txt (the full crash dump incl. the block
  tables), dmesg_final.txt (the per-phase ccs resets), SHA256SUMS.
- `r260/devcores/` — 8 devcoredumps (both cards × 4 variants) +
  watch.log.
- `launch.log` — the rig-side driver log (the four boot/probe/
  verdict cycles + the W14 restore).
