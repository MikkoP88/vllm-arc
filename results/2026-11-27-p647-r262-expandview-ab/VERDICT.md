# P6.47 r262 — the e2e A/B: the expand-view block-table stride defect IS the gemma DEVICE_LOST, and the bridge guard IS the fix (a_probe=0 c_suite=0 b_ctrl=2)

**Date**: 2026-11-27 · **Leg**: r262 (plan §12 rung; the r261
next-rung: the e2e A/B of the fix) · **Image**:
`vllm-arc:v0.30.0-b70-cert29` (cert29_fork_sha=16effb49, the r255
crash image) with an `mx_kv` overlay mounted by the wrapper's
`ARC_MXKV_OVERLAY` hook · **Rung chain**: r255 (3 crashes) → r256
(OFF-branch delta) → r257 (faithful replay reproduced; minimal clean)
→ r258a/b → r259 (eager crash: graphs exonerated) → r260 (face
bisect) → r261 (real walk clean offline; static audit) → **r262
(this leg: the defect and the fix are e2e-proven; the fix is landed
in the tree)**.

## What ran

`tools/ar/r262_expandview_leg.sh` (driver) with the overlays built by
`tools/ar/r262_patch.py` from the image's own bytes, three modes:

- **probe** — the fix plus a one-shot dump of exactly the rows the
  wheel dereferences on the synth path (once per block table and
  pool);
- **on** — the fix only (the landing candidate: the bridge
  materializes a non-contiguous `bt_rows`);
- **off** — a boot marker only (the control; `engine.py` byte-identical
  to the image's).

Same image, same posture (gemma-4-31b, mxfp8 KV, `ARC_MXKV_NATIVE=1`,
`ARC_GRAPH=0` eager, spec off, LMCache off), same probe
(`r260_bisect_probe.py`, the longprobe v3 class semantics: cold →
rehit → rehit2 at temp 0 / 1536 tokens, then t1a/t1b at temp 1 /
seed 1234 / 768; byte-identical prompts within a face), one fresh
boot per phase:

| phase | overlay | faces | generations | result |
| --- | --- | --- | --- | --- |
| a | probe | multi | 5/5 | clean |
| c | on | essay code multi simple multi | 25/25 | clean |
| b | off | multi | 2/5 | CRASH at the first multi rehit |

`R262_SUMMARY a_probe=0 c_suite=0 b_ctrl=2` (0 = clean, 2 = crash).

Every boot passed the same gates before its faces ran
(`launch.log`): overlay byte identity — the sha256 of the host overlay
equals the sha256 of the files inside the container
(`R262_OVERLAY_MOUNTED mode=probe` / `on` / `off`); the runtime
marker `MXKV native bridge r262 overlay: mode=<mode>`
(`bridge_marker_lines=2`, `bridge_mode_lines=2`); the posture
(`graph_capture_lines=0`, `eager_posture_lines=3`,
`native_lines=6`, `lmcache_lines=0`, `tracebacks=0`).

## Verdict

**The expand-view stride defect is the cause of the gemma MX
DEVICE_LOST, and the bridge guard is the fix.** The predicted triple
landed exactly:

- with the fix ON the multi rehit completes every time it is
  attempted: four 118-row rides in phase a (rehit, rehit2, t1a, t1b)
  and nine in phase c (the multi class twice; by the probe's
  construction every generation after a face's first is the same
  118-row continuation) — 30 generations across both phases,
  `DEVICE_LOST=0`, `TRACEBACK=0`, and **not one kernel message between
  container start and phase b's crash** (no engine reset, no
  coredump, across both fixed boots);
- with the fix OFF the same boot-to-rehit sequence that crashed in
  r255 through r260 crashed again: the cold completes, the rehit
  returns HTTP 500 at its first step (the 118-row synth forward,
  `native call rows=118 window=1024` on both ranks at 09:49:02, the
  runner error at 09:49:03), ccs engine resets on BOTH cards, two
  devcoredumps (`devcores/`).

The only delta between phases c and b is the bridge's
`.contiguous()` normalization (the `off` overlay differs from `on` by
that block and its marker string — `r262/overlay_on/` vs
`r262/overlay_off/`); so mounting an overlay, adding a python guard,
or the probe itself does not defuse the crash — the stride
normalization does.

The fix is landed in the tree
(`vllm_gaudi/xpu/patches/mx_kv/native_decode.py`, with a contract
comment at the synth call site in `engine.py`); its bytes are
semantically the `on` overlay. What the A/B validated is the overlay
on the crash image; the landed tree bytes get their own validation on
the rebuilt image (next rung).

## The probe dump — what the wheel was dereferencing (phase a)

The probe reads the same bytes the wheel addresses — row r at
`base + r * bt_width` — through `as_strided`, once per (block table,
pool) at the first synth ride of the boot (the multi rehit). Every
line is in `r262/a_probe_fixon/btprobe.txt`.

- **100 calls = 50 windowed layers x 2 TP ranks.** gemma-4-31b has 60
  text layers: 50 sliding-window and 10 full-attention (5:1); the 50
  windowed layers sit in five windowed KV-cache groups (five block
  tables, 64 rows x 640 columns each, 163,840 bytes) and share ten
  pool tensors per rank. The dump holds 5 tables x 10 pools = 50
  classes per rank, 50 native calls per synth forward per rank.
- **The call line** (identical for all 100):
  `MXKVBTPROBE call q=118 width=640 row_stride=640 table_rows=64
  storage_bytes=163840 row0_storage_off=0 pool_blocks=2240
  block_size=64 window=1024 seq_len=3446`. `row_stride=640` is the
  pitch of the table row the probe measured — the pitch the wheel
  assumes; the expand view handed to the wheel has stride 0.
- **Row 0 — the request's real row**: `sl=3329 cols=36..52
  ids=[594, 593, ..., 579, 485] min=485 max=594 nonzero=17 oob=0`.
  The windowed tables are position-aligned, with the out-of-window
  columns holding block 0: 36 zeros at columns 0..35, the 16 window
  blocks at 36..51, and the fresh tail from column 52. The scheduler
  dump of the crash request (`b_ctrl_fixoff/req_lines.txt`) shows the
  same shape — 36 zeros + 18 ids per windowed row (the two fresh tail
  blocks are 485 and 396); the full-attention group (page 128) holds
  26 cached blocks + 1 fresh, which is why the prefix hit stops at
  3328 = 26 x 128 and the tail is 118 tokens.
- **Rows 1..63 — the table's following rows**: all-zero ids
  (`nonzero=0`), 17 columns each, in all 100 calls.
  As block ids that is the null block: windowed attention silently
  OFF on those rows.
- **Row 64 — past the table**: `BEYOND past_end_bytes=0
  READ_FAILED RuntimeError('setStorage: ... out of bounds for storage
  of size 163840')`, 100 times. torch's storage-bounds check refused
  the read, so **the bytes past the table are still UNOBSERVED** — a
  raw read needs a different trick (and may itself fault; see open
  items).
- **The "slot mapping adjacent to the table" guess is refuted**:
  `smap_adjacent=False` in every class; the five slot-mapping buffers
  sit at other addresses (the address map below).

## Mechanics of the defect

The engine's continuation paths by query length (`engine.py`):
`q == seq_len` is the first-chunk flash path; `1 < q <= 8` is the
split-row dispatch (per-row gathered tables — contiguous); `q <= 128`
is the synth path (`bt_row.expand(q_len, -1)` — stride (0, 1));
anything larger is the bulk prefill. A pure-decode step takes the
decode path with the real table (`attn_metadata.block_table`,
contiguous by construction); a one-row continuation in a mixed batch
takes the synth path with a size-1 view, contiguous by definition.
Decode never misreads and never copies.

The defect lives in the synth window `9 <= q <= 128` on
sliding-window layers:

| synth rows | what the wheel reads for rows 1..q-1 | effect |
| --- | --- | --- |
| q <= 64 | the table's zeroed rows (block 0) | silent: windowed attention off |
| 65 <= q <= 128 | rows 64..q-1: device memory past the table, as block ids | out-of-bounds walk; hang or garbage |

It re-reads every rung of the chain without exception:

- colds are one full forward (the flash path) → always clean, in
  every rung;
- the essay and code rehits are 56- and 57-row rides (2104 = 16 x 128
  + 56; 5433 = 42 x 128 + 57) → inside the table → always clean;
- the multi rehit is a 118-row ride (3446 = 26 x 128 + 118) → 54 rows
  past the table → the crash site in r255, the r257 suite, r259, all
  four r260 variants and this leg's phase b;
- the r257 minimal pair survived the SAME 118-row ride (its serve
  log carries the `rows=118` native call on both ranks) — so a ride
  past the table is not fatal by itself: it depends on the bytes it
  lands in;
- r261's 60 materialized tables were all contiguous by construction →
  always clean, which is exactly why the offline oracle could not
  reproduce.

The lane is young: the expand view dates from the original MX KV
lane (#111, 2026-11-13) where it rode the stride-aware Triton lane,
and the native windowed synth routing arrived with the P-C2 windowed
support (#148, 2026-11-26). The certified release cert28 predates it
and is unaffected; the cert29 gate was the first gate to run the
lane, and it caught it.

**Address map (TP0, phase a)** — tables at `...50a00000`,
`...50a28000`, `...52764000`, `...5278c000`, `...527b4000`; the five
slot-mapping buffers (393,216 bytes each) at `...50a50000` through
`...50a90000`. A 118-row ride over-reads 54 rows = 138,240 bytes past
each table. Past three of the five tables lies the NEXT table (its
rows 0..53: in-range ids and zeros); past the second table lies the
run of slot-mapping buffers; past the fifth, an unprobed region. A
slot id is 64x its block id (block 485 → slot 31,040), so slot ids
read as int32 block ids point thousands of blocks past a 2,240-block
pool (gigabytes past a 605 MB pool). Whether such an address is
mapped depends on the ids live at the time and on the heap layout —
a candidate for why the same ride survives on a fresh boot (r257)
and dies in suite state. **An inference from the map, not an
observation.**

## Output hashes — what the fix changes and what it does not

Per ride, unfixed (the r260 bank) against fixed (phases a and c):

| ride | unfixed | fixed |
| --- | --- | --- |
| essay / code: cold, rehit, rehit2, t1a, t1b | 5c80063c79a1 / fc2dddcc7ce6 / fc2dddcc7ce6 / 74616f540de4 / 74616f540de4 | identical |
| simple: rehit, rehit2, t1a, t1b | 124ef8c98c7c / 124ef8c98c7c / 4f6c814ca19a / 4f6c814ca19a | identical |
| multi cold, not first after boot | 5c80063c79a1 | 5c80063c79a1 |
| multi rehit, rehit2, t1a, t1b | CRASH | fc2dddcc7ce6 / fc2dddcc7ce6 / 74616f540de4 / 74616f540de4 |
| multi cold, first after boot | 886716bff6e5 (r260 V3, phase b) | 07b7b69d1e6c (phase a) |
| simple cold, first after boot | ccb04f8dcaf8 (r260 V4) | not run fixed |
| multi "cold" on the repeat face (itself a prefix rehit) | not run unfixed | fc2dddcc7ce6 (phase c) |

The last row is the ninth phase-c ride: the repeat multi face repeats
the same prompt, so its "cold" is a 118-row rehit and lands in the
rehit sha class. Phase a holds four rides, phase c nine.

- **The r261 prediction "rehit outputs are expected to change once
  the fix lands" is RETRACTED.** Fifteen comparable ride classes are
  byte-identical across the fix. On these faces the temp-0
  continuation is prompt-independent: the cold sha and the rehit sha
  are each the same for the essay, code and multi prompts (three
  different prompts), the attractor class the cert batteries already
  bank. The sha classes therefore cannot see whether windowed
  attention was live on the continuation rows. A sha-stability gate
  on these faces is blind to this defect class; only the crash
  exposed it. A gate face that does not collapse to the attractor is
  a follow-up candidate.
- **The first-after-boot cold shas remain an open curiosity, not a
  fix effect.** The multi-first cold gave `886716bff6e5` on both
  unfixed boots (r260 V3, phase b) and `07b7b69d1e6c` on the
  fixed+probe boot, while the essay-first cold gives the common
  class `5c80063c79a1` on both (r260 V1 unfixed, phase c fixed). A
  cold is one flash-path forward followed by decode-path steps, so
  no changed line executes in it. A fixed-overlay multi-first
  control without the probe would separate "fix" from "that boot";
  not run.

## Corrections to earlier banks

- **r261, "the five windowed layers"**: there are fifty windowed
  layers in five windowed KV-cache groups (the probe's 50 classes per
  rank). The "five" is the number of block tables, not layers. The
  r257/r258b/r259 wording "five windowed pools/rows" means the five
  per-group tables.
- **The `xpu_graph_lines=0` marker is vacuous** in the r259 and r260
  banks: the helper took `-i` as its pattern and never ran the regex.
  The eager posture of those legs is nonetheless proven by their
  serve logs (0 capture lines, 3 x `graph posture cudagraph_mode=NONE`
  and 4 x `VLLM_XPU_ENABLE_XPU_GRAPH=False` per serve log). This
  leg's driver forwards every grep argument and uses two
  discriminating markers instead: `graph_capture_lines` (must be 0)
  and `eager_posture_lines` (must be >= 1).

## The W14 restore hang — root-caused (operational)

The leg's own restore hung twice (`W14_BOOT_HANG` 09:58:31Z,
`W14_RESTORE_FAIL` 10:07:44Z): the serve boots to "JIT kernel warmup
finished" and then waits forever on `No available shared memory
broadcast block found in 60 seconds`, port refused, both GPUs idle.
The r261-era explanation ("a container restart is the remedy") is
superseded — a restart alone reproduces it.

`py-spy dump` on the host pids (`w14_hang/pyspy_before.txt`) shows
both workers in `wait (torch/utils/file_baton.py:50)` under
`_jit_compile (torch/utils/cpp_extension.py)` ← `_host_view_ext
(vllm_gaudi/models/qwen4_exp/xpu/expert_offload.py:194)` ← the first
forward. The W14 image JIT-builds its host-view extension into a
PER-PID directory `/tmp/iexp_host_view_<pid>` (~27 s build). A boot
killed inside that window leaves a 0-byte `lock` behind; a fresh
container PID namespace is deterministic (Worker_TP0=561, TP1=562),
so every later clean boot finds the baton held and waits forever.
The poison carries the timestamps 2026-11-26 17:58:27Z — a boot that
died inside its build window, which the leg logs place at a leg
entry's `docker stop` of a restore boot in its first minutes; the
07:50Z "successful" restart only worked because diagnostic
`docker exec`s had bumped the pid counter (581/582).

**Remedy** (`tools/ar/r262_w14_restore.sh`, verified
`W14_HEALTH_OK 2026-11-27T12:54:41Z`, `W14_GEN_OK`): restart →
delete every `/tmp/iexp_host_view_*` directory that has no
`iexp_host_view_ext.so` (`W14_STALE_BATON_DIRS_REMOVED=2`) →
GPU-free wait → serve → health + one real generation. The restore
boot got pids 569/570 and built fresh (`host_view_dirs_after.txt`).

## Open items

1. **The bytes past the table** — never read (torch refuses the
   storage overrun). A raw read (enlarging the storage view on a
   sacrificial boot) would also show whether the fifth table's
   over-read region is mapped. Diagnostic only; the fix does not
   depend on it.
2. **Graph posture** — the fix is proven in EAGER posture only. The
   synth path is the same python in graph mode (it runs outside the
   captured graph), and the copy is a device op (capture-safe), but
   that is an argument, not a leg: m8g / m4g / mkg are re-legged on
   the rebuilt image.
3. **The first-after-boot cold shas** — see above.
4. **Hardening (later)**: teach the wheel `block_table.stride(0)` so
   the bridge need not copy; a non-collapsing gate face for the gemma
   rehit class.

## Next rung

Rebuild the image from `main` (new fork_sha) → re-leg m8g / m4g / mkg
on it → re-gate (the r255-class gate that crashed) → the cert29 cut.

## Bank

- `launch.log` — the driver log: overlay identities, per-boot gate
  lines, every generation, the phase summary, the W14 restore lines.
- `r262/a_probe_fixon/`, `r262/c_suite_fixon/`,
  `r262/b_ctrl_fixoff/` — per phase: `serve.log`, `probe.out`,
  `gen_summary.txt`, `erraudit.txt`, `markers.txt`,
  `overlay_identity.txt`, `ov_sha.txt`, `ctr_sha.txt`, `boot.txt`,
  `btprobe.txt`, `req_lines.txt`, `devicelost_context.txt`,
  `dmesg_final.txt`, `longprobe_seq_p47r262{a,b,c}.json` / `.txt`.
- `r262/devcores/` — the phase-b devcoredumps of both cards
  (`card1_094903`, `card2_094903`) with their kernel-log excerpts and
  the watcher log.
- `r262/overlay_probe/`, `r262/overlay_on/`, `r262/overlay_off/` —
  the mounted overlays as run (`engine.py`, `native_decode.py`).
- `r262/r262_patch.py`, `r262/r262_expandview_leg.sh`,
  `r262/r260_bisect_probe.py` — the tools as run; `r262/SHA256SUMS`
  (rig-side).
- `w14_hang/` — the stale-baton evidence: `pyspy_before.txt`,
  `host_view_dirs_before.txt`, `host_view_dirs_after.txt`,
  `cleanup.txt`, `pids_after.txt`, `gen.txt`, `restore.log`,
  `serve_posture_before.log`.
- `SHA256SUMS` — covers every banked file except itself and this
  verdict.
