# r257 — the faithful replay (P6.47)

**VERDICT: REPRODUCED — the faithful OFF-branch suite crashed with the
exact r255 DEVICE_LOST signature, and the minimal pair (same bytes,
pristine state) ran CLEAN. The state delta is isolated: the trigger is
the re-request's block-address walk in the suite-fragmented allocator,
not the prompt bytes. Two devcoredumps captured for the first time
(`Reason: LR job cleanup` — the GuC hang class, BOTH GPUs).**

## What ran

`tools/ar/r257_faithful_replay.sh` (driver, rig-side
`/root/build/r257_faithful_replay.sh`, md5
`8a23fef767fa054d7b71578bbbd303ae`) + `tools/ar/r257_faithful_probe.py`
(probe, rig-side `/root/build/p647/r257_probe.py`, md5
`e5f7f3288083ee93d8639025ed97ebff`) on
`vllm-arc:v0.30.0-b70-cert29` (cert29_fork_sha
`16effb497cd36cb25df7c19cb6f1938d2d45cdc5`). Launched
2026-11-27T02:26Z, complete 02:51Z. `R257_SUMMARY replay=2 faithful=0
fullstate=3 (2=crash 0=clean 1=fail 3=skipped)`.

Phase order note: this supersedes the journal's A-then-B listing —
running the minimal pair first would seed the multi blocks and turn
the suite's cold multi into a prefix-hit, which is NOT the r255
sequence. The suite replayed first, uncontaminated; the minimal pair
ran second on its own fresh boot.

Posture (both boots): `mx_backend_lines=1`, `native_lines=54`,
`pc2_windowed_impl=100`, `lmcache_lines=0`, `tracebacks=0`, KV pool
54,426 tok @1.33x — byte-identical to the r255 m8g crash leg.

## Phase B (replay) — CRASHED, the r255 sequence reproduced

The ACTUAL `/root/build/p647/p647_longprobe.py` full suite (the
identical binary probe that crashed r255; the kwarg
`{"enable_thinking": False}` rides every request — the OFF branch).
essay + code faces complete (their REP-LOOP garble is the known
native-KV baseline, per the r255/r253 classifier law). The multi face
died:

```
[multi] ERROR <HTTPError 500: 'Internal Server Error'>
[simple] ERROR URLError(ConnectionRefusedError(111, 'Connection refused'))
```

`erraudit: DEVICE_LOST=3 TRACEBACK=4`; engine dead at 02:46:18. dmesg:
**ccs (compute) engine resets on BOTH GPUs at 02:45:59** (da:00.0
guc_id=28, b1:00.0 guc_id=22), `Xe device coredump has been created`
on both — and the watcher banked both dumps 19 s later:
`card1_024618.devcore` (403,916 B) + `card2_024618.devcore` (395,108
B) with dmesg snapshots. Dump header: `Reason: LR job cleanup,
guc_id=22`, kernel `6.17.0-10` — **the lost-job/timeout class: a HUNG
compute batch**, not an OOB fault. This matches the 13-18 s
reset-before-python-surface timing in r255: the GuC hang detection
window.

The crashed request's scheduler dump (the full SchedulerOutput is in
`r257/replay/devicelost_context.txt` and `req_lines.txt`):
`prompt_token_ids_len=3446, num_computed_tokens=3328,
num_scheduled_tokens=118` — the exact r255 geometry.

## Phase A (faithful minimal) — CLEAN: the state delta is ISOLATED

Fresh boot, the r255 request byte-exact (UNCUT multi text + the
enable_thinking kwarg), fresh -> A/A resend:

```
CALIB on_branch_tokenize=94 off_branch_engine=47 delta=47
FULL_MULTI_ON_BRANCH_N 3493
RUNG {"face": "multi-uncut-kwarg", "n": 3446, "fresh": {...ok, dt 2.9...},
      "resend": {...ok, dt 1.7...}, "alive": true}
```

The RUNG's engine count = **3446 exactly** — the OFF-branch branch
delta is confirmed (+47 constant) and the replay is byte-faithful.
`erraudit: DEVICE_LOST=0 TRACEBACK=0`. **Same bytes, pristine state =
clean. Bytes + suite state = crash. The prompt bytes are exonerated;
the allocator/block-layout state is the trigger.**

## The crash-request block table (banked in the SchedulerOutput dump)

The dumped `block_ids` exposes the 6-group hybrid KV structure for
the first time:

- **5 windowed rows** (page-64: 54 entries for 3,446 tok): each =
  **36 zero-pads + 18 real ids**. The zero-pads = the pre-window
  region (36x64 = 2,304 tok; window = 18x64 = 1,152 tok). The real
  ids are **heavily fragmented recycles** from the essay/code faces
  (row 1: 62, 49, 134, 109, 21, 82, 184, 107, 46, 100, 132, 71, 19,
  105, 153, 52, 177, 231; rows 2-4 carry consecutive runs 180..183,
  201..212, 267..282, 338..353 with outliers — the free-list order
  after the suite's alloc/free churn).
- **1 full-attention row** (page-128: 27 entries = ceil(3446/128)):
  all 27 real, all scattered (483, 710, 403, 416, 469, ... 217).
- `num_common_prefix_blocks=[0, 0, 0, 0, 0, 27]` — **the prefix cache
  covers ONLY the full-attn group**; the windowed groups take fresh
  allocations on every re-request (0 common blocks), and
  `new_block_ids_to_zero=null` — the fresh windowed blocks are NOT
  zeroed.

The hang therefore walks the windowed page-64 twin over 18
freshly-allocated, never-zeroed, fragmented blocks — geometry no
pristine-state request ever produces (r256's minimal pairs and this
leg's phase A allocated from clean free lists).

## The crash ordinal drifted — and that strengthens address-dependence

- r255 m8g crashed at the multi **t0 rehit** (dumped params:
  `temperature=0.0, top_p=1.0, top_k=0, seed=None` — greedy sample 2).
- r257 crashed at a multi **temp1 sample** (`temperature=1.0, top_p=0.95,
  top_k=64, seed=1234, max_tokens=768`) — the t0 triple SURVIVED here.

Same request state (3446/3328/118), different re-request ordinal.
The trigger is not a fixed sample sequence — it is the block-address
walk each re-request lands on, which drifts as the face's own
alloc/free churn progresses. r255's layout hung at the first
re-request; r257's at the third. The faithful minimal's pristine
layout never hangs. This is consistent with a value-dependent
address/index path in the windowed twin (the page-64 split/plan
arithmetic is the candidate surface — the p204 forced-plan and
engine-page-64 laws both point at `mx_get_num_splits`/plan-len walks).

## What the r256 negative is now fully explained by

ON-branch bytes (the +47 head shift) changed the tail-window CONTENT,
and every r256 request ran on pristine-state boots (minimal pairs +
ladder rungs with stable prefix hits) — neither the faithful bytes nor
the fragmented layout was ever present. r257 supplied both and the
crash returned.

## Next rung (r258)

1. **Decode the devcoredumps** (the hung batch state: guc_id, ring,
   engine registers) — first-ever captures of this defect.
2. **Bank the clean twin's block table** — `dump_input.py` fires only
   on death, so the faithful phase's table was never surfaced; capture
   it with a request-state dump at INFO (or a debug serve) for the
   pristine-layout differential.
3. **The kernel-path walk**: the windowed page-64 twin's
   block-table/address consumption in the P-C2 native impl (build34
   wheel) under a fragmented table — the split/plan arithmetic
   (`mx_get_num_splits`, plan-len) is the candidate hang surface.
4. Then: the fix, re-leg m8g/m4g/mkg, re-gate (r255-class), and the
   cert29 cut sequence.
