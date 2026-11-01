# P6.13 — rung 3: sliced final-span store (feasibility + design + predictions)

**Owner directive (2026-11-01):** "run deep analyze is this possible fix
and if is write plan to fix and start to implementeing start-to-end fix"
— following the END-stall deep analysis
(`results/2026-11-01-p65-end-stall-analysis/ANALYSIS.md`, commit
1890174e).

**Status: GATED (2026-11-01 owner window) — see `VERDICT.md`. Short form:
every safety/regression face GREEN (integrity 5/5 + determinism 3/3, run7n
dead-center 300.48 / 84.41-84.57, U32k/C32k families unchanged, zero
ERROR/WARN-fork-added); mechanism confirmed exactly (63 232-class = 0 all
session, 38 = 2×19 windows on the faces run, U64k TTFT 50.72 / 48.79-cold
vs 52.25 bar); the pre-registered max-gap ≤600 ms bar mechanically MISSED
(1 598) and is documented as mis-set (r48e nolmc floor 2 347 ms — the bar
contradicted banked evidence; corrected claim confirmed). Engine code merged
default-OFF; the enabling profile stays experimental/uncertified.**

## 1. Feasibility verdict: YES — with one honest boundary

- **Fixable fork-side:** the M1/M2 end-boundary cost — the deferred
  one-shot whole-span store sitting ~4.3 s (64k, ~0.24 GB/s contended,
  ~16.5 KB/tok/worker) inside ONE bystander-shared, pre-first-token
  step — can be spread across steps with bounded per-step cost by a
  pure adapter-side change. Total bytes (and roughly total store wall)
  are UNCHANGED; what changes is the concentration. This is a fairness
  fix (max-gap), not a throughput fix — though TTFT should IMPROVE at
  64k because the one-shot currently sits inside the pre-first-token
  step (wfs fires in the kv-connector CM's finally, before
  sample/postprocess — the p43a measured law).
- **NOT fixable here:** M4 (end-as-admission aliasing — the freed
  budget admitting the next long prefill whose chunk-step stall lands
  time-aligned with the END) is scheduler-side; the P6.6 lane already
  rejected the scheduler lever family on evidence (r66a/b/c). The M4
  discriminating capture (empty-queue U64k end-window) remains the
  owner-window item from the END-stall annex. M3/M5 are ms-class —
  untouched by design.

## 2. Why the fix is safe: call-shape equivalence

The slice re-uses EXACTLY the call shape lmcache already serves for
per-chunk rows — no new store semantics:

| | per-chunk row (stock, served today) | sliced window (this fix) |
|---|---|---|
| token_ids | cumulative prefix through chunk end | cumulative prefix through window end |
| mask | True only on the chunk's span | True only on the window's span |
| slot_mapping | cumulative slots (same length) | cumulative slots (same length) |
| offset | first True position | first True position |

The window arithmetic (`_lmc_slice_store_window`, module-level in the
adapter) enforces: contiguous mask tail invariant, non-final edges
aligned DOWN to `_lmcache_chunk_size` (the adapter's own truncation
law, wfs :2550-2556 analogue), progress guaranteed (window ≥ one chunk
even if budget < chunk), final window keeps the capture's own end.

**Correctness laws preserved** (all three, unchanged):
1. **Store lag** — slicing happens inside `_lmc_flush_pending`; bytes
   are still one step stale; the flush's `torch.xpu.synchronize()`
   still precedes every gather.
2. **Mamba-first ordering** — under the deferral profile the boundary
   registry banks ALL boundaries at the final-chunk wfs (one sweep,
   before any attn slice executes); every attn slice is ≥1 step later.
   Non-deferred rows unchanged.
3. **Clean-miss-on-race** — chunks appear atomically per store call; a
   lookup racing the slice train hits only stored chunks and recomputes
   the rest (the r48b law). Mid-drain warm replays therefore pay a
   partial recompute (bounded by unstored bytes) — integrity-gated in
   the predictions below.

Failure semantics: a slice exception drops the remainder (earlier
slices stay cached; later chunks recompute) — identical to today's
single-shot failure law.

## 3. The hunk

`vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py`:
- NEW module-level `_lmc_slice_store_window(...)` (after
  `need_gpu_interim_buffer`).
- `_lmc_flush_pending` attn loop: honors `LMC_STORE_SLICE_TOKENS`
  (default `"0"` = OFF = byte-identical certified behavior — the
  original full-span call runs untouched when the env is unset/0).
  Remainders re-queue in `self._lmc_pending_stores` (cursor rides the
  tuple's offset slot; original token_ids/mask/slot preserved), so the
  next wfs (fires every step, decode included) executes the next
  window. FIFO order per request is preserved (remainders precede new
  captures appended later in the same wfs).

Env profile (extends the certified long-context profile; NONE of this
is default): `--long-prefill-token-threshold 2048` +
`LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` +
**`LMC_STORE_SLICE_TOKENS=4096`** (candidate; the gate also runs 2048).

## 4. Numbers (from banked measurements)

- Bytes/token/worker: 0.51 GB / 31 616 tok ≈ 16.5 KB (r48k mechanism).
- Contended big-store regime ~0.24 GB/s ⇒ 70 ms per 1024 tok.
- Effective window with S=4096, chunk=1664: 3328 tok (aligned down) ≈
  55 MB ≈ ~230 ms + per-flush fixed (sync+submit, the 79 ms/event
  regime's components) ⇒ **~260–310 ms added to each of ~19 consecutive
  decode steps at 64k** (63 232 = 19 × 3328 exactly), instead of ~4.3 s
  in ONE pre-first-token step.
- S=2048: same aligned 3328-tok windows (2048//1664=1 → chunk floor) —
  i.e. at chunk=1664 the two budgets COLLAPSE to the same window size;
  the gate's S=2048 face is therefore a control, not a different
  profile (recorded so the gate cannot "discover" a fake S-sensitivity).

## 5. Pre-registered predictions (gate faces vs r48k/r48l bars)

Posture: cert4-lineage probe image, thr2048+LMC+DEFER+MIN_TOKENS=40960,
6 bystanders, P65_W2MT=4000, greedy, k3, graphs (the r48l posture).

| check | bar (no slice) | predicted (slice 4096) |
|---|---|---|
| U64k TTFT | 51.90 / 52.25 s | **47.5–50.5 s** (one-shot leaves the pre-first-token step; slice-1 ~0.3 s remains) |
| U64k W gaps p50/p95/max | 1287/1585/1612 (r48l) | p95 ≤ 1 100; **max ≤ 600 ms** |
| drain window | jam-then-break | ~19 consecutive steps at base+~300 ms, flat (no super-linear growth) |
| U32k TTFT / W gaps | 22.88 s / 995/1300 | **identical family** (below MIN_TOKENS ⇒ deferral off ⇒ slicing never engages) |
| C32k replay | ~1.2 s | unchanged |
| C64k mid-drain replay (NEW integrity face) | n/a | correct output; TTFT between C32k-class and U64k-class (partial-span recompute bounded by unstored bytes) |
| store histogram (64k face) | 2×63 232-class finals | **63 232-class = 0**; ~38×3 328-class (2 workers × 19) |
| run7n agg/solo | 300.05 / 84.44 | 290–300 / 83–84 (±5 %) |
| engine WARN/ERROR | boot-accepted only | same (no new fork-added lines) |
| integrity battery | 5/5 | **5/5** |

Gate decision rule (pre-registered): ADOPT the profile iff U64k max gap
≤ 600 ms AND TTFT ≤ r48l bar AND run7n ±5 % AND integrity 5/5 AND U32k
family unchanged; else record RED and keep default-OFF.

## 6. Local validation (this session, no rig)

- `slice_window_mirror_test.sh` — bash/awk mirror of the window
  arithmetic (dev box has no python; the rig boot's py_compile is the
  syntax gate, per the r48i law): **ALL PASS** — 64k tiling (19 × 3 328,
  exact coverage, no gaps/overlaps, ≤budget, aligned edges), 32k
  tiling, single-window ≡ un-sliced call shape, skip-leading row,
  cursor resume, budget<chunk progress guarantee, nothing-storeable.
  Raw: `mirror_gate.out`.
- Paren-balance note: whole-file awk counts −1 vs HEAD — the `[f, n)`
  interval notation inside the new docstring (a close-paren with no
  open — false positive; all CODE paren pairs balance: def signature,
  max-block, return-block, both store calls, tuple append).
- In-boot checks for the gate (r48j lesson — name the module path, never
  shared strings): `[…@lmcint] src-override OK` marker + committed-file
  md5 + env-in-engine-process (`/proc/<pid>/environ`:
  `LMC_STORE_SLICE_TOKENS=4096`). Behavioral activation = the store
  histogram, which fully discriminates the three states: slicing LIVE
  (63 232-class = 0, ~38×3 328-class), deferral-live/slicing-dead
  (63 232-class finals present), neither (1 664-class per-chunk stores
  present). No probe logging needed; the repo copy carries none.

## 7. What was NOT done (and why)

- No rig boot / no image build — GPU work is owner-window-only. The
  gate script below is ready; it reuses `/root/build/p65/` harness
  (p65_timeline_v51.py) and the r48l probe-image flow (docker commit,
  the race-free path proven by r48i/r48iv2's failure).
- No default-posture changes: cert4 remains byte-identical by default
  (env unset). No docs currency change (no new certified image until
  the gate passes; MIGRATION row marked pending).
- No M4 scheduler work (out of scope per P6.6 record).
