# VERDICT — r268c trace leg (THE DEFECT IS NAMED: the global pool pages at 128, the decode walks it at 64)

Banked 2026-11-28 from rig bank `/root/build/p647/r268c_trace` after
`R268C_DONE rc=0 10:14:32Z` + `W14_RESTORE rc=0 10:19:06Z` (transfer
oracle 12/12 byte-match; address audit clean after byte-exact
redaction; the 29.8 MB serve.log banked gzipped — `trace_disc.txt`,
the MXKVTRACE-only subset, is derivable via `grep MXKVTRACE` and not
banked). Boot: cert29, gemma lawful g3 skin, `kv_cache_dtype=mxfp8` +
native windowed lane, `ARC_GRAPH=0` (eager — the e2/c3 posture that
reproduces), `ARC_MXKV_OVERLAY=/root/build/r268c_overlay` (engine.py
md5 `76e3bb12` — the r268 overlay with `_TRACE` **hardcoded True**,
the fix for the ARC_MXKV_TRACE dead letter: serve_arc_b70.sh's
curated export list never passes the var into the container).

## The leg

The disc set (`p268c_disc.py`, md5 `64726f9d`) reproduced the r268b
pattern on a traced boot — **DISC_RED=7/8**: T0 control (ptok 6,
total 14) CLEAN; T1 plain raw 122+16 RED (400-nan); T2 gate chat
40+64 RED finite-wrong (`'Paris<end_of_turn>\n<start_of_turn>user…'`);
T3 card chat 33+96 RED (400); T4 the probe's g4 short templated raw
20+96 RED (400 — kills the 128-crossing law: 116 total never crosses
128, but crosses 64); T5 the probe's g3 templated raw 40+96 RED (400);
T6 the 2-way burst both RED. **The trace fired: 145,680 lines
(48,480 store + 46,920 decx)** — per-layer store slot maps and decode
block-table rows, joined by the per-module `id=`.

## THE DEFECT (the G3 metadata value, named)

**The gemma GLOBAL-attention pool (the 10 full-attention layers,
D=512, Hk=2/rank — cache shape `(2240, 128, 2, 1056)`) pages at
128 — while the engine's five `mx_decode` call sites omit
`block_size`, so the Triton decode walks the 128-paged glob table
with `BLOCK_SIZE=64`.**

Proven by three independent trace facts:

1. **The pool shapes**: slide pool `(2240, 64, 8, 528)` — page 64;
   glob pool `(2240, 128, 2, 1056)` — page **128**. (stride
   `(270336, 2112, 1056, 1)` — contiguous; `2240×128×2×1056`.)
2. **The block-table widths**: slide bt stride **640 = 40960/64**;
   glob bt stride **320 = 40960/128** — the glob table is BUILT
   per-128-pages. The glob store resolves its page correctly
   (`kv_cache.shape[1]`, engine.py:749 — stores land right; r267's
   store-audit bit-exact is consistent).
3. **The walk**: the wrapper default 64 (the five audited call sites:
   engine.py 354/823/995/1239/1546 + the 1528/1537 rows rides) feeds
   `BLOCK_SIZE` into the kernel's `bt[b, offs_n // BLOCK_SIZE] +
   offs_n % BLOCK_SIZE` pitch.

The consequence, exactly matching every observed signature:

- tokens 0-63: `row[0]` pos 0-63 — **correct** (both page semantics
  agree below 64 when the slot base is 128-aligned).
- tokens 64-127: reads `row[1]` — for any request with total ≤ 128
  the glob table has only ONE valid entry, so `row[1]` is **past the
  valid prefix**: stale-written → finite-wrong (the `'our'` class /
  T2's fabricated turns); never-written → non-finite (the 400-nan
  class; the r267 uninit census).
- tokens 128+: reads `row[2+]` — e.g. T1 (total 138): the glob table
  is `[620, 642]` (tokens 0-127 in 620, 128-137 in 642); the 64-walk
  reads tokens 64-127 from block **642 pos 0-63** — only 10 of those
  64 positions were ever written → non-finite → 400.
- **qwen is immune by construction**: uniform single-pool geometry
  (one head size, page 64) — the default-64 walk is exact → the
  certified-clean history. **Gemma is the only hybrid** (slide p64 +
  glob p128) — every MX gemma lane walks the glob pool wrong.

## What this corrects

- The r266 batch-2 note "both pools page 64 — D-B exonerated" was
  **wrong**: the banked serve args' `block_size: 64` is the ENGINE
  CONFIG (the 64-token slot space), not the glob pool tensor's page.
  The original p128 suspicion was right for the glob pool.
- The r267 canary's **pagewidth face already reproduced this exact
  defect** (synthetic p128 D=512: mxfp8 + mxk8v4 default-64 walk
  corrupts past ln=64; pagefix holds) and was demoted to
  "latent-hardening" on that false premise — it was the root-cause
  reproduction all along. The canary's other faces ran BOTH pools at
  p64, so they could not catch it.
- The r268/r268b request-conditionality is explained: every probe
  request runs max_tokens 64-96 over a 20-92-token prompt (totals
  84-138 > 64 — always past the break); the r268 clean workload was
  three requests of total ≤ 26 (always below it); the A-ladder's
  short crossings (totals 68-118) are **not provably clean** — 4-16
  positions of stale-garbage KV under a robust repetition attractor
  is finite-wrong below the ladder's detectors (nan / `' our our'` /
  http) — consistent with the one 67-token "clean" card chat (its
  40 crossing positions read stale-written rows). The exact
  per-request crossing algebra (slot-base alignment in the 128-space)
  is runtime-state — matching r268b's runtime-state finding.

## Consequence

**r269 — the fix-gate A/B — launched 10:30:44Z** (leg `f3_fixm8g_29`,
graphs ON = the production posture, c3 mxfp8+native posture +
`ARC_MXKV_OVERLAY=/root/build/r269_fix`): the WRAPPER-LEVEL class fix
— all three decode entry points (`mx_decode`/`mx_decode_rows`/
`mx_decode_rows_dot`) take `block_size: None` and resolve it to the
cache's OWN page at body start (glob → 128, slide → 64), closing
every current and future engine call site (overlay md5s: engine
`d1fa3c17` = the image's own, `triton_mx_decode` `e33aa7b7` with the
fix at lines 1404/1427-1432 + 1501/1515-1520). The leg runs the FULL
p266 matrix probe (the 323/324-RED set) + the r264 g1 battery: leg
GREEN = the fix turns gemma MX-KV clean → land the wrapper fix
in-tree → m8g/m4g/mkg re-gates WITH logprobs → full battery → cert29
per Law 11. Leg RED with the same signature = residual defect →
the trace leg re-runs under the fix overlay.

Analysis artifacts banked: `analysis_windows.txt` (per-step window
summary), `analysis_evolution.txt` (per-layer bt-row evolution),
`analysis_join.txt` (the store-vs-walk join whose 64-division
"violations" isolate the glob group and, recomputed at 128, prove the
stores/rows consistent — the mismatch is the WALK's page),
`analysis_focus_t1/t3.txt` (the per-layer focus with pool shapes and
strides).
