# P6.5 annex — why the END of a long-context request stalls bystander streams

**Question (owner, 2026-11-01):** "deep analyze why ending of prompt
request answer stall all other concurrent streams on long context — I
think start when prefill hits maybe can stall, but why also end stalls?"

**Scope:** analysis only, from banked evidence (r48a–r48l, r66 series,
r43a/b probes) + code walk of the live connector/scheduler paths. No rig
work; the one confirming capture this analysis proposes is listed as an
owner-window item at the end. Parent lane: P6.5
(`results/2026-10-31-p65-r48a-stall-capture/VERDICT.md`, corrected
matrix after the r48f harness reversal).

## 0. The frame: bystander ITT = step time, at every lifecycle boundary

In vLLM v1 every request's lifecycle work rides shared engine steps. One
step = one scheduler iteration = one forward pass; any cost inside a step
is paid by EVERY stream scheduled in that step. r48a measured the law
exactly: an 8192-token prefill chunk at ~2036 tok/s prefill makes the
step ~4.0–4.3 s, and all co-scheduled bystanders show exactly that as
their inter-token gap (4 gaps 3.64/3.70/3.99/4.30 s = 4 chunks of
31 842 tok). The scheduler packs decodes + chunk prefills into the SAME
step (scheduler_v030.py:581-665, budget :661-680).

The same law governs the END — with one asymmetry the code walk makes
explicit: **the engine has no "quiet teardown."** Every end-event
(final-span store, deepest mamba boundary, block frees) executes either
(a) inside the first steps AFTER the request's last row — steps that now
contain only bystanders — or (b) on the CPU in output processing, which
gates what the NEXT step admits.

## 1. First, the counter-result: at 32k on the certified posture the END is measurably CLEAN

r48a, certified posture (thr0, k3, graphs, 25-var chain), U32k fresh
face, bystander table around the long stream's completion:

- **W_F1_end (finish + store): p95 59.5 ms, max 302 ms** vs baseline
  decode p50/p95 = 53.1/54.3 ms. Cadence "clean 53 ms through the long
  stream's decode AND its END."
- Candidate **[iii] request-end store flush: dismissed at that scale.**
  The naive mirror-theory ("the engine must flush the whole context's KV
  when the request ends, like stock lmcache 0.5.6's synchronous finish
  flush") is FALSE on this fork, for two designed reasons:
  1. Stores are **incremental + lagged**, not end-concentrated: each
     prefill chunk's aligned span stores as it completes
     (`num_saved_tokens` advances per chunk — adapter
     `from_request_tracker`, lmcint:342-407), decode KV stores per decode
     row (`save_decode_cache`), and every store executes ONE step later
     inside `_lmc_flush_pending` (lmcint:2619-2691) after a full
     `torch.xpu.synchronize()` (:2635). At end there is simply nothing
     big left to flush.
  2. The connector's `request_finished` does no store work at all
     (lmcint:3177-3196 returns `False` — no delayed free either), and
     block freeing is CPU-side (`_free_request`,
     scheduler_v030.py:2561-2588), overlapped with the in-flight GPU
     step under async scheduling (:2603-2610 deferred frees).

So the owner's observation is real but scale/posture-dependent: the
end-stall appears on LONGER contexts (≥64k), on the thr2048+LMC
postures, and/or in workloads with queued follow-on requests. Five
mechanisms, evidence-graded:

## 2. M1 — the end-of-prefill boundary carries the store tail (lag = +1 step, bystander-only execution)

`wait_for_save` fires INSIDE every step (kv-connector CM's finally, the
p43a measured law, lmcint:2389-2395). A store captured at step k executes
at step k+1. The **final prefill chunk** (`is_last_prefill`) stores the
final span — and its snapshot executes in the **first step after the
prefill row** = the first step of the answer's own streaming, shared
with bystanders.

- Per-event raw cost: 20.5 ms p50 (1664-tok attn span) / 10.0 ms (832-tok
  mamba) — but **~79 ms fixed overhead per event under concurrency**
  (lag-flush device sync + from_gpu gather + submit per scheduler step);
  r48b micro-cost closure: the freeze leg's 312 events cost ≈4.4 s raw
  yet +24.6 s wall — the overhead, not the bytes, is the stall.
- **One-shot whole-span regime** (the p65 deferral profile, cert4 opt-in
  `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960`): the final
  chunk stores the ENTIRE context in one operation. Measured: 31.6k tok
  ≈ 0.51 GB at ~0.24 GB/s contended ≈ 2+ s concentrated in that single
  first-decode step (the r48k 32k-face +4.2 s TTFT regression mechanism;
  at 64k the finals are 2×63 232-token stores).

Measured end-of-prefill bystander gaps with 6 live bystanders
(W2MT=4000): no-defer 1238/1969/2242 ms p50/p95/max (r48iv2); ungated
defer 1224/1598/2008 (r48k); gated defer 1287/1585/**1612** (r48l). So
on ≥64k faces or the deferral profile, the boundary at the START of the
answer carries the single largest gap of the request's whole life.

## 3. M2 — the store backlog drains THROUGH the answer ("hangover")

r48c's finding: chunked KV enqueues faster than the save pipeline drains
(**4992 tok/s** both workers); bystander gaps GROW linearly during
prefill and **the jam breaks exactly when the last chunk stores
(~+96 s)**. Corrected 64k matrix: thr2048+LMC 76.89 s with bystander
p50/p95/max = 3723/5607/5608 ms vs thr2048+nolmc 52.27 s (1111/2027) —
the +24.6 s is the store path ([vii-a], a 64k×concurrency phenomenon;
at 32k it is zero: r48h LMC 22.78 s ≈ nolmc 22.39 s, gaps 729/1291 vs
705/1262).

To a bystander this reads exactly as the owner describes: the stall
starts at prefill, keeps worsening, and only releases around the long
request's boundary events — so "it also stalls at the end."

## 4. M3 — the deepest mamba boundary banks AFTER the request's last row

p43a measured law (in-code, lmcint:2389-2411): wfs fires before
sample/postprocess, so the LAST 512-token boundary (including ones
crossed by the final generated tokens) materializes at postprocess —
after the request's final wfs row. The re-serve sweep stores it at the
NEXT wfs: **the first step after the request finished, a bystander-only
step**, using the stashed token witness. Cost ≈9 ms per boundary at
4.2 GB/s (r48k's 220×832-class stores ≈ free next to 51.90 s walls).
Structurally an end-event inside a shared step; benign per event.

## 5. M4 — END is an admission event: the next long prefill starts the moment the old one frees (end→start aliasing)

When the request finishes, output processing frees its hundreds of
blocks (a 64k context's whole table) — exactly the budget the head of
the WAITING queue needs. The next `schedule()` admits it and packs its
first 8192-token chunk into the next step → bystanders take a fresh
~4 s-class gap **time-aligned to the previous request's END**. Timeline
correlation says "the end caused it"; the mechanism is the next START
riding the end's admission. On the certified ≤32k posture — where the
end's own work is ≤302 ms (W_F1_end) — this aliasing is the DOMINANT
perceived end-stall. The owner's client patterns make it prominent:
r48g's wave-2 drained near-solo (+89 s) precisely because faces admitted
after the long one; a sequential client (next request fires on
completion) makes every end coincide with a fresh prefill by
construction.

## 6. M5 — CPU-side teardown scaling (minor)

`finish_requests` (scheduler_v030.py:2498) → encoder-cache free →
`kv_cache_manager.free(request)` (block-table + prefix-cache hash-queue
updates, O(context/block_size)) → deferred-free fencing under async
scheduling; plus pin cleanup classes (the soak's 98 LMCache pin_monitor
warnings are this family — self-resolving, serve-invisible, r70 close-out
audit). Bounded at 32k by W_F1_end max 302 ms; grows linearly with
context; never measured dominant.

## 7. Synthesis — answer to "why also the end stalls"

The START stall is chunk-step compute ([ii]/[viii]: bystander ITT = step
time). The END stall is NOT its mirror — the engine does no long compute
at end. What the owner sees at long context is the superposition of:

1. **M1** — the store tail (final-span/one-shot store +1 step) landing in
   the answer's first, bystander-shared step: 1.6–2.2 s max-gap class at
   ≥64k or on the deferral profile; ~0 at 32k default.
2. **M2** — the prefill's own store backlog still draining through early
   decode, releasing only at/after the last-chunk store (+24.6 s class
   at 64k×12 bys; +6.2 s at 6 bys; 0 at 32k).
3. **M4** — the freed budget immediately admitting the next queued long
   prefill: seconds of fresh chunk-step stall aliased onto the end
   moment. Dominant on the certified ≤32k posture.
4. **M3+M5** — genuinely end-owned costs: ms-class per boundary, ≤302 ms
   total at 32k. Never the answer.

Unifying invariant: bystander ITT = step duration, and every lifecycle
boundary (admission, last chunk, finish, post-finish sweep) does its
work inside shared steps or gates the next step's composition.

## 8. Levers (status)

- **Shipped:** the lag+incremental store design already killed the
  synchronous end-flush ([iii] dead — that is WHY ends are clean at 32k).
- **Shipped, opt-in (cert4):** the p65 gated deferral improves the 64k
  end-of-prefill boundary NET — p95 −19 %, max −28 %, and the fix's gap
  profile BEATS the no-connector posture (1287/1585/1612 vs nolmc
  1111/2027/2347's p95/max class) — but it CONCENTRATES cost: the
  one-shot store could be further sliced (bounded-size store chunks per
  flush, e.g. ≤16 MB/step) to turn the ~1.6 s end-boundary max-gap into
  several ~300 ms gaps. NOT implemented, NOT validated — a P6.5 rung-3
  candidate requiring capture legs.
- **The remaining floor is scheduler-side [viii]** (any step carrying an
  N-token chunk costs every bystander N/2036 s): admission pacing /
  step-cap tradeoffs are the P6.6 prefill lane (thr2048 → ~1.1 s
  bystander ITT, ~4× better, TTFT +5–8 %; MNBT-16k r66c already
  quantified). A pacing law that keeps a fresh long prefill's first
  chunk OUT of bystander-shared steps would also erase the M4 aliasing.

## 9. Confirming capture (owner-attended-window item, not run)

One discriminating leg, no new code: U64k fresh face on the certified
default posture with **zero queued requests behind it** (wave-2
disabled), bystander table over [last long token, +3 s):
- gap present with empty queue → M1/M2 share (store tail + hangover);
- gap absent until a queued request is added → M4 share (admission
  aliasing).
Faces: (a) empty queue, (b) queued SHORT request, (c) queued LONG
request. Reuses `p65_timeline_v51.py` (W2MT=0 variant) on cert4; ~15 min
engine window + restore. Banked as the decision input for rung 3 / P6.6
priorities.

## Provenance

- Evidence: `results/2026-10-31-p65-r48a-stall-capture/` (VERDICT.md
  r48a–r48l + sub-leg logs), `results/2026-10-31-p66-*` (r66a/b/c),
  r70 close-out (`results/2026-10-31-p70-r70-cert4-comprehensive/`).
- Code (repo, cert4 lineage): `vllm_gaudi/xpu/patches/lmcint/
  vllm_v1_adapter.py` — deferral hunk :396-403, wfs :2119, flush call
  :2187, mamba banking loop :2244-2396, re-serve sweep :2389-2496, attn
  store loop :2498-2594, `_lmc_flush_pending` :2619-2691 (sync :2635),
  `request_finished` :3177-3196. Scheduler (extracted v0.30 in r48a
  evidence): running loop :581-665, chunk budget :661-680,
  `finish_requests` :2498, `_free_request` :2561-2588, deferred frees
  :2603-2636.
