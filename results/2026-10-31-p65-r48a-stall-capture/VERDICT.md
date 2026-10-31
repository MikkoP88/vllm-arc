# r48a–r48f — P6.5 capture legs: concurrent-stream stall attribution (cert3)

**VERDICT (r48a): stall CONFIRMED and attributed — candidate [ii] chunked-prefill
step granularity is the dominant mechanism at the certified posture (15.6 s of
the 17.8 s window at 32k context); candidates [i] minor (~2 s tail), [iii]/[v]/[vi]
no signature at 32k.**

**VERDICT (r48b, CORRECTED by r48c, RELENGTHED by r48f/r48g): fix rung 1
(threshold 2048) is LENGTH-DEPENDENT — catastrophic for 64k-under-load
(41.6→76.9 s fresh TTFT, bystander p95 up to 4947 ms), mild at 32k
(18.05→22.78 s, gaps 3996→1291 ms p95 — r48g), exposing a SECOND,
dominant-under-chunking mechanism. r48b's first attribution
(per-chunk-admission lookup walk) is OVERTHROWN by r48c: the lookup runs ONCE
per request (one elide line per boot; elide bought back only ~5 s of the ~59 s
regression). The real mechanism is [vii] store-queue backpressure — see the
r48c section. (The original "U32k 17.8→76.9" framing compared a 32k face to a
64k face — the closure bug made every r48b+ "fresh" face 64k content.)**

**VERDICT (r48f — REVERSAL): the r48e/r48d "false-prefix-hit correctness bug"
is RETRACTED — it was a HARNESS BUG. The v2–v4 timer lambda closed over the
comprehension's `tag` cell instead of using its bound default `tg`, so EVERY
face sent the LAST entry of its `P65_FACES` subset; the "fresh" faces after
the first were legitimate same-prompt replays (ground truth: r48f usage events
report prompt_tokens=63796 for all three long faces — the p64 text, 63 784+12
template tokens). All r48b–r48f faces are relabeled below; the first-send
faces of every leg were genuine and their TTFTs stand as valid cells. The
attribution matrix SURVIVES with corrected magnitudes — see the r48f section.**


## Harness (p65_timeline.py)

Cert3 image, certified posture (k3 + graphs + 25-var chain), greedy.
6 bystander streams ("Count from 1 to 1000.", max_tokens 2000, SSE chunk
timestamps client-side) + injected long-context stream (~32k synthetic
tokens, max_tokens 48) at t=+20 s. Warmup request first (first-traffic
law). Raw: `p65_timeline.json` (3118 chunks, 30 events).

## Measurements

| face | TTFT | bystander impact |
|---|---|---|
| U32k uncached (31 842 tok) | **17.82 s** | **frozen**: p95 gap 3991 ms, max 4302 ms; ~93 % starvation in the window |
| C32k cached replay | **1.09 s** | (window empty — wave drained; bounded by TTFT anyway) |
| baseline decode | — | p50 53.1 ms / p95 54.3 ms (healthy, ~2.7 tok/chunk k3) |

Per-bystander step pattern (identical on all streams, `byst0`/`byst3`
dumped): healthy 53-58 ms cadence → four stall gaps **3.64 / 3.70 / 3.99 /
4.30 s** (= the four 8192-token chunk steps; 31842 tok + spec ÷ 8192 ≈ 4
chunks) → two ~1.0 s gaps (post-prefill book/handshake before first token)
→ clean 53 ms cadence through the long stream's decode AND its END.

**W_F1_end (finish + store): CLEAN** — p95 59.5 ms, max 302 ms: the
request-end KV store does not stall bystanders at 32k (candidate [iii]
dismissed at this scale; the 0.5.6 store path is deferred/async).

## Attribution matrix (plan P6.5 (b))

| candidate | verdict | evidence |
|---|---|---|
| [i] sync-admission lookup walk | minor (~2 s of 17.8) | the two ~1 s gaps sit between last chunk and first token; per-request ZMQ roundtrips are 1.6-3.5 (p42 probe29 law, flat) |
| [ii] chunked-prefill starvation | **DOMINANT — refined**: decode is NOT excluded from chunk steps (a bystander chunk lands at every step end); the step itself takes ~4.3 s because it carries an 8192-token prefill chunk — bystander ITT = step time | 4 stall gaps ≈ 4 × chunk compute; scheduler running-loop schedules decodes and chunk prefills in ONE step up to max_num_batched_tokens=8192 (scheduler_v030.py:581-665) |
| [iii] request-end store flush | none at 32k | W_F1_end p95 59.5 ms |
| [iv] mamba snap at chunk ends | no separate signature | subsumed in chunk-step time; no inter-chunk spikes beyond step time |
| [v] XPU-graph recapture | no signature | no capture/WARN lines in window (serve log) |
| [vi] CPU-pool pressure | no signature | 100 GiB pool; no alloc failures/Evicts in window |

Physics: with prefill throughput ~2 036 tok/s (measured: 15.63 s for
31 890 tok), ANY step carrying an N-token chunk makes every co-scheduled
stream's inter-token gap ≥ N/2036 s. At chunk=8192 → 4.0 s. The only lever
that bounds bystander ITT while preserving prefill throughput is the
per-step chunk cap: `--long-prefill-token-threshold` (upstream
SchedulerConfig, vLLM 0.30 CLI flag verified in-image: arg_utils.py:1609;
only encoder-decoder resets it; 2048 is 64-aligned for the mamba split —
scheduler_v030.py:661-680).

Projected with threshold 2048: step ≈ 2048 tok ≈ 1.0 s + decode →
bystander ITT ~1.1 s (**~4× better**), 16 steps vs 4 (TTFT +~5-8 % step
overhead). The plan gate face (gap ≤ 2× baseline = 108 ms) is physically
unreachable at 32k context on this hardware (would need chunk ≤ ~220 tok →
~150 steps → TTFT +~12 s); gate re-derived for r48b as: max bystander gap
≤ ~1.2 s AND no throughput regression on the certified faces (±5 %).

## Harness defects found (fixed for r48b)

1. Bystander wave drained at ~45 s (2000 tok ÷ ~44 tok/s) — late windows
   empty. Fix: wave 2 at t=60 s (6×1500).
2. `long_prompt` used one fixed seed → the 8k prompt was a strict PREFIX
   of the 32k prompt → F3 measured the cached path (TTFT 1.10 s ≈ F2's
   1.09), not a fresh 8k prefill. Fix: per-face seeds; faces now U8k /
   U32k / U64k / C32k.

## Artifacts

- `p65_summary.txt` / `p65_timeline.json` — r48a faces + raw chunk/event timeline
- `p65_timeline.py` / `p65_gate.sh` — the leg tools
- `gate.log`, `scheduler_v030.py` (extracted from the cert3 image, in-tree
  read evidence for the step-budget code)
- rig: `/root/build/p65/` (engine log `serve_full_r48a.log`)

## r48b — rung 1 (threshold 2048) FALSIFIED (appended same session)

Posture: certified + `--long-prefill-token-threshold 2048` (flag verified
live in the config dump). Corrected harness (unique seeds — r48a's F3 was
accidentally a prefix of F1; second bystander wave at t=60).

| face | TTFT (r48a certified) | TTFT (r48b +LPT2048) |
|---|---|---|
| U32k uncached | 17.82 s | **76.89 s** |
| C32k cached replay | 1.09 s | **65.26 s** (queued behind the freeze) |
| U8k uncached | (invalid face in r48a) | 50.27 s |
| U64k uncached | — | 35.27 s |

Bystanders (12 streams): degraded through EVERY window (p50 68-2959 ms,
p95 up to 4947 ms) — generation collapsed to 8-21 tok/s engine-wide for
~80 s (stats lines), `Deferred: 1 reqs`, KV usage climbed 15→83 % then the
log-jam broke (prompt 6378 tok/s, KV →33 %). Engine stats + store cadence
(~40 k tok/10 s continuously) prove chunks kept completing: this is NOT
chunk compute; effective prefill throughput for the 64k-fresh face fell
1532 (r48f thr0) → 830 tok/s. *(The original "1789→414" divided by a 32k
premise — the face was 64k content.)*

**Mechanism (r48b first pass, r48c-corrected):** r48b attributed the collapse
to the per-chunk re-admission lookup walk. r48c falsifies that (below); the
corrected mechanism is [vii] store-queue backpressure: chunked KV (attn +
mamba-state + 12–14 streams' decode KV via `save_decode_cache: True`) is
enqueued into the LMCache save pipeline faster than it drains; the
synchronous `wait_for_save` in the forward then stretches every step by the
growing FIFO queue depth. The r48b log signature (gen collapse 8–21 tok/s,
store cadence continuous, effective U32k prefill 1789→414 tok/s) is exactly
this: everything crawls together at the growing step time.

Also observed (hygiene, P6.5-close item): `VLLM_QUANTIZE_Q40_LIB` unknown-env
WARNING fires 1×/boot in cert3 — the var is inherited from the BASE image
(`vllm-iexperimental:base-v0.26.0` `.Config.Env`), not our Dockerfile (the
P6.3 stanza removal was real but incomplete — the r47c "Q40 gone" claim is
corrected hereby).

### r48b artifacts

- `gate2.log`, `p65_summary_lpt2048.txt`, `p65_timeline_lpt2048.json`
- `run7n_t0_lpt2048.out` — PERFECT after the freeze (agg 300.07 / solo
  84.45 = certified warm numbers): the posture is healthy at steady state
- `serve_full_r48b.log` — stats lines + store cadence evidence

Next: r48c — boot A: `VLLM_ARC_LOOKUP_ELIDE=65536` alone (discriminates
the walk [i] from the mamba snap [iv]); boot B: ELIDE + threshold 2048
(the fix face: walk elided + step time bounded).

## r48c — lookup walk FALSIFIED; [vii] store-queue backpressure identified

Two boots (gate3.log): A = certified + `VLLM_ARC_LOOKUP_ELIDE=65536`;
B = A + `--long-prefill-token-threshold 2048`.

| face | A (elide, thr 0) | B (elide + thr 2048) | certified (r48a) |
|---|---|---|---|
| U32k uncached | **18.05 s** | ~~71.53 s~~ *(relabeled r48f 2nd pass: 64k FRESH content — compare against r48f's 41.61, not 17.82)* | 17.82 s |
| C32k cached replay | 1.45 s | 53.66 s (queued behind freeze) | 1.09 s |
| U64k uncached | — | ~~35.04 s = FULL SPEED~~ *(relabeled r48f: queued p64 REPLAY — not a fresh face)* | — |

- **Boot A ≡ certified** (18.05 vs 17.82; same four ~4-s chunk-step gaps,
  bystander W_U32k p95 3996/max 4302 ms): the lookup walk at 4 admissions
  is ~0. Elide is behavior-neutral at the certified posture.
- **Boot B: elide bought back only ~5 s** (76.89 → 71.53). The lookup walk
  is NOT the collapse mechanism. Exactly **one** `lookup-elide` line per
  boot (fired at a request's final admission: r48cA `tot=31817 engine=30784`) —
  `get_num_new_matched_tokens` runs once per request lifetime, never per
  chunk. r48b's "16 admissions × walk" model is dead; the p42 memo/elide
  machinery is irrelevant here.
- **Bystander gaps GROW linearly through the freeze** (per-second max gap
  1.09 s at U32k send → 4.22 s at +96 s; `p65_align.py` alignment of
  p65_tl_elide_lpt.json vs serve_full_r48cB.log). Not a fixed step cost —
  an accumulating queue.
- **Metric trap corrected:** `Avg prompt throughput` books at request
  completion, not per chunk — the healthy r48cA boot ALSO reports 0.0
  tok/s while U32k's chunks compute (then 3176 tok/s in the completion
  window). The r48cB "prompt 0.0" lines do NOT mean the prefill stalled;
  stores flowed at a constant **4992 tok/s (both workers) = ~2496 tok/s
  per worker** through the whole freeze (= the save pipeline draining at
  max rate: one 832-token attn store + one 1664-token mamba-state store
  per ~1.09 s cycle per worker, cost 9–20 ms each, i.e. burst 40–90k tok/s
  but pipeline-effective 2496).
- **No preemption, no re-admission churn:** zero preempt events (the 4 grep
  hits are LMCache config dumps); `Running: 7` constant, `Waiting: 0`
  through the freeze ("Deferred: 1" appears once, at C32k's send).
- **The jam breaks exactly when U32k's last chunk stores (~+96 s)** —
  inflow drops, the queue drains in seconds, and U64k (sent +85, queued
  behind the jam) then runs its tail at **0.97 s/chunk**. *(r48f relabel:
  that U64k face was a queued p32 REPLAY — its chunks carried retrieval +
  small tail compute, not fresh 2048-token prefill; "small chunks per se
  are fast" is WEAKENED to a replay-context observation until r48g's
  fresh 64k face re-measures it.)* The collapse needs small chunks × many
  concurrent streams.
- run7n t0 after boot B: PERFECT (agg 299.82/266.12, solo 84.47/84.44 =
  certified warm numbers). C32k's external retrieval proven live again:
  `Retrieved 39936 out of 39936 required tokens ... cost 971.6 ms`
  (chatcmpl-b19076ad, both workers, +59 s) — P6.1 conclusion re-confirmed.

### Attribution update (supersedes the r48b paragraph)

**[vii] store-queue backpressure** (new candidate, plan matrix extension):
with threshold 2048 every ~1-s step enqueues its chunk KV (attn 832-granular
+ mamba-state, `mamba_pool_ratio 2`) plus 12–14 streams' decode KV
(`LMCACHE_SAVE_DECODE_CACHE=True`); the forward's synchronous `wait_for_save`
(vllm_v1_adapter; connector comment: "LMCache stores synchronously within
the forward pass") waits on the FIFO save queue, so a marginally-saturated
pipeline turns into a linearly growing per-step stall for EVERY co-scheduled
stream. Explains: linear gap growth; engine-wide gen collapse; U64k
full-speed post-jam; run7n unaffected; threshold-0 immunity (the 4-s chunk
compute hides save latency — 9984 tok drain per 4-s step ≥ inflow); r48b ≈
r48cB (elide-independent); C32k's one-off ~1 s retrieval spike (sync load).

**r48d discriminator (launched):** boot A = threshold 2048 +
`LMCACHE_SAVE_DECODE_CACHE=False` (env-only, single variable — removes the
decode-stream store inflow). Predicted if [vii] holds: U32k TTFT ~20–30 s,
bystander gap ~1.0–1.2 s flat, U64k full speed, run7n unchanged. If the
freeze persists unchanged, the chunk/mamba store path itself is the driver
(narrowed [vii], next leg).

**Pre-registered refinement (codeside, before the r48d result):**
`ReqMeta.from_request_tracker` skips stores below the chunk boundary
(lmcint adapter:351-356), so 12 decode streams at 14–21 tok/s engine-wide
contribute ≈1 store per ~40 s — the freeze-time 3-stores/s cadence was
dominated by the CHUNK's own spans (2048 tok ≈ 2.46 × 832-span/step).
Therefore SDC=False removes only ~1–2 % of the inflow: the likely outcome
is **freeze persists (~70 s)** ⇒ [vii] narrows to the chunk/mamba store
path executing per ~1-s step (from_gpu gather + LMC_STORE_LAG device-sync
flush + pool alloc + 512-boundary mamba banking — `wait_for_save` fires
every step, adapter:2085-2153). A freeze that PERSISTS with SDC=False and
VANISHES with the connector dropped (boot B candidate: no kv-transfer-config)
would fully convict the store path.

## r48d — decode-store inflow EXONERATED; freeze persists (SDC=False)

Single boot (gate4.log): certified + threshold 2048 +
`LMCACHE_SAVE_DECODE_CACHE=False` — both knobs verified live in the engine
config dump (`save_decode_cache': False`, `long_prefill_token_threshold':
2048`).

| face | r48d (SDC=F) | r48cB (SDC=T) | healthy (thr 0) |
|---|---|---|---|
| U32k uncached | **69.77 s** *(64k FRESH content — relabel 2nd pass)* | 71.53 s *(64k FRESH)* | 41.61 s @64k (r48f) / 17.8–18.1 s @32k |
| C32k cached replay | 52.28 s (queued) | 53.66 s | 1.1–1.5 s |
| U64k uncached | ~~22.28 s (~2863 tok/s post-jam!)~~ *(r48f relabel: queued p64 REPLAY — the "~50% skip" reading is dead)* | 35.04 s | — |

- **The freeze PERSISTS essentially unchanged** (W_U32k_start p50 1404 ms /
  stall 560 s vs 1387/587 in r48cB): decode-phase store inflow — the only
  thing this boot removed — is NOT the driver. Pre-registered refinement
  confirmed (decode streams contribute ~1 store per ~40 s; the freeze-time
  cadence was the chunk's own spans).
- ~~U64k post-jam at ~2863 tok/s: small chunks intrinsically FAST~~ —
  *(r48f relabel: that face was a queued p32 REPLAY; withdrawn. Fresh
  small-chunk rate is unmeasured until r48g.)*
- run7n t0: PERFECT (agg 299.58, solo 84.44/84.37).

**[vii] narrowed to: the chunk/mamba store path executing per ~1-s step**
(`wait_for_save` fires every scheduler step: flush-prev-snapshot after a
device sync + gather + pool alloc + memcpy + 512-boundary mamba banking).
Conviction leg r48e: threshold 2048 + NO connector (`ARC_LMC=0`, env gate
serve_arc_b70.sh:165) — clean U32k (~16-18 s) convicts the store path 100 %;
residual growth would expose a second scheduler-side mechanism.

### r48d artifacts

- `p65_sum_sdcf_lpt.txt` / `p65_tl_sdcf_lpt.json` / `serve_full_r48d.log` /
  `boot_sdcf_lpt.out` / `run7n_t0_sdcf_lpt.out` / `gate4.log` / `p65_gate4.sh`

## r48e — conviction leg (no connector): the freeze SPLITS into two mechanisms

Boot B (gate5.log): certified posture MINUS the LMCache connector
(`ARC_LMC=0`, env gate serve_arc_b70.sh:165; engine log confirms zero
LMCache/kv_transfer lines) PLUS threshold 2048.

| face | r48e (no LMC) | r48cB (LMC on) | healthy (thr 0) |
|---|---|---|---|
| U32k uncached | **52.27 s** (64k FRESH) | 71.53 s (64k FRESH) | 41.61 s @64k (r48f) |
| U64k uncached | **1.75 s (!!)** (same-prompt replay) | 35.04 s | — |

- **The freeze does NOT vanish without the connector** — U32k 52.27 s,
  bystander W_U32k_start p50 1111 ms / max 2347 ms, same growing-gap shape,
  same jam-breaks-at-completion pattern (engine stats: gen 19→17 tok/s
  crawl, then a 6367.8-tok/s completion burst). ~~The regression splits:
  [vii-a] ≈ 19 s + [viii] ≈ 34 s~~ *(r48f relabel: this split was computed
  against a 32k baseline while the r48e "U32k" face was a 64k fresh — the
  corrected split, from the clean 64k matrix, is [vii-a] +24.6 s /
  [viii] +10.7 s; see r48f section.)*
- run7n t0: PERFECT (agg 300.06, solo 84.51/84.46).
- `_mamba_block_aligned_split` exonerated as a chunk-shredder: for
  block-aligned 2048 chunks every stop is one-time or end-near
  (scheduler_v030.py:413-526 re-read in full); chunks stay 2048.
- `prefix_cache_retention_interval` already at its semantic-only default
  (image cache.py: `Field(default=0)`) — no retention knob left to pull.
- Mamba align semantics (image cache.py doc): align caches state at the
  last token of each scheduler step when block-aligned — with 2048-token
  chunks that is EVERY chunk end (16 snapshots vs 4 at threshold 0):
  candidate [iv] revived as the leading [viii] suspect (materialization
  cost per chunk end), but it does not by itself explain the linear gap
  growth — needs a probe boot with per-step timing to settle.

### r48e anomaly — FALSE full-prefix hit on a fresh unique prompt (correctness class)

> **RETRACTED by r48f** (harness closure bug — see the r48f section): the
> r48e `P65_FACES="U32k,U64k"` subset made the "U64k" face the SECOND send
> of the same p64 text the "U32k" face had just sent. TTFT 1.75 s =
> a legitimate local-prefix replay of an identical prompt. No correctness
> defect. The text below is kept as the capture record; the "1 shared
> token" prompt-diff check was run against the WRONG pair (p32 vs p64 —
> neither is what the face actually sent).

U64k TTFT **1.75 s for a 63 784-token prompt** = 36 448 tok/s — physically
impossible (solo eager prefill ≈ 2036 tok/s). The engine computed only
~1 400 tokens of it (the 00:15:06 window books 139.3 tok/s avg; the only
prompt burst, 63 678 tokens, is U32k's completion one window earlier).
The prompts share **1 token** (verified rig-side: first diff at char 38;
seeds 1010101 vs 3030303) — no legitimate prefix-cache hit is possible.
Local "Prefix cache hit rate" jumped 0.0→44.1 % exactly at U64k's
admission. **The engine skipped ~97.8 % of a fresh unique long prompt and
generated from uncomputed KV** — output-content sanity was not captured
(harness v3 discarded text); v4 (r48f) fixes that.

The same anomaly appears PARTIALLY with the connector on: r48d's U64k
22.28 s ≈ 2863 tok/s (> solo rate) with no completion-sized booking burst
(align of p65_tl_sdcf_lpt.json: only ~5 195 tokens booked near its
completion window) ⇒ ~50 % skipped. During-jam admissions (r48b 35.27 s,
r48cB 35.04 s) computed fully. The trigger correlates with admission
AFTER prior long requests completed and their blocks freed (post-drain),
not with the connector.

**r48f (launched): reproduction attempt on the CERTIFIED posture** (no
seds — the shipped config), harness v4 = usage `cached_tokens` capture +
first-240-chars content capture per stream, faces U32k/C32k(control)/U64k.
If `cached_tokens >> 0` for the fresh U64k on the certified posture, this
is a shipped-posture correctness bug (P6.5 sub-item, engine-side —
likely the hybrid/mamba prefix-cache path matching freed-block state).

### r48e artifacts

- `p65_sum_nolmc_lpt.txt` / `p65_tl_nolmc_lpt.json` / `serve_full_r48e.log` /
  `boot_nolmc_lpt.out` / `run7n_t0_nolmc_lpt.out` / `gate5.log` / `p65_gate5.sh`

### r48c artifacts

- `p65_sum_elide.txt` / `p65_tl_elide.json` / `serve_full_r48cA.log` /
  `boot_elide.out` — boot A
- `p65_sum_elide_lpt.txt` / `p65_tl_elide_lpt.json` / `serve_full_r48cB.log` /
  `boot_elide_lpt.out` / `run7n_t0_elide_lpt.out` — boot B
- `gate3.log`, `p65_gate3.sh` (also `.tmp/` + rig `/root/build/p65/`)
- `p65_align.py` (`.tmp/`, rig) — client-timeline ↔ engine-log per-second
  alignment tool (the bys_max_gap/stored/retrieved columns above)

Also observed: engine config dumps `'chunk_size': 832` although the chain
exports `LMCACHE_CHUNK_SIZE=1024` — parked (pre-existing, both certs;
not P6.5-blocking).

## r48f — REVERSAL: harness closure bug; false-hit RETRACTED; matrix corrected

Boot (gate6.log): **pure certified posture** (no seds, no env deltas — the
shipped config), harness v4 (usage capture + per-stream first-240-chars
content). The leg was launched to reproduce the r48e "false full-prefix
hit". It did something better: it exposed the harness bug that had been
shaping every leg since r48b.

### The bug (p65_timeline.py v2–v4)

```python
timers = [at(at_s, (lambda tg=tag: sse_call(tg,
          {"messages": [..., "content": prompts[tag]}, ...)))
          for tag, at_s, ... in FACES]
```

The lambda binds `tg=tag` as a default but its **body reads `prompts[tag]`**
— a closure over the comprehension's `tag` cell, whose value after the
comprehension completes is the **LAST** face of the filtered `FACES` list.
Every timer therefore sent the same prompt: `prompts[last entry of the
P65_FACES subset]`. Timers fire long after the comprehension finishes (face
times +25…+85 s), so every face — whatever its label — sent the last entry's
content.

**Ground truth (server-side usage events, captured by v4's
`stream_options.include_usage`):** all three long faces report
`prompt_tokens=63796` (= the p64 text, 63 784 + 12 template tokens);
bystanders correctly report 23. The harness's own `prompt_tokens` dict
records the *intended* sizes (U32k=31805, U64k=63784, U8k=7961) — the
client built distinct prompts and then didn't send them.

### Relabel table (what each face actually was)

| leg | P65_FACES subset | last entry | all faces sent | first-send face (genuine) | later faces |
|---|---|---|---|---|---|
| r48a | v1 harness (no bug) | — | per-face, genuine | U32k 31 842 ✓ | ✓ |
| r48b | full: U32k,C32k,U8k,U64k | U64k | **p64 63 784** | "U32k" = **64k FRESH** ✓ | C32k/U8k/U64k = queued same-prompt replays |
| r48cA | U32k,C32k | C32k | **p32 31 805** | "U32k" = **32k FRESH** ✓ | "C32k" = replay (label correct by luck) |
| r48cB | U32k,U64k,C32k | **U64k** *(filtered list keeps FACES order U32k,C32k,U64k — corrected 2nd pass)* | **p64** | "U32k" = **64k FRESH** ✓ | queued replays |
| r48d | U32k,U64k,C32k | **U64k** (same) | **p64** | "U32k" = **64k FRESH** ✓ | queued replays |
| r48e | U32k,U64k | U64k | **p64** | "U32k" = **64k FRESH** ✓ | "U64k" = same-prompt replay |
| r48f | U32k,C32k,U64k | U64k | **p64** | "U32k" = **64k FRESH** ✓ | replays (usage ground truth) |

*(The first relabel pass (written before r48g landed) had r48cB/r48d ending at
C32k by reading the P65_FACES string order — the filter preserves FACES
declaration order, so those legs also ended at U64k. Consequence: every
"U32k"-labeled face in r48b/r48cB/r48d was a 64k fresh prefill; NO genuine
32k-fresh thr2048+LMC cell existed until r48g. The 64k matrix below was
already correct; the 32k rows are corrected here.)*

Cross-checks that validate the relabeling: r48cA's "U32k" booked 31 765
tokens at completion (= p32); r48f's usage = 63 796 (= p64); every leg's
first-send face ran at physics-consistent rates; every "impossible" TTFT
(r48e/r48f U64k 1.75/1.85 s) was a second send of an identical prompt.

### Retractions

1. **False-prefix-hit "correctness bug": RETRACTED.** No engine defect. The
   r48e "U64k" face was the second send of the same p64 text the "U32k"
   face had completed seconds earlier: TTFT 1.75 s = a legitimate
   local-prefix replay (r48f's analog retrieved 44 928/44 928 required
   tokens). The r48d "~50% skip" = a queued replay computing only its tail.
   The "post-drain admission trigger" correlation = simply "2nd/3rd send
   after the 1st completed and its blocks were still cached".
2. **r48f's "contaminated content"** (U32k/C32k/U64k all answering
   "stage 4212 / Note 128") = same prompt → same answer. The identical
   texts plus identical usage counts were in fact the smoking gun that
   exposed the bug.
3. **"Fresh small-chunk full speed" claims withdrawn** (r48cB 0.97 s/chunk,
   r48d 2863 tok/s): both measured REPLAY tails, not fresh 2048-token
   chunk compute. Unmeasured until r48g.
4. The **"usage anomaly"** (all faces prompt_tokens=63796) is the bug
   itself, not engine misaccounting. `cached=None` on every usage chunk:
   this vLLM does not populate `prompt_tokens_details.cached_tokens` —
   no engine-side cached-token reporting exists in this build, so
   skip-evidence must come from bookings/content/TTFT physics (as done).

### Corrected matrix (single image, k3, graphs, 12 bystanders, greedy)

**64k fresh prefill TTFT (all cells genuine first-sends):**

| posture | TTFT | eff. tok/s | bystander W_start p50 / p95 / max gap |
|---|---|---|---|
| thr0 + LMC (**certified**) | **41.61 s** (r48f) | 1532 | **3723 / 5607 / 5608 ms** |
| thr2048 + LMC | **76.89 s** (r48b) | 830 | 1399 / 4829 / 4947 ms (growing) |
| thr2048 + LMC + elide | 71.53 s (r48cB) | 883 | 1387 / 3824 / 4219 ms (growing) |
| thr2048 + LMC + SDC-off | 69.77 s (r48d) | 914 | 1404 / 3072 / 3413 ms (growing) |
| thr2048 + LMC (near-solo*) | 49.91 s (r48g U64k) | 1278 | (windows empty — see below) |
| thr2048 + LMC (6 bys, W2MT 4000) | 56.10 s (r48iv2, defer no-op) | 1137 | 1238 / 1969 / 2242 ms |
| thr2048 + nolmc | **52.27 s** (r48e) | 1220 | 1111 / 2027 / 2347 ms (growing) |

\* r48g's 64k face ran at +85 s after both bystander waves had drained
(v5 wave-2 of 1500 tok ended ~+89 — W_U64k windows n=0): ~near-solo. Its
49.91 s ≈ r48e's full-concurrency nolmc 52.27 s is consistent with
[vii-a] needing concurrent streams, and flags that leg-to-leg freeze
severity is concurrency- and phase-dependent (76.89 vs 49.91 same posture
family).

→ **[vii-a] connector store-path amplification = +24.6 s TTFT** (r48b −
r48e, matched 12-bystander concurrency) and p95 gap 4829 vs 2027 ms.
**[viii] engine-side small-chunk cost = +10.7 s** (52.27 − 41.61 ≈ 25%
throughput; two-variable caveat — the thr0+nolmc cell is missing; r48h
CLOSED the connector variable at fixed threshold at 32k: nolmc ≡ LMC
there, so the 32k gap floor is pure [viii]).

**32k fresh prefill TTFT:**

| posture | TTFT | bystander W_start p50 / p95 / max |
|---|---|---|
| thr0 + LMC (+elide, inert) | **18.05 s** (r48cA; r48a v1: 17.82) | p95 3996 / max 4302 ms (4×4.2-s chunk steps) |
| thr2048 + LMC | **22.78 s** (r48g, v5 labels) | **729 / 1291 / 1309 ms** |
| thr2048 + nolmc | **22.39 s** (r48h) | **705 / 1262 / 1276 ms** — identical to +LMC |

**Store-event scaling ([vii-a]'s physical basis):** r48f booked **112**
`Stored … out of …` lines (a p64 triple: fresh + 2 replays, thr0) vs
r48cB **304** (the same p64 triple, thr2048) — same content, ~2.7× the
events ≈ **5.5× the per-token store-event density**. At thr2048 every
~1-s step enqueues its attn spans + a mamba state per chunk end (align
mode materializes at every block-aligned chunk end — 16 vs 4 snapshots
per 32k) + boundary banking; `wait_for_save` fires every step (lmcint
adapter:2085+). The freeze's linear gap growth = backlog accumulation at
this event rate.

### The reframe (what P6.5 actually is)

- **The certified posture itself FAILS the gate, hard**: during a 64k
  fresh prefill bystanders see p50 3.7-s gaps (8 chunk steps × ~4.3–5.6 s;
  70× the 53-ms baseline); at 32k, 4×4.2 s. This IS the owner's reported
  stall ("streams stop decoding when one stream with long context starts").
- The upstream knob built for exactly this (`--long-prefill-token-threshold`)
  is **length-dependent with the connector as-is**: at 32k it nearly meets
  the gate already (r48g: 22.78 s, gaps 729/1291/1309 ms vs certified
  3996/4302 — 3× better gaps for +26% TTFT); at 64k under load it is
  catastrophic (76.89 s vs 41.61, gaps growing to ~5 s — [vii-a]).
- **Without the connector the knob nearly meets the gate at 64k** (r48e:
  p50 1111 ms ≈ the ~1.2-s bar; p95 2027 still over — [viii] residual).
- **Local + external replay is healthy everywhere** (1.45–1.85 s replay
  TTFTs, retrieval 44 928 tok sub-second; the r48cB `Retrieved 39936 …
  971.6 ms` line stands). Cached streams are NOT part of the problem; the
  problem is exclusively FRESH long prefill under co-scheduled decode.
- run7n stays PERFECT on every posture tested (thr0/thr2048/SDC/nolmc) —
  steady-state throughput is not at risk in any direction.

### Fix ladder (rung decision)

1. ~~Knob alone (thr2048 + LMC as-is)~~ — REJECTED as a shipping posture:
   near-gate at 32k (+26% TTFT buys 3× better gaps) but catastrophic at
   64k-under-load (1.85× TTFT, gaps still growing to ~5 s), and the +26%
   32k TTFT cost itself exceeds the ±5% gate.
2. **Rung 2 (glue-path, our adapter file): chunk-set store deferral.**
   Under `--long-prefill-token-threshold` chunking, stop enqueuing
   per-chunk attn-span stores + per-chunk-end mamba banking every step;
   bank spans through the chunk set and store once at set completion (or
   request end). Removes [vii-a]'s per-step events; then thr2048 should
   approach the r48e profile with the connector's benefits intact.
   Gate face: 32k/64k fresh TTFT ≤ certified +~10% AND bystander p50
   ≤ ~1.2 s, run7n ±5%.
3. Rung 3 (engine-side [viii], ≈+25% prefill cost at 64k): mamba-align
   per-chunk-end materialization — only if rung 2 leaves the gate unmet;
   engine-code → cert4.

### r48g / r48h (launched — harness v5, `prompts[tg]` fixed)

Boot A (r48g): certified + thr2048, faces **U32k FRESH + C32k replay +
U64k FRESH** (label-correct thr2048+LMC cells at both lengths — no prior
leg has them) + run7n t0 = the **pre-fix baseline** for the rung-2 patch.
Boot B (r48h): thr2048 + `ARC_LMC=0`, face **U32k FRESH** = direct
[vii-a] isolation at 32k (completes the matrix; also re-measures fresh
small-chunk prefill rate cleanly).

## r48g — label-correct thr2048+LMC cells: 32k is NEAR-GATE; boot-B crash

Boot A completed (gate7.log; harness v5 — usage events confirm true
per-face prompts):

| face | TTFT | notes |
|---|---|---|
| U32k FRESH (31 805 tok) | **22.78 s** (1396 tok/s) | bystanders W_U32k_start **p50 729 / p95 1291 / max 1309 ms** — vs certified thr0's 3996/4302 |
| C32k replay | **1.20 s** | W_C32k_start p50 60 ms — replay path clean under thr2048+LMC |
| U64k FRESH (63 784 tok) | **49.91 s** (1278 tok/s) | near-solo: both bystander waves drained by +89 s (v5 wave-2 = 1500 tok); W_U64k windows n=0 |

- run7n t0: PERFECT (agg 296.5, solo 84.39/84.39 — certified numbers).
- **First genuine 32k-fresh thr2048+LMC cell ever measured** (all prior
  "U32k" thr2048 faces were 64k content — see the second relabel
  correction): the knob+connector at 32k is mild — +4.7 s TTFT (+26%) for
  3× better bystander gaps. The [vii-a] collapse is a 64k×concurrency
  phenomenon.
- The 49.91 s near-solo 64k ≈ r48e's 52.27 s full-concurrency nolmc —
  [vii-a] needs concurrent streams, and freeze severity varies 49.9–76.9 s
  across legs at the same posture family (phase/concurrency dependence;
  any future gate face must fix concurrency explicitly).
- Harness limitation found: wave-2 (1500 tok) drains ~+89 s — late faces
  run near-solo. v5.1 adds `P65_W2MT` (r48i runs 3000).
- **Boot B (r48h) FAILED to start**: the arc-test container was SIGKILLed
  (exit 137) at launch, right after boot A's `docker rm -f` under
  post-run7n load (OCI exec error). W14 was restored by the EXIT trap.
  Retried in gate8 with a 30-s drain between boots.

### r48g artifacts

- `r48g/p65_sum_lpt_v5.txt` / `r48g/p65_tl_lpt_v5.json` — faces + usage
  events (true per-face prompts: U32k=31805-class, U64k=63784-class) +
  bystander tables (raw)
- `r48g/serve_full_r48g.log` — 306 `Stored … out of total` store events
  (the thr2048 per-chunk cadence; vs 112 in r48f thr0)
- `r48g/boot_lpt_r48g.out`, `r48g/run7n_t0_lpt_v5.out`, `r48g/gate7.log`,
  `r48g/p65_gate7.sh`, `r48g/p65_timeline_v5.py`

### r48i / r48h (gate8, launched — deferral probe + nolmc retry)

**r48i** = r48g's posture + `LMC_DEFER_CHUNK_STORE=1` + the installed
adapter patched in-container (docker cp before the engine's first lmcache
import; `arc p65 fork delta` marker grep + py_compile verified in-boot).
The patch (29 diff lines, `adapter_installed.py` → `adapter_p65.py`):
scheduler-side, `from_request_tracker` — prefill-chunk rows
(`input_token_len < prompt_len`) keep their row (loads/mamba addressing
ride it) but get `SaveSpec.can_save=False` and leave `num_saved_tokens`
un-advanced, so the final-chunk row stores the whole span in ONE shot and
the mamba registry banks every boundary in one sweep. Decode rows and
disagg rows untouched. Pre-registered predictions:
- U32k fresh: 19–25 s (≈ r48g 22.78 minus the 32k [vii-a] share if any);
  bystanders p50 ≤ 800 ms
- C32k replay: ~1.2 s (unchanged)
- U64k fresh (P65_W2MT=3000, live bystanders): **42–52 s** — the fix
  face; ≈ r48e's nolmc profile if [vii-a] is the store path
- `Storing KV cache` rows: **ONE per long face** (vs ~16) — the
  mechanistic signature
- run7n: 290–300 agg / 84–85 solo
**r48h** = thr2048 + `ARC_LMC=0`, U32k only (32k [vii-a] isolation;
predicted 18–21 s, p50 ~0.9–1.2 s).

**r48i RESULT: NO-OP — the deferral did not intercept the stores.**
In-boot checks all passed (ENV_CHECK `LMC_DEFER_CHUNK_STORE=1` from
`/proc/<pid>/environ`, `PATCH_MARKER=1` on disk, py_compile OK, adapter
live — "LMCache hybrid mode" in log), yet the store histogram is
byte-identical to r48g's (118×1664-tok attn spans + 128×832 + 24 more
832), i.e. `defer_chunk_store` never evaluated True. Its 53.29 s 64k TTFT
is therefore a **no-fix measurement** (and cannot be read as fix
validation OR failure). Attribution narrowed before retry: `ReqMeta(` is
constructed at exactly ONE site (`from_request_tracker`,
vllm_v1_adapter.py:485), called from 4 places (lines 2718/2812/2851/2883
of the installed file), and `lmcache_connector_v1.LMCacheConnectorV1Dynamic`
is a pure delegating wrapper — so the hunk sits in the only row path.
**Root cause (ruling out code paths): import race.** `vllm serve` imports
the adapter at config validation (~T+2–4 s after exec); the `docker cp`
lands ~T+1–2 s after `docker run`; once any parent process imports the
old bytes, forked children (EngineCore workers) keep them from memory
regardless of the disk file. The cp and the import are a coin flip — and
the disk-file greps cannot see which side won. **r48i-v2 (gate9)** removes
the race class entirely: the patched adapter is `docker commit`-ed into a
probe image (`vllm-arc:v0.30.0-b70-cert3-p65probe`, committed-file md5
`f07c955e…` == `adapter_p65.py`) and the normal serve script boots that
image; behavioral activation check = the 1664-token attn-span store class
collapses 118 → ~0. Faces U32k/C32k/U64k, `P65_W2MT=4000` (r48g's 3000
drained before the 64k face had live bystanders; 4000 keeps 6 streams
through +60→+134, covering the U64k prefill window), predictions
unchanged from r48i. r48h rides the same gate with a **graceful
teardown** (`pkill -f "vllm serve"` → 10 s drain → `rm -f`) — both prior
r48h deaths (gate7 B, gate8 §5) followed `rm -f` of a loaded engine
(exit 137/OCI at next container start).

**r48i-v2 RESULT (gate9): still a NO-OP — import-race theory DEAD.**
Patch baked into the image layer (`docker commit`, committed-file md5
verified `f07c955e…`), env verified in the engine process environ
(`/proc/<pid>/environ`: `LMC_DEFER_CHUNK_STORE=1`), yet the store
histogram again matches no-defer (122×1664 attn + 236×832 mamba; r48g
had 118×1664 + 128×832). `defer_chunk_store` never evaluates True at
runtime. The physics replicated cleanly at the same time: U32k 22.89 s
(r48g: 22.78), C32k replay 1.21 s, U64k **56.10 s** (vs 53.29 with
6 bystanders, 76.89 with 12 — the [vii-a] freeze cost scales with
concurrency/phase, consistent with the 79 ms/event-per-step model),
W_U32k gaps 732/1295 (r48g: 729/1291), W_U64k gaps 1238/1969/max 2242,
baseline 53.5 ms, run7n 295.1 agg / 83.7 solo (cert-grade posture).
Code-side ruling (read of the installed 0.5.6+arc1 adapter): the chunk
continuation rows DO flow through `from_request_tracker`
(`build_connector_meta` → cached_reqs loop: `tracker.update()` appends
each chunk, `prompt_len` = full prompt, `token_ids` = cumulative), so
every term of the condition looks correct on paper — the contradiction
is resolved by measurement, not reading: **r48j (gate10)** bakes an
INSTRUMENTED adapter (rate-limited WARNING per prefill row logging
`in_len/prompt_len/num_saved/chunk_sz/skip_save/defer/disagg/env`) into
a probe2 image; one 32k boot discriminates (a) rows absent while stores
continue → rows built elsewhere, (b) rows present + defer=False → the
term values show why, (c) defer=True while stores continue → worker
side ignores can_save.

**r48j RESULT (gate10): the no-op is a WRONG-FILE patch — scenario (a).**
`ROWLOG_LINES=0` across a full 32k fresh face (TTFT 23.54 s, stores
44×1664 + 106×832): the lmcache-integration adapter's
`from_request_tracker` never executed ONCE while the stores ran. The
live row-builder is the **fork's lmcint copy**: serve config
`kv_connector='LMCacheConnectorV1'` with
`kv_connector_module_path=None` → vllm's factory imports
`vllm…v1.lmcache_connector`, and
`vllm_gaudi/xpu/patches/lmcache_hma_connector._SrcOverrideFinder`
(meta-path hook installed by the fork) **exec's
`lmcint/vllm_v1_adapter.py` source in place of the vendored
lmcache_integration module at import time** (boot marker per process:
`[…@lmcint] src-override OK`; no hash gate — the docstring md5s are
provenance records, not enforcement). r48i/r48iv2/r48j patched
`lmcache/integration/vllm/vllm_v1_adapter.py` — the lmcache package's
own adapter, **not in this boot's import path at all**. The two
lineages share log strings ("LMCache hybrid mode", identical
`from_request_tracker` body), which is why every liveness check
(env-in-process, marker-on-disk, hybrid-mode line) passed while the
behavior never changed. Lesson banked: **adapter-liveness checks must
name the module path** (`src-override OK` / `__module__`), never shared
strings. Fix leg r48k (gate11): probe3 image commits
`adapter_lmc_p65.py` — installed-lmcint + the SAME deferral hunk
already sitting in the repo copy
(`vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py` since r48i prep) +
a probe-only `ARC_P65_DEFER` WARNING per deferred row — and re-runs the
r48iv2 posture exactly (U32k+U64k, `P65_W2MT=4000`, thr2048+LMC+DEFER,
run7n) for the clean A/B.

**r48k RESULT (gate11): THE DEFERRAL IS LIVE — 64k wins, 32k regresses,
length gate required.** `SRC_OVERRIDE=9` (lmcint exec'd in every engine
process), `DEFER_ROWS=64` with exactly the designed shape (`in_len`
1664→3328→4992→6656 growing, `num_saved=0` held un-advanced, EngineCore
pid, module path = the src-override'd lmcache_integration name), and the
store histogram TRANSFORMED as predicted: **1664-class 122 → 0**; new
one-shot whole-span finals 2×31616 (32k face) + 2×63232 (64k face) +
2×4992/2×7488 (warmup). Performance, same posture as r48iv2 (6 bys,
W2MT 4000, thr2048+LMC):

| face | no-defer (r48iv2) | defer (r48k) | delta |
|---|---|---|---|
| U64k TTFT | 56.10 s | **51.90 s** | **−4.2 s (−7.5%)** |
| U64k W gaps p50/p95/max | 1238/1969/2242 ms | 1224/1598/2008 ms | p95 −19%, max −10% |
| U32k TTFT | 22.89 s | **27.06 s** | **+4.2 s (+18%) — REGRESSION** |
| U32k W gaps p50/p95 | 732/1295 ms | 1013/2231 ms | worse |
| run7n agg/solo | 295.1/83.7 | **300.6/84.4** | zero steady-state cost |

Mechanism of the regression: the one-shot final store moves ~0.51 GB
(31.6k tokens) in a single contended operation (~0.24 GB/s big-store
regime ≈ 2+ s concentrated at the final chunk), while r48h PROVED the
per-chunk events it replaced cost ≈0 at 32k (LMC == nolmc). At 64k the
trade inverts (event backlog compounding > one-shot cost). The mamba
832-class (220 stores, ~9 ms each at 4.2 GB/s, worker-side page
materialization path) is untouched by row deferral and BENIGN at this
concurrency: r48k's 64k with 220 of them (51.90 s) ≈ r48e's nolmc with
zero (52.27 s). **Fix shape (rung 2 final): length-gate the deferral —
`LMC_DEFER_MIN_TOKENS` (default 0 = defer all; certified chain sets
40960, between the 32k face where [vii-a]=0 and the 64k face where it
compounds).** r48l (gate12, probe4) validates: U32k must return to the
22.5–23.5 s family with 1664-class stores present and ZERO defer rows;
U64k must stay 50–52.5 s with one-shot finals; run7n unchanged.

**r48l RESULT (gate12): GATED FIX VALIDATED — EVERY PREDICTION HELD.**

| check | predicted | observed |
|---|---|---|
| defer rows | 0 for U32k, ~38 for U64k | **38, ALL prompt_len=63796** |
| 1664-class stores | return for 32k, gone for 64k | 48 (32k family), 0 for 64k |
| one-shot finals | ~63k store present | 2×63232 |
| U32k TTFT | 22.5–23.5 s | **22.88 s** (r48g 22.78, r48iv2 22.89 — family restored) |
| U32k W p50/p95 | ~700–900 / ~1300 | 995 / 1300 (r48g 729/1291 family) |
| U64k TTFT | 50.0–52.5 s | **52.25 s** (win held: no-defer 56.10) |
| U64k W p50/p95/max | ~1100–1300 / ≤1700 | 1287 / 1585 / 1612 |
| run7n agg/solo | 290–300 / 83–84 | **300.05 / 84.44** |

**P6.5 fix closure (rung 2, glue-path, env-gated + length-gated):**
`LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` in
`vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py` (repo copy carries
the exact hunk; probe legs r48k/r48l validated it in-image). Net effect
at the validated posture: 64k-fresh TTFT −7.5 % (56.10→51.90/52.25),
bystander p95 gap −19 % / max −28 % during 64k prefill — **and the fix's
gap profile now BEATS the no-connector posture** (1585/1612 vs r48e
nolmc 2027/2347): with stores deferred, remaining 64k-prefill gap
dilution is [viii] engine-side chunk compute. 32k faces byte-identical
to certified behavior. Steady state (run7n) untouched across r48g/
r48iv2/r48k/r48l (295–301 agg, 83.7–84.4 solo — noise band). Remaining
exposure, explicitly out of rung 2: the ~1.3–1.6 s bystander gap FLOOR
during any ≥32k chunked prefill ([viii], present with or without
LMCache; r48h) — that is the P6.6 prefill lane.

**r48h RESULT (gate9, graceful teardown fixed the crash): thr2048 +
nolmc U32k = 22.39 s, gaps p50 705 / p95 1262 / max 1276 ms —
statistically IDENTICAL to thr2048+LMC (r48g: 22.78 s, 729/1291/1309).**
At 32k the LMCache store path costs ≈0.4 s TTFT and ≈0 bystander gap:
**[vii-a] is a 64k-scale phenomenon** (40+ chunk rows × bigger spans ×
bystander concurrency), and the corrected matrix's 64k row (52.27
nolmc vs 76.89 LMC, +24.6 s) is the only face where the store-path fix
can pay. The 32k bystander floor (~700 ms p50 during ANY 32k chunked
prefill, LMC or not) is the engine-side [viii] dilution — prefill
compute monopolizes steps regardless of connector. Zero store lines in
the r48h log (nolmc confirmed); C32k/U8k/U64k faces not needed for this
cell.

**[vii-a] micro-cost closure (r48b log, pre-registered before r48i):** the
freeze leg's 312 store events cost only ≈4.4 s of raw execution (124×
1664-tok spans at p50 20.5 ms + 188× 832-tok at p50 10.0 ms) — the store
BYTES are cheap. The remaining ~24.6 s over ~312 events ≈ **79 ms of
per-event fixed overhead** (lag-flush device sync + from_gpu gather +
submit per scheduler step under concurrency). Removing the events — not
making them faster — is therefore the correct fix shape; if r48i's
deferral collapses the freeze to ≈ r48e's nolmc profile, this model is
confirmed. (Also visible in r48b: 2×41 600/51 584/59 904-token single
stores — the mamba sweep batches many boundary objects into ONE
cache_engine call, i.e. the deferred one-shot pattern already exists
upstream for mamba banking.)

### r48f artifacts

- `r48f/p65_sum_cert_v4.txt` / `r48f/p65_tl_cert_v4.json` — faces, texts,
  usage events, bystander table (raw)
- `r48f/serve_full_r48f.log` — 112 store lines, 44 928-tok retrieval,
  hit-rate 0.0→31.2% trace, 43 WARN/ERROR lines (all upstream-accepted set)
- `r48f/boot_cert_r48f.out`, `r48f/gate6.log`, `r48f/w14_restore_r48f.log`,
  `.tmp/p65_gate6.sh`, `.tmp/p65_timeline.py` (v5; the v4→v5 delta is the
  one-character `prompts[tag]`→`prompts[tg]` fix + comment, quoted above —
  the rig's v4 as-run copy was overwritten by the v5 scp)

### r48i / r48j / r48k / r48l artifacts (deferral probe → fix validation)

- `r48i/` (gate8, no-op leg): `adapter_installed.py` (post-gate8-cp
  container copy — actually the PATCHED lmcache-integration file,
  md5 `f07c955e`; the pristine is `56de9ee3`, see r48j for why this
  file was the wrong target), `adapter_p65.py`, `adapter_p65.diff`
  (34 lines), capture/serve/boot/run7n logs, `p65_gate8.sh`,
  `p65_timeline_v51.py` (v5.1 = v5 + `P65_W2MT`)
- `r48iv2/` (gate9, race-free no-op + r48h boot): `gate9.log`,
  `p65_gate9.sh`, capture + `serve_full_r48iv2.log` (358 stores:
  236×832 + 122×1664), run7n, boot outs; probe image
  `vllm-arc:v0.30.0-b70-cert3-p65probe` (docker-commit of cert3 +
  patched lmcache-integration adapter)
- `r48h/` (gate9 boot B, first SUCCESSFUL boot after graceful-teardown
  fix): capture + `serve_full_r48h.log` (ZERO store lines — nolmc
  confirmed), boot out
- `r48j/` (gate10, instrumented probe): `adapter_p65v2.py` (deferral +
  `ARC_P65_ROW` row logger), `serve_full_r48j.log` — **ROWLOG_LINES=0**
  with 44×1664 + 106×832 stores (the wrong-file proof), capture, boot,
  `p65_gate10.sh`, `gate10.log`; probe2 image `…-p65probe2`
- `r48k/` (gate11, fix LIVE, ungated): `adapter_lmc_p65.py` (installed
  lmcint + hunk + `ARC_P65_DEFER` warning), `serve_full_r48k.log`
  (DEFER_ROWS=64; stores: 220×832 + 2×31616 + 2×63232 + 2×4992 +
  2×7488, **1664-class 0**), capture, run7n (300.6/84.4), boot,
  `p65_gate11.sh`, `gate11.log`; probe3 image `…-p65probe3`
- `r48l/` (gate12, GATED fix validation): `adapter_lmc_p65_gated.py`
  (= repo copy hunk + probe warning; MIN_TOKENS gate),
  `serve_full_r48l.log` (38 defer rows all prompt_len=63796; stores:
  230×832 + 48×1664 + 2×63232), capture (U32k 22.88 / U64k 52.25),
  run7n (300.05/84.44), boot, `p65_gate12.sh`, `gate12.log`; probe4
  image `…-p65probe4`. Rig-side derived scripts:
  `serve_arc_b70_defer.sh` (thr2048+DEFER) and
  `serve_arc_b70_defer2.sh` (thr2048+DEFER+MIN_TOKENS=40960).
- Repo: `vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py` carries the
  final gated hunk (uncommitted at bank time — ships via the p65 kit
  patch into the vllm_gaudi wheel; the lmcache wheel is NOT touched by
  this fix).

