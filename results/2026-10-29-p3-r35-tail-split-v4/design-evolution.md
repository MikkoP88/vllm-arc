# r35 design — finish-flush external warm path (v2, corrected by code read)

> **OUTCOME ADDENDUM (pinned after the legs):** v2/v3 were FALSIFIED and v4
> was BANKED — see VERDICT.md. The finish-snap of v2 was dead code twice
> over (preprocess_mamba never runs under MTP+align — ppm=0 probe; and the
> mid-chunk 8064 boundary never materializes — the p41c exact-boundary law,
> already probe-proven at r31). v3 (split without backoff) banked 8064
> perfectly on every side but broke golden 3/5 — the composition-coherence
> law (VERDICT) forbids chunk ends deeper than the pop. v4 = the same delta
> scoped to prompts where upstream banks nothing (the 1k face): N3 warm
> 9.2 s → 544.7 ms clean GATE PASS, everything else at parity, golden 5/5.
> The 8k warm ≤750 gate moves to the overhead lane (r36): p41-leg WARM5 was
> 755.1 ms; r33's adapter changes cost ~266 ms on the warm path.

Pinned 2026-10-29 after r34 + engine code reads (sources: image site-packages,
copied to .tmp/r35read/ and /root/build/r35read/). Supersedes the v1 sketch
at the bottom.

## Corrected mechanism (code-pinned)

- The eagle "block drop" is a LOOKUP-SIDE pop, not a hash removal:
  single_type_kv_cache_manager.py:618 "If eagle is enabled, drop the last
  matched block to force recompute the last block to get the required
  hidden states for eagle drafting head" (:827 `if drop_eagle_block and
  hit_length > 0`). The hash cache RETAINS every completed block
  (N1: blocks 0..8 hashed at finish). The warm's LOCAL match pops the last
  → local hit 7168 → residual 1024 → 1021 ms warm. This pop is structural
  for eagle-family resume and STAYS (r34 proved bypassing it breaks
  integrity — GPU-resident block-8 bytes under pool churn are not
  trustworthy; outputs became engine-history-dependent).
- CRITICAL r33 EVIDENCE: N4 warms loaded [0,8064) EXTERNALLY (LMCache
  `Retrieved 8064` ×6, byte-identical 4/4, golden 5/5 around them) —
  loading THROUGH the would-be-popped block via the EXTERNAL pool is
  certified byte-true. The r34 hazard was the GPU-APC path (zero Retrieved
  lines on the fix boot); the external CPU-pool copies are stable.
- Therefore the ≤750 warm path is: EXTERNAL load to the deepest boundary
  (8064 for N1, 896 for N3) + residual 128 + decode — GPU-local behavior
  untouched. Budget: load ~210 ms + residual ~90 ms + overhead ~300 ms
  (sched + spec verify + 54-58 ms output sync) ≈ 600-650 ms ≤ 750 ✓.
  Load-cost reduction (r33 follow-up, ~250 ms) is the buffer lane.

## The two store-side gaps (both at request finish)

1. MAMBA never banks the final partial chunk's deepest boundary: the glue
   snap fires only when a carry stages (next chunk's preprocess); the last
   chunk has no successor. Fix: finish-snap in the p41 glue —
   idx = floor(num_computed_total/896)-1, block j = state after (j+1)*896
   committed tokens (mid-chunk materialized; a finished request runs no
   later pass, so no nulling before the first post-finish read). N1 banks
   8064, N3 banks 896, N4 re-snaps 8064 (same-B replace, no-op).
   ORDERING CRUX: registry entry must land BEFORE the adapter's
   wait_for_save consumes the req's entries.
2. ATTN external store stops at the last CHUNK END (N1: "Stored 7168");
   block 8 [7168,8064) completes mid-final-chunk but is never stored.
   Fix: finish-flush in the lmcint adapter's store seam — store
   completed-unstored blocks at finish (committed prompt blocks only;
   the partial tail block never — same law as the mamba snap).

Lookup/load side: UNCHANGED (p41 joint lookup + p43a clamp; the connector
already prefers/supplements external as N4 cells proved).

## Build order

1. Read the lmcint adapter store seam (wait_for_save call site, finish
   hooks, request_finished_all_groups in lmcache_hma_connector.py) → pin
   the ordering for the mamba snap + the attn flush site.
2. Implement: glue finish-snap (vllm_gaudi/xpu/patches/mamba_lmcache_glue.py)
   + adapter finish-flush (patches/lmcint/vllm_v1_adapter.py, kit
   source-override law). One leg, two file-local changes, one mechanism.
3. Probe boot: expect N1 snap B=8064 at finish + attn Stored 8064 +
   Retrieved 8064 on warms; N3 banks 896. Gates: N1 ≤750, N3 ≤750,
   N4/N2/N5/golden/spec-pos1/survival battery, rig → W14.

## v1 sketch (superseded — kept for the record)

The first sketch proposed re-hashing block 8 into the GPU attn hash at
finish (engine-core patch). WRONG TARGET: the hash already has block 8;
the pop is on the match side, and bypassing the pop (r34) is what broke.
The v1 mamba finish-snap analysis (block j law, nulling window, lazy-snap
ordering problem) carries over unchanged into v2 gap 1.
