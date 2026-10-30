# p41 — hybrid-GDN mamba state caching (worker glue + snap law)

## What

Mamba (conv/ssm) state store/retrieve through LMCache — upstream lmcache
has NOTHING like it; D/A lineage converged on one delta generation
(v136 glue re-anchored per stack; dual-donor md5 evidence in the p41
commit).

- **Worker glue** `vllm_gaudi/xpu/patches/mamba_lmcache_glue.py` (e02
  half): registry symbols `_LMC_BOUNDARY_REGISTRY` (per-req 8 / global FIFO
  48) + `_LMC_SEED_OVERRIDE` — the arc1 adapter hard-imports BOTH from
  `vllm.v1.worker.mamba_utils` (adapter lines 1741/2049); without this
  module the store path ImportErrors on the first finished request and the
  joint lookup clamps every hit to 0. v136 verbatim semantics re-anchored
  onto the 0.30 fused-kernel body: seed consume/write at loop top, v135
  fp16 seed repair before a carry stages (LIVE for our float16 mamba cache;
  knob `VLLM_MAMBA_FP16_SEED_REPAIR=0`), boundary-row collection in the
  carry branch, snap loop after the copies, one-shot seed pops on
  finished/preempted/resumed.
- **Snap-site law** (p41c, r32): snapshot host INSIDE `preprocess_mamba`'s
  carry branch — when a carry stages, the deepest full boundary block
  `floor(computed/bs)*bs` is final and visible; same-B re-snaps replace.
  Delivered as `vllm_gaudi/xpu/patches/mamba_hybrid_snap_glue.py` (model
  states side); exact-boundary snap fixed integrity 0/5 → 5/5.
- **r34 falsification law (do-not-redo)**: `disable_eagle_block_drop` A/B
  improved warm faces (N1 1390→836 ms) but broke 0-drift integrity
  deterministically — outputs became engine-history-dependent via
  spec-draft contamination of retained trailing blocks (upstream's warned
  class). Reverted; control returned 5/5. The spec-protective drop stays.
- Not carried (REVAL ledger in p41 commit): v138c sanitize-carry
  (fallback-path-only in both donors; our posture is fused align),
  LMCDbg*/v52f debug telemetry.

## Test

Integrity 5/5 cold-vs-rehit after the full battery (zero drift under load —
`r42_integrity.py`); boundary telemetry present; align posture REQUIRED
(the snap law assumes `--mamba-cache-mode align`; App F posture-pin law).

## Validation

- r31: external-load chain proven live end-to-end; root cause #1 (registry
  import) found+fixed (b6835611).
- r32: root cause #2 (exact-boundary snap) fixed — 0/5 → 5/5 (c954c29d).
- r42/r43: integrity 5/5 + determinism 3/3 on the certified image, both k4
  and k5 postures.
