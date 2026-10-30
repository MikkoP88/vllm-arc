# p42 — lookup-elide (skip LMCache lookup when engine prefix covers span)

## What

`VLLM_ARC_LOOKUP_ELIDE` — an **INT token count, not a bool; default 0 =
OFF (opt-in only)** — in
`vllm_gaudi/xpu/patches/lmcache_hma_connector.py:235`; port of D
champion's extyield2 / `IEXP_LOOKUP_MEMO=1` (§7 ENV map row; provenance
comment in `vllm_gaudi/envs.py`).

Mechanism: when the engine's own prefix cache already covers the request
span AND the residual is ≤ the knob's token count, both ZMQ lookups are
skipped — the retrieve path would return nothing the engine doesn't
already have. This is why D's certified matrix shows `Retrieved=0` in
steady state while Stored>0 (App E note): GPU prefix cache absorbed
repeats + elide skipped the lookups. The churn bench (lowered
`LMCACHE_MAX_LOCAL_CPU_SIZE`, forced eviction) exists precisely to prove
`Retrieved>0` when the external pool is the only copy.

**r31 disposition (default flipped 4096→0, 2026-10-28)**: elide was scoped
when every warm answer was provably 0 (APC-only world) and the round trips
were pure loss; with p40d (torch-baseline KV transfer → real attn stores)
+ p41c (live-site boundary snaps → real joint loads) the same faces now
claim real bytes — residual-1024 faces would elide and FORFEIT the load.
The guard stays as an explicit opt-in. **The certified serve line never
sets it — elide is OFF in every certified posture since r31.**

Companion knob `VLLM_ARC_LOOKUP_MEMO` (default 1 = ON, async-mode only,
kill-switch `=0`): per-rid post-clamp hit-memo replay of the resolved
depth without re-paying the round trips — the half of D's extyield2 that
remains profitable in our posture.

## Test

Churn cell: `Retrieved>0` (evicted spans must come back from the external
pool); integrity 5/5 unchanged; no regression on run7n faces.

## Validation

- r30/r31: measured with elide ON (4096) then dispositioned to opt-in
  after the p40d/p41c live-load findings (connector header records both).
- r42/r43/r44 certified legs: battery green with the knob at its default
  (OFF — full lookups, memo ON).
- r45 audit: consumed-here confirmation + citation recorded
  (`results/2026-10-30-p5-r45-env-log-audit/` evidence C/E). r45g
  deep-audit: registry/docs default corrected to the code truth (int 0 =
  OFF); the earlier "default true" claim in envs.py/docs was a
  bool-declaration artifact, not the shipped behavior.
