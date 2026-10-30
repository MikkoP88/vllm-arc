# p42 — lookup-elide (skip LMCache lookup when engine prefix covers span)

## What

`VLLM_ARC_LOOKUP_ELIDE` (default **true**) in
`vllm_gaudi/xpu/patches/lmcache_hma_connector.py` — port of D champion's
extyield2 / `IEXP_LOOKUP_MEMO=1` (§7 ENV map row; provenance comment in
`vllm_gaudi/envs.py`).

Mechanism: when the engine's own prefix cache already covers the request
span, the external lookup is skipped — the retrieve path would return
nothing the engine doesn't already have. This is why D's certified matrix
shows `Retrieved=0` in steady state while Stored>0 (App E note): GPU
prefix cache absorbed repeats + elide skipped the lookups. The churn bench
(lowered `LMCACHE_MAX_LOCAL_CPU_SIZE`, forced eviction) exists precisely to
prove `Retrieved>0` when the external pool is the only copy.

## Test

Churn cell: `Retrieved>0` with elide ON (evicted spans must come back from
the external pool); integrity 5/5 unchanged; no regression on run7n faces
(elide only removes work the engine would redo).

## Validation

- Adopted with the p40 connector boot (r30+) — every certified leg since
  runs default-true; r42 battery green on the standalone image with the
  knob at default.
- r45 audit: consumed-here confirmation + citation recorded
  (`results/2026-10-30-p5-r45-env-log-audit/` evidence C/E).
