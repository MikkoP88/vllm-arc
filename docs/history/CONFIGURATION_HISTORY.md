# CONFIGURATION history — archived superseded content

Per-doc archive (main-currency law): everything here was superseded in
the living [`../CONFIGURATION-arc.md`](../CONFIGURATION-arc.md) and is
kept verbatim for evidence-chain continuity. Per-cert/per-phase
entries: [`../../CHANGELOG.md`](../../CHANGELOG.md).

## Stall-lever column provenance (era note, verbatim)

> Era note: the default and `MNBT 4096` columns are the cert6 P6.15
> matrix (`results/2026-11-02-p615-stall-analysis/`; KV row above
> re-verified natively on cert7, p618e — 400,739/427,666); the p65 column
> is the cert4/cert5-era record face (`results/2026-10-31-p65-cert4/`,
> `results/2026-11-01-p65-cert5/`). Both live levers were re-verified
> natively on cert7 (p618e, same harness): default TTFT 18.27/39.91 s +
> gaps p95 4001/5273 ms; MNBT 4096 TTFT 19.51/42.34 s + gaps p95
> 2052/2523 ms — all in-family
> (`results/2026-11-03-p618-cert7/PERF.md`).

## `LMC_STORE_SLICE_TOKENS` — the full experimental gate record (P6.13, gated 2026-11-01)

`LMC_STORE_SLICE_TOKENS=4096` — spreads the deferred one-shot final-span
store across ~19 bounded windows (aligned 3328 tok each at runtime chunk
1664), one per step, re-using the exact per-chunk store call shape; default
unset = byte-identical certified behavior. Gate
(`results/2026-11-01-p65-rung3-slice/`): all safety faces green (integrity
5/5 + determinism 3/3, run7n 300.48 / 84.4, U32k family unchanged, zero
fork-added WARN/ERROR), mechanism landed exactly (`63232`-class = 0 all
session; 38 = 2×19 windows; U64k TTFT 50.72 s / 48.79 s-cold vs the 52.25 s
bar; full-hit + mid-drain replay integrity clean at 1.74/1.82 s). NOT
adopted into the certified profile: the pre-registered ≤600 ms bystander
max-gap bar was mechanically missed (1598 ms) and is recorded as mis-set
(the r48e nolmc floor is 2347 ms with zero stores — chunk compute [viii]
is the floor; the corrected claim, store-side contribution removed, is
confirmed). Enabling it is a caller decision on the corrected bar.
Reproducible on the certified image since cert5 (the hunk ships in
`rel/v0.30.0-b70-cert5`, default-OFF): the cert5 release gate re-ran the
slice face on the tag-built image — U64k TTFT **50.77 s** (cert4 same-face
51.95), W_U64k_start max **1602 ms** vs cert4's 1997 (the one-shot tail
gone), `63232`-class **0** / `3328`×38 exact, U32k unchanged
(`results/2026-11-01-p65-cert5/`). The cert6 release gate re-ran the same
face in-family (U64k TTFT 51.66 s, `63232`-class 0, `3328` signature
exact — `results/2026-11-02-p614-cert6/`).

## TurboQuant corruption-era record (P6.18, verbatim)

> The P6.18 record (reproduction battery, roundtest, guard gate) stays
> banked at `results/2026-11-03-p618{b-repro,c-roundtest,-cert7}/` — its
> Class-B/Class-A split stands; its precision-ceiling ROOT-CAUSE
> conclusion was corrected by P6.19 (the roundtest exonerated the kernels,
> and the true spec-verify defects were found above them).

## KV-cache dtype identity provenance (cert8/cert9/cert10 gate chain, verbatim)

> the corruption bar (REP-LOOP/JUNK/BLEED/EMPTY, garble) is **zero across
> all five dtypes** on cert8 AND cert9, cold and warm, churn included
> (cert9 evidence: the r84 gate + r84b re-char — dominant identity hashes
> byte-identical across the two cert gates, so the P6.20 off-thread store
> changed no output distribution; cert10 carried the k8v4 leg —
> essay/code/simple hashes byte-identical to the stock cert9 k8v4 leg at
> the r99 gate, so the oneCCL overlay is absent from the TQ decision
> path).

## envs.md recount narratives (superseded header + naming-map text)

- Header (superseded): "Generated at P5 hardening (r45, 2026-10-30;
  serve-line identity updated at the r46t close-out) from
  `vllm_gaudi/envs.py` arc-lane markers + the certified serve line
  (`r43_serve_k5.sh`, spec depth 3 per the r46t recommendation — a
  serve-time CLI change). … (25 vars incl. `LMCACHE_TRACK_USAGE=false`,
  P6.3): from cert9 through cert23 every release added ZERO certified
  serve-line vars."
- Naming-map live-var recount (superseded): "**18 live vars** — the r45g
  audit's 10 + `VLLM_ARC_TQ_SPEC_FIX`/`VLLM_ARC_TQ_VERIFY_MODE`/
  `VLLM_ARC_TQ_MQ_SPLITS` (p619) + `VLLM_ARC_LMC_ASYNC_STORE`/
  `VLLM_ARC_LMC_EVSYNC` (p620) + `ARC_MXFP4_MODE` (p630) +
  `ARC_MXFP8_MODE` (p631) + `ARC_SKINNY_M` (p633); grep-verified against
  code consumers at the P6.23 docs audit + the 2026-11-08 P6.31 registry
  catch-up (the p619 trio and the p630 gate had gone live docs-only; the
  `vllm_gaudi/envs.py` registry now mirrors the docs) — + 3 dropped at
  r45d + 1 dead since r47b".
- Host-script var recount (superseded, P6.33 count 23; the P6.37 close-out
  added `ARC_KVDTYPE` → 24): "the base 11 plus the P6.28–P6.33 lane knobs
  … most as set-only pass-throughs … the 11th is `ARC_FP8`, the P6.24
  cert10 overlay pass-through … the 10th is `ARC_SPEC_METHOD`, the
  draft-sampling posture knob: P6.17 — unset = `probabilistic` = the
  CERTIFIED default …".
- `LMCACHE_MAX_LOCAL_CPU_SIZE` was listed `100` until the P6.37 close-out
  made 16 (the 64 GB-RAM host default: `--shm-size 16g`, 16 GB LMCache
  CPU tier) the documented certified value.
