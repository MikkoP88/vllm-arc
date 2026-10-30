# r35 — scheduler tail-split: v3 general delta FALSIFIED (composition-coherence law), v4 scoped delta BANKED (1k-face warm 9.2 s → 545 ms GATE PASS)

Date: 2026-10-29 (rig legs 17:55–18:50). Posture: r34-reverted product
(p2 + p40/p41/p41b/p41c + lmcint kit + r33 TWO_HS) + this leg's
`vllm_gaudi/xpu/patches/sched_tail_split.py` via a sitecustomize hook on
`vllm.v1.core.sched.scheduler` (tuple `arc_r35_patch`, knob
`VLLM_ARC_R35_TAIL_SPLIT`, default on).

## Hypothesis chain (v1 → v3, all superseded — see design-evolution.md)

Warm ≤750 ms on the 8k face needs the joint hit at 8064 (residual 128);
r33 proved external loads byte-true at 7168, r34 proved the pop must stay.
v3 = plan the mamba-aligned chunk end WITHOUT the eagle backoff (split
only; lookup pop untouched) so chunk1 ends 8064 aligned → the p41c glue
snaps the boundary (materialized, lawful) and the adapter stores
[0,8064) externally → warms load 8064 + 128 residual.

## v3 result — store side perfect, integrity broken (FALSIFIED)

Probe boot (17:55): everything designed happened. Split `new=8064` then
`new=128`; `p41c-snap ncc=8064 B=8064 idx=8` (mid-request aligned end);
`Stored 8064 out of total 8064` (0.1307 GB) + `Stored 896` (mamba obj);
every N1 warm `Retrieved 8064` (≈210 ms) + `Retrieved 896` (the seed obj)
+ `p41c-seedmd5 postwrite==seed_bytes` + carry `src_col=8` — the full
external 8064 path worked and was sha-stable (N1 5 warms 7eb702…).

But: golden **3/5**, and the timeline decode pinned why:

- Golden warms ride LOCAL paths (zero Retrieved) — local pop-hit 7168 +
  recompute [7168,8192) — while the colds now chunk [0,8064)+[8064,8192).
  integrity[0]: cold `27be8461…` (a NEVER-seen value), warm `8898f039…`
  = the r33 CANONICAL. integrity[3]: cold `d9f601923574` = the r34
  FIX-BOOT value (same 8064 chunking!), warm `a15739023154`.
- N4 warm[0]: the cold's WFS store landed at 18:03:21, warm[0] ran
  18:03:15–22 → external missed → FIRST-EVER pure-local N4 warm
  (6272-hit + 1792 recompute; the `Stored 1792` line is its store) →
  sha `7eb702…` = N1's certified sha — a composition the cold never ran.

## The composition-coherence law (this leg's core finding)

**A warm resume is byte-coherent only at depths that are cold chunk-end
states; the drop-backoff pins chunk ends to popped depth; chunking deeper
changes the COLD's own numeric path, and every warm that does not follow
the new composition (local pop-hit recomputes) flips marginally-sensitive
prompts.** r33's 5/5 worked because cold chunks ended exactly at 7168 =
what local hits reuse (warm recompute ≡ cold chunk2, same carried state,
bitwise). Consequences:

- The 8064-depth warm on 8k prompts is STRUCTURALLY UNREACHABLE without
  breaking 0-drift: the state@8064 the upstream cold computes is
  mid-chunk (p41c exact-boundary law — never a discrete checkpoint), and
  producing it requires moving the cold chunk end (v3, r34 flag) which
  breaks every other path's composition. This closes the r34 "open" item
  (attn span [7168,8064) via external pool) as impossible-under-integrity.
- r34's break and this one share the gate symptom but differ in
  mechanism: r34 = history-dependent bytes (pop-off GPU resume, churn);
  r35-v3 = deterministic composition mismatch. Same lesson: the pop and
  its backoff are one coherent design.

## Control (E-law): knob-off boot, same battery

`VLLM_ARC_R35_TAIL_SPLIT=0` fresh boot: N4-then-golden = **INTEGRITY
5/5**, every canonical sha restored ([0] 8898f039, [2] 8fa239e9, [3]
a1573902 — within-run consistent). Causality proven.

## v4 — the one coherent exception (BANKED)

Apply the no-backoff split ONLY when upstream banks NOTHING:
`deepest_full = n - n%bs > 0` and `max(deepest_full - bs, 0) == 0`
(token counts in (896, 1792] at bs=896). There the delta ADDS the
deepest-full-block chunk end instead of moving one: cold chunks
[0,896)+[896,n) and banks 896 (snap + store at the aligned end); warm =
external load 896 + residual [896,n) whose composition EQUALS the cold's
chunk2 from the same frozen state → within-run coherence holds.

Probe-boot proof: `r35probe split` fired 7× — ALL n=1024 (the 1k cell;
every 8k request rode the passthrough). 28 B=896 snaps, 30
Retrieved-896 lines. N3 sha `8a5edab2…` stable cold + 5 warms.
Known accepted trade: inside the class the cold composition changes vs
upstream's single exempt chunk (within-run determinism measured; the
cross-posture marginal class is the documented cross-boot flip class).

## E4 re-gate battery (clean boot 18:36, first-traffic, timers-off)

| cell | integrity | warm | note |
|---|---|---|---|
| N4 8064 | 9c1623f0… | 1297.9–1346.0 ms | r33/r34 parity (Retrieved 7168-class) |
| Golden 5×8k | **5/5** | 2.1–2.3 s | zero drift |
| N2 7168 | 2b4c342f… | med 1260.5 ms | parity (r34: 1257.9) |
| N5 7680/8960 | cc7e4412…/7eb70257… | 1287.4 / 1256.4–1349.4 | parity |
| WARM5 8192 | 8a5edab2… within-run | med 1025.5 ms | parity (r34: 1021.6); ≤750 FAIL → r36 |
| N1 8192 | 7eb70257… | med 1037.7 ms | parity (r34: 1033.1); ≤750 FAIL → r36 |
| **N3 1024** | **8a5edab2…** | **med 544.7 ms ≤750 PASS** | **was ~9200 (nothing banked)** |

Probe stripped at E4 (ARC_SNAP_PROBE out of r30_serve.sh; the r35probe
print is env-gated, silent off).

## Follow-ups (r36 lane)

The 8k warm ≤750 gate is now bounded by the coherence law: depth is
capped at 7168 (local, 1033–1038 ms) — the remaining lever is overhead,
not depth. Prime suspect: the p41-leg WARM5 was **755.1 ms** and r33's
adapter changes (pool-view + seam transforms) took it to ~1021 — a
~266 ms lookup/lag regression on the warm path. r36 = phase-timer
capture of the warm path (lookup / WAITING_FOR_REMOTE_KVS window /
residual forward / output sync), pin the dominant term, then slim the
adapter's per-request work. Native SYCL TWO_HS load (0.63→2+ GB/s) is
the secondary lane for external-path faces.
