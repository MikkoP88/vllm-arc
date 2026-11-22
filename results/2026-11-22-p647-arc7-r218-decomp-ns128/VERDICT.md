# r218 — the ns128-conditioned lane table: the split win decomposed (union −10.7, floor −2.7); remaining gap 14.3 = floor 5.6 + union 8.7; kv_tile-64 is the next surgery

**Date**: 2026-11-22 · **Leg**: r218 (arc7 Phase-8 decomposition re-price) ·
**Stack**: the r217 wheel (md5 `9ded4fb67e15ba092aac607d58db1b72`) +
`vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay; every boot pinned
`ARC_MXKV_NSPLIT=128` · **Run**: `r218_decomp.log` 07:00–07:26 UTC (four
boots, c32768 n1, env severance per r215; zero code changes)

## Verdict

**GATE_PASS — the r215 table re-priced at the banked posture: the
splits win (70.2 → 56.8) attacked the K/V union hardest (−10.7) and the
floor (−2.7); the anchor reproduces the posture at −0.02% (third ns128
full-boot sample: 56.7 / 56.8 / 56.8).**

### Lane table @c32768 n1 (ms/tok-step; r215 auto-priced in parens)

| pool | @auto-40 | @ns128 | Δ |
|---|---|---|---|
| full | 70.2 | **56.8** | −13.4 |
| floor (v8, const-KV skeleton) | 50.8 | **48.1** | −2.7 |
| K lane (v2−v8) | +14.4 | +7.2 | −7.2 |
| V lane (v6−v8) | +17.8 | +8.0 | −9.8 |
| K/V union (full−v8) | +19.4 | **+8.7** | −10.7 |
| MX total (full−fp8 42.5) | +27.7 | **+14.3** | −13.4 |

Rows=148 on every cell (bucket shape held); chunks 52/102/174/192 =
acceptance collapse on the severed cells as designed (itl stays
per-step); 0 errors.

### Readings

1. **Both lanes halved under deeper splits** (K 14.4→7.2, V 17.8→8.0):
   the walk was parallelism-starved, not bandwidth-saturated — more
   WGs hide the per-tile latency, exactly the r216 mechanism note.
2. **Still massively subadditive** (K+V = 15.2 vs union 8.7,
   overlap 6.5): the r201 K/V interleave keeps doing real work; the
   union is a SHARED bandwidth/joint-schedule pool — lane-local
   surgery stays structurally dead.
3. **The floor dropped too** (50.8→48.1) despite the reduce volume
   GROWING with splits (128 vs ~40): the skeleton's launch/addressing
   side also parallelized. Floor above fp8: 5.6 (was 8.3).
4. **Remaining gap 14.3 = floor 5.6 + union 8.7.** A wider kv_tile
   (16→64) attacks BOTH — 4× fewer tile-iterations (floor: less
   per-tile overhead; union: wider loads = more outstanding memory
   requests per instruction). That is the r212c2 road and the next
   wheel surgery after r219.

### Dispatch follow-ups shipped with this leg

- r219 wheel edits staged on the rig tree (build in flight at bank
  time): `mx_get_num_splits` reduce multiplier 4→13 slots/head
  (auto lands ~130 at the bucket ≈ the validated 128 posture; the
  r210 4-slots knee was probe-priced — fourth sign-inversion-class
  instance), `policy_split_cap` 64→256, kernel ceiling 128→256 (tail
  probe ns192/ns256); backups `.r219edit` + `.r217bak` on rig.
- Overlay telemetry now logs the splits arg (`splits=<int|auto>`,
  `_ns` hoisted above the log block — py_compile cannot catch a
  NameError) — auto becomes observable behaviorally AND textually.

**Reverted**: nothing — env-instrument leg; probes read per-call in C++.
