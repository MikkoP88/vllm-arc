# VERDICT — r45 env/log audit (§10 P5 row) — GREEN, rig-free leg

The P5 "env/log audit (no markers, no IEXP_*)" gate, assessed 2026-10-30 on
the certified tree (feat/arc-p60-p5-hardening @ ac4db26d content = image
p45b). All sweeps, classifications, and the 24/24 env-citation table in
[evidence.txt](evidence.txt) (sections A–G). Zero rig boots; W14 stayed
serving throughout (health 200 verified before the leg).

## Gate disposition

| §10 P5 audit item | disposition | evidence |
|---|---|---|
| no markers in tracked tree | **PASS** — only hits are upstream's own `hpu_modelopt.py` bytes (never edited; fork law) | evidence A |
| no markers in commit bodies | **PASS** — 19/19 commits main..HEAD marker-free, no trailers (App I #8 law) | evidence B |
| no `IEXP_*` / legacy `[iexp026s]`-class markers | **PASS** — IEXP_ appears only as 3 donor-provenance comments in `envs.py` beside live `VLLM_ARC_*` names (law #6 requires the citation); no legacy log markers in xpu source | evidence C |
| source debris (prints/TODO/debug) | **PASS** — every project-own print is boot-once or first-3-capped visibility (p45 design; r31 logger law); all TODOs live in the byte-pinned C-donor `lmcint/vllm_v1_adapter.py` | evidence D |
| env chain fully cited (no-unevidenced law) | **PASS — 24/24 serve-line vars + 2 docker-run vars cited** (§7 ENV map, App E certified set, App F/G champion boot registry, or per-leg law); CLI posture chain cited leg-by-leg | evidence E |
| log surfaces of record | **PASS** — r42 hook_audit 7/7 zero-FAILED one-line visibility; serve_keylines posture-only; r43/r44 cells clean, timers-off | evidence G |

## Findings (all zero image impact — none set in the certified line)

- **F1** `VLLM_ARC_YIELD_TOKENS` declared, consumed nowhere (p51 never
  built; §7 row stale).
- **F2** `VLLM_ARC_ASYNC` declared, consumed nowhere; comment promises a
  p50 reval that CLOSED at P2 (r22a was the async A/B; 0.30 core default
  governs).
- **F3** `VLLM_ARC_COMPACT_GDN` is a dead twin of the LIVE
  `VLLM_COMPACT_GDN` (auto-set `hpu_model_runner.py:1410`, consumed
  `platform.py:501`).

Disposition: **p61 env-normalization drops F1–F3** (no-unevidenced law:
unevidenced knobs don't ship). Recorded, not touched — this is an audit
leg; code changes get their own leg.

## P5 row after r45

| P5 item | state |
|---|---|
| env/log audit | **DONE (this leg)** |
| docs/envs.md | **DONE** — generated this leg ([../../docs/envs.md](../../docs/envs.md)) |
| MIGRATION addendum | **DONE** — [../../docs/MIGRATION-arc.md](../../docs/MIGRATION-arc.md) |
| write-up + bank | draft = this dir + r42/r43/r44 VERDICTs; final close-out writes after soak |
| 8 h soak zero errors zero drift | **REMAINING** — r46, day-window rig leg (overnight law: rig stays W14 overnight; script prepped) |
| squash-merge on §10 gates | after r46 green |
