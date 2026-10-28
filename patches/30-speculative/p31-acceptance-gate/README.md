# p31 — spec-acceptance gate (pos1 ≥ 0.5 hard, acceptance ≥ 0.85 target)

## What

- `tools/arc_spec_gate.py` — bench-side gate reading upstream-0.30-native
  spec telemetry:
  - `/metrics` counters: `vllm:spec_decode_num_{drafts,draft_tokens,
    accepted_tokens}_total` + `..._accepted_tokens_per_pos_total{position="i"}`
    (vllm/v1/spec_decode/metrics.py, verified in image
    vllm-arc:v0.30.0-b70-p12).
  - Engine-log window cross-check: last `SpecDecoding metrics: ...
    Per-position acceptance rate: p1, p2, ...` line (D-law lineage — rig
    `p2r3_leg_v1.sh:387-391`).
  - Gates: **pos1 = per_pos[0]/drafts ≥ 0.5 HARD** (exit 2 — the
    wrong-logits/mojibake family detector: D r1 signature 0.000x4 vs healthy
    0.914-0.954, no middle state ever observed, probe B vs probe D);
    **acceptance = accepted/draft ≥ 0.85 target** (superiority contract,
    champion 0.871; recorded by default, exit 4 under `--strict-target`
    at certification).
  - `--expect-off`: asserts spec counters ABSENT — non-spec leg posture
    purity.
  - Exit 3 = telemetry absent on a spec leg (D fail-law: absence is a
    telemetry break, not silence-by-design).
- `tools/arc_matrix.sh` — section 3b: every leg gated. Spec legs run the
  gate after bench (JSON row archived); non-spec legs run `--expect-off`
  (posture leak = leg FAIL). Boot line now takes `ARC_LEG_GRAPH/
  ARC_LEG_SPEC/ARC_LEG_LMC` pass-throughs for the P2+ posture matrix.

## CI note (interpretation recorded)

Plan asks for the gate "in bench + CI". This repo runs no external CI —
no-internet-publishing law; `.github/workflows/` are upstream's files,
never edited (fork law). The certification harness `arc_matrix.sh` IS the
CI surface here: it is the thing every leg and every phase gate runs
through, and it now hard-fails the leg on a pos1 miss.

## Validation

- Parser unit (rig, synthetic exposition with labels + HELP/TYPE noise):
  all 7 counters parsed, champion-shaped numbers verified
  (acceptance 0.87, pos1 0.95) — `PARSER_UNIT_OK`.
- Live expect-off against the running arc-test engine (eager non-spec):
  `SPEC_GATE expect-off PASS`, RC=0.
- Spec-path (load → deltas → pos1/acceptance + log window) validates at
  the first P2 spec leg; expected shape per D certified signature
  (pos1 ~0.95, avg draft 86-87.5%, mean length 4.45-5.0).
