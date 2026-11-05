# P6.23 — docs-currency audit (owner directive 2026-11-05, task #70)

Directive: "deep analyze all documents so all information is based on the
latest certified image and is clean from old information, containing only
valid information." Scope: the six living docs (README,
CONFIGURATION/PERFORMANCE/TESTING/MIGRATION-arc.md, envs.md) audited in
full against the current certified truth — **cert9
(`rel/v0.30.0-b70-cert9`) remains the certified release; P6.21/P6.22
added no image (analysis lanes, closed negative)** — plus mechanical
recounts and a paste-block byte-diff. Precedent: the P6.19f/P6.20f
docs flips; this pass is the post-P6.22 truth alignment.

Method: full read of every doc (no spot-checks), then three mechanical
checks — (a) evidence-dir/VERDICT recount vs the banked claims, (b)
grep count of live `VLLM_ARC_*` code consumers vs the envs.md naming-map
claim, (c) byte-diff of the README serve block against CONFIGURATION
Step 2 (README:183 claims byte-identity; CONFIGURATION is canonical).
Config-parity standard: docs claiming validated configs must byte-diff
clean; no shorthand in paste-blocks; verify counts.

## Findings (F1–F12, all fixed in this item)

| # | Doc : site | Stale claim | Fix | Basis |
|---|---|---|---|---|
| F1 | README:8 status bullet | "P0–P6.20 complete"; P6.20 tail "kernel-level compute is the only remaining lever — AR 30.4 %/FA 20.3 %" | status → **P0–P6.22 complete**; tail extended with the P6.21/P6.22 negative closes (0.33× ESIMD, 2.06× AR) | P6.21a/b/c + P6.22 evidence dirs; commit 5d40c6e5 |
| F2 | README:9 lanes bullet | "Open improvement lanes … ESIMD dense-FA kernel project r67b re-scope" + "TP2 allreduce path question is closed — r71" | rewritten: **ledger CLOSED** — every slice dispositioned (GEMM/AR/FA/GDN/host); remaining routes (quantized AR, AR/compute overlap) are owner-naming-required, not open lanes; r71 pointer kept as transport-clean evidence | P6.21c + P6.22 verdicts; r83 device-saturation |
| F3 | README:287 | "108 evidence dirs, 101 VERDICTs" | **113 dirs** / 101 VERDICTs (recount: `ls results/ \| grep -c '^2'` = 113 ≡ `git ls-tree -d HEAD:results` = 113; `find -iname 'VERDICT*.md'` = 101, unchanged) | mechanical, 2026-11-05 |
| F4 | README serve block | README:183 claims "byte-identical to CONFIGURATION-arc.md Step 2" but two comment lines differed (`CCL_ZE_IPC_EXCHANGE` comment; `LMCACHE_TRACK_USAGE` 1-line vs 4-line) | README aligned to canonical CONFIGURATION wording; post-fix diff of the script body (#!/bin/bash → serve_full.log): **58/58 lines byte-identical** | config-parity standard; diff banked below |
| F5 | README "What this fork adds" | no Telemetry row despite step_trace (P6.20) + ar_trace (P6.22) instruments shipping in-tree | row added (silent-off diagnostics class, registry-exempt) | `vllm_gaudi/xpu/patches/{step_trace,ar_trace}.py` |
| F6 | TESTING:232 | "108 dirs / 101 VERDICTs" | → 113 / 101 | mechanical |
| F7 | TESTING validation record | table ended at the P6.20 row | four rows appended: P6.21a (ar_quant negative), P6.21b (FA ladder), P6.21c (ESIMD v2 negative), P6.22 (AR lane negative + census instrument, no cert10) | phase commits 0bb39f49…5d40c6e5 |
| F8 | MIGRATION §6 evidence table | ended at the P6.20 row | two rows appended (P6.21 lanes, P6.22 lane) — both marked no-engine-change; header unchanged: cert9 still current | same |
| F9 | envs.md:155 naming map | "(**10 live vars** + 3 dropped + 1 dead)" | → **15 live vars** with composition named (r45g's 10 + p619 ×3 + p620 ×2), grep-verified against code consumers | `grep -rhoE 'VLLM_ARC_[A-Z0-9_]+' vllm_gaudi/ tools/serve_arc_b70.sh \| sort -u` → 15 consumed names (+3 dropped +1 dead mentioned only) |
| F10 | PERFORMANCE cert9 prefill section | "the residual U32k/U64k gap is kernel compute … the remaining lever — see the P6.20 record" | extended with the P6.21c/P6.22 negative dispositions; "the prefill device ledger is fully dispositioned" | P6.21c/P6.22 evidence dirs |
| F11 | PERFORMANCE r67b consequences | "ESIMD FA-kernel rung … open as a kernel project"; "TP2 allreduce family (~30 %) is the largest unattacked slice" | historical analysis text kept as decision record; both forward claims annotated with their measured closes (P6.21c 0.33×/ceiling 0.83×; P6.22 2.06×, dominant class 76,677,120 B = 88 % of family, 27.3 % of the r94 window) | P6.21c/P6.22 evidence dirs |
| F12 | CONFIGURATION knob table | the "recommended settings" summary table (added r45g-era) lacked all five p619/p620 knobs (covered only in prose sections) | five rows added: `VLLM_ARC_TQ_SPEC_FIX`, `VLLM_ARC_LMC_ASYNC_STORE`, `VLLM_ARC_TQ_VERIFY_MODE`/`VLLM_ARC_TQ_MQ_SPLITS`, `VLLM_ARC_LMC_EVSYNC` | envs.md truth tables (already complete); p619/p620 VERDICTs |

## Count-drift disclosure

The "108 evidence dirs" figure was itself stale at the P6.20f banking:
the p620f squash commit (`14b6f5b4`) already carried **109** tracked
results dirs (`git ls-tree -d 14b6f5b4:results | grep -c '^2'` = 109) —
the 108 was carried forward without recount. Since then p621a/p621b/
p621c/p622 each banked one dir: 109 + 4 = **113 = the verified current
count** (working tree ≡ `git ls-tree -d HEAD:results`). VERDICT count is
methodology-stable (VERDICT*.md at depth 2): 101 before, 101 after —
the P6.21/P6.22 lanes bank READMEs, not VERDICT files.

## Deliberately NOT changed (truth-preservation rulings)

- **cert9-image docs correctly exclude `ar_trace`**: the P6.22 instrument
  lives on main but NOT in the cert9 image (tag predates it); docs
  describing the certified image (TESTING suite 9's "18 hook installs",
  PERFORMANCE cert9 rows) correctly describe the image, not the tree.
  It rides the next certified image. envs.md documents the code control
  surface, so it carries ARC_AR_TRACE — correct as-is.
- **PERFORMANCE provenance law intact**: the current record is
  cert6-native prob-lane measurement re-verified at the cert7/8/9 gates —
  that is the comparator law (history never mixes into record tables),
  not staleness. All history blocks remain image/posture-labeled.
- **MIGRATION header** ("most recently cert9 (P6.20)") stays — no image
  since; P6.21/P6.22 rows explicitly say so.
- TESTING suite count (14), PERFORMANCE posture laws, envs.md certified
  25-var chain: verified current, byte-checked against the serve block.

## Gates

- `tools/leak_sweep.sh` → exit 0 (no new addresses; this pack adds none).
- `tools/check_currency.sh` → exit 0 (all docs still point at cert9 —
  correct: no new image; docs-only item).
- Serve-block parity: README ≡ CONFIGURATION Step 2 script body,
  58/58 lines byte-identical (diff empty).
- No rig window, no image build, no W14 interaction (docs-only item).

## Reproduce

```bash
ls results/ | grep -c '^2'                       # 113
find results -maxdepth 2 -iname 'VERDICT*.md' | wc -l   # 101
grep -rhoE 'VLLM_ARC_[A-Z0-9_]+' vllm_gaudi/ tools/serve_arc_b70.sh | sort -u
awk '/^#!\/bin\/bash/{f=1} f{print} /serve_full\.log 2>&1/{if(f)exit}' README.md  > /tmp/a
awk '/^#!\/bin\/bash/{f=1} f{print} /serve_full\.log 2>&1/{if(f)exit}' docs/CONFIGURATION-arc.md > /tmp/b
diff /tmp/a /tmp/b                                # empty
```
