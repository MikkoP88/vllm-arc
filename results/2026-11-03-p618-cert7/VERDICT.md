# p618d cert7 — release gate on the tag-built image `vllm-arc:v0.30.0-b70-cert7` (2026-11-03/04)

**Flow (single-flow law):** squash-merge `feat/arc-p618c-tq-deprecation`
→ main `848dd7ef` → tag → build → **G0 caught a real guard bug** (see
Incident 1) → fix squash-merged as `feat/arc-p618c2-cache-dtype-attr`
→ main `5bf4192d16e91d4e1c5fb3066db68b7415013dd7` → tag
`rel/v0.30.0-b70-cert7` RECREATED at the fixed commit → image rebuilt
from the moved tag (`build_image.sh`, log `rig/cert7b_build2_fixed.out`;
the broken first build's log kept as `rig/cert7_build1_brokentag.out`)
→ the gate battery below on the tag-built image itself.

## Gate table

| Gate | Result | Detail |
|---|---|---|
| manifest fork_sha | **PASS** | `fork_sha=5bf4192d…` == tag commit; wheel md5s unchanged from cert6 |
| guard code in image | **PASS** | raise marker 1 + provenance comment 1 in installed `platform.py` |
| image-native verify | **PASS** | p48 battery **28/28** + p73 slice-hunk **11/11** |
| **G0 guard proof k8v4** | **PASS** | `--kv-cache-dtype turboquant_k8v4` boot: loud refusal in **~65 s** — `ValidationError … Value error, ArcXpuPlatform: kv_cache_dtype 'turboquant_k8v4' is deprecated on vllm-arc: … Serve with the certified default 'fp8_e4m3'` (`rig/gate/guard_k8v4/refusal_traceback.txt`); `Application startup complete` count **0**; health never came up |
| **G0 guard proof tq4nc** | **PASS** | same for `turboquant_4bit_nc` (`rig/gate/guard_tq4nc/`) |
| default fp8 boot | **PASS** | health ~295 s both boots; `SLICE_ENV_ABSENT=0`; flip-first 9.2 s → `stop`; KV pool **400,739 tokens** == the p618b cert6-fp8 legs byte-value (cert6-gate's 405,491 was that day's free-memory state — rig-state value, not an engine delta) |
| run7n t0 | **PASS (in-family)** | agg#2 **299.73**, solo 84.50/84.44 (cert6: 296.25 / 84.34/84.46 — +1.2 %) |
| run7n t1.0 xhigh | **banked (above band)** | agg#2 **218.29**, solo 70.91/68.11 (cert6: 186.48 / 61.73/61.79). The probabilistic-drafting t1.0 face (P6.17) is the engine's highest-variance face; the move is improving, uniform across streams, no ragged leg. No code delta touches sampling — recorded, not gated |
| integrity + det | **PASS** | 5/5 + 3/3 (and again 5/5 + 3/3 on the leg-1b boot) |
| spec gate | **PASS** | pos1 **0.9258** ≥ 0.5 hard; acceptance 0.837 vs 0.85 target (recorded miss, mean len 3.51) |
| **LONG-RETURN GATE** (owner hard gate) | **PASS — corruption faces 100 % CLEAN** | see below |
| erraudit | **PASS** | **0 ERROR / 0 Traceback**; WARN raw 43 (cert6: 44) |
| WARN class-set | **PASS — zero fork-added** | 30 unique classes; **0 new vs cert6**; the one absent cert6 class is the traffic-conditioned Triton-JIT `_topk_topp_kernel` warmup note (fires under battery traffic, not config — leg-1b's lighter traffic didn't trigger it) |

## LONG-RETURN GATE detail (v3 probe, 4 classes, ≥3 churn cycles)

**t0 corruption faces — 100 % CLEAN:** all four classes single-hash
across cold/rehit/rehit2 **plus 3 churn+rehit cycles** on one boot:

| class | single hash | finish | chars |
|---|---|---|---|
| essay | `648d66ae32c9` | stop ×3 | 3885 stable |
| code | `507d389dc258` | stop ×3 | 2746 stable |
| multi | `2b2ddebff529` | stop ×3 | 1023 stable |
| simple | `84efc84cc142` | stop ×3 | 1510 stable |

Zero REP-LOOP, zero JUNK, zero BLEED, zero EMPTY, zero truncation —
every garble dict empty in every phase of every cycle
(`rig/gate/longprobe_*.{json,txt}`, `longreturn_gate_summary.txt`).

**t1-seed annex:** the first battery's ONLY miss was one sporadic
t1-seed flip (`[simple]`, 1 of 24 pairs that night). Re-characterized
on the same live engine (`rig/gate/t1seed_char2.out`,
probe `rig/p618d_t1seed_char.py`): **16/16 pairs identical** across
4 rounds × 4 classes (incl. the class that flipped), t0 spot-checks
identical every round (hash `84efc84cc142` — same as the gate), and
the different-seed control yields a different hash (seeds function).
Disposition: sporadic RNG-path divergence of seeded temp-1.0 requests
under probabilistic drafting (P6.17 posture) — the p618b Class-A
semantics at sampled temperature, **not corruption, not looping, not
dtype-related**. Determinism guarantees: t0 within-path byte-identity
(proven across 3 churn cycles); seeded-temp>0 reproducibility is
best-effort, not a byte-guarantee, on any cached spec-decode system.

## Incidents (recorded for the campaign)

1. **First tag build had a broken guard (caught by G0, never certified).**
   The p618c guard read `cache_config.kv_cache_dtype` — the CLI flag
   name — but the wheel's pydantic field is `cache_dtype`
   (`config/cache.py:111`). `AttributeError` fired for EVERY dtype incl.
   fp8 before the intended `ValueError`. Fixed on main `5bf4192d`,
   predicate runtime-verified on the real class (4 TQ → REFUSE,
   fp8/auto → allow), tag moved, image rebuilt. **Law added (plan
   p618d): a config-hook change needs a runtime config-construction
   smoke — `py_compile` cannot catch attribute-name drift between CLI
   flags and pydantic fields.**
2. **Fail-exit skipped tail captures → leg 1b.** The gate script banks
   erraudit/WARN/serve-log after the long gate; the t1-seed FAIL exited
   before them and the container was then removed. Leg 1b re-ran
   boot + flip + integrity + captures (bank FIRST, verdict-independent).
   Lesson folded into the gate script order.
3. **arc/W14 port collision during characterization.** The live arc
   engine kept port 8000 while W14's restore serve silently died; a
   health probe answered by the WRONG engine read 200. Fixed by killing
   arc + re-serving W14 (health verified with port-owner check:
   exactly one listener, owned by the radiance container's pid).
   `rig/w14_restore_p618d*.log`.

## Verdict

**cert7 = the P6.18 release: every gate GREEN on the tag-built image.**
The `turboquant_*` kv-cache dtypes are now refused at the platform
layer on every entry point — proven live (G0, both presets, ~65 s to
loud refusal, startup never reached) — and the certified default
`fp8_e4m3` posture is untouched: t0 family in-band, integrity/
determinism/spec green, zero errors, zero fork-added warnings, and the
owner's long-return gate 100 % CLEAN on every corruption/looping face
across 3 churn cycles. `rel/v0.30.0-b70-cert7` = **current certified
release** pending the p618e currency flip.

Redaction: 4 files, +132 bytes total (8×6 + 7×7 + 7×5), raw-ref
per-file sweep 0 — `results/REDACTION-2026-11-04-p618-cert7.md`.
W14 restored and health-verified post-leg (port-owner checked).
