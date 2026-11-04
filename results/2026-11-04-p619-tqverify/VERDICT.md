# VERDICT — P6.19e cert8 release gate (2026-11-04)

Image `vllm-arc:v0.30.0-b70-cert8` (id `b212fbca59ec`), tag-built from
annotated tag `rel/v0.30.0-b70-cert8` on main `e47739b0` (the P6.19 fix
squash). Gate script `rig/p619e_gate.sh` (console `rig/gate.out`,
per-leg evidence `rig/gate/`), post-gate annexes `rig/p619e_annex.sh`
(console `rig/annex.out`, evidence `rig/repchar/`, `rig/mqannex2/`,
`rig/w14ctrl/`). Redaction: second pass +816
(`results/REDACTION-2026-11-04-p619-tqverify.md`).

## Gate table

| gate | verdict | evidence |
|---|---|---|
| [0] manifest + fix-in-image + guard-gone + p48/p73 image-native | **PASS** | `manifest.txt` fork_sha `e47739b0…` == tag; hook_entries=2 (`p619` entry live); zero `deprecated on vllm-arc` (P6.18 guard removed); `p48_verify.out` 28/28, `p73_verify.out` 11/11 |
| [1] fp8 posture (health/flip/integrity+det/run7n t0+t1.0/spec) | **PASS** | health ~295 s, KV pool 400,739 (byte-value cert7), SLICE_ENV_ABSENT=0, fix-engaged-on-fp8=0 (control: hooked module never imports), integrity 5/5 + det 3/3, run7n t0 agg#2 300.65 / solo 84.53-84.57 + t1.0 221.36 / solo 70.98-68.14, spec pos1 0.9258 hard-PASS (acceptance 0.837 vs 0.85 target — recorded, identical posture to cert7) |
| [2] fp8 long-return (full + 3 churn cycles) | t0 corruption lane **100 % CLEAN**; t1-seed content lane: 1 window-repetition flag `[code]` — **annex-dispositioned benign** (below) | 4 classes single-hash cold/rehit/rehit2 + all churn rehits: essay `648d66ae32c9`, code `507d389dc258`, multi `0627e973e7b1`, simple `84efc84cc142` (code/simple = the cert7 family values byte-exact) |
| [3] TQ all-dtype (owner bar: no corruption on ANY kv-cache dtype) | **PASS** — zero garble on every dtype × lane; two identity faces dispositioned (Class-A / 3-bit floor), neither corruption-class | k8v4 bar PASS (below); k3v4/q4nc/q3nc full longprobes below; posture line `v6 installed (active=True mode=split mq_kernel=True)` on every TQ boot |
| [4] mq annex (opt-in `VLLM_ARC_TQ_VERIFY_MODE=mq`) | **PASS after re-run** — gate FATAL was a script sed bug (incident 2); annex re-run: `mode=mq` engaged, flip stop 9.4 s, tps solo 53.83/54.31, **agg8 284.81** (1.21× split, in-family of t8g 289.23) | `rig/mqannex2/` |
| [5] W14 restore + verify | **PASS** (via EXIT trap after the [4] FATAL; generation-verified in the annex) | `rig/incidents/`, annex console `[C]` |

## TQ all-dtype long-return detail

| dtype | verdict | hashes (t0 lanes) | KV pool | notes |
|---|---|---|---|---|
| `turboquant_k8v4` (split, certified) | **bar PASS** (full + 3 churn cycles) | essay `6084223df0a5` · code `4065e97dc61c` · multi `{3bb45b51520b, f0dce5b55f4f, f978fd6b15e2}` · simple `fe080656ac71` | 519,972 (== t6p graphs byte-value) | multi = the t9 cross-boot pair + one adjacent coherent variant — Class-A identity face, garble-clean, stop-finish, stable chars throughout (r2 rehit drifted between the two known variants, r3 stable); tps 54.42/54.59 solo, agg8 235.97 (t6p 54.50/240.86 in-family) |
| `turboquant_k3v4_nc` | runner CORRUPTION_SIGNAL → **Class-A disposition** | essay `8ccfa3b73aa1`/`372efab9884d`/`8ccfa3b73aa1` · code `df99aacc45be`/`9d135c11263a`/`df99aacc45be` · multi `9ec8a5001cfd` ×3 · simple `cc082a03dea2` ×3 | 843,202 | cold + rehit2 **byte-identical to the t9 primaries on ALL FOUR classes** (cross-boot attractor stability intact); the single rehit1 round took the adjacent coherent variant (essay 4165/4134/4165 chars, code 2985/4835/2985), all garble-empty, stop-finish — the p618b-documented warm-resume face |
| `turboquant_4bit_nc` | **CLEAN** (all classes identical) | essay `56a2e47a4325` · code `ccef33edf9b1` · multi `ce38cfd6df8c` · simple `84efc84cc142` (cross-dtype family match with fp8 simple) | 748,549 | zero flags of any kind |
| `turboquant_3bit_nc` | runner CORRUPTION_SIGNAL → **3-bit precision-floor disposition** | essay `{f166666b80cb, b862a6d1c3bd, ef28cc2d495b}` · code `{ac0f3b6787ad, 518803c40182, 2c53475b05b4}` · multi `a3b5f407f146` ×3 · simple `e0414695aa10` ×3 | 963,764 | essay+code take a different coherent adjacent variant per round (all garble-empty, adjacent chars); NOT the p618b Class-B attractor set (those were cross-dtype byte-identical loops with junk) — this is the 8-level value-precision ceiling (P6.18c root cause; upstream's own table +20.59 % PPL on 3bit): outputs remain coherent English/code, multi+simple fully stable |

Post-hoc literal zero-garble sweep (`grep -cF` over every banked
longprobe `.out`, 15 files × 4 class lines): **15/15 OK** — every t0
lane on every dtype is `garble={'cold': [], 'rehit': [], 'rehit2': []}`
/ `{'r1': [], 'r2': []}`.

## The fp8 `[code]` t1-seed REP-LOOP flag — disposition (benign, evidence-complete)

What fired: the gate bar's `t1_garble` clause — the seeded temp-1.0
code text (seed 1234, the runner default) contains a 25-char window
(stride 5) repeated ≥ 4×. What did NOT fire: the corruption lane —
t0 single-hash on all 4 classes + all churn rehits; within-boot seeded
determinism INTACT (t1a==t1b); the runner's own bar (t0_identical +
cold-garble + t1_seed_identical) = CLEAN; finish stop ×3, chars stable.

1. **Cross-boot trajectory variance** (p618d law): cert7's same-lane
   code text `085255571bb1` (clean) ≠ cert8-gate `71ee4c4adcff`
   (flagged) ≠ cert8-annex-boot `3346edf045b5` (clean, pair-stable
   across rounds 1+4). The flag is a property of one sampled
   trajectory, not of the build.
2. **W14 incumbent control** (`rig/w14ctrl/`): the untouched incumbent
   (stock flash-next stack, zero fork code) fired the IDENTICAL flag
   class on the same request lane — 1/4 seed-1234 requests, top window
   `` '`\n        *   If `i % 5 =' `` ×4 — a repeated bulleted list
   item in legitimate code output. Window-repetition in temp-1.0
   sampled code text is a **model-lane property**, engine-independent.
3. **Full re-characterization on cert8 fp8** (`rig/repchar/`, 24
   records: 4 rounds × 4 classes × pairs + t0 spots + different-seed
   controls, texts banked — the runner discards them, tooling gap):
   **zero garble flags anywhere**; 15/16 seeded pairs byte-identical;
   the single divergent pair (r3 essay) coherent on both halves
   (`a81882add278` vs `18ff99786c73`, both clean, stop-finish) — the
   p618d sporadic-pair class at 1/16 (p618d saw 1/24).
4. **fp8 cannot carry the fix** (control proof t7/t9 + this gate:
   fix-engaged-on-fp8=0) — the flag cannot be a fix defect by
   construction.

Verdict: content-lane false-positive of the repetition heuristic on
probabilistic (temp-1.0) sampled code — the deterministic serving
guarantee (t0 within-path byte-identity) is untouched. The owner bar
("no corruption on ANY kv-cache type") reads the corruption class
(p618b Class-B: attractor loops/junk/bleed on the deterministic lane):
**zero occurrences across all five dtypes, cold and warm, churn
included.**

## WARN class-set (+1 vs cert7)

`_topk_topp_kernel` jit_monitor note — traffic-conditioned Triton JIT
warmup (first top-k/top-p-filtered sample; the t1.0 lane). The three
sibling jit_monitor classes were already in cert7's set; cert7's own
VERDICT recorded the same traffic-conditioning for its absent cert6
class. Zero fork-added WARN sources (upstream wording, stock wheel
sampler kernel). 31 classes total.

## Incidents

1. **`garble_check` BRE miss**: the gate's zero-garble grep escapes
   brackets (`\[\]`) which rig GNU BRE does not match as literals →
   regex count 0 → false `ZERO-GARBLE: FAIL` noise on every leg (and a
   polluted `GATE_FAIL` exit code). Real verdicts read from the JSON
   summaries; post-hoc literal `grep -cF` sweep above = 15/15 OK. Not
   hot-fixed in-place (bash reads scripts incrementally by byte offset
   — in-place edit of a running script risks abort).
2. **mq-annex sed FATAL killed the gate**: the env-injection sed
   targeted a literal `-e VLLM_XPU_ENABLE_XPU_GRAPH=1` docker flag that
   does not exist — the serve script builds `GRAPH_ENV="export …"` —
   and `boot_dtype`'s `exit 1` runs in the gate's main shell, so the
   whole script died at [4] and [5] never ran (the EXIT trap restored
   W14 anyway). Fixed sed appends the export INSIDE `GRAPH_ENV`
   (`rig/mqannex2/serve_used.sh`); re-run green. LAW: probe helpers
   must `return 1`, not `exit`, from functions.
3. **Dual W14 launch → crippled incumbent**: the trap's restore
   (15:08) + a redundant manual `w14_serve_only.sh` (15:12) raced two
   engines at gmu 0.955; after the duplicate was killed the survivor
   passed `/health` + `/v1/models`, then 500'd on the first generation
   and died — **health-only restore verify is insufficient**. Clean
   restart (stop/start + single serve launch) + a real 16-token
   generation probe = verified; the probe is now part of the annex [C]
   verify (`W14_GEN_VERIFIED`). `w14_serve_only.sh` has no guard —
   never invoke it twice without an intervening engine check.
4. **W14 same-seed non-determinism** (control observation): the
   incumbent returns different texts for identical seeded t1 requests
   within one boot (4/4 differ); the arc engine is pair-deterministic
   (15/16 here, 16/16 in p618d) — the fork stack is the MORE
   deterministic engine on this face.

## Performance summary (cert8-native, certified split posture)

| face | solo (tok/s) | agg8 (tok/s) | KV pool (tok) |
|---|---|---|---|
| fp8 (default, certified) | 84.53 / 84.57 | 300.65 (agg4×256#2) | 400,739 |
| fp8 t1.0-xhigh | 70.98 / 68.14 | 221.36 | — |
| `turboquant_k8v4` split | 54.42 / 54.59 | 235.97 | 519,972 (1.98×) |
| `turboquant_k8v4` mq (opt-in, experimental) | 53.83 / 54.31 | 284.81 | 519,972 |
| `turboquant_k3v4_nc` | — | — | 843,202 (3.22×) |
| `turboquant_4bit_nc` | — | — | 748,549 (2.86×) |
| `turboquant_3bit_nc` | — | — | 963,764 (3.68×) |

spec pos1 0.9258 / acceptance 0.837 (target 0.85 recorded-miss, the
cert7 posture); 0 ERROR / 0 Traceback on every boot (fp8, 4 TQ dtypes,
mq annex); integrity 5/5 + determinism 3/3.

## Disposition

**SHIP — `rel/v0.30.0-b70-cert8` = the certified release.** The owner
bar holds: zero corruption-class occurrences on every kv-cache dtype
(fp8 + all four turboquant presets), cold and warm, churn included;
TQ presets are servable again (split dispatch certified, mq opt-in);
the two observed identity faces are the documented engine-class
Class-A warm-resume variants and the 3-bit value-precision floor —
dispositioned with cross-boot byte-evidence, the incumbent control,
and the 24-record re-characterization.
