# p616c — ARC_SPEC_METHOD serve hook: end-to-end validation (2026-11-03)

Close-out of the owner P6.16 directive (*"run full analyze and research
how to implement fix inline … start-to-end implementations and fix
deployment"*): the P6.16b-proven probabilistic draft posture ships as an
**opt-in serve-layer knob** — `ARC_SPEC_METHOD=probabilistic` in
`tools/serve_arc_b70.sh` (branch `feat/arc-p616-spec-method-hook`,
commit `f22acdbd`). This leg validates the DEPLOYMENT on the rig: the
hook script scp'd **byte-unchanged (ZERO seds — unlike p616a/b, the env
knob IS the injection path)**, flag-probe expansion + two full boots on
the cert6 image (fp8 default dtype, graph law, ARC_SPEC=3, W14 trap,
preexisting arc-test snapshot law).

## The one-line verdict

**PASS end-to-end — merge as serve-layer.** The default path emits the
exact certified spec-config bytes (probe + runtime `ps` + engine log
three-way proof, junk values byte-identical to unset), and the opt-in
path delivers the p616b probabilistic faces through the knob (t1.0
68.14/64.14 solo + 226.87 agg vs p616b's 67.46/64.10 + 241.10 —
uniform, +22 % over the greedy bank; t0 in-band both boots, integrity
5/5 + det 3/3 both boots, 0 ERROR / 0 Traceback both boots). **No cert7
image**: the serve script is not shipped in the image (Dockerfile
carries only `tools/wheel_guard.sh`) and the wheel/engine are unchanged
— a new image would be content-identical. cert6 remains the certified
image; the hook deploys from the repo.

## Stage 0 — flag probe (extracted from the deployed script, no boot)

`SPEC_FLAG` expanded for all four env paths:

| Path | SPEC_FLAG |
|---|---|
| `ARC_SPEC_METHOD` unset | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` |
| `ARC_SPEC_METHOD=typical` (junk) | byte-identical to unset (`cmp` IDENTICAL) |
| `ARC_SPEC_METHOD=probabilistic` | `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'` |
| `ARC_SPEC=0` (spec off) | empty |

Gates: unset==junk **IDENTICAL**; unset == the certified bytes
**(1/1 grep)**; probabilistic appends exactly the p616b-validated
string **(1/1)**; spec-off empty **(1/1)**. The awk extraction
(`hook_block.sh`) is the deployed script's own SPEC block — 8 lines,
nothing else.

## Boot A — default path (`ARC_SPEC_METHOD` unset = the certified posture)

| Face | boot A | bank | Δ |
|---|---|---|---|
| t0 agg#2 | 296.69 | 296.25 | +0.15 % |
| t0 solo ×2 | 84.47 / 84.44 | 84.51 / 84.41 | in band |
| t1.0 greedy agg#2 | 184.86 | 186.48 | −0.9 % |
| t1.0 greedy solo ×2 | 61.79 / 61.07 | 61.73 / 61.79 | in band |
| TTFT U32k / U64k | 17.47 / 40.51 s | 17.34 / 40.24 s | +0.7 / +0.7 % |
| spec gate | PASS pos1 0.9221, acc 0.8254 (target-miss recorded — the known fp8 class) | — | — |
| integrity / det | **5/5 / 3/3** | — | — |
| ERROR / Traceback | **0 / 0** | — | — |

Runtime injection proof: `ps` serve line carries
`num_speculative_tokens":3` and **zero** `draft_sample_method`
occurrences (count 0); the engine config dump has **no** method key
(`spec-method NOT in log`). The certified posture is byte-identical
through the hook.

## Boot B — opt-in path (`ARC_SPEC_METHOD=probabilistic`)

| Face | boot B | comparator | Δ |
|---|---|---|---|
| engine parsed the key | config dump `draft_sample_method': 'probabilistic` | — | ✓ |
| runtime serve line | `num_speculative_tokens":3` + `draft_sample_method":"probabilistic"` (count 1) | p616b sed bytes | identical |
| t0 agg#2 | 297.12 | 296.25 | +0.3 % |
| t0 solo ×2 | 84.42 / 84.54 | 84.51 / 84.41 | in band |
| t1.0 agg#2 | **226.87** | p616b 241.10 / greedy bank 186.48 | −5.9 % cross-boot / **+21.7 %** |
| t1.0 solo ×2 | **68.14 / 64.14** | p616b 67.46 / 64.10 | byte-close; uniform, no collapse leg |
| TTFT U32k / U64k | 17.34 / 42.18 s | p616b 17.37 / 40.30 s | −0.2 % / **+4.7 % flagged** |
| spec gate | PASS pos1 0.9181, acc 0.8218 (target-miss recorded) | — | — |
| integrity / det | **5/5 / 3/3** (fp8 clean, as in p616b) | — | — |
| ERROR / Traceback | **0 / 0** | — | — |

Both solo legs ≥ 55 tok/s and ≥ the greedy bank 61.73; the agg sits
+21.7 % over the certified greedy face and −5.9 % under p616b's
241.10 — ordinary cross-boot variance on the sampled face (p616a/b
same-face spread was wider), direction and magnitude both consistent
with the banked probabilistic fp8 record. U64k TTFT +4.7 % is the one
out-of-family face (single measurement; same class as p616b's flagged
3bit U64k +7.2 % — boot-day variance on the slowest-flip leg); U32k
dead-on. t0 is posture-provably identical (argmax drafts at temp 0,
`gumbel.py:102,117`; accept-iff-argmax verify,
`rejection_sampler_utils.py:571-585`) — and measures in-band.

## Gates

| Gate | Result |
|---|---|
| (0a) unset == junk flag bytes == certified bytes | **PASS** (probe, `cmp` IDENTICAL + byte grep) |
| (0b) probabilistic appends exactly the p616b-validated string; spec-off empty | **PASS** |
| (A1) default path runtime: no method key in ps + engine log; spec config = certified bytes | **PASS** |
| (A2) default path t0 within ±1.5 % (agg/solo/TTFT) | **PASS** (+0.15 % agg) |
| (A3) default path t1.0 greedy face in family | **PASS** (−0.9 % agg, solo in band) |
| (A4) integrity 5/5 + det 3/3 + 0 ERROR/Traceback | **PASS** |
| (B1) opt-in path: engine parses the key; runtime bytes = p616b sed | **PASS** |
| (B2) opt-in path t0 within ±1.5 % | **PASS** (+0.3 % agg) |
| (B3) opt-in path t1.0: both solo ≥ 55 and ≥ 61.73; agg ≥ 186.48 | **PASS** (68.14/64.14; 226.87 = +21.7 %) |
| (B4) integrity 5/5 + det 3/3 + 0 ERROR/Traceback | **PASS** |
| (B5) TTFT within −3 % | **PASS except U64k +4.7 % (flagged, single face)** |

## Disposition

- **Merge** `feat/arc-p616-spec-method-hook` (serve-layer; engine/wheel
  byte-unchanged). Default posture = greedy = certified, byte-identical
  by construction (only the `= probabilistic` branch adds anything).
- **No cert7 image** — the serve script is rig-side/repo-side, not in
  the image; a rebuilt image would be content-identical (unevidenced
  artifact). cert6 stays the certified image; certification of the
  opt-in posture = p616a + p616b + this leg, all on cert6.
- Docs: PERFORMANCE-arc (posture section + TQ t1.0 reading pointer),
  CONFIGURATION-arc (`ARC_SPEC_METHOD` knob + menu), TESTING-arc
  (validation row), MIGRATION-arc (evidence row), envs.md (host-layer
  var), README (status + counts).

## Provenance

Leg: 2 boots + probe, cert6 image, fp8 default dtype, ARC_GRAPH=1,
ARC_SPEC=3, harnesses `bench_run7n_t5.py` (t0 + t1.0-xhigh),
`arc_spec_gate.py`, `r42_integrity.py`, `p70_solo_ttft.py` (U-bars);
deployed script sha256 `fb631d35…` == repo `f22acdbd`
(`serve_used.sh`); leg script `p616c_hook_validate.sh`; W14 restored by
trap (health 200 verified post-leg). Serve logs + preexisting json
redacted byte-exact: bootA log +34, bootB log +104, json +5 = **+143**;
raw-ref grep 0 — `../REDACTION-2026-11-03-p616c.md`.
