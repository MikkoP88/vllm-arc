# Testing — vllm-arc on dual Intel Arc Pro B70

Test methodology, hardware, workloads, and pass criteria for the
qwen3.8-27b-fp8 certified posture. Performance numbers live in
[`PERFORMANCE-arc.md`](PERFORMANCE-arc.md); the complete control surface
(every env/knob) lives in [`envs.md`](envs.md); run recipes in
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md).

## Hardware

| Component | Specification |
|---|---|
| GPU | 2 × Intel Arc Pro B70 (Xe2), 48 GB each, TP2 (one card per socket) |
| Host CPU | 2 × Intel Xeon Gold 5218 @ 2.30 GHz (64 CPUs) |
| Host RAM | 377 GB |
| Container | `--privileged --network host --ipc host`, shm 64 GB |
| Stack | vllm-arc image (see `/etc/vllm-arc/IMAGE_MANIFEST` inside the image: fork SHA, wheel pins, dependency closure) |

## Model and engine posture (all tests)

Model `qwen3.8-27b-fp8` (dense hybrid-GDN, MTP-capable), served with:
TP2, block size 64, FP8 weights with the block-quant W8A16 route (p44),
FP8 e4m3 KV cache, float16 Mamba SSM cache, `--mamba-cache-mode=align`
(posture-pinned; see laws below), prefix caching ON, hybrid KV manager
ON, MTP speculative decoding `num_speculative_tokens: 3`
(**recommended depth, r46t**; certified history also measured at 4 and
5 — k5 dropped from the recommendation), XPU graphs ON
(`VLLM_XPU_ENABLE_XPU_GRAPH=1`). The exact serve line is
`tools/serve_arc_b70.sh` (posture knobs) and the certified env chain in
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md).

## Test suites

Every battery runs **first-traffic** on a fresh boot (engine restart,
health 200 before any request) with timers/profilers OFF unless a suite
says otherwise. **Sampling**: the certified suites (1–4, 6–8) send
`temperature: 0` (greedy) — required for the byte-identical
integrity/golden faces and kept uniform across throughput suites; every
request also carries the inert-at-greedy real-world set (top_p 0.95,
top_k 20, min_p 0, presence_penalty 0, repetition_penalty 1 — argmax
survives top_k/top_p at temp 0, penalties are no-ops). Suite 12 defines
the real-world (sampled) regimes; the `vllm bench serve` cells (suite 5
and the 64k face) run at server-default sampling (note in suite 5).
Tools are md5-pinned in the evidence dirs; they are host-side python
scripts using only the OpenAI-compatible HTTP API.

### 1. Boot + survival flip
First request after boot ("survival flip"): engine must return HTTP 200
and complete the first generation (cert4-native: 6.2–7.3 s end-to-end
for the flip request, r70; cert-era history value 6.09 s). Catches
first-traffic death classes (graph capture, warmup interlock,
connector init).

### 2. Throughput battery — `bench_run7n.py` (`tools/bench_run7n.py`)
- **solo200**: single request, `max_tokens=200`, streaming off, tokens/s
  = `200 / wall` *including* TTFT.
- **agg4x256**: 4 concurrent threads × `max_tokens=256` each; aggregate
  tokens/s = total tokens / wall across the 4-thread join.
- Prompt: `"Count from 1 to 1000."`, temperature 0,
  `chat_template_kwargs: {"enable_thinking": false}`, non-streaming.
- Run **twice per leg** (run7n ×2); the two invocations must agree
  within ~1.5% (steady-state check).

### 3. Long-context TTFT ladder — `bench_longctx.py` + `vllm bench serve`
- **t1 (cold turn)**: fresh document of N tokens + `"Reply with the
  single word OK."`, streamed; TTFT = time to first streamed token.
  Sizes: 8k / 16k / 32k / 64k (7950 / 16237 / 32081 / 65536 tokens).
- **t2 (warm reprompt)**: SAME prompt + tiny delta suffix — measures
  same-session cache-hit latency (see warm-hit below).
- Documents are deterministic word-list compositions (seeded RNG over a
  fixed 54-word list; `random.Random(1009 * doc_id + 7)`), ~1.3
  chars/token — reproducible byte-for-byte from the script.
- 64k face measured with the vllm `bench serve` random dataset
  (`--random-input-len 65536 --random-output-len 512 --num-prompt 1
  --ignore-eos`), matrix-cell compatible (same default-sampling note
  as suite 5).

### 4. Warm-hit (prefix cache re-hit after churn)
`bench_prefix.py`: external churn sequence, then re-issue a previously
seen prompt; the response first-token latency must land ≤ 750 ms at the
8k face (bar set on the cert-era image). This exercises the prefix
reuse path end-to-end. **Tier note (r70)**: which tier serves a re-hit
is now measured, not assumed — a clean same-boot re-request without
eviction is served by **engine APC** (r70: external/LMCache tier 0.0 %
in window, warm t2 = 0.99–1.11 s at 8k–32k, warm-hit re-fires 0.80–0.91
s); the churn face here can push the prefix out of the GPU-resident
pool and exercise the **LMCache connector** retrieval + boundary snap +
TWO_HS path — a different, deeper face than the no-churn re-hit.

### 5. 12-cell load matrix — `arc_matrix.sh` (r44-certified)
`vllm bench serve`, random dataset, `--random-output-len 512
--ignore-eos --request-rate inf`:
input lengths {1024, 16384, 32768, 131072} × concurrency {1, 4, 16}.
Captures TTFT / TPOT / ITL / request throughput / output token
throughput / speculative acceptance (rate, mean length, per-position)
per cell. 12/12 cells must complete with zero errors.
**Sampling note**: `vllm bench serve` in v0.30 does not set
`--temperature` by default (`temperature=None` in the cell config; the
tool warns it no longer forces greedy) — matrix and 64k cells run at
the server/model default sampling. TTFT faces are
sampling-insensitive; matrix acceptance % are therefore
default-sampling values (the greedy acceptance figures come from the
temp-0 suites 2/8).

### 6. Output integrity (0-drift) — integrity suite
5 canonical prompts (fixed seeds 101/202/303/404/505 + 11/22/33
determinism set), temperature 0, re-issued **after the full battery +
churn**; SHA-256 of every completion must be byte-identical to the
banked reference (5/5 integrity, 3/3 determinism). Any sha shift after
warm/churn traffic = cache-integrity defect (the class fixed by p41c +
p43).

### 7. Golden logit capture — `golden_run` (5 cells)
5 fixed cells, logits captured and compared **byte-identical** (md5) to
the banked golden reference — guards numerics across posture changes
(e.g. k4→k5 adoption left all 5 golden cells byte-identical).

### 8. Speculative-decoding gate — `arc_spec_gate.py`
Counters from `/metrics` (`vllm:spec_decode_*`): drafts, draft tokens,
accepted tokens → aggregate acceptance, tokens/step, and per-position
acceptance (positions 0–4). Gate: acceptance length and pos-1 rate in
the certified band (see PERFORMANCE-arc.md for measured values).

### 9. Patch-hook audit
Serve-log audit at HEALTHY + after first traffic: **18 hook installs,
0 failed** (per-target table in `hook_audit.txt`), FP8 W8A16 reroute
ACTIVE ×4, kvfmt detector loaded. Any FAILED hook = leg invalid.

### 10. Soak — r46 (8 h, cert-era) and r70 (4 h, cert4-native)
Continuous cycles of: run7n → longctx → prefix/churn → integrity probe
(r46, 8 h) — pass = zero engine errors / tracebacks in the serve log,
integrity 5/5 at every probe, health 200 throughout, hooks stable.
Evidence: `results/2026-10-30-p5-r46-soak/`. The cert4-native soak
(r70): cycles of run7n + 24-prompt integrity probe every ~5 min for 4 h
on the certified image — **GREEN: 56/56 cycles, battery 0 fail,
integrity 56/56 bad=0, health 200 throughout, 0 ERROR/Traceback, no
throughput drift** — evidence
`results/2026-10-31-p70-r70-cert4-comprehensive/` (VERDICT §Leg C
close-out + `c/` mirror; PERFORMANCE-arc "Stability" row).

### 11. CI parity (static)
`ruff` (0.11.7) whole-tree pass, `yapf` (0.43.0) 0-hunk idempotence,
`mypy` (1.11.1, upstream `tools/mypy.sh 0 local` face) with
**delta = 0 vs upstream v0.30.0** (identical 21-error sets in an
identical container environment, A/B-proven; all errors are in
upstream's own files). Run in a disposable container per leg.

### 12. Real-world temperature × MTP-depth campaign — r46t
`tools/bench_run7n.py` with flags (`--temperature`, `--thinking`,
`--sampling alt`), one fresh boot per posture from the certified serve
line (spec depth sed'd; non-MTP leg = line minus `--speculative-config`).
Two regimes: **A** temperatures {0.2, 0.4, 0.6, 0.8, 1.0} ×
{enable_thinking: true, reasoning_effort: "xhigh"} + real-world set
(top_p 0.95 / top_k 20 / min_p 0 / presence_penalty 0 /
repetition_penalty 1); **B** alt set (temperature 0.7 / top_p 0.80 /
top_k 20 / min_p 0 / presence_penalty 1.5 / repetition_penalty 1.0),
non-thinking. Acceptance deltas from pre/post `vllm:spec_decode`
counter snapshots around every battery. Depths k3/k4 measured in both
regimes (k5 measured in A, then dropped from recommendation and
testing — its alt-leg boot was cut by the rig watcher before
starting). Output: the recommendation table in
[`PERFORMANCE-arc.md`](PERFORMANCE-arc.md) +
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md). Pass = every battery
completes 4/4 streams × all cells, zero call failures, health 200
throughout.

### 13. Garbage/corruption hardening — `arc_garbage_probe.py` (v2)
15 probes against the live server: well-formed-but-hostile content
(binary mojibake `bytes(range(256))×64`, control chars, ~0.5 MB blob,
100 k no-space run, empty content, unicode salad incl. ZWSP/RTL/emoji)
and structurally invalid requests (null content, bad role,
missing/empty/string-typed messages, string-typed max_tokens/
temperature, raw truncated JSON, raw non-JSON body). After EVERY
probe: `/health` must be 200; on 200-responses the content shape must
be valid; a 400 within the expected set (e.g. oversized-payload
rejection) counts as correct robust behavior. Suite closes with a
temp-0 fixed-prompt generation run twice — sha256 must byte-match
(engine state not corrupted). Pass = 15/15 + state stability. **v2
note**: v1 misread two correct behaviors as failures (template
thinking-on default → content in reasoning; oversized-payload 400) —
post-capture root-caused as a probe defect; v1 outputs preserved as
superseded evidence, v2 pins `enable_thinking: false` on valid probes.

### 14. Long-return gate — `p618_longprobe.py` (v3; owner hard gate at every release certification since cert7)

Long-form generation integrity under cache churn — the gate that
exposed the P6.18 TurboQuant corruption (the earlier integrity suite
samples short generations; repetition attractors live in long output).
Four classes (essay/code/multi/simple), each generated to natural
finish at t0 with a pinned seed, then re-hit cold/warm/warm2 and
re-hit again after cache-churn cycles (100/200/300-token foreign
generations between hits). PASS requires, per class, across ALL
phases: a **single sha256** for the full completion (byte-identity),
zero REP-LOOP / JUNK / BLEED / EMPTY signals (the probe's garble
detectors), `finish_reason == stop`, stable output length. Cert7 ran
4 classes × (cold + rehit + rehit2 + 3 churn+rehit cycles) — 100 %
CLEAN; plus a t1-seed annex (seeded temp-1.0 pairs; see the
determinism-semantics law below). Cert8 ran the full suite on **all
five dtypes** (fp8 + every `turboquant_*` preset, churn included —
zero garble everywhere; the owner bar "no corruption on ANY kv-cache
type"). Evidence: `results/2026-11-03-p618-cert7/rig/gate/longprobe_*`,
`results/2026-11-04-p619-tqverify/rig/gate/`.

## Laws that every test posture keeps

- **align law**: never remove `--mamba-cache-mode=align`; never ship
  `--no-enable-prefix-caching` (silently flips cache mode to "none",
  kills the boundary registry and long-context cells).
- **graph law**: all graph legs run `VLLM_XPU_ENABLE_XPU_GRAPH=1`.
- **pinned-KV law**: pinned USM >16–32 GiB/proc segfaults — the live
  gate is `LMCACHE_DISABLE_PINNED_MEMORY=1` (certified 25-var chain;
  the 5-op torch-baseline transfer re-pin is unconditional wheel-side,
  `e01-xpu-platform-ops.patch` — the old `VLLM_ARC_LMC_TORCH_KV` knob
  is dead since r47b, see envs.md); boundary snap ON (integrity).
- **timers off** for every performance face (profilers perturb decode).
- **determinism-semantics law (P6.18, from the cert7 t1-seed annex)**:
  the guarantee is **t0 within-path byte-identity** — same seed, same
  request, same engine state → identical bytes (proven across cold/
  warm/churn phases and 3 churn cycles by the long-return gate).
  Seeded **temp > 0 reproducibility is best-effort, not a
  byte-guarantee**, on any cached spec-decode system: probabilistic
  drafting samples the draft distribution, and tiny resume-state
  numerical differences can shift a sampled trajectory (a sporadic
  seeded-t1.0 flip — 1 of 24 pairs — was characterized 16/16 identical
  on re-run and dispositioned as RNG-path divergence, not corruption;
  `results/2026-11-03-p618-cert7/` VERDICT annex). Verify t0 for
  byte-claims; verify t1 only statistically.

## Evidence layout

Each leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT,
raw bench outputs, hook audit, integrity shas, serve keylines) — 113
dirs / 101 VERDICTs of record to date; the naming convention is
documented in [`envs.md`](envs.md) (naming map).

## Validation record

| Gate | Status | Evidence |
|---|---|---|
| P0 venv ground truth | GREEN | `results/2026-10-27-p0-venv-groundtruth/` |
| P1 platform + first serve | GREEN | `results/2026-10-27-p1-platform/` |
| P2 spec/graph defects fixed (survival, capture, speculator) | GREEN | `results/2026-10-28-p2-*` (34 legs) |
| P3 cache lane (connector, hybrid state, elide/memo, tail-split) | GREEN | `results/2026-10-2{8,9}-p3-r3*` (r30 banked 10-28, r31–r36 under 10-29) |
| P4 certification battery + certified image + k5 adoption + 12-cell matrix | GREEN | `results/2026-10-30-p4-r42/r43/r44/` |
| P5 env/doc/lint/Dockerfile audits + mypy delta=0 | GREEN | `results/2026-10-30-p5-r45-env-log-audit/`, rig leg `arc-p5-r46` |
| 64k TTFT face | GREEN (40,926 ms, concurrent-load conservative — cert-era); cert4-native solo U64k: 39.19–40.26 s | rig leg `arc-p5-r46/ttft64k_t1.txt`; `results/2026-10-31-p70-r70-cert4-comprehensive/` |
| r46 8 h soak | GREEN — 46 cycles, zero errors, integrity 46/46, golden 5/5 zero-drift, pos-1 0.8919 steady, health 200 end-state | `results/2026-10-30-p5-r46-soak/` |
| r46t real-world campaign (temps × depths × thinking × garbage) | GREEN — k3 recommended; garbage v2 5/5 suites 15/15; MTP 1.57–2.05× over no-spec | `results/2026-10-30-p5-r46t-temps/` |
| r70 comprehensive cert4 measurement (current record cert4-native) | GREEN legs a+b (greedy/sweep/ladder/warm/alt + no-spec A/B + tier attribution + r46t parity) + leg c soak GREEN (56/56 cycles, 0 battery/integrity/engine failures, no drift — `c/` mirror banked by the close-out commit) | `results/2026-10-31-p70-r70-cert4-comprehensive/` |
| P6.13 cert5 release (2026-11-01: cut the certified image) | GREEN single-flow on the tag-built image: manifest fork_sha == tag `3e268627`, p48 28/28 + p73 slice-hunk 11/11 image-native, SLICE_ENV_ABSENT=0 + flip 8.3 s, run7n t0 297.19/84.48-84.47 + t1.0 189.41/62.35-61.85 in-family, integrity 5/5 + det 3/3, WARN class-set identical to cert4 (44 == 44 lines, comm empty both ways), 0 ERROR/Traceback; slice face on the same image — U64k TTFT 50.77 s, W_U64k_start max 1602 ms, 63232-class 0 / 3328×38 exact | `results/2026-11-01-p65-cert5/` |
| P6.14 TurboQuant lane (2026-11-02: fix `turboquant_*` lagged store, validate all presets, certify) | GREEN — fix (TQ branch first in `_arc_attn_pool_view`, kernel-block→page merge, uint8 KV_2LTD init) validated per preset on the candidate image: 0 ERROR/Traceback, kit D2H live per slot (262/388/230/198), integrity 5/5 + det 3/3, spec ≥ target, KV 526k/757k/853k/975k (2.01/2.89/3.25/3.72×), U32k/U64k TTFT −30…−35 % vs fp8; default face inert (WARN class-set identical, all faces in-family); cert6 release gate single-flow GREEN on the tag-built image (default battery + slice face 3328×38 exact + TQ4 face: 0 errors, spec 0.9714/0.9381, KV 757,641, TTFT 15.85/33.98 s); patch-audit phase 4/4 PASS | `results/2026-11-02-p614-{dbg,preflight,l1v1,l1v2,l2,l3,cert6}/` |
| P6.16 draft-sampling posture (2026-11-03: research the upstream-direction fix, implement inline, retest all numbers, merge only on real impact, certified configs untouched) | GREEN — **opt-in `ARC_SPEC_METHOD=probabilistic`** serve hook (`tools/serve_arc_b70.sh`, branch `feat/arc-p616-spec-method-hook`): p616a boot+smoke (boots, t0 safe, cadence unchanged — gumbel/triton inside FULL draft graphs), p616b full 5-dtype matrix on cert6 (serve-line sed, 0 ERROR/Traceback, det 3/3 everywhere: t1.0-xhigh agg fp8 +29 % / tq4 +44 % / k3v4 +54 % / 3bit +91 % / k8v4 EV-neutral; every measured solo pair uniform, zero 26-tok/s legs; t0 unchanged every dtype; honest correction — the deterministic divergent-trajectory collapse remains on TQ dtypes; integrity flakes classified by the b2 discriminator: posture × churn, not reproducible ×4), p616c hook deployment (flag probe unset==junk==certified bytes + prob appends the validated string + spec-off empty, 4/4; Boot A default path full battery in-family with zero `draft_sample_method` in ps+log; Boot B opt-in engine-parsed + runtime bytes = the sed, t1.0 226.87 agg + 68.14/64.14 solo, t0 +0.3 %; 11/11 gates PASS). **Default stays greedy (k8v4 fails the every-dtype ≥ +10 % rule); no new image — cert6 remains certified (serve-layer change, wheel unchanged)** | `results/2026-11-03-p616a-probsmoke/`, `…-p616b-probmatrix/`, `…-p616c-spechook/` |
| P6.17 prob-lane certification (2026-11-03: the certified line is `ARC_SPEC_METHOD=probabilistic`, ALL record values from that lane, missing faces measured; prob lanes are the superior lanes) | GREEN — comprehensive p617 campaign on cert6 (5 fresh boots: prob legs a/b, no-spec leg c, default-path boot d, greedy opt-out boot e): every record face re-measured natively on the probabilistic lane (t0 agg 291.4–303.1 / solo 83.7–84.9; t1.0 steady 222.7–241.3 family + uniform 64–71 solo, zero collapse pairs in 20; TTFT 3.77/7.90/17.38/40.32; warm t2 0.99/1.08/1.07 via the fresh-seed warm2 probe (t1 colds 3.89/8.39/17.44 in-family with the ladder) + warm-hit 0.80–0.85 s; temp sweep 214.9–224.7 steady t0.2–0.8; alt 271.5–281.3/82.7–92.1; per-face acceptance windows 0.57–0.61 sampled / 0.859 t0 battery / 0.91 alt; MTP-over-no-spec 2.35–2.53× by face); gates — integrity 5/5 + det 3/3 every measured boot, golden 5/5 zero-drift, garbage 15/15, spec gate hard-PASS everywhere (acceptance 0.83 gate window, posture-invariant — r70 greedy gate identical), 0 ERROR/Traceback ×5, survival 200 ×5; serve-default FLIP validated: flag probe 5/5 via PATH-stubbed docker on the real script (unset/probabilistic/junk → prob bytes, `greedy` → the exact pre-P6.17 bytes, `ARC_SPEC=0` → no flag), default-path boot engine bytes IDENTICAL to the record legs (ps audit d==a), greedy opt-out boot carries the old certified bytes with t0 in-family; honest transients disclosed (first-sampled-face-after-t0 ~134–137, posture-invariant — r70 greedy identical class; one unreproduced t0 solo pair); rule override recorded (default flipped despite the k8v4 EV-neutral clause); no new image — cert6 remained certified through P6.17 (serve-layer default change, wheel/engine untouched) | `results/2026-11-03-p617a-probcert/` |
| P6.18 TurboQuant corruption campaign + cert7 (2026-11-03/04: reported `turboquant_k8v4` corruption → deep analysis, reproduction, root cause, fix, new certified image, full gates + long-return validation) | GREEN — p618a/b: corruption split into **Class B** (deterministic degenerate generation, REP-LOOP attractors, ALL FOUR presets, cold-included, LMC-independent, cross-dtype byte-identical) and **Class A** (t0 warm-resume trajectory flips — dtype+LMC-independent, settled as inherent warm-resume-vs-cold numerics by the diag/diag2 zero-retrieval canaries; disposition documentation); p618c: **roundtest proves the SoA store→decode Triton pair numerically exact on XPU** (decode self-coherence rel_err ≤ 6e-6, store spec-conformant, fully deterministic, all presets + engine tables + model geometry) → root cause = 4-bit-class value-precision quality ceiling → fix = `ArcXpuPlatform.check_and_update_config` refuses any `cache_dtype` startswith `turboquant` (loud ValueError, every entry point, no override; upstream-pattern audit banked); p618d: **cert7 release gates ALL GREEN on the tag-built image** (`a8cd765b0380`): manifest fork_sha `5bf4192d` == tag, p48 28/28 + p73 11/11, **G0 guard proof** k8v4 + tq4nc (~65 s loud refusal, startup 0), default fp8 posture PASS (KV pool 400,739 == p618b cert6-fp8 legs), run7n t0 299.73/84.50/84.44 in-family, integrity 5/5 + det 3/3 (×2), spec pos1 0.9258 PASS, 0 ERROR/Traceback, WARN class-set zero fork-added, **long-return gate 100 % CLEAN** (suite 14; 4 classes single-hash across cold/rehit/rehit2 + 3 churn cycles, zero REP-LOOP/JUNK/BLEED/EMPTY) + t1-seed annex (16/16 identical characterization → determinism-semantics law); first tag build's guard bug caught live by G0 (CLI-flag-name vs pydantic-field drift — law: config-hook changes need a runtime config-construction smoke); reproducibility audit: image ⊆ repo 144/144 md5, every image input tracked or pinned | `results/2026-11-03-p618b-repro/`, `results/2026-11-03-p618c-roundtest/`, `results/2026-11-03-p618-cert7/` |
| P6.19 TurboQuant corruption FIXED + cert8 (2026-11-04: re-opened as "no corruption on ANY KV cache type" — end-to-end telemetry, 3-root-cause fix, all-dtype long-return gate, new certified image, full metrics, all docs) | GREEN — root causes (t2i telemetry campaign): **RC1** captured verify graphs baked the raw-KV flash fast-path (fixed by donor-v19b capture forcing — the wheel builder's own `build_for_cudagraph_capture` contract), **RC2** the multi-row synthetic-decode dispatch was replay-broken under graphs (replaced by **split dispatch**: q_len single-row stock decode calls, per-row `seq_lens` as recorded device op — the certified lane), **RC3** the MQ kernel read the AoS cache with SoA addressing (first draft read live cache as zeros; `patches/tq_mq_aos.py` re-ports it to AoS — opt-in experimental, MQVAL5 4/4 exact vs the engine's own pair); fix ships as `patches/tq_spec_fix.py` default-ON (`VLLM_ARC_TQ_SPEC_FIX=1`, kill-switch for diagnosis only), P6.18 refusal LIFTED; t9 all-dtype gate: zero garble on all 4 TQ presets + fp8 control (control proof: the hooked module never imports on fp8 — the identity faces are engine-class probabilistic drafting); **cert8 release gates on the tag-built image** (`b212fbca59ec`, tag `rel/v0.30.0-b70-cert8` @ `e47739b0`): manifest fork_sha == tag, guard gone, p48 28/28 + p73 11/11 image-native, fp8 posture PASS byte-family cert7 (KV 400,739; run7n t0 300.65/84.53-84.57, t1.0 221.36/70.98-68.14; spec pos1 0.9258; integrity 5/5 + det 3/3), fp8 long-return t0 lane 100 % CLEAN (code/simple byte-exact cert7 hashes; the one `[code]` t1-seed REP-LOOP flag dispositioned benign: identical flag fired on the zero-fork W14 incumbent — a temp-1.0 model-lane property — plus a 24-record re-characterization, zero garble, 15/16 seeded pairs identical), **TQ all-dtype zero garble** (k8v4 bar PASS full+3 churns, KV 519,972 byte == t6p, tps 54.42/54.59 solo / agg8 235.97; k3v4 rehit-variant Class-A face with cold/rehit2 byte-matching the t9 primaries on all 4 classes; q4nc fully clean incl. t1; q3nc 3-bit precision-floor coherent-variant face, multi/simple stable), mq annex agg8 284.81 = 1.21× split, post-hoc literal zero-garble sweep 15/15, WARN +1 traffic-conditioned Triton-JIT note (dispositioned), 0 ERROR/Traceback ×7 boots; incidents banked (garble_check BRE tooling miss, mq sed FATAL with exit-in-function killing the gate — EXIT trap saved the restore, dual W14 launch → crippled incumbent → **restore verify now requires a real generation probe**) | `results/2026-11-04-p619-tqverify/` (dev campaign + `rig/` gate evidence + `VERDICT.md` + `BEST-PRACTICES-AUDIT.md`) |
| P6.20 prefill host-path + cert9 (2026-11-04/05: eliminate prefill host bottlenecks + the U32k/U64k p95 gap — telemetry-driven code optimization, new certified image, full metrics, all docs) | GREEN — **step telemetry first** (`patches/step_trace.py`, `ARC_STEP_TRACE`, silent-off diagnostics class: 9 segments spanning scheduler → prep → execute_model → sample → draft → wfs flush/body/total): r80 L1 evsync NEUTRAL (in-order stream — event sits behind the chunk forward; kept as the cheap body default), r81 **L3 off-thread store ADOPTED** (interleave A/B/A/B: U32k −0.74 s / U64k −1.64 s / U8k −0.08 s; zero-error; store worker ACTIVE iff async — default ON, `VLLM_ARC_LMC_ASYNC_STORE=0` kill switch; device ops stay on the engine stream, one RLock serializes), r82 gate battery GREEN + instrument lesson banked (the lmcint kit is a source-override — wrap the ENGINE'S module, not the patch file) + t1-seed annex KNOWN_FACE (8/8 pairs, cert7 method), r83 **L6 dispatch-late** FULL GREEN no annex (spec byte-exact cert8: pos1 0.9258/acc 0.837), r83 trace window = **lane closure: U64k/U32k device-compute-bound** (wfs_total waits 4.1→5.1 s/chunk = device-queue drain behind chunk forwards; worker bodies ≤108 ms; ~89 % of TTFT device-side — host path clean; L6a/L6b NOT-built, zero value on a device-bound face; remaining lever = kernel compute AR 30.4 %/FA 20.3 %); **cert9 release gates on the tag-built image** (`a9f8252451b6`, tag `rel/v0.30.0-b70-cert9` @ `aa2ee17d`): manifest == tag + all 4 p620 files md5-identical to the tag tar, p48 ALL_GREEN + p73 11/11 image-native, **default fp8 posture with ZERO env overrides** (proves default-ON L3: STORE_WORKER_ACTIVE, LMC_ENV_SET=0; step_trace silent-off verified), integrity 5/5 + det 3/3, run7n t0 296.91/84.54-84.48 in-family (t1.0 240.06 steady + 133.47 documented first-sampled transient), spec pos1 0.9258 acc 0.837 **byte-exact cert8**, fp8 LONG-RETURN PASS with ZERO t1-seed misses + WARN class-set diff 0/0, p65 image-native **U32k 16.93 / U64k 38.93 / U8k 3.76 / C32k 1.09**, TQ all-dtype zero garble (k8v4 bar PASS + tps 54.64/54.85/236.15 ≈ cert8; k3v4/q4nc sporadic identity faces annexed by re-char on fresh boots — **dominant hashes byte-identical to the cert8 gate boots** = same variant families, per-boot draft lottery only chooses which face shows; q3nc = its P6.14-record drift character), mq annex rerun with the cert8-precedent fixed GRAPH_ENV sed, 0 ERROR/Traceback every boot; gate-engineering incidents banked (the p619e-mode sed FATAL repeats — boot env injection must target the serve script's GRAPH_ENV line; garble_check's full-probe key-set artifact on rehit files — the python corruption bar is the gate); W14 restored via `w14_boot.sh fullres` + real generation probes | `results/2026-11-04-p620-prefill-opt/` (dev campaign + `rig/` r80–r84b evidence + README + REDACTION note) |
| P6.21a W13H ar_quant donor A/B (2026-11-05: quantized-AR donor port, quality + perf gate) | **NEGATIVE, closed unadopted** — quality GREEN (all identity/integrity faces clean) but perf regresses on every U-bar: U64k **+9.2 %** / U8k +7.3 % / U32k +4.2 % slower TTFT; the donor's fp16-compress AR path loses to stock oneCCL at prefill sizes | `results/2026-11-05-p621-arquant/` (squash of feat/arc-p621a-arquant-negative) |
| P6.21b FA headroom ladder (2026-11-05: r87–r91 — how much of the 20.3 % FA slice is reachable) | **Analysis of record** — stock `_vllm_fa2_C::varlen_fwd` runs at **25 % of the measured 147 TF/s dual-card fp16 peak** (S_dual 3.96); every patch-level route falsified by measurement (mask export, GQA head-pack, LSE elision, split-kv — each individually benched, none moves the needle); v1 OOB defect proven; ESIMD v2 build opened as the P6.21 vehicle | `results/2026-11-05-p621b-fa-headroom/` |
| P6.21c ESIMD dense-FA v2 kernel (2026-11-05: the kernel project the ladder opened) | **NEGATIVE, closed** — numerics GREEN (full verification ladder PASS incl. OOB canaries), perf **0.33× stock** across U32k; ablation-decomposed ceiling (floor 77.5 + stores 59 + compute 89 ms) bounds the best case at **~0.83× stock — under the 2× bar**; dpas N-by-operand-width + VNNI/scatter cost laws banked in-source | `results/2026-11-05-p621c-fa-esimd-v2/` |
| P6.22 TP2 allreduce lane (2026-11-05: r94/r95 — the last un-dispositioned device slice, ~30 % of rank0 device time) | **NEGATIVE, closed** — census instrument `ar_trace.py` (ARC_AR_TRACE, silent-off, installed 4/4) + message-size census (dominant AR = 76,677,120 B = 7488×5120 fp16 ×1,584/worker = 88 % of family) + trace (in-situ 7.85 ms/AR) + ceiling bench: **stock oneCCL ring 7.78 ms = 2.06× FASTER than the measured one-shot ceiling 16.03 ms** (concurrent ring 9.85 GB/rank/s vs torch copy engines 4.92 GB/s/dir; pre-registered 1.3× gate fails by 2.66×); survived a rig power cut (census via corpse docker cp); no cert10 at THIS phase — instrument only, nothing serve-affecting (the one-shot lane stayed closed; the quantized-wire route it left open was taken by P6.24) | `results/2026-11-05-p622-ar-lane/` (+ REDACTION note) |
| P6.23 docs-currency audit (2026-11-05: all six living docs on cert9 truth) | GREEN — 12 findings fixed across README/PERFORMANCE/TESTING/MIGRATION/CONFIGURATION/envs + build guides: stale forward claims cleared, counts corrected (113 evidence dirs / 101 VERDICTs at that point), README serve block realigned byte-identical to canonical CONFIGURATION Step 2 (58/58 lines), coverage gaps filled (Telemetry row; P6.21/P6.22 validation + MIGRATION rows; the five p619/p620 knobs); truth-preservation rulings banked (cert9-image docs correctly exclude ar_trace, rides next image); docs-only, cert9 stayed current | `results/2026-11-05-p623-docs-audit/` |
| P6.24 oneCCL fp8-wire allreduce + cert10 (2026-11-05: the P6.22 owner-named quantized-AR route taken end-to-end — r96 fidelity, r97 gate, r98 quality, r99 cert10 release, r100 bottleneck scan) | **GREEN — cert10 released** (`vllm-arc:v0.30.0-b70-cert10`, image `2f86fc2411fb`; cert9 engine byte-identical by exhaustive `vllm_gaudi/` md5 walk, + two-file oneCCL overlay, manifest rows `oneccl=2022.0.0+arc1`): **r96** same-version 2022.0.0 rebuild FIDELITY PASS (baseline reproduces the r95 anchor exactly — 7.782 vs 7.780 ms at the dominant size; overlay +0.3 % gated; canonical staging paths + tracked-tree pristine checks banked); **r97** e4m3-wire ring v4 GATE PASS **1.65×** at 76,677,120 B (7.801→4.731 ms; 1.57–1.69× all large sizes; FP8=0 control == anchor; numerics rel-rms 0.02285 == the offline bit-exact floor; sector law + K1-atomics laws banked); **r98** quality ladder task-level parity (same container one-env-line A/B: tps 1.005/0.998/0.954, run7n t0 0.998, spec +0.008 << 0.03 bar, integrity/det/long-return/erraudit all green, `simple` byte-identical = min-count law live; two analysis-layer script defects found+fixed same-day, measured data never affected); **r99** cert10 release gate GATE PASS (identity gates incl. tree walk; [1] default-posture battery green with FP8_ENV_SET=0 proven — one measured disposition: code t1 REP-LOOP boot-stochastic, non-reproducing fresh-boot, t0 green everywhere, r99c annex, bar amended REPORT-with-annex; [2] k8v4 TQ carry — essay/code/simple byte-identical to stock cert9 k8v4; [3] FP8 knob leg + in-situ wire proof on the shipped image, 40 `CCL fp8` STATS lines count 7,459,840 both-rank scale agreement; [4] A/B/A/B perf interleave **beats the banked cert9 baseline on EVERY p65 face**: U32k 16.29 (−3.8 %), C32k 1.04 (−4.6 %), U8k 3.27 (−13.0 %), U64k 35.73 s (−8.2 %), gated ratios 0.920/0.916 in [0.80,1.02]); **r100** bottleneck scan ALL DOORS CLOSED with wire-rate measurement over the real peer IPC mapping (remote writes 9.85 GB/s flat 4/8/16 B — fabric ceiling, wider stores dead; reads 4.54 GB/s 2.2× slower — read-reduce dead; concurrent rw collapses 1.49 — pipelining dead; local 380 GB/s 38× headroom; control +0.0 % at the dominant size) — v4 = 82 % of the wire-bound floor, final; probe build env-gated diagnostic, never ships; runner laws banked (r84 kill_arc retry after a transient OCI setns death; three cwd-class script defects caught, data intact, reconstruct tools banked); W14 restored + verified every window | `results/2026-11-05-p624-fp8-ar/` (r96–r100 evidence + VERDICT + REDACTION notes + `external_patches/oneccl/`) |
| P6.24b in-line-best-practice patch audit (2026-11-06: post-release deep review of the landed P6.24 squash — patch format/content, env registration, kernel patterns, build script, Dockerfile, docs parity, sha evidence chain) | **GREEN — three findings, all dispositioned** (audit record: the patch-audit section atop `results/2026-11-05-p624-fp8-ar/VERDICT.md`): tracked probe patch reconciled to the as-built rig bytes (delta = one trailing newline; sha chain `cd66f238…` reproducible again, zero content change); res-scale invariant documented precisely in the oneccl README (2*max = primarily the cross-rank AGREEMENT mechanism; bound domain + generalization formula banked; no gate could expose the residual domain gap by construction — v4 stays FINAL); build-script header + bootstrap pipefail; frozen-artifact law banked (gated patch files never edited — their sha256s ARE the evidence chain). Everything else verified clean: v4 sha chain tracked==VERDICT==rig kit==manifest, series excludes the probe patch, env registration upstream-pattern with docs-matching defaults, kernel patterns, Dockerfile copies byte-identical + LIB_SHA-gated, CRLF-clean, knob-table parity | `external_patches/oneccl/README.md` (invariant section) + the VERDICT patch-audit section |
| P6.25 overlap-door audit (2026-11-06: r101 windows 1+2 — the one door P6.20–P6.24 left unmeasured: cross-stream AR∥GEMM co-tenancy, the async-TP class; owner directive "eliminate all remaining prefill bottlenecks") | **NEGATIVE, closed MEASURED — no engine change, cert10 stays current (no cert11: nothing landed)**: pre-registered gate (best-k mock ≥1.08× serial) read **FAIL on both wires** — mock-layer A/B on real per-rank shapes (M=7488, L=8, GLU included, `tools/ar/ar_overlap_bench.py`, cert10 image, torchrun TP2 xccl): best-k **0.969× stock (k=2) / 0.966× fp8 (k=3)** — overlap slower than serial at every k (k-sweep flat → class closed, schedule- and wire-independent); root cause measured directly — **coten gemm_inflation 1.974× stock / 1.580× fp8** (concurrent AR nearly doubles GEMM latency: the device time-slices AR and GEMM kernels on the same execution resources — no idle-EU lane), while enqueue itself is async (0.182/0.263 ms — viable at the API, dead on the device); census bonus closes the GEMM lane: layer GEMMs 17.173 ms = 148.6 TF/s rank-local ≈ **~99 % of the p66 devprobe's 147–150 TF/s per-card dense fp16 ceiling** (the audit draft's 73.5 TF/s/rank reference was wrong — the "147 dual" figure was per-card); anchors re-validated in-window (stock AR 7.809 ms == the r94/r100 ceiling; fp8 4.597 ms = 1.70×, in-family with r97's 4.731/1.65×); two bench shape bugs (o-slice, GLU) found+fixed BEFORE the gate was read — gate read only on correct-shape data; `overlap_bitwise:false` = mock artifact (k-split changes GEMM M-dims → different tile/split-K accumulation order; the AR per-token-independence property itself holds); residue map now EXHAUSTED — every prefill door measured-closed, remaining headroom bounded ≤1–2 % class (AR-internal pipelining ~1 % r100; fp8 quant-pass fusion = novel cross-library contract that only relocates the same EU work); W14 restored + generation-probed post-window | `results/2026-11-06-p625-overlap-door/` |
| P6.26 MXFP4 validation + fix + cert11 (2026-11-06: owner directive — full testing of `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` + `RedHatAI/Qwen3.8-27B-MXFP4xFP8`; r102 recon, r103/r103b phase A, r104 phase B, r105/r105t/r105t2/r105b release chain) | **GREEN — fix landed (7ea0f27b) + cert11 released** (`vllm-arc:v0.30.0-b70-cert11`, tag `rel/v0.30.0-b70-cert11` = c987c2cd): **r102** pre-boot wheel recon predicted the quark failure byte-exact (core `XPUPlatform.supported_quantization` omits `quark` → `Platform.verify_quantization` ValueError at config validation; the wheel's QuarkConfig routes OCP-MX W4A4 through the platform-neutral `init_mxfp4_linear_kernel` → native `XPUMxFp4LinearKernel` on Xe2 — a format-origin omission, not a hardware limit); **r103/r103b phase A** — rhmxfp4 boots UNMODIFIED on cert10; qurk fails with the predicted error; patched retry boots clean (9.03 GiB/rank in 2.51 s, native kernel line, no Simulated warning); fix = two `ArcXpuPlatform` overrides (HpuPlatform pattern, wheel-free, runtime-inert for existing lanes — kernel selection table-keyed); **r104 phase B** full certified posture both legs: coherent G1 probes, spec gates 0.9853/0.9951 (above the fp8 incumbent's 0.9258), decode penalty 1.21×/1.29× < the pre-registered 2× trigger, TTFT 1.6–2.4× (REPORT) → disposition: mxfp4 = certified weight-constrained lane, fp8 stays performance default; **r105** engine battery on cert11c — run-1 VOID (concurrent 30 GB host download through the window; archived `r105-contaminated/` + quiescence guard added), run-2 clean PASS (300.48/84.55–84.56, TTFT 17.37/1.09/3.77/38.82, LONGRETURN PASS, ERROR=0) vs **provenance-corrected anchors** — the pre-registered bars were memory-recalled, never re-grepped (disclosed gate defect); corrected anchors grep-cited from banked r99 perf_off1/off2 (17.715/1.21/3.78/39.025), verdict recomputed mechanically from banked run-2 artifacts (`r105/verdict_recomputed.txt`); **r105t/r105t2** same-day cert10 discriminator — cert10 ≡ cert11c face-for-face (17.66/1.19/3.77/38.97), thermal hypothesis refuted with xpu-smi telemetry (no throttle reason, 63–92 W of 230 W cap; r105t-void = killed-by-self-matching-pkill attempt, archived with the operator-error record); **r105b** final-image gate on the tag-built overlay — raw GATE_FAIL was the awk `$2`-vs-`$3` column bug (fixed PR #79), recomputed **PASS** from banked artifacts (manifest + oneccl rows, walk 294→294 changed set exactly platform.py+pyc, overlay shas ×2, thin shim, freeze 0, HEALTH 295 s, FLIP_FIRST 9.1 s, SPEC_GATE pos1 0.9217, ERROR=0); PRs #77/#78/#79 → main both remotes; W14 restored + generation-probed every window | `results/2026-11-06-p626-mxfp4/` (+ REDACTION note; fix commit 7ea0f27b; runners `tools/ar/r10[2-5]*.sh`) |
| P6.27 MXFP8 AutoRound validation (2026-11-06: owner directive — full testing of `intel-ai/Qwen3.8-27B-MXFP8-MXFP8MTP-CT-AutoRound`; r106 pre-boot recon in plan §P6.27, r106 phase A isolation, r106b phase B full posture — both legs on the CERTIFIED cert11 image) | **GREEN — all gates PASS, ZERO fixes, no cert12 (zero engine/wheel bytes changed; the lane is a config-driven model route on cert11)**: **recon held byte-for-byte** — quant_method `compressed-tensors` passes the platform list (no P6.26-class rejection), `_is_mxfp8` matches all six predicates → `CompressedTensorsW8A8Mxfp8`, `_POSSIBLE_MXFP8_KERNELS[XPU]` → `XPUMxFp8LinearKernel` on oneDNN `fp8_gemm` — the pre-registered dispatch line `Using XPUMxFp8LinearKernel for MXFP8 GEMM` (kernels/linear/__init__.py:923) captured in BOTH postures; **phase A** (eager/spec0/lmc0): weights 14.74 GiB/rank in 21.1 s, coherent probe, 0/0 erraudit — P-1 (param mismatch) refuted; **phase B** (graph FULL_DECODE_ONLY + MTP k3 prob + LMCache): health 285 s incumbent-identical, graph captured 71 sizes (P-3 refuted — `xpu_mxfp8_quantize` replays in-graph), quantized MTP drafter loads + drafts, G1 coherent (train probe correct), **SPEC_GATE PASS pos1 0.9626 / acceptance 0.9174** (P-2 refuted; fp8 incumbent 0.9258, mxfp4 lanes 0.9853/0.9951), KV pool 6.25 GiB, 0/0 erraudit; **G2 REPORT matrix** (all under the 2× trigger): TTFT U32k 26.67 / C32k 1.52 / U8k 6.36 / U64k 59.52 s vs r105-corrected anchors = 1.50×/1.26×/1.68×/1.53× — **below both mxfp4 lanes on every TTFT face**; bystander decode p50 81.8 ms = **1.53× — ABOVE both mxfp4 lanes (64.7/68.8)**: the pre-registered "below 1.2–1.3×" decode expectation was wrong and is explained (decode is weight-byte-bound: W8 = 2× the W4 bytes/step + per-linear MX-scale handling vs the incumbent's fused block-scaled lane; prefill amortizes the quant pass — hence faster than mxfp4 there); known-warning disposition ("no KV cache group identified as the draft model's": 1× here + 1× both mxfp4 legs + 0× fp8 incumbent = pre-existing CT-quant class, benign — spec gates pass on every leg showing it); disposition: mxfp8 = the AutoRound-import lane (no capacity win vs fp8, no quality loss), fp8 stays the performance default; W14 restored + generation-probed post-window | `results/2026-11-06-p627-mxfp8/` (+ REDACTION note; runners `tools/ar/r106*.sh`) |
| P6.28 gemma-4 validation + G2 freeze + cert17 (2026-11-06/07: owner directive — full gemma-4 testing/validation, the G2 long-prefill freeze "solved fully", prefill perf diagnostic) | GREEN — G2 freeze closed in 3 layers (cert16 warm-gate v4 cache-key split; probe-time compile accepted; cert17 KVGATE + `ARC_LMC=0`); layer-2b = the 300 s execute-model RPC timeout (`ARC_RPC_TIMEOUT=1200`, face120k 2950.3 s zero-jit); L1 fp8 battery all faces green (31.5/283.4/2950.3/9.8 s; decode 18.4–18.5 solo / 69.9 agg8 tok/s); L2 mxfp8 battery green (35.4/297.4/1819.5/10.3 s; decode 11.9/54.6; G1 reason-probe miss flagged — broader quality battery follow-up); erraudit 0/0 everywhere; prefill root cause CLOSED (hd512 full-attn L2-bandwidth wall, r108aa/r108ab bare-kernel benches — kernel-structural, upstream shares the cap, certified table at ceiling) | `results/2026-11-06-p628-gemma4/` (phaseB `tile_sweep/`; phaseC `cert15dbg2_boot/`, `cert16_boot/`, `cert16_nolmc/`, `cert17_boot/` incl. `gate_fire.txt`, `cert17_l2/`, `cert17_decode/`, `micro_realshape/`, `micro_sweep/`) |
