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
raw bench outputs, hook audit, integrity shas, serve keylines) — 108
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
