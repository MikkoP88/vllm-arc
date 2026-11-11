# Testing — vllm-arc on dual Intel Arc Pro B70

Test methodology, hardware, workloads, and pass criteria for the
qwen3.8-27b-fp8 certified posture. Performance numbers live in
[`PERFORMANCE-arc.md`](PERFORMANCE-arc.md); the complete control surface
(every env/knob) lives in [`envs.md`](envs.md); run recipes in
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md). The per-phase validation
ledger archive: [`history/TESTING_HISTORY.md`](history/TESTING_HISTORY.md);
per-cert/per-phase changelog: [`../CHANGELOG.md`](../CHANGELOG.md).

## Hardware

| Component | Specification |
|---|---|
| GPU | 2 × Intel Arc Pro B70 (Xe2/BMG-G31, 256 XMX), 32 GB each (31.89 GiB visible), TP2 — both cards on one CPU socket (`ZE_AFFINITY_MASK=0,1` exposes both device indices) |
| Host CPU | 2 × Intel Xeon Gold 5218 @ 2.30 GHz (64 CPUs) |
| Host RAM | 377 GB (validation rig); the recommended configs run on 64 GB-RAM hosts (`--shm-size 16g`, 16 GB LMCache CPU tier) |
| Container | `--privileged --network host --ipc host`, shm 16 GB |
| Stack | vllm-arc image (see `/etc/vllm-arc/IMAGE_MANIFEST` inside the image: fork SHA, wheel pins, dependency closure) |

## Model and engine posture (all tests)

Model `qwen3.8-27b-fp8` (dense hybrid-GDN, MTP-capable), served with:
TP2, block size 64, FP8 weights with the block-quant W8A16 route (p44),
FP8 e4m3 KV cache, float16 Mamba SSM cache, `--mamba-cache-mode=align`
(posture-pinned; see laws below), prefix caching ON, hybrid KV manager
ON, MTP speculative decoding `num_speculative_tokens: 3`
(**recommended depth**; k4 is the documented alternative, k5 is
measured and dropped for the certified battery — decision record:
CHANGELOG), XPU graphs ON (`VLLM_XPU_ENABLE_XPU_GRAPH=1`). The exact
serve line is `tools/serve_arc_b70.sh` (posture knobs) and the
certified env chain in
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
and complete the first generation (6.2–8.2 s end-to-end for the flip
request). Catches first-traffic death classes (graph capture, warmup
interlock, connector init).

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
8k face. This exercises the prefix reuse path end-to-end. **Tier
note**: which tier serves a re-hit is measured, not assumed — a clean
same-boot re-request without eviction is served by **engine APC**
(external/LMCache tier 0.0 % in window, warm t2 = 0.99–1.11 s at
8k–32k, warm-hit re-fires 0.80–0.91 s); the churn face here can push
the prefix out of the GPU-resident pool and exercise the **LMCache
connector** retrieval + boundary snap + TWO_HS path — a different,
deeper face than the no-churn re-hit.

### 5. 12-cell load matrix — `arc_matrix.sh`
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
warm/churn traffic = cache-integrity defect.

### 7. Golden logit capture — `golden_run` (5 cells)
5 fixed cells, logits captured and compared **byte-identical** (md5) to
the banked golden reference — guards numerics across posture changes.

### 8. Speculative-decoding gate — `arc_spec_gate.py`
Counters from `/metrics` (`vllm:spec_decode_*`): drafts, draft tokens,
accepted tokens → aggregate acceptance, tokens/step, and per-position
acceptance (positions 0–4). Gate: acceptance length and pos-1 rate in
the certified band (see PERFORMANCE-arc.md for measured values).

### 9. Patch-hook audit
Serve-log audit at HEALTHY + after first traffic: **hook installs,
0 failed** (per-target table in `hook_audit.txt`), FP8 W8A16 reroute
ACTIVE ×4, kvfmt detector loaded. Any FAILED hook = leg invalid.
The authoritative per-patch inventory (all four loading layers, every
module with its knob and provenance) is
[`PATCHES-arc.md`](PATCHES-arc.md) — the count grows with the image;
the gate audits the class, PATCHES-arc audits the census.

### 10. Soak (4 h, certified image)
Continuous cycles of run7n + 24-prompt integrity probe every ~5 min for
4 h — **GREEN: 56/56 cycles, battery 0 fail, integrity 56/56 bad=0,
health 200 throughout, 0 ERROR/Traceback, no throughput drift**.
Evidence: `results/2026-10-31-p70-r70-cert4-comprehensive/`.

### 11. CI parity (static)
`ruff` (0.11.7) whole-tree pass, `yapf` (0.43.0) 0-hunk idempotence,
`mypy` (1.11.1, upstream `tools/mypy.sh 0 local` face) with
**delta = 0 vs upstream v0.30.0** (identical 21-error sets in an
identical container environment, A/B-proven; all errors are in
upstream's own files). Run in a disposable container per leg.

### 12. Real-world temperature × MTP-depth campaign
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
regimes (k5 measured then dropped — decision record: CHANGELOG).
Output: the recommendation table in
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
note**: v2 pins `enable_thinking: false` on valid probes (v1 misread
two correct behaviors as failures and is superseded).

### 14. Long-return gate — `p618_longprobe.py` (hard gate at every release certification)

Long-form generation integrity under cache churn — the gate that
exposed the TurboQuant corruption class (the earlier integrity suite
samples short generations; repetition attractors live in long output).
Four classes (essay/code/multi/simple), each generated to natural
finish at t0 with a pinned seed, then re-hit cold/warm/warm2 and
re-hit again after cache-churn cycles (100/200/300-token foreign
generations between hits). PASS requires, per class, across ALL
phases: a **single sha256** for the full completion (byte-identity),
zero REP-LOOP / JUNK / BLEED / EMPTY signals (the probe's garble
detectors), `finish_reason == stop`, stable output length. The full
suite runs on **all five dtypes** (fp8 + every `turboquant_*` preset,
churn included — the release bar "no corruption on ANY kv-cache
type"). Evidence: `results/2026-11-03-p618-cert7/rig/gate/longprobe_*`,
`results/2026-11-04-p619-tqverify/rig/gate/`.

## Laws that every test posture keeps

- **align law**: never remove `--mamba-cache-mode=align`; never ship
  `--no-enable-prefix-caching` (silently flips cache mode to "none",
  kills the boundary registry and long-context cells).
- **graph law**: all graph legs run `VLLM_XPU_ENABLE_XPU_GRAPH=1`.
- **pinned-KV law**: pinned USM >16–32 GiB/proc segfaults — the live
  gate is `LMCACHE_DISABLE_PINNED_MEMORY=1` (certified 25-var chain);
  the 5-op torch-baseline transfer re-pin is unconditional wheel-side
  (`e01-xpu-platform-ops.patch`; the dead `VLLM_ARC_LMC_TORCH_KV`
  knob: envs.md); boundary snap ON (integrity).
- **timers off** for every performance face (profilers perturb decode).
- **determinism-semantics law**: the guarantee is **t0 within-path
  byte-identity** — same seed, same request, same engine state →
  identical bytes (proven across cold/warm/churn phases and 3 churn
  cycles by the long-return gate). Seeded **temp > 0 reproducibility
  is best-effort, not a byte-guarantee**, on any cached spec-decode
  system: probabilistic drafting samples the draft distribution, and
  tiny resume-state numerical differences can shift a sampled
  trajectory. Verify t0 for byte-claims; verify t1 only statistically
  (characterization: `results/2026-11-03-p618-cert7/` VERDICT annex).

## Evidence layout

Each leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT,
raw bench outputs, hook audit, integrity shas, serve keylines) — 128
dirs / 102 VERDICTs of record to date; the naming convention is
documented in [`envs.md`](envs.md) (naming map).

## Validation record

Every phase of the project closed GREEN (or measured-negative and
closed) with banked evidence. The per-phase validation table — gate,
status, and `results/` pointer for all of P0 through P6.38 — is
archived in [`history/TESTING_HISTORY.md`](history/TESTING_HISTORY.md).
Per-cert/per-phase changelog entries:
[`../CHANGELOG.md`](../CHANGELOG.md). Release summaries:
[`release_notes.md`](release_notes.md).
