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
and complete the first generation (measured 6.09 s end-to-end for the
flip request). Catches first-traffic death classes (graph capture,
warmup interlock, connector init).

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
on the certified image — evidence
`results/2026-10-31-p70-r70-cert4-comprehensive/` (in flight at this
commit; close-out row lands in PERFORMANCE-arc "Stability" with the
follow-up commit).

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

## Laws that every test posture keeps

- **align law**: never remove `--mamba-cache-mode=align`; never ship
  `--no-enable-prefix-caching` (silently flips cache mode to "none",
  kills the boundary registry and long-context cells).
- **graph law**: all graph legs run `VLLM_XPU_ENABLE_XPU_GRAPH=1`.
- **pinned-KV law**: `VLLM_ARC_LMC_TORCH_KV=1` default (native pinned
  ops segfault >16–32 GiB/process); boundary snap ON (integrity).
- **timers off** for every performance face (profilers perturb decode).

## Evidence layout

Each leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT,
raw bench outputs, hook audit, integrity shas, serve keylines) — 77
dirs to date; the naming convention is documented in
[`envs.md`](envs.md) (naming map).

## Validation record

| Gate | Status | Evidence |
|---|---|---|
| P0 venv ground truth | GREEN | `results/2026-10-27-p0-venv-groundtruth/` |
| P1 platform + first serve | GREEN | `results/2026-10-27-p1-platform/` |
| P2 spec/graph defects fixed (survival, capture, speculator) | GREEN | `results/2026-10-28-p2-*` (10 legs) |
| P3 cache lane (connector, hybrid state, elide/memo, tail-split) | GREEN | `results/2026-10-29-p3-r30..r36/` |
| P4 certification battery + certified image + k5 adoption + 12-cell matrix | GREEN | `results/2026-10-30-p4-r42/r43/r44/` |
| P5 env/doc/lint/Dockerfile audits + mypy delta=0 | GREEN | `results/2026-10-30-p5-r45-env-log-audit/`, rig leg `arc-p5-r46` |
| 64k TTFT face | GREEN (40,926 ms, concurrent-load conservative — cert-era); cert4-native solo U64k: 39.19–40.26 s | rig leg `arc-p5-r46/ttft64k_t1.txt`; `results/2026-10-31-p70-r70-cert4-comprehensive/` |
| r46 8 h soak | GREEN — 46 cycles, zero errors, integrity 46/46, golden 5/5 zero-drift, pos-1 0.8919 steady, health 200 end-state | `results/2026-10-30-p5-r46-soak/` |
| r46t real-world campaign (temps × depths × thinking × garbage) | GREEN — k3 recommended; garbage v2 5/5 suites 15/15; MTP 1.57–2.05× over no-spec | `results/2026-10-30-p5-r46t-temps/` |
| r70 comprehensive cert4 measurement (owner directive: current record cert4-native) | GREEN legs a+b (greedy/sweep/ladder/warm/alt + no-spec A/B + tier attribution + r46t parity); leg c soak in flight at this commit — close-out follows with `c/` evidence | `results/2026-10-31-p70-r70-cert4-comprehensive/` |
