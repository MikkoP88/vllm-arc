# P6.47 — cert24 §10 gate (r154): run1 verdicts + gemma probe root causes (2026-11-12)

Image `vllm-arc:v0.30.0-b70-cert24` (manifest fork_sha `40fa6830`, cert23 base).
Gate: `tools/ar/r154_cert24_gate.sh` (md5 `f8f3e3e3e35d9f8089b41dc6e5d6187e` at
run3; run1 script `7e8b118923ee36b02f4d0c1607d8dd81`). Legs: ctrl/m8q/m4q/m8g/m4g
on `feat/arc-p647-mxkv-engine-vstride-fix`. Raw evidence: `raw/cert24/<leg>/`,
`raw/gatelog/gate_0841.log` (run1) + `gate_run2_1040.log` (run2 partial).

## Run1 verdicts (gate_0841.log, 2026-11-12 07:41–10:17 UTC)

| leg | posture | suite-13 | suite-14 long-return | verdict |
|---|---|---|---|---|
| ctrl | qwen champion chain, KV dtype default (fp8_e4m3) | PASS | — (short leg) | **PASS** — `mx_backend_lines=0` (default-off neutral) |
| m8q | qwen + `ARC_KVDTYPE=mxfp8` (LMC=1 spec3) | PASS 15/15 | PASS (LONGRETURN + T1_SEED, all 4 classes) | **FULL PASS** |
| m4q | qwen + `ARC_KVDTYPE=mxfp4` | PASS 15/15 | **FAIL — essay REP-LOOP under churn** | **REAL quality-limit failure** |
| m8g | gemma W8A16 + mxfp8 KV | 15/15 (run2) | INVALID (run1: longprobe 404s) | probe bugs — engine exonerated |
| m4g | gemma W8A16 + mxfp4 KV | INVALID (run1) | INVALID (run1) | probe bugs — engine exonerated |

### m8q control: perfect cross-cycle determinism (raw/cert24/m8q/longprobe_full.out)

```
[essay]  t0_identical=True hashes=['997102a41c3b' x3] garble={cold:[],rehit:[],rehit2:[]} chars=[3771,3771,3771]
[code]   t0_identical=True hashes=['8ae6a2992644' x3] ... chars=[3699,3699,3699]
[multi]  t0_identical=True hashes=['6fe272b48561' x3] ... chars=[588,588,588]
[simple] t0_identical=True hashes=['84efc84cc142' x3] ... chars=[1510,1510,1510]
```

T1_SEED PASS, Paris OK, erraudit ERROR=0. The 8-bit MX lane reproduces
native-class behavior through identical churn mechanics (4×256-token churn
between probes).

### m4q: 4-bit-K long-generation quality limit (the REAL finding)

```
[essay]  t0_identical=False hashes=['d748a1b1493b','484f71e94d10','484f71e94d10']
         garble={cold:[], rehit:['REP-LOOP'], rehit2:['REP-LOOP']}
         chars=[4053, 13832, 13832]  t1_seed_identical=True  prompt_tok=2145
[code]   t0_identical=True hashes=['835e22925745' x3] no garble  chars=[2547,2547,2547]
[multi]  t0_identical=True hashes=['17d67b534410' x3] no garble  chars=[548,548,548]
[simple] t0_identical=True hashes=['84efc84cc142' x3] no garble  chars=[1510,1510,1510]
```

Interpretation (m8q control isolates the variable):

1. **Not a cache-reuse bug** — m8q walks the SAME churn schedule with zero
   drift; if reuse/eviction were corrupting, both lanes would fail.
2. **One-way transition to a stable attractor** — cold essay is clean
   (4053 chars, stop-terminated); after churn the essay class regenerates
   13832 chars of REP-LOOP, and the attractor hash `484f71e94d10` is stable
   across BOTH rehits. Deterministic degradation, not stochastic corruption.
3. **Mechanism**: mxfp4 stores K at e2m1 (measured rel-L2 ≈ 12%, trace27).
   Long-generation retrieval over ~2.1k-token contexts with 4-bit keys
   loses the precision to re-rank near-tied history — the essay class
   (repeated long-form structure) falls into repetition. Code/multi/simple
   survive: shorter, higher-margin retrieval targets.
4. Same-hash `simple` across m8q AND m4q (`84efc84cc142`) — short-context
   output is dtype-identical; only long-context retrieval degrades.

This matches the wheel's own quality doctrine (P6.15 B6: turboquant_k8v4 =
"quality-safest preset" — 8-bit KEYS; key precision is the retrieval
bottleneck, value precision tolerates 4 bits). **Action: mxk8v4** — e4m3 K +
e2m1 V, 200 B/slot @D=128 (0.78× fp8-KV bytes, 2.56× fp16 capacity): the
TRUE turboquant_k8v4 analogue, implemented same day on this branch.

## Gemma probe root causes (run1 invalid → run2 → run3)

Run1 gemma legs INVALID for probe reasons, NOT engine reasons:

1. **404 longprobe cells** — longprobe hardcoded the qwen served name;
   gemma legs 404'd (16 ERRORs each leg, empty hashes). Engine exonerated:
   m8g==m4g battery texts nearly identical across KV dtypes.
2. **Completions-endpoint semantics** — gemma /v1/completions does
   document-continuation (prompt echo + more questions), dtype-independent;
   fixed by per-model chat endpoint (r154 run2).
3. **`message.reasoning` field (run2 live curl, arc-c24 m8g)** — gemma-4 is
   a thinking model that IGNORES `enable_thinking:false` (kwarg accepted,
   template unaffected) and vLLM 0.30 emits the stream as `message.reasoning`
   (NOT `reasoning_content`). Battery + longprobe read only
   content/reasoning_content → `PROMPT_OK 0 ''` on all three battery
   prompts while the live JSON carried `"reasoning": "<|im_start>user\nThe
   capital of France is Paris.\n..."` (raw/cert24/m8g.run2/, gate_run2 log).

Fixes (run3 script): battery + derived longprobe + suite-13 all capture
`content or reasoning_content or reasoning`; run2 partial m8g banked as
`m8g.run2` (garbage suite 15/15 PASS with the model fix — probes got real
text through content; state_stability deterministic).

## Laws extracted

- Probe text extraction must capture EVERY response channel the server
  emits (`content`, `reasoning_content`, `reasoning`) — thinking models
  move text between channels and vLLM's field name is version-dependent.
- A thinking model's template may ACCEPT `enable_thinking` while ignoring
  it — acceptance is not compliance; verify on the raw JSON.
- 4-bit K is a long-generation quality limit, not a corruption bug: the
  discriminator is a stable one-way attractor with clean control-lane
  behavior under identical churn (bank this exact signature).
- Kill a running gate with SIGKILL-safe teardown in mind: bash does NOT
  run EXIT traps on untrapped SIGTERM — restore W14 manually after
  `kill <gate-pid>` (done 2026-11-12 11:35 UTC, radiance-test Up).

Next: run3 (m8g/m4g with all three fixes) → mkq/mkg gate legs (mxk8v4) →
cert24 rebuild with mxk8v4 (new content commit) → r155 leg-D perf.
