# VERDICT — r22a: async ACTIVE verified; mamba-align theory FALSIFIED; the 58 ms sync is honest in-order pipeline wait whose length the gdn build sets. p10 is the single TPOT lever.

Posture: p12b2 image (fork ca75f1d8+), product flags verbatim except
`--no-enable-prefix-caching` (the A/B lever). Two boots: profiler boot
(r22a_serve.sh, traces + survival) then profiler-free boot
(r22a_serve_noprof.sh, bench). Rig restored to W14 after the leg.

## Finding 1 — async scheduling is ACTIVE (P2 row condition 1 closed)

- Source law: core 0.30 auto-enables in the `else` branch of
  config/vllm.py:1420-1500 with NO log line; every disable gate
  (pooling / spec family / executor) logs a warning. Both boot logs:
  zero async-disable warnings, align count 0.
- Empirical: async_utils.py frames present in both traces (scheduler
  thread tid 7732 runs get_output + the sync; worker tid 7384 runs the
  builds — they run in PARALLEL).
- Boot-log check + core skip-condition check both PASS.

## Finding 2 — mamba align-mode is NOT the 58 ms (theory falsified)

Chain: prefix caching OFF → MambaModelConfig (models/config.py:617) forces
mamba_cache_mode 'none' → gpu_model_runner.py:2119 skip condition true →
needs_cpu_accepted_counts path skipped. Clean A/B, same load:

| metric               | r21b (align) | r22a (none) |
|----------------------|--------------|-------------|
| step wall            | 90.1 ms      | 87.1 ms     |
| Event.sync p50       | 58.02 ms     | 58.58 ms    |
| get_output p50       | 58.04 ms     | 58.60 ms    |
| prepare_attn total   | 5.75 s       | 5.87 s      |

Identical within noise ⇒ the 58 ms is AsyncOutput.get_output's
copy_event.synchronize() on the scheduler thread, mode-independent.
"4.2% GPU busy" in traces is an artifact (kernels inside command-graph
replays are invisible).

## Finding 3 — stepprobe: step walls without a profiler (new instrument)

Streaming completions; with MTP each engine step = one SSE chunk, so
inter-chunk gap = step wall. Counting-prompt acceptance from the same
window (chunks/stream × 4.1 tok = max_tokens ✓):

- c1 step = 64.4 ms (p90 64.7) — FLAT, tight
- c8 step = 79.6 ms (p90 79.9) — batch scaling only +15 ms

Cross-checks: solo run7n 57.9 tok/s ⇒ 64.4/3.73 per-token ✓ (spec-gate
mean length 3.81); matrix random-prompt TPOT c1 66.7 ms ≈ step/1 (acc≈1
on random tokens) ✓. All TPOT faces reconcile to ONE step-wall law:
TPOT ≈ step / acceptance.

## Finding 4 — evprobe: XPU Event.synchronize is HONEST (no pathology)

| probe                                          | result          |
|------------------------------------------------|-----------------|
| completed-event sync, idle GPU                 | 0.003 ms        |
| query() spin, completed                        | 0.002 ms        |
| waiting sync behind 368 ms enqueued matmuls    | 274 ms (drained honestly) |
| Event(blocking=True) ctor                      | REJECTED by torch.xpu.Event |
| side-stream AsyncOutput replica, completed     | 0.010 ms        |

⇒ No fixed event cost, no blocking-mode semantics. The product's 58 ms
is real in-order GPU pipeline time: copy_stream waits on the main
stream, which holds graph replay + the gdn build's eager tail
(aten::index/_to_copy/copy_ chains, ~5.2 s totals per trace window) +
D2H; and the next step's replay cannot be enqueued until the build's
~50 ms CPU section completes on the worker thread. The build sets the
step length TWICE (CPU serial + stream tail).

## Finding 5 — bench A/B (uncached vs cached posture)

- Posture-clean cells: c1 rows identical to r18 (TPOT 66.7-76.9 ms;
  TTFT 25718/58173 ms vs 25764/58546); run7n agg 183-218 vs
  195.7-213.1; solo 57.8-58.1 vs 57.5-57.7. k1k-c8 (fresh prompts in
  BOTH postures): 135.39 → 96.11/96.95 ms.
- Uncached k≥8k c≥2 cells collapse (TPOT 689-3641 ms): decode starves
  behind other streams' chunked prefills — r18's cells there were
  cache-aided (TTFT 3.2 s vs 16.9 s at k8k-c4). The matrix is a
  cache-dependent instrument at c≥2; no-cache is NOT a shippable
  posture (P3 lmcache builds on caching).
- Spec gate: pos1 0.8821 ≥ 0.5 hard PASS; acceptance 0.7028 vs 0.85
  target (miss recorded, P4 tuning). Onset count 0 through the full
  suite — interlock holds with caching OFF too.

## Disposition

- P2 row: async-ACTIVE ✓ (this leg), decomposition published ✓
  (step_r21b/step_r22a + stepprobe + evprobe), TPOT c8 still OPEN
  (steady ~80 ms step / 96-97 measured > 60 gate).
- p10 (gdn metadata build ≤ 5 ms/step) is THE lever and now has a
  mechanism-backed budget: build 50 ms → ≤5 ms cuts the step to
  ~30-35 ms at c8 ⇒ TPOT c8 (acc≈1) ≈ 35 ms ≤ 60 gate; solo ≈
  35/3.7 ≈ 9.5 ms/tok ≈ 105 tok/s > 79.74 bar. p50 (output-sync
  overlap) is secondary polish once the pipeline is short.
- New laws banked: (a) profiler aggregation hang — with_stack +
  torch_profiler_dump_cuda_time_total=True wedges workers ≥15 min in
  device_time_total after traces are written; profile legs LAST on a
  boot, always set dump_cuda_time_total false; (b) stop radiance-test
  BEFORE arc legs (port-8000 collision = false-HEALTHY + engine death).

## Artifacts (legs_pull/)

step_r21b.txt / step_r22a.txt (trace A/B); stepprobe.txt (c1/c8 step
walls); evprobe.txt (event semantics); pymatrix.txt (18 cells) +
k1k_warm.txt (re-pass) + spec_gate.json/.txt; run7n_1/2.txt;
survival.txt (flip payload 200 in 6.72 s, onset 0); onset_count.txt;
serve/boot/bench/survival/prof scripts + step/stepprobe/evprobe
analyzers. Raw traces stay on the rig (/root/build/r22/*rank0*).
