# VERDICT — r25 GPU-side phase split (the plan's "r24 step-split law", Appendix B): step 100% accounted — forward replay 49.0–54.5 ms (~75%, 0.77 ms/layer unfused content vs ~23 ms BW floor) + tail 16.6–17.4 ms (~26%: sample+verify 2.4–2.8 healthy, draft = 4 tiny replays 14.1–14.6 ms at ~3.5 ms fixed per-replay overhead, tailres ≈ 0) + **fgap=bgap=0** — stream saturated end-to-end; overlap/sync lanes DEAD, the only levers are replay CONTENT (fusion, collectives, replay count).

Posture: p12b2 image, product flags verbatim (mode=none + FULL_DECODE_ONLY,
MTP k4, prefix caching ON), p10 fastpath + the r25 diagnostic phase timer via
PYTHONPATH=/root/p25cap (sitecustomize meta-path hooks). Boot healthy
17:16:38; timer installed in both TP workers.

## Instrument (legs_pull/scripts/r25_phase_timer.py)

Wraps the **V2 model runner** — vLLM 0.30 defaults to Model Runner V2 here
(`XPUModelRunnerV2` = `vllm.v1.worker.gpu.model_runner.GPUModelRunner`, no
V1-fallback warning in the boot log; `use_v2_model_runner` defaults True when
Triton present and no unsupported feature). The V2 async step is SPLIT:
`execute_model` (forward: prep + target verify replay launch) then the async
continuation `sample_tokens` (sample+verify, MTP drafter `propose`, post-
process/copies). Timer wraps forward/tail/sample/draft with XPU event pairs
(`enable_timing=True` — plain events return elapsed 0), logs per-phase p50s
every 96 steps, both TP ranks. Cross-phase stream gaps recorded as
pseudo-phases fgap (forward_end→tail_start) / bgap (tail_end→next
forward_start).

Upstream ships the SAME pattern (`StepTimingCollector`,
`vllm/v1/worker/gpu/async_utils.py:48`) but wires it only to
adaptive-verification capture profiling — this leg wired it to steady state.
V1-module wraps (`vllm/v1/worker/gpu_model_runner.py`) are DEAD on this
posture — future instrumentation must target the V2 module.

## The split (steady decode, TP0 ≡ TP1)

| phase | p50 | note |
|---|---|---|
| forward | 49.0–54.5 ms | c1 49.0 / c8 54.5; target verify FULL replay, ~0.77 ms/layer |
| tail | 16.6–17.4 ms | sample_tokens continuation |
| — sample+verify | 2.4–2.8 ms | healthy |
| — draft (4 MTP replays) | 14.1–14.6 ms | ~3.5 ms fixed per-replay overhead vs ~1–2 ms compute |
| — tailres (post/copies) | ≈ 0 (max 6.2) | |
| fgap | 0.0 (max 0.2) | tail enqueued behind forward, no idle |
| bgap | 0.0 (max spikes = inter-leg idle) | scheduler turnaround nil |

Reconciliation: 49.0+16.6=65.6 ≈ stepprobe-under-timer c1 65.3 ✓;
54.5+17.4 ≈ c8 80.0 ✓.

## Controls

- spec gate PASS pos1=0.8804, acc 0.7045, len 3.82 (product band 0.8768–0.9082)
- onset=0; p10 fastpath intact (p50 0.48–0.51 ms, budget law held)
- **Timer distortion law**: enable_timing costs +3.8% step wall (62.9→65.3)
  / −3.5% solo (59.1→57.0) ≈ +2.4 ms/step event stalls → ALL gate and cert
  legs run TIMERS-OFF; timers are for profile legs only, profile legs LAST.

## Disposition (per plan §12 r24 revision)

- p50-for-wall / output-sync overlap: DEAD (fgap=bgap=0 — nothing to overlap).
- Next levers: **S1/p32** FULL_AND_PIECEWISE compile posture (forward → ≤45 ms
  c1; BW floor ~23 ms = 27 GB fp8 ÷ TP2 ÷ 608 GB/s; W14 33 ms whole-step
  control), **p33** draft-replay chain (draft → ≤7 ms), AR-attribution leg.
- Instrument laws banked: hook the V2 module (`gpu.model_runner`), not V1;
  `find_spec`'s `target` PARAM shadows same-named closure vars (boot-1 miss);
  XPU events need `enable_timing=True` (boot-3 empty splits).

## Artifacts (legs_pull/)

phase_log.txt (all splits, TP0+TP1); stepprobe.txt; run7n_1.txt (solo
distortion control); spec_gate.{json,txt}; p10_stats.txt; onset_count.txt;
scripts/{p25_serve,p25_boot,p25_bench}.sh + p25_sitecustomize.py +
r25_phase_timer.py. Rig restored to W14 (health 200, 17:33).
