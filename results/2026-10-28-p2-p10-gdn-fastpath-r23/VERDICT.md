# VERDICT — p10 GDN metadata fast path: BUDGET LAW MET (0.49 ms p50, max 0.81 ms vs ≤5 ms); TPOT improves everywhere but the step wall barely moves — the remaining wall is the GPU pipeline itself (compile=none replay), not CPU serialization.

Patch: `vllm_gaudi/xpu/patches/gdn_metadata_fastpath.py` (branch
feat/arc-p10-gdn-metadata-fastpath), deployed via PYTHONPATH sitecustomize
on the p12b2 image, product posture verbatim (r18 flags, prefix caching ON
→ mamba align mode). Stock `build` kept for prefill/mixed/zero-spec/over-
limit/non-contiguous shapes (telemetry: fallbacks exercised 252× through
the suite, correctness by delegation).

## Mechanism (probe.txt, idle-GPU micro-probe on the product stack)

| op behind ~290 ms of enqueued work (40× 4096² matmul) | wall |
|---|---|
| `bt[cpu_bool_mask, :5]` (core build's row select) | **678.7 ms** — stream drain PLUS implicit syncs (variable-shape result forces mask nonzero + shape readback) |
| `na[cpu_bool_mask]` (1-D) | 263.3 ms — stream drain via blocking mask H2D |
| `mask_gpu.nonzero()` + `index_select` (patch v1 design) | **275.0 ms — ALSO BLOCKS**: nonzero's count readback is a D2H sync |
| same ops, idle GPU | 0.42 ms |
| equality of results | True / True |

Design consequence: v1 (nonzero+index_select) was discarded mid-leg BEFORE
serve; v2 uses only SLICES — spec rows are front-contiguous (the same
invariant core itself relies on when it slices
`query_start_loc[: num_spec_decodes + 1]`, "padded sequences are always at
the back"), guarded at runtime (`non-contiguous` → stock build). No mask
indexing, no H2D, no nonzero, no 0-dim-GPU `fill_` (would `.item()` and
sync — tail fills use CPU-known scalars; in this branch
`query_start_loc[num_spec_decodes]` == `num_spec_decode_tokens`).

## Budget law (p10 row, plan §4): MET

Serve-time build walls (in-patch sampler, every 256 builds, both workers):
- v2: **p50 0.49 ms, p90 0.59-0.62 ms, max 0.74-0.81 ms** over 4608 fast
  builds (capture + full suite). ≤5 ms/step law met with 6× margin.
- v1 (nonzero) for contrast at capture: p50 1.0 ms but p90 22-60 /
  max 77-115 ms — the nonzero block, confirming the probe.
- Pre-p10 (r19-r22a traces): ~50 ms CPU serial in build, one slow build
  per step (p90 69 ms), gaps "before IndexKernelFunctor" ~63 ms.

## Bench (vs r18 product posture, same seed/cells)

| face | r18 | r23 (p10) | Δ |
|---|---|---|---|
| stepprobe c1 step | 64.4 ms | 62.9 ms | −2.3% |
| stepprobe c8 step | 79.6 ms | 77.4 ms | −2.8% |
| run7n solo | 57.5-57.7 | 59.07-59.33 | **+2.5%** |
| run7n agg | 195.7-213.1 | 206.7-222.3 | +3-5% |
| k1k-c8 TPOT | 135.39 | 115.66 / 110.75 re-pass | **−14.6%** |
| k8k-c8 | 150.81 | 148.14 | −1.8% |
| k32k-c8 | 185.13 | 182.23 | −1.6% |
| k64k-c4 | 133.56 | 131.22 | −1.8% |
| c1 rows (all ctx) | 66.75-75.59 | 64.53-72.72 | −3.3% |
| c16 rows | 275-328 | 282-326 | noise |
| TTFT c1 rows | ≈ | ≈ | unchanged |
| spec gate | pos1 0.8821, acc 0.7028, len 3.81 | pos1 0.8768, acc 0.6979, len 3.79 | identical (correctness) |
| onset count | 0 | **0** | interlock holds |

## The decisive finding: the step wall moved to the GPU pipeline

Removing 50 ms of CPU-serial build moved the c1 step only 64.4→62.9 ms.
The build was running on the worker thread IN PARALLEL with the
scheduler-thread copy_event wait (r22a thread topology); with the build
off the critical path, the step is now paced by the GPU pipeline itself:
replay + eager tail + D2H ≈ 63 ms at c1 (5 tokens, 27B fp8, TP2) ≈
~1 ms/layer — the compile=none unfused replay. W14's 33 ms c8 steps run
graphs+compile; champion D's ~50 ms c8 TPOT runs compiled+MTP. The next
TPOT lever is the **compile A/B** (plan P4 item — evidence now promotes it
into the P2 critical path), plus output-sync overlap (p50 polish).

## P2 gate status after r23

- async ACTIVE ✓ (r22a), decomposition published ✓ (r22a stepprobe +
  r23), metadata build ≤5 ms ✓ (r23, this leg).
- TPOT c8 ≤60 ms: still OPEN — matrix c8 110.75-182.23, steady step 77.4.
  Path: compile A/B (replay length) — the CPU side is no longer the wall.
- Correctness: spec-gate acceptance identical, onset 0, fallback paths
  exercised; stock build handles every non-steady shape.

## Artifacts (legs_pull/)

probe.txt (+ p10_probe.py) — mechanism evidence incl. the nonzero
falsification; p10_stats.txt — budget-law serve-time sampler lines;
stepprobe.txt — step walls before/after; pymatrix.txt (18 cells) +
k1k_warm.txt; run7n_1/2.txt; spec_gate.json/.txt; onset_count.txt;
p23_serve/boot/bench.sh; p10_sitecustomize.py (deployment);
gdn_metadata_fastpath.py is the repo file (deployed copy verified by
matching log line numbers in serve_r23_final.log on the rig).
