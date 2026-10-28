# VERDICT — r18: §10 P2 gate sweep on the defect-b fix, product posture (no injection)

Image `vllm-arc:v0.30.0-b70-p12b2` (fork ca75f1d8). First boot where
`first_request_warmup` installs from the **platform wiring**
(`_XPU_PATCH_MODULES` → `check_and_update_config`) — interlock line present
in the FRONTEND (APIServer pid=16) before any request, plus EngineCore and
both TP workers. All four product patches live in the same log
(graph_safe_output, speculator_full_capture, first_request_warmup, r12b
tracer under ARC_TRACE_FLIP=1).

Posture: mode=none + FULL_DECODE_ONLY (1..320) + **MTP k4** + TP2 — the
§10 P2 gate posture (defects (a)+(b) + spec on graphs).

## Gate rows

| §10 P2 gate | result | evidence |
|---|---|---|
| defects (a)+(b) fixed: FULL graph + MTP boots and survives first request | **PASS** | the exact r16/r17 832-tok flip payload as FIRST traffic (md5 8cc5358e…): interlock fired → small 2.2 s + max-batch 5.6 s (7.8 s total — trimmed sizing from r17a confirmed) → **HTTP 200**, follow-ups 0.43 s / 0.82 s, `ZOMBIE_ONSET` count **0** through the entire suite |
| acceptance telemetry live (p31 gated, every leg) | **PASS** | `SPEC_GATE` counters + log window live on first spec leg on this chain: pos1 0.8756 (hard ≥0.5), log window "Per-position acceptance rate: 0.862, 0.815, 0.789, 0.779, Avg Draft acceptance rate: 81.1%", mean acceptance length 4.24 |
| TPOT c8 ≤ 60 ms | **FAIL — blocked on p11 (by design)** | c8 cells 113.8–185.1 ms (k1k-c8 113.8 warm re-pass; eager tax was ~113 ms flat). Known mechanism (Appendix B QSA floor law): Triton QSA splitk at 0.8% XMX = 58% of the c8 step. This matrix is the **Triton comparison cell** for the p11 ESIMD ≥7× A/B in the same gate row. |
| QSA ESIMD ≥7× vs Triton | **NOT RUN** | p11/p90/p91 not yet ported — the remaining P2 implementation item |

Contract bars (run7n agg 195.7–213.1, solo 57.5–57.7 vs 274.41/79.74):
gap is the planned lever stack (QSA ESIMD → P3 cache/prewarm → P4
acceptance tuning), not a regression; the eager→graphs+MTP chain is
upstream-0.30-untuned by construction.

## Warmup cost accounting (product posture)

First traffic pays 7.8 s (small + ~2-chunk max-batch stage, trimmed r17a
sizing). Every later request is un-taxed (follow-ups 0.43–0.82 s; whole
18-cell suite + run7n ×2 + spec gate ran on the same boot, zero onset).

## Artifacts (legs_pull/)

survival.txt (large-first/small2/large2 + onset), interlock_log.txt
(install lines across processes + warmup stage timings), pymatrix.txt +
k1k_warm.txt (18 cells + warm re-pass), run7n_1/2.txt, spec_gate.txt +
spec_gate.json, r18_serve.sh / r18_boot.sh / r18_survival.sh / r18_bench.sh,
payload_md5.txt (8cc5358e… / 4fd1b3d5…), onset_count.txt.

## Disposition

- Defect-b fix: validated at the **§10 P2 gate posture** (graphs+MTP,
  product wiring). r16 mechanism + r17 counterfactual + r18 gate posture =
  complete evidence chain.
- P2 remains open on ONE row: p11 QSA ESIMD (≥7× + TPOT c8 ≤60 ms re-run).
  Everything else in the P2 row is green.
- p12b branch merge to main: deferred until the P2 row closes (per §11
  merge law — gates then squash-merge).
