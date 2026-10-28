# VERDICT — P2 posture pivot: graphs-OFF eager serves the full matrix clean; defect-b closed at the posture level

## The leg (arc-p2-graph-20261028-032625, image 02961955b15a, fork 986b2c49)

ARC_LEG_GRAPH=0 → VLLM_XPU_ENABLE_XPU_GRAPH unset → serve log states
`graph posture cudagraph_mode=NONE (VLLM_XPU_ENABLE_XPU_GRAPH=False)` —
no capture sessions exist, so the recording poison of p12b rounds 1-7
cannot arm.

Results — FIRST fully-passing p2 leg in the project:

- Full k1k matrix: every cell ok=N/N (k1k/k8k/k16k c1-c16, k32k-c1, k64k-c1)
- 315 HTTP 200s, ZERO requests failed, ZERO zombie/storm/recording lines
  over the entire leg (the flip window that killed every graph leg came
  and went with no effect)
- p31 gate expect-off PASS (non-spec posture pure)
- Platform gates PASS (ArcXpuPlatform, XE2=True, meminfo, model identity
  qwen3.8-27b-fp8)

## First eager numbers (k1k sanity matrix — NOT the contract bench)

| cell | tpot_ms | agg_tps |
|---|---|---|
| k1k-c8 | 94.0 | 75.8 |
| k1k-c16 | 94.0 | 136.1 |
| k8k-c16 | 99.4 | 129.8 |
| k16k-c16 | 123.4 | 97.1 |

Contract shape: TPOT c8 ≤ 60 ms and agg ≥ 274.41 are NOT met by eager
non-spec decode (~94 ms / ~136 agg) — as expected: the decode lever in
the plan is MTP k4 speculative decode (the spec leg, next in §12 order),
not the graph lane. Graph mode only ever wrapped decode steps; prefill
runs eager in BOTH postures, so prefill parity is unaffected by the pivot
(measured by its own harness later).

## Posture decision recorded

- Graph lane: ABANDONED for serving (p12b r7 verdict — permanent on-use
  queue flip; results/2026-10-28-p2-graph-p12b-r7/).
- p12b hygiene layers: dormant on this lane (guards arm only on the first
  flip, which never comes without graphs); retained in-tree as proven
  integrity keepers.
- Next: SPEC leg (ARC_LEG_GRAPH=0 ARC_LEG_SPEC=1) — p32 coercion, MTP k4,
  p31 acceptance (pos1 ≥ 0.5 / 0.85) — then the Run-7n contract bench on
  eager+spec.

Raws: this directory (serve_full.log = full serve log; k1k_sanity.txt =
matrix cells; rig leg dir arc-p2-graph-20261028-032625).
