# r211 E1' — 1K timeline decomposition (MX-native paged decode)

**Date**: 2026-11-21 · **Leg**: r211_e1 (arc7 Phase-8, mxfp8 short-ctx
floor pathology) · **Wheel**: arc7mx r210 keeper `7247e83b` ·
**Image**: `vllm-arc:v0.30.0-b70-cert25zc` · **Tool**:
`tools/ar/p147_mxkv/p202_timeline.py` + `r211_e1.sh`

## Verdict

**PASS — H1' CONFIRMED (host-dispatch leg), H2'/H3 demoted.** At
ctx=1024 the MX-native decode call is **HOST-LAUNCH-BOUND**: the device
timeline mirrors the host enqueue timeline (per-call `dev ≈ host`,
inter-call device gap ≈ 2 µs = the python loop overhead). The 0.44-0.87
ms "floor cells" of the r208/r209 chains measured the **host interface
path** (arg rebuilds + C++ split-plan build + launch marshalling), not
kernel occupancy. Even the near-empty mode-3 floor kernel "reads"
330-840 µs. SLM/occupancy (H3) cannot bind at this grid, exactly as
the GRID CORRECTION predicted; the pathology is not in the kernel.

## Method

Per-rep event pairs with NO intermediate sync (queue never drained
artificially): `dev[i]` = elapsed(ev_s[i], ev_e[i]); `gap[i]` =
elapsed(ev_e[i], ev_s[i+1]); `host[i]` = perf_counter_ns around
(record + call + record); plus the classic bracketed cell (one event
pair around N calls) that produced the r208/r210 numbers. One docker
process per `ARC_MXKV_VPROBE` mode (kernel reads env at launch).
Geometry: B2/HK2/HQ8/D128, fragmented block-16 tables, ctx 1024,
splits 5 (r210 1K optimum posture) and 1, 60 reps. Canary vs splits=1
reference per cell (NaN-poisoned out-buffer, r207 law).

## Results (medians, µs; r210 wheel)

### Mode 0 — full kernel

| format | splits | dev | gap | host | bracketed | canary |
|---|---|---|---|---|---|---|
| mxfp8 | 5 | 573.4 | 1.9 | 576.3 | 648.3 | 9.77e-4 |
| mxfp8 | 1 | 697.6 | 2.0 | 699.6 | 717.3 | 0 |
| mxfp4 | 5 | 568.5 | 2.5 | 571.0 | 559.1 | 9.77e-4 |
| mxfp4 | 1 | 521.3 | 1.8 | 520.5 | 538.9 | 0 |
| mxk8v4 | 5 | 528.3 | 2.0 | 527.4 | 540.9 | 9.77e-4 |
| mxk8v4 | 1 | 486.8 | 2.0 | 486.9 | 519.9 | 0 |

### Mode 3 — floor (skip both stages)

| format | splits | dev | gap | host | bracketed |
|---|---|---|---|---|---|
| mxfp8 | 5 | 839.2 | 3.7 | 840.9 | 445.5 |
| mxfp8 | 1 | 469.8 | 2.0 | 470.4 | 461.2 |
| mxfp4 | 5 | 449.8 | 49.2 | 498.9 | 373.9 |
| mxfp4 | 1 | 330.3 | 2.2 | 333.9 | 337.5 |
| mxk8v4 | 5 | 450.8 | 58.9 | 506.8 | 411.1 |
| mxk8v4 | 1 | 318.6 | 1.9 | 319.8 | 312.4 |

### Mode 8 — gemm/epilogue floor

All six cells CANARY_NAN — mode-8's const-P epilogue leaves the NaN
poison in `out` (floor-mode output semantics), NOT a no-launch: the
dev times are real queue-resident event-pair times. Mode-8 cells stay
structural floors (const-fill law). Numbers: dev 232-454, host 311-547,
**gap 80-90 µs** — mode 8 shortens the host path enough that the host
runs ahead and the device idles 80-90 µs between calls: the first
direct bound on true per-call device cost at 1K (≤ ~90 µs incl.
launch turnaround, vs 445-648 µs host).

## Findings

1. **`dev ≈ host` + `gap ≈ 2 µs` in every mode-0 cell** — the events
   bracket host-paced enqueue, not kernel occupancy. Corollary to the
   const-fill floor law: standalone short-ctx floor cells measure the
   host interface, so the r208 "mxfp8 3× floor pathology" was a
   host-path asymmetry, never a kernel property.
2. **Format asymmetry is host-side and splits-conditional**: at s=1
   mxfp8 host 700 vs mxfp4 520 (Δ180 µs); at s=5 they converge
   (576/571). The fatter e4m3-V arg marshalling (scale tensors, plan)
   costs host time the empty kernel never hides at short ctx.
3. **Common ~0.5 ms host term dominates all formats** — vs v2's
   ~0.22 ms total at 1K. The 1K blocker (0.47-0.63× of v2) is the
   `mx_paged_decode` host path: per-call `torch.arange`/`torch.tensor`
   H2D rebuilds (`cu_q`, `sl`, `host_kv_lens`), C++ split-plan build,
   launch marshalling.
4. **Bimodal first-loop host cost** (med 576 / p90 1405; second
   bracketed loop avg 648) — transient warm-up class, not steady
   state; medians reported.
5. 32K/8K regimes are unaffected (kernel time >> host cost there) —
   the 1.80×/2.02×/4.73× results stand.

## Next ladder (E2' replaces E2-E4)

- **E2' host-path A/B**: prebuild `cu_q`/`sl`/`bt`/`host_kv_lens`
  once + cache the split plan; rerun this timeline — expect host to
  collapse, `gap` to open (device exposed), reading the true kernel
  time per format at 1K.
- **E2'' XPU-graph capture+replay** of N calls per format (engine
  posture) — kernel-only per-call cost, graph-captured.
- **E3 engine call-site audit**: what the `vllm_gaudi/xpu` mx_kv lane
  rebuilds per decode step (the telemetry face paid this same host
  path at B=1/1K: 0.385/0.290/0.277 ms) — the fix lands there
  (cache/prebuild per-step constants), then 1K telemetry re-chain.
- E4 (ocloc SLM/register report) DEMOTED to engine-grid-only question.

## Artifacts

`p202_timeline_m{0,3,8}.json`, `r211_e1.log`, `r211_e1_m{0,3,8}.log`
(rig: `/root/build/p199/out`, `/root/build/p199/logs`).
