# vllm-arc — status & release ledger

Current-state holder: the certified-release detail (this cert only), the
phase ledger (one line per phase), and the improvement-lane disposition.
Prior-release narratives and per-cert history live in
[`../CHANGELOG.md`](../CHANGELOG.md) (main-currency law — this file never
carries superseded-cert detail). Nothing here is separate evidence — every
claim traces to the banked `results/` legs and the two ledger docs
([`TESTING-arc.md`](TESTING-arc.md) — validation record,
[`MIGRATION-arc.md`](MIGRATION-arc.md) — evidence table).

**Currency: `rel/v0.30.0-b70-cert23` (`vllm-arc:v0.30.0-b70-cert23`).**
This file rides the main-currency law: at every certification close-out
the README anchors repoint (checklist in `tools/check_currency.sh`) and
this ledger's release section is rewritten to the new cert in the same
change.

## Certified release — `rel/v0.30.0-b70-cert23`

Every gate GREEN with banked evidence. cert23 = **cert22 + the P6.35
graph-exec double-buffer** (r135 five-leg gate + r135b control; one
patch module, no wheel change, no serve-line var — the engine bytes
elsewhere are cert22's):

- **The defect it removes** — torch 2.13 `XPUGraph.replay()` on the
  SAME exec object blocks host-side until that exec's prior launch
  retires (p135i microbench: 13.96 ms same-graph in-flight vs 8–12 µs
  for two execs sharing one capture pool alternating). The fix
  (`vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`) captures every
  graph ×2 into the manager's shared pool and alternates replays per
  desc (swap-technique wrappers; second-capture failure degrades to
  stock semantics with a WARN — never a boot death). Leg J: replay max
  24 → 0.37 ms, engine serial work 40.8 → 12.7 ms/step, alternation
  proven (12 XRP ids × 59 over 119 steps).
- **E2E-honest position** — the gate A/B reads **NEUTRAL at every
  measured posture**: fp8-k3 94.5 (db on) vs 94.3 tok/s (db off),
  ITL p50 40.49 both; mx4k5 121.0 vs 123.9 (sweep). Leg-F's spiky
  81.6 "stock baseline" was ~3.8 ms/step instrumentation tax — the
  clean engine's step was never replay-limited (the block overlaps
  the device-floor work the step waits out anyway; leg K: the
  copy_event wait waits out real device work). Ships as a **costless
  driver-level hygiene fix**: +29 s boot (the ×2 captures), zero
  extra memory (shared pool), byte-identical semantics.
- **Knob** — `VLLM_ARC_GRAPH_EXEC_DB` (default `1` = ACTIVE; `=0` =
  stock single-exec replays — kill-switch proven in the gate: dboff2
  boot shows zero db markers and the stock face).
- **The P6.35 performance deliverable is POSTURE, not engine** (all on
  stock cert22-class bytes; serve-line choices on cert23): spec depth
  **k5 = +41% decode** (115.2 tok/s solo U8k vs 81.6 at k3); **mxfp4
  W4A16 + k5 = 123.9 tok/s (+52%) with TTFT 3.98 s < the 4.55 s
  baseline** — the recommended decode posture (`ARC_QUANT=mxfp4
  ARC_MXFP4_MODE=w4a16 ARC_QUANTCFG='{"ignore":["*visual*",
  "*linear_attn*","mtp.fc"]}' ARC_GMU=0.85 ARC_SPEC=5 ARC_FP8=0`, BASE
  checkpoint). mxfp8 W8A16 is a decode dud (83.8 ≈ baseline — no
  weight savings, bf16-activation tax). Rate accounting corrected:
  spec-decode SSE streams multi-token chunks (600 tokens/127 chunks at
  fp8k4) — honest numerator is max_tokens. Evidence:
  `results/2026-11-10-p635-telemetry/` (legs A–K + `sweep/` +
  `cert23/`).
- **P6.35-V full validation matrix (cert23, r136+r136b)** — 11 cells,
  both models, all online quants × spec depths, uniform protocol, 0
  tracebacks, db markers correct at every depth. Qwen: mx4-W4A16 wins
  every spec column — **k5 re-read 128.3 tok/s (+36% vs the clean
  fp8-k3 base 94.5, project best; 5.93 tok/chunk acceptance)**;
  spec0 column fp8 32.8 / mx8 28.2 / mx4 36.1. Gemma (spec0): **mx4
  37.1 tok/s = +43% vs mxfp8 26.0** (ITL 27.0 vs 38.5 ms — the W4A16
  weight-streaming win is model-agnostic); **online fp8 boots + serves
  (26.6) — new capability**; mxfp8 26.0 repeatable. mxfp8 deprecated
  for decode (below fp8 and mx4 at every depth, both models). TTFT:
  fp8-k5 3.48 s best; mx4 prefill cost bounded (worst 5.11 s at k5).
  Winner postures: qwen `ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 QCFG-ignore
  GMU0.85 SPEC5 FP8=0` (BASE ckpt); gemma same quant + `GMU0.8 LMC=0
  RPC=1200 MAXLEN=40960`. Evidence: `results/2026-11-10-p635-telemetry/
  vmatrix/` (incident note: 27B container rm >30 s caused a
  stale-record cascade — data banked before every marker; watchdog +
  hardened r136b teardown re-ran the 3 affected cells clean).
- **P6.36 — the final bottleneck decomposition (2026-11-10, evidence-only;
  cert23 stands, no engine bytes changed)**: the prefill window is
  **device drain** — per-rank epoch telemetry (ARR/SCHE/ESTEP/WEXE/REXE)
  decomposed U32k = 22 ms dispatch | 2.6 s exec walls | **10.9 s (80 %)
  engine-core `future.result()` waits on chunk device completion** |
  0.5 s tail; cross-rank skew ZERO (lockstep to 0.1 ms — the banked
  "1.87 s rank-skew block" was the drain seen through a host-blocking
  profiler); scheduler/queue sub-ms; decode round trip ≤2 ms ⇒ the
  inter-chunk gaps ARE device completion (device prefill rate
  ≈ 2.1–2.9 k tok/s at every dtype and chunk size). B discriminators
  ALL negative: MNBT 16384 wins U8k −0.6 s but collapses U32k +8 s;
  4k neutral; bf16 wide-math loses everywhere (Xe2's fp8
  counter-operand-conversion path is prefill-OPTIMAL). Decode D2 host
  tail split (4.4 ms enqueue + ~1.5 ms round trip vs ~34–35 ms device
  floor) ⇒ ceiling +8–10 %, not worth the risk to the champion.
  **No remaining lawful lever ≥ 10 %.** Evidence:
  `results/2026-11-10-p636-legA/` (PR #96).
- **P6.37 — the owner close-out (2026-11-10)**: per-config performance
  reports ([`docs/reports/`](reports/README.md) — model × quant × MTP,
  decode/TTFT/prefill/ITL at 1K–128K, the MTP-k3 temperature sweep,
  line charts, copyable configs; bank `results/2026-11-10-p637-reports/`
  — r138/r138b, 13 cells, 61 faces, 0 tracebacks/timeouts, every 8K
  point re-verified in-band vs P6.35-V; no engine bytes changed —
  cert23 stands); the 64 GB-RAM host compatibility layer (close-out F:
  `--shm-size 64g → 16g`, `LMCACHE_MAX_LOCAL_CPU_SIZE 100 → 16` —
  validated by the r138 run itself); the kv-cache-dtype comparison
  (close-out G: `ARC_KVDTYPE` knob — `fp8` ≈ the fp8_e4m3 incumbent,
  `auto` (fp16 KV) refuses at maxlen 262144 (8.73 > 6.78 GiB) but
  serves at 131072 slightly faster, `fp8_e5m2` hard-refused (no e5m2
  KV path in esimd FA2 on Xe2); TurboQuant presets cited from the
  cert8/cert9 banks); and the patch inventory
  ([`docs/PATCHES-arc.md`](PATCHES-arc.md)).
- History: every prior cert (cert → cert22) — full content, validation
  legs, fork SHAs, supersession notes: [`../CHANGELOG.md`](../CHANGELOG.md).

### Prior certs — one line each (full narratives: CHANGELOG)

| Cert | Phase | Content (one line) | Evidence |
|---|---|---|---|
| cert22 | P6.33 | skinny-M kernel family (arc6 wheel), correct but falsified e2e under graph replay — **banked OFF**, serve ≡ cert21 | `results/2026-11-08-p633-skinny-m/` |
| cert21 | P6.31 | online-mxfp8 **W8A16 route** (arc5 wheel): qwen 1.37–1.55× / gemma 1.57–1.85× over stock W8A8; checkpoint-fp8 keeps the perf crown | `results/2026-11-08-p631-mxfp8-w8a16/` |
| cert20 | P6.30 | online-mxfp4 **W4A16 route** (arc4 wheel): superior to online mxfp8 on every axis 1.47–1.75×; W4A4 penalty root-caused (act-quant + oneDNN JIT-gate) | `results/2026-11-08-p630-mxfp4-perf/` |
| cert19 | P6.29 | sliding-window hd256 **FA2 route**: 18–25× class microbench; gemma faces 4.9/17.2/110.9/0.9 s, decode 28.4/195 tok/s | `results/2026-11-07-p629/` |
| cert18 | P6.28 | full-attention hd512 **FA2 route**: 34× class; gemma faces 2.2–16.3× faster, qwen discriminator untouched | `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/` |
| cert17 | P6.28 | gemma-4 validation chain (cert12 launcher table + cert16 warm-gate v4 + cert17 KVGATE); G2 freeze closed in three layers | `results/2026-11-06-p628-gemma4/` |
| cert11 | P6.26 | `quark`/`supports_mx` platform patch — both MXFP4 checkpoints certified; fp8 default unchanged | `results/2026-11-06-p626-mxfp4/` |
| cert10 | P6.24 | oneCCL **fp8-wire allreduce overlay** (default OFF): 1.65× at the dominant AR; every p65 TTFT face beats cert9 opt-in | `results/2026-11-05-p624-fp8-ar/` |
| cert9 | P6.20 | prefill host-path: L3 off-thread LMCache store (default ON) + L6 dispatch-late + `step_trace`; prefill proven device-bound | `results/2026-11-04-p620-prefill-opt/` |
| cert8 | P6.19 | **TurboQuant spec-verify FIX** (`tq_spec_fix.py`): all four presets serve clean; capacity 1.98×–3.68× | `results/2026-11-04-p619-tqverify/` |
| cert7 | P6.18 | TQ corruption reproduction + interim dtype refusal (superseded by cert8 next day) | `results/2026-11-03-p618-cert7/` |
| cert6 | P6.14 | TQ-aware store fix + all four presets validated (the P6.17 record faces were measured on this image) | `results/2026-11-02-p614-*/` |
| cert5 | P6.13 | sliced final-span store, default OFF (`LMC_STORE_SLICE_TOKENS`) | `results/2026-11-01-p65-cert5/` |
| cert4–cert | P6.5– | earlier image milestones (cert4 posture image; cert3 tag-built process; cert2/cert pre-date later fixes) | CHANGELOG |

## Project status — P0–P6.37 complete (close-out delivered)

All phases closed with evidence. One line each; per-phase records:
[`TESTING-arc.md`](TESTING-arc.md) (validation record) and
[`MIGRATION-arc.md`](MIGRATION-arc.md) (evidence table); per-cert
narratives: [`../CHANGELOG.md`](../CHANGELOG.md).

| Phase | Closed | One-line outcome | Evidence |
|---|---|---|---|
| P0–P5 | 2026-10-27…30 | ground truth → first serve; engine hardening (P2); cache lane (P3); certification battery + certified image + 12-cell matrix (P4); audits + 8 h soak + real-world campaign, k3 recommended (P5) | per-phase `results/` dirs |
| P6.1–P6.15 | 2026-10-30…11-03 | measurement/analysis phases: external hit-rate, lmcache gates, p65 stall capture (cert4), prefill anatomy, dtype/stall probes + collapse mechanism, draft-sampling research | per-phase `results/` dirs |
| P6.16/P6.17 | 2026-11-03 | **probabilistic drafting = the certified default** (the correct sampler at temperature); every record face re-measured on that lane | `results/2026-11-03-p617a-probcert/` |
| P6.18 | 2026-11-03 | TQ corruption reproduced + split (Class A/B); interim dtype refusal = cert7 | `results/2026-11-03-p618-cert7/` |
| P6.19 | 2026-11-04 | **TQ definitive fix** — three stacked spec-verify defects; all-dtype gate zero garble = cert8 | `results/2026-11-04-p619-tqverify/` |
| P6.20 | 2026-11-04 | prefill host-path (L3 off-thread store + L6); prefill proven device-bound = cert9 | `results/2026-11-04-p620-prefill-opt/` |
| P6.21/P6.22 | 2026-11-05 | device-compute residue attacked head-to-head — ar_quant, FA ladder, ESIMD v2, one-shot AR: **every lane closed NEGATIVE** | `results/2026-11-05-p621*/`, `…-p622-ar-lane/` |
| P6.23 | 2026-11-05 | docs-currency audit — 12 findings fixed | `results/` audit notes |
| P6.24 (+b) | 2026-11-05/06 | **oneCCL fp8-wire overlay** (default OFF, 1.65× dominant AR) = cert10; post-release patch audit dispositioned | `results/2026-11-05-p624-fp8-ar/` |
| P6.25 | 2026-11-06 | overlap-door audit: cross-stream AR∥GEMM closed MEASURED (slower than serial at every k) — the in-scope ledger is CLOSED | `results/2026-11-06-p625-overlap-door/` |
| P6.26 | 2026-11-06 | MXFP4 checkpoints certified (`quark` platform patch) = cert11; fp8 default unchanged | `results/2026-11-06-p626-mxfp4/` |
| P6.27 | 2026-11-06 | MXFP8 AutoRound checkpoint validated zero-change on cert11 (no cert needed) | `results/2026-11-06-p627-mxfp8/` |
| P6.28 | 2026-11-06/07 | gemma-4 campaign: G2 freeze closed 3 layers (cert12/16/17); hd512 FA2 route 34× (cert18) | `results/2026-11-06-p628-gemma4/` |
| P6.29 | 2026-11-07 | sliding-window FA2 route 18–25× class (cert19); spec parity dispositioned closed for gemma | `results/2026-11-07-p629/` |
| P6.30 | 2026-11-07/08 | online-mxfp4 **W4A16 route** — superior online lane (cert20); W4A4 penalty root-caused | `results/2026-11-08-p630-mxfp4-perf/` |
| P6.31 | 2026-11-08 | online-mxfp8 **W8A16 route** 1.37–1.85× (cert21); W4A8 joint dtype banked closed-negative | `results/2026-11-08-p631-mxfp8-w8a16/` |
| P6.32 | 2026-11-08 | Xe2 FP8-nativeity verdict: no native FP8 dot on Xe2; kernel-feasibility map | analysis notes |
| P6.33 | 2026-11-08/09 | skinny-M kernels built + falsified under graph replay — **banked OFF** = cert22 (serve ≡ cert21) | `results/2026-11-08-p633-skinny-m/` |
| P6.34 | 2026-11-09 | joint-matrix ladder research: serialization + SLM-staging laws; never certifies — closed research-only | `results/2026-11-09-p634-jm-ladder/` |
| P6.35 (+V) | 2026-11-10 | **graph-exec double-buffer = cert23** (E2E-neutral hygiene); posture findings: mx4+k5 128.3 tok/s project best | `results/2026-11-10-p635-telemetry/` |
| P6.36 | 2026-11-10 | final decomposition: prefill = device drain, skew zero, B discriminators negative — **no remaining lawful lever ≥ 10 %**; cert23 stands | `results/2026-11-10-p636-legA/` |
| P6.37 | 2026-11-10 | owner close-out: per-config reports, 64 GB-RAM host defaults, `ARC_KVDTYPE` knob + dtype report, patch inventory | `results/2026-11-10-p637-reports/` |

## Improvement-lane status — the in-scope ledger is CLOSED (P6.24; audited exhaustive P6.25)

Every prefill device slice is dispositioned by measurement — GEMM 39.7 % (fastest on-image route already deployed, r40/r41; r101 census: ~99 % of the per-card dense ceiling — no headroom), allreduce ~30 % (**quantized e4m3 wire deployed on cert10** — 1.65× at the dominant AR opt-in, 82 % of the measured wire floor; r100 proved the fabric itself the binding limit: writes 9.85 GB/s flat across store widths, reads 2.2× slower, overlap collapses the link), FA ~20 % (ESIMD v2 ceiling-bound at 0.33× stock — P6.21c), GDN ~5 % (marginal), norms/act ~5 % (small), host path (clean — P6.20). The stack sits at the practical in-scope floor. The two P6.22 candidate routes are now: quantized AR **DEPLOYED** (cert10, default-off env `CCL_SYCL_ALLREDUCE_FP8=1`), and AR/compute overlap CLOSED at every level: r83 same-stream (the device stream is compute-saturated at prefill), r100 fabric (concurrency collapses the link), and **r101 cross-stream co-tenancy (P6.25)** — the async-TP-class door: enqueue is async (0.18 ms) but the device time-slices AR and GEMM kernels (concurrent AR inflates GEMM latency 1.97× stock / 1.58× fp8), so k-sub-batch overlap is slower than serial at every k on both wires (best 0.969×/0.966×) — closed schedule- and wire-independent. Residual headroom is bounded ≤1–2 % class (AR-internal pipelining r100 ~1 %; fp8 quant-pass fusion into GEMMs — a novel cross-library contract that only relocates the same EU work). Transport was proven clean earlier (r71: direct P2P, 10.23 GB/s D2D, all collectives on the `topo sycl` device path; `tools/pcie/README.md`).
