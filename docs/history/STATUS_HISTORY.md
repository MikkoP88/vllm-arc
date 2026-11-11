# STATUS history — prior certs and the phase ledger

Archived from `docs/STATUS-arc.md` in the P6.38d docs restructure
(2026-11-11). This file is an **append-only archive**: it records
superseded cert identities and closed phases, and never carries
current state — the living doc ([`../STATUS-arc.md`](../STATUS-arc.md))
holds current-cert state only. Per-cert changelog entries:
[`../../CHANGELOG.md`](../../CHANGELOG.md).

## Prior certs — one line each

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
| cert4–cert | P6.5– | earlier image milestones (cert4 posture image; cert3 tag-built process; cert2/cert pre-date later fixes) | CHANGELOG per-cert entries |

## Phase ledger P0–P6.37

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
| P6.37 | 2026-11-10 | close-out: per-config reports, 64 GB-RAM host defaults, `ARC_KVDTYPE` knob + dtype report, patch inventory | `results/2026-11-10-p637-reports/` |
