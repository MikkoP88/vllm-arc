# MIGRATION history — the phase evidence table

Archived from `docs/MIGRATION-arc.md` §6 in the P6.38d docs
restructure (2026-11-11). Append-only archive; the living doc
([`../MIGRATION-arc.md`](../MIGRATION-arc.md)) carries current
identities and procedures. One row per phase/leg-group; the "what it
proved" details live in the cited banks.

| Phase | Evidence (under `results/`) | One-line verdict |
|---|---|---|
| P4 r42–r44 | `2026-10-30-p4-r42-certified-image/` … `-r44-matrix-certified/` | certified-image boot + battery; self-contained image (Fix A/B); 12-cell regression baseline |
| P5 r45/r46/r46t | `2026-10-30-p5-r45-env-log-audit/`, `-r46-soak/`, `-r46t-temps/` | env/log audit + migration addendum; 8 h soak GREEN; real-world campaign — k3 recommended |
| P6.1–P6.7 | `2026-10-30-p6-r47a-ext-hitrate/` … `2026-10-31-p67-r67a-compile-ab/` | external hit-rate artifact; lmcache gates + comparator law; cert3 process; p65 stall capture + cert4; prefill anatomy; compile REVAL rejected |
| P6.10 / P2P | `2026-10-31-p70-r70-cert4-comprehensive/`, `-p71-r71-p2p-analysis/` | cert4-native comprehensive re-measurement + 4 h soak; TP2 transport proven clean (direct P2P 10.23 GB/s) |
| P6.13 | `2026-11-01-p65-end-stall-analysis/`, `-p65-rung3-slice/`, `-p65-cert5/` | END-stall analysis; sliced final-span store default-OFF; cert5 |
| P6.14/P6.15 | `2026-11-02-p614-*`, `2026-11-02-p615-*`, `2026-11-03-p615c-*` | cert6 TQ presets + store fix; bystander-stall taxonomy; t1.0-xhigh collapse mechanism pinned |
| P6.16/P6.17 | `2026-11-03-p616a-c*/`, `-p617a-probcert/` | probabilistic drafting measured then certified as the serve default |
| P6.18 (cert7) | `2026-11-03-p618b-repro/`, `-p618c-roundtest/`, `-p618-cert7/` | TQ corruption reproduced/split; interim dtype refusal; reproducibility audit |
| P6.19 (cert8) | `2026-11-04-p619-tqverify/` | TQ fixed — three spec-verify defects; all-dtype long-return gate zero garble |
| P6.20 (cert9) | `2026-11-04-p620-prefill-opt/` | L3 off-thread store default-ON + L6; prefill proven device-bound |
| P6.21/P6.22 | `2026-11-05-p621*/`, `-p622-ar-lane/` | device-compute residue closed NEGATIVE; AR census instrument banked |
| P6.24 (cert10) | `2026-11-05-p624-fp8-ar/` | oneCCL fp8-wire overlay certified default-OFF; + b/d/e audits |
| P6.25 | `2026-11-06-p625-overlap-door/` | cross-stream AR∥GEMM closed MEASURED — the in-scope improvement ledger is CLOSED |
| P6.26 (cert11) | `2026-11-06-p626-mxfp4/` | quark/MXFP4 lane certified; fp8 default byte-identical |
| P6.27 | `2026-11-06-p627-mxfp8/` | MXFP8 AutoRound checkpoint validated zero-change (no cert) |
| P6.28 (cert12/16/17/18) | `2026-11-06-p628-gemma4/` | gemma-4 chain: G2 freeze closed 3 layers; hd512 FA2 route 34× |
| P6.29 (cert19) | `2026-11-07-p629/` | sliding-window FA2 route; gemma decode +45 %; spec parity closed |
| P6.30 (cert20) | `2026-11-08-p630-mxfp4-perf/` | online mxfp4 W4A16 route — superior online lane |
| P6.31 (cert21) | `2026-11-08-p631-mxfp8-w8a16/` | online mxfp8 W8A16 route 1.37–1.85×; gemma 8-bit capacity law |
| P6.32 | (analysis, in-tree notes) | Xe2 FP8-nativeity verdict + kernel-feasibility map |
| P6.33 (cert22) | `2026-11-08-p633-skinny-m/` | skinny-M kernels banked OFF, serve ≡ cert21 |
| P6.34 | `2026-11-09-p634-jm-ladder/` | joint-matrix ladder laws; research lane closed |
| P6.35 (cert23) | `2026-11-10-p635-telemetry/` | graph-exec double-buffer certified |
| P6.35-V | `2026-11-10-p635-telemetry/vmatrix/` | 11-cell matrix green; project best 128.3 tok/s |
| P6.36 | `2026-11-10-p636-legA/` | prefill = device drain; no remaining lawful lever ≥ 10 % |
| P6.37 | `2026-11-10-p637-reports/` | per-config reports + 64 GB-RAM defaults + `ARC_KVDTYPE` knob |
