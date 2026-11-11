# TESTING history — the per-phase validation record

Archived from `docs/TESTING-arc.md` ("Validation record" table) in the
P6.38d docs restructure (2026-11-11). Append-only archive; the living
doc ([`../TESTING-arc.md`](../TESTING-arc.md)) carries the current
suites, gates, and laws. The authoritative narratives for every row are
the cited `results/` VERDICTs.

| Gate | Status | Evidence |
|---|---|---|
| P0 venv ground truth | GREEN | `results/2026-10-27-p0-venv-groundtruth/` |
| P1 platform + first serve | GREEN | `results/2026-10-27-p1-platform/` |
| P2 spec/graph defects fixed (survival, capture, speculator) | GREEN | `results/2026-10-28-p2-*` (34 legs) |
| P3 cache lane (connector, hybrid state, elide/memo, tail-split) | GREEN | `results/2026-10-2{8,9}-p3-r3*` |
| P4 certification battery + certified image + k5 adoption + 12-cell matrix | GREEN | `results/2026-10-30-p4-r42/r43/r44/` |
| P5 env/doc/lint/Dockerfile audits + mypy delta=0 | GREEN | `results/2026-10-30-p5-r45-env-log-audit/`, rig leg `arc-p5-r46` |
| 64k TTFT face | GREEN (40,926 ms, concurrent-load conservative — cert-era); cert4-native solo U64k: 39.19–40.26 s | rig leg `arc-p5-r46/ttft64k_t1.txt`; `results/2026-10-31-p70-r70-cert4-comprehensive/` |
| r46 8 h soak | GREEN — 46 cycles, zero errors, integrity 46/46, golden 5/5 zero-drift, pos-1 0.8919 steady, health 200 end-state | `results/2026-10-30-p5-r46-soak/` |
| r46t real-world campaign (temps × depths × thinking × garbage) | GREEN — k3 recommended; garbage v2 5/5 suites 15/15; MTP 1.57–2.05× over no-spec | `results/2026-10-30-p5-r46t-temps/` |
| r70 comprehensive cert4 measurement | GREEN legs a+b (greedy/sweep/ladder/warm/alt + no-spec A/B + tier attribution + r46t parity) + leg c soak GREEN (56/56 cycles, 0 battery/integrity/engine failures, no drift) | `results/2026-10-31-p70-r70-cert4-comprehensive/` |
| P6.13 cert5 release | GREEN single-flow on the tag-built image: manifest fork_sha == tag `3e268627`, p48 28/28 + p73 slice-hunk 11/11 image-native, integrity 5/5 + det 3/3, WARN class-set identical to cert4 (44 == 44 lines), 0 ERROR/Traceback; slice face U64k TTFT 50.77 s, `63232`-class 0 / 3328×38 exact | `results/2026-11-01-p65-cert5/` |
| P6.14 TurboQuant lane | GREEN — TQ-aware store fix validated per preset: kit D2H live per slot, integrity 5/5 + det 3/3, spec ≥ target, KV 526k/757k/853k/975k (2.01/2.89/3.25/3.72×), U32k/U64k TTFT −30…−35 % vs fp8; cert6 release gate single-flow GREEN | `results/2026-11-02-p614-{dbg,preflight,l1v1,l1v2,l2,l3,cert6}/` |
| P6.16 draft-sampling posture | GREEN — opt-in `ARC_SPEC_METHOD=probabilistic` serve hook validated on cert6 (5-dtype matrix, t1.0-xhigh agg fp8 +29 % / 3bit +91 %, t0 unchanged every dtype); default stayed greedy at P6.16 | `results/2026-11-03-p616a-probsmoke/`, `…-p616b-probmatrix/`, `…-p616c-spechook/` |
| P6.17 prob-lane certification | GREEN — every record face re-measured natively on the probabilistic lane (t0 agg 291.4–303.1 / solo 83.7–84.9; TTFT 3.77/7.90/17.38/40.32; warm t2 0.99–1.08; MTP-over-no-spec 2.35–2.53×); serve-default FLIP validated (flag probe 5/5, default-path boot bytes IDENTICAL to record legs); no new image | `results/2026-11-03-p617a-probcert/` |
| P6.18 + cert7 | GREEN — corruption split Class B (REP-LOOP attractors, all presets) / Class A (t0 warm-resume flips); roundtest proved the store→decode pair numerically exact → interim config-time refusal of every `turboquant_*` dtype; cert7 gates ALL GREEN on the tag-built image incl. the long-return gate 100 % CLEAN + t1-seed annex (determinism-semantics law) | `results/2026-11-03-p618b-repro/`, `…-p618c-roundtest/`, `…-p618-cert7/` |
| P6.19 + cert8 | GREEN — three root causes fixed wheel-side (`tq_spec_fix.py` default-ON, refusal lifted); all-dtype long-return gate zero garble on all 4 TQ presets + fp8 control; cert8 gates on the tag-built image (fp8 byte-family cert7; TQ all-dtype zero garble; mq annex agg8 284.81 = 1.21× split) | `results/2026-11-04-p619-tqverify/` |
| P6.20 + cert9 | GREEN — L3 off-thread store ADOPTED (U32k −0.74 s / U64k −1.64 s); L6 dispatch-late; lane closure: U64k/U32k device-compute-bound (~89 % of TTFT device-side); cert9 gates on the tag-built image (default posture with ZERO env overrides proves default-ON L3) | `results/2026-11-04-p620-prefill-opt/` |
| P6.21a ar_quant donor | NEGATIVE, closed unadopted — quality green, perf regresses every U-bar (U64k +9.2 %) | `results/2026-11-05-p621-arquant/` |
| P6.21b FA ladder | Analysis of record — stock FA2 at 25 % of the 147 TF/s dual-card fp16 peak; every patch-level route falsified | `results/2026-11-05-p621b-fa-headroom/` |
| P6.21c ESIMD dense-FA v2 | NEGATIVE, closed — numerics green, perf 0.33× stock; ceiling ~0.83× < the 2× bar | `results/2026-11-05-p621c-fa-esimd-v2/` |
| P6.22 TP2 allreduce lane | NEGATIVE, closed — census + trace + ceiling bench: stock oneCCL ring 2.06× faster than the one-shot ceiling; instrument banked (nothing serve-affecting) | `results/2026-11-05-p622-ar-lane/` |
| P6.23 docs-currency audit | GREEN — 12 findings fixed across the six living docs + build guides | `results/2026-11-05-p623-docs-audit/` |
| P6.24 + cert10 | GREEN — r96 rebuild fidelity, r97 e4m3-wire gate **1.65×**, r98 quality parity, r99 release gate (opt-in beats cert9 on every p65 TTFT face), r100 wire-rate closure (v4 = 82 % of the wire floor) | `results/2026-11-05-p624-fp8-ar/` |
| P6.24b patch audit | GREEN — three findings dispositioned (probe-patch sha reconciliation, res-scale invariant documented, frozen-artifact law) | `external_patches/oneccl/README.md` + VERDICT patch-audit section |
| P6.25 overlap-door audit | NEGATIVE, closed MEASURED — cross-stream AR∥GEMM slower than serial at every k on both wires (best 0.969×/0.966×; coten GEMM inflation 1.974×/1.580×); GEMM census closes at ~99 % of the per-card ceiling | `results/2026-11-06-p625-overlap-door/` |
| P6.26 + cert11 | GREEN — quark failure predicted byte-exact, two-override platform fix (7ea0f27b), both MXFP4 checkpoints validated end-to-end; cert11 gate recomputed PASS from banked artifacts after an awk column bug | `results/2026-11-06-p626-mxfp4/` |
| P6.27 MXFP8 AutoRound | GREEN — zero-change validation on cert11 (native kernel, quantized drafter, spec 0.9626/0.9174, TTFT below both mxfp4 lanes, decode 1.53× explained); no cert | `results/2026-11-06-p627-mxfp8/` |
| P6.28 gemma-4 + cert17 | GREEN — G2 freeze closed 3 layers; L1 fp8 battery all green; L2 mxfp8 battery green; prefill root cause closed at the kernel level | `results/2026-11-06-p628-gemma4/` |
| P6.28 FA2 route + cert18 | GREEN — hd512 FA2 34× at production geometry; image validation faces 2.2–16.3× faster; qwen discriminator untouched | `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/` |
| P6.29 + cert19 | GREEN — sliding-window FA2 18–25× prefill / 11–15× decode; gemma faces 4.9/17.2/110.9/0.9 s (qwen-parity at 32k); ngram lane closed negative (upstream assert); qwen discriminator in-band | `results/2026-11-07-p629/` |
| P6.30 + cert20 | GREEN — W4A16 root-cause chain r113→r120d (oneDNN gemmstone gate math); FIX-P arc4 wheel op; e2e superiority 1.47–1.75× on every axis; 4 engineering laws banked | `results/2026-11-08-p630-mxfp4-perf/` |
| P6.31 + cert21 | GREEN — kernel matrix (m816 1.74×/2.68×); qwen 1.37–1.55× / gemma 1.57–1.85× e2e; W4A8 closed-negative (oneDNN refuses the joint dtype); gemma 8-bit capacity law; r125 three-lane gate green | `results/2026-11-08-p631-mxfp8-w8a16/` |
| P6.32/P6.33 + cert22 | GREEN (kernels correct, route falsified, banked OFF) — r126 correctness gates; r127 five-leg e2e falsification (gemma −38/−42 %); attribution chain r127b–f (stock oneDNN 420–680 GB/s under replay); r127g2 cert leg exact-ref | `results/2026-11-08-p633-skinny-m/` |
| P6.34/P6.35 + cert23 | GREEN — `graph_exec_doublebuffer.py` default-ON; r135 five-leg gate + r135b control; E2E-neutral at every posture; posture records k5 +41 %, mx4+k5 +52 % | `results/2026-11-10-p635-telemetry/` |
| P6.35-V matrix | GREEN — 11/11 cells, 0 tracebacks; project-best qwen mxfp4 k5 128.3 tok/s; gemma mxfp4 +43 %; mxfp8 deprecated for decode | `results/2026-11-10-p635-telemetry/vmatrix/` |
| P6.36 decomposition | GREEN (analysis-of-record) — prefill = device drain (80 % of U32k), rank skew ZERO, B discriminators negative, decode D2 ceiling +8–10 % < bar | `results/2026-11-10-p636-legA/` |
| P6.37 close-out | GREEN — r138 13-cell report matrix (61 faces, 0 tracebacks, every 8K point re-verified); 64 GB-host defaults shipped + validated; `ARC_KVDTYPE` boundaries measured; PATCHES-arc.md inventory | `results/2026-11-10-p637-reports/` |
