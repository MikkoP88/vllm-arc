# P6.24 r97 — fp8-wire allreduce GATE: PASS (2026-11-05)

Lane: oneCCL 2022.0.0 same-version rebuild + e4m3 wire patch (Route A),
patch `arc-fp8-allreduce.patch` v4 (sha256 effe9a27fae37ff596a43f25cfb9cc504f8e4ab5889bebcb0a92fdce7e14a07d),
fat lib libccl.so.1.0 sha256 3a120e8b352fbd6be45ab61a7f4e927d80c9bc6618ff646e6db68b0263eca4a7.

## Gate (pre-registered: >=1.3x at the dominant 76,677,120 B; <=6.0 ms)

| size (B)    | stock (ms) | fp8-wire (ms) | speedup | GB/rank/s |
|------------:|-----------:|--------------:|--------:|----------:|
| 150,994,944 | 15.352     | 9.076         | 1.69x   | 16.64     |
| 83,886,080  | 8.535      | 5.149         | 1.66x   | 16.29     |
| **76,677,120** | **7.801** | **4.731**  | **1.65x** | **16.21** |
| 68,157,440  | 6.936      | 4.196         | 1.65x   | 16.24     |
| 15,779,840  | 1.621      | 1.030         | 1.57x   | 15.32     |
| 2,621,440   | 0.327      | 0.353         | 0.93x   | stock path (sub-threshold, jitter band) |

- R97_CONTROL: PASS — fp8-off overlay reproduces the r95/r96 anchor
  (7.801 vs 7.780 ms; randn 0.00022; every size within noise).
- R97_GATE: PASS at 1.65x — inside the pre-registered 1.6-1.8x band.
- Numerics: PASS — rel-rms 0.02285 == the offline bit-exact noise floor
  (fp8_noise_sim.py: 0.022861 at this exact size; bar re-registered 0.035
  with a 5.7x margin to the >=0.2 corruption regime). Both ranks identical.
- Steady-state stage telemetry (dominant, STATS=1): k1_amax 0.51 ms,
  k2_quant 2.01, barrier 0.03-0.11, k3_res 0.02, k4_reduce 1.97, k5_dequant
  0.40, barrier 0.02 — sums to the measured 4.73 ms; each remote-writing
  kernel moves 19.2 MB at ~9.6 GB/s = full wire rate. First message pays a
  one-time ~350 ms SYCL JIT across the five kernels (steady-state medians
  are unaffected).

## Laws banked this lane (see r97_debug_ledger.md for the full capture chain)

1. Sector law: stores to the peer's IPC-mapped memory must be packed to
   full sectors. Byte-granular e4m3 stores ran 0.41 GB/s (sub-sector RMW
   per store); u32-packed stores restore 9.6 GB/s. Convicted by the v3
   stage split: all-local K5 with identical converters ran 124 GB/s while
   K2/K4 with byte remote writes took 45-47 ms each.
2. e4m3 numerics floor: one round trip per output element on randn-scale
   data = rel-rms ~0.023 on the bench metric — bars tighter than that
   fail correct implementations (v2 was correct and "failed" 0.02).
3. K1 atomics: single-address fetch_max is a CAS loop on Intel GPUs;
   per-element = ~100 ms, 262144 threads = ~15 ms, 64 (16384-thread grid
   with register-max) = microseconds.
4. Torch 38M-element fp32 dot/norm inflates the printed cosine ~1.0145x
   (stock leg prints 1.0142 for an exact sum; exact cos is 1.0000028).
   Never gates; rel-rms is the discriminator.

## Phase state

- Phase 0 (r96 fidelity): PASS — same-version rebuild is ABI/perf-identical.
- Phase 1 (r97 gate): PASS (this verdict).
- Phase 2 (r98 quality ladder): **PASS** (below).
- Next: r99 serve A/B + cert10 image. Env trio CCL_SYCL_ALLREDUCE_FP8 /
  _MIN_COUNT / _STATS (defaults 0 / 2097152 / 0 — silent off; the
  certified posture keeps FP8=0 and serves stock unless explicitly
  enabled).

## r98 phase 2 — quality ladder: PASS (2026-11-05)

Same-day A/B on ONE container (cert9 image, same caches): leg A = stock
cert9 bytes, leg B = v4 overlay + exactly one env line
(`CCL_SYCL_ALLREDUCE_FP8=1`). Suite = the t5/t9-class instruments
(flip/integrity+det/run7n t0 + t1.0-xhigh/spec/tps/LONG-RETURN full +
3 churn cycles). All pre-registered bars green:

| bar | leg A (stock) | leg B (FP8=1) | ratio / bar |
|---|---|---|---|
| tps solo1 | 60.89 | 61.21 | 1.005 in [0.90,1.10] |
| tps solo2 | 61.39 | 61.26 | 0.998 |
| tps agg8 | 329.12 | 314.13 | 0.954 (single-wall agg) |
| run7n t0 agg | 286.11 | 285.43 | 0.998 |
| run7n t0 solo | 84.50 | 84.55 | 1.001 |
| run7n t1.0-xhigh agg | 184.80 | 184.42 | 0.998 |
| spec pos1 | 0.9177 | 0.9258 | delta +0.0081 (bar 0.03) |
| integrity/det | 5/5 + 3/3 | 5/5 + 3/3 | PASS |
| LONG-RETURN within-leg | PASS (4 classes single-hash, 3× CLEAN cycles) | PASS | hard bar |
| erraudit | 0/0 | 0/0 | hard bar |

- Cross-leg t0 drift: essay/code/multi hashes differ (the pre-registered
  e4m3 prefill-noise disposition — 0.023 rel-rms flips near-tie argmax;
  coherence gated by zero-garble + finish-stop); **`simple` byte-identical
  across legs** — its 38-token prefill never crosses the 2M-elem
  threshold, so it never touches the wire: the min-count law confirmed
  live in serve.
- In-situ wire proof (STATS probe, one ~15.6k-token prefill): 40
  `CCL fp8` lines, e.g. `count 7459840` (= 1457×5120, mid-size prefill
  chunks) with both ranks agreeing on `res scale 0.00235421`; first call
  pays the one-time JIT (k1 11.5 ms) then steady (k2 0.417 ms).
- KV pool note: 400,739 (A) vs 398,363 (B) tokens (−0.6%) — the rebuilt
  lib's init footprint shifts pool sizing slightly; quality/parity bars
  unaffected (all measured faces above).
- Two analysis-layer script defects found and fixed same-day (measured
  data never affected): the r84-inherited garble_check BRE `\[]` escapes
  never match (grep -F does) — every prior cert gate carried false
  ZERO-GARBLE FAIL lines dispositioned by summary PASS; and the r98
  rehit longprobe labels lacked per-cycle suffixes (three rehit jsons
  collapsed into one) — cross-cycle evidence reconstructed from the
  per-cycle console outs by `r98_reconstruct.py` (hashes/garble/flags
  print verbatim). Both fixes carry into r99. A ghost JSONDecodeError
  (char 0 on a provably-valid json, seen twice in first-run summary
  passes, never reproducible since) was fenced with a raw-bytes debug
  fallback; the hardened run completes deterministically.

Evidence: `r98/` (both legs' full artifact sets + statsprobe + corrected
verdicts), `r98_gate.log` (window console), `r98_quality.sh`,
`r98_reconstruct.py`.

Artifacts: r96_*.json/log (phase 0), r97_v4_*.json/log (phase 1),
fp8_noise_sim.py (converter twin + bar calibration), r97_debug_ledger.md
(v1->v4 capture-first chain), r98/ + r98_gate.log (phase 2),
external_patches/oneccl/ (patch + README).
