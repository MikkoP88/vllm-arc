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
- Next: r98 quality ladder (task-level parity vs cert9), r99 serve A/B +
  cert10 image. Env trio CCL_SYCL_ALLREDUCE_FP8 / _MIN_COUNT / _STATS
  (defaults 0 / 2097152 / 0 — silent off; the certified posture keeps
  FP8=0 and serves stock unless explicitly enabled).

Artifacts: r96_*.json/log (phase 0), r97_v4_*.json/log (this gate),
fp8_noise_sim.py (converter twin + bar calibration), r97_debug_ledger.md
(v1->v4 capture-first chain), external_patches/oneccl/ (patch + README).
