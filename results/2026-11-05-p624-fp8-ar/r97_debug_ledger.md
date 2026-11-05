# r97 debug ledger — fp8-wire gate iterations (2026-11-05)

Capture-first ledger for the three gate attempts; every fix below is
preceded by measured evidence (E-law). All legs: cert9 image, r95/r96
conditions byte-identical, torchrun 2-proc, guard randn.

## v1 (patch 1fbe25ff) — silent no-op then wrong+slow
- Staging bug (banked in build_oneccl_arc.sh): frozen container mounts let
  the arc build stage into stage-stock; legs silently raced the STOCK lib.
- After restage: numerics FAIL (cos 0.567), 13x slow (0.74 GB/rank/s).
- Root cause A (encoder): first-pass converter kept the top 8 bits of the
  raw fp16 E5:M field — halves the exponent instead of re-biasing 15->7;
  1.0 encoded as 256. Convicted by stats telemetry: scales provably
  identical across ranks, hand-arithmetic on the amax chain.
- Root cause B (K1): atomic fetch_max PER ELEMENT — 19M serialized
  same-address CAS, ~100 ms.

## v2 (patch d6f5fcd2) — correct numerics, still 0.85 GB/rank/s
- Encoder rewritten with proper RTNE on both e4m3 paths (fp16 intermediate);
  K1 register-max + one atomic/thread, grid capped 262144; stats fprintf.
- leg A control PERFECT: 7.800 ms at dominant == r95 anchor; randn PASS.
- leg B numerics: rel-rms 0.02285 == offline bit-exact noise floor
  (fp8_noise_sim.py: 0.022861 at n=38,338,560; stock cross-check reproduces
  the rig's 0.00022 to three digits). The algorithm was CORRECT; the
  original 0.02 bar sat under the true e4m3 floor -> re-registered 0.035.
- Perf 90.5 ms dominant, LINEAR per element (~2.3 ns/elem). v1->v2 delta
  (~15 ms) = the K1 cap benefit; ~75 ms unexplained by atomics.
- Metric note: rig cos prints ~1.0145x the true cosine (present in the
  stock leg too: 1.0142 for an exact fp16 sum; offline exact cos is
  1.0000028). torch 38M-element fp32 dot/norm artifact; never gates.

## v3 (patch 7683910f) — stage telemetry convicts the WIRE WRITES
- Converters branchless fp32-native (~8 ops; exact rare path < 2^-6 with
  the zero-law: zeros must encode 0 for sparse activations); 630k-value
  exhaustive-grid twin: 0 mismatches. K1 grid 2048. Stage timing added
  under FP8_STATS (declared perturbation).
- Stage split at dominant (per rank, both ranks agree):
  k1_amax 3.2 ms | k2_quant 47.3 ms | barrier 0.7 | k3_res 0.02 |
  k4_reduce 45.2 ms | barrier 0.02 | k5_dequant 0.31 ms | barrier 0.02
- LAW (banked): K5 — same converters, ALL-LOCAL traffic — runs 0.31 ms
  (124 GB/s). K2/K4 differ in exactly one respect: e4m3 SINGLE-BYTE stores
  into the peer's IPC-mapped memory. 19.2 MB at 45-47 ms = 0.41 GB/s.
  Byte-granular stores to uncached peer memory pay a sub-sector
  read-modify-write each; stock fp16 stores are full 32B sectors and
  sustain 9.85 GB/rank/s. Peer-memory stores must be packed to full
  sectors.
- Same ledger explains v2 entirely: 47+45+3.2(+15 ms of K1 CAS in v2) ~
  the 90-105 ms linear floor.

## v4 (patch effe9a27) — packed wire writes: GATE PASS
- K2/K4 pack 4 e4m3 codes per u32 store (16-lane subgroup = 64B, 2 full
  sectors, 4x fewer store instructions); scalar tail (data_count % 4) on
  thread 0; K1 grid 16384 (2048 was latency-bound at 12 GB/s).
- RESULT: 4.731 ms at the dominant (1.65x, 16.2 GB/rank/s), 1.57-1.69x
  across all large sizes, control PASS, numerics PASS (0.02285 == floor).
  Steady stages: k1 0.51 / k2 2.01 / k4 1.97 / k5 0.40 / barriers ~0.1 —
  both remote-writing kernels at ~9.6 GB/s = full wire rate. First message
  pays ~350 ms one-time SYCL JIT. Full tables in VERDICT.md.
