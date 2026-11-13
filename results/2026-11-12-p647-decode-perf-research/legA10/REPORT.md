# A10 telemetry attribution -  (n_attn=16)

| rung | conc | q8 itl p50/p90 | m8 itl p50/p90 | m4 itl p50/p90 | q8 ttft | m8 ttft | m4 ttft | q8 tps | m8 tps | m4 tps |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1 | 39.1/39.3 | 42.2/42.6 | 41.5/42.2 | 0.29 | 0.29 | 0.39 | 76.3 | 60.9 | 74.2 |
| 1024 | 8 | 55.9/56.7 | 71.0/76.9 | 71.7/76.0 | 2.01 | 2.07 | 1.48 | 46.6 | 39.3 | 44.8 |
| 8192 | 1 | 40.4/40.4 | 64.1/64.3 | 58.2/58.3 | 3.03 | 2.98 | 2.97 | 64.9 | 46.9 | 50.4 |
| 8192 | 8 | - | - | - | - | - | - | - | - | - |
| 32768 | 1 | 42.5/42.7 | 137.2/137.6 | 112.0/112.2 | 15.59 | 17.42 | 16.99 | 26.6 | 14.9 | 17.2 |
| 32768 | 8 | 77.1/213.3 | 830.4/1995.4 | 3714.0/3826.9 | 71.47 | 67.24 | 58.45 | 4.4 | 2.5 | 5.8 |

## Attribution (conc=1: attn-path delta per step, per layer)

| rung | m8 dITL ms | m8 per-layer ms | m4 dITL ms | m4 per-layer ms | m8 dTTFT s | m4 dTTFT s |
|---|---|---|---|---|---|---|
| 1024 | 3.1 | 0.19 | 2.5 | 0.15 | -0.00 | 0.10 |
| 8192 | 23.7 | 1.48 | 17.9 | 1.12 | -0.05 | -0.07 |
| 32768 | 94.6 | 5.91 | 69.5 | 4.34 | 1.83 | 1.40 |

Reading: per-layer ms compares directly against the A9 kernel
timings at the same shape (D=256, HK=4, N_REP=6). A per-layer
delta far above the kernel ms means the cost is NOT in the
decode kernel - look to launches, engine casts, store path, or
the spec/draft loop before touching the kernel again.
