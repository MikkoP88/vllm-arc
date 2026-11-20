# r200h — arc7 MX-native decode: floor decomposition (probe modes 6/7/8) + splits law

Date: 2026-11-20 · Leg: P6.47 Phase 8 (arc7_mx_attn) · Wheel: r200h
(`vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl`, md5
7bcda414, archived `fix/...+arc7mx-r200h.whl`) · Branch:
feat/arc-p647-arc7-mx-native · Rig batch (W14 kept down, R198/R200C/R200H
KEEP_DOWN).

## VERDICT: PASS — frontier moves from V lane to K lane; floor is 5x under v2

Change: probe modes 6/7/8 compiled in (const-K severs the K copy+dequant+
reorder chain; mode 7 = compute floor = const-K + const-V + skip staging;
mode 8 = mode 7 + const-P, skipping softmax) + `r200c_probe.sh` sweeps
0-8 + `p200h_splitbench.py`/`r200h_splitbench.sh` (num_kv_splits sweep).

Gates: synth-v2 both formats **ALL_STAGES_OK, C 0/2048**; faces **42/42
PASS**, bitwise 0/1024 x3 (`r200h_faces_summary.txt`); modes 0-5 within
noise of the r200g matrix (no regression from the probe gates).

### Splits law (`p200h_splits.json`, wheel 15b98a29)

All prior p198/p199 telemetry ran through `native_call(..., splits=0)` ->
interface maps 0 -> -1 -> stock FMHA TileScheduler AUTO-splits: the 4.76 ms
full / 3.16 ms floor at 32K were already split-parallelized. Explicit
uniform splits (32K mxfp8): 1 -> 74.26 ms (serial per-WG walk = ~36 us per
k-tile), 16 -> 5.66 ms (worse than auto 4.76 — uniform wastes the short
seq). Fit t(s)=A/s+B: **B ~= 1.7 ms constant split overhead alone is 2.2x
v2's 0.77 ms total — more splits cannot close the gap**. Canary
max|splits_n - splits_1| = 4.9e-4 (bf16 rounding; real gate = r198 forced8
face).

### Floor decomposition (32K device ms, auto splits, `p200c_probe_*.json`)

| mode | meaning | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|---|
| 0 full | everything | 4.732 | 3.674 | 4.422 |
| 1 | skip V staging | 4.353 | 3.446 | 4.202 |
| 2 | const-V (no gather) | 3.437 | 2.591 | 3.345 |
| 3 | floor (no V lane) | 3.140 | 2.409 | 3.237 |
| 4 | no barrier | 4.724 | 3.677 | 4.419 |
| 5 | no scales | 4.698 | 3.661 | 4.400 |
| 6 | const-K | 1.843 | 1.525 | 1.520 |
| 7 | compute floor | 0.170 | 0.177 | 0.184 |
| 8 | gemm/epilogue floor | 0.363* | 0.142 | 0.148 |

*mxfp8 mode-8 non-monotonic vs mode-7 (0.363 > 0.170): const-P=1 quirk,
floors are lower bounds regardless (const-fill ablation law).

Attribution (mxfp8): **K load chain m0-m6 = 2.889 ms = 61% of the walk**;
V lane m6-m7 = 1.673 (cross-check (m0-m2)+(m0-m1) = 1.674 exactly);
softmax+split-P m7-m8 = 0.19; gemm+loop+epilogue floor = 0.14-0.18 —
**5x below v2's 0.77 ms total**. mxfp4/mxk8v4 same shape (K 58%/66%,
floors 0.142/0.148). e4m3-specific dequant ALU ~= 0.74 (mxfp8 K chain minus
mxfp4's shared loads+reorder ~2.15).

## Next bottleneck (always-loop)
The K lane: no prefetch exists (the mainloop prefetches V only —
`prefetch(prefetch_vb, ...)`), so per-tile K LSC load latency (~us-scale,
JM-serialization-law family) is exposed on the critical path. Ladder:
(a) r201 K-lane cross-tile prefetch/double-buffer (overlap K loads with
softmax+PV+loop body), (b) bitcast/LUT e4m3 decode (~0.74 slice),
(c) reduce the 1.7 ms split overhead (reduce kernel). Stop condition:
superiority vs v2 (0.77/0.66/1.58), then r200-e2e fastcycle ->
full battery -> cert26.
