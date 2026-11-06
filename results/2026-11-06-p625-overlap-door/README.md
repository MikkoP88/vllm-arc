# P6.25 — the overlap door: measured CLOSED (r101, 2026-11-06)

Owner directive (verbatim, 2026-11-06): *"Identify and eliminate all remaining
Prefill bottlenecks, performance bottlenecks, optimize critical paths, minimize
runtime overhead, and eliminate remaining latency gaps, this is code
optimization work no existing knob testing or hardware stuff. Do realy deep
analyze and research, and update master plan and start implement fix
start-to-end, use in line best practise project patches, all patches and
changes has to be in line best practices and upstream, use comprehesive
telemetry capture points end-to-end to locate root issue, sources and etc...,
create new certificated image on end and run full performance metric and
update all documents."*

Plan §P6.25 mapped the residue after P6.20–P6.24 and found exactly ONE door no
leg had measured-closed: **cross-stream AR(comm) ∥ GEMM(compute) co-tenancy**
(async-TP class). r83 closed same-stream overlap (queue never idle); r100
closed fabric rd+wr concurrency (1.49 GB/s collapse) and AR-internal
pipelining (~1 % class); neither answered whether a side stream can hide the
AR wire behind GEMM EUs.

## Pre-registered gate (plan §P6.25, before any window)

> best-k mock overlap ≥ 1.08× serial → r102 in-situ prototype;
> else the overlap door closes measured.

## Vehicle

- `tools/ar/ar_overlap_bench.py` — five modes (`async`/`census`/`ar`/`coten`/
  `mock`) on the serve engine's own stack: torchrun `--standalone
  --nproc-per-node 2`, backend `xccl`, cert10 image, the certified oneCCL env
  block, real qwen3.8-27b TP2 per-rank shapes at M=7488 (the dominant prefill
  chunk): qkv 5120→4096, o 3072→5120 (q-slice input), gate_up 5120→17408,
  GLU → 8704, down 8704→5120; both row-parallel ARs at [7488, 5120] fp16.
- `tools/ar/r101_overlap_bench.sh` (window 1: stock leg) and
  `tools/ar/r101b_overlap_bench.sh` (window 2: stock `mock` + full fp8 leg
  with `CCL_SYCL_ALLREDUCE_FP8=1`). W14 stopped for each window, restored by
  trap, generation-probed after (rig law).

Bench-defect note (both fixed BEFORE any gate reading; the gate was read only
on correct-shape data): window-1 `mock` died on two shape bugs — o_proj
consumes the qkv q-slice (`[:, :3072]`), and the gate_up→down path needs the
GLU (`silu(gate)*up` → 8704 cols) that the first draft skipped.

## Results (rank 0, M=7488, medians)

| mode | stock | fp8 wire |
|---|---|---|
| async enqueue | 0.182 ms — async-enqueue | 0.263 ms — async-enqueue |
| AR solo [7488,5120] | **7.809 ms** (== banked r94/r100 ceiling 7.78–7.80 ✓) | **4.597 ms** (1.70× ✓ matches cert10's 1.65× in-situ) |
| census layer GEMMs | 17.173 ms total = 148.6 TF/s | — |
| coten gemm_inflation | **1.974×** (9.013→17.792 ms) | **1.580×** (9.204→14.539 ms) |
| coten ar_inflation | 1.092× | 1.156× |
| mock serial (L=8) | 273.11 ms | 222.78 ms |
| mock k=2 / k=3 / k=4 | 0.969 / 0.966 / 0.951 | 0.960 / 0.966 / 0.945 |
| **best-k speedup** | **0.969 (k=2) — FAIL** | **0.966 (k=3) — FAIL** |

Census per-GEMM: qkv 2.093 ms / 150.1 TF/s, o 1.618 / 145.5, gate_up 9.002 /
148.3, down 4.46 / 149.6. The JSON's `pct_peak` fields (≈204 %) were computed
against the window-1 constant 73.5 TF/s/rank — an assumption this census
itself falsified: the practical dense fp16 ceiling on these shapes is ~150
TF/s rank-local (the earlier "147 TF/s dual-card" phrasing was per-card).
Read against the corrected reference the GEMMs sit at **~99 % of the measured
dense ceiling — the GEMM lane closes as already-optimal**.

## Interpretation

1. **The API permits overlap** (all_reduce is async-enqueue on both wires) —
   the door was worth measuring.
2. **The device does not.** Concurrent AR nearly doubles GEMM latency
   (1.974× stock; 1.580× fp8): the AR kernels and the GEMM kernels contend
   for the same execution resources — the device time-slices them rather
   than running them on independent EU-vs-wire lanes. Hiding 15.6 ms of AR
   per layer costs as much GEMM slowdown as it saves, minus schedule and
   event overhead — which is exactly what the mock shows: every k on both
   wires is ≤ 0.97× serial. This is schedule-independent (k-sweep flat) and
   wire-independent (both stock ring and fp8 quant-path fail), so it closes
   the async-TP class for this hardware, not just the two-phase prototype
   schedule.
3. **fp8 wire composes with nothing further**: its lighter kernel mix
   contends less (1.58× vs 1.97×) but still far too much to win.
4. `overlap_bitwise: false` (stock) is a **mock artifact, not an AR-numerics
   finding**: the k-split changes GEMM M-dims → different tile/split-K
   accumulation order in fp16 → ULP diffs compounding over 8 layers. The
   actual safety property — row-parallel AR is per-token independent, so a
   row split changes no element's reduction order — remains true. (Moot:
   nothing lands.)

## Verdict

**The pre-registered gate FAILS on both legs → the overlap door closes
measured; no r102.** With this leg the P6.25 residue map is exhausted — every
prefill critical-path door is now measured-closed:

| door | closed by |
|---|---|
| GEMM efficiency | r101 census: ~99 % of measured ~150 TF/s rank-local dense ceiling |
| stock AR rate | P6.24/r94/r100: 7.809 ms == fabric ceiling (9.85 GB/s writes flat in width); re-anchored today |
| AR wire bytes | P6.24 cert10 fp8 opt-in: 4.597 ms (1.70×), re-anchored today |
| same-stream overlap | r83: queue never idle (also closes host-gap runtime overhead) |
| fabric rd+wr concurrency | r100: 1.49 GB/s collapse |
| AR-internal pipelining | r100 v5: ~1 % class, below any gate |
| cross-stream AR∥GEMM co-tenancy | **r101 (this leg): mock ≤0.97× at every k, both wires** |
| FA/attention kernel | P6.21b anatomy lane (separate, already falsified) |

**No cert11**: nothing landed — the engine is byte-unchanged, so cert10
(`rel/v0.30.0-b70-cert10`) remains the certified image and a re-release would
certify no change (no-unevidenced-customization law). The best-performance
posture remains the certified cert10 defaults with the P6.24 FP8 opt-in
(`ARC_FP8=1` / `export CCL_SYCL_ALLREDUCE_FP8=1`; docs/CONFIGURATION-arc.md
"Best-performance postures"). The full performance matrix is unchanged from
the P6.24 certification (r96–r100); this window re-validated its two key
anchors in isolation (7.809 ms stock, 4.597 ms fp8). Remaining residue is
bounded ≤1–2 % class (AR-internal pipelining; fp8 quant-pass fusion into
GEMM epilogues — requires a novel cross-library contract on the frozen
oneCCL overlay and, per the co-tenancy result above, only relocates the same
EU work): below every gate this project has ever set.

## Files

- `r101_stock_{async,census,ar,coten,mock}.json`, `r101_fp8_{async,ar,coten,mock}.json` — raw rank-0 results
- `r101_stock_mock.log`, `r101_fp8_leg.log`, `r101b_window2.log` — window-2 leg logs (window-1 leg log was overwritten by window 2's stock-mock run; its four mode results live in the JSONs)
- Bench + runners, tracked: `tools/ar/ar_overlap_bench.py`, `tools/ar/r101_overlap_bench.sh`, `tools/ar/r101b_overlap_bench.sh`
- Image: `vllm-arc:v0.30.0-b70-cert10` (engine byte-identical to cert9 + two-file oneCCL overlay)
