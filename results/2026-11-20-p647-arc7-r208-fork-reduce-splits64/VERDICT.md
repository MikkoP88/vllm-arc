# r208 — fork-owned split reduce: splits 17-64 legal, correct, and fast

**VERDICT: PASS — MxReduceSplitK (fork-owned stock-reduce copy with the
thread-strided loader and the plan-aware combine guard) restores the
canary to one-bf16-ULP at EVERY split count 1-64 while holding the
r207-walk times. Chain on the same wheel: ALL faces PASS (bitwise
0/1024 per format, splits, robust, envelope) and mode0 at 32K drops to
mxfp8 0.427 / mxfp4 0.327 / mxk8v4 0.333 ms — 1.80x / 2.02x / 4.73x of
the certified v2 totals. The MXFP4/MXFP8 lanes now BEAT FP8-class
decode performance at 32K by ~2x (goal met and exceeded).**

- Wheel: arc7mx-r207r208 md5 `611a8dba` (archived
  `fixsrc/+arc7mx-r207r208.whl`); sweep driver
  `tools/ar/p147_mxkv/r207_splitbench.sh`; bench
  `tools/ar/p147_mxkv/p200h_splitbench.py` (NaN canary, WGTARGET=64).
- Gates: `invalid_launch_count=0`; canary <= 2.44e-4 at every cell
  (incl. the 8K table in `p208_splits_clean.json`); WGTARGET default 2
  keeps the vendor-parity plan for the engine path (byte-equal plan
  arithmetic, `ARC_MXKV_WGTARGET` unset).

## 32K device ms (plan tracks the cap; canary clean throughout)

| s | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| 1 | 19.879 | 14.074 | 13.186 |
| 8 | 2.705 | 2.033 | 1.882 |
| 16 | 1.550 | 1.197 | 1.104 |
| 24 | 1.150 | 0.923 | 0.850 |
| 32 | 0.945 | 0.800 | 0.741 |
| 48 | 0.790 | 0.735 | 0.676 |
| 64 | 0.719 | 0.667 | 0.628 |

16 -> 64 scaling is ~2.2x (not the ideal 4x): the reduce/tail slice
grows with splits and the walk asymptotes. 8K bottoms near s=32-48
(~0.38-0.42 ms).

## The two stock-reduce bugs fixed (mx_reduce_split.hpp header has the
full derivation)

1. Loader `thr_id < effective_splits` — one thread per split; past 16
   threads (page-16 WG = 16 threads) SLM stats were unwritten and the
   combine folded garbage. This is WHY the stock author capped splits
   at threads-per-WG.
2. Combine-loop break `i*num_blocks_per_split >= windowed_k_blocks`
   assumes uniform-ceil partitioning; the host plan's divmod partition
   puts tiles in tail splits the guard calls empty (drops them). The
   break now applies only on the legacy no-plan path.

The strided loader degenerates to the stock one-thread-per-split
mapping at s <= 16 — s<=16 cells reproduce the r207-wheel values
(1.550/1.197/1.104 vs 1.533/1.147/1.100, layout-dice-scale wiggle),
so bitwise faces at the default posture are not expected to move.

## Open (improvement loop continues)

- AUTO posture: policy cap is now 64 but the C++ auto heuristic still
  lands at 40 splits at 32K (reduce cap = 4*num_wg_slots/Hq = 40).
  mode0 probe chain on this wheel is the next leg; if mode0 sits at
  the s=40 interpolation (~0.85 mxfp8), lift the reduce-cap knee.
- Engine path (native_decode.py) passes no host_kv_lens -> legacy
  uniform splits; the plan-knob (ARC_MXKV_WGTARGET) only affects
  host_kv_lens callers (bench/probes).
- Walk base at s=64 (~0.63-0.72 ms) is still ~2.5x above the 0.25 ms
  pure-overhead signature; the reduce/tail slice is the next lever
  (probe modes m3/m4 decompose it).

## Chain on wheel 611a8dba (r208_chain.sh, 2026-11-20 20:51-20:57)

Faces (r198 offline harness, all PASS — full log in r208_chain.log):

- bitwise (seq_len==1 dequant oracle): 0/1024 mismatches x3 formats.
- native-exact / native-v2 / robust (adversarial, signed, len1/15/16/17):
  d_rounded <= 1.53e-5, cos >= 0.9999985 everywhere.
- splits face (forced8 + default plan): one-bf16-ULP class, unchanged
  from r204 — the auto-posture change (16 -> 40 splits) did NOT move
  the numerics envelope.
- telemetry (always-on native-vs-v2 datapoint, THIS wheel's headline):
  32K: mxfp8 1.40x (0.548 vs 0.769), mxfp4 1.70x (0.389 vs 0.662),
  mxk8v4 3.79x (0.414 vs 1.567); 8K: 0.94x / 0.96x / 2.12x;
  1K: 0.63x / 0.52x / 0.54x (short ctx still v2-favored).

Probes (p200c_probe_0..8.json, 32K device ms, mode 0 = full):

| fmt | r204 mode0 | r208 mode0 | r208 m6 const-K | r208 m8 gemm floor |
|---|---|---|---|---|
| mxfp8 | 1.278 | **0.427** | 0.258 | 0.198 |
| mxfp4 | 1.037 | **0.327** | 0.234 | 0.118 |
| mxk8v4 | 0.971 | **0.333** | 0.246 | 0.090 |

mode0 sits ~1.7x above the const-K anchor and ~2.2x above the gemm
floor — in-walk headroom remains but the big structural gap is closed.
synth2 x3 formats: A/C mismatches 0, B ok (unchanged law posture).

## Posture notes

- mode0 = auto (no host_kv_lens): C++ heuristic at 32K now yields 40
  splits (policy cap 64, reduce cap 4*num_wg_slots/Hq = 40 binds).
- 1K ctx remains the relative weak spot (0.52-0.63x of v2): short ctx
  requests few splits and pays the same fixed overhead — next loop
  candidate (auto request floor for tiny ctx + launch-path trim).
