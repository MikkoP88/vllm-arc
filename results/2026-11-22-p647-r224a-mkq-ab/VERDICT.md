# r224 — mxk8v4 native refusal root-caised to the wheel kernel (NATIVE_WRONG)

Phase P6.47, iteration r224 (2026-11-22/23). Fork content base:
cert26 (`rel/v0.30.0-b70-cert26`, content sha cac63931). Branch
`feat/arc-p647-r224-mkq-strides`.

## Question

The cert26 battery refused the native mxk8v4 lane at the long-return
gate (deterministic REP-LOOP, all four classes, T1_SEED PASS) while the
r220c offline value gate had passed it. Where does the native lane
diverge from the certified Triton v2 lane — engine-side pool geometry,
or inside the kernel?

## Method (one instrument, three analyses, all on the engine's own tensors)

- **r224a** (`ARC_MXKV_AB` in `mx_kv/engine.py`, legs
  `r224a_mkq_ab.sh`): at the native dispatch, rerun the certified
  Triton `mx_decode` on the IDENTICAL `(q_rows, kv_cache, bt_rows,
  sl_rows)` and bank divergence + payload (q/bt/sl, pool window
  indexed by row 0's real blocks, true shape/strides/storage_offset,
  both outputs). Eager re-serve from a byte-faithful /proc
  argv+environ reconstruction, graph-off (pageable-H2D capture law),
  `ARC_MXKV_AB_MINSL=1000` so the budget lands on real essay-class
  decode steps (attempt 1's budget was consumed by the eager dummy
  executions).
- **r224b** (`r224b_ab_analyze.py`): pool-layout attribution,
  divergence stats, per-row/per-head localization, context buckets.
- **r224c** (`r224c_bisect.py`, GPU replay): reconstruct the exact
  pool tensor from a payload (window bytes at `kv_off + block *
  stride0`), replay BOTH kernels, then bisect factors (offset0,
  bt-width) against the Triton replay oracle.
- **r224d** (`r224d_ref.py`, CPU): float64 ground truth from the
  payload's own pool bytes via `mx_quant.decode_kv` (the byte-layout
  ground truth), GQA G=6, D=256 — arbitrate which output is correct.

## Findings

1. **In-vivo divergence is real, deterministic, format-specific**
   (r224a/r224b): 120/120 essay-class captures (sl 5463-8199, both TP
   workers) divergent, dmax up to 2.27, no NaNs, `exact` fraction
   0.05-0.11. Present from the first dummy batch already (SHORT).
2. **Pool geometry exonerated** (r224c): replay of both kernels on the
   reconstruction is BIT-EXACT vs the banked outputs (dmax=0.000e+00,
   both kernels, all payloads); `offset0` variant changes nothing —
   the hybrid pool views' per-layer 435.2 MB storage offsets are
   handled correctly. Pool is mx_layout-conformant: shape
   `(8500, 64, 2, 400)`, dense strides `(51200, 800, 400, 1)`, dim3 =
   `slot_bytes(256, mxk8v4)` = `[K e4m3 x256][Ks e8m0 x8][V e2m1
   x64B][Vs x8]` (264+136), dim2 = Hk=2.
3. **The native kernel is wrong; the Triton lane is right** (r224d):
   vs float64 ground truth, `out_triton` errs 4.8e-4..3.6e-3 while
   `out_native` errs 7.7e-2..5.5e-1 — ratio 156-245x on three payloads
   across both workers and depths.
4. **Error signature = magnitude-dependent softmax/exp-path defect**:
   per-head error correlates with per-head `max|q.K*scale|`
   (corr 0.19/0.57/0.78 across the three payloads) and grows with
   layer depth (layer 0: 7.7e-2 ~ the bf16 noise floor 5.9e-3..5.1e-2;
   layer 29: 5.5e-1). Same defect class as the banked
   fmha-decode-softmax law (bf16-P + fast exp2), not a dequant/layout
   defect (those do not scale with logit magnitude). This is why the
   r220c offline gate passed: its synthetic random data never
   produced real-activation score magnitudes.

## Consequence

The refusal is NOT an engine/store/layout bug — the fix belongs in the
vxk kernel source (`mx_paged_decode` mxk8v4 path: softmax/exp2
precision relative to the certified mxfp8/mxfp4 paths). The r224c
replay harness is the correct offline gate for that fix: replay against
these banked payloads and require native-vs-oracle at the Triton lane's
error level before any e2e re-entry. mxk8v4 native stays REFUSED until
then; the certified mxk8v4 posture remains the Triton lanes.

## Evidence

- `leg_attempt1.log`, `leg_attempt3.log` — arc windows (boot, /proc
  capture, instrument injection md5-verified, re-serve, traffic).
- `r224b_report.txt` — analyzer: 128 captures, bucket rollup,
  `R224B_VERDICT: DIVERGENT captures=128 divergent=107` (attempt 1:
  dummy-class SHORT) and `DIVERGENT captures=120` LONG (attempt 3).
- `r224c_run3.log` — bit-exact replays + offset0 no-op + bf16 floors.
- `r224d_out.txt` — `R224D_VERDICT: NATIVE_WRONG` x3 with per-head
  profiles and score correlations.
- `gen_essay.json` — the REP-LOOP refusal class reproduced in vivo
  (verbatim prompt repetition) under the diagnostic boot.
- `serve.log` — full diagnostic serve log (host addresses redacted,
  see `REDACTION-serve-log.md`).
- `SHA256SUMS.payloads` — hashes of the 128-payload bank (~313 MB,
  stays on the rig at `/root/build/p647/r224a/` per the 100 MB
  artifact law; selected payloads re-derivable from hashes).

## Tools shipped this iteration

`tools/ar/r224a_mkq_ab.sh`, `tools/ar/r224b_ab_analyze.py`,
`tools/ar/r224c_bisect.py`, `tools/ar/r224d_ref.py`, and the
`ARC_MXKV_AB`/`ARC_MXKV_AB_N`/`ARC_MXKV_AB_MINSL` instrument in
`vllm_gaudi/xpu/patches/mx_kv/engine.py` (eager-only, self-guarding
under capture, never kills serving).
