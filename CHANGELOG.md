# Changelog — vllm-arc

Every certified release and closed phase, newest first. The living docs
(README, `docs/*`) carry **current state only**; superseded doc content
is archived per-doc under `docs/history/` (main-currency law —
`tools/check_currency.sh`). Facts in each entry are
the durable summary of the banked evidence; full narratives live in the
cited `results/` legs and the close-out PRs. Tags are annotated and point
at the certified `main` commit; `IMAGE_MANIFEST` inside each image carries
the fork SHA it was built from.

Phase labels P0–P6.38 refer to the plan's execution phases; every phase's
raw evidence is banked under `results/YYYY-MM-DD-<phase>-*/`.

## rel/v0.30.0-b70-cert23 — P6.34/P6.35: graph-exec double-buffer (2026-11-10)

- cert22 engine + one patch module, `vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`:
  every XPU graph captured ×2 into the manager's shared pool, replays
  alternate execs — removes the torch-2.13 same-exec `XPUGraph.replay()`
  host block (p135i microbench: 13.96 ms same-graph vs 8–12 µs
  alternating). Replay max 24 → 0.37 ms; engine serial work 40.8 →
  12.7 ms/step; **E2E-neutral at every measured posture** (fp8-k3 94.5 vs
  94.3 tok/s on/off; mx4k5 121.0 vs 123.9) — a costless driver-level
  hygiene fix: +29 s boot, zero extra memory. Knob
  `VLLM_ARC_GRAPH_EXEC_DB` (default 1; `=0` kill-switch proven).
- The P6.35 decode gains are **posture** (serve-line choices on the same
  engine): spec k5 +41 % (115.2 tok/s); mxfp4-W4A16 + k5 = 123.9 tok/s
  (+52 %) with TTFT 3.98 s < the 4.55 s baseline — the recommended decode
  posture. mxfp8-W8A16 is a decode dud.
- P6.35-V full validation matrix (r136/r136b, 11 cells, both models, all
  online quants × spec depths, 0 tracebacks): qwen mx4-W4A16 k5 re-read
  **128.3 tok/s = +36 % vs the clean fp8-k3 base 94.5 (project best)**;
  gemma mx4-W4A16 **37.1 tok/s (+43 % vs mxfp8)**; gemma online fp8
  boots + serves (26.6, new capability); mxfp8 deprecated for decode;
  best TTFT fp8-k5 3.48 s. Rate accounting corrected (spec SSE streams
  multi-token chunks — numerator = max_tokens).
- Evidence: `results/2026-11-10-p635-telemetry/` (legs A–K + sweep +
  `cert23/` gate + `vmatrix/`).

## rel/v0.30.0-b70-cert22 — P6.33: skinny-M kernel family, banked OFF (2026-11-09)

- cert21 engine + the arc6 `vllm_xpu_kernels 0.1.14.1+arc6` wheel
  (three GEMV-regime M≤8 weight-only GEMM ops —
  `_xpu_C.skinny_fp8_gemm_w8a16` / `skinny_fp8_gemm_w8a16_mxfp8` /
  `skinny_fp4_gemm_w4a16`: TN weight streaming, register dequant, fp32
  accumulate, deterministic fixed-order split-K; plus W4A8 dispatch arms
  and the v0.1.17 Option-B backports #611/#645/#650/#600; sha256
  `484e06c1…`, ~352 MB — rig-only per the 100 MB artifact law) +
  `vllm_gaudi/xpu/patches/skinny_m.py` route behind `ARC_SKINNY_M`.
- Kernels correct (r126 GATE-1/2/3) and winning the eager kbench regime,
  but the r127 five-leg e2e battery **falsified the reroute**: under
  FULL_DECODE_ONLY XPU-graph replay stock oneDNN runs 420–680 GB/s
  device-side (r127f 50-call amortized probe; the motivating ~197 GB/s
  was an eager-launch host-rate artifact; the size-1 decode graph pads
  M→2), so gemma solo regressed −38 %/−42 % while every over-cap path sat
  on ref. r127c `ARC_SKINNY_M=off` reproduced the r124b ref exactly.
- **Route ships default OFF** (unset/off = stock routes = the certified
  default); serve behavior byte-identical to cert21 (r127g2: gemma
  27.7/192.9 exact, zero route markers). TC (joint-matrix) redesign is
  the open future lane behind the r127f gate.
- Content fork SHA `e2457389`; evidence `results/2026-11-08-p633-skinny-m/`.

## rel/v0.30.0-b70-cert21 — P6.31: online-MXFP8 W8A16 route (2026-11-08)

- cert20 engine + the arc5 `vllm_xpu_kernels 0.1.14.1+arc5` wheel
  (`_xpu_C.fp8_gemm_w8a16_mxfp8`: bf16 activations × f8 weights,
  group-32 e8m0 weight scales, zero per-call act quant; carries the
  oneDNN-refused `fp4_gemm_w4a8` probe op; sha256 `219268cf…`) +
  `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` reroute behind `ARC_MXFP8_MODE`
  (unset/`w8a16` = route, default; `w8a8` = stock control).
- Same-image A/B (r124/r124b): **qwen 1.37–1.55× over stock W8A8 on
  every face** (U8k 3.61 s / U32k 17.07 s, 53.8–55.6 solo / 265.9 agg8
  tok/s, spec pos1 0.9804); **gemma 1.57–1.85×** (3.87 / 20.94 s,
  27.7 / 192.4) at `ARC_MAXLEN=40960` (the 8-bit-weights capacity law:
  ~29 GiB TP2 weights leave 5.31 GiB KV).
- Adjudication (pre-registered): checkpoint-fp8 stays the perf default;
  mxfp4-w4a16 stays the superior online lane; W4A8 (e4m3×f4) banked
  closed-negative (oneDNN refuses the joint dtype at primitive-create).
- r125 three-lane gate green (fp8 discriminator + w8a16 + w4a16 carry).
  Content fork SHA `c8fdf32a`; evidence
  `results/2026-11-08-p631-mxfp8-w8a16/`.

## rel/v0.30.0-b70-cert20 — P6.30: online-MXFP4 W4A16 route (2026-11-08)

- cert19 engine + the arc4 `vllm_xpu_kernels 0.1.14.1+arc4` wheel
  (`_xpu_C.fp4_gemm_w4a16`: bf16 activations × f4 weights, group-32
  e8m0 weight scales → oneDNN `jit:gemm` DPAS via weight decomposition;
  sha256 `c675f1a3…`) + `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` reroute
  behind `ARC_MXFP4_MODE`.
- Root-caused the stock W4A4 penalty (r113→r120d): per-call activation
  quant ≈ GEMM cost at decode Ms, and the (f4,f4) joint dtype can never
  pass oneDNN gemmstone's JIT gate → `matmul:ref`, structurally. W4A16
  drops both (verbose-proven `jit:gemm`, numerics rel 1.66e-3).
- Same-image A/B (r120c, qwen): **U8k 3.66 s / U32k 17.12 s, 62.5–65.2
  solo / 301–311 agg8 tok/s — superior to online mxfp8 on every axis
  1.47–1.75×**, 1.74–1.87× TTFT over the W4A4 control, spec pos1 1.0000;
  gemma (r120g) 1.90–2.21× over W4A4. The v3.13 online-fp8 question
  closed mechanistically (group-scale handling inside `jit:gemm`); plain
  qwen fp8-online is an upstream interface gap (r120h traceback banked).
- Engineering laws banked: `_xpu_C` needs explicit submodule import
  (package import does not dlopen); ≤24 icpx jobs when a W14 window is
  live; pip validates wheel filenames; gemma legs need `ARC_LMC=0` +
  `ARC_RPC_TIMEOUT=1200` + dir `/models/gemma4-31b` (no dash).
- r121 gate green. Content fork SHA `a918bee9`; evidence
  `results/2026-11-08-p630-mxfp4-perf/`.

## rel/v0.30.0-b70-cert19 — P6.29: sliding-window FA2 route (2026-11-07)

- cert18 engine + `external_patches/vllm/patch_arc_xpu_fa2_slide_route.py`
  (`tools/cert19.Dockerfile`): sliding-window head-256 calls (gemma-4's
  50/60 local layers) dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel
  windowed in the same triton unified-attention wrapper. `ARC_ATTN_FA2=0`
  disables both routes (this and cert18's).
- Microbench (r109a/r109a3): triton 0.147–0.158 s/call, k-INDEPENDENT vs
  FA2 windowed 0.006–0.009 s = **18–25× prefill, 11–15× decode**. The
  r109a4–a6 "ref misses" were a per-chunk-softmax bug in the TEST refs
  (both kernels correct).
- Image validation (r111): gemma faces 4.9 / 17.2 (qwen-parity, 1.016×
  of the 16.93 record) / 110.9 / 0.9 s; decode 28.4 solo / 195 agg8
  tok/s (+45 % / 2.64× over cert18); both markers live per rank; qwen
  discriminator in-band (3.83/17.58 s). Spec parity for gemma-4
  dispositioned closed: no MTP head in the checkpoint; ngram
  structurally refused by the upstream v1 spec path on pure-attention
  hybrids (r110/r111 banked) — decode posture stays `ARC_SPEC=0`.
- Content fork SHA `904a3de8`; evidence `results/2026-11-07-p629/`.

## rel/v0.30.0-b70-cert18 — P6.28: FA2 hd512 route (2026-11-07)

- cert17 engine + `external_patches/vllm/patch_arc_xpu_fa2_route.py` v3
  (`tools/cert18.Dockerfile`): full-attention head-512 calls (gemma-4's
  10 global layers) dispatch to the ESVD FA2 kernel in the triton
  unified-attention wrapper. `ARC_ATTN_FA2=0` escape hatch.
- r108ac/r108ad: 0.3197 s vs 10.81 s per k32k fp8-KV call = **34×**
  (k98k ~30×, decode 1.5 ms/call); the fp16-KV control refuted the
  earlier L2-bandwidth-wall reading — the triton kernel is slow per
  K-tile iteration; the engine shipped the fast kernel all along.
- Image validation (r108ae): gemma faces 2.2–16.3× faster (face32k
  283.4→41.5 s, face120k 2950.3→181.5 s), decode +6 %, qwen
  discriminator marker-0 in-band (3.77/17.49 s) — incumbent lane
  untouched. The v2 boot crash (warm-gate full-storage descale form) was
  root-caused and fixed in v3.
- Content fork SHA `3136adfe`; evidence
  `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/`.

## rel/v0.30.0-b70-cert17 — P6.28: gemma-4 validation chain (2026-11-07)

- The cert11 engine + three build-kit fixes that made gemma-4-31B
  servable: cert12 `a964e047` (B70 prefill launch-config table in the
  triton unified-attention launcher + engine warmup gate), cert16
  `5d895fef` (warm-gate v4 — the triton cache-key int-softcap fix that
  closed the G2 long-prefill freeze layer 1), cert17 `e5e725b4` (lmcint
  kit hardening — KV-format detector BS floor 16 + the fail-loud
  `[ARC-P628-LMC-KVGATE]` two-head-size boot gate).
- G2 freeze closed in three layers (warm-gate cache key; ~31 s probe
  compile accepted; LMCache store wedge → KVGATE + `ARC_LMC=0` posture).
  Layer-2b: the face120k 300 s cliff was the stock execute-model RPC
  watchdog — `ARC_RPC_TIMEOUT=1200` joins the certified gemma posture.
- Both gemma lanes validated end-to-end (faces + decode + quality
  probes, erraudit 0/0); prefill root cause diagnosed at r108aa/r108ab
  (hd512 wall reading, later refined by r108ac — see cert18).
- Evidence: `results/2026-11-06-p628-gemma4/`.

## rel/v0.30.0-b70-cert11 — P6.26: MXFP4/quark lane (2026-11-06)

- cert10 engine + a two-override `ArcXpuPlatform` patch (`quark` in
  `supported_quantization` + `supports_mx() → True`, commit `7ea0f27b`) —
  the ONLY byte changes vs cert10 are `vllm_gaudi/xpu/platform.py` +pyc
  (r105b walk 294→294); the fp8 default posture is byte-identical,
  proven face-for-face same-day (r105t/r105t2, thermal hypothesis
  refuted with xpu-smi telemetry).
- Both pre-quantized MXFP4 checkpoints validated end-to-end under the
  full posture with the native `XPUMxFp4LinearKernel`; spec gates above
  the fp8 incumbent (0.9853/0.9951 vs 0.9258); decode penalty 1.21× /
  1.29× under the 2× trigger — mxfp4 = certified weight-constrained
  lane, fp8 stays the performance default.
- Tag `c987c2cd`; evidence `results/2026-11-06-p626-mxfp4/`.

## rel/v0.30.0-b70-cert10 — P6.24: oneCCL fp8-wire allreduce overlay (2026-11-05)

- cert9 engine (exhaustive `vllm_gaudi/` md5 walk: diff 0) + a two-file
  oneCCL 2022.0.0 same-version rebuild overlay with the e4m3-wire
  large-ring allreduce patch (u32-packed peer stores per the sector law,
  branchless converters, min-count threshold 2M elems). **Default OFF**;
  `CCL_SYCL_ALLREDUCE_FP8=1` opts in.
- r96 rebuild fidelity PASS (ABI/perf-identical); r97 gate **1.65×** at
  the dominant 76,677,120 B AR (7.801→4.731 ms); r98 task-level quality
  parity; r99 release gate — the opt-in beats cert9 on every p65 TTFT
  face (U8k −13 %, U64k −8.2 %, U32k −3.8 %, C32k −4.6 %); r100 closed
  every remaining AR door with wire-rate measurement (v4 = 82 % of the
  wire floor, final).
- Base fork SHA `aa2ee17d` (carried from cert9); evidence
  `results/2026-11-05-p624-fp8-ar/`.

## rel/v0.30.0-b70-cert9 — P6.20: prefill host-path (2026-11-04)

- cert8 engine + `patches/step_trace.py` diagnostics (silent-off),
  **L3 off-thread LMCache store (default ON**, `arc-lmc-store` worker —
  device ops stay on the engine stream) and L6 dispatch-late. TTFT
  −0.74 s U32k / −1.64 s U64k / −0.08 s U8k; spec/identity faces
  byte-exact cert8. The r83 window trace closed the lane: U64k/U32k are
  device-compute-bound (~89 % of TTFT device-side; host path clean).
- Fork SHA `aa2ee17d`; evidence `results/2026-11-04-p620-prefill-opt/`.

## rel/v0.30.0-b70-cert8 — P6.19: TurboQuant spec-verify FIX (2026-11-04)

- The P6.18 TQ refusal lifted: the corruption reported under
  `turboquant_k8v4` was three stacked spec-verify defects — RC1 captured
  verify graphs baking the raw-KV flash fast-path, RC2 a replay-broken
  multi-row synthetic-decode dispatch (fix: **split-row dispatch**),
  RC3 the MQ-kernel draft targeting the wheel's never-used SoA ops tree
  (fix: `tq_mq_aos.py` re-port to the engine's AoS layout). Fix ships
  wheel-side as `vllm_gaudi/xpu/patches/tq_spec_fix.py`, default ON.
- All-dtype long-return gate: **zero garble on all four TQ presets AND
  the fp8 control** (the hooked module never imports on fp8). fp8 faces
  byte-family cert7 (KV pool 400,739; t0 300.65 / 84.53–84.57; spec
  pos1 0.9258). TQ lanes servable again with capacity multipliers
  1.98×–3.68× (k8v4 spec tps 54.4 solo / 236 agg8; MQ experimental mode
  284.8 agg8, +21 %).
- Fork SHA `e47739b0`; evidence `results/2026-11-04-p619-tqverify/`.

## rel/v0.30.0-b70-cert7 — P6.18: TurboQuant interim refusal (2026-11-03)

- The corruption report (k8v4 corrupt returns) reproduced and split into
  Class B (deterministic REP-LOOP attractors, all presets) and Class A
  (t0 warm-resume trajectory flips); p618c roundtest proved the SoA
  store→decode pair numerically exact on XPU, pinning the then-diagnosis
  to a 4-bit-class precision ceiling → interim fix = config-time refusal
  of every `turboquant_*` dtype. **Superseded next day by cert8** — the
  diagnosis was wrong, the refusal is lifted, and the dtype serves
  clean. Long-return gate + determinism-semantics law + reproducibility
  audit banked here.
- Fork SHA `5bf4192d`; evidence `results/2026-11-03-p618b-repro/`,
  `…-p618c-roundtest/`, `…-p618-cert7/`.

## rel/v0.30.0-b70-cert6 — P6.14: TurboQuant presets (2026-11-02)

- The TQ-aware store fix (`_arc_attn_pool_view` TQ branch first,
  kernel-block→page merge, uint8 KV_2LTD init) + all four presets
  validated per-preset: KV capacity 526k/757k/853k/975k tokens
  (2.01/2.89/3.25/3.72×), U32k/U64k TTFT −30…−35 % vs fp8, spec ≥
  target, integrity green. (The TQ spec-verify defects surfaced later —
  see cert7/cert8.)
- Fork SHA `986e97d6`; evidence `results/2026-11-02-p614-*/`.

## rel/v0.30.0-b70-cert5 — P6.13: sliced final-span store, default OFF (2026-11-01)

- `LMC_STORE_SLICE_TOKENS` (default OFF): the sliced final-span store —
  safety/regression faces GREEN, mechanism landed exactly
  (63232-class = 0; 38 = 2×19 windows; U64k TTFT −1.5…−3.5 s); the
  pre-registered ≤600 ms max-gap bar was mis-set and recorded as such —
  merged default-OFF, enabling profile experimental/uncertified.
- Fork SHA `3e268627`; evidence `results/2026-11-01-p65-cert5/`.

## rel/v0.30.0-b70-cert4 — P6.5: certified posture image (2026-11-10-31 era; 2026-10-31)

- The p65 stall-capture campaign + the candidate battery on the shipped
  image; the long-context serve profile (`--long-prefill-token-threshold
  2048` + LMC deferral pair) and the stall-lever menu that still
  document CONFIGURATION's optional postures.
- Evidence: `results/2026-10-31-p65-cert4/`, `…-p65-r48a-stall-capture/`.

## rel/v0.30.0-b70-cert3 / cert2 / cert — image-process milestones (2026-10-31 and earlier)

- cert3: the tag-built image process (manifest A/B + image-native 28/28)
  — `results/2026-10-31-p6-r47e-cert3/`.
- cert2: 0.5.5-era lmcache line — valid, older.
- cert: predates the r46u glue-alias fix — never build from it.

## Serve-layer and analysis phases (no cert — engine bytes unchanged)

- **P6.43 canonical rung ladder + no-chord padding + gemma 128K completion
  (2026-11-11)**: every context axis now keeps the canonical rung ladder
  (1K, 8K, 32K, 64K, 128K) up to its highest measured x, centralized in the
  two plotters. (1) **Labeled empty 64K slot on gemma**: the gemma boots
  never measured 64K, so the 17 gemma charts that reach 128K previously
  jumped 32K→128K with no 64K position; they now render the full 5-rung
  ladder with an empty, still-labeled 64K slot. (2) **Rung padding
  enforces no-chord centrally**: `chart()`/`band_chart()` inject an
  explicit `(rung, None)` pair for every rung a series did not measure,
  so the existing run-splitting breaks the line at the empty slot — this
  also fixed a latent defect where 4 of 5 `tq_speed_ctx` series had been
  chording 32K→128K across the unmeasured 64K rung since P6.42 (invisible
  to the tick-only audit). (3) **Gemma 128K completion**: the solo detail
  table gains its 131072 row (annex-boot faces, t1.0 decode 26.9 tok/s;
  `r140` faces carry `None` fields, cells are now None-safe), and the
  annex boot's t1.0@128K face joins the temperature comparison (both bars
  at 128K: 28 vs 27 tok/s) with the exec-success counts updated. Queue
  and efficiency tables carry the 64K "-" row with a generalized empty-slot
  note; the annex intro's stale "not merged into the charts above" is now
  the merge provenance sentence. METHODOLOGY gains the canonical-ladder
  law. (4) **Band-tag de-collision**: the picture-capture validation pass
  exposed that all 26 TTFT charts drew the right-margin class-threshold
  tags (instant/responsive/noticeable/slow) at 1.9–6.3 px baseline gaps —
  the four bands compress under a 0–300 s axis. The new `_hline_tags`
  helper (shared by all three plotters) spreads the stack order-
  preservingly (11.5 px pitch, centred, clamped inside the plot) only
  when a collision exists; uncompressed sets keep the line-hugging
  position, so the other 149 charts are byte-identical. Oracle diff vs
  P6.42 output: 136 charts byte-identical, 39 changed (19 ladder-set +
  26 band-tag-set, 6 in both) — qwen/fp8/mxfp4 outputs outside those
  sets untouched. All 175 charts rasterized at 2x (Edge headless) and
  validated: vision checks on the ladder/chord/band-tag/bar fixes, plus
  exact-box checkers — 0 overlapping pairs across 1603 value labels,
  0 across 313 band tags, 0 tag-vs-tick.
  Generator only; no engine bytes changed — cert23 stands.

- **P6.42 no-jump full-ladder axes + gemma 128K annex merge (2026-11-11)**:
  closes the last x-axis defects found by a full 175-chart tick audit.
  (1) **Break, never chord**: a `None` inside a series (or at a band edge)
  now splits the polyline into contiguous measured runs — a line can no
  longer bridge an unmeasured context with a straight segment that
  implies data there. (2) **Empty slots instead of axis jumps**: derived
  metrics that are unmeasurable at a context (queue wait, efficiency at
  the solo-only 64K rung) render that rung as a labeled empty slot on
  the full ladder (`xdomain`), instead of dropping it and making the
  axis jump 32K→128K; rows absent from a table show "-". (3) **Gemma
  128K annex merge**: `gemma-4-31b-mxfp4.md` main tables and charts no
  longer stop at 32K — the 128K annex boot's faces merge into the main
  report's data layer (cross-boot equivalence already delta-table-proven
  in the annex section: same image, same quantization posture), so the
  exec summary, per-user speed/TTFT/aggregate charts, solo decode/TTFT/
  ITL charts, queue table, KV-pool table (128K row uses the annex boot's
  pool, footnoted), use-case bars, and the cross-model all-chart now all
  run 1K→128K, with a provenance sentence under the exec summary. The
  tq per-user speed line is completed to the full five-rung ladder the
  same way. Oracle diff vs P6.41 output: 115 charts byte-identical,
  60 changed, all in the intended families; 1,602 value labels,
  0 overlaps. No engine bytes changed; cert23 stands.
- **P6.41 linear x-axis + full measured prefill coverage (2026-11-11)**:
  every context-length line chart re-spaced from log2 to a **linear
  equal-slot axis** — one evenly spaced slot per measured context (no
  log compression, no value-proportional bunching), all middle labels
  kept, and the axis stops at the highest context actually measured.
  The qwen concurrency sweep skipped 64K (sweep = 1K/8K/32K/128K) while
  the solo faces measured it: the "1 request" line, its band edge, and
  the prefill table/chart are now completed from each configuration's
  solo faces (the same single-request measurement on the same certified
  image — cross-boot drift bounded near zero by the same-contexts delta
  tables), so the prefill chart spans the full measured 1K-128K with
  64K restored; loaded lines keep honest gaps and band shading is now
  drawn per contiguous measured run (never bridges an unmeasured
  context). 12 reports / 175 charts (unchanged counts); 1,581 value
  labels, 0 overlaps. No engine bytes changed; cert23 stands.
- **P6.40 presentation overhaul — context-length x-axis + range bands
  (2026-11-11)**: every line chart re-anchored so the x-axis measures
  **context length** and concurrency becomes the plotted series (the
  prior x = concurrency charts retired); per-user speed and TTFT gain
  millstone-style **Range band charts** (the measured envelope: solo
  edge vs heaviest tested load) with dashed quality-threshold guide
  lines on the use-case bars; every chart now ships a data-precise
  *Interpretation* paragraph and sits directly under its table in the
  same section. Near-duplicate reports eliminated:
  `gemma-4-31b-mxfp4-128k.md` merged into `gemma-4-31b-mxfp4.md` as a
  full 128K annex (same-contexts delta table proves both boots agree)
  and `gemma-4-31b-mxfp8-128k.md` into `gemma-4-31b-mxfp8.md` as the
  refused-131K control — 12 reports / 175 charts (was 14 / 108).
  Generator refactor (`p139_report.py`: label-placement extraction,
  `band_chart`, per-concurrency series builders) proven
  behavior-preserving by oracle diff — 41 untouched charts
  byte-identical; 1,545 point labels, 0 overlaps across all 175.
  No engine bytes changed; cert23 stands. Reanalysis of the banked
  P6.37/P6.39/P6.39c data, no new banks.
- **P6.39 concurrency reports + gemma 128K annex (2026-11-11)**: every
  per-config report gains the measured concurrency 1–8 series and the
  paired temperature 0.0/1.0 faces; all line-chart points carry value
  labels (staggered collision-free placement — 1,360 labels, 0 overlaps
  across 108 charts); the gemma MXFP4 128K annex (max_model_len 131072,
  KV pool 288,972 tokens: solo 28.4 tok/s @ 128K, TTFT 175.9 s,
  capacity 2 concurrent requests) with the mxfp8 131K boot refusal
  banking the 8-bit boundary; the TurboQuant KV presets and the gemma
  fp8 auto-KV wall (~13.3K ctx) measured. Engineered boundary find: the
  API server enforces **prompt_len + max_tokens ≤ max_model_len**
  (probe-banked verbatim) — gemma's sentencepiece round-trip inflates
  1.026x, so the harness now sizes server-side prompt budgets
  (ctx − max_tokens − 64). Docs tooling hardened: `validate_docs.py`
  (0 errors across 29 fork docs) and a two-pass `leak_sweep.sh`. No
  engine bytes changed; cert23 stands. Banks
  `results/2026-11-11-p637b-conc/` + `results/2026-11-11-r140-gemma128k/`.
- **P6.38 docs restructure (2026-11-11)**: current-state-only living docs
  + the two-tier history architecture — `CHANGELOG.md` as the pure
  changelog (the consolidated-history section removed), per-doc archives
  `docs/history/{STATUS,TESTING,MIGRATION,PERFORMANCE,CONFIGURATION}_HISTORY.md`
  (every outgoing table/narrative banked verbatim), and
  `docs/release_notes.md` as the upstream-style release index. Era
  labels (rNN leg IDs in prose, cert-era tags, phase prefixes) stripped
  from README + all living docs; two latent staleness bugs fixed
  (CONFIGURATION Step-1 docker image cert19→cert23; draft-sampling
  ARC_IMAGE example cert19→cert23) plus the envs.md currency bugs
  (`LMCACHE_MAX_LOCAL_CPU_SIZE` 100→16, host-var count 23→24 with
  `ARC_KVDTYPE`). No engine bytes changed; cert23 stands.
- **P6.37 close-out (2026-11-10/11)**: the per-config performance reports
  (`docs/reports/`), 64 GB-RAM host defaults (`--shm-size 16g`,
  `LMCACHE_MAX_LOCAL_CPU_SIZE=16`), the `ARC_KVDTYPE` knob + measured
  dtype boundaries, `docs/PATCHES-arc.md`. Bank
  `results/2026-11-10-p637-reports/`.
- **P6.36 prefill-window decomposition (2026-11-10)**: the prefill
  window is device drain (80 % of U32k = engine-core waits on chunk
  device completion); rank skew ZERO; every B discriminator negative;
  decode D2 ceiling +8–10 % → **no remaining lawful lever ≥ 10 %**,
  cert23 stands. `results/2026-11-10-p636-legA/`.
- **P6.34 joint-matrix skinny-kernel research (2026-11-09)**: the
  fragment-op serialization law + SLM-staging law banked; ladder
  numerics never certify — lane closed research-only.
  `results/2026-11-09-p634-jm-ladder/`.
- **P6.32 Xe2 FP8-nativeity verdict (2026-11-08)**: no native FP8 dot on
  Xe2 (counter-operand conversion only; B70 = BMG-G31, 256 XMX); the
  kernel-feasibility map + Option-B backport policy.
- **P6.27 MXFP8 AutoRound validation (2026-11-06)**: the intel-ai MXFP8
  checkpoint validated zero-change on cert11 (native
  `XPUMxFp8LinearKernel`, quantized MTP drafter, spec 0.9626/0.9174) —
  a config-driven model route, no cert. `results/2026-11-06-p627-mxfp8/`.
- **P6.25 overlap-door audit (2026-11-06)**: cross-stream AR∥GEMM
  co-tenancy closed MEASURED (overlap slower than serial at every k on
  both wires; the device time-slices AR and GEMM) — every prefill door
  now measured-closed. `results/2026-11-06-p625-overlap-door/`.
- **P6.23/P6.24b/P6.24d/P6.24e doc + patch audits (2026-11-05/06)**:
  docs-currency audit, P6.24 patch audit, STATUS-ledger move,
  best-performance-postures consolidation.
- **P6.21/P6.22 device-compute lanes (2026-11-05)**: ar_quant donor
  (−4…−9 %, unadopted), FA headroom ladder (stock at 25 % of peak but
  every patch route falsified), ESIMD dense-FA v2 (0.33× stock, ceiling
  ~0.83× — under bar), TP2 one-shot allreduce (stock ring 2.06× faster
  than the one-shot ceiling) — all closed NEGATIVE.
  `results/2026-11-05-p621*/`, `…-p622-ar-lane/`.
- **P6.15 dtype/stall probes + collapse mechanism (2026-11-02/03)**:
  bystander-stall taxonomy + MNBT Pareto; the t1.0-xhigh
  sampled-collapse mechanism pinned; 12-face master matrix.
- **P6.17 prob-lane certification (2026-11-03)**: the certified line is
  `ARC_SPEC_METHOD=probabilistic` (serve-script default); every record
  face re-measured on that lane; the P6.16 opt-in hook became the
  default. `results/2026-11-03-p617a-probcert/` (+p616 lineage).
- **P6.16 draft-sampling posture (2026-11-03)**: the probabilistic
  serve hook measured (t1.0 agg +29 % fp8 … +91 % 3bit) — merged
  serve-layer. `results/2026-11-03-p616*/`.
- **P6.1–P6.13 (2026-10-30 … 2026-11-01)**: external-hit-rate,
  0.5.6+arc1 lmcache gates, p65 stall capture + cert4, p66 prefill
  anatomy (device-bound, MNBT-16384 not adopted), p67 compile REVAL
  rejected, p70 cert4-native comprehensive re-measurement + 4 h soak,
  p71 P2P transport proven clean, p73 END-stall analysis + the sliced
  store. Evidence per `results/` phase dirs.
- **P0–P5 (2026-10-27 … 2026-10-30)**: venv ground truth, platform +
  first serve, spec/graph defect fixes, cache lane (connector, hybrid
  state, elide/memo, tail-split), certification battery + certified
  image + 12-cell matrix, env/doc/lint audits + 8 h soak + the r46t
  real-world campaign (k3 recommended). Evidence per `results/` phase
  dirs.
