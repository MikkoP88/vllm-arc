# r215 — the engine-shape lane decomposition: the FLOOR dominates, lane marginals are tiny, and the pool structure is now priced

**Date**: 2026-11-22 · **Leg**: r215 (arc7 Phase-8 e2e decomposition) ·
**Stack**: r212aa wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5 `9a01790b...`; template-split
shipping body) + `vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay
bridge · **Run**: `r215_engine_decomp.log` 02:35–03:00 UTC (four boots,
one c32768 n1 face each) · **Instrument**: probe severance THROUGH the
engine — the C++ config `run()` reads `ARC_MXKV_VPROBE` /
`ARC_MXKV_DIRECTBF16` from the serve env (two inert passthrough lines
added to `serve_arc_b70.sh`, zero kernel/wheel changes). Bucket shape
held at `rows=148` on every cell (r213 reference); acceptance-collapse
mechanics visible in chunk counts (52/101/169/192) without moving the
verify-step shape.

## Verdict

**GATE_PASS — all four cells clean (0 errors, 0 tracebacks), dispatch
`rows=148` on all cells (no bucket drift), env knobs confirmed live
(cell values diverge far beyond the 0.0-0.1% A/A band).**

| cell | env | itl p50 (ms/step) | meaning |
|---|---|---|---|
| full (r213) | — | 70.2 | shipping state (DIRECTBF16 on) |
| dbf0 | `ARC_MXKV_DIRECTBF16=0` | 75.6 | float decode tail, correct output |
| v6 | `ARC_MXKV_VPROBE=6` | 68.6 | K severed (V+P+base) |
| v2 | `ARC_MXKV_VPROBE=2` | 65.2 | V severed (K+P+base) |
| v8 | `ARC_MXKV_VPROBE=8` | 50.8 | floor (K+V+P severed/const) |
| fp8 bar | — | 42.5 | stock path (ctrlA24 cert25zc) |

Engine lane table (per step, c32768 n1, m4 lane):

- isolated K lane (v2−v8): **+14.4** · isolated V lane (v6−v8): **+17.7**
- **marginal K given V live (full−v6): +1.6** · **marginal V given K live
  (full−v2): +5.0** · K+V union (full−v8): **+19.4** vs isolated sum
  32.1 → **12.7 of shared/overlapped cost** (the r201 prefetch
  interleave is doing real work)
- **floor over the whole fp8 step: 50.8 − 42.5 = +8.3** (K/V/P const,
  all structure live, still 1.20x fp8 with FREE KV)
- decode-tail pool (dbf0−full): **+5.4 — already banked** (DIRECTBF16
  ships on; dbf0 prices what we'd be without it)
- MX total gap: **+27.7**

## Readings

1. **The floor is the first-class target.** At 8K the m4−fp8 gap is
   8.1 ≈ the 8.3 floor; at 32K it is 27.7 ≈ floor + the 19.4 K/V union.
   The floor is ctx-flat structural cost: the native kernel's skeleton
   (tile walk, staging structure, const-operand GEMMs, per-call
   overhead) costs more than the ENTIRE stock fp8 attention step. Even
   with zero-cost KV we would sit 1.20x fp8.
2. **Lane marginals kill the K-walk lever class at engine shape.**
   Severing the whole e2m1 K lane saves only +1.6 ms/step when V is
   live; V is +5.0. The r212n probe magnitudes (K ~178 µs dominant per
   call) are shape-scoped and do NOT transfer — consistent with the
   sign-inversion law. r214's +12.3 e4m3-K excess is therefore NOT an
   isolated-K-lane cost: it must ride the SHARED pool (bandwidth /
   codegen basin of the joint schedule), because the isolated K lane
   is only worth 14.4 and the marginal 1.6.
3. **The K+V union is subadditive by 12.7** — K and V traffic overlap
   heavily (r201 interleave). Further "overlap one more thing" levers
   (r212z/r212ab class) attack marginals worth ≤5 — dead at engine
   shape. The union's remaining 19.4 is real byte traffic + scheduling
   at half fp8's KV bytes.
4. **The decode-tail (ALU/output-path) pool is 5.4 and already
   banked** by DIRECTBF16 in the shipping state — the SWAR
   2-nibbles-per-op class attacks work inside the lanes whose
   marginals are 1.6/5.0: structurally incapable of closing the 27.7
   gap even if free.

## Consequence — the lever ranking flips

The remaining pool to fp8-level decode splits as **floor 8.3 (flat) +
K/V union 19.4 (scales with ctx)**. Both are kernel-structural, not
lane-local. Next (r216): the zero-code e2e tile-tune sweep —
`ARC_MXKV_V5_TFLOOR` / `ARC_MXKV_V5_SPLITS` / `ARC_MXKV_V5_BN` are
already env-threaded into the serve script; r193's offline "TFLOOR 2
wins ≤8K, neutral 32K" verdict is pre-template-split and probe-priced,
so it re-prices lawfully at the engine bucket on the r212aa wheel
(~6 min/cell, e2e face arbiter).

**Reverted**: nothing — env-instrument leg on the converged shipping
wheel; no code changes. The two serve-script passthrough lines are
inert when the env is unset (shipping posture unchanged).
