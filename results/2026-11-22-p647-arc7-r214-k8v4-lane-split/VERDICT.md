# r214 — the K-vs-V e2e lane split: the e4m3 K side carries the dtype asymmetry (5.4x at 32K)

**Date**: 2026-11-22 · **Leg**: r214 (arc7 Phase-8 e2e attribution) ·
**Stack**: r212aa wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5 `9a01790b...`; template-split
shipping body) + `vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay
bridge · **Run**: `r214_k8v4.log` 02:23–02:30 UTC (single mxk8v4 boot,
faces c8192/c16384/c32768 n1)

## Verdict

**GATE_PASS — zero errors, zero tracebacks, 156 native-call telemetry
lines, dispatch `rows=148` on both workers (no bucket drift vs the r213
reference).** The r213-lawful engine-level attribution the offline
probe cells can no longer provide: a real-quality hybrid boot (e4m3 K +
e2m1 V) differenced against the r213 A/A-confirmed m4/m8 means on the
same wheel.

| ctx | k8v4 | m4 (e2+e2) | m8 (e4+e4) | K-excess | V-excess | chk |
|---|---|---|---|---|---|---|
| 8192 | 50.8 | 48.5 | 51.6 | +2.3 | +0.8 | ok |
| 16384 | 61.6 | 55.9 | 63.2 | +5.7 | +1.6 | ok |
| 32768 | 82.5 | 70.2 | 84.8 | **+12.3** | +2.3 | ok |

(ITL p50 ms/token, n1, MTP spec 3, graph on. K-excess = k8v4−m4 =
e4m3-K over e2m1-K; V-excess = m8−k8v4 = e4m3-V over e2m1-V; the
consistency check K+V = m8−m4 passes at every ctx. Per-user MTP tps
rides along: k8v4 71.4/61.3/43.3 vs m4 75.8/62.6/51.8.)

## Reading

1. **The e4m3 K side carries the asymmetry**: at 32K the K excess is
   +12.3 vs the V excess +2.3 (5.4x); the ratio holds ≥2.9x at every
   ctx. The m8 e2e regression (+3.8% at 32K, r213) is therefore a
   K-path phenomenon — e4m3's wider element walk on the K side costs
   disproportionately at engine shapes.
2. **The engine K lane responds strongly to decode-path changes**
   (+12.3 ms/tok between two K dtypes at the same bucket shape) — the
   K walk/decode lever class (SWAR 2-nibbles-per-op, walk restructuring)
   is re-opened at engine level despite the offline probe ladder
   (r212n probe magnitudes are shape-scoped per the sign-inversion law).
3. The V side is nearly dtype-symmetric (+2.3 at 32K): the V SLM-staged
   decode path (`dequant_vslm`) transfers well across element widths;
   no V-side lever is warranted.
4. k8v4-vs-fp8 gap at 32K is +40.0 ms/tok vs m4's +27.7: the hybrid
   confirms e2m1-K is the right KV compression posture (m4 leads k8v4
   everywhere).

**Next**: r215 — the engine-shape lane decomposition (probe severance
through the engine via `ARC_MXKV_VPROBE`/`ARC_MXKV_DIRECTBF16` env,
zero code changes) to price K/V/base/decode-tail pools at the engine
bucket shape, then pick the K-side decode/walk lever with the e2e face
as arbiter.

**Reverted**: nothing — readout leg on the converged shipping wheel;
no tree changes.
