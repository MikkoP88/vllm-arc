# r212m — direct-bf16 decode (exponent-folded scales): bit-exact, +13% same-wheel at 32K

**Date**: 2026-11-21 · **Leg**: r212m (arc7 Phase-8, decode-ALU iteration +
offline validation) · **Stack**: r212m wheel (`vllm_xpu_kernels-0.1.14.1
+arc7mx`, md5 `64affaf81da725e95bd6cf54cc5c6f36`; r212l tree + probe 9 +
the r212m patch, ninja resume builds `r212m_build1/2.log`) +
`vllm-arc:v0.30.0-b70-cert25zc`, harness `p204_graphtime.py` at engine
geometry (D=256 / HREP=6 / BLOCK=64, graph 32×50, SPLITS=auto) · **Runs**:
`r212m_leg1.log` (probe pre-flight 12:35-12:40 UTC: m0+m9 × 32768) +
`r212m_val1.log` (validation 13:13-13:20 UTC: m0 × {1024, 8192, 32768} +
DIRECTBF16=0 × 32768)

## Verdict

**The decode tail IS the bottleneck — and the direct-bf16 assembly is
bit-identical while reclaiming +12.6%/+13.5% same-wheel at 32K. mxfp4
lands 759.1 µs (−26.2% net vs r212l); mxfp8's cross-wheel numbers are
layout-dice-polluted (its own float path moved +29%) with a
register-pressure suspect queued for the r212m2 trim.**

### The probe law (pre-flight, `r212m_leg1.log`)

Probe 9 severs ONLY the decode tail (e2m1/e4m3 bit assembly + scale
multiply + f32→bf16 convert) at all three dequant sites; loads,
extracts, selects and fragment writes stay:

| fmt | full m0 | m9 | decode slice | share |
|---|---|---|---|---|
| mxfp4 | 1027.8 µs | 291.0 µs | 736.8 µs | **71.7%** |
| mxfp8 | 1361.8 µs | 589.0 µs | 772.7 µs | **56.7%** |

The m9 floor (291 µs) sits at the memory-bandwidth floor (~298 µs
estimated for the 268 MB/call working set) — the decode ALU is fully
EXPOSED, not hidden under load latency. (Gate NOTE in the log: m9
canary "clean" is the harness not computing canary in probe mode —
confirmed against banked r212k/r212l m2/m6 cells, all 0.0.)

### The iteration (source: `tools/ar/p147_mxkv/p212m_patch.py`)

`e2m1_scaled_bf16` / `e4m3_scaled_bf16` assemble the bf16 bits of
payload × 2^k directly with the e8m0 exponent folded in: every
e2m1/e4m3 value times a power of two is exactly representable in bf16
(≤4 significant bits vs 8), so in the normal band [1, 254] the direct
bits are IDENTICAL to float-assembly+mul+convert — no rounding
anywhere. base_e = code−1 (e2m1; the subnormal 0.5 lands on base_e,
zero nibbles return ±0) / code−7 (e4m3; normals only — payload
subnormals fall back). Out-of-band codes (denormal/overflow edges,
code 0's legacy semantics) route slow via per-group band checks (one
predictable branch per group). Env switch `ARC_MXKV_DIRECTBF16`
(default 1; 0 = float path) provides the same-wheel A/B oracle.
Scale reads became code-domain (same gmem traffic); slow elements
derive floats on demand.

### The result (`r212m_val1.log`, canaries 0.0 at EVERY cell)

Fulls vs r212l (cross-wheel — dice-polluted) and same-wheel A/B:

| fmt | ctx | fast | r212l base | net | same-wheel float (dbf0) | same-wheel gain |
|---|---|---|---|---|---|---|
| mxfp4 | 1024 | 406.5 | 594.5 | −31.6% | — | — |
| mxfp4 | 8192 | 311.3 | 420.6 | −26.0% | — | — |
| mxfp4 | 32768 | 759.1 | 1028.3 | −26.2% | 868.8 | **+12.6%** |
| mxfp8 | 1024 | 863.9 | 658.3 | +31.2% | — | — |
| mxfp8 | 8192 | 606.5 | 541.7 | +12.0% | — | — |
| mxfp8 | 32768 | 1514.4 | 1359.0 | +11.4% | 1750.3 | **+13.5%** |

1. **BIT-EXACTNESS PROVEN**: every canary 0.00e+00 — the assembly
   matches the float path exactly on live cache data, both formats,
   both K and V lanes, at every context.
2. **The fast path is strictly better on its own wheel** (+12.6/+13.5%
   at 32K): the patch STAYS, default on.
3. **The mxfp8 cross-wheel regression is NOT the fast path**: this
   wheel's float path itself moved +29% (1750.3 vs 1359.0) —
   binary-layout dice plus a structural suspect: the kc[8]+be[8]+sc[8]
   register set stays live in the e4m3 lanes (mxfp4's e2m1 path skips
   sc when in-band — exactly the lane that gained). The r212m2 trim
   (drop the be arrays, inline `int(code)−7`; derive sc only for rare
   subnormal bytes) targets it.
4. mxfp4's −26 to −32% cross-wheel includes favorable dice (dbf0
   868.8 vs r212l 1028.3 = −15.5% on unchanged-behavior float path);
   the honest attributable gain is the same-wheel +12.6%.

### Disposition

- r212m wheel is the working tree state (backups: `*.r212lbak`
  sources, `...whl.probebak`/`...whl.r212lbak`/`...whl.build21bak`
  wheels — one per generation, all in `wheels-arc-vxk/`).
- Next (r212m2, ≤30-min cycle): register-pressure trim as above, then
  re-probe 9 on the trimmed wheel for the fresh floor; after that the
  remaining decode-tail levers are the paired u32 assembly (2 nibbles
  → 2 bf16s per op chain) and tile width (the ~147 µs softmax/P+staging
  pool from r212k).
- e2e faces deferred to convergence (improvement-convergence law);
  the mxfp8 lane must first prove its cross-wheel regression is dice
  (r212m2 must bring the float path back near 1359) before any image
  cut.

## Artifacts

`offline/p212m_m0_32768.json` + `p212m_m9_32768.json` (probe),
`offline/p212m2_m0_{1024,8192,32768}.json` + `p212m2_dbf0_32768.json`
(validation), `r212m_build2.log`, `r212m_val1.log`. Rig:
`/root/build/p199/out/p212m*`, `/root/build/p199/logs/r212m*`,
`/root/build/r212m_{leg1,val1,build1,build2}.log`. Tools:
`p212m_probe_patch.py` (probe 9), `p212m_patch.py` (the iteration),
`r212m_probedec.sh`, `r212m_val.sh`.
