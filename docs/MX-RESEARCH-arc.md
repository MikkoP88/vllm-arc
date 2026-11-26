# vllm-arc — MX design-space research (P-E)

The complete MX lever map for the target stack (qwen3.8-27b-fp8 /
gemma-4-31b on dual Arc Pro B70, TP2): every model-weights-side and
KV-side MX lever with its **existence proof in the pinned stack**
(`torch 2.13.0+xpu`, the oneCCL pin — verified by the r247 census,
`results/2026-11-26-p647-r247-mx-surface/`), its **Xe2 hardware
mapping** (BMG-G31, 256 XMX; no native FP8/MX dot — counter-operand
conversion only, the xe2-xmx-fp8-nativeity verdict), its **lawful
scope** (fork law: the `vllm_gaudi/xpu/` lane), its **disposition**
(closed / certified / open candidate), and the **evidence pointer**.

This is the P-E deliverable (plan phase order: P-A..P-D closed →
P-E → the cert29 cut). Ranked open candidates feed the improvement
loop (§5); everything else is dispositioned with a measured or
structural reason. Currency: cert28 + the build6 lineage (rides
cert29).

## 1. Pinned-stack surface (r247 census)

| fact | value |
|---|---|
| torch | `2.13.0+xpu` (oneCCL-pinned; upgrades closed by the 2026-11-05 verdict) |
| MX dtypes | `float4_e2m1fn_x2` (packed e2m1), `float8_e4m3fn(+uz)`, `float8_e5m2`, `float8_e8m0fnu` — all present |
| public scaled dot | `torch.nn.functional.scaled_mm` (recipe API: per-operand scales + `_ScalingType`/`_SwizzleType`), `scaled_grouped_mm` |
| private | `torch._scaled_mm`, `torch._scaled_grouped_mm` (builtins) |
| XPU dispatch | `aten::_scaled_mm` has 4 XPU entries (kernels registered; the dot is conversion-based on Xe2) |
| aten MX format ops | **none** — no float4/e2m1/e8m0/mx-named aten ops; MX packing is not an aten surface |
| torchao | **not installed** — the torchao mx_formats branch is closed on this stack |
| de-facto MX compute | the fork's C-ext ops (`fp4_gemm_w4a16`, `fp4_gemm_w4a8` in the onednn family) + the mx_kv wheel kernels (`libattn_kernels_xe_2.so`) |

## 2. Model-weights-side levers

| lever | existence | Xe2 mapping | disposition |
|---|---|---|---|
| **mxfp4-W4A16, group-32 E8M0** | fork C-ext `fp4_gemm_w4a16` (onednn; arc4 wheel lineage) | bf16 activations, packed f4 weights, e8m0 group-32 weight scales on the `int4_gemm_w4a16` decomposition route | **CERTIFIED** — best online-quant lane: qwen mx4-W4A16 + k3 = 103.3 tok/s @ 8K, edge widens with context (`ARC_MXFP4_MODE=w4a16`; `results/2026-11-24-p647-r240-reports/`) |
| mxfp4-W4A4 (joint dtypes) | stock `XPUMxFp4LinearKernel` + `_xpu_C.fp4_gemm` | per-call activation quant to mxfp4 group-32 then joint-dtypes gemm | **closed (measured inferior)** — the p630 FIX-P reroute exists precisely to skip the per-call activation quant; shape-dependent gemm losses (`results/2026-11-08-p630-mxfp4-perf/`) |
| mxfp8-W8A16 | torch `float8_e4m3fn` + conversion | no weight savings vs fp8, bf16-activation tax | **closed (measured dud)** — ≈ baseline decode (`results/2026-11-08-p631-mxfp8-w8a16/`, decomposition `results/2026-11-10-p635-telemetry/vmatrix/`) |
| checkpoint-fp8 | certified incumbent path | native lane | **CERTIFIED** — the perf default |
| **mxfp4-W4A8** | C-ext `fp4_gemm_w4a8` (per-token activation quant + packed f4 weights — the op ships in the onednn family) | e4m3 activations on the counter-operand-conversion dot (the mechanism that runs W8A16 at ~1.03× bf16); Intel's own `greedy_w4a8` quant-GEMM kernels are xe_3-only (xe2 verdict class b) — our lane is the conversion dot, not a native-rate claim; the old W4A8 refusal was OUR dispatch default, not hardware | **OPEN candidate** (§5.3) — the op exists, the lane is hardware-legal, nothing measured |
| group size 16 / 64 | MX standard is 32; the C-ext op contract and the mx_kv layout are group-32 | 64 halves scale bytes (1 B/32 → 1 B/64); 16 doubles them | **open (offline-oracle first)** — no measured basis; coarser scales risk the K-precision acceptance class (§5) |
| per-channel E8M0 (weights) | composition of existing pieces | coarser than per-group | **open (low)** — marginal size win, error-profile risk |
| torchao mx_formats / `to_mx` | **not installed** | — | **closed (absent + scope law)** — new dependency for a surface the fork owns more directly |
| torch native `scaled_mm` recipe API | present (§1) | conversion-based dot | **closed as a vehicle today** — the dedicated C-ext ops already beat the generic path at our shapes; re-open only on a torch/oneCCL rebase (watch 2.14, upgrade verdict) |

## 3. KV-side levers

| lever | existence | disposition |
|---|---|---|
| mxfp4 / mxfp8 / mxk8v4 KV | the `mx_kv/` engine + wheel kernels (`libattn_kernels_xe_2.so`), E8M0 group-32 scales 1 B/32, K+V+scale packed per slot | **CERTIFIED** — qwen (r234–r238 banks) and, since P-C, gemma at D=512/page-128 (r245: 37/37, all three dtypes). MX KV: 0.81–0.84× fp8 decode with 1.28–1.88× KV-token capacity; n1 ITL mxfp4 1.37× at 32K |
| E8M0 scale encoding | `mx_quant.py`: `2^ceil(log2(amax/EMAX))`, codes [0,254], 255 = NaN | **closed (implemented, oracle'd)** — the packing walk is scale-inline (`mx_layout.py`) |
| **K/V precision mixing** | the acceptance-k-precision law (r238): **e2m1 K costs ~10 % MTP acceptance/step (deep positions), e2m1 V is free**; mxk8v4 = K:e4m3 + V:e2m1 is the certified mixed lane | **the frontier** — see §5 |
| per-head / per-token scale granularity | current = group-32 per (token × head-group) inline; per-head coarser, per-element not MX | **open (low)** — scale bytes are ~3 % of payload at g32; coarser risks corruption bands |
| int4 KV, per-token-per-head fp16 scales | banked wheelref only (`mx_quant.py` survey reference) | **closed as reference** — outside the MX lane family, no measured basis |

## 4. The mixed-precision frontier (K/V)

The acceptance law is asymmetric: **the K operand's precision drives
MTP acceptance (e2m1 K ≈ −10 %/step at deep positions), the V
operand's is free** (r238, `results/2026-11-24-p647-r238-n8-accept-mining/`).
That asymmetry defines the whole mixed-lane space:

| lane | K / V | acceptance | capacity | status |
|---|---|---|---|---|
| mxk8v4 | e4m3 / e2m1 | parity (V-side free) | V-half | **certified** (r245: 18.7 tps @ 1K n1, pool 71,856) |
| mxfp4 | e2m1 / e2m1 | −10 % class | both halves | **certified with known acceptance cost** (r238) |
| mxfp8 | e4m3 / e4m3 | parity | none (1.0×) | **certified** (r234–r237) |
| **k4v8** | e2m1 / e4m3 | −10 % class (K-side cost) | **none — byte-identical to mxk8v4 at every D** (r250) | **closed-dominated** (r250: no face where it can beat mxk8v4) |
| k4v4 | e2m1 / e2m1 | −10 % class | both | == mxfp4 (same operands) — covered |

## 5. Ranked open candidates (feeds the improvement loop)

1. **group-64 E8M0 reach** — halves scale bytes everywhere (weights +
   KV); offline oracle first; acceptance probe mandatory (coarser
   scales, K-side risk — the same operand the r250 disposition
   convicts).
2. **mxfp4-W4A8 weights lane** — the op ships (`fp4_gemm_w4a8`), the
   lane is hardware-legal per the xe2 verdict; activation quant is
   per-token (not per-call group-32) so the p630 tax class does not
   apply. Measure decode + ITL vs W4A16 at 8K–128K.
3. per-channel E8M0 / per-head KV scales — marginal; offline-oracle
   only if 1–2 leave headroom.

(The former #1 — the k4v8 KV lane — closed dominated at P-I r250;
see §6.)

Post-cert29 rungs by default (§10 gates first); nothing here blocks
the cert29 cut — every blocking issue is closed (P-C kernel, P-D
gate).

## 6. Closed channels (with the reason)

- k4v8 KV lane (P-I r250, 2026-11-26): dominated by certified
  mxk8v4 — slot bytes are symmetric in the K/V width mix, so k4v8 is
  byte-identical at every D (200/400/800 B at 128/256/512 — the
  r247 lever map's "K-half capacity" framing was the slip); it pays
  the twice-convicted e2m1 K (r238 acceptance −10%/step; cert24
  gate run1: the one REAL quality-limit failure, essay REP-LOOP
  under churn) on the only axis that gates, to upgrade the operand
  two batteries measured quality-free (e2m1 V: r238 sub-noise, r245
  full grid). Zero rig time — banked evidence + arithmetic.
  `results/2026-11-26-p647-pi-r250-k4v8-disposition/`.
- torchao branch: absent from the pinned image + scope law.
- W8A16: measured decode dud.
- W4A4: measured inferior (per-call activation quant).
- torch native `scaled_mm` as the compute vehicle: conversion-based
  dot, already beaten by dedicated ops; re-open on torch rebase.
- int4-KV per-token-head: reference bank only.
- Upgrade levers (torch 2.15 / oneCCL 2022.1.2): closed negative
  2026-11-05 (`torch215-oneccl2121-upgrade-verdict`).

Evidence index: r247 census
(`results/2026-11-26-p647-r247-mx-surface/`), MX KV record legs
(r234–r238 banks + `results/2026-11-24-p647-r239-record-lanes/`),
P-C phase (r242–r245 banks), W4A16 record
(`results/2026-11-24-p647-r240-reports/` + the telemetry
decomposition `results/2026-11-10-p635-telemetry/vmatrix/`), W8A16
dud (`results/2026-11-08-p631-mxfp8-w8a16/`), p630 reroute
(`results/2026-11-08-p630-mxfp4-perf/`), xe2 nativeity verdict
(master plan §6178–6207, five evidence classes; silicon
`results/2026-11-08-p631-mxfp8-w8a16/rig_raw/kbench/r123_kbench.out`),
upgrade verdict (2026-11-05 close-out).
