# r212z — double-buffered K decode: OVERLAP PROVEN, WHEEL NET-LOSS (runtime-parity operand demotion); reverted

**Date**: 2026-11-21 · **Leg**: r212z (arc7 Phase-8 instrument) · **Stack**:
r212z wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`c8396abbc69b562e88d9996d16c2a637`; r212y tree + `p212z_patch.py` —
depth-1 software pipeline: prologue decodes (K,0); inside iteration D
the lookahead decodes (K,D+1) BEFORE gemm(D); `tSrK`/`tSrK1` B
fragments alternate by D parity via `auto& tKrCur = (D & 1) ? tSrK1 :
tSrK;`; TU clean, `r212z_build1.log`) + `vllm-arc:v0.30.0-b70-cert25zc`
· **Run**: `r212z_leg1.log` 23:09–23:16 UTC (three same-wheel cells: m0
/ m11 / m17 × 32768)

## Verdict

**The overlap mechanism is REAL: k_lane collapsed +606.3 → +112.4 µs.**
Decoding (K,D+) one step ahead of gemm(D) genuinely hides the
tSrK-liveness chain (r212x's ~600 µs = loads → staging → decode →
operand → DPAS serialization) under the DPAS pipe — ~490 µs of the K
lane is overlap-coverable in principle. Canary 0.0 on every cell: the
pipeline's parity mapping is byte-correct.

**But the wheel is a NET LOSS — the loop structure itself collapsed the
GEMM-1 region.** The anchors catch what the k_lane delta hides:

| cell | r212z | r212y | delta |
|---|---|---|---|
| m0 (K pipelined + V real) | 1094.1 | 1006.6 | +87.5 |
| m11 (K const + V light) | **1107.0** | 414.0 | **+693 (×2.7)** |
| m17 (K pipelined + V light) | 1219.4 | 1020.3 | +199.1 |
| k_lane (m17−m11) | +112.4 | +606.3 | −494 |
| m8 m0 oracle (mxfp8) | 2765.4 (+109.4%) | 1769.2 (+34.0%) | +996 |
| m11 mxfp8 (rode along) | 1361.2 (+3.1%) | — | in band |

m11 runs NO decode work at all (const-fill only) yet nearly tripled —
the cost is the `tKrCur` runtime-parity reference itself. An MMA B
operand selected by a runtime ternary cannot live in statically bound
DPAS operand registers; the compiler demotes the fragments to
addressable/indexed storage and every `tKrCur` access pays a
round-trip. All four touching paths collapsed on the e2m1
instantiation (const-fill, real decode, gemm) — the r212t scratch-wall
law at fragment scale. "k_lane 112" is two regressed cells
differencing, not a win; the leg script's pre-registered
`FORK: OVERLAP WON` label fired on the naive threshold and is
overridden by the anchors (script's fork fixed post-hoc: m11 > 800
→ REGION COLLAPSE takes precedence).

**Path-dependence asymmetry (rode-along evidence)**: on the e4m3
instantiation the CONST path stayed in band (m11 mxfp8 = 1361.2,
+3.1% vs r212q's 1320.5) while the REAL path exploded (m0 mxfp8
2765.4, m17 mxfp8 2731.8). Two separately instantiated lanes took
different demotion layouts from the same edit — consistent with
register-pressure-driven layout collapse, not a uniform cost.

**Reverted in full**: wheel restored from `.r212ybak`
(`b64bd6602a92ca866a22944533fabdab`), tree reverted to the exact
r212y state via `p212z_revert.py` (mainloop md5 `ddf8ef96f16086166218d442df2e213c`,
r212y marker present, r212z marker and `tSrK1` absent; dequant
untouched at `523debbe` — the p212x2 guard). The r212z-state mainloop
is preserved as `mx_paged_decode_mainloop.hpp.r212zbak`
(md5 `675a9863f89c24855a83fc6d5c18ea8d`). The r212z wheel
(c8396abb) was not snapshotted before the restore — reconstructible
in ~16 min from `.r212zbak` + the standing inner build script.

## The fork

**r212aa: template-split the probe apparatus out of the shipping
instantiation (the principled attractor escape — now top priority).**
Reasoning: (1) the two biggest known recoverable chunks are the
attractor basin taxes — ~290 µs on mxfp4 (m0 1006.6 attractor vs 718.0
clean r212u) and ~450 µs on mxfp8 — and the split attacks both at
once; (2) every future K-lane result needs trustworthy cross-wheel
absolutes, which only a clean shipping instantiation provides; (3) the
K overlap is proven worth ~490 µs but needs **compile-time parity**
(peel the D-loop by 2 so each half statically names its buffer — the
ternary folds; doubles the K-branch body, dice risk) or a
single-buffer restructure — deferred behind the split so its result
lands on clean absolutes. Design: `mainloop_core<bool kInstr>` (or a
templated lambda over the two GEMM regions) with every probe branch
under `if constexpr (kInstr)`; dispatch once on `params.vlane_probe ==
0`; cells 0/11/17 + m8 oracle ride as before, m8 in band = escape
confirmed.
