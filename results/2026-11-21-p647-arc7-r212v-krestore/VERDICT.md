# r212v — k-restore rung: K DOMINANT (directional); wheel dice-regressed, m8 FAIL

**Date**: 2026-11-21 · **Leg**: r212v (arc7 Phase-8 instrument) · **Stack**:
r212v wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`0969aff271937793c158a491c6b17344`; r212u tree + `p212v_patch.py` —
probe 17 = K REAL + V LIGHT: exempt 17 from the GEMM-1 const-K gate
(`>= 6 && != 17`) and the GEMM-2 const-V gate, extend ONLY the vslm
light range to 17 (dtile keeps `<= 16` so the K walk runs its real
fast loop; mode 17 falls to the plain m11-shape tail); mainloop-only,
zero dequant edits; TU preflight clean, ninja-resume build
`r212v_build1.log`) + `vllm-arc:v0.30.0-b70-cert25zc` · **Run**:
`r212v_leg1.log` 20:20–20:26 UTC (three same-wheel cells:
m0 / m11 / m17 × 32768)

## Verdict

**GATE_FAIL (m8 stability oracle +33.8%) — but the attribution fork
is decisive and directional: K DOMINANT, V AT FLOOR.**

| cell | µs | note |
|---|---|---|
| m0 (K real + V real) | 1024.4 | +42.7% vs r212u's 718.0 — wheel regressed |
| m11 (K const + V light) | 423.8 | +11.7% vs r212u's 379.4 |
| m17 (K real + V light) | 1048.8 | |
| pool (m0−m11) | 600.7 | inflated by the wheel regression |
| **k_lane (m17−m11)** | **625.0 = 104% of pool** | the K walk |
| **v_pool (m0−m17)** | **−24.4 = −4% ≈ 0** | V fast ≡ V light |
| telescoping | 0.00 drift | k_lane + v_pool = pool ✓ |

The m8 e4m3 byte-identity oracle came back **1766.5 vs 1320.5
(+33.8%, FAIL)** — the whole kernel got slower on this wheel: the
`!= 17` edits are semantically null for vprobe 0 (all three gates
evaluate the same for 0), so this is a compilation-layout regression
centered on the edited hot region (the GEMM-1 K loop — which is also
where m8's e4m3 arm lives). Absolute numbers on this wheel are not
interpretable; the ±6% dice band does not cover +34%.

What IS robust (internal to one wheel, all cells sharing the code):
**the V fast loop adds NOTHING over the light tail with K real
(v_pool ≈ 0, slightly negative) — the V direct-bf16 walk is at its
floor. The pool is the K walk.** This closes the r212m–r212u
mis-attribution story: the six-step falsification chain was
measuring K-severed pairs correctly (V-side deltas all ≈ 0) and
calling their baseline difference "the V assembly pool"; the actual
cost was the K walk (copy class + dequant_dtile + reorder + gemm
interleave) that every m-cell severed.

### The fork

**Next (r212w): K-DECOMPOSITION RUNGS on a fresh-dice wheel** —
m18 = K real + reorder SEVERED (dequant_dtile runs, tSrK const-filled
after) prices loads+staging+decode; m20 = K real + dtile LIGHT TAIL +
real reorder prices staging+reorder; with m0/m11/m17 anchors the
full split is: reorder = m17−m18, (a)+(b) = m18−m11,
(a) = (m20−m11)−(m17−m18), (b) = (m18−m11)−(a). The m8 oracle gates
whether the new wheel's absolute numbers count; if the dice lands
bad again, the layout sensitivity itself becomes the investigation.

## Artifacts

`offline/p212v_{m0,m11,m17}_32768.json`, `r212v_build1.log`,
`r212v_tu.log`, `r212v_leg1.log`. Rig: `/root/build/p199/out/p212v_*`,
`/root/build/r212v_{build1,tu,leg1}.log`. Tools: `p212v_patch.py`,
`r212v_krestore.sh`.
