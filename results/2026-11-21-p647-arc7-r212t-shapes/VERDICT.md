# r212t — store-shape split: BOTH CHEAP; the pool is the fast-loop interaction itself

**Date**: 2026-11-21 · **Leg**: r212t (arc7 Phase-8 instrument) · **Stack**:
r212t wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`9a672744e9449d06251eb1f98f1fed71`; r212s tree + `p212t_patch.py` —
probe_kops → probe_mode passthrough, rungs 14/15: the m11 light tail
+ the REAL `e2m1_scaled_bf16_u32` helper call (full assembly incl.
the ternary select, base 120) with the convert-shaped store, and
trivial bits with the memcpy-shaped store; fast path byte-identical;
TU preflight clean, ninja-resume build `r212t_build1.log`) +
`vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212t_leg1.log` (four
same-wheel cells: m0 / m11 / m14 / m15 × 32768)

## Verdict

**BOTH CHEAP: m14 379.6 / m15 379.5 ≡ m11 379.9 (deltas −0.3 / −0.4
µs, noise).** The full bit-assembly INCLUDING the select is free, and
the memcpy-shaped store is free — each placed in the unrolled light
tail. Yet m0 (the real fast loop) pays +337.4 µs over the same tail.
Neither remaining suspect survives:

| cell | µs | delta vs m11 |
|---|---|---|
| m0 (fast full) | 717.3 | +337.4 = the pool |
| m11 (light) | 379.9 | — |
| m14 (light + assembly+select) | 379.6 | **−0.3** |
| m15 (light + memcpy store) | 379.5 | **−0.4** |

Combined with the chain (branches r212o, loop form r212r, domain
r212q −24 µs, converts r212p, int ops r212s), every COMPONENT of the
fast loop's body is individually priced FREE in the light context.
The 337 µs materializes only in m0's combination. The one fast-loop
feature no unrolled rung has replicated: the fast loop is a RUNTIME
loop whose body reads RUNTIME-INDEXED local arrays — `kc[r]` in vslm,
`st[v0][jb>>2]` in dtile — while every probe rung runs UNROLLED
(compile-time indices → arrays promote to registers). The codebase
documents the hazard in situ: "a runtime index here would push st[]
to scratch". Scratch loads per element are exactly a 300-µs-class
wall, and r212q ≡ r212r exactly (717.7 both) is consistent with SROA
failing to promote `kc` in BOTH builds (the prep writes/reads `kc[r]`
with a runtime loop index before the walk).

m8 byte-identity oracle 1332.2 (+0.9% vs r212q 1320.5, in band);
canaries 0; m0 on this wheel 717.3 (r212q/r212r's 717.7 reproduces —
same-binary law held across four consecutive wheels).

### The fork

**Next (r212u): VERBATIM-BODY RUNG** — probe 16 = the fast loop's
exact body as a RUNTIME loop inside the light path (kc 120-initialized
under probe_light so the read cannot be UB-folded). m16 ≈ m11 → body
innocent even runtime-indexed → CONTEXT (placement/branch) → the
disassembly route decides. m16 ≈ m0 → BODY: the runtime-indexed array
reads (scratch) → fix = constant-index everything (unroll prep + walk)
so the arrays fully promote.

## Artifacts

`offline/p212t_{m0,m11,m14,m15}_32768.json`, `r212t_build1.log`,
`r212t_tu.log`, `r212t_leg1.log`. Rig: `/root/build/p199/out/p212t_*`,
`/root/build/r212t_{build1,tu,leg1}.log`. Tools: `p212t_patch.py`,
`r212t_shapes.sh`.
