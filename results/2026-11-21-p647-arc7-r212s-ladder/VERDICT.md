# r212s — int-op pricing ladder: FLAT; scalar int ops are FREE, the pool is the store shape or the select

**Date**: 2026-11-21 · **Leg**: r212s (arc7 Phase-8 instrument) · **Stack**:
r212s wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`7ba93472270b6537b4e23d6970844da8`; r212r tree + `p212s_patch.py` —
probe rungs 12/13: the m11 light tail + 4 / +8 dependent scalar int
ops per element (a rung = 2 shifts + xor + and, chained through the
value so nothing folds); TU preflight clean, ninja-resume build
`r212s_build1.log` 18:06–18:22 UTC) + `vllm-arc:v0.30.0-b70-cert25zc` ·
**Run**: `r212s_leg1.log` 18:23–18:30 UTC (four same-wheel cells:
m0 / m11 / m12 / m13 × 32768)

## Verdict

**THE LADDER IS FLAT: m11 379.6 / m12 (+4 ops) 379.2 / m13 (+8 ops)
380.2 µs — slope ≈ 0.08 µs/op, noise. SCALAR INT OPS ARE FREE.**
Combined with the chain (branches r212o, loop form r212r, domain
r212q −24 µs, converts r212p), every arithmetic theory for the ~337 µs
m0−m11 pool is eliminated. What remains is the difference in STORE
SHAPE and one unpriced instruction class:

| cell | µs | delta vs m11 |
|---|---|---|
| m0 (fast full) | 717.9 | +338.3 = the pool |
| m11 (light, 0 ops) | 379.6 | — |
| m12 (light + 4 int ops) | 379.2 | **−0.4** |
| m13 (light + 8 int ops) | 380.2 | **+0.6** |

Remaining candidates, precisely: **(a)** the 2-byte `__builtin_memcpy`
into the 2-byte `value_type` local + store-from-local (the fast loop's
store shape; every probe tail instead stores `value_type(float(...))`
through the convert path), **(b)** the ternary SELECT inside
`e2m1_scaled_bf16_u32` (the ladder's dummy ops had no select), or
**(c)** an assembly-specific interaction. m8 oracle 1340.7 (+1.5%, in
band); canaries 0; m0 on this wheel 717.9 (r212q/r212r's 717.7
reproduces — same-binary law held).

### The fork

**Next (r212t): STORE-SHAPE SPLIT** — probe 14 = light tail + the REAL
helper call (full assembly incl. the select, base 120) with the
convert-shaped store; probe 15 = light tail + trivial bits with the
memcpy-shaped store. m14−m11 ≈ 0 ∧ m15−m11 big → MEMCPY CONFIRMED →
store-constructor fix (no 2-byte local; bitcast/from_bits form).
m14−m11 big ∧ m15−m11 ≈ 0 → SELECT → algebraic select removal. Both
cheap → the fast-loop interaction itself → disassembly route decides.

## Artifacts

`offline/p212s_{m0,m11,m12,m13}_32768.json`, `r212s_build1.log`,
`r212s_tu.log`, `r212s_leg1.log`. Rig: `/root/build/p199/out/p212s_*`,
`/root/build/r212s_{build1,tu,leg1}.log`. Tools: `p212s_patch.py`,
`r212s_ladder.sh`.
