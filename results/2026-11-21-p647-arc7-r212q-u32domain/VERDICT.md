# r212q — u32-domain assembly: partial (−3.2%); the emulation tax was real but small

**Date**: 2026-11-21 · **Leg**: r212q (arc7 Phase-8 iteration) · **Stack**:
r212q wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`b974b603a1c1cc58c47d11ee9149f84e`; r212p tree + `p212q_patch.py` —
u32-native `e2m1_scaled_bf16_u32` helper + both e2m1 fast loops
rewritten in the u32 domain: the nibble extract fused into ONE native
u32 shift (`u >> (8*j + 4*phase) & 0xF`), u32 field shifts/ORs with no
mid-chain truncation, single 2-byte memcpy at the fragment store;
TU preflight clean, ninja-resume build `r212q_build1.log` 17:02–17:18
UTC) + `vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212q_leg1.log`
17:19–17:21 UTC (one cell: m0 × 32768, both formats — m4 readout + m8
stability oracle)

## Verdict

**The u32 domain recovers −3.2% (m4 m0 717.7 vs the r212o unrolled-u16
baseline 741.8) — 24 µs of the 361 µs bit-assembly pool. The 16-bit
GRF-op-emulation tax was REAL but SMALL. With branches falsified
(r212o), tail arithmetic free (r212p, m11 ≡ m9), and domain emulation
now mostly innocent, the surviving ~337 µs of assembly cost is
STRUCTURAL: unrolled code size (icache), register-pressure
serialization, or chain ILP — not the op domain, not op count.**

### The numbers (mxfp4/mxfp8 32K)

| cell | r212q | baseline | delta |
|---|---|---|---|
| m4 m0 (fast full, u32) | **717.7** | 741.8 (r212o) | **−3.2%** |
| m8 m0 (byte-identity oracle) | 1320.5 | 1323.1 (r212o) | +0.2% ✓ |
| surviving assembly (m4 − tail 380.7) | 337.0 | 361.1 | −24.1 |
| canaries | 0 / 0 | — | ✓ |

The m8 oracle held to +0.2% — the e4m3 paths are untouched and
reproduce exactly; the m4 move is the e2m1 fast-loop rewrite itself.

### Disposition

- **PATCH STAYS** (u32 form is strictly cleaner and −3.2%); r212q wheel
  is the working state (`wheel.r212qbak` = b974b603 preserved; r212p
  backups beneath it).
- e2e: −3.2% offline does not justify a face rerun; the r212m4 faces
  remain the e2e record.
- Disassembly route attempted on the 1.5 GB `libattn_kernels_xe_2.so`
  (fatbin-embedded) and parked as a tool-fighting time sink — a cheaper
  decisive instrument exists for the top theory.
- **Next (r212r)**: no-unroll probe — the SAME u32 body under a runtime
  loop (drop `CUTE_UNROLL` from both fast loops). The unrolled body is
  ~400 instructions vs the light probe's ~100; if code size /
  cross-element interleave is the wall this collapses it, if flat the
  code-size theory is innocent and SWAR pairing (r212s) takes over.

## Artifacts

`offline/p212q_m0_32768.json`, `r212q_build1.log`, `r212q_tu.log`,
`r212q_leg1.log`. Rig: `/root/build/p199/out/p212q_m0_32768.json`,
`/root/build/r212q_{build1,tu,leg1}.log`. Tools: `p212q_patch.py`,
`r212q_u32.sh`.
