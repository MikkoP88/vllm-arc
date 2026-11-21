# r212p — light-probe pricing: the tail machinery is free; the bit-assembly codegen is the wall

**Date**: 2026-11-21 · **Leg**: r212p (one-cell instrument on the r212o
tree) · **Stack**: r212p wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`6286dae5c7115700077c55f5a2f28944`; r212o tree + `p212p_patch.py` —
probe 11 light tail + prep severing at the two e2m1 sites; ninja-resume
build `r212p_build1.log` 16:23–16:39 UTC) · **Run**: `r212p_leg1.log`
16:43–16:45 UTC (single cell: VPROBE=11 × 32768)

## Verdict

**m11 (light tail: extracts + one float convert + store, prep severed)
= 380.8 µs ≡ m9 (heavy tail: + nibble decode + inline e8m0_to_scale +
mul + extra converts) = 380.6 µs. The tail machinery costs ZERO
incremental time — the walk pipelines any float-op tail freely. The
fast bit-assembly path pays +361.0 µs over this tail-invariant base
while doing FEWER ops. The wall is the u8/u16 scalar bit-surgery
CODEGEN (16-bit GRF op emulation / masked inserts), not op count, not
branches (r212o falsified), not the scale machinery.**

### The three-way split (mxfp4 32K, r212o-wheel)

| cell | µs | meaning |
|---|---|---|
| m0 | 741.8 | fast full (direct-bf16 assembly) |
| m9 | 380.6 | heavy probe tail |
| **m11** | **380.8** | light probe tail — ≡ m9 |
| c8 | 248.7 | const floor |
| m0 − m11 | **361.0** | the bit-assembly codegen's cost |
| m9 − m11 | −0.2 | tail arithmetic is hidden |
| m11 − c8 | 132.1 | extracts + stores + walk loads + softmax/P + staging |

(m8's m11 cell = 592.1 is partial — `dequant_reg` keeps its full
decode under probe 11 by the e4m3 byte-identity discipline; not an m8
instrument.)

### The fork decision

**CODEGEN LEVER confirmed.** Next (r212q): rewrite the fast-path
assembly in the u32 domain — the nibble extract, the field shifts and
the bit ORs as native 32-bit ops with a single 2-byte memcpy at the
store (current: u8 extracts + u16 assembly + u16 memcpy). If the
emulation tax is the cause this should collapse a large fraction of
the 361 µs; pairwise SWAR (2 bf16s per op chain — adjacent dst
elements i, i+1 are same-phase nibbles of ADJACENT bytes in the same
u[r], so in-WI pairing IS legal, unlike the closed input-byte
nibble-pairing) is the follow-on multiplier. If u32 is null, the spill
hypothesis (register pressure in the fast loop) takes over and the
disassembly route decides.

## Artifacts

`offline/p212p_m11_32768.json`, `r212p_build1.log`, `r212p_leg1.log`.
Rig: `/root/build/p199/out/p212p_m11_32768.json`, `/root/build/r212p_*
{build1,leg1}.log`. Tools: `p212p_patch.py`, `r212p_light.sh`.
