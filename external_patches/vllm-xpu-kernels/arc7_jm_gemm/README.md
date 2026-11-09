# arc7 kit — JM (joint-matrix / DPAS tensor-core) skinny-M GEMM (P6.34)

Owner directive 2026-11-09: take the skinny family to ALL-AROUND
SUPERIORITY over stock on every shape x lane x M cell, for ALL model
types (shape-generic; not just the two target models). This kit is the
tensor-core redesign the P6.33 close-out left open.

## What ships

| file | role |
|---|---|
| `csrc/xpu/jm_gemm/jm_gemm_kernel.hpp` | DPAS stage-1 kernel + stage-2 reduce + launcher (`jm_launch`, formula cfg) |
| `csrc/xpu/jm_gemm/jm_gemm_interface.cpp` | three torch ops; layout/scale resolvers lifted verbatim from arc6 |
| `apply_arc7.py` | rig-side patcher: copies module, ops.h decls, bindings regs, CMake line |
| `vxk_arc7_inner.sh` | in-container build (arc6 inner flow, `+arc7` version) |

Additive ops (stock + arc6 skinny + JM coexist in one wheel — probe A/B
in-process):

- `jm_fp8_gemm_w8a16(A, B, B_scale?, bias?)` — checkpoint fp8 (w16pt/w16pc)
- `jm_fp8_gemm_w8a16_mxfp8(A, B, B_scale, out_dtype?, bias?)` — MXFP8 group-32
- `jm_fp4_gemm_w4a16(A, B, B_scale, out_dtype?, bias?)` — MXFP4 packed e2m1

## Design (frozen in the kernel header; see plan §P6.34)

- **dpas.hf.hf.8.8** per 16-k step (asm verbatim from cutlass-sycl
  `XE_DPAS_TT<f,hf,hf,f>`, BSD-3): M=8 rows (real rows + zero pad — free),
  f32 accumulate.
- Fragment geometry per cutlass `mma_traits_xe` layouts: 16-thread subgroup
  owns N-tile [n0,n0+16); lane t holds weight **column n0+t, k0..k0+15** —
  exactly the stock k-contiguous uint4/uint2 → **NO weight repack in v1**.
  A: row t%8, k stride-2 (bf16→f16 in-register); D: half-row chunks
  (coalesced 16B stores).
- Packed **pair decoders** f8/f4→f16 (~7-9 ALU ops per 2 elements,
  lane-isolated SIMD-in-register) replace the ~10-15-op/byte f32 decodes —
  the measured ALU wall of the scalar family.
- MX group scales by **accumulator flush** per 32-k group
  (`d_tot += 2^e8m0 * d_grp`, f32): exact over the full e8m0 range, cheaper
  than per-element folds.
- Split-K two-stage identical to arc6 (`[S][8][N]` raw fp32 partials,
  fixed-order reduce; epilogue scale applied exactly once).
- Launch law formula-derived for ANY shape (all-model-types clause):
  tiles=ceil(N/16), subgroup floor + group-divisor + kc>=1024 + golden-
  ratio swizzle. Probe knobs `ARC_JM_SGF`/`ARC_JM_SWZ`; winners baked for
  cert.

## Bars (r128a, 50-call XPUGraph replay)

stock f8 big shapes 490-530 GB/s; small L2-resident 780-885 GB/s;
f4 lane ~300-380 GB/s actual bytes (softest). Device envelope: DRAM read
589, L2 992, copy 535 GB/s. Superiority bar: JM < stock on EVERY cell;
ceiling-bound cells close as tie-at-ceiling with math banked.

## Apply/build (rig)

```
# kit ships to /root/build/arc7kit (scp + md5 verify)
python3 /root/build/arc7kit/apply_arc7.py          # vxk-arc6-src -> vxk-arc7-src
# host-swap the builder bind to the arc7 tree (arc6 flow), then:
docker exec vxk-arc3-builder /root/vxk_arc7_inner.sh > /root/logs/arc7_build.log 2>&1
```

Gates: r128b correctness (incl. decoder exhaustive 256/256-code checks,
odd-shape cells) -> graph50 per-cell A/B vs stock+scalar -> r128 e2e
five-leg both models -> cert23.

## CLOSE-OUT (2026-11-09, cut-10): lane CLOSED — kit preserved as negative evidence

The `jm_gemm_kernel.hpp` here is the **build9 (arc9) version** (md5
`33330004…b627`): the PP arm is the SLM-V24 form (A loaded direct
from a launcher-built k-blocked f16 buffer; B staged per-sg into
`local_accessor` SLM slots, one barrier), `ARC_JM_VAR` selects
per call (5 = the proven V5 global-scratch control, byte-unchanged;
else = SLM form). Build8 (V14 ping-pong arm) never built — runtime
env-read passed as template arg; see
results/2026-11-09-p634-jm-ladder/ Part 6.

Cut-10 verdict (arc9 wheel, sha256 `54aa858f…d187`, rig-only): GATE1
80/80 exact BOTH variants — SLM staging is wheel-exact — but stage B
**0 WIN / 0 TIE / 56 LOSS**: best-of-SGF jm14 at 0.28–0.53× stock
(27–47% of the DRAM floor vs stock's 60–90%). Stock oneDNN M=1 GEMM
is already near-optimal on B70; the only exact lawful B-path (SLM
staging) bills ~2× stock. The lane is CLOSED; nothing from this kit
ships in a certified image. The four staging laws + the ladder/wheel
calibration live in results/…-jm-ladder/ Parts 5-7 and memory
`jm-global-scratch-vs-slm-law`.

| file | role |
|---|---|
| `skinny_m_jm_route.py` | the validated opt-in route (ARC_SKINNY_M=jm); banked copy of rig `r128a/skinny_m_route.py` (md5 `c7021e7a…5705`) — NOT in the live tree (reverted to cert22) |
| `vxk_arc9_inner.sh` (tools/ar/) | build9 in-container build script (resume mode via `VXK_ARC9_CLEAN=0`) |

Any future custom-GEMM lane must beat stock's 60–90%-of-floor bar
before it is worth a wheel build — and, per the r129 register-DPAS
probe (same day, results README Part 8), through **joint_matrix only**:
every register-fed lowering (raw-asm `XE_DPAS_TT`, builtin/
`__spirv_SubgroupMatrixMultiplyAccumulateINTEL`, AOT and JIT) traps on
bmg/oneAPI-2026.0; register-input dpas is unimplementable on this
stack. Custom skinny-GEMM is CLOSED on all vehicles.
