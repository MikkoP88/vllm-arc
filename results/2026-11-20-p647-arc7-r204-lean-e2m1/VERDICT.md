# r204 — lean branchless e2m1 decode + dtile law-arithmetic walk

**VERDICT: PASS — all correctness gates green (synth-v2 ALL_STAGES_OK x2,
faces 42/42, bitwise 0/1024 x3) and the e2m1 lanes drop hard: mxfp4
mode0 -48% (1.991 -> 1.037), mxk8v4 -31% (1.408 -> 0.971 = 1.62x the
certified v2 Triton decoder); mxfp8 byte-identical (1.278 vs 1.277 —
clean attribution, its lane was untouched).**

- Wheel: `arc7mx-r204` md5 `2fb3b721e71e0984f32418d5688fe59b`
  (archived `fix/vllm_xpu_kernels-0.1.14.1+arc7mx-r204.whl`).
- Sources: `mx_types.hpp` `1cd65f86` (lean e2m1), `mx_dequant.hpp`
  `76f775fc` (dtile lean walk; dequant_reg tight walk REVERTED — see
  below).

## What changed

1. **LEAN BRANCHLESS `e2m1_to_float`** (mx_types.hpp): normals assemble
   directly `s|(126+ee)<<23|m<<22`; the subnormal band (ee==0) is
   `m * 0.5` as a float multiply with the sign on a negate select
   (keeps -0.0 bitwise). The old `if (ee == 0)` defeated predication
   on both e2m1 hot walks.
2. **DTILE LAW-ARITHMETIC WALK** (dequant_dtile staged branch, the
   e2m1-K lane of mxfp4): FOUR upfront scale loads (rows t1/t1+8 x
   groups grp0/grp0+1) + `off = t0 + 2*v1 + 16*v2`,
   `goff = (off + org_elem%32) >> 5` (carry law, valid for any tile
   origin) — retires the r201b kr[8]/sc0/sc1 guard machinery on the
   one lane whose fragment form IS pinned by `assert_staged_tv_form`
   (the dtile dst fragment, ((2,8),((2,8),N)) since r199b).
3. **dequant_reg tight walk REVERTED** (compile fix): the same law on
   the COPY fragment failed preflight — `src.tv_layout()`'s thr first
   sub-mode is NOT extent 2, i.e. the copy fragment's distribution
   form differs from the dequant dst fragment's even though r199b's
   register-index 1:1 correspondence holds. Kept the r201b guarded
   walk (bitwise-proven). Next cycle: dump the copy form before
   leaning it again. (New law in CLAUDE memory:
   copy-frag-vs-dst-frag-tv-form-law.)

## 32K device ms, mode0 (r200h -> r201b -> r203b -> r204 vs v2)

| fmt | r200h | r201b | r203b | r204 | v2 | ratio |
|---|---|---|---|---|---|---|
| mxfp8 | 4.732 | 2.810 | 1.277 | **1.278** | 0.768 | 0.60x |
| mxfp4 | 3.674 | 2.363 | 1.991 | **1.037** | 0.660 | 0.64x |
| mxk8v4 | 4.422 | 2.616 | 1.408 | **0.971** | 1.575 | **1.62x** |

Telemetry face (wall, native vs v2): mxk8v4 32K 0.951 vs 1.569
(1.65x); mxfp4 0.981 vs 0.660; mxfp8 1.398 vs 0.765.

## Lane decomposition (stable anchors; the floor modes m3/m6/m8 bounce
run-to-run — const-fill law, read as lower bounds only)

- V lane (m0-m2): mxfp8 0.377 (=), mxfp4 0.836 -> 0.484, mxk8v4
  0.576 -> 0.403 (the lean e2m1 decode alone; dequant_vslm unchanged
  besides the shared decode).
- staging (m0-m1): mxfp4 0.285 -> 0.105, mxk8v4 0.165 -> 0.018.
- m2 (const-V full walk incl. K chain): 0.901 / 0.553 / 0.568; compute
  floor (m7): 0.418 / 0.163 / 0.184 -> remaining K load+dequant chain
  ~0.48 / ~0.39 / ~0.38.

## Next levers (r205)

1. e2m1 PAIR decode (one byte -> lo/hi floats, single bit extraction)
   on the V lanes (0.38-0.48 ms) and the dtile K walk.
2. Copy-form dump -> tight law walk for dequant_reg (e4m3-K chain of
   mxfp8/mxk8v4).
3. Walk base (const-V ~0.55-0.90): split-reduce + wave shaping — v2
   runs the same byte volume 1.6x faster on mxfp8, so beating v2 on
   the pure-e4m3/mxfp4 lanes eventually needs gemm/walk work, not
   just dequant lanes.
