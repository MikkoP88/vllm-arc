# VERDICT — r242: MX-native paged decode at D=512 (gemma full-attention lane)

**Date**: 2026-11-26 (leg 06:10–06:11 UTC on build5; build1 defect run 2026-11-25 21:20Z)
**Image**: `vllm-arc:v0.30.0-b70-cert28` (offline harness, NO engine boot)
**fork_sha**: `7555082f8b93953bac0ab75496e1b05840b635db`
**Wheel**: `vllm_xpu_kernels-0.1.14.1+arc7mx` **build5/r243a** md5 `ead038b1f1515d6144a0246ea8d31762`
**Harness**: `tools/ar/p147_mxkv/p242_d512_offline.py` via `tools/ar/r242_d512_offline.sh` (W14 quiesced, EXIT-trap restored — `W14_RESTORE_FIRED_fullres` in `logs/r242.out`)
**Cells**: d512p64 (D=512 page 64 G=8 — the gemma-4-31b full-attention geometry) + d256p64 control (certified qwen geometry, same-wheel regression guard)

## VERDICT: PASS — the hd512 kernel-instantiation gap is closed

All 24 faces green on the build5 wheel: exact (fp64 on dequantized
cache, d ≤ 9.74e-04 / cos ≥ 0.9999986, all three formats), v2 triangle
(vs the certified Triton decoder on the same cache, d ≤ 9.76e-04),
bitwise (seq_len==1 dequant bit law, 0/8192 mismatches × 3 formats),
splits (forced-8 ReduceSplitK + default plan at ctx 8192, d ≤ 4.24e-04),
boundary (block-64 edge lens through shuffled tables, d = 1.95e-03),
ctrl (d256p64 exact+bitwise clean — no regression on the certified
geometry), telemetry.

## The defect ladder this run closes (P-C1)

1. **build1 (2026-11-25 21:20Z, md5 `3c0b9b163edc103acf3eb55d80c6bdfd`):
   16/16 D=512 faces FAILED** — corruption confined to V columns > 256:
   at D=512 q8 the output head size exceeds the single-slice V stage
   (kDecodeMaxShapeOutV=256), so the kernel runs grid.x=2 V-slices, and
   every blk_v=1 workgroup re-staged slice 0's payload and scale groups
   0..7 (slice 1 duplicated slice 0, cos ≈ 0.49). The ctrl cell passed
   clean — the defect was strictly the V-slice path. **Fix (r242b)**:
   `v_slice_bytes = blk_v * kVStagePitch` added at all three dense-frame
   V byte-load build sites in `mx_paged_decode_mainloop.hpp` + the
   MX scale-line cache window made slice-local (`[v_slice_grp0,
   +kVSliceGroups)` in `mx_dequant.hpp`). D ≤ 256 stays byte-identical.
2. **build30 crash**: the live tree's python interface was stale against
   the P-C2 schema — `splits_per_seq_dev=None` landed on
   `window_size_right` at positional 10. **Fix (r242c/build31)**: the
   interface ships wl/wr at schema positions 9/10, `None` → (-1,-1).
3. **build3 forced-8 splits FAIL** (cos ≈ 0.51, all formats; default
   passed): the host split plan priced 64-row tiles at page 64
   (`_kv_tile_from_block_size`) while the kernel walks 16-row sub-tiles
   (r212c `_16` policy) — work items covered only the first quarter of
   each seq. **Fix (r242d/build32)**: `kv_tile = 16` in the interface's
   plan call (the legacy/auto path always priced 16 — unaffected).
4. **build4 PASS** (r242b+r242c+r242d authority, 2026-11-26 04:4x);
   **build5 re-run PASS** (this bank) proves the r243a work-list clamp
   (see the r243 bank) is a no-op on windowless paths: forced-8 values
   byte-identical to build4 (mxfp8 d = 4.24e-04, mxfp4 2.43e-04,
   mxk8v4 2.44e-04).

## Telemetry — native vs the certified Triton v2 decoder (D=512)

| ctx | mxfp8 | mxfp4 | mxk8v4 |
|-----|-------|-------|--------|
| 1024 | 0.25x | 0.30x | 0.23x |
| 8192 | 1.33x | 1.82x | 1.53x |
| 32768 | 2.96x | 3.44x | 3.09x |
| 40960 | 3.35x | 3.95x | 3.53x |

Native D=512 is 3.0–4.0x faster than v2 at the gemma maxlen class
(32–40K ctx; mxfp4 peaks 252.8 GB/s effective); v2 wins below ~4K ctx.
The crossover makes the engine-side dispatch question a real one —
wiring lands with the gemma e2e grid, not in this offline bank.

## What stands

- The r240b `gkv_m*` gemma first-decode refusal (kernel-instantiation
  gap) is closed offline: D=512 page 64 is dispatched, correct, and
  fast on the same wheel that carries the window surface.
- Nothing in the certified qwen path changed behavior (ctrl clean,
  windowless forced-split values identical across build4/build5).
- Engine routing for gemma D=512 layers remains to be validated e2e —
  the gemma grid leg (r240b shape, maxlen 40960) is the next rung.

## Evidence (SHA256SUMS at bank root covers all files)

- `out/r242_summary.txt` — face-by-face results (the VERDICT source)
- `out/p242_d512.json` — full machine-readable results + telemetry
- `logs/r242.out` — full harness log (wheel md5, faces, VERDICT,
  W14 restore)
