# VERDICT — r243: the MX-native paged-decode sliding-window surface (P-C2)

**Date**: 2026-11-26 (build4 defect run 04:51–04:52 UTC; build5 authority run 06:04–06:05 UTC)
**Image**: `vllm-arc:v0.30.0-b70-cert28` (offline harness, NO engine boot)
**fork_sha**: `7555082f8b93953bac0ab75496e1b05840b635db`
**Wheel**: `vllm_xpu_kernels-0.1.14.1+arc7mx` **build5/r243a** md5 `ead038b1f1515d6144a0246ea8d31762`
**Harness**: `tools/ar/p147_mxkv/p243_window_offline.py` via `tools/ar/r243_window_offline.sh` (W14 quiesced, EXIT-trap restored)
**Cells**: d256p64 (gemma sliding geometry, page 64) + ctrl256p16 (qwen-geometry masked twin, page 16)

## VERDICT: PASS — the masked kernel twin is correct at both pages

All faces green on build5: exact (windowed native W=64 vs window-masked
fp64, d ≤ 1.95e-03 / cos ≥ 0.9999986, all three formats, both pages —
with the d_blind window-blind discriminator 0.70–0.83, proving the mask
binds), nonbinding (lens ≤ W: windowed vs windowless twin agree),
boundary (W=64 lens 1/32/63/64/65/127/128 and W=1024 lens
1000/1023/1024/1025 through shuffled tables), bitwise (seq_len==1
dequant bit law, 0 mismatches × 3 formats × 2 pages), splits (W=64 and
W=2048, forced-8 + default at ctx 8192 — d ≤ 1.95e-03), widewin (W=1024
at ctx 8192/32768/40960 mxfp4, d ≈ 4.8e-04), telemetry.

## The defect this run caught (r243a) — windowed work-list overrun

The build4 run (preserved in `pre-r243a-failrun/`) died at the splits
face: **mxfp4 forced-8 W=64 at ctx 8192 →
`UR_RESULT_ERROR_DEVICE_LOST`**. Root cause: the kernel computes the
walk bound `k_block0 = max(L-1-wl, 0)/16` and adds it to every work
item's tile start (`kv_split_offset = k_block0 + wl_tile_start`,
`mx_paged_decode_kernel.hpp:326`), but the interface's split plan was
built from FULL seq lens — at W=64/ctx 8192, split 0's item walked
tiles 508..571 of 512, past the block-table row. mxfp8 survived the
same shape on mapped garbage (every OOB column lands outside the
window and gets masked — correct output, undefined reads); mxfp4
faulted. Two-sided fix (build33):

1. **Interface** (`mx_attn_interface.py`): with a window riding the
   call, the plan prices each seq at the WINDOWED tile extent
   `(ceil(L/16) - k_block0)` padded to a tile multiple — the device
   walk now ends exactly at `k_blocks` for every (L, W) pair (verified
   exhaustively over the edge lattice).
2. **Kernel** (`mx_paged_decode_kernel.hpp`): the work-list walk is
   clamped to `k_blocks` regardless — a mispriced plan now
   under-covers visibly (a correctness failure the oracle catches)
   instead of killing the device.

The harness's splits face grew a W=2048 cell (128 windowed tiles — a
genuine 8-way windowed split through ReduceSplitK; the W=64 window
collapses to ≤ 4 windowed tiles per seq, under the ≥4-tiles/split
law): d ≤ 4.43e-04 on all formats. The windowless regression proof is
the r242 bank's build5 re-run (byte-identical forced-8 values).

## Telemetry — windowed vs windowless native decode (W=1024, d256p64)

| fmt | ctx 8192 | ctx 40960 |
|-----|----------|-----------|
| mxfp8 | 0.403 vs 0.839 ms (0.48x) | 0.271 vs 1.147 ms (0.24x) |
| mxfp4 | 0.208 vs 0.472 ms (0.44x) | 0.209 vs 0.925 ms (0.23x) |
| mxk8v4 | 0.284 vs 0.570 ms (0.50x) | 0.260 vs 1.117 ms (0.23x) |

The 40x decode-work reduction law materializes as 2.0–4.3x wall-clock
at these ctx (walk shrinks to W=1024 rows; the residual is the pool
walk overhead + softmax/reduce floor). At the gemma lane (mxfp4,
40960 ctx) windowed decode is **4.4x faster** than the windowless twin
— the sliding-layer half of gemma's attention now has a native lane.

## What stands

- The masked twin (LocalMask=true) is compiled ONLY for D=256 at page
  16/64 × qgroup ≤ 8 × 3 formats (the instantiation discipline);
  other geometries refuse loudly at the interface — unchanged.
- Engine routing for sliding impls (engine.py pass-through, native
  routing, triton-fallback refusal) rides P-C2's tracked surface and
  is validated e2e by the coming gemma grid — this bank is the kernel
  authority.
- The stock TQ-lane prefill window (r244 bank) complements this: both
  halves of the gemma window story are now offline-proven.

## Evidence (SHA256SUMS at bank root covers all files)

- `out/r243_summary.txt` — face-by-face results (the VERDICT source)
- `out/p243_window.json` — full machine-readable results + telemetry
- `logs/r243.out` — full authority-run log
- `pre-r243a-failrun/` — the build4 DEVICE_LOST run, preserved as the
  defect-discovery record (r243.out / r243_summary.txt / SHA256SUMS)
- `build-launchers/` — the build-ladder launcher scripts (r242c = build31,
  r242d = build32, r243a = build33), tracked provenance for the wheel
  md5 chain (`.r242d.md5` → `.r243a.md5`)
