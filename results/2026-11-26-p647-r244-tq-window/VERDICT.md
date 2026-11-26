# VERDICT — r244: the stock TQ-lane prefill sliding window

**Date**: 2026-11-26 (run banked on the rig 2026-11-26 early UTC)
**Image**: `vllm-arc:v0.30.0-b70-cert28` — **stock cert28 venv, NO wheel**
(the probe validates the vendor varlen op the engine already ships;
the build5 wheel is irrelevant here by design)
**fork_sha**: `7555082f8b93953bac0ab75496e1b05840b635db`
**Harness**: `tools/ar/p147_mxkv/p244_varlen_window.py` via `tools/ar/r244_varlen_window.sh` (W14 quiesced, EXIT-trap restored)
**Cells**: d128 (HQ=8 HK=2) + d256 (HQ=8 HK=2) — the TQ prefill lane shapes

## VERDICT: PASS — the stock varlen op honors window_size; use it

The stock `_xpu_ops.flash_attn_varlen_func` already threads
`window_size=(left, right)` — the fork's engine simply never passed it
(`_flash_attn_varlen` override was window-blind). This probe proves
the op's own window math correct at both lane shapes, so the P-C2
engine edit (pass `window_size=(sliding_window-1, 0)`) rides a
validated vendor surface:

- **Masking proven, not assumed**: the `d_blind` discriminator (the
  same output vs a window-BLIND reference) reads 0.92–1.12 — the
  windowed output is far from the window-blind one and sits at the
  stock floor vs the window-masked fp64 reference (d 7.2e-03–1.0e-02
  ≈ the stock face's own d 8.7e-03–9.4e-03).
- **Precision envelope calibrated honestly**: the stock varlen op
  emits bf16 from fp32 accumulation — native error ~1e-3/2-ulp
  against fp64 (NOT the MX kernel's 2e-3 band). Gates: per-cell stock
  floor (measured by the windowless stock face) + 2e-3 for windowed
  faces; the stock face itself gates at an absolute 5e-2 breakage
  guard. No blind loosening.
- **Boundary**: window-edge lens (63/64/65 at W=64; 1023/1024/1025 at
  W=1024) exact to the envelope.
- **Telemetry (d256, ctx 8192)**: windowed W=1024 prefill **1.266 ms
  vs 3.898 ms windowless = 3.1x faster** — the r240b gf8 cells' 40K
  first-chunk prefills (50 sliding layers) shed 75% of their prefill
  walk.

## What stands

- The r240b gf8 stock cells banked window-wrong prefill output —
  re-measure on cert29 with the P-C2 engine routing (already noted in
  the r240b bank; this probe is the fix's offline authority).
- The qwen target is unaffected (no sliding impls → the engine edit is
  a pure pass-through when `sliding_window is None`).
- e2e gemma grid (next rung) validates the engine routing end-to-end;
  this bank is the vendor-surface authority for it.

## Evidence (SHA256SUMS at bank root covers all files)

- `out/r244_summary.txt` — face-by-face results (the VERDICT source)
- `out/p244_varlen.json` — full machine-readable results + telemetry
- `logs/r244.out` — full harness log
