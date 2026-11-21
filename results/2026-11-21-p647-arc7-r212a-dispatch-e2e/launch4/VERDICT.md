# r212a launch4 — first full native e2e: CORRECT everywhere, dispatch frozen native, perf verdict split

**Date**: 2026-11-21 · **Leg**: r212a launch4 (arc7 Phase-8) · **Wheel**:
build21 `ae8961bf…` (split fix) · **Image**: `vllm-arc:v0.30.0-b70-cert25zc`
+ mx_kv overlay (engine.py md5 `f34349e7…` — D=256 thresholds
16384 / 12288:8192 / 8192; native_decode.py `3236351b…` bf16 bridge)
· **Run**: `r212a_launch4.log` 05:37-06:20 UTC, full r196b grid (3
formats × 9 faces) · **Banks**: `m4qa/ m8qa/ mkqa/` (per-face
p139 JSON + metrics + serve snapshot), `r212a.txt` (p181 report)

## Outcome

**Numerics: PASS. Acceptance: PASS. Dispatch design: FAIL (one law).
Perf: native-frozen is 3-8.6× the control at every matched face.**

1. **The bridge chain is CORRECT e2e**: all 27 faces CONC_OK,
   ACCEPT_KNEE_PASS, zero knees (acc 73-94% across formats/ctxs/n1/n8)
   — fp16-cast → D=256/page-64 `_16`-policy dispatch → split-priced
   walk all hold in the real engine, both TP workers
   (`decoder=native fmt=… source=native-env` at boot capture).
2. **The side check failed: TRITON NEVER ENGAGED**
   (`R212A_{M4QA,M8QA,MKQA}_TRITON_NOT_OBSERVED`, rc=1). Root cause
   is the estimator's capture-freeze interaction: the branch prices
   the captured bucket's bt **capacity** (width ~4100 blocks, sized
   by max_model_len — constant across buckets), not the walked
   length, so `walked_cap = rows × bt_width × stride` exceeds every
   threshold on EVERY captured graph. The lane freezes native at
   capture (first bucket = max-width bs37×r4) and no runtime step can
   change it (decode replays never re-run plan-build python). The
   ctx-adaptive thresholds are dead code in the engine path.
3. **Rates A/B vs r212a_ctl (dispatch-inert control, same image)** —
   matched n1 faces, ITL p50 ms/token:

   | face | control | launch4 | ratio |
   |---|---|---|---|
   | m4qa 4096 n1 | 49 | 150 | 3.1× |
   | m4qa 8192 n1 | 58 | 267 | 4.6× |
   | m4qa 32768 n1 | 112 | 960 | 8.6× |
   | m8qa 4096 n1 | 54 | 160 | 3.0× |
   | m8qa 8192 n1 | 55 | 286 | 5.2× |
   | m8qa 32768 n1 | — | 1038 | — |

   Native's e2e ctx-slope is ~12× the control's. Decomposition: both
   lanes share the ~35-40 ms weight-pass base; the gap is entirely
   in the per-ctx walk term.
4. **Control-slope anomaly (flagged, not resolved)**: the control's
   4K→32K ITL growth (49→112 ms) sits ~4× BELOW the bandwidth floor
   for walking 32K×2×272B×16 layers (~0.23 ms min at 1.2 TB/s). The
   control predates build21 and ran the lean grid; its absolute
   slope is not fully trusted until re-verified on the current boot
   stack (r212b).

## Next (r212b — isolate, then fix)

- **A/B isolator, zero new code**: same image + overlay, boot with
  `ARC_MXKV_NATIVE_MINLEN=mxfp4:999999999` (all buckets triton) vs
  default (native-frozen), one face class (c16384 n1+n8, m4qa).
  Clean lane A/B on the shipping stack — if "all-triton" lands at
  ~5 µs/layer class the control reproduces and native is the
  regression; if it lands near launch4 the regression is elsewhere.
- **Capture-time telemetry** (the telemetry directive): one posture
  line per captured bucket — rows, bt_width, walked_cap, chosen
  lane, and `num_kv_splits` (from temp_out shape [rows, heads×splits,
  D]) — rides the same leg.
- Dispatch redesign candidates after the A/B: price by rows-only at
  capture (bt width is constant), or capture per-bucket lane
  branches, or accept native-everywhere only if the walk term is
  fixed (splits suspicion: engine buckets are bs37×r4=148-row class
  vs offline B=32 — the auto heuristic may choose splits=1).
- r212d multi-SG p64 tile and the mxfp4 v2/v5 retune remain queued.

## Artifacts

`m4qa/ m8qa/ mkqa/` (27 faces: `p139_conc_c*.json`, metrics
before/after, serve snapshots), `r212a.txt`, `overlay_md5.txt`,
`serve_patch.txt`, `r212a_launch4.log`. Rig:
`/root/build/p647/r212a/`, log `/root/build/p199/logs/`.
