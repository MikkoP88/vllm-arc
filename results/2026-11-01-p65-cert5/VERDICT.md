# P6.13 cert5 — certified release close-out (sliced final-span store, default-OFF)

**Date**: 2026-11-01 · **Phase**: P6.13 close (plan §12; owner directive
2026-11-01: "cut the certified image")
**Release**: tag `rel/v0.30.0-b70-cert5` = main `3e268627` (P6.13 merge
`1195dffa` + the W14-recreate-law correction; squash lineage
`feat/arc-p65-rung3-slice`) · **Image**:
`vllm-arc:v0.30.0-b70-cert5` (ID head `6413d8aeec8a`, 45.5 GB, tag-archive
built 2026-11-01T13:30:57Z, build 726.2 s).

cert5 = **cert4's engine + the P6.13 slice hunk** in
`vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py`
(`_lmc_slice_store_window` + the flusher re-queue). The hunk is
**default-OFF** (`LMC_STORE_SLICE_TOKENS` unset ⇒ behavior byte-identical
to cert4); the experimental long-context profile (defer2 +
`LMC_STORE_SLICE_TOKENS=4096`) is opt-in. Full attribution + fix history +
the probe-leg gates: `results/2026-11-01-p65-rung3-slice/VERDICT.md`
(Gate-A lineage); the mis-set 600 ms bar and the not-adopted profile
decision live there (§2) and are unchanged by this release.

## Gate — tag-built release image, single flow (`rig/`, gate_cert5.log)

Build + verify + default posture + slice face all on the SAME tag-built
image (no separate rc leg — the code was probe-gated same-day on the
docker-commit rc image `cert4-p73probe`, adapter md5-identity proven there).

| Check | Result |
|---|---|
| Build (tools/build_image.sh, 4 ground-truth wheels) | BUILD done 726.2 s; WHEEL_GUARD — manifest lists all 4 pins with cert4-exact md5s (vllm `1f14d00b…`, triton `708e93ed…`, vxk `a51df187…`, lmcache `352295b4…`) |
| IMAGE_MANIFEST `fork_sha` | **3e26862765c412bba23c3ffffe7e018c42645033 == tag SHA** (`manifest.txt`) |
| p48_verify (28 checks, image-native) | **ALL_GREEN** (`p48_verify.out`) |
| p73_verify (slice-hunk battery, image-native) | **11/11 PASS** (`p73_verify.out`) — src-override identity, P6.5 defer hunk intact + 3 `can_save` guards (regression hold), `_lmc_slice_store_window` present, env default `'0'`, flush→helper call, chunk-row call shape (mask+slot+offset), cursor re-queue, no probe markers, compiles |
| Certified default boot (25-var, NO p65/slice envs) | HEALTH ~315 s; **SLICE_ENV_ABSENT=0** (hunk inert); first flip `stop 8.3 s` |
| run7n t0 (default posture) | steady agg#2 **297.19** / solo **84.48, 84.47** — family 294.7–300.5 / 83.7–84.9 ✓ (agg#1 272.91 = post-flip warmup face) |
| run7n t1.0 xhigh | agg **189.41** / solo **62.35, 61.85** — family 186.2 / 61.1, +1.7 % / +2.0 % (inside ±5 %) ✓ |
| r42 integrity + determinism | **5/5 + 3/3 PASS** |
| WARN/ERROR audit vs cert4 accepted set | **AUDIT_CLASSES_MATCH** — 44 filtered lines == cert4's 44; distinct message-class set identical in BOTH directions (comm empty both ways); zero fork-added classes; **0 ERROR / 0 Traceback**; 0 pin_monitor warnings (only benign `PinMonitor started` INFO ×2/log) |
| Slice face boot (defer2 + SLICE=4096, same image) | ENV_CHECK `LMC_DEFER_CHUNK_STORE=1`, `LMC_DEFER_MIN_TOKENS=40960`, `LMC_STORE_SLICE_TOKENS=4096` all in engine environ; HEALTH ~295 s |

## The slice face on the certified image (3-face capture, 12 bystanders, k3, W2MT 4000)

| Face | cert5 (slice 4096) | cert4 tag-image (no slice) | delta |
|---|---|---|---|
| U64k TTFT | **50.77 s** | 51.95 s (3-face) / 51.16 (Gate-B face) | **−1.18 s** |
| W_U64k_start p50/p95/max | 1190 / 1569 / **1602 ms** | 1242 / 1585 / **1997 ms** | max **−394 ms** (one-shot tail gone) |
| W_U64k_end p50/p95/max | 54 / 116 / 119 ms (clean) | 55 / 65 / 67 ms (clean) | +50 ms p95 — the drain tail moves HERE by design (post-answer window) |
| U32k TTFT | **23.16 s** | 23.15 s | unchanged |
| W_U32k_start p50/p95 | 749 / 1299 ms | 746 / 1297 ms | unchanged (r48g-exact) |
| W_U32k_end | 55/59/65, 0 stall rows | 55/59/61, 0 stall rows | clean both |
| Store histogram (full session) | **63232-class = 0**; **3328 × 38** (= 2 workers × 19 windows, the slice signature); 1664 × 44; 832 × 220 | **63232 × 2** (the one-shot); 1664 × 48; 832 × 230 | the signature flipped exactly as designed |

Cross-leg continuity (extends cert4 VERDICT's table): r48l 52.25 s /
1585-1612 ms / 2×63232 → rung3 probe (rc image) 50.72 s / 1561-1598 ms /
0×63232+38×3328 → **cert5 tag image 50.77 s / 1569-1602 ms / 0×63232+38×3328**
— the probe-leg mechanism result reproduces on the release artifact
(TTFT within 0.05 s, window stats in-family, histogram exact).

Default-posture histogram (whole default session: flip + t0 + t1.0 +
integrity, no 64k traffic): 832 × 86, 7488 × 10, 1664 × 6, 6656 × 2,
3328 × 2 — small-store classes only, no large one-shot (consistent: no
defer env, no 64k prompt on that face).

## Posture & disposition

- Default certified posture: **unchanged** (25-var chain, no p65/slice
  envs) — SLICE_ENV_ABSENT=0, class-identity audit, family-exact run7n
  (both temperatures) and integrity/determinism prove byte-identical
  default behavior. Cert4 defaults remain the certified serve posture.
- The slice profile stays **experimental / opt-in** per the rung3
  pre-registered rule (max-gap bar miss, recorded there §2); cert5 makes
  it *reproducible on a certified image* (defer2 profile +
  `LMC_STORE_SLICE_TOKENS=4096`), documented in CONFIGURATION-arc.
- Identities re-pointed at cert5 (README, CONFIGURATION-arc, MIGRATION,
  PERFORMANCE-arc, envs.md); evidence row added to MIGRATION.
- W14 incumbent restored by the gate's EXIT trap per the corrected
  recreate law (`w14_boot.sh fullres` → `w14_boot_cert5.log`: W13H staging
  chain + sidecar + env chain; engine init → health **200**, FRANCE probe
  correct, `W13H_BOOT_DONE`).
- P6.13 **CLOSED** (code merged default-OFF, release certified).
