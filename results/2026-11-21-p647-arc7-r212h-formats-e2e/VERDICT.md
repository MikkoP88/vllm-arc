# r212h — fix generalization: mxfp8 + mxk8v4 e2e

**Date**: 2026-11-21 · **Leg**: r212h (arc7 Phase-8) · **Stack**:
build21 wheel + mx_kv overlay (native_decode.py md5 `19f662b8…` —
the r212f fix, same bridge for all formats) +
`vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212h_launch1.log`
07:46-07:59 UTC · **Tool**: `tools/ar/p147_mxkv/r212h_e2e.sh`
(two boots — mxfp8, mxk8v4 — faces c8192/c16384/c32768 n1;
compared against the launch4 pre-fix banks on the same stack)

## Verdict

**PASS — the capacity-priced split fix generalizes unchanged to
both remaining MX formats.**

| format | face | pre-fix (launch4) | fixed (r212h) | speedup |
|---|---|---|---|---|
| mxfp8 | c8192 n1 | 285.5 ms | **50.9 ms** | 5.61× |
| mxfp8 | c16384 n1 | 536.5 ms | **61.4 ms** | 8.73× |
| mxfp8 | c32768 n1 | 1038.1 ms | **84.3 ms** | 12.31× |
| mxk8v4 | c8192 n1 | 269.0 ms | **48.9 ms** | 5.51× |
| mxk8v4 | c16384 n1 | 503.2 ms | **57.3 ms** | 8.78× |
| mxk8v4 | c32768 n1 | 971.0 ms | **78.3 ms** | 12.41× |

ITL p50 ms/token. Numerics clean on both boots: 6/6 faces ok,
0 errors / 0 tracebacks, MTP acceptance in the normal class
(3.43-3.62 tokens/chunk at k3). Telemetry confirms the fixed
argument at every capture for both formats (mxfp8 cap 262400 /
stride 544; mxk8v4 cap 263296 / bt width 4114 / stride 800 — the
format-dependent pool geometry prices its own capacity).

### Lane state after r212g + r212h

All three MX formats (mxfp4, mxfp8, mxk8v4) now run the native
kernel 5.5-13× faster than their pre-fix posture and supersede
their Triton lanes at every measured face. Against the
fp8-checkpoint bar (ctrlA24: 40.4/41.1/42.5/66.8 at c8192 n1 /
c16384 n1 / c32768 n1 / c16384 n8): mxk8v4 48.9/57.3/78.3 =
1.21×/1.39×/1.84×; mxfp8 50.9/61.4/84.3 = 1.26×/1.49×/1.98×;
mxfp4 (r212g) 57.4/74.1 = 1.40×/1.74×. The excess scales with ctx
(the MX dequant walk vs fp8's FA2 walk) — the next kernel
iteration's target, after convergence + full battery per the
improvement-convergence law.

## Artifacts

`m8qa/`, `mkqa/` (per-face p139 JSON + metrics + serve snapshots +
telemetry/dispatch/erraudit), `overlay_md5.txt`, `wheel_sha256.txt`,
`r212h_launch1.log`. Rig: `/root/build/p647/r212h/`, log
`/root/build/p199/logs/`.
