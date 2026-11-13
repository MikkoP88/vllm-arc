# P6.47 v3 flavor + combo sweep (cert24x, 2026-11-12)

Probe: `tools/ar/p147_mxkv/p157_v3_flavor.py` (repo, deterministic seeds),
run in `vllm-arc:v0.30.0-b70-cert24x` after leg A7b. Closes the A7b tune
gap — the A7b tune face swept v3join only, and telemetry had shown v3shift
as the mxk8v4 champion. ctx=32768, HK=8/Hq=32 (N_REP=4), B=1, D=128.

## Verdict (baked into `_MX_TUNE` as (BLOCK_N, num_warps, num_stages, flavor))

| fmt | winner | ms @32K | notes |
|---|---|---|---|
| mxfp8 | join (64,4,s2) | 1.576 | flavor-flat (shift s3 1.575); join s3 1.914 |
| mxfp4 | join (64,8,s2) | 1.156 | double-e2m1 keeps join (shift best 2.446) |
| mxk8v4 | **shift** (64,8,s2) | 2.520 | join best 3.124 — V-only e2m1 dequant is the hot op under the shared walk; shift = no cross-lane join |

## Raw rows (ms, 20-rep mean)

- mxk8v4: join n64w4s3 3.322 / n64w4s2 3.124 / n64w8s2 4.164 / n64w8s3 4.155 / n128w4s3 5.804; shift n64w4s3 3.721 / n64w4s2 3.803 / n64w8s2 **2.520** / n64w8s3 5.227 / n128w4s3 4.933
- mxfp8: join n64w4s3 1.914 / n64w4s2 1.576 / n64w8s2 2.430 / n64w8s3 2.432 / n128w4s3 3.497; shift n64w4s3 1.575 / n64w4s2 1.576 / n64w8s2 2.431 / n64w8s3 2.434 / n128w4s3 3.493
- mxfp4: join n64w4s3 3.936 / n64w4s2 3.611 / n64w8s2 **1.156** / n64w8s3 1.432 / n128w4s3 7.809; shift n64w4s3 5.420 / n64w4s2 5.412 / n64w8s2 2.458 / n64w8s3 2.448 / n128w4s3 7.609

Run-to-run variance on this rig is ~10-15% per config (A7b vs A8 same-config
spreads); winners are consistent across runs. Correctness is the A/B battery's
gate (legA7 exact faces: v3 flavors bit-equal to v2 at 549 tokens, d=0.000).
