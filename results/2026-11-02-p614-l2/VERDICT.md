# p614 L2 — default-face non-regression battery on p614c2 (2026-11-02)

**Context:** owner directive 2026-11-02. The fix's gate law: with
`--kv-cache-dtype fp8_e4m3` (the certified default) NOTHING may change
vs cert5. Image `vllm-arc:v0.30.0-b70-p614c2` (manifest fork_sha
6f1a09d9, `rig/manifest.txt`). Batteries mirror the cert5 gate
(`../2026-11-01-p65-cert5/`) on the same rig.

## Image-native + posture

| Check | result |
|---|---|
| p48 battery (in-container) | **28/28 ALL_GREEN** (`rig/p48_verify.out`) |
| p73 slice-hunk checks (in-container) | **11/11 PASS** (`rig/p73_verify.out`) |
| slice envs leaked into default face | none (`SLICE_ENV_ABSENT=0`) |
| flip-first 832 tok | **8.3 s — byte-identical to cert5** |

## Default face vs cert5 family

| Face | cert5 | **p614c2 L2** | verdict |
|---|---|---|---|
| run7n t0 agg#2 / solo ×2 | 297.19 / 84.48+84.47 | **300.27 / 84.53+84.51** | in family |
| run7n t1.0 xhigh agg#2 / solo ×2 | 189.41 / 62.35+61.85 | **184.37 / 61.83+61.84** | in family (no raggedness — fp8 quality intact) |
| r42 integrity / determinism | 5/5 + 3/3 | **5/5 + 3/3** | PASS |
| WARN/ERROR lines + class-set diff vs cert5 | 44, identical | **44, `WARN_CLASS_DIFF_vs_cert5=0`** | identical class-set |
| ERROR/Traceback | 0 | **0** | PASS |

## p73 slice face on the same image

| Face | cert5 slice | **p614c2 L2 slice** |
|---|---|---|
| U32k TTFT | 23.16 s (fp8 family) | 23.53 s |
| U64k TTFT | 50.77 s | 51.66 s (cert4 same-face 51.95) |
| W_U32k_start gap p95/max | 1299/1310 ms | 1313/1324 ms |
| W_U64k_start gap p95/max | 1569/1602 ms | 1578/1598 ms |
| store classes | 3328 signature | **220×832 + 44×1664 + 38×3328; 63232-class = 0** |
| ERROR/Traceback | 0 | **0** |

All in family. (W_U64k_end shows a 15-row 110 ms tail, 0.86 s total —
minor, not a gate.)

## Operational note (first slice attempt)

The first slice-face boot FATAL'd on a rig-level race, NOT an image
defect: the battery's `docker rm -f arc-test` raced the default-face
serve's slow graph teardown — the container died Exited(137) but was
not removed; the p73 serve script's same-image freshness guard then
reused the corpse and its `docker exec -d` hit a stopped container
(daemon: "container … is not running", the sole line in the first
`boot_p73face.out`; the corpse's own log is banked as
`rig/serve_full_p73face_partial.log` — it shows a healthy boot mid-
progress when SIGTERM'd). Rerun script `rig/p614_l2_slice.sh` removes
arc-test and WAITS until `docker inspect` confirms it gone before the
30 s drain; the rerun completed cleanly (numbers above).

## Disposition

**L2 PASS — the fix is provably inert on the certified default face
(perf in family, WARN class-set identical, 0 errors) and the p73 slice
face is in family on the same image.** Rig addresses/hostname redacted
byte-exactly (+97/+34/+97/+5; appended to
`../REDACTION-2026-11-02-p614.md` below). W14 restored by trap.
