# VERDICT — P6.47 cert28 section-10 gate battery (r233) — PASS

**Image under gate**: `vllm-arc:v0.30.0-b70-cert28` = cert27 content +
the r231d arc7mx build28 wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`,
md5 `e6d8d337e43b907504537061414128f8`, BAKED; installed `.so` md5s
asserted in-image: `libattn_kernels_xe_2` `61d2f5a365f05084b3dd72c4ae1da3e0`,
`_vllm_fa2_C.abi3.so` `6e6c48508d40cb4c5007f27f79df97cf`, `libmqa_logits`
`190e261edc79b3b497d8d8e801e65757` — the latter two identical to cert27's,
not relinked by the header-only patch). Manifest:
`cert28_fork_sha=43f84cf977f9444900ff21387715ca5865db4c1a` (the r231d
offline+e2e evidence merge). Kernel delta over cert27: the
seqused-tightened split clamp only — `seq_num_kv_splits =
min(num_kv_splits, max(1, windowed_k_blocks/4))` on the legacy scalar
path, mirrored in `mx_paged_decode_kernel.hpp` + `mx_reduce_split.hpp`
(provenance `tools/ar/r231d_apply.py`). The mx_kv overlay is
byte-identical to cert27's (inherited via FROM; every overlay marker
re-asserted in-image).

**Harness**: `tools/ar/r233_cert28_gate.sh` (shipped md5
`af935185aefbc444588e2b86de53c298`) — derived from r228 (the cert27
battery) by `tools/ar/r233_transform.py`, which refuses on any
provenance-anchor drift; final diff = 12 hunks, no functional c27
tokens remain. Leg semantics unchanged from r228: qwen absolute,
gemma comparative vs the same-image g0 native control. Ran as stage 2
of the gated chain `tools/ar/r232_cert28_launch.sh`
(`R232C_LAUNCH_START 23:39:39` → `R232C_LAUNCH_DONE 05:10:16`
2026-11-24 UTC; banked chain log `bank/launch.log`).

## Per-leg gates (8 legs, one boot each)

| Leg | Posture | mx_backend | native_lines | LMC | LONGRETURN | T1_SEED | erraudit |
|---|---|---|---|---|---|---|---|
| ctrl | fp8 default (no ARC_KVDTYPE) | 0 (default-off PASS) | 0 | on (4 lines; certified posture) | n/a (FULL=0 by design) | n/a | 0/0 |
| m8q | mxfp8 native | 1 | 108 | auto-off (0) | PASS | PASS | 0/0 |
| m4q | mxfp4 native | 1 | 108 | auto-off (0) | PASS | PASS | 0/0 |
| mkq | mxk8v4 native | 1 | 108 | auto-off (0) | PASS | PASS | 0/0 |
| m8g | gemma mxfp8 (Triton lane) | 1 | 0 | off | bank-only | PASS | 0/0 |
| m4g | gemma mxfp4 (Triton lane) | 1 | 0 | off | bank-only | MISS simple (bank-only) | 0/0 |
| mkg | gemma mxk8v4 (Triton lane) | 1 | 0 | off | bank-only | PASS | 0/0 |
| g0 | gemma native-fp8-KV control | 0 | 0 | off | baseline banked | PASS | 0/0 |

- SUITE13 garbage probes: PASS 15/15 on every FULL leg (qwen + gemma).
- Suite-14 longprobe full + 3 churn rehits banked per FULL leg; the
  gemma REP-LOOP/finish-length prints are the confirmed native-KV
  baseline (reploop-classifier-baseline law, r164 disposition) — gated
  bank-only, never absolute.
- **GEMMA_COMPARE: PASS** — no new flag modes vs g0 (identical
  {REP-LOOP} × 4 classes on all three MX gemma legs), no window
  amplification (topwin 0 everywhere), known-class churn banked.
- Every leg's battery/text/decode probes ran clean; `P647_CERT28_GATE_COMPLETE -- PASS 04:32:12`.

## Verdict

**PASS** — cert28 carries cert27's full correctness posture with the
r231d clamp wheel baked: all three qwen MX lanes LONGRETURN+T1_SEED
absolute PASS at native_lines=108, ctrl postures intact, gemma
comparative green. No gate regressions vs the cert27 battery.

## Hygiene

- Law-8: 376 address occurrences redacted rig-side BEFORE pull
  (12 serve logs; `10.20.3.<host>`→`10.20.3.REDACTED`,
  `10.100.8.<client>`→`10.100.8.REDACTED`, perl -pi byte-exact);
  post-pull local sweep clean. `SHA256SUMS` regenerated after the
  pass (193 files, incl. the chain `launch.log`).
- W14 restored by the chain's EXIT trap: `R234: W14 HEALTH 200
  (05:10:16)`; `R232C_LAUNCH_DONE 05:10:16`.
- Evidence: `bank/` (per-leg dirs: boot/markers/battery/garbage/
  longprobe full+3 rehits/longreturn_summary/serve.log/telemetry/
  erraudit/metrics) + `gemma_compare.txt` + `launch.log`.
