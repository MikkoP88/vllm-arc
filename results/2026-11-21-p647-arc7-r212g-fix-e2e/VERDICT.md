# r212g — e2e validation of the real split fix: native supersedes Triton

**Date**: 2026-11-21 · **Leg**: r212g (arc7 Phase-8) · **Stack**:
build21 wheel + mx_kv overlay (native_decode.py md5 `19f662b8…` —
passes `max_seqlen_k=_max_sk`, the r212f fix) +
`vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212g_launch1.log`
07:35-07:43 UTC · **Tool**: `tools/ar/p147_mxkv/r212g_e2e.sh`
(one native-posture boot, m4qa, the r212b face set; guard checks
the ARGUMENT string `max_seqlen_k=_max_sk`, per the r212f audit
law — a telemetry-only edit is a no-op)

## Verdict

**PASS — the capacity-priced walk fix lands e2e; the MX-native
decode now BEATS the Triton lane at every face.**

| face | A all-Triton (r212b) | B native pre-fix (r212b) | B FIXED (r212g) | vs pre-fix | vs Triton |
|---|---|---|---|---|---|
| c16384 n1 | 76.1 ms | 497.8 ms | **57.4 ms** | 8.67× faster | **1.33× faster** |
| c32768 n1 | 111.9 ms | 960.3 ms | **74.1 ms** | 12.96× faster | **1.51× faster** |
| c16384 n8 | 382.6 ms | 549.7 ms | **109.5 ms** | 5.02× faster | **3.49× faster** |

ITL p50 ms/token; A/B sides are the banked r212b runs on the same
stack (identical wheel/image/geometry).

### Numerics

Clean: 9/9 requests ok (n8 all 8 slots), 0 errors / 0 tracebacks in
the serve log, MTP acceptance in the normal class (3.37-3.69
tokens/SSE-chunk at k3 — matches the pre-fix acceptance; no
collapse from the capacity-priced arg). Correctness independence
was already proven (per-row device-side limits bound the walk;
r212f canaries 0 at 28.6× slowdown).

### What closed the loop

The full causal chain, each step evidenced: capture dummy dilutes
`metadata.max_seq_len` to 9/13/18/5 (r212b telemetry) → the bridge
passed it as `max_seqlen_k` (r212f git audit — the earlier r212c
"falsification" was a no-op) → C++ prices `kv_tiles =
ceil(max_seqlen_k/16)` = 1 → splits frozen at 1 per captured graph
→ one workgroup per (row, kv-head) walks the full real KV
(28.61× kernel-level, r212f) → 16 MX layers ≈ the ~465 ms
kineto-silent graph wall (r212e) → the 6.5-8.6× e2e regression.
The one-line fix (`max_seqlen_k = bt_width × block_size` =
262400): kernel-free (r212f cap cell 1265.4 µs), engine-safe
(capture clean, 5:05 boot), and e2e this table.

### Posture after r212g

- The frozen-native dispatch (estimator capacity pricing) is now
  benign: native wins every walked-Σ face offline and every e2e
  face here. The `_NATIVE_MINLEN` thresholds remain dead code in
  the engine path (documented law) — retained as the tuning
  surface.
- mxfp4 decode target state: MX-native > Triton by 1.3-3.5× at
  16-32K. Remaining formats (mxfp8, mxk8v4) ride the same bridge —
  r212h validates them against the launch4 pre-fix banks.

## Artifacts

`B_native_fixed/{c16384_n1,c32768_n1,c16384_n8}/` (p139 JSON +
metrics + serve snapshots), `decoder/dispatch/telemetry/erraudit`,
`overlay_md5.txt`, `wheel_sha256.txt`, `r212g_launch1.log`.
Rig: `/root/build/p647/r212g/`, log `/root/build/p199/logs/`.
