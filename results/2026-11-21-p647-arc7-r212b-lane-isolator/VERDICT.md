# r212b — same-stack lane isolator: the split-starvation law

**Date**: 2026-11-21 · **Leg**: r212b (arc7 Phase-8) · **Stack**:
build21 wheel `ae8961bf…` + mx_kv overlay (engine.py `f34349e7…`) +
`vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212b_launch1.log`
06:29-06:47 UTC · **Tool**: `tools/ar/p147_mxkv/r212b_ab.sh` (two
boots, m4qa, faces c16384 n1 / c32768 n1 / c16384 n8)

## Verdict

**PASS — root cause isolated in one leg.** Two boots, everything
identical but the lane branch (A: `ARC_MXKV_NATIVE_MINLEN=mxfp4:
999999999` → all-Triton on the treatment stack; B: default →
native-frozen), plus the r212b capture telemetry in the bridge
(`native_decode.py` one line per (fmt, rows, max_seqlen_k, bt_width)
class — md5 `d52d6b6b…` at run time).

### A/B (ITL p50 ms/token)

| face | A all-Triton | B native | ratio |
|---|---|---|---|
| c16384 n1 | 76.1 | 497.8 | 6.5× |
| c32768 n1 | 111.9 | 960.3 | 8.6× |
| c16384 n8 | 382.6 | 549.7 | 1.4× |

The dispatch-inert control REPRODUCED on the current stack (A
c32768 n1 111.9 vs the old r212a_ctl 112) — the launch4 A/B table
stands; native's regression is real, not a stale control.

### The smoking gun (B telemetry)

```
MXKV native call fmt=mxfp4 rows=148 max_seqlen_k=9  bt_width=4100 stride=544
MXKV native call fmt=mxfp4 rows=4   max_seqlen_k=13 bt_width=4100 stride=544
MXKV native call fmt=mxfp4 rows=4   max_seqlen_k=18 bt_width=4100 stride=544
MXKV native call fmt=mxfp4 rows=256 max_seqlen_k=5  bt_width=4100 stride=544
```

**SPLIT-STARVATION LAW**: `max_seqlen_k` freezes per captured graph
at the capture DUMMY's diluted `metadata.max_seq_len` (9/13/18/5 —
the same dilution the frozen-native law exposed), and the C++
auto-split heuristic prices the split count from it
(`kv_tiles = ceil(max_seqlen_k/16)` → 1 → `num_kv_splits = 1`
frozen). At replay, each (row, kv-head) walks its full REAL KV on a
single workgroup — the device-side per-row limits keep it correct
(accuracy passed everywhere), but the walk is serialized. The
ratio's batch-dependence is the tell: n8 (32 rows × 2 heads = 64
walk WGs) narrows to 1.4× while n1 (8 WGs) blows out to 8.6×.

This also reconciles the offline-vs-engine magnitude gap: the p204
harness always passed the TRUE ctx as max_seqlen_k (splits priced
correctly); the engine froze 1. The "~100× offline-vs-engine" gap
in the launch4 verdict was split starvation, not harness overhead
alone.

## Fix (r212c, in flight)

Price the frozen `max_seqlen_k` at the bucket's walk CAPACITY —
`bt_width × block_size` (4100×64 = 262400): the per-row device
limits bound the true walk regardless (correctness independent of
this value), so the frozen split count simply covers the worst-case
replay class and short rows early-exit their splits. One-line
bridge change (`native_decode.py` md5 `19934f1c…`), validated by
`r212c_min.sh` (single native boot, same three faces) against the
banked A-side numbers.

## Artifacts

`A_triton/ B_native/` (per-face p139 JSON + metrics + serve
snapshots + decoder/dispatch/telemetry/erraudit), `overlay_md5.txt`,
`wheel_sha256.txt`, `r212b_launch1.log`. Rig: `/root/build/p647/
r212b/`, log `/root/build/p199/logs/r212b_launch1.log`.
