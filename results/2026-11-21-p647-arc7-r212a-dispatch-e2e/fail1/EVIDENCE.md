# r212a fail1 — first native e2e attempt (fp16 query gate)

**Date**: 2026-11-21 · **Launch**: `r212a_launch2.log` (rig
`/root/build/p199/logs`) · **Bank (rig)**: `/root/build/p647/r212a_fail1`
(preserved via `mv` before the rerun; overlay engine.py md5
`d1019b78714737f016446474694def0b`, native_decode.py pre-fix)

## Outcome

**All three cells (m4qa/m8qa/mkqa) BOOT_FAIL at warmup graph capture** —
identical root cause, fixed by a one-line bridge cast; rerun is
`r212a_launch3.log` / bank `/root/build/p647/r212a`.

## Root cause

The certified engine boots **fp16 activations** (`dtype=torch.float16`,
qwen3.8-27b-fp8 posture — see the non-default-args line in the serve log);
the Triton v2/v5 lanes are dtype-generic, but the wheel's native kernel is
bf16-specialized. The bridge passed the engine's fp16 `q_rows` straight
through:

```
RuntimeError: mx_paged_decode: query must be [total_q, heads, D] bf16
```

(`external_patches/vllm-xpu-kernels/arc7_mx_attn/csrc/xpu/attn/xe_2/mx/
mx_paged_decode_xe2.cpp:172-174` — `TORCH_CHECK(q.dtype == BFloat16 &&
q.dim() == 3)`.) The cache-pool check that precedes it PASSED, so the
uint8 MX slot-pool layout contract holds e2e. Crash site:
`compile_or_warm_up_model → _run_decode_step → _split_verify_decode →
mx_decode_native` — i.e. the FIRST bucket that froze native (all three
formats, first captured graph each).

**Fix**: `native_decode.py` casts `q_rows.to(torch.bfloat16)` (device op,
capture-safe; ≤2^-9 relative round-trip inside the fp16-dot class the MX
lanes already accept; the engine's existing `out.to(query.dtype)` closes
the loop). md5 after fix `3236351b292faedcc5796a0bddae4b22`.

## Dispatch-posture lines captured before the crash (per cell, both TP workers)

| cell | fmt | first-bucket rows | walked_cap | minlen | side |
|---|---|---|---|---|---|
| m4qa | mxfp4 | 148 | 38,835,200 | 32768 | native |
| m8qa | mxfp8 | 148 | 38,910,976 | 262144 | native |
| mkqa | mxk8v4 | 148 | 38,967,808 | 32768 | native |

The first captured decode bucket is the max-width one (rows=148 = bs37 ×
r4; bt width ≈ 4100 blocks = max_model_len class) — walked_cap ~38.8M
freezes native there as designed. No triton-side line was reached (crash
inside that first native capture); the rerun's dispatch.txt carries both
sides.

## Artifacts

Rig: `serve.log`/`boot.txt` per cell under `/root/build/p647/r212a_fail1`
(full tracebacks; `Traceback` counts 5/7/7), `serve_patch.txt`,
`overlay_md5.txt`, `r212a.txt` (report with R212A_*_NOT_OBSERVED gates).
Local: this note (extracted evidence — the fail1 serve logs stay on the
rig; the rerun bank supersedes them).
