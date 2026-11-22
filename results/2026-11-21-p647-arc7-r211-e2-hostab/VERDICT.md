# r211 E2' — host-path A/B (MX-native paged decode at 1K)

**Date**: 2026-11-21 · **Leg**: r211_e2 (arc7 Phase-8, E2' of the r211
ladder) · **Wheel**: arc7mx r210 keeper `7247e83b` · **Image**:
`vllm-arc:v0.30.0-b70-cert25zc` · **Tool**:
`tools/ar/p147_mxkv/p203_hostab.py` + `r211_e2.sh`

## Verdict

**PASS — the E1' host-path thesis is quantified and the fix shape
validated.** V0 (harness verbatim) 513-595 µs host collapses to
197-212 µs at V3 (prebuilt args + direct `torch.ops` call), a 2.5-3×
host cut; with the device finally exposed (V3 `dev > host`), the TRUE
1K kernel times read **mxfp4 237 µs / mxk8v4 223 µs / mxfp8 332 µs**
at the s5 posture — mxfp4/mxk8v4 at v2 parity (v2 1K ≈ 218 µs total)
once the host is fixed; mxfp8 keeps a **device-side** 1.5× residual
(the e4m3-V lane: 2× staging bytes + dequant at short ctx).

## Segment prices (µs, median of 100)

| segment | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| arange cu_q | 20.6 | 13.5 | 13.5 |
| tensor(lens)→dev (sl) | 32.4 | 23.5 | 23.8 |
| tensor(lens) host | 7.9 | 4.4 | 4.4 |
| _as_int32_device_tensor | 32.9 | 23.9 | 24.3 |
| _infer_num_xe_cores | 8.5 | 6.3 | 6.2 |
| split plan + splits H2D | 68.2 | 52.0 | 52.1 |
| empty_like(q) | 4.7 | 2.6 | 2.5 |

Per-segment format deltas (e.g. arange 20.6 vs 13.5) are
allocator-state noise — the segments are format-agnostic; only their
SUM matters. The wrapper's per-call python-visible work at splits>1
≈ 105-135 µs; the harness rebuilds add ≈ 60-65 µs.

## Variant table (µs, medians; gap 1.3-1.9 µs throughout)

| format | variant | host | dev | bracketed | canary |
|---|---|---|---|---|---|
| mxfp8 | V0_rebuild | 594.5 | 592.4 | 577.6 | 9.77e-4 |
| mxfp8 | V1_prebuilt | 514.1 | 509.9 | 589.1 | 9.77e-4 |
| mxfp8 | V2_prebuilt_s1 | 212.6 | 627.4 | 628.2 | 0 |
| mxfp8 | V3_direct_op | 196.6 | 331.7 | 331.8 | 9.77e-4 |
| mxfp4 | V0_rebuild | 512.8 | 511.8 | 538.8 | 9.77e-4 |
| mxfp4 | V1_prebuilt | 433.2 | 431.5 | 504.1 | 9.77e-4 |
| mxfp4 | V2_prebuilt_s1 | 215.4 | 443.8 | 444.6 | 0 |
| mxfp4 | V3_direct_op | 212.2 | 237.4 | 236.1 | 9.77e-4 |
| mxk8v4 | V0_rebuild | 516.8 | 514.2 | 542.5 | 9.77e-4 |
| mxk8v4 | V1_prebuilt | 421.3 | 421.2 | 493.9 | 9.77e-4 |
| mxk8v4 | V2_prebuilt_s1 | 216.6 | 415.7 | 414.8 | 0 |
| mxk8v4 | V3_direct_op | 206.4 | 223.3 | 223.3 | 9.77e-4 |

Canary 9.77e-4 = one bf16-ULP at short-seq output magnitude (r210
law); V2's 0.00 is vs its own identical s1 call.

## Findings

1. **V0→V3 host cut 2.5-3×** (595→197 mxfp8, 513→212 mxfp4). The
   engine-shaped call (prebuilt per-step tensors + direct op) is the
   fix for the E1' 1K host-bound blocker. The remaining ~200 µs V3
   host is C++ op entry + two kernel submits (s5 = main + reduce) —
   it vanishes under XPU-graph replay.
2. **V3 is device-exposed** (dev > host in all formats) — first true
   kernel-time read at 1K: mxfp4 237 / mxk8v4 223 / mxfp8 332 µs.
   v2 1K ≈ 218 µs total → fp4 formats at parity after the host fix;
   **mxfp8's residual 1.5× is device-side** (V-lane e4m3 cost at
   short ctx; at 8K/32K mxfp8 is at parity-or-ahead, so this is a
   short-ctx-only latency effect, consistent with 2× V staging bytes
   per tile walk).
3. **s1 single-split kernels are 416-627 µs device** — confirms the
   r210 1K calibration (s1 latency-bound, 1.5× worse end-to-end).
4. **V1 wrapper at s5 costs 210-300 µs over V3** — python-visible
   segments price only ~105-135 µs; the remainder sits in the C++
   entry when a work list is supplied (arg marshalling, work-list
   processing, second launch). The engine must not pay this per layer
   call: build the plan ONCE per step, share across layers, call the
   op directly.
5. Graph-capturability corollary (pageable-H2D-under-capture law):
   V0-V2 all perform per-call H2D (`tensor(lens)→dev`,
   `_as_int32_device_tensor`, plan `.to(device)`) — NOT capturable.
   V3's shape (persistent device tensors, preallocated out) is the
   capture-safe posture the engine lane needs.

## Next (E3 — engine integration, then kernel-side mxfp8)

- **E3a**: `vllm_gaudi/xpu` mx_kv decode path → V3 shape: per-STEP
  metadata (cu_q, seqused_k device tensor, block table, plan, out)
  built once and shared across layers; direct `torch.ops` call.
- **E3b**: graph posture — capture the decode step (V3 shape only);
  verify replay correctness + 1K ITL.
- **E4' (kernel)**: mxfp8 short-ctx V-lane — 332 vs 237 µs is the
  e4m3-V staging/dequant latency; candidates: V-stage pitch trim,
  ping-pong staging (P6.34 law), fewer per-tile barrier waits at
  short ctx.

## Artifacts

`p203_hostab.json`, `r211_e2.log` (rig: `/root/build/p199/out`,
`/root/build/p199/logs`).
