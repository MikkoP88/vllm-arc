# r231a — the split-ladder timing decomposition (P6.47, 2026-11-23)

**VERDICT: MEASURED — the MX ITL gap root issue decomposed and
reconciled end-to-end.** The r230 question (what is the
format-independent ~6.5 ms/step ITL constant + the 0.27–0.47 ms/K
slope vs ctrl) is answered by a ctx × nsplit timing ladder of the
wheel kernel (`mx_paged_decode`) on real banked payload geometries,
bare cert27 container (`tools/ar/r231a_timing.sh` +
`tools/ar/r231_splittime.py`, rows=4 n1 verify class, pool at real
geometry, scattered block tables).

## The ladder (auto heuristic, ms per call, rows=4)

| fmt | 2K | 8K | 32K | 128K | 262K | slope ms/K |
|-----|-----|-----|------|-------|-------|------------|
| mxfp4 | 0.552 | 0.643 | 1.011 | 2.723 | 5.064 | 0.0173 |
| mxk8v4 | 0.517 | 0.667 | 1.213 | 3.494 | 6.616 | 0.0234 |
| mxfp8 | 0.510 | 0.698 | 1.285 | 3.794 | 7.347 | 0.0262 |

Forced-ns ladder (per format, in the probe logs): ns1 is 68× slower
at 262K (187–512 ms — the split parallelization is essential); auto
tracks the best forced ns within ~0.1 ms at ≥8K; at 2K a tighter
ns32-class would save only 0.14–0.29 ms/call.

## The reconciliation (e2e = 16 × per-call, net of ctrl flash)

qwen3.8-27b is a 64-layer 3:1 hybrid (config `layer_types`): **16
full-attention layers** each pay one native verify call per step; the
60 linear-attention (GDN) layers walk no KV. Predicted vs measured
(r230 e2e deltas):

- m4q slope: 16 × 0.0173 = **0.277** vs measured **0.27** ms/K — exact.
- mkq slope: 16 × 0.0234 = 0.374 vs 0.40; m8q: 16 × 0.0262 = 0.42 vs
  0.47 — within the ctrl-flash subtraction (~0.003–0.005 ms/K/layer).
- Fixed term: 16 × ~0.5 = ~8 ms gross at 8K-class ctx; net of the
  ctrl lane's own flash decode (~0.1 ms/layer at 8K) ≈ **6.5 ms —
  the r230 intercept, reproduced**.

## Root-issue statement (the always-loop target list)

1. **Per-layer multiplication is the structural frame**: no single
   call is expensive (≤1.3 ms at 32K) — 16 calls/step is the gap.
2. **Fixed term (~0.5 ms/call)**: launch + barrier + frozen-128
   over-provisioned early-exit splits + the 128-partial combine. The
   strong over-provisioning hypothesis is REFUTED (early-exits cost
   only ~0.2–0.3 ms/call at 2K, ~0 at ≥8K where auto is already
   ns128-optimal). Recoverable by capacity-tighter split pricing only
   at ≤8K walked ctx: ~0.15–0.29 ms/call ≈ 2.4–4.6 ms/step, and only
   if 262K-token requests stay covered (the graph freeze prices the
   table-width capacity — an engine-side lever with a real
   long-request tradeoff).
3. **Walk slope (the dominant long-ctx term)**: dequant-ALU bound —
   MX bytes/token are 0.53–1.03× the fp8 lane's (544/1056/800 vs
   1024 B/token/rank), yet the slope is 2–3× ctrl's net. Xe2 has no
   native MX dot (xe2-xmx-fp8-nativeity verdict) → dequant is
   structural; the lever is walk inner-loop ALU work (wheel build26
   class).

## Method + hygiene

Bare container (`sleep infinity`, no serve boot — kernel timing needs
only the wheel + idle GPU), W14 quiesced/restored via guarded trap
(HEALTH 200 at each close). Refuses: cert27 image + manifest
`cert27_fork_sha=d39ab5a26bf737d006d47a3f3125a404dca6e928` +
installed-`.so` md5 (build25 pin). Payloads: one per format bank
(r224a=mxk8v4 r229a=mxfp8 r229b=mxfp4 — fmt read from the payload
field, verified). Timing: torch.xpu events, 3 warm + 10 reps, p50.

Debug history (all six launch logs banked): a1 whole-window slice
bug; a2 multi-GB kv_off reproduction removed (offset 0); a4 the
`:fmt` arg override miscast the mxfp8 cell (payloads carry fmt — the
override was pure harm); a5 the banks' colliding basenames made the
wrapper's second docker cp overwrite the first (slot-272 refusal
surfaced it); a6 clean. rfc1918 scan on the bank + logs: zero.

## Next rung

The slope lever is the only one that moves 32K-class ITL materially
(fixed-term recovery caps at ~8K-class ctx): wheel-side walk inner-
loop ALU reduction on the arc7mx tree (builder /root/build/vxk-arc-src
law), offline ladder + r229 calibrated replay gates, then the r230
e2e face harness.
