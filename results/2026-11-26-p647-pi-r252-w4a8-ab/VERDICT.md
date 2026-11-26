# VERDICT — P6.47 P-I r252: W4A8 measured dominated by W4A16 at every face — the weights frontier is closed; P-I completes (2026-11-26)

**What**: P-I improvement cycle 3 of 3 — the corrected lever map's
last pre-cert29 open candidate (#1, MX-RESEARCH §5.1). The first e2e
measurement of the `mxfp4-W4A8` weights lane: a same-boot A/B on
cert28 (`vllm-arc:v0.30.0-b70-cert28`) at the r240 q43 record lane —
online mxfp4 of the BASE `qwen3.8-27b` checkpoint
(`ARC_QUANTCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'`,
`ARC_GMU=0.85`, `ARC_LMC=1`, `ARC_FP8=0`, spec3, graphs, KV default
fp8_e4m3 — the weights lane isolated). One boot per mode
(`ARC_MXFP4_MODE` is read at install time), w4a16 control first,
w4a8 probe second. Runner: `tools/ar/r252_w4a8_ab.sh` (byte-parity
banked as `r252_w4a8_ab.sh.shipped`, rig md5 a16f96871f8ec1fe5b80e7ee34c9008a).

## The gate evidence (the lane legally ran)

- **w4a8 op present**: `mxfp4 w4a8 probe ACTIVE` at
  `mxfp4_w4a16.py:158` in serve.log — BOTH the APIServer install AND
  the Worker (`Worker pid=1026`) line; the existence-only-green-gate
  law is satisfied (the probe lane actually rode
  `_xpu_C.fp4_gemm_w4a8`, not stock W4A4). w4a16 leg shows the
  matching `reroute ACTIVE` line.
- **Both cells clean**: erraudit ERROR=0 TRACEBACK=0 on both boots;
  every face n_ok full (1/8/1); POOL identical 599,525 tokens (same
  weights bytes, same KV lane — the A/B isolates the GEMM path).
- **Coherence**: both legs produce on-topic coherent text at temp 0
  (banked `coherence.text`); acceptance canary parity (below).

## The A/B (ab_table.txt, computed from the banked p139 JSONs)

| face | metric | W4A16 | W4A8 | delta |
|---|---|---|---|---|
| c8192 n1 | per-user mean TPS | 80.77 | 45.40 | **−43.8 %** |
| c8192 n1 | ITL p50 | 37.16 ms | 67.02 ms | **+80.4 %** |
| c8192 n1 | TTFT | 3.61 s | 5.21 s | +44.4 % |
| c8192 n8 | per-user mean TPS | 19.90 | 13.15 | **−33.9 %** |
| c8192 n8 | agg TPS | 31.30 | 17.91 | −42.8 % |
| c32768 n1 | per-user mean TPS | 86.91 | 49.43 | **−43.1 %** |
| c32768 n1 | ITL p50 | 39.34 ms | 69.26 ms | **+76.0 %** |

**Dominated at every face, solo and concurrent, decode and prefill.**
The n8 ITL p90s (1.5–2.2 s both legs) are the pre-existing churn band
(ramp-interleaved ITL law — p50s are the steady-state read); solo
faces are the clean discriminator and they show a near-doubling of
inter-token latency.

**Acceptance canary — parity.** Solo 8K acceptance is byte-identical
(131/186 accepted, acc/position 0.704 on both legs — same seed, temp 0);
32K solo is +0.7 % on w4a8 (0.821 vs 0.815); n8 shows −2.3 %
(0.839 vs 0.859), within the n8 churn band. The per-token e4m3
activation quant does not materially move MTP acceptance at these
faces — consistent with the draft being lane-invariant and verify-side
activation perturbation being mild. There is no compensating quality
gain to trade against the speed loss — and informationally there
cannot be one: e4m3 activations are a subset of bf16's range on the
W4A16 route.

## The mechanism (what the r247 framing got wrong)

The r247 lever map claimed "activation quant is per-token (not
per-call group-32) so the p630 tax class does not apply". Measured:
per-token granularity softens the REDUCTION (a row-amax, not a
per-32-group amax) but the per-call activation-quant tax class very
much applies — every decode step pays the abs.amax → divide → clamp →
cast pass per linear layer ([M,1] f32 scale bookkeeping included), and
the (e4m3 × f4) joint dot runs slower than the W4A16
weight-decomposition route at our shapes. The two costs are not
separated by this A/B (both are removed together on the incumbent
lane); the disposition does not need the split — the total is
+76–80 % ITL at every solo face.

## Disposition: CLOSED-DOMINATED

r122's kbench prediction ("dominated by w4a16") is confirmed e2e on
the certified image at the target faces. The W4A8 lane joins §6's
closed channels: **no reopen trigger** short of a native xe2
W4A8 GEMM kernel class change (the vendor kernels are xe_3-only per
the xe2 verdict) — the lane's ceiling is the conversion dot we already
ride on W4A16, minus the activation-quant tax, so it cannot win on
this silicon generation.

**P-I phase complete**: all three cycles dispositioned —
r250 k4v8 (closed-dominated, byte-arithmetic), r251 group-64
(closed-deferred, band measured, K-side empty), r252 W4A8
(closed-dominated, e2e measured). No open improvement candidate
remains pre-cert29; the residual §5 items (per-channel E8M0, per-head
KV scales) stay open-low for the post-cert29 loop.

**NEXT RUNG: the cert29 cut** (r232 recipe on the cert28 base — the
P-H interop fix, the law-8 gate hardening, the serve-script parity,
P-H4's posture-aware P-D gate — then the full battery on cert29).

## Attempt 1 note (banked `attempt1-refused-fp8vl/`)

The first launch pointed at the fp8 VL checkpoint with
`ARC_QUANT=mxfp4` and refused at ModelConfig validation
("quantization method fp8 in the model config does not match the
`quantization` argument (mxfp4)") — the online-MX lanes run on the
BASE checkpoint with `ARC_QUANTCFG` per the r240 runner. Both banks'
tracebacks preserved; no rig time lost (refusal at ~50 s × 2).
