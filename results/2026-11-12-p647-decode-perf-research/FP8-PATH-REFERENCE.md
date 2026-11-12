# FP8 KV reference-path analysis — why the baseline is fast and always clean

**Date:** 2026-11-12 · **Leg:** decode-perf research (A10 line) · **Owner directive:** "do
deep analyze end-to-end --kv-cache-dtype FP8 path why it has so good performance and is
always corruption free even this hardware do not support native FP8."

This document maps the ctrl lane's fp8_e4m3 KV path end-to-end on cert25x-class images
and derives the property list the MX lane must match. It is the reference side of the
A10 telemetry attribution: every MX-vs-ctrl delta should trace to a property the fp8
path has and the MX path lacks — or the delta is not structural and must be re-measured.

## 1. What actually serves the ctrl lane

Banked serve logs (cert24/cert25 ctrl legs, qwen champion chain):

```text
[flash_attn.py:1116] Using FlashAttention version 2
[xpu.py:199] Using Flash Attention backend.
```

No TQ, no MX routing — the stock vllm_gaudi FlashAttention v2 XPU backend with
`kv_cache_dtype=fp8_e4m3` admitted natively (`supported_kv_cache_dtypes` includes
`fp8`/`fp8_e4m3`, gated by `flash_attn_supports_kv_cache_dtype`).

## 2. The end-to-end path

| Stage | What runs | Kernel count per layer-step |
|---|---|---|
| Prefill/verify store | stock `set_kv_buffer` elementwise e4m3 cast (+ per-tensor scale) | 1 small cast kernel |
| Decode attention | ONE fused FA v2 varlen kernel: paged walk (page_size = block_size), fp8 load, counter-operand convert, dot, online softmax, split-combine via in-kernel scheduler metadata | 1 kernel |
| MTP verify (q_len=k+1) | the SAME single kernel — extra rows are just more cu_seqlens entries; causal masking is positional | 1 kernel |
| Host logic | none per layer beyond the varlen call | 0 |

`get_scheduler_metadata(..., qkv_dtype=fp8_dtype, num_splits=max_num_splits)` — the
split-KV combine that MX implements as a separate second kernel is, in FA, scheduled
INSIDE the single launch (persistent split work distribution + in-kernel reduce).

## 3. Why it is fast on hardware without native FP8

Xe2/B70 has no native FP8 dot ([[xe2-xmx-fp8-nativeity-verdict]]: counter-operand
conversion only). The fp8 lane does not need one:

1. **Decode is bandwidth-bound.** The win is bytes: 256 B/slot/head-D256-rank vs 512 B
   fp16 — the ALU cost of converting each loaded fp8 to fp16 for the XMAD dot is
   hidden behind the memory pipe.
2. **Conversion is fused into the attention kernel.** No dequant pass, no extra
   tensors, no intermediate fp16 KV materialization.
3. **One launch per layer per step.** At agg8 × ~17 attn layers the ctrl path issues
   ~17 decode kernels per step; per-launch device dispatch (~µs each) is invisible.

## 4. Why it is always corruption-free

1. **Uniform layout.** An e4m3 slot is D bytes, no scale interleaving, no group
   deinterleave arithmetic to get wrong. The MX analogue ([K][Ks][V][Vs] with 1B
   E8M0 per 32 elements) adds two classes of addressable mistakes (scale-lane
   indexing, group-boundary handling) the fp8 lane simply does not have.
2. **Positional causal masking.** Flash's varlen causal handling derives each row's
   limit from cu_seqlens positions. The per-row-limit class of bug (our LIMIT LAW:
   MTP allocator-extent seq_lens leaking the next draft's K/V) is structurally
   impossible in the flash shape.
3. **Uniform per-layer geometry.** One dtype cast, one head size per spec — the
   hybrid two-head-size pools (gemma: 50×256-D + 10×512-D per rank) exercise nothing
   MX-specific in the store path.
4. **Upstream maturity.** The kernel and the fp8 KV plumbing are mainline vllm code
   paths with fleet-scale coverage; our MX lane is 100% first-party code.

## 5. The MX gap table (pre-fix → post-fix)

| Property | fp8 ctrl | MX pre-cert25y | MX post-batched-verify |
|---|---|---|---|
| Decode kernels per layer-step | 1 | ~1 (kernel math fine) | 1 |
| Verify-batch kernels per layer-step | 1 | ~16/req (per-row mx_decode + per-req inversion cascade) | 1 batched (+~8 once per step, layer-cached) |
| Per-step host kernels at agg8 ×17 layers | ~17 | ~2,000+ | ~80 |
| Store per layer-step | 1 cast kernel | 1 mx_store (quantize+pack) — comparable | unchanged |
| Split-combine | in-kernel | separate tiny combine kernel | unchanged (watch in A10) |
| Per-row limit derivation | structural (positional) | slot→position inversion law | same law, batched once per step |
| Bytes per slot (D=256) | 256 (fp8) | 528 (mxfp8) / 272 (mxfp4) / 400 (mxk8v4) | unchanged — the format law (VERDICT.md §1) |

**Reading:** the offline kernel batteries (A4-A8) showed the v3 kernel itself at or
below flash-class timings at equal shape; the e2e deficit was the engine structure
around it — overwhelmingly the MTP verify dispatch. `_split_verify_decode`
(engine.py, cert25y) collapses the per-step kernel count to the fp8 lane's order.
The remaining structural deltas (separate combine kernel, separate store
quantize) are the A10 telemetry's next attribution targets.

## 6. Corruption-freedom obligations for the MX lane

From §4, the MX lane's equivalent guarantees are: the LIMIT LAW per-row limits
(slot-derived, never allocator-extent seq_lens), the KV-store separate-strides law
([[kv-store-separate-strides-law]]), the two-head-size pool audit (gemma KV_2LTD —
per-group block tables, page 64 vs 128), and the end-to-end corruption battery on
every dtype × both models (r161 L1-L6). The m8g disposition (native-g0 longprobe,
r164) applies §4.3/§4.4 as the control.

## 7. A10 measured: fp8 vs MXFP8/MXFP4 end-to-end on cert25y (2026-11-12)

The rebake + batched-verify + native-q image (fork_sha d97d4a89), p139_conc
streaming, medians across slots (REPORT.md in legA10/):

| rung | conc | q8 ITL p50 | m8 ITL p50 | m4 ITL p50 | q8 TTFT | m8 TTFT | m4 TTFT |
|---|---|---|---|---|---|---|---|
| 1K | 1 | 39.1 ms | 42.2 (+8%) | 41.5 (+6%) | 0.29 s | 0.29 s | 0.39 s |
| 1K | 8 | 55.9 | 71.0 (+27%) | 71.7 (+28%) | 2.01 | 2.07 | 1.48 |
| 8K | 1 | 40.4 | 64.1 (+59%) | 58.2 (+44%) | 3.03 | 2.98 | 2.97 |
| 32K | 1 | 42.5 | 137.2 (3.2x) | 112.0 (2.6x) | 15.59 | 17.42 | 16.99 |
| 32K | 8 | 77.1 | 830.4 (10.8x) | 3714.0 (48x) | 71.47 | 67.24 | 58.45 |

Per-layer attn delta (conc=1, n_attn=16): m8 +0.19 ms @1K, +1.48 @8K,
+5.91 @32K; m4 +0.15 / +1.12 / +4.34.

### The three structural differences that matter, measured

1. **Short ctx: parity achieved.** +0.15-0.19 ms/layer @1K is inside the
   ctrl kernel's own per-layer cost — the engine structure (verify
   batching, kernel count, casts) no longer differentiates the lanes.
   The residual is the two-launch structure + scalar dequant.
2. **ctx scaling = the split cap.** s_eff = min(32, ...) freezes the
   split count: at 8K each program serially walks 8 tiles (256 tokens),
   at 32K 32 tiles (1024). The per-layer delta tracks that serial chain
   (0.19 -> 1.48 -> 5.91 ms). ctrl's FA kernel splits IN-KERNEL with
   dynamic work redistribution, so its cost stays ~flat (39 -> 42.5 ms
   across the whole rung). Model check at 32K/n8: R=32@2K batched 4.5 ms
   x 16 walk ratio ~ 72 ms/layer predicted, ~52 measured (m8) — the 32K
   cliff is kernel-walk-dominated, occupancy clawing some back.
3. **The occupancy cliff inverts the byte law (m4 @32K/n8 = 3714 ms).**
   At R=32 rows x 24 heads x 32 splits = 24,576 programs (8 warps each)
   the device is compute-oversubscribed and the binding resource flips
   from bytes to per-element ALU: e2m1 nibble dequant costs ~2-3x fp8's
   convert, so mxfp4 (half the bytes) is 4.5x WORSE than mxfp8 at the
   extreme cell — the offline batteries (B<=8) never reach this
   oversubscription, which is why A9 said mxfp4 fastest. MXFP4 pool is
   1.8x ctrl (722K tok) — its TTFT actually leads at 32K/n8 (58.4 vs
   71.5 s) — the deficit is decode-only.

### Why ctrl is immune to all three

ONE fused kernel per layer-step: GQA-packed (each KV byte is read once
per kv-head program and dotted against all 24 q-heads x 4 verify rows),
fp8->fp16 counter-operand convert hidden in the XMX MMA pipe (the "no
native fp8" penalty vanishes in a bandwidth-bound kernel), split-combine
scheduled in-kernel, zero host-side per-layer work. The MX lane's
per-q-head grid re-walks and re-dequants the same KV up to 24x (6 q-heads
x 4 verify rows) in scalar ALU.

### Hidden axis: MTP acceptance

ITL ratio @1K/n1 is 1.08 but per-user tps is 76.3 vs 60.9: chunk-token
yield (600 tok / 194 chunks = 3.09/chunk vs ctrl ~3.6+) — quant noise
costs a little draft/verify agreement. Acceptance metrics are banked per
leg in serve logs (SpecDecoding lines); r161 carries them.

### A11 levers, ranked by expected close

1. `_MX_SPLIT_MAX` 64/128 (read at launch; mid-cache keyed on it) —
   cuts the serial chain 2-4x at >=8K; targets axis 2 directly.
2. Verify-row walk sharing (4 rows per program): the 4 rows' limits
   differ by <=3 tokens (only the last tile masks), state is HALF of
   v3's spilled N_REP=6 unroll — 4x fewer walks+dequants without the
   register cliff that killed GQA sharing at D=256.
3. XMX tl.dot reopening: A6's pointer-chase premise is refuted at true
   shape (v3 loses; m4's byte-law inversion = ALU-bound) — a
   tl.dot((R_TILE*N_REP, D) @ (D, N)) score path converts the hot ALU
   to MMA, the same trick ctrl uses.
4. Combine-kernel fusion: minor (per A10 the two-launch cost is inside
   the @1K residual only).
