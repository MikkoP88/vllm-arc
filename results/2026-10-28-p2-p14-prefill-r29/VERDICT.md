# VERDICT — r29 p14 prefill-parity + P2 close-out numbers: TTFT gap DIAGNOSED as GPU
# content — GEMM tiles 64.5% + oneCCL AR 22.2% of a 97%-busy window; both pools map
# onto the in-scope C bake-chain reval lane (§12-14). TPOT gate ≤60 ms timers-off verdict:
# 63.4 c1 / 77.9 c8 — MISSED-with-proof (BW-wall law; gate number predates it). p14 stays
# OPEN as the §12-14 reval (no posture change this leg; NO MERGE).

Posture: product stack VERBATIM (mode=none + FULL_DECODE_ONLY [1..320], MTP k4,
p10 fastpath only via /root/r29cap, timers OFF, torch profiler armed
with_stack=false) — the gates/cert posture per the r25 distortion law. Boot
19:22→healthy 19:25; battery full; rig restored to W14 after teardown.

## 1. TTFT ladder (c1, cold, seeded random-id prompts, max_tokens=1, non-stream)

| ctx | r29 cold (ms) | bar (D champion) | delta | r18 (timers-era) |
|---|---|---|---|---|
| 8192 | 5469.3 | 3944 | **+38.7%** | 5794 (+47%) |
| 16384 | 11285.9 | 8083 | **+39.6%** | 11997 |
| 32768 | 24207.0 | 17447 | **+38.7%** | 25764 |
| 65536 | 54781.8 | 41565 | **+31.8%** | 58546 |

The gap is ~constant-fraction across ctx → a RATE difference (content), not fixed
overhead; slightly better than r18 (+47%→+39%: timers-off + p10; r18 had no p10).
Flag-diff half already done: our boot line is flag-identical to the D-champion serve
line except kv-transfer (P3 lane), async-scheduling (silently auto-enabled here,
r22a law), and compile mode (falsified ×3) → **engine-path (0.30 vs 0.26), no
prefill-relevant flag delta exists**.

## 2. 8k-prefill attribution (single 8192-token request, eager-visible — prefill is
eager on FULL_DECODE_ONLY; profile window = exactly this request + 2 decode steps)

Span wall 5555 ms; kernel content 5402 ms = **97% of wall** (GPU busy — launch/CPU
overhead is NOT the gap; 5270 kernels, urEnqueue CPU 286 ms fully hidden).

| category | ms | % | calls | notes |
|---|---|---|---|---|
| gemm-fp8 | 3481.9 | 64.5% | 623 | `_xpu_C::fp8_gemm` 520 @ **6.65 ms avg** (3.458 s) + aten::mm aux 103 @ 0.23 ms |
| collective | 1198.5 | 22.2% | 1334 | `c10d::allreduce_` **265 @ 4.505 ms** (ReduceCopy 2.165 + ring_write 2.162 per AR) + 267 barriers 50 ms + allgatherv 7 |
| fa-attn | 240.9 | 4.5% | 34 | varlen_fwd 34 @ 7.06 ms (16 full-attn layers × 2 forwards + 2) |
| gdn | 182.3 | 3.4% | 672 | ChunkFwdO 64.5 + launcher 60.7 + WU 33.5 + A 17.2 + InverseOpt 5.5 — GDN is NOT the prefill problem |
| quant | 109.8 | 2.0% | 520 | per_token_group fp8 |
| norm+act+copy+other | 188.7 | 3.5% | — | fused_add_rms_norm 76.9, act 51.4, memcpy D2D 36.9 |

TP0 ≈ TP1 (GEMM 3482 vs 3525 ms; AR 1198 vs 1151) — symmetric, no rank-skew story.

**Arithmetic**: 8k prefill FLOPs/rank = 2×27e9×8192/2 = 2.21e14 → GEMM pool runs at
**~63.5 TFLOPS/rank effective fp8**. 265 ARs = 2 forwards (target + draft prefill:
2×(128 layer ARs + misc)); each AR moves ~84 MB (8192×5120×2B) at ~4.5 ms →
**~19 GB/s effective** — both pools far from silicon limits.

**Cross-check vs the bar**: our GEMM+FA+GDN content alone (3905 ms) ≈ D's ENTIRE 8k
TTFT (3944 ms) → D's engine must run the same math materially faster (≈90 TFLOPS
GEMM tiles and/or cheaper AR). No D-side kernel trace exists in the v1235-certified
archive (client TTFTs only: 7950-tok text prompt → 4.399 s, consistent with the bar)
— so the split of D's 3944 between GEMM-rate and AR-path is not directly observable;
our own decomposition is the actionable evidence.

**Diagnosis**: the 1.5 s/8k delta lives in (1) fp8 GEMM tile rate (64.5% pool) and
(2) oneCCL large-AR staging (22.2% pool) — exactly the C bake-chain reval lane of
plan §12 item 14 (arstage / stall / wedge / v60 / v63 / **v64-TTFTFIX** — a named
TTFT fix in that chain). Recommended reval order: AR-staging first (single largest
non-GEMM pool, 1198 ms, algorithm-level not tile-level), GEMM-tile bakes second
(biggest pool but shape-dependent). Every port passes the scope-law check (line C
provenance) BEFORE implementation. In-scope, no new lanes opened.

## 3. TPOT gate cell, timers-off (§10: k1k-c8 ≤60 ms random-token acc≈1, AS MEASURED)

stepprobe c1 63.4–63.5 ms / c8 77.9 ms (c1-again 63.4; chunk-gap = step wall).
**MISS vs 60 ms bar — with proof**: the step is at the BW wall (r27: 320
layer-GEMMs ×0.15 ms ≈48 ms floor + spec structure; ~27 GB/rank weight reads
@~300 GB/s); TPOT moves only via acceptance length (P4) or step structure. The 60 ms
gate number predates the BW-wall law.
**Distortion-law amendment**: r25/r26/r28 stepprobes ran timers-ON (77.6/92.8);
today timers-OFF = 63.4/77.9 ≡ r22a's original 64.4/79.6 → the stepprobe face
carried ~+14 ms timer overhead — larger than the +3.8% bench-wall distortion.
Gate-face numbers must come from timers-off legs; r29 is now the reference cell.

## 4. Regression cells

- survival (832-tok flip payload as FIRST traffic): **200 / 200**, onset count 0 —
  defects a/b held (r17/r18 interlock line intact).
- acceptance gate: pos1 **0.8798** PASS hard ≥0.5 (acc 0.7115, mean len 3.85 —
  normal cross-leg band 0.877–0.912; target 0.85 miss recorded, P4 lane).
- p10 fastpath: installed both ranks (serve log echo); build term not re-measured
  this leg (timers off) — banked 0.49 ms (r25/r26/r28).

## Evidence

`legs_pull/`: ttft_ladder.txt, prefill_analyze.txt (both ranks; NB the "avg=u"
column is actually ms — cosmetic label bug), prefill_req.txt (5604.2 ms same-shape
client wall), prof_prefill/ (rank0/rank1 chrome traces 1.7 MB gz each +
profiler_out_0/1.txt torch aggregation + async_llm trace), spec_gate.{json,txt},
stepprobe.txt, onset_count.txt, serve_r29_final.log. `scripts/`: r29_boot.sh,
r29_serve.sh, r29_sitecustomize.py, r29_battery.sh, r29_teardown.sh,
r29_prefill_analyze.py. Rig restored to W14 (teardown verified).

Next: P2 close-out package (this leg supplies the last two numbers: TPOT
timers-off + TTFT ladder), then P3 per §12. p14 follow-up = §12-14 C-bake reval
(arstage/v64-TTFTFIX first), owner-gated scheduling.
