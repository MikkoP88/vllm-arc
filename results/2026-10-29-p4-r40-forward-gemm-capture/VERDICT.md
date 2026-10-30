# r40 — E-LAW forward-pass GEMM capture: ROOT CAUSE PINNED (block-fp8 linear routing), fix scoped to a stock-wheel op

Leg: P4 r40 (capture-only; zero serve boots, zero product change).
Instrument: `r40_gemm_microbench.py` — same silicon (B70 d0, ZE_AFFINITY_MASK=0),
both wheels, no serve. Six per-rank TP2 projections of qwen3.8-27b-fp8 (128x128
block scales), M sweep {1,2,4,5,8,16,64,128,896,2048} (5 = c1 verify k4+1, 1 =
draft replay, 16≈c4 verify 4x(4+1), 896/2048 = prefill class). Paths probed:
per-token input-quant alone; our production path (input-quant +
`_xpu_C.fp8_gemm` W8A8); stock `_xpu_C.fp8_gemm_w8a16` (fp16 activations,
block scales); champion ESIMD `esimd_gemm_fp8_blockscale` (D image only).
Outputs: rig `/root/build/r40/r40_ours.out` (60 rows + DONE) and
`r40_champ.out` (60 rows + DONE), pulled to `.tmp/r39read/`.

## 1. DEFINE (from r39, unchanged)
Solo 59.1-60.0 vs bar 79.74 (-26%); agg 195.6-209.2 vs 274.41 (-24..-28%);
prefill 1.27-1.39k vs 1.84k (-31%); our step 63-70 ms c1 / 83.4 c4 vs D
43/46; our forward ~49 ms (47% of 608 GB/s spec) vs D ~25-28 (72-90%).

## 2. CAPTURE (per-GEMM ms, eager, identical conditions both images)

M=5 (c1 verify pass), format quant | w8a8 | w8a16 (both wheels) | ESIMD (D):

| shape (N,K)      | in-quant | fp8_gemm W8A8 | fp8_gemm_w8a16 | ESIMD block |
|------------------|----------|---------------|----------------|-------------|
| gdn_qkvz 8192x5120  | 0.215 | 0.338 | 0.077 | 0.098 |
| gdn_out 5120x3072   | 0.216 | 0.331 | 0.064 | 0.037 |
| attn_qkvz 7168x5120 | 0.249 | 0.328 | 0.064 | 0.092 |
| mlp_gate_up 17408x5120 | 0.211 | 0.334 | 0.167 | 0.220 |
| mlp_down 5120x8704  | 0.213 | 0.328 | 0.081 | 0.121 |
| lm_head 76032x5120  | 0.211 | 0.801 | 0.680 | 0.973 |

- Our production path pays quant+GEMM = 0.43-0.55 ms per layer GEMM at decode
  Ms; `fp8_gemm_w8a16` = 0.064-0.167 ms. **3-8x per-GEMM at decode Ms.**
- W8A8 is a FIXED-COST path at small M: ~0.33 ms flat for M=1..128 on every
  layer shape regardless of weight bytes (31 MB gdn_out same 0.33 as 178 MB
  gate_up) => dispatch/multi-kernel fixed cost, not bandwidth. Effective BW at
  M=5: 48-268 GB/s (the r39 re-classed "302 GB/s achieved" class, now
  MEASURED per-op).
- w8a16 effective BW at M=5: 247-579 GB/s (gdn_qkvz 547, attn_qkvz 579,
  gate_up 536, down 549, lm_head 573) = exactly the ~537 GB/s D-implied
  achieved class. r39's re-attribution is confirmed at the op level.
- Numerics: w8a16 relerr vs fp32 dequant-block reference = 3.5e-4..5.5e-4 on
  every shape (fp16 activations, no input quant => strictly better than W8A8
  which adds activation-quant error). Block scales are accepted natively
  (oneDNN). (W8A8 probe relerr printed Infinity — bench artifact: our As was
  [M,1] per-token, production block path uses group As; timing class valid,
  numerics field void.)
- Draft replay M=1: ours 0.547 (0.214+0.333) vs w8a16 0.064-0.088 vs ESIMD
  0.038-0.084. Champion's only ESIMD wins are gdn_out/attn_qkvz at M<=2
  (1.4-1.8x vs w8a16) and lm_head M=1 is a wash (0.658 vs 0.668). ESIMD
  total step upside ~0.1 ms/step => SECOND-ORDER, deferrable.
- **torch 2.12-vs-2.13 suspect CLOSED**: champ image's own `fp8_gemm` W8A8 is
  the same slow class (gdn_out M=5: champ 0.346 vs ours 0.331; gdn_qkvz
  0.349 vs 0.338) and champ w8a16 is class-equal to ours (attn_qkvz M=5:
  champ 0.083 vs ours 0.064 — ours FASTER there). D does not have faster
  stock ops; it ROUTS differently.
- Prefill Ms: w8a16 2.2-2.4x W8A8 across the board (M=896: gate_up 1.154 vs
  2.454, down 0.620 vs 1.639; M=2048: lm_head 12.95 vs 25.94, gate_up 2.92
  vs 6.20). Same lever covers the prefill -31% face.

## 3. DECOMPOSE (~100% of the forward gap)
Layer inventory: 64 layers x ~4-5 GEMMs (GDN: qkvz/out/gate_up/down; attn:
qkvz+o_proj(same 5120x3072 shape)/gate_up/down) + 5 lm_head calls.
- Ours under graph (r27 eager-proxy): GEMM avg ~0.15 ms x 320 = 48 ms
  forward. Microbench shows WHY: fixed-cost W8A8 path + per-call input quant.
- D same silicon, same model: routes ALL block-fp8 linears to
  `fp8_gemm_w8a16(a16, B.t(), scale.t().contiguous(), None)` (oneDNN) with
  `apply_input_quant=False` ("optimized XPU paths consume the original fp16
  activation"), ESIMD only for M<=2 GEMV corners, thresholds
  `_XPU_BLOCK_QWEN36_27B_ONEDNN_MIN_M=4` on our five shapes (D dispatch
  table maps 1:1 to our per-rank TP2 projections).
- Projected: 48 ms GEMM share -> 12-20 ms at w8a16-class 0.064-0.17/GEMM =>
  forward 25-28 ms = D's 43-46 ms step minus tail/sample. Bars clear at our
  CURRENT acceptance (emitted 4.03 needs step <=~42; forward ~27 + tail
  15.2 + sample 2.5 = 44.7; c4 benefits the same way at M~20 flat region).

## 4. ROOT CAUSE (pinned, one dominant)
**Our stack routes every block-fp8 linear through per-token input-quant +
`_xpu_C.fp8_gemm` W8A8 — a fixed-~0.33-ms-eager path at decode Ms — while
the champion routes the same weights through `fp8_gemm_w8a16` with fp16
activations (oneDNN block-scale native), 3-8x faster per GEMM, skipping the
quant entirely. The op already exists in our stock 0.30 wheel with identical
signature (our wheel file `scaled_mm/xpu.py` XPUW8A16FP8LinearKernel calls
it); only the BLOCK kernel's dispatch and the input-quant step differ.**

## 5. FIX (r41, one change, then gates)
Reroute `XPUFp8BlockScaledMMKernel.apply_block_scaled_mm` (overlay in
`vllm_gaudi/xpu/`, src-override pattern) to
`torch.ops._xpu_C.fp8_gemm_w8a16(x_fp16, B.t_view, scale_kn.t(), bias)` with
`apply_input_quant=False` semantics; weight stays [N,K] (bench passed the
non-contiguous `.t()` view exactly as D calls it); stock path kept as
fallback. NO new binaries, NO ESIMD port in r41 (deferred as optional
micro-leg; D's own ESIMD is worth ~0.1 ms/step). Gates: one TIMERS-OFF
clean boot + full battery first-traffic + integrity 5/5; E4 strip after.

## 6. Corrections to banked legs
- r39 VERDICT said "AR share publish — r24's never-published promise":
  WRONG. r27 already published forward AR = 0-8 ms (<=~8%, central 2-5).
  This leg supersedes that phrasing; no re-edit of the banked file.
- r37/r38 D-cell miscomparisons (P2r3 vs P2r6-CKPT) already corrected in
  r39; no further action.

## 7. Rig end-state
No boots, no serve; read-only `docker run --rm --privileged` microbenches in
vllm-arc:v0.30.0-b70-p41 and vllm-iexperimental:v1.0.0 only. radiance-test
up (W14 HEALTH=200), arc-test absent. Overnight-safe.
