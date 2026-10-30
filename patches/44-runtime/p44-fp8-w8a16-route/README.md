# p44 — block-fp8 linear reroute to `fp8_gemm_w8a16` (SUPERIORITY FIX)

## What

`vllm_gaudi/xpu/patches/fp8_block_w8a16.py` — reroute
`XPUFp8BlockScaledMMKernel.apply_block_scaled_mm` to the wheel's STOCK
`fp8_gemm_w8a16(x_fp16, B.t(), scale_kn.t(), bias)` with
`apply_input_quant=False` semantics (fp16/bf16 activations in, block
scales, oneDNN path). Stock W8A8 path stays as fallback; kill-switch
`VLLM_ARC_FP8_W8A16_DISABLE`. Zero new binaries — the op already ships in
our 0.30 wheel. Registered in-place (`_XPU_PATCH_MODULES` via platform.py;
the leg form used the cap sitecustomize hook — one source, two callers).

## Why (E-law chain, r39→r41)

- r39: PARD lane CLOSED (architecture-gated); D-gap re-attributed to
  FORWARD step time from the champion cell's own telemetry (acceptance
  exonerated — our emitted 4.03–4.30 beats champion 3.46).
- r40 (root cause, microbench on the six per-rank TP2 shapes, M∈{1..2048}):
  our production path = per-token input-quant 0.21–0.30 ms + `fp8_gemm`
  W8A8 = a FIXED ~0.33 ms at M≤128 on every shape (48–268 GB/s); stock
  w8a16 = 0.064–0.167 ms (247–579 GB/s = D's achieved ~537 class) —
  **3–8×/GEMM at decode Ms, 2.2–2.4× at prefill**; relerr 4–5e-4
  (numerics strictly better: no activation quant). Decomposition ~100% of
  the 48 ms GEMM share → 12–20 ms at w8a16 class → forward 25–28 ms.
- r41 (fix): one change, bars beaten — see Validation.

## Test

TIMERS-OFF clean boot + full first-traffic battery with D-certified tools
unchanged; golden drift = expected numerics change (re-banked at cert);
integrity 5/5 + determinism 3/3; `fp8 reroute ACTIVE` lines in the serve
log (×4 sites).

## Validation

- r41 (3e340999): solo **86.31–86.64** vs bar 79.74 (+8.3%; r37 baseline
  59.1 = +46%); agg#2 **300.60–308.07** vs 274.41 (+9.6–12.2%); TTFT t1
  8k/16k/32k **3729/7788/16630** all under parity bars; c1 step 63–70 →
  ~47–48 ms; pos1 0.8919 MET; integrity 5/5.
- r42 certified image: solo 85.0–85.2, agg#2 297.7–297.9, TTFT
  3445/7252/15552 (−10…−13% vs bars), pos1 0.9020 MET with decay profile
  improved, warm-hit 740 ms MET; golden re-banked (e02cbb07).
- r43 k5 on this forward: solo 86.39–87.13, agg#2 303.5–305.4; golden
  byte-identical to k4. r44 matrix: c1 ITL 46.8–52.2 ms ≤32k (the w8a16
  arithmetic band), TPOT 8.6–20.7 ms c1.
