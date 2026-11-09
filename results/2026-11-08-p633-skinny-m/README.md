# P6.33 — skinny-M decode GEMM kernels (arc6 wheel): built, benchmarked, e2e-falsified, banked OFF

Date: 2026-11-08/09 · Branch: `feat/arc-p633-skinny-m` · Image lineage:
`vllm-arc:v0.30.0-b70-cert22` (= cert21 + arc6 wheel + skinny-M patch,
route default OFF) · Evidence root: `rig_raw/` (this directory).

## Verdict (read this first)

The arc6 wheel ships three **correct** (r126 GATE-1, rel < 2e-2 on every
gate cell across all shapes/Ms/lanes) skinny-M weight-only GEMM kernels
that **win the eager/host-bound benchmark regime** — and the e2e serve
battery **falsified the reroute**: under FULL_DECODE_ONLY XPU-graph
replay (the certified decode path) stock oneDNN executes its
tensor-core gemmstone kernels at **420–680 GB/s device-side**, and the
skinny scalar-FMA streaming design (ceiling ~300 GB/s) loses 1.6–2.6×
per call recorded into the replayed graph. Gemma solo decode regressed
−38% (m816) / −42% (g16); every over-cap path stayed exactly on ref.

The certified posture therefore ships the route **default OFF**
(`ARC_SKINNY_M` unset/off; `=on` opts in). Behaviorally cert22 ≡ cert21
at serve; the kernels and route remain in the tree, correct and
available — the eager-regime wins are real (host-bound decode lanes
outside graph replay), and the measurement-methodology findings below
are the item's transferable value.

## What was built

Wheel `0.1.14.1+arc6` (sha256
`484e06c18d0c8ea1cf6dc36dda24828e37b5c800688e9d4087d779cb6dfb7156`,
built incrementally in `vxk-arc3-builder` on the arc5 tree), sources
`external_patches/vllm-xpu-kernels/arc6_skinny_v017bp/csrc/xpu/skinny_gemm/`:

- `skinny_fp8_gemm_w8a16` — checkpoint-fp8 lane (per-tensor/per-channel
  fp32 scales), same signature as the stock champion op
- `skinny_fp8_gemm_w8a16_mxfp8` — online MXFP8 lane (group-32 e8m0)
- `skinny_fp4_gemm_w4a16` — online MXFP4 lane (packed e2m1 + e8m0)

Design: one work-item streams TN weight columns K-contiguously (128-bit
loads), dequantizes in registers (f8 bit-identity re-bias, e2m1 value
set, e8m0 exp2 group scales), accumulates fp32, fixed-order split-K
reduce (deterministic). Split law v2.7 (final): uniform group-divisible
slices, kc ≥ 1024, swizzle always-on (A=0x9E3779B1 over tiles), per-h
activation decode at M=1, work-item floor 32768, split cap 8.

Python route `vllm_gaudi/xpu/patches/skinny_m.py` (last in
`_XPU_PATCH_MODULES`, chains after the p630/p631 reroutes): wraps
`apply_weights` of the three kernel classes, detects lane by weight
dtype, routes within per-lane M caps (kernel-level evidence):
`w16 → M=1`, `m816/g16 → M ≤ 2`. Gate `ARC_SKINNY_M`: **unset/off =
stock routes (certified default after r127)**, `on` = opt-in.

## Kernel iteration (r125–r126, ten windows)

| ver | change | read |
|---|---|---|
| v1–v2.3 | streaming kernel + split-K + 128-bit loads + swizzle probe | correctness green; M=1 wins emerge |
| v2.4 | per-tensor scale `.item()` in resolve path | p50 tax (later understood — see v2.5.1) |
| v2.5 | bake: swizzle always-on, kc≥1024, 64-aligned kc, per-h a-decode | w16pc 6/6 at 0.80–0.92 (w8, per-iteration A/B) |
| v2.5.1 | `.item()` D2H sync removed — scale passes as device pointer, stride 0 broadcasts scalar | w8b round-interleaved caught 14.3/30.3 ms p90 tails on exactly w16pt; the p50 pipeline-stall tax had made the lane look 1.16–3.03× behind stock |
| v2.6 | floor-driven splits (640 stream floor), non-uniform last slice | w9: won gemma.up ~2% but regressed qwen.o 5–10% on all four lanes — falsified |
| v2.7 | v2.5 split law restored (qwen 5×1024, gemma 4×1344, downs 8×{1088,1344}) | w10: GATE-1 0/192, w16pc 6/6 mean 0.871, g16 6/6 mean 0.851, m816 4/6 mean 0.972 — certification candidate |

Round-interleaved kbench (`tools/ar/r126_kbench_ri.py`): every
(shape, lane, M) cell benched once per round × 30 rounds, one queue
sync per round — every cell median spans the same mode episodes of the
rig's ±30–40% bimodality; per-iteration pairing (v1 kbench) kills
clock drift inside a pair but not cell-order mode contamination
(window8 read w16pt 18% slower than w16pc on identical kernel work —
the tell that unraveled v2.4's `.item()` and v2.6's refine poison).

## e2e battery (r127, five legs, arc6test image v2.7 + per-lane caps)

Postures identical to the banked refs (cert21 legs A/B/C, gemma r124b
ml40k, gemma r120g). `ARC_SKINNY_M` unset (= on in the pre-falsification
build). Route proof per leg: `skinny-M wrap installed` ×12, the lane's
`skinny-M route FIRST <lane>` marker, zero `route FAILED`.

| leg | lane | solo tps (ref) | Δ | agg8 tps (ref) | spec |
|---|---|---|---|---|---|
| A qwen fp8 | w16 | 60.5 (60.10) | +0.7% | 302.1 (302.50) | 0.9258 ✓ |
| B qwen mxfp8 | m816 | 53.1 (54.30) | −2.2% | 267.2 (266.80) | 0.9804 ✓ |
| C qwen mxfp4 | g16 | 63.4 (61.80) | +2.6% | 303.8 (299.60) | 1.0000 ✓ |
| D gemma mxfp8 | m816 | **17.1 (27.7)** | **−38%** | 192.2 (192.4) | n/a |
| E gemma mxfp4 | g16 | **23.3 (40.5)** | **−42%** | 261.3 (262.0) | n/a |

TTFTs unchanged on every leg (D: 3.88/20.92 vs 3.87/20.94). Qwen legs
are inert by construction: with MTP k3 the target decode is always the
M=4·B verify batch (over every cap) and the MTP draft head is excluded
from quantization — legs B/C show only an M=2 capture-variant marker
that the 4·B-token verify regime never replays.

## Attribution chain (why D/E regressed)

1. **r127b qkv probe** — the r126 matrix benched up/down/o only; qkv
   (5376→8192/rank, 60 layers) and the real o shape benched standalone:
   qkv m816 M=2 1.19×, everything else ≤ 1.03× — and stock read a flat
   ~0.158 ms across 22–44 MB shapes: **the standalone loop is
   launch/host-bound, not compute-bound** (the tell).
2. **r127c EXP-1** — same image, `ARC_SKINNY_M=off`: solo **27.7 tps,
   exact match to the r124b ref** (agg 192.3 vs 192.4). Causality
   proven; the arc6 wheel stock path validated against the arc5-era
   reference. (EXP-2 eager boot OOMs during memory profiling — eager
   posture is not bootable at GMU 0.8, itself confirming the serve
   path is graphs-or-nothing.)
3. **r127e/r127f graph probes** — 1-call XPUGraph replay is ~0.25 ms
   overhead-dominated (useless); the 50-call amortized graph gives the
   device-side truth at M=2 m816:

   | shape | stock in-graph | skinny in-graph | sk/st | stock GB/s |
   |---|---|---|---|---|
   | gemma.up | 0.1122 ms | 0.1822 ms | 1.62× | 516 |
   | gemma.down | 0.1132 ms | 0.1807 ms | 1.60× | — |
   | gemma.qkv | 0.0852 ms | 0.1617 ms | 1.90× | 516 |
   | gemma.o | 0.0425 ms | 0.1089 ms | 2.56× | 680 |

   **Stock oneDNN under graph replay runs 420–680 GB/s** — the
   ~197 GB/s "stock level" that motivated the kernel was an
   eager-launch-bound artifact (host dispatch ~0.15–0.3 ms/call caps
   every op eagerly; the round-interleaved kbench compared host rates).
   The scalar-FMA streaming ceiling (~300 GB/s, r123 bytes-floor math)
   cannot beat tensor-core paths at any M — and the size-1 decode graph
   pads the token dim to 2 (legs D/E's only marker was M=2, fired
   during capture), so even M=1-capped routing executes as M=2 work
   inside the replayed graph.

## Posture

- `ARC_SKINNY_M` unset/off = stock routes — **certified default**;
  cert22 behaviorally ≡ cert21 at serve. `=on` opts in (eager-regime
  wins; graph-replay regression documented above).
- rig `serve_arc_b70.sh` gained the `ARC_SKINNY_M` pass-through
  (mirrors `ARC_MXFP8_MODE`) so the knob reaches the container.
- Certification leg (r127g2): gemma m816 default posture on the cert22
  content image (`vllm-arc:v0.30.0-b70-arc6test`, FORK_SHA e2457389)
  → solo **27.7** / agg8 **192.9**, the exact r124b reference values,
  with zero `skinny-M wrap installed` / `route FIRST` markers in the
  serve log (`rig_raw/r127g2_certleg.log`; the first r127g run caught
  a stale-flat-file ctx22 build — see the trap note in the plan's
  P6.33 execution log). The image's installed
  `vllm_gaudi/xpu/patches/skinny_m.py` was md5-verified
  (`61ff347a13b5b21bbf57244c7b1667aa` = repo HEAD, contains the
  default-off gate) before the leg.

## Open (future item, out of P6.33 scope)

A tensor-core (joint-matrix) skinny kernel family could contest the
in-graph 420–680 GB/s bar at M ≤ 2; the scalar streaming family is at
its architectural ceiling. The r127f 50-call amortized graph probe is
the instrument that any such item must gate on — eager-loop wins are
host-rate illusions on this stack.
