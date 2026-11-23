# r231c — fixed-term decomposition: e2e per-call model CLOSED (P6.47, 2026-11-23)

**VERDICT: PASS (analysis leg — no wheel change, no image). Two probes
on the certified build25 wheel close the e2e MX per-call model at
every record ctx within ~5%, re-attribute the r231a "fixed term", and
price the remaining levers. The record path's residual headroom is the
structural dequant-machinery bound; shallow-ctx split tightening is
real but off the record path.**

## Probe v1 (`tools/ar/r231c_fixedterm.py`, `fixedterm.log`)

Banked 96-block window payloads, reps=50, idle GPU, ns sweep at
sl ∈ {1, 64, 256, 1024, 2048}:

- **True per-call floor (sl=1): 51-108 µs** (ns1 51-73, ns32 65-104,
  auto 74-108) — launch + plan + fill + epilogue + recover is
  ~0.1 ms-class, NOT the ~0.4-0.5 ms the r231a model assigned to a
  "fixed term". Against ctrl flash's ~0.1 ms/layer the floor roughly
  cancels.
- **Single-split cliff**: `windowed_k_blocks < kMinBlocksForSplit`
  (=32 k-tiles = 512 tokens, tile_n=16) forces `is_single_split` →
  ns1 == ns32 == auto; sl=256 (16 k-tiles serial on one WG, ~30
  µs/k-tile) costs 377-510 µs while sl=1024 (splits engaged) costs
  153-176 µs — 4 serial pages cost ~3× 16 parallel pages.
- **L2-hot bias**: the ~5 MB window keeps reps L2-resident — 228 µs
  @2K hot vs the cold-pool 439 (v2) / splittime's ~510 — the e2e walk
  class is memory-cold.

## Probe v2 (`tools/ar/r231c_e2egeom.py`, `e2egeom.log`)

E2e geometry: thousands-of-blocks pool, window bytes tiled, walked
block ids stride-scattered (L2-defeating); `max_seqlen_k` at engine
capacity (bt width 4114 × 64 = 263,296 — the r212f frozen posture) vs
window capacity. auto/ns32/ns128 at sl ∈ {512, 2048, 8192, 32768},
µs/call (rows=4):

| sl | mxk8v4 auto | mxfp8 auto | mxfp4 auto | mxfp4 ns32 | mxfp4 ns128 |
|------|--------|-------|-------|-------|-------|
| 512 | 138 | 147 | 131 | 129 | 130 |
| 2048 | 439 | 448 | 428 | **202** | 427 |
| 8192 | 566 | 599 | 530 | 491 | 529 |
| 32768 | 1131 | 1229 | 950 | 1644 | 948 |

(auto ≈ ns128 at capacity pricing — the heuristic prices ~128 splits
from max_seqlen_k, confirmed.)

## The closed model

e2e MX gap per face ≈ 16 × [probe auto(ctx) − ctrl ~0.1 ms]:

- 8K face (r230 m4q gap 9.0 ms): 16 × (0.53 − 0.10) = 6.9 vs 9.0 —
  walked ctx at the 8192-face ≈ prompt+gen ≈ 8.2K-class; residual
  ~2 ms = engine-pool cold-miss class + measurement posture.
- 32K face (gap 15.5 ms): 16 × (0.95 − 0.10) = 13.6 vs 15.5 — same
  ~12% residual class. The SHAPE is closed at both ends; the r231a
  intercept attribution was wrong in kind, not in magnitude.

## Corrected attribution + priced levers

1. **The "fixed term" is split over-provisioning at shallow ctx**:
   capacity-priced auto (128 splits) costs 428-448 µs at 2K where
   ns32 costs 202-244 — a +92% cliff — and it evaporates by 8K (auto
   530 vs ns32 491, mxfp4). A device-side `splits_per_seq` (the
   plumbing exists: `p.splits_per_seq[idx_b]`) tightened from
   `seqused_k` would recover ~0.2 ms/call — **but only at ≤2-4K walked
   ctx, OFF the 8-32K record path** (r231a lever-2's "≤8K only"
   estimate confirmed and sharpened).
2. **Walk machinery is the record-path residual**: at 8K, ~0.44 of the
   0.53 ms is machinery (DRAM floor for MX bytes at 8K ≈ 22 µs — a
   ~20× overhead): per-page dequant ALU + V-staging barrier structure,
   structural on Xe2 (no native MX dot; r231a law). r231b proved the
   vehicle-risk class of naive hoists here. Remaining in-knob ideas
   (V-stage double-buffering to hide the per-page barrier pair,
   d-tile-batched dequant) are build26-class surgery with r231b-class
   risk and bounded ceiling (~half of 0.44 at best).
3. **Floor**: 65-130 µs/call (launch, Oaccum memset at split width,
   recover, epilogue) — ~2.1 ms e2e at full elimination (unreachable;
   tighter ns at shallow ctx trims the memset/reduce share).

## Conclusion for the line

The MX decode per-call model is closed end-to-end (r231a slopes +
r231b vehicle boundary + r231c floor/split attribution). At the
certified record ctx the remaining gap to ctrl flash is the Xe2
dequant-machinery bound: mxfp4 1.36× ctrl @32K, mxk8v4 1.45×, mxfp8
1.51× (r230) with MX capacity 2-4× better than fp8 (the point of the
lane). Further walk-side micro-optimization has priced, bounded,
high-risk headroom; the superiority record stands on cert27.

NEXT RUNG (owner's call per plan §12 gates): either (a) the V-stage
pipelining build26-class attempt (one knob, offline gates, r231b
discipline), or (b) declare the MX walk line converged at the
structural bound and move to the next plan phase.

## Hygiene

Bare-container probes on the image .so (nothing swapped, no build);
W14 quiesced per arc window and restored to HEALTH 200 after each
(`*_launch*.log` tails); rig bank `/root/build/p647/r231c/` carries
the raw logs + shipped probes. v1's graph-replay arm printed
GRAPH_UNAVAILABLE (torch.xpu graph API absent in this torch build) —
the host-vs-GPU split it targeted is meanwhile bounded by the v2
capacity-arm agreement with the graph-replayed e2e record (residual
~12%), so the arm was not pursued further.
