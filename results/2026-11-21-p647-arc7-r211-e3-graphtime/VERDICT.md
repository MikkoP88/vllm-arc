# r211 E2'' — graph-replay device-only times (1K fair comparison)

**Date**: 2026-11-21 · **Leg**: r211_e3 (arc7 Phase-8, E2'' of the
r211 ladder) · **Wheel**: arc7mx r210 keeper `7247e83b` · **Image**:
`vllm-arc:v0.30.0-b70-cert25zc` · **Tool**:
`tools/ar/p147_mxkv/p204_graphtime.py` + `r211_e3.sh` (torch face:
`torch.xpu.XPUGraph`; CUDAGraph is absent — first launch exit-3'd on
the wrong name probe)

## Verdict

**PASS with a story flip — the native 1K deficit is REAL KERNEL
LATENCY, 5-7× v2 device-side, and splits do NOT move it.** v2's
graph-replay device time at 1K is 45.5-50.3 µs (fp4/fp8) — its ~218 µs
harness number was nearly all launch host. The E2' conclusion "fp4 at
parity" compared against v2's host number and is DEAD. Native
device-only at 1K: mxfp8 328-338, mxfp4 234-244, mxk8v4 219-229 µs —
5.2-6.8× v2 (fp8/fp4) and 2.6× (mxk8v4, whose v2 lane pays 86 µs for
the K8 dequant). The forced-splits ladder s5→s32 is FLAT within 2%:
the cost is a FIXED per-call term, not the per-WG tile-walk chain —
the r210 s=5 "optimum" was an artifact of the host-bound harness.
Capture smoke: ALL formats captured and replayed clean (canaries
0 / 9.77e-4 one-ULP) — the native op is graph-capturable in the V3
shape; the engine lane's XPU-graph posture is safe.

## Results (device-only µs/call; graph = 32 captured calls × 50 replays)

| format | v2 | native auto | native s5 | s10 | s16 | s32 |
|---|---|---|---|---|---|---|
| mxfp8 | 50.3 | 327.9 | 330.8 | 336.9 | 338.0 | 337.9 |
| mxfp4 | 45.5 | 233.6 | 236.2 | 242.5 | 243.6 | 243.6 |
| mxk8v4 | 86.1 | 219.4 | 222.2 | 228.3 | 229.4 | 229.4 |

Canary: native_auto 0 (self-identical); forced splits 9.77e-4 vs the
auto reference (reduction-order one-ULP class, r210 law); v2 0 vs its
own eager ref.

## Findings

1. **Fair 1K comparison**: native/v2 device-only = 6.5× (mxfp8),
   5.1× (mxfp4), 2.5× (mxk8v4). At 8K/32K the native lane is at
   parity-or-better, so this is a short-ctx-only latency deficit.
2. **Splits flat (≤2% s5→s32)** — per-WG chain length is NOT the
   cost driver; adding WGs (12→~60) changes nothing. The ~220-330 µs
   is a fixed per-call term: kernel startup (SLM alloc, barrier
   tree, work-list read), the serialized main→reduce pair on the
   captured stream, or per-kernel teardown gaps. The E4-lite leg
   (graph floors per probe mode) locates it.
3. **auto ≈ s5 within 1%** — the r210 heuristic lands on the same
   device-time plateau at 1K; no posture change available from the
   plan side.
4. **The engine's 1K decode step under graphs = pure device time**:
   native 220-330 µs/layer vs v2 46-50 — the honest engine gap to
   close is this kernel-side latency, NOT the host path (graphs
   already amortize it; the bridge's device-op shape is capture-law
   clean).
5. mxk8v4's v2 = 86 µs (vs fp8-v2's 50) shows v2 pays a real price
   for the mixed-format dequant — the native mxk8v4 deficit is only
   2.5×, and its absolute device time (219) is the LOWEST native
   cell.

## Next (E4-lite in flight; r212 design)

- **E4-lite** (r211_e4): graph floors per probe mode (0/3/7/8) at
  1K auto — m3 (skip both stages) device time prices the fixed
  startup; m7/m8 price the compute/gemm skeletons. If m3 ≈ 300 µs
  the mainloop is nearly irrelevant at 1K and the fix is structural
  (merge reduce into main / drop the staging protocol at short ctx);
  if m3 ≈ 20 µs the walk is the target.
- **r212 candidates** (kernel-side, E-lite-gated): single-kernel
  decode at short ctx (fold the reduce; skip SLM staging protocol —
  a direct-from-GMEM register path, the v2 shape inside CUTLASS),
  barrier-count reduction per tile, or launch-shape change (one WG
  per (seq, head-pair) with in-register accumulation).

## Artifacts

`p204_graphtime.json`, `r211_e3.log` (rig: `/root/build/p199/out`,
`/root/build/p199/logs`).
