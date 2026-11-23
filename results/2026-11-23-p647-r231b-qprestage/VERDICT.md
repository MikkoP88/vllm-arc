# r231b — wheel-side Q pre-stage: CLOSED NEGATIVE (P6.47, 2026-11-23)

**VERDICT: FAIL as a perf lever — correctness gates all PASS, timing
ladder a large regression. The K-invariant Q-operand redundancy in the
MX decode mainloop is real but overlap-hidden; hoisting it through SLM
costs far more than the redundancy ever did. Wheel stays build25.**

## The lever (from the r231a VERDICT next-rung)

The MX decode mainloop re-derived the K-INVARIANT Q A-operand at every
(k-tile, d-tile): `copy(copy_q, ...)` + `reorder(tQrQ, tSrQ)` × 4
d-tiles per 16-row k-tile = 8192 redundant copy+reorder pairs per
32K-token call; r212w priced a same-class reorder at ~595 µs of a 32K
walk. The patch (`tools/ar/r231b_apply.py`) staged each d-tile's
reordered fragment to SLM once per body() (SG0 writes, one barrier,
16 KB `q_stage`) and reloaded per use — bitwise-neutral by construction
(bf16→float widening exact; the bf16 ctor of the same value returns
identical bits).

## Gates (bare cert27 container, r231a pattern, W14 quiesced)

- ORACLE build25-vs-build27 bitwise: **PASS 12/12 BITWISE_EQUAL**
  (`oracle_cmp.txt`; 4 walk-fit payloads × r224a mxk8v4 / r229a mxfp8 /
  r229b mxfp4, banked bt-geometry replay, payload fmt wins; saved
  tensors in `outA/` `outB/`).
- R229 calibrated band cells: **12/12 PASS, 0 FAIL**
  (`r229_cells.log`; BAND_DMAX 2.0e-1 / BAND_MISM 16 / BAND_ULP 64 /
  BAND_TRITON 1e-2 / BAND_CROSS 4.0e-1 — observed NATIVE_VS_WALK
  dmax ≤ 1.5e-2, mism ≤ 1/3072, triton bit-exact).
- LADDER (r231a-identical inputs, first-file-per-bank): ran clean,
  **REGRESSED** — see table. `R231B_DONE 20:02:29`, W14 restored 200.

## Ladder: auto column ms/call, build25 (r231a) → build27 (r231b)

| ctx | mxk8v4 | mxfp8 | mxfp4 |
|------|--------|-------|-------|
| 2K | 0.517 → 0.646 (+25%) | 0.510 → 0.625 (+23%) | 0.552 → 0.537 (−3%) |
| 8K | 0.667 → 0.991 (+49%) | 0.698 → 1.071 (+53%) | 0.643 → 0.821 (+28%) |
| 32K | 1.213 → 2.578 (+113%) | 1.285 → 2.694 (+110%) | 1.011 → 1.731 (+71%) |
| 128K | 3.494 → 8.829 (+153%) | 3.794 → 9.304 (+145%) | 2.723 → 5.496 (+102%) |
| 262K | 6.616 → 17.165 (+159%) | 7.347 → 18.076 (+146%) | 5.064 → 10.484 (+107%) |

Full forced-split table: `r231b_ladder.txt`. Side signatures:
`best_ns` collapsed ns128 → ns32 everywhere (≥32K); `recover_ms` rose
from 0.005–0.02 to a flat ~0.3 across all cells including ns1; ns32
columns moved ≤ +4% (dice band) while ns64/ns128 exploded — the
regression compounds with WORKGROUP COUNT, not just ctx.

## Root cause

Two compounding vehicle costs, both proportional to (k-tiles × 4):

1. **Scalar SLM reload storm.** The in-loop reload issues 32 scalar
   4-byte SLM loads per thread per (k-tile, d-tile). At 262K: 16384
   k-tiles × 4 d-tiles = 65536 sites → ~2.1M scalar SLM loads per
   thread (~268M per 128-thread WG). What it replaced — a vectorized,
   L1-hot copy (Q is 4 KB, resident after first touch) plus a register
   reorder — was largely hidden behind DPAS/memory latency in build25.
   The redundancy was real but FREE in the pipeline; the replacement
   traffic is not.
2. **SLM capacity / occupancy.** `q_stage` added 16 KB to
   SharedStorage (v_stage is 2–4 KB) — a 4–8× per-WG SLM growth. High
   split counts (ns64/ns128, and auto at ≥32K) schedule many WGs per
   sequence; the occupancy loss compounds there first, exactly where
   the table breaks (ns32 ≈ par, ns128 +160%).

The r212w 595 µs pricing does not transfer: that priced the K-operand
reorder on the per-step B path (full-width, every walk step), not the
A-operand re-derivation the MMA pipeline already hides.

## Consequence for the MX ITL line

The walk-side inner-loop ALU lane is measured NOT the lever the r231a
model hoped: even a zero-cost Q path would trim at most the (small)
exposed fraction of copy+reorder. The e2e MX gap decomposition (r231a)
stands: at engine ctx the split is slope ~0.0173 ms/K × ctx (m4q) vs a
FIXED ~6.5 ms (16 layers × ~0.4 ms/call constant). The fixed term is
now the dominant unattacked share (e.g. @32K: 15.5 ms gap = 8.9 slope
+ 6.5 fixed).

**NEXT RUNG: r231c = fixed-term decomposition of the shallow-ctx call**
(2K auto = 0.51–0.65 ms/call total; ~0.4 ms of it constant): launch +
host plan + recover kernel + V-staging setup, probes 0/2/3/6/8 per the
telemetry law, one knob at a time. Walk-side ALU lane closed unless a
register-vehicle variant (no SLM, unrolled fragment array) is cheap to
try opportunistically — expected value low, overlap-hidden.

## Build + artifact identity (build27 — NOT shipped)

Incremental build27 (VXK_ARC7MX_CLEAN=0, 43 min, rc=0; wheel
363,673,555 B; pre-patch wheel preserved `.r231bprebak`). Patched tree
md5 bf3402ec57a3eda9abd1f0d57eccf200 → RESTORED pristine
a02286184bdb07a088e0321a22bb0111 after the verdict (verified, 0
markers). build27 artifacts (rig-only, 100 MB law — never tracked):
`libattn_kernels_xe_2.so` md5 cff66d624b85ed4bc49bba57ff79873d and
`_vllm_fa2_C.abi3.so` 6e6c48508d40cb4c5007f27f79df97cf at
`/root/build/p647/r231b/`; build25/cert27 image pin
2d73ff7a3c2aa2caab86179288ca3711. **cert27 remains the certified
image; nothing in this leg ships.**

## Missteps (both harness-side, both now laws)

1. **r231b-1 misextraction**: the first extraction filtered for
   `libmqa_logits` (inherited from a sibling leg's refuses line) →
   md5-identical to build25 while the real patched artifact sat in the
   wheel. `nm -D` proof: the MX kernels live in
   `libattn_kernels_xe_2.so` (37 mx_paged_decode dynsyms; libmqa_logits
   carries ZERO), reached via the `_vllm_fa2_C.abi3.so` stub. Fixed
   `tools/ar/r231b_extract.py` extracts both; the libmqa pin in older
   refuses lines was vacuous.
2. **r231b-2 walk-fit**: run 1's r229 cells died 6-of-12 pre-verdict
   (`IndexError: index 128 ... size 96` in the fp64 walk): bank idx
   1–17 payloads carry sl≈8198 over a 96-block banked window (6144-token
   capacity); idx 18+ fit (sl≈5462–5473). Probe `tools/ar/
   r231b_pickprobe.py`; cells re-picked IDX="18 24 36 48". The abort
   also exposed a trap-order bug: the W14-restore guard matched the
   script's OWN container before its rm (restore skipped, rig left
   down) — fixed rm-first in `tools/ar/r231b_offline.sh`; W14 manually
   restored to 200 between runs, and the clean run's trap restored it
   again (log tail).

## Rig state at close

W14 radiance-test HEALTH 200 (`r231b_offline_launch2.log` tail);
builder tree pristine; build27 artifacts preserved under
`/root/build/p647/r231b/` for any future register-vehicle A/B.
