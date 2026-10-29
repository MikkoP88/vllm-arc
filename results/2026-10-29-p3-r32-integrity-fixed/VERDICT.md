# r32 — root cause #2 FIXED (exact-boundary snap law): ALL correctness gates GREEN, integrity 5/5

Date: 2026-10-29 (rig legs 05:18–06:07). Posture identical to r31 (p2 product
+ p40/p41/p41b/p41c + lmcint kit with r31 fork deltas), one delta: the p41c
snap half now banks ONLY exact chunk-end boundaries (`ncc % bs == 0`).
Serves: probe boots `/root/r31_serve_dbg.sh` (ARC_SNAP_PROBE=1), gate boots
`/root/r30_serve.sh` (timers-off, clean).

## Root cause #2 — FOUND (probe E4, `e4_probe_chain.txt`)

The E3 md5 chain (r31 close) showed warm resumes re-seeding byte-identical
garbage: pool roundtrip was byte-exact, but the CONTENT stored at B=8064 was
never a real state. E4's pre-orig/post-orig per-step md5s + the fused-kernel
decision inputs pinned it:

- Cold 8k prompts step **ncc=7168 (sidx=7) → ncc=8192 (sidx=9)** — there is
  NO ncc=8064 step. The final chunk `[7168,8192)` is 1024 tokens and crosses
  boundary 8064 MID-CHUNK.
- Mid-chunk crossing semantics (upstream 0.30, `mamba_utils.py`):
  `preprocess_mamba_align_fused_kernel` advances
  `new_state_idx = ceil(computed_after/bs)−1 = 9` and the pre-copy carries
  `bt[7] → bt[9]` DIRECTLY — **column 8 is skipped entirely**. The
  postprocess migration only fires when `aligned_new_computed ≥
  num_tokens_running_state` (chunk end aligned); here 8064 < 8192 → no
  migration. `bt[8]` is a stale/foreign page **no write ever touched**
  (pre==post md5 at that step; the copy guard `src_col == dst_col` = 9==9
  no-ops).
- p41c's old snap law ("deepest complete block j = ncc//bs−1") banked that
  garbage page as the "state-after-8064" → every warm resume re-seeded it
  (E3's `seed_bytes == cold blk8` was byte-perfect delivery of garbage).
- Downstream symptoms all explained: integrity 0/5, golden len=4096 drift
  under traffic, ladder L=7168 deterministic mismatch. Exact-ending faces
  passed because their boundary blocks WERE written.

**Fix** (p41c snap half, one guard): snap only when `ncc % bs == 0` — bank
B = ncc iff the previous chunk ended exactly on B. Uniform over prefill and
decode (decode crossings migrate via postprocess on the aligned running
state). The adapter's deepest-PRESENT m-lookup then books the joint hit at
the true deepest banked boundary (7168 for 8k prompts); the warm resume
recomputes `[7168,8192)` as the same 1024-token chunk shape the cold chain
ran → bit-identical numerics. Also verified from source + E4: bias-0
pre-copies are verbatim (conv window shift 0, SSM `bt[src+0]→bt[dst]`) —
the token_bias-rotation hypothesis is DEAD.

## Gates (CLEAN boot — no prior traffic, certified battery order)

| cell | result | verdict |
|---|---|---|
| survival | 200/200 (large-first 7.34 s) | **PASS** |
| spec pos1 | 0.872 (hard 0.5; target 0.85 miss recorded, P2 band) | **PASS** |
| golden vs P2 product ref | **5/5 zero drift incl. len=4096** | **PASS** |
| LMCache integrity 5×8k | **5/5 byte-identical re-hits** (warm 2.0–2.3 s) | **PASS** (was 0/5) |
| warm-hit 8k ≤ 750 ms | cold 5864.7 / warm 1047.1, sha-identical | FAIL → p43 lane |
| churn | re-hits 1040/1027 ms warm-class both ends; no pool pressure at 24×8k vs 100 GB | N/A (redesign pending) |
| run7n | agg 196.2/215.7, solo 59.1/59.1 | P2 band, no regression |

Probe-boot integrity (fix verification): 5/5 (27be8461…, 28f2a474…,
8fa239e9…, d9f60192…, 13827225…).

## The len=4096 golden face — context-sensitive knife-edge (documented, not a defect)

The 4096 face generates a degenerate greedy loop (` "` repeated, 48 chars) —
one near-tie flip rewrites the whole tail sha. Observations:

- CLEAN boot, gate order: **0825deba = P2 ref, 5/5** (posture parity proven).
- After heavier prior traffic (integrity runs before golden; or the
  mamba-snap-off boot): 99955622, deterministic within that context.
- The mamba half is exonerated (drifts with VLLM_ARC_LMC_MAMBA_SNAP=0).
- Integrity prompts [2]/[3] show the same law: their TRUE-cold shas flip
  across boots (37a00b32 vs 8fa239e9 for [2]) — within-boot re-hits always
  match. Earlier "cross-boot pool persistence" hypothesis DISPROVEN (no
  orphan processes; every value traced to a same-container run).

Disposition: the golden gate runs in the certified battery order where it is
5/5 vs the P2 ref. A P3-context ref snapshot is banked
(`battery/golden_ref_p3.json`, captured post-battery) for reference.
Cross-context sha flips on knife-edge faces are inherent to greedy decode of
random-id prompts and are NOT load-path corruption (integrity 5/5 is the
load-path truth).

## Warm-perf anatomy (p43 lane, mechanisms now named)

- Clean-battery warm 1047.1 ms: external joint hit at 7168 → recompute the
  final 1024-token chunk + seed retrieve/park/H2D (41 MB ×2 ranks). p43:
  final-boundary snap at request finish (kills the −1024 recompute — bank
  B = final ncc from the running block), async pinned-buffer snaps
  (event-synced at store; kills the cold-side +300 ms sync stalls),
  seed-load overlap.
- Store-lag sensitivity: an immediate re-hit after the cold (~0 gap, first
  polluted sequence) measured 4831 ms — the async store hadn't banked the
  boundary yet → shallow hit. The integrity cell's ~1 s sleep never sees
  this. p43 candidate: store-completion signal before lookup bookable.

## Rig state at close

arc-test STOPPED, radiance-test (W14) restored + health 200 (session log).
Never left on arc image overnight.

## Files

- `e4_probe_chain.txt` — full E4 probe chain (724 lines, both TP ranks) —
  the mid-chunk-crossing proof.
- `serve_e3.log` / `serve_e4.log` — preserved probe-boot serve logs.
- `battery/` — clean-boot battery artifacts + P3 golden ref snapshot.
