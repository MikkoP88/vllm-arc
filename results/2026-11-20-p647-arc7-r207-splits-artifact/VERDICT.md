# r207 — extended splits sweep: the s>=24 "cliff" is a failed-launch artifact

**VERDICT: GATED NEGATIVE (measurement invalid) — forced num_splits_kv
> 16 never launched a kernel. The apparent 16 -> 24 collapse (1.67 ->
0.25 ms, all formats, both contexts) was host overhead plus a
false-passing canary. Two durable laws extracted; the r206 inferences
"B ~ 0.57 ms" and "auto effective ~24-26 splits" are retracted.**

## What happened

`p200h_splitbench.py` was extended to s in {24, 32, 48, 64} to test
whether the block-16 policy cap clamps forced splits or real headroom
exists. The results looked like a miracle: at s >= 24 all three formats
at BOTH 8K and 32K converged to an identical ~0.255 ms "floor" with the
corruption canary still passing — apparently below every v2 total.

None of it was real:

1. `mx_paged_decode_kernel.hpp` capped `max_num_kv_splits` at
   `SGPerWG * sg_size` = 16 at page 16. `can_implement` returns false
   for anything larger; `run()` prints "Invalid Problem Size" and
   returns `kErrorInvalidProblem`, which the torch op swallows —
   **no kernel is launched and no error reaches python**.
2. The bench's `out` is `torch.empty_like(q)`; the caching allocator
   hands back the same block still holding the PREVIOUS cell's correct
   output, so the canary (compare vs the splits=1 reference) measured
   stale-but-correct bytes and passed at the usual one-bf16-ULP.
3. The timed 0.25 ms per call is pure host-side dispatch (arg marshal +
   allocations + the refused launch) — hence format- and
   context-independence.

Evidence: `grep -c 'Invalid Problem Size'` on the bench log = **576** =
24 calls/cell (20 timed + 3 warm + 1 canary) x 4 illegal splits x 6
cells (3 formats x 2 contexts) — every s >= 24 cell, nothing else.
Wheel under test: arc7mx-r204 md5 `2fb3b721` (driver log). Full log on
the rig at `/root/build/p199/logs/r206_splitbench.log`; the invalid
JSON is banked as `p207_extended_sweep_invalid.json` (every s >= 24 row
in it is void).

## Law 1 — silent-failed-launch stale-buffer canary

A kernel wrapper that returns its output tensor while internally
swallowing a failure Status makes `torch.empty_like`-style outputs
FALSE-PASS any correctness canary: the allocator recycles the previous
correct bytes. Detection: write the canary call into a **NaN-poisoned
preallocated `out`** — a non-launching cell surfaces as CANARY_NAN.
(Implemented in `p200h_splitbench.py` from r207 on; the C++ side should
eventually TORCH_CHECK the run() Status instead of printing.)

## Law 2 — num_splits_kv is a cap, not a request

`build_decode_split_plan` treats `num_kv_splits` as a per-seq cap; the
per-seq split count comes from `target_tiles_per_wg` (2x Xe-core
oversubscription). The r206 "uniform forced splits" table was therefore
a PLAN-splits ramp ([1,1] -> [7,8] -> [7,14]), never uniform splits, and
its A/s+B fit was a coincidence over that ramp. Any true splits sweep
must scale the plan target with the cap (ARC_MXKV_WGTARGET knob, r207)
and log the emitted plan per cell.

## Standing after the artifact

- All s <= 16 cells of the r206 table remain valid (they launched).
- Auto (mode0) rides the C++ heuristic: `mx_get_num_splits` -> 16 at
  block 16 (policy cap) -> 1.278 / 1.037 / 0.971 ms at 32K.
- No measurement exists past 16 splits. The r207 wheel (build15)
  raises `max_num_kv_splits` to 64 (split coord derives from the WG-id
  decomposition, not the thread layout — the SG-count bound was sanity,
  not structure) and the block-16 policy cap to 64; the re-sweep with
  the NaN canary + WGTARGET=64 measures the true curve.
