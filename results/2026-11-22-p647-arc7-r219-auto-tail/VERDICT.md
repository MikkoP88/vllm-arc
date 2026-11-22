# r219 — the tail REGRESSES (192/256 +7–13%: 128 is the optimum) and the auto re-price hit the wrong branch (page-64 ELSE); r219b fix staged

**Date**: 2026-11-22 · **Leg**: r219 (arc7 Phase-8 auto re-price + tail
probe) · **Stack**: the r219 wheel (md5 `fafc6d4dedda71ea1a6957f88d849856`
— multiplier 4→13 slots/head, block16 policy cap 256, kernel ceiling
256, overlay `splits=` telemetry) · **Run**: `r219_auto_tail.log`
08:10–08:31 UTC (three boots, faces c8192/c16384/c32768 n1)

## Verdict

**GATE_FAIL (auto cell) — but both questions the leg asked are answered,
cleanly and cheaply:**

### 1. The tail beyond 128 REGRESSES — the curve's asymptote is real

| cell | 8192 | 16384 | 32768 | tps @32K | chunks |
|---|---|---|---|---|---|
| ns128 (banked) | 48.5 | 51.2 | 56.8 | 66.7 | 52–57 |
| ns192 | 51.9 (+7.1%) | 55.7 (+8.8%) | 63.2 (+11.2%) | 59.9 | 52–56 |
| ns256 | 55.9 (+15.3%) | 58.8 (+14.8%) | 64.4 (+13.3%) | 58.8 | 52–56 |

Quality healthy on both (chunks 52–56, 0 errors — the r217 TORCH_CHECK
plus ceiling 256 make deep splits run FOR REAL, and real says: worse).
Mechanism: past 128 the reduce volume + scheduling overhead grows
faster than the walk parallelism pays. **ns128 is the optimum of the
split surface; the ceiling stays 256 as headroom only.**

### 2. Auto mis-picked 64 — my edit hit the wrong branch

`auto` @32K = 62.9 (+10.66% vs the posture) ≈ ns64's 62.2, and the new
`splits=` telemetry confirms `auto` was requested. Root cause: the
engine's page size is **64**, so `mx_get_num_splits` takes the **ELSE
branch** (`kv_tile=64, sg_per_wg=4`) whose `policy_split_cap = 64` I
did not touch (I edited the block-16 branch). The multiplier edit DID
land (reduce cap = 13×120/12 = 130) — arithmetic: splits = 341 (bw) →
130 (reduce) → **64 (policy ELSE)** = observed. The r219 wheel also
proves the ceiling-256 + telemetry edits live (ns256 ran real, the
`splits=` line prints).

Device facts pinned by this leg: W = num_wg_slots = 120 at page 64
(sg_per_wg 4), H_q = 12, wgs_per_split = 296 at the bucket.

### r219b fix (staged, build in flight at bank time)

ELSE-branch `policy_split_cap` 64→**128** (AT the validated optimum,
since 192/256 regress); block-16 branch aligned to 128 for consistency.
Auto arithmetic becomes 341 → 130 → **128**. r219b validates: `auto`
(expect ≈56.8 @32K within 2%) + `ns128r` repro on the new wheel.

### Housekeeping

The r217 wheel bytes (`9ded4fb6`) were overwritten by the r219 build
without a rotation backup (resume mode skips it) — the r217 tree state
is reconstructable (`.r217bak` file backups + the documented xe2.cpp
edit-inverse) and r217's banks pin the md5; superseded by r219b anyway.

**Reverted**: nothing yet — the r219 wheel is transitional; r219b's
wheel (both caps at 128) becomes the candidate posture wheel.
