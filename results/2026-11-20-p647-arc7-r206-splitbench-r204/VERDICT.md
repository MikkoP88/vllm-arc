# r206 — splits re-sweep on the converged r204 wheel (zero-build)

**VERDICT: MEASURED (no code change) — the r200h split-overhead law is
STALE on the converged wheel: B collapsed 1.7 -> ~0.57 ms, and B alone
now sits at/below v2's totals. Remaining splits headroom is real but
modest (~0.15-0.2 ms to s=64); the walk-base lever is the reduce/tail
slice inside B.**

- Wheel under test: `arc7mx-r204` md5 `2fb3b721` (verified in the
  driver log before the run).
- Tool: `tools/ar/p147_mxkv/p200h_splitbench.py` (unchanged),
  driver `tools/ar/p147_mxkv/r206_splitbench_r204.sh` (r200c
  quiesce/restore pattern). W14 auto-restored (fullres) on exit.

## 32K device ms (uniform forced splits; canary <= 2.4e-4 everywhere)

| s | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| 1 | 19.884 | 14.062 | 13.214 |
| 2 | 10.123 | 7.231 | 6.836 |
| 4 | 5.168 | 3.710 | 3.534 |
| 8 | 2.712 | 1.971 | 1.904 |
| 16 | 1.651 | 1.270 | 1.230 |

Fits t(s) = A/s + B (consecutive-doubling, consistent across pairs at
high s): **A ~ 17.0 / 11.2 / 10.7, B ~ 0.59 / 0.57 / 0.56**. mode0
(1.278 / 1.037 / 0.971) implies auto's EFFECTIVE parallelism is ~24-26
splits — below the block-64 policy cap of 64, so either the heuristic
requests fewer (splits_bw = kv_tiles/12 = 42 at 512 tiles; reduce cap
= 4*num_wg_slots/Hq) or the effective parallelism saturates in tail
waves / reduce serialization. Projections: t(32) ~ 1.12/0.92/0.89,
t(64) ~ 0.86/0.75/0.73.

## 8K: uniform-forced LOSES to auto

mxfp8 8K: s=16 forced = 0.606 device vs telemetry auto ~0.40 — the
per-seq auto assignment beats uniform splits at mixed/short lengths
(r200h already saw this). Any heuristic change must preserve the
per-seq path.

## Laws / next

1. B ~ 0.57 ms is the current splits-floor at 32K — BELOW v2's mxfp8
   total (0.768) and near mxfp4's (0.660): with enough splits the
   native kernel's asymptote already reaches v2 class. The B slice
   (reduce kernel + wave tails) is the walk-base lever to attack, not
   more splits per se.
2. Cheap follow-up (~2-line host change + rebuild): raise the auto
   request toward the cap for large ctx (splits_bw divisor 12 -> 8,
   or lift the reduce cap knee) and re-probe mode0 — expected
   mxfp8 1.278 -> ~1.1 if s_eff 24 -> ~40.
3. The 1/s scaling degrades between s=8 and s=16 (1.64x not 2x) —
   consistent with B, not with a new bottleneck.
