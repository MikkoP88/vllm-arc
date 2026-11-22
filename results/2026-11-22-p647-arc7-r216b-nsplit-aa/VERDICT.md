# r216b — ns64 A/A-CONFIRMED as the posture; ns96 exposed as a silent-skip artifact (kernel split ceiling 64) that incidentally prices the pure model step

**Date**: 2026-11-22 · **Leg**: r216b (arc7 Phase-8 confirmation) ·
**Stack**: r212aa wheel (`9a01790b...`) +
`vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay with the r216
NSPLIT pin · **Run**: `r216b_nsplit.log` 03:31–03:45 UTC (two boots,
faces c8192/c16384/c32768 n1 each)

## Verdict

**GATE_PASS — ns64 confirmed on a fresh boot at spread 0.03% (62.18
vs 62.2), acceptance healthy (chunks 52-56 ≈ the r213 pattern), 0
errors, rows=148. ns96 is INVALID as a performance reading: silent
kernel skip past the ceiling.**

### The ns64 posture (real)

| ctx | itl (auto) | itl (ns64) | dev | tps (auto) | tps (ns64) |
|---|---|---|---|---|---|
| 8192 | 48.5 | 47.3 | −2.4% | 75.8 | 73.4 |
| 16384 | 55.9 | 52.3 | −6.5% | 62.6 | 65.7 |
| 32768 | 70.2 | 62.2 | **−11.4%** | 51.8 | **60.9** |

Gap to the fp8 bar at 32K: 1.65x → **1.46x**. Best-banked mxfp4
posture on every axis; the 8K tps −3% rides cross-boot acceptance
variance (itl improved; chunks 56 vs 52).

### The ns96 artifact (and what it bought us)

ns96's itl 39.9/40.4/41.4 "beat fp8" — but per_user_tps collapsed to
31.8/50.2/36.4 with chunks 96-152 (vs 52-56): acceptance collapsed on
garbage output. Root cause: `max_num_kv_splits = 64`
(`mx_paged_decode_kernel.hpp:100`; the reduce's LSE arrays are sized
to it, `mx_reduce_split.hpp:106-108`) and `can_implement` returns
false above it WITHOUT a loud TORCH_CHECK in the binding — the kernel
silently never ran and the engine streamed uninitialized attention
(the silent-failed-launch stale-buffer canary law, now observed at the
binding layer).

**The free calibration anchor**: ns96 therefore measured the pure
model step with NO attention kernel: **model_step ≈ 39.9/40.4/41.4 ms
(mildly ctx-scaling via the GDN state walk)**. Consequences:

- fp8's stock attention kernel ≈ 42.5 − ~41 ≈ **1.5 ms/step total**
  (16 layers × ~94 µs) — the fp8 bar is nearly all model.
- MX kernel at ns64 ≈ 62.2 − ~41 ≈ **21 ms/step** (16 × ~1.3 ms/call)
  — was ~29 at auto-40. The 14x per-call gap to stock persists.
- The deeper-splits trend is real (timing improved monotonically
  40 → 64 → "skip"): raising the ceiling is a LAWFULLY promising
  wheel-side lever — but only with the reduce arrays, SLM budget, and
  correctness faces re-validated, and a loud can_implement check added
  (fail, never silently skip).

## Posture

- **Banked now**: `ARC_MXKV_NSPLIT=64` is the mxfp4 engine posture
  (env pin; validating legs r216+r216b; the C++ auto-reduce-cap fix
  and the ceiling lift ride the next wheel iteration).
- **Queued (r217 class)**: (a) raise `max_num_kv_splits` past 64 with
  the reduce re-sized + offline correctness faces, then e2e A/A; (b)
  add the binding-level TORCH_CHECK on `can_implement`; (c) the
  kv_tile 16→64 walk policy (floor+union attack) unchanged in
  priority — the ceiling lift is the smaller surgery and the trend
  says it pays.

**Reverted**: nothing — env-instrument legs; the overlay pin is inert
when the env is unset.
