# r217 — split-ceiling lift lands: ns128 = 56.7 ms/tok @32K (1.33x fp8, was 1.65x at auto); ns96 proven REAL at 58.2; the old fast-garbage cell formally closed

**Date**: 2026-11-22 · **Leg**: r217 (arc7 Phase-8 wheel surgery e2e) ·
**Stack**: the r217 wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`9ded4fb67e15ba092aac607d58db1b72` — r212x2 + `max_num_kv_splits`
64→128 + the loud `can_implement` TORCH_CHECK) +
`vllm-arc:v0.30.0-b70-cert25zc` + the mx_kv overlay (r216 NSPLIT pin) ·
**Run**: `r217_ceiling_lift.log` 06:06–06:35 UTC (three boots, faces
c8192/c16384/c32768 n1 each) · **A/A confirmation**: r217b (ns128 fresh
boot, follows this bank)

## Verdict

**GATE_PASS — the ceiling lift pays immediately and perturbs nothing:
ns64r reproduces the banked posture at −0.07%, ns96r is now a VALID
cell at 58.2 @32K (the old-ns96 39.9 was the silent skip, formally
closed), and ns128 takes the best-ever banked mxfp4 cell: 56.7 @32K.**

### Results (mxfp4 n1; itl ms/tok; quality-gated on chunks ≤ 80 + tps)

| ctx | auto (r213) | ns64 (r216b) | ns96r | ns128 | ns128 tps | chunks |
|---|---|---|---|---|---|---|
| 8192 | 48.5 | 47.3 | **47.2** | 48.5 | 71.8 | 53–57 |
| 16384 | 55.9 | 52.3 | 50.9 | **51.2** | 68.4 | 56 |
| 32768 | 70.2 | 62.2 | 58.2 | **56.7** | **65.4** | 52–53 |

- **ns64r repro −0.07% @32K** (62.16 vs 62.2 banked; 47.3/52.3 exact at
  8K/16K): the lift + TORCH_CHECK edits are inert at ns64 — the
  ns128 posture is not a wheel-difference artifact.
- **ns128 @32K: 56.7 = −8.79% vs ns64, −19.2% vs auto; 1.33x the fp8
  bar (was 1.65x at auto, 1.46x at ns64); tps 65.4 = +26% vs auto
  (51.8), and 8K→32K tps droop flattened (73.4→65.4 vs 75.8→51.8).**
- **ns96r REAL: 58.2 @32K (−6.45% vs ns64), chunks 53, tps 63.9** —
  confirms the r216b diagnosis end-to-end: same env value that
  fast-garbaged on the old wheel now runs the kernel and lands on the
  curve (39.9 was never a timing; it was uninitialized memory).
- 8K optimum stays ~96 (ns128 +2.46% there); 16K/32K prefer 128.

### The surface (32K): decelerating but not flat

40 (auto) 70.2 → 64 62.2 (−11.45%) → 96 58.2 (−6.45%) → 128 56.7
(−2.6% per step now). Deltas halve per octave of splits — asymptote
reads ~54–55. A further lift to 192/256 is priced at another ~2h20m
build for ≲2%: queued behind the auto-reduce-cap fix, not ahead of it.

### Wheel-side state (r217 tree vs r212x2, backups `.r217bak` on rig)

- `mx_paged_decode_kernel.hpp:~105`: `max_num_kv_splits` 64→128 (reduce
  SharedStorage scales with it: 2 × 128 × 4B = 1 KB SLM — budget fine).
- `mx_paged_decode.hpp:~557`: `launcher.run(...)` now under
  `TORCH_CHECK(... == cutlass::Status::kSuccess, ...)` — over-limit or
  unimplementable args fail loudly; the silent-skip class is closed at
  the binding layer.
- Overlay pin `ARC_MXKV_NSPLIT` unchanged (env instrument; unset = auto).

## Posture

- **Pending r217b A/A** (fresh boot, ns128 only): if ≤2% spread, bank
  `ARC_MXKV_NSPLIT=128` as the mxfp4 engine posture (96 at 8K-lean
  deployments is within noise of the same pin).
- Queued next (priority order): (a) C++ auto re-priced for engine
  geometry — kill the env pin by making `mx_get_num_splits` pick this
  region itself; (b) ceiling 192/256 probe (decelerating); (c) the
  kv_tile 16→64 walk policy — the floor+union pools (r215) remain the
  structural gap; splits exhausted their headroom at ~13 ms of the
  27.7 ms pool.

**Reverted**: nothing — the wheel keeps the lift + check (safety-neutral
at ns64, proven −0.07%); env pins inert when unset.
