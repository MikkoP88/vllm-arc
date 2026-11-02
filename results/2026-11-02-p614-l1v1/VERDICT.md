# p614 L1v1 — first fix image (p614c1, fix v1) TQ4 leg (2026-11-02)

**Context:** owner directive 2026-11-02 (fix turboquant lagged-store).
Fix v1 (commit preceding 6f1a09d9's amend) taught `_arc_attn_pool_view`
the TQ packed layout and gave the engine a uint8 TWO_HS geometry with
`hidden_dim_size = H_kv * 268` from the then-current static reading of
the wheel (head 268). Image `vllm-arc:v0.30.0-b70-p614c1`. This leg is
the FIRST post-fix TQ4 leg; it FAILED its store gates and produced the
decisive partial results that pinned the remaining root cause (banked in
`../2026-11-02-p614-dbg/`).

## Gates (leg `rig/` artifacts, script `p614_tq_leg.sh` on rig)

| Gate | result | note |
|---|---|---|
| G1 ERROR/Traceback == 0 | **FAIL** | 110 pairs, same single class as the A/B (lagged attn store) |
| G2 `[r43kvfmt]`/`[r43xfer]` kit lines | **FAIL** | 0 kit lines — detector still refusing |
| G3 external prefix hit rate | informational | 0.0 (as in cert5 legs; nothing evicts at this headroom) |
| G4 MemoryObj GC warnings == 0 | **FAIL** | 270 (downstream of the 110 store failures) |
| G5 integrity + determinism | **PASS** | 5/5 + 3/3 (A/B was 3/5) |
| G6 spec gate | **PASS** | pos1 0.9406, acceptance 0.8889 |
| G7 t0 family | **PASS** | agg#2 299.00, solo 85.61 (band 282.33–312.05 / 80.26–88.70) |
| flip-first 832 tok | 9.7 s | in family |
| U32k / U64k TTFT | 16.69 s / 33.46 s | TQ wins hold (−28 % / −34 % vs fp8 family) |
| KV capacity | 757,641 tokens | +1.87× at the same 6.92 GiB pool |
| bystander gaps p95/max | 3184/3288 (U32k), 3500/3504 ms (U64k) | unchanged vs A/B — inherent TQ cost, not the store bug |
| t1.0 xhigh | agg 92.76, solo 77.35/25.86 | still collapsed — 4-bit quality cost (owner-call material), not the store bug |

## What L1v1 proved / disproved

- The engine-init branch RAN (boot log kv geometry `(17, 2, 1664, 1,
  262)`), but `hidden_dim_size` 268 ≠ live slot half 262 → the static
  head-268 reading was wrong; live geometry is head 256 / H_kv 2 @TP2.
- Integrity 5/5 with the store path still failing (0 kit lines, 110
  errors) confirms the A/B's 3/5 was recompute flakiness, NOT proof of a
  working store: **integrity alone cannot certify the store path; the
  `[r43kvfmt]`/`[r43xfer]` lines + 0 ERROR are the real gates** (this is
  now G1/G2 in the leg script).
- All the A/B performance faces reproduce on p614c1 (capacity, TTFT,
  t0, gaps) — the fix changed nothing user-visible except leaving
  integrity at 5/5.

## Disposition

L1v1 = root-cause evidence leg, not a candidate. The remaining failure
chain (kernel-block 128-token pool views + fp8-branch swallow at
H_kv=2 → detector `BS >= 256` refusal) was pinned by the instrumented
boot in `../2026-11-02-p614-dbg/` and fixed in commit 6f1a09d9 (fix v3:
TQ branch first + page-granularity merge), validated by
`../2026-11-02-p614-preflight/` (PRE_FLIGHT_V2: OK, 28/28) and rerun as
L1v2 on image p614c2.

Rig addresses in `rig/serve_full_tq.log` redacted byte-exactly
(+89 bytes; see `../REDACTION-2026-11-02-p614.md`). W14 restored by the
leg's EXIT trap.
