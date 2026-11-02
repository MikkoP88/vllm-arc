# p614 L1v2 — fix v3 image (p614c2) TQ4 leg: ALL GATES PASS (2026-11-02)

**Context:** owner directive 2026-11-02 (fix turboquant lagged store).
Image `vllm-arc:v0.30.0-b70-p614c2` (fork_sha 6f1a09d9, repo ==
image-native adapter md5 `33812d19…`), fix v3 = TQ branch first in
`_arc_attn_pool_view` + kernel-block→page merge + wheel-derived slot.
One variable vs the certified posture: `--kv-cache-dtype
fp8_e4m3` → `turboquant_4bit_nc` (substitution verified 1/0,
`rig/serve_diff.txt` line 52). Baselines: A/B leg
(`../2026-11-02-p65-tq4-ab/`, cert5 image, stores broken) and L1v1
(`../2026-11-02-p614-l1v1/`, p614c1 fix v1, stores broken).

## THE FIX IS PROVEN

| Gate | L1v1 (fix v1) | **L1v2 (fix v3)** |
|---|---|---|
| G1 ERROR/Traceback | 110/110 FAIL | **0/0 PASS** |
| G2 `[r43kvfmt]`/`[r43xfer]` kit lines | 0 FAIL | **`NB=500 BS=1664 TWO=2 HS=262` ×2 + `TWO_HS interleaved branch live … H=262 dir=D2H` ×2 PASS** |
| G4 MemoryObj GC warnings | 270 FAIL | **0 PASS** |
| G5 integrity / determinism | 5/5, 3/3 | **5/5, 3/3 PASS** |
| G6 spec gate | PASS | **PASS pos1 0.9533, acceptance 0.9159** |
| G7 t0 agg#2 / solo | 299.00 / 85.61 | **300.06 / 86.31+86.38** (band 282.33–312.05 / 80.26–88.70) |

The lagged attn store now SUCCEEDS end-to-end: 0 `Lagged attn store
failed` lines, healthy `Stored 1664 out of total 1664 tokens … 3.36
GB/s offload` store lines, both TP workers detect
NL_X_NB_BS_NH_TWO_HS at page granularity on the FIRST try, and — first
time under any TQ preset — the external tier participates: External
prefix cache hit rate climbs 2.2 → 4.1 → 5.9 % through the integrity
re-hit faces (every prior TQ leg was pinned 0.0 with stores failing).
Integrity 5/5 now means the store path is actually exercised, not
bypassed.

## Performance faces (same harness, same rig)

| Face | fp8 cert5 family | TQ4 L1v2 | note |
|---|---|---|---|
| KV capacity | 405,491 tok (1.55×) | **757,641 tok (2.89×)** | +1.87× at the same 6.92 GiB pool |
| flip-first 832 tok | 8.3 s | 10.3 s | in family |
| run7n t0 agg#2 / solo ×2 | 297.19 / 84.48+84.47 | **300.06 / 86.31+86.38** | above family top |
| run7n t1.0 xhigh agg#2 | 189.41 | 93.05 (101.62 first) | still collapsed — 4-bit QUALITY cost (ragged early-finish streams), NOT the store bug |
| run7n t1.0 xhigh solo ×2 | 62.35 / 61.85 | 77.60 / 25.94 | same shape as A/B (77.34/25.85) |
| U32k TTFT | 23.16 s | **15.91 s** | −31 % win holds |
| U64k TTFT | 50.77 s | **34.02 s** | −33 % win holds |
| W_U32k_start gap p95/max | 1299/1310 ms | 3218/3334 ms | unchanged vs A/B — inherent bystander cost |
| W_U64k_start gap p95/max | 1569/1602 ms | 3785/3918 ms | unchanged vs A/B |
| W_*_end windows | clean | clean (0 stall rows) | |

## Disposition

Store-path defect: FIXED (owner directive's "fix all issues" for the
lagged store under TQ — confirmed at live geometry HS=262 BS=1664).
Remaining TQ4 costs are QUALITY-side and inherent to 4-bit sampled
throughput (t1.0 xhigh raggedness, bystander gaps) — owner-call
material per the A/B disposition; `turboquant_k8v4` (HS=388) is the
quality-safest candidate and is characterized next. Default fp8 face:
non-regression battery (L2) on the same image; then remaining preset
legs (k3v4_nc HS=230, 3bit_nc HS=198).

Rig addresses/hostname redacted byte-exactly (+125/+5; see
`../REDACTION-2026-11-02-p614.md`). W14 restored by the leg's EXIT
trap.
