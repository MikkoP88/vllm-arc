# VERDICT — P6.47 Phase 7 closing rung: TILE_FLOOR=2 e2e confirmation (r195)

**LEG**: 2026-11-19-p647-tf2-e2e (rig bank /root/build/p647/tfe2A26)
**IMAGE**: vllm-arc:v0.30.0-b70-cert25zb; ARC_MXKV_V5_TFLOOR=2 via the
new serve_arc_b70.sh pass-through trio — the single delta vs the
banked baseline chains (r185b m4q/mkq cells, S1 m8q cell).
**GATES**: ACCEPT_KNEE_PASS required; acceptance must stay
baseline-level; ITL compared against the banked baselines.

## Result table (acceptance% / mean-len / ITL p50 / eff / ss, ms)

m4qtf2 (decoder v2 — TFLOOR-exempt by design; the null lane):

| ctx | n1 base itl | n1 tf2 itl | Δ | n8 base ss | n8 tf2 ss |
|---|---|---|---|---|---|
| 4096 | 49 | 49 | 0 | 228 | 249 |
| 8192 | 58 | 58 | 0 | 457 | 513 |
| 16384 | 76 | 76 | 0 | 895 | 917 |
| 32768 | 112 | 112 | 0 | 1978 | 1971 |

(n1 byte-identical — v2's split rule is untouched by the knob; n8
aggregate ITLs also identical to r185b (131/201/341/1547); the ss
column's +9-12% at 4K-8K is the known ramp-noise band for this column
class.)

m8qtf2 / mkqtf2 (decoder v5 — the knob's lanes), n1 ITL:

| ctx | m8q base | m8q tf2 | Δ | mkq base | mkq tf2 | Δ |
|---|---|---|---|---|---|---|
| 4096 | 65 | 65 | 0 | 58 | 58 | 0 |
| 8192 | 69 | 69 | 0 | 66 | 64 | −2 |
| 16384 | 75 | 70 | **−5** | 85 | 83 | −2 |
| 32768 | 106 | **95** | **−11 (−10.4%)** | 126 | 121 | −5 (−4.0%) |

n8 ss: m8q 193/409/735/1639; mkq 253/485/946/1893 — no regressions vs
the lane class.

Quality: acceptance healthy at every cell (72.7-94.0%, KNEE=none x3);
24/24 groups CONC_OK, no KV caps, no timeouts.

## VERDICT: FLIP CONFIRMED — TILE_FLOOR default 4 -> 2

- The e2e confirms the r193 kernel verdict with upside: m8q n1 gains
  −10.4% at 32K (kernel-level predicted ~−1%; the extra is the plan
  replay at deeper splits — the B1-noted plan-shape effect runs in
  reverse at tf2), mkq −4%, and NO lane regresses (m4q null as
  designed; the knob cannot touch the v2 path).
- The kernel-level <=8K win did not surface e2e (4K/8K flat) — the
  short-ctx cells sit near engine-fixed costs; the long-ctx cells are
  where the split-depth pays.
- **Disposition**: TILE_FLOOR launcher default flips to 2 (kernel
  constexpr stays 4 — the certified-signature freeze; the env default
  is the flip site). Ships in cert25zc together with the G-adaptive
  m4q dispatch (the B1-banked n8 lever: v2 below the knee, v5 above;
  ARC_MXKV_V5_G_MIN, default 8, mxfp4-only) — r196 fastcycle on the
  new image validates both, then the full §10 battery, then release
  per cert sequence.

**OUTCOME: ACCEPT_KNEE_PASS, no regressions, m8q/mkq n1 real wins
(−4..−10% at 16-32K), flip CONFIRMED, cert25zc path open.**
