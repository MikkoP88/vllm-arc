# VERDICT — P6.47 P-I r250: the k4v8 lane dispositioned CLOSED-DOMINATED by banked evidence (zero rig time)

**What**: P-I improvement cycle 1 of 3 (the P-E lever map's ranked
candidate #1, `docs/MX-RESEARCH-arc.md` §5.1). The k4v8 KV lane —
e2m1 K + e4m3 V, the mirror of certified mxk8v4 — is dispositioned
**CLOSED-DOMINATED by analysis**: strictly dominated by the already-
certified mxk8v4 lane on every axis that gates. No engine wiring, no
rig cycle. This follows the fast-cycle law's intent: cycles are for
candidates with unknown signs; this one's sign is fully determined by
three banked laws plus arithmetic.

## The proof (four links, each banked)

1. **Bytes — the lever map's own arithmetic slip.** The lever table
   priced k4v8 as "capacity: K-half"; the slot layout is symmetric
   in K/V operand widths, so halving K bytes doubles V bytes:
   `slot = D*k_bits/8 + D/32 + D*v_bits/8 + D/32` (E8M0 scale
   bytes per 32-element group per operand,
   `vllm_gaudi/xpu/patches/mx_kv/mx_layout.py`). Verified at every
   certified head size (slot_arithmetic.txt, this bank):

   | D | mxk8v4 (8/4) | k4v8 (4/8) | mxfp4 (4/4) | mxfp8 (8/8) |
   |---|---|---|---|---|
   | 128 | 200 B | 200 B | 136 B | 264 B |
   | 256 | 400 B | 400 B | 272 B | 528 B |
   | 512 | 800 B | 800 B | 544 B | 1056 B |

   **k4v8 ≡ mxk8v4 at every D — the lane buys zero pool capacity
   over the certified mix.** The "K-side capacity headroom at deep
   context" framing in the r247 lever map was a ref-math slip (the
   ref-harness-bug-audit law: audit the ref's math — here the ref
   IS the certified `mx_layout` arithmetic).

2. **K side — two independent banked laws convict e2m1 K.**
   - r238 acceptance mining
     (`results/2026-11-24-p647-r238-n8-accept-mining/`): e2m1 K
     costs ~0.23 acc/step (−10%, ~5 SE, the only above-noise claim
     of the factorial), concentrated at the deepest verify
     positions (pos2 0.636 vs 0.767); e2m1 V costs ~0.03
     (sub-noise). The verdict's own closing line names the optimal
     mix: "the mkq split (e4m3 K + e2m1 V) avoids it while keeping
     half the KV savings."
   - cert24 gate run1
     (`results/2026-11-12-p647-cert24-gate/`): m4q (mxfp4, e2m1 K)
     is the gate's one **REAL quality-limit failure** — essay
     REP-LOOP under churn on suite-14 long-return — while m8q
     passed fully. This is the evidence that built the mxk8v4
     quality doctrine (`mx_layout.py`: "8-bit keys keep retrieval
     precision, 4-bit values ride the byte savings").

3. **V side — e4m3 V buys quality where none was lost.** e2m1 V
   rode the full r245 certification grid on the mxk8v4 lane
   (`results/2026-11-26-p647-r245-page128/`: 37/37 + 8 KV_CAP at
   D=512/page-128) and the r238 acceptance battery (sub-noise).
   k4v8's e4m3 V improves an operand two independent batteries
   measured as quality-free at the 4-bit point — while paying the
   byte cost of the upgrade.

4. **Dominance.** Same slot bytes as mxk8v4 (link 1), strictly worse
   on the K axis (link 2), strictly better on an axis measured
   worthless (link 3), at strictly more wiring surface (a new name
   through the selector, dtype map, kv-mode map, engine routing
   tables, V5/native thresholds, layout, quant, dequant, native and
   Triton decode kernels — the ~15-site mxk8v4 chain, mirrored).

   There is no face — capacity, acceptance, corruption, or
   throughput — where k4v8 can beat mxk8v4. A rig cycle to confirm
   arithmetic contradicts the fast-cycle law's budget.

## The lever-map correction (same increment)

`docs/MX-RESEARCH-arc.md` §4/§5 updated: the k4v8 row moves to the
closed channels with this verdict's citation chain; the "K-half"
capacity framing is corrected to "none — byte-identical to mxk8v4";
group-64 E8M0 becomes the open candidate #1, mxfp4-W4A8 #2 (the
per-channel/per-head E8M0 scales stay #3).

## Reopen condition

The disposition is analytic — it rests on the banked faces of THIS
engine. It reopens the moment a face shows **e2m1-V sensitivity** (a
corruption or quality class traceable to V-side precision that the
r238 acceptance battery and the r245 grid did not cover — e.g. a
future long-ctx coherence face at D=512 with V-side attribution).
Then k4v8 is the natural probe lane (the only mix isolating V at
8-bit with K held at the 4-bit baseline).

## NEXT RUNG (P-I cycle 2)

group-64 E8M0 reach: offline oracle first (corruption bands vs
group-32 at the target geometry, e4m3 K), acceptance probe on the
rig only if the offline band passes — the K-side risk is the same
operand this verdict convicts, and the capacity gain is 2–3%.
