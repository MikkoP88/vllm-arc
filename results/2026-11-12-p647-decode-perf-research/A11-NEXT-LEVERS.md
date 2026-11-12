# A11 next levers — v4 verify-row walk sharing + XMX reopening (design sketch)

**Date:** 2026-11-12 (post-A10/A11-sweep) · **Status:** design for the next
kernel line; nothing implemented. Levers 1 (split cap) is REFUTED
(legA11_summary.txt); lever 4 (combine fusion) is minor. This note scopes
levers 2+3 — the structural close toward the fp8 lane's GQA-packed
economy (FP8-PATH-REFERENCE section 7).

## The structural deficit

v2's per-q-head grid re-walks and re-dequants the same KV tiles up to
24x per step: 6 q-heads per kv-head (GQA) x 4 verify rows (MTP). The A9
GQA-sharing attempt (v3) LOST at D=256 because 6 unrolled accumulator
rows spilled; the split-cap sweep shows the walk is not parallelism-
starved — it is per-byte ALU work. So the win must REDUCE WORK, not
redistribute it.

## Lever 2 — verify-row sharing (v4): 4x work cut at half v3's state

One stage-1 program per (request, q-head, split) serving ALL 4 verify
rows against ONE walk:

- **Limits make it near-exact:** the 4 rows' causal limits are
  ctx-3..ctx (LIMIT-LAW slot inversion upstream); they differ only in
  the final tile. Walk to ctx; mask rows against their own limit only in
  the last tile (a (4, BLOCK_N) mask, negligible).
- **State is HALF of what killed v3:** 4 accumulator rows x D=256 fp32
  (1024 floats) vs v3's 6; q is (4, D); m/l are (4,). The score step
  stays per-row `tl.sum(q_r * k)` — unrolled `tl.static_range(4)`, no
  3D intermediate, no tl.dot required.
- **Economy:** K/V load + dequant once per tile for 4 rows — the walk,
  gather, and dequant amortize 4x. Program count at n8 drops 4x
  (R=32 -> 8 row-groups), directly relieving the 24,576-program
  oversubscription cell where m4 collapsed.
- **Interface:** `_split_verify_decode` already gathers q rows and
  duplicates bt rows — v4 takes the same call shape with a row-group
  axis instead; `seq_lens` becomes per-row limits (sl_rows already is).

## Lever 3 — XMX tl.dot reopening: the score path to MMA

A6's "no tl.dot" non-goal rested on the pointer-chase diagnosis; at
true shape that premise is refuted (v3 loses; the m4 byte-law inversion
at oversubscription says ALU-bound). tl.dot((R_TILE*N_REP, D) @ (D,
BLOCK_N)) with fp16 operands / fp32 accumulator converts the hot
multiply-reduce from scalar FMA to XMX — the same counter-operand
economy ctrl's FA kernel uses (fp8 dequant -> fp16 feeds the MMA pipe).
Numerics: reassociation only; gate d <= 2e-3 like every v2/v3 face.
Combines naturally with lever 2 (the dot's M dimension = 4 rows x 6
q-heads = 24 — MMA-shaped by construction).

## Validation plan

p157-style probe faces (bit-equality vs per-row v2 at the LIMIT-LAW
boundary shapes; timing at the A10 problem cells: 8K/32K x n1/n8, R=32
verify @2K/32K), then the e2e ladder re-run (r162 cells) on the
converged image — the m4@32K/n8 cell is the acceptance-metrics question
r161's serve logs settle first.
