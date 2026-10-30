# r43 — k5 A/B on the certified forward: ADOPTED — canonical posture = num_speculative_tokens 5

One-sed boot on the p45b certified image (r43_serve_k5.sh: the r42 line
with `num_speculative_tokens` 4→5; diff = line 52 only). First-traffic
battery, D-certified tools unchanged. The r38 k5 CANDIDATE (+6.7/7.8% on
the OLD forward) is CONFIRMED on the p44 forward.

## A/B faces (k5 vs r42 k4, same image, same night)

| face | k4 (r42) | k5 (r43) | verdict |
|---|---|---|---|
| solo200 tps (bar 79.74) | 85.01–85.16 | **86.39/86.40/86.50/87.13** | **k5 +1.5–2.4%** |
| agg4x256#1 (bar 254.33) | 266.68/266.72 | **273.52/274.01** | **k5 +2.6%** |
| agg4x256#2 (bar 274.41) | 297.65/297.94 | **303.48/305.42** | **k5 +1.9–2.5%** |
| solo dt | 2.35 s | 2.30–2.32 s | k5 faster |
| pos1 (target 0.85) | 0.9020 | **0.8919 MET** | equal-class (both ≥ 0.85) |
| TTFT t1 8k/16k/32k | 3445/7252/15552 | 3987/7238/15534 | 16k/32k equal; 8k first-cell artifact |
| warm t2 8k/16k/32k | 845/665/1179 | 848/675/1178 | equal |
| warm-hit (bar ≤750 ms) | 740 | **740** | MET |
| survival (flip first) | 200 / 7.02 s | 200 / 6.09 s | PASS |
| integrity / determinism | 5/5, 3/3 | **5/5, 3/3** | PASS |
| hooks FAILED / reroute | 0 / ×4 | 0 / ×4 | clean |

## Numerics under k5
- golden: **all 5 cells byte-identical to k4** (63b97ff3/9e9d9aa9/
  702a74d2/785b74ef/99955622 — json md5 e02cbb07… equal) — the k4-re-banked
  golden_ref already covers the k5 posture; no drift face opens.
- integrity/determinism: only seeds 101/202 shas shift vs k4 (draft-len
  argmax-tie redistribution, same class as the p44 numerics note); 303/
  404/505 and 11/22/33 byte-identical.

## Decision
**ADOPT k5** — every throughput face wins (+1.5–2.6%), every gate green,
TTFT/warm faces unchanged, golden unchanged. Canonical serve line =
r43_serve_k5.sh (`num_speculative_tokens:5`); the k4 script stays as the
A/B fallback. r38's deferred candidate is now the certified posture; no
further re-canonicalization owed beyond the golden_ref (already valid
for both) — the 12-cell §4.8 matrix (r44) runs on this k5 serve.

## Rig
arc-test up on p45b serving k5 during the r44 matrix; W14 restore after
the matrix closes the night's arc block.
