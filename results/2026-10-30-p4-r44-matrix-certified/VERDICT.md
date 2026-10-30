# r44 — §4.8 bench matrix on the certified image (p45b, k5 canonical) — 12/12 cells

The certification-set matrix face (the earlier owed-notes' "18-cell" was a
mis-carry: the §4.8 certified face is 12 cells — input 1k/16k/32k/128k ×
concurrency 1/4/16, output 512, --ignore-eos, whole cell queued at once).
Run against the live k5 serve (r43 canonical posture) via the arc-adapted
port of /root/build/matrix_cells.sh (r44_matrix.sh; deltas = container
name, served-model name, offload-counter line → spec counters). Cell
files are the evidence; summary_clean.txt is the parsed table (the
in-script summary greps missed vllm bench 0.30's multi-space alignment —
extraction-only defect, re-parsed post-hoc, no re-run).

## Reference faces (regression baseline for P5 + future stack bumps)

| cell | TTFT ms | TPOT ms | ITL ms | tok/s | acc len | pos profile |
|---|---|---|---|---|---|---|
| i1024-c1 | 583.67 | 8.60 | 46.77 | 102.79 | 5.46 | 92.6/90.4/89.4/87.2/86.2 |
| i1024-c4 | 1985 | 11.07 | 64.47 | 264.69 | 5.85 | 98.3/96.9/96.9/96.3/96.3 |
| i1024-c16 | 6237 | 28.97 | 126.85 | 325.55 | 4.40 | 85.3/75.3/69.0/57.5/52.7 |
| i16384-c1 | 7381 | 9.13 | 49.11 | 42.50 | 5.40 | 90.5/87.4/87.4/87.4/87.4 |
| i16384-c4 | 13263 | 31.51 | 160.63 | 66.43 | 5.12 | 90.0/84.0/80.3/79.1/78.8 |
| i16384-c16 | 46422 | 111.19 | 523.98 | 72.38 | 4.73 | 84.7/78.7/74.5/69.5/66.1 |
| i32768-c1 | 16084 | 10.83 | 52.19 | 23.68 | 4.82 | 91.5/86.8/77.4/66.0/60.4 |
| i32768-c4 | 34287 | 44.64 | 210.74 | 34.47 | 4.73 | 90.3/82.0/72.5/66.7/61.9 |
| i32768-c16 | 97008 | 161.02 | 708.16 | 34.84 | 4.41 | 82.0/73.2/66.8/61.3/58.2 |
| i131072-c1 | 101095 | 20.71 | 66.15 | 4.58 | 3.19 | 86.3/64.6/42.2/16.8/8.7 |
| i131072-c4 | 162170 | 162.68 | 854.78 | 6.24 | 5.29 | 90.0/87.2/85.4/84.3/82.0 |
| i131072-c16 | 903980 | 184.16 | 843.54 | 4.78 | 4.60 | 87.3/78.3/70.0/65.2/58.8 |

## Reading
- c1 step (ITL): 46.8 / 49.1 / 52.2 / 66.2 ms at 1k/16k/32k/128k —
  attention growth ≈ +19 ms/step to 128k; decode class at ≤32k stays in
  the r41/r42 arithmetic band (c1 ≈ 47–52 ms).
- k5's 5th draft position pays at ≤32k everywhere (pos4 52–87%); at
  128k-c1 it decays to 8.7% (len 3.19) — the MTP long-context decay
  class (champion's pos4 is dead at ALL depths; ours is alive ≤32k).
  128k-c4's len 5.29 on random data is a per-cell sample; the canonical
  acceptance gate remains the 4-stream spec gate (pos1 0.8919 MET).
- TPOT (per accepted token) 8.6–20.7 ms at c1 — all faces comfortably
  inside the ≤60 ms gate class.
- 128k-c16 mean TTFT 904 s = the certified whole-cell-queued shape (16 ×
  128k prefills at ~2.3k tok/s + decode) — reference, not a gate.

Rig: matrix finished 12/12 at ~00:20; W14 restored after (see r44
end-state in plan §12). Artifacts: rig
/root/build/matrix_arc/leg-2026-10-29-r44-certified/ + this dir.
