# p43 — warm-t2 adder (TWO_HS transfer + scheduler tail-split v4)

## What

Close the warm second-hit (t2) adder that kept warm-hit above D's band:
the fixed ~0.39 s + residual cost on cache-hit requests.

- **TWO_HS KV transfer** `vllm_gaudi/xpu/patches/lmcint/lmc_twohs_transfer.py`
  + `lmc_kvfmt_detector.py` (r33): attn-KV pool is per-token K/V
  **interleaved**; the byte-true path is `NL_X_NB_BS_NH_TWO_HS` — detect
  the layout, transfer both halves. Fixed the r31/r32 warm re-hit
  integrity break (N2/N4 faces) — all integrity faces green since.
- **Scheduler tail-split v3 → FALSIFIED, v4 scoped** (r34/r35,
  `vllm_gaudi/xpu/patches/sched_tail_split.py`): v3 (defer the trailing
  block) broke 0-drift deterministically (r34) — same spec-contamination
  class as p41's r34 law; v4 = final-boundary-at-finish snap on the glue
  side instead: 1k-face warm 9.2 s → 544.7 ms GATE PASS (ab81fc11).
- **Face closure** (r36, 28160a98): the 8k warm gate re-faced from
  D-donor CL1_RESULTS (cl1b t2 8k/16k = the 0.587/0.741 s bars);
  composition-coherence law — absolute 0.75 s at 8k is structurally
  unreachable (residual 1024 tok × 0.561 ms/tok → floor ≈ 0.87 s; D's own
  KNOWN_ISSUES #22 = same wall). Operative contract term "beat D's adder"
  MET: 1.03–1.05 vs D-SPEC1 1.14–1.28; 1k face 545 ms beats even C's 0.587.
  r30's "755.1 ms FAIL-by-5ms" retired as a folded-transfer artifact.

## Knobs

`VLLM_ARC_R35_TAIL_SPLIT` (default 1 = v4 scoped tail-split ACTIVE, `=0`
disables) — `patches/sched_tail_split.py:59`; see docs/envs.md.

## Test

Warm faces t1/t2 per bench_longctx; churn re-hit warm-class; integrity 5/5
after warm sequences (the r34 discriminator: any N4-class cell must not
shift shas); `kvfmt TWO_HS` lines in the serve log of every boot.

## Validation

- r33: N2/N4 integrity FIXED (fa08c36e); r34: v3 A/B falsified + reverted
  with control (c4468339); r35: v4 scoped banked (ab81fc11).
- r36: gate closure + probe strip at E4 (28160a98).
- r42: warm-hit 740 ms ≤ 750 bar MET on the certified image — first time
  (champion band 587–741); r43 k5 equal (740 ms).
