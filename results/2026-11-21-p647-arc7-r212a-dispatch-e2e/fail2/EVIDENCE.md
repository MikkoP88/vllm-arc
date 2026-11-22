# r212a fail2 — fp16 gate cleared; D=256/page-64 geometry law

**Date**: 2026-11-21 · **Launch**: `r212a_launch3.log` (rig
`/root/build/p199/logs`) · **Bank (rig)**: `/root/build/p647/r212a`
(overwritten-by-rerun slot at the time; fail1 preserved at
`/root/build/p647/r212a_fail1`) · **Overlay**: engine.py
`d1019b78714737f016446474694def0b`, native_decode.py WITH the bf16 cast
(`3236351b292faedcc5796a0bddae4b22`)

## Outcome

**All three cells BOOT_FAIL at the same first-native-capture point —
one contract layer deeper.** The fail1 fp16 cast fix WORKED (the
bf16/dim-3 TORCH_CHECK passed; the uint8 slot-pool layout check before
it also passed at the engine's real slot widths):

```
RuntimeError: mx_paged_decode: head_size 256 not yet compiled
(first-cut arc7 instantiations cover D=128; widen
mx_paged_decode.hpp's dispatch set to add buckets)
```

per cell: m4qa 02:10:04, m8qa 02:14:00, mkqa 02:17:58 (~4 min
boot-to-fail each; run rc=1, W14 restored 02:18:14).

## The geometry law (fail2's discovery)

The e2e target **qwen3.8-27b-fp8** carries `head_dim = 256`
(`num_attention_heads = 24`, `num_key_value_heads = 4` → TP2: HQ 12 /
HK 2 / **G 6** → the kernel's `_8` qgroup bucket), full-attention
layers 16 of 64, engine pool **page 64** (walked_cap 148×4100×64
exact). The entire r208-r212 offline program ran **D=128 / BLOCK=16 /
G=4** — calibrated off the config's `linear_key_head_dim = 128` (the
GDN linear heads), not the full-attention `head_dim` the MX KV lane
serves. Consequences:

1. **Every E5b crossing magnitude is D=128-priced** — the engine.py
   thresholds {mxfp8 262144, mxfp4 8192:32768, mxk8v4 32768} are
   provisional until recalibrated at the real geometry (r212c).
2. The wheel's dispatch widens to {128,256}×{16,64} using the VENDOR's
   own policies (`decode_policy_qpacked_head` carries q8/q16 × h64-h576
   × p16/p32/p64/p128; the `_64` policy services any 64-multiple page
   via the page-table indirection) — the MX fork's kernels are
   shape-generic (mx_paged_decode.hpp law; staging derives from policy
   tile shapes, walk takes runtime page_size). Build19 (incremental,
   VXK_ARC7MX_CLEAN=0) produces the widened wheel; the r210 keeper is
   preserved as `..._r210keeper.whl`.
3. p204 harness geometry env-parameterized (P204_D/P204_HREP/
   P204_BLOCK) — r212c runs 256/6/64 (engine cell, thresholds source),
   256/6/16 (head-width isolation), 128/4/16 (certified-cell regression
   vs the banked E5b numbers, kernel binary-layout dice band).

## Artifacts

Rig: per-cell `serve.log`/`boot.txt` under `/root/build/p647/r212a`,
`r212a.txt` (report), `serve_patch.txt`, `overlay_md5.txt`. Local: this
note; the model-config extract (head_dim/heads) is quoted above.
