# VERDICT — r266 batch-2 (gemma issue-analysis, legs e2/e1/b5/b4/c1/c2/b3)

Banked 2026-11-28 from rig bank `/root/build/p647/r266_gemma` after
`R266_END 2026-11-28T06:51:14Z leg_failures=0` + `W14_RESTORE rc=0`
(transfer oracle 77/77 byte-match; redaction clean; skin audit 7/7
SKIN_PASS — see `skin_audit.md`). Batch-1 (c3/c4/c5/w1/w2/b2 + defect
census) is banked at `results/2026-11-28-p647-r266-gemma-analysis/`.

## The completed discriminator matrix

| leg | image | kv | weights | graphs | chat t0 ok | reploop | verdict |
|---|---|---|---|---|---|---|---|
| b5_g3_19 | cert19 | fp8 | fp8 | on | 48/54 | 0 | CLEAN |
| b4_g3_20 | cert20 | fp8 | fp8 | on | 48/54 | 0 | CLEAN |
| c1_bf16_29 | cert29 | fp8 | bf16 | on | 48/54 | 0 | CLEAN |
| c2_fa2off_29 | cert29 | fp8 | fp8 (FA2 off) | on | 48/54 | 0 | CLEAN |
| e1_g3_eager_29 | cert29 | fp8 | fp8 | OFF | 48/54 | 0 | CLEAN |
| b3_g4_29 | cert29 | fp8 | fp8 | on | 60/81 | 18 | PARTIAL (official g4 template degenerates — the g4-base-checkpoint law; not a corruption) |
| e2_m8g_eager_29 | cert29 | mxfp8+native | fp8 | OFF | 0 (raw 0/27) | attractor | **RED — byte-identical to c3** |

Batch-1 completes the picture: c3 (mxfp8), c4 (mxfp4), c5 (mxk8v4) all
RED with the same signature (token 1 correct, `' our our our'`
attractor to finish=length, 324x 400-nan on logprobs requests); w1/w2
(MX weights + KV fp8) CLEAN.

## Verdict

1. **The lawful g3 skin serves gemma CORRECTLY at every image era**
   (cert19/20/29), every weight path (fp8/bf16), both attention routes
   (FA2 on/off), graphs on/off. The r264 m8g/m4g/mkg REDs were POSTURE
   ARTIFACTS (qwen skin violence — `tmpl=__qwen_default__
   rp=<qwen3> tp=<qwen3_coder>`), fully superseded by this batch: no
   gemma defect exists in the skin/serving surface.

2. **The corruption is MX-KV-LANE-SPECIFIC and geometry-specific**:
   qwen (uniform layers, windowless Triton decode at D=128) is
   certified clean on the same lane; gemma + MX-KV is RED at every fmt
   (c3/c4/c5) and both graph modes (c3/e2). What gemma uniquely
   exercises on MX: the WINDOWED NATIVE decode lane (50 sliding layers,
   D=256 Hk=8/rank GQA2 W=1024 — the mxfp8 windowed face was validated
   offline as mxfp4 ONLY, r243) and the Triton windowless lane at
   D=512 Hk=2 GQA8 (10 global layers — r242 validated offline at
   d512p64 with hand-quantized caches; the engine's call shape was
   never exercised).

3. **e2 eliminated the XPU-graph hypothesis** (eager reproduces c3
   byte-identically); **e1 exonerated the eager path** (KV-fp8 eager is
   fully clean). D-B (the five engine call sites omitting block_size)
   is EXONERATED as the cause: both engine pools are page 64 (serve
   args `block_size: 64`, page-size unification, p242's certified
   d512p64) — the earlier "global pool p128" census claim was an
   unverified inference, corrected. D-B remains latent-hardening (the
   r269 wrapper fix is still correct to land).

4. **Next probe (decisive)**: the r267 offline canary v4 (shipped,
   engine-true cells: slide D=256 p64 Hk=8 GQA2 W=1024 + glob D=512
   p64 Hk=2 GQA8, engine-dispatch lanes, stepwise onset face + store
   bit-audit + widewin + autosplit). Canary FAIL names the kernel/lane
   defect offline; canary PASS -> the r268 EAGER trace leg reads the
   live store-smap vs bt-walk addressing (G3 metadata values — the
   surviving hypothesis class).

Gates: `R266_PREFLIGHT_OK`, all 7 legs `RESULT=OK` (probe rc=0),
kernel health clean on every leg, W14 restored rc=0 with GEN_OK.
