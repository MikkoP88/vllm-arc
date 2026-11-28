# VERDICT — r267 offline NaN-canary (P6.47 P-D, the decisive probe)

Banked 2026-11-28 from rig bank `/root/build/p647/r267_canary` after
`R267_DONE rc=0 08:59:17Z` + `W14_RESTORE rc=0` (transfer oracle 7/7
byte-match; address audit clean). Ran inside
`vllm-arc:v0.30.0-b70-cert29` against the IMAGE'S OWN arc7mx wheel
bytes (md5-provenance printed; no pip install, no engine boot) — the
exact kernel/store code the r266 legs ran.

## Harness provenance

Two runs are banked:

- `logs/launch_v4_rc1.log` + `logs/r267_v4.out` — the v4 run
  (harness md5 `ac5f46f5`): **rc=1 was a harness bug, not an engine
  reproduction**. `build_paged` counted pool blocks as
  `total//blk + 2`, which under-allocates at `n_seq=6`/page-64
  (autosplit needs 154 blocks, 152 allocated) and the `order[]`
  slice then truncates *silently* → `IndexError` at the last
  sequence's tail — the run died after `pagewidth`, before
  autosplit/uninit/verdict. Every face that ran in v4 was clean.
  The old page-16 geometry had masked the bug (601 ≤ 602); the v4
  engine-true p64 cells exposed it. The v5 fix counts blocks from
  the per-sequence ceil sum and asserts the slice length.
- `logs/launch.log` + `logs/r267.out` — the v5 run (harness md5
  `c5f90789`): complete, `VERDICT: PASS`, zero FAIL lines, rc=0.

## The verdict

**PASS — kernel/store math is GREEN at the engine's true lanes,
pages, and gemma geometry. The engine MX-KV decode corruption is
WIRING (G3: engine metadata VALUES), not kernel or store math.**

| face | cells × fmts | result |
|---|---|---|
| ctrl | 1 × 3 (both arms: wheel native + engine-true Triton-default) | ok, d≈4.9e-4, cos 0.9999987 |
| stepwise (THE onset face) | slide + glob × 3 | all 24 incremental decode-step stores+decodes per cell ok, d≈5-7e-4 |
| store-audit | slide + glob × 3 | 24 incremental stores bit-clean K AND V, sentinels intact |
| widewin | slide + glob × 3 at ctx 8192/32768/40960 | ok; d_blind 0.17-0.21 (the W=1024 mask binds) |
| autosplit | slide (auto/forced8/s1) + glob (Triton) × mxfp8/mxk8v4 | ok, interleaved 6-request tables exact |
| uninit (evidence, no gate) | slide + glob, 0xFF and empty pools | non_finite=4096/4096 and 8192/8192 at every head |
| pagewidth (evidence, no gate) | synthetic p128 D=512 × 3 | D_B reproduced (mxfp8, mxk8v4): default-64 walk corrupts past ln=64, pagefix holds |

Key closures:

1. **The mxfp8 × W=1024 gap is closed CLEAN** — r243 had validated
   the windowed native lane as mxfp4 only; the canary ran the real
   slide cell (D=256 p64 Hk=8 GQA2 W=1024) at mxfp8/mxk8v4/mxfp4 to
   ctx 40960 (the engine's bt width 640) with the window proven to
   bind. Exact.
2. **The Triton windowless lane at the engine's call shape is
   clean** — glob (D=512 p64 Hk=2 GQA8) through the default-64 call
   (== the pool's true page) at bulk, long-ctx, and interleaved
   multi-request shapes. Exact.
3. **Store math is bit-exact at gemma cells** — pointed-row vs
   `mx_quant` reference, K AND V separately, at the v source's
   divergent token+head strides; sentinel rows byte-frozen. The
   stepwise face reproduced the engine's decode-step store→decode
   interleave (the token-1-clean/token-2-broken onset shape) exactly.
4. **The wrong-row read signature is characterized**: decoding over
   never-written rows (0xFF or `torch.empty` pool) yields
   **non-finite at every head, 100% of positions** — the NaN source
   for the engine's 324× 400-nan logprobs. A *stale* (wrong-but-
   written) row read stays finite-but-wrong — the ' our' attractor
   class. Both are metadata-addressing failures, not math failures.
5. **D-B stays latent-hardening only**: on the synthetic p128 cell
   the default-64 walk corrupts past ln=64 while the true-page walk
   holds — banked as the wrapper-fix evidence (`block_size: None ->
   cache.shape[1]`, r269 overlay). Both engine pools are page 64, so
   D-B is exonerated as the corruption cause (r266 serve args).

## Consequence

The surviving hypothesis class is **G3 — the engine hands the
kernels wrong metadata VALUES at decode time** (slot_mapping /
block-table contents / seq lens), making decode read stale or
never-written rows. The next leg is the pre-staged **r268 EAGER
trace** (overlay `0ce0954c`, shipped to `/root/build/r268`): boots
the e2 posture (ARC_GRAPH=0, mxfp8+native — proven to reproduce c3
byte-identically) and prints the live store-source shapes/strides
per store call and the decode bt stride/contiguity/max_seq_len,
comparing store-smap vs bt-walk addressing. Graph-capture is
eliminated (e2/e1 pair), the skin is eliminated (batch-1/2), the
math is eliminated (this canary) — what remains is the wiring.
