# r205 — copy-form law dump + tight K walk (REVERTED: binary-layout regression)

**VERDICT: GATED NEGATIVE — correctness PASS (synth-v2 ALL_STAGES_OK x2,
faces 42/42, bitwise 0/1024 x3; the dumped law is RIGHT) but the walk
REGRESSED mxfp8 mode0 +6.4% (1.278 -> 1.360) via a binary code-layout
side effect, not its own execution. Source REVERTED to the r204 state
(commit 0beefa91); r204 wheel 2fb3b721 remains the best/live state.**

- Wheel: `arc7mx-r205` md5 `b3a4705d4d5c16ba13c4b392fbcb9285`
  (archived `fix/`, NOT shipped).
- Source md5 at test: `mx_dequant.hpp` `977ff229`.

## The law (the durable win)

Preflight type-dump (incomplete-template error print) of the COPY
fragment's full tv_layout, 16x64 byte tile:

```
thr ((4,4)) strides (1:B1, 1:B0)   -> a = w%4 advances COL, b = w/4 advances ROW
val ((4,8),2) strides ((4:B0,4:B1),32:B1)
row = (w/4) + 4*(i%4)   col = (w%4) + 4*((i/4)%8) + 32*(i/32)
```

FOUR rows per WI (w/4 + {0,4,8,12}), 4x16 sub-block per WI — NOT the
dst fragment's ((2,8),((2,8),N)) form. **Consequence: r201b's kr[8]
slot law (slot = (i&1)+2*(i>>4)) NEVER matched this form** — e.g. i=2
has true row w/4+8 but kr[0] = w/4, so the `direct` guard tripped at
runtime and the walk silently rode the per-element fallback
(LLVM-CSE'd to ~8 scale loads). r201b's -41% was real but its "law"
was dead code; the bitwise gates passed via the fallback, and the
"direct never tripped" claim in the r201b verdict was unevidenced.

The r205 tight walk (8 upfront scale loads, 4 rows x 2 groups, pure
shift/mask walk + goff select) is bit-correct and pinned by a new
`assert_copy_tv_form` — but see below.

## Why reverted

mode0 32K device ms: mxfp8 1.278 -> 1.360 (+6.4%), mxk8v4 0.971 ->
1.005, mxfp4 1.037 -> 1.010. Probe rerun on the SAME wheel reproduces
every number to 0.001 (rig is repeatable; NOT noise). Attribution:

- m2 (const-V: K path REAL) — FLAT 0.901 -> 0.902: the new walk's own
  execution is not slower.
- m6 (const-K: `dequant_reg` DEAD — probe >= 6 skips the whole K
  branch) — REGRESSED 0.643/0.545 -> 0.728/0.586: the regression fires
  with the changed code never executing. => binary-level effect
  (register allocation / i-cache / scheduling dice) from recompiling
  the TU, landing on the V-carrying modes (m0/m1/m6) while compute
  floors (m7/m8) stay flat.
- mxk8v4's m2 (K chain) DID improve 0.568 -> 0.511 (-0.057): the walk
  is genuinely a bit faster where it runs — just not enough to pay the
  layout tax.

Net across formats (summed mode0): r204 3.286 < r205 ~3.34-3.39. r204
wins; revert. Future K-walk attempts must A/B against the
same-binary rerun bar (0.001 reproducibility) and watch the
path-disabled modes for layout dice before attributing causation.

## Matrix (r205, 32K device ms; rerun values for m0/m2/m6 identical)

| mode | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| 0 | 1.360 | 1.010-1.026 | 0.978-1.005 |
| 2 | 0.902 | 0.543 | 0.511-0.521 |
| 6 | 0.728 | 0.554-0.564 | 0.586 |
| 7 | 0.426 | 0.163 | 0.173 |
| 8 | 0.349 | 0.138 | 0.146 |

## Next (r206)

With r204 restored (mxfp8 0.60x / mxfp4 0.64x / mxk8v4 1.62x of v2):
the remaining levers are the ~0.4 ms V lanes (bf16-direct integer-add
scale fold — corner bands need per-group guards), the K chains, and
the walk base itself (TileScheduler granularity + split-reduce — v2
runs the same byte volume 1.6x faster on mxfp8, so parity there
eventually needs mainloop-structure work, not just dequant lanes).
