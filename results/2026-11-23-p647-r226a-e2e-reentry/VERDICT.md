# r226 — mxk8v4 native e2e re-entry: PASS on fresh engine tensors

Phase P6.47, iteration r226 (2026-11-23). Fork content base: cert26
(`rel/v0.30.0-b70-cert26`, content sha cac63931) plus the r225t-gated
build25 wheel (md5 `aeca45daa740e9973a1c51bfc5ea2bc4`, fixes 1-5).
Branch `feat/arc-p647-r224-mkq-strides`.

## Question

r225 fixed the kernel offline (root causes 1-3 + the fix-5 forced-S==1
regression) and gated it 12/12 on the r224a banked payloads at the
q-cast-matched criterion. Does the fixed native lane hold up in the
ENGINE — on freshly generated essay-class decode traffic, judged at the
same corrected criterion?

## Method

r224a harness verbatim plus one new step (3b: `pip force-reinstall` of
the build25 wheel, `SO_MD5` verify): refusal-posture boot
(`ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1`) -> byte-faithful /proc
argv+environ capture -> wheel install -> instrumented `engine.py`
overlay (md5-verified) -> eager re-serve with `ARC_MXKV_AB=1`, `N=60`,
`MINSL=1000`, graph-off (pageable-H2D capture law) -> one small gen +
one essay-class prompt -> harvest payloads FIRST, class-scan them, then
run the r225t q-cast-matched scorer (`base` cell: each lane vs the fp64
walk of its OWN input) on the first 8 COVERED payloads.

## Payload classes (first-run discovery)

The instrument banks at most 96 blocks (6144 tokens) per payload. The
essay window produces two deterministic call classes: 8199-token calls
(129 blocks needed -> banked window TRUNCATED, not offline-walkable)
and 5463-token calls (86 blocks -> COVERED). r224's own bank held the
same mix (its analyzer report shows calls 1-17 at sl_mean 8199, calls
18+ at 5463) — its three replayed payloads were covered-class by
choice. The first r226a run picked `sort | head -8` = truncated-class
payloads and the scorer aborted at the walk (`index 128 out of bounds
for dimension 0 with size 96`); the abort also lost the payloads (the
harvest loop ran after the scoring gate, and the EXIT-trap teardown
removed the container). Fixed in the wrapper: explicit class scan
(`max_sl <= nbt*64`), scoring only COVERED payloads, harvest-before-gate
ordering, and a strict cell gate (a crashed scorer cell aborts the leg
instead of silently shrinking the scored set). First-run artifacts are
preserved under `first/`.

## Findings

1. **Engine A/B collapsed an order of magnitude** (the instrument's
   `dmax` is the raw lane gap `|native(q.bf16) - triton(q.fp16)|` on
   live tensors — it INCLUDES the r225r q-cast input difference by
   construction, so it is an upper bound, not the verdict):

   | run (same instrument, same traffic) | min | p50 | max | nan |
   |---|---|---|---|---|
   | r224 refusal (build21-era wheel) | 5.981e-2 | 3.052e-1 | 2.273e+0 | 0 |
   | r226 run 1 (build25) | 3.906e-3 | 5.859e-2 | 2.422e-1 | 0 |
   | r226 run 2 (build25, banked here) | 3.906e-3 | 5.469e-2 | 3.906e-1 | 0 |

   120/120 lines `nan=0/0` in every run; the refusal signature was
   magnitude, and the magnitude is gone.

2. **The decomposed verdict passes on FRESH payloads**: 8/8 COVERED
   cells PASS (r226c_score.log). Triton replay bit-exact vs the banked
   engine outputs 8/8 (`dmax=0.000e+00` — the r224c replay law holds on
   fresh traffic); native bitwise-stable in-process with `nan=0` 8/8;
   `NATIVE_VS_WALK` dmax 3.866e-3..1.548e-2, mism 0-3/3072, max_ulp
   0.50-15.51; `QBORACLE_VS_WALK` 4.835e-4..1.838e-3;
   `NATIVE_VS_QMATCHED_ORACLE` <= 1.562e-2. The 15.51-ulp cell sits at
   mism 3/3072 (frac_gt2 = 9.8e-4), inside the pre-registered
   tie-outlier allowance (frac <= 1e-3 AND max_ulp <= 16) written into
   the r225 VERDICT gate plan before any r225t/r226 run; every other
   band unchanged.

3. **Generation stays coherent**: the essay continuation is verbatim
   greedy continuation of the repeated prompt, `finish_reason=length`
   (gen_essay.json) — same class as the certified Triton lanes.

## Consequence

The r225 kernel fixes are live end-to-end: the engine's own instrument
shows the raw lane gap collapsed ~an order of magnitude with zero NaN,
and the q-cast-matched criterion passes 8/8 on freshly banked engine
tensors (both TP workers represented across 86 COVERED payloads). The
mxk8v4 native lane is un-refused at the diagnostic posture. The
CERTIFIED posture flips at cert27 — new image carrying the build25
wheel, full performance battery, and the mxfp4/mxfp8 payload-replay
gate class per the improvement-convergence law (same ReduceSplitK path
serves all three formats).

## Evidence

- `launch.log` — full leg: refuses (exact wheel md5), boot, /proc
  capture (pid=13, argc=42, env=31), wheel install (`SO_MD5
  190e261edc79b3b497d8d8e801e65757`), instrument md5, re-serve,
  traffic, class scan (86/34), 8/8 scoring, teardown, W14 restore.
- `payload_classes.txt` — per-payload `max_sl`/`nbt` + COVERED/TRUNCATED.
- `r226c_score.log` — the 8 scorer cells with full stat lines
  (`R226C_INVOKED=8 SCORED=8 PASSED=8`).
- `ab_lines.txt` — 120 engine A/B verdict lines (nan=0/0 on all).
- `r224b_report.txt` — lane-gap analyzer rollup on the fresh bank.
- `serve.log`, `first/serve.log` — diagnostic serve logs (host
  addresses redacted, see `REDACTION-serve-log.md`).
- `SHA256SUMS.payloads` — manifest of the 120-payload bank (~600 MB
  stays rig-side at `/root/build/p647/r226a/` per the 100 MB artifact
  law; payloads re-derivable from hashes).
- `first/` — the aborted first run: same two-class mix (and the same
  mix visible in r224's own banked report — no prompt/tokenizer
  mystery; `essay_prompt.json` byte-identical across runs), the
  window-cap scorer crash, and the engine-side fix evidence that
  survived (ab_lines nan=0/120, dmax floor at e-3).

## Tools shipped this iteration

`tools/ar/r226a_mkq_ab25.sh` — the r224a leg wrapper extended with the
wheel-install step, the payload class scan, harvest-before-gate
ordering, and the strict scorer cell gate.
