# P6.47 r225 — mxk8v4 native residual: three root causes found, fixed, gated

Status: CLOSED — R225T_GATE: PASS 12/12 cells on build25 (wheel md5
aeca45daa740e9973a1c51bfc5ea2bc4). Root causes 1-3 (e2m1 subnormal
mantissa, imprecise combine exp2, bf16 split partials) + fix 5 (S==1
sentinel LSE, a fix-3-introduced regression) close the defect; the
residual gate3/gate4 "failure" was a REFERENCE mismatch, not a kernel
defect (see "Root cause 4 hunt").

## The refusal

cert26 shipped the MX-native decode line with `mxk8v4` refused at the
engine gate: banked r224a A/B payloads showed native-vs-fp64-reference
dmax of 7.7e-2 / 5.5e-1 / 1.5e-1 while the Triton reference lane sat at
3.6e-3 / 6.0e-4 / 6.0e-4 (0.06-0.38 output-ulp).

## Root cause 1 — e2m1 subnormal mantissa drop (r225 fix 1)

`e2m1_scaled_bf16` (mx_dequant.hpp) built the bf16 result by inserting
the e2m1 mantissa WITHOUT the implicit bit for ee==0 (subnormal)
payloads — every subnormal V payload decoded to +-0.0. Deep-layer V
bytes are ~1/6 subnormal, so the V side of the output collapsed.

Fix: set the implicit bit for the normal band (ee!=0) and decode
subnormals as m*0.5 per OCP. Proof in-wheel (r225d, build22 wheel):
base == dbf0 BITWISE on all three payloads (dmax 0.0 — the fast
direct-bf16 path converged onto the float path, as every e2m1 value is
exactly representable in bf16 once the bug is gone); base moved off
the banked broken output; native error collapsed 8-25x
(7.748e-2 -> 7.685e-3, 5.525e-1 -> 6.874e-2, 1.479e-1 -> 5.999e-3).

## Root cause 2 — native::exp2 in the split-KV combine (r225 fix 2)

NECESSARY HYGIENE, NOT SUFFICIENT: the build23 gate (fix1+fix2,
r225d_gate2.log + fix2fail/) came back with a residual IDENTICAL to
fix1's — the fix was proven live in the gated bytes (single .o
rebuilt, fresh .so, new wheel md5 ebc5845b) yet max_ulp did not move.
native::exp2's ~1e-3-relative error and the true dominant source
(below) are the same error class; removing the smaller left the max
unchanged.

Fix 1 left a 7e-3..7e-2 residual = 10-20x the Triton lane's error,
uniform err/|O| p50 ~ 2^-9.4 (exactly one bf16 half-ulp of P),
worst absolute error on LARGE |O| — a multiplicative signature.

Falsification chain (each leg banked in this directory):

- r225e profile: uniform across heads/rows, uncorrelated with seqlen.
- r225f columns: no per-MX-group scale structure (medians +-2e-4).
- r225g P-ladder: base equidistant from exact/P1/P2/P3 fp64 rungs —
  not a P-rung depth artifact.
- r225h rank: no rank-1/SVD error structure.
- r225i synthetic sweep: kernel BITWISE EXACT vs fp64 model at
  seqlen 1..65536 on random-band synthetics (e8m0 codes [120,134]) —
  the residual lives in what synthetics did not cover.
- r225j surface swaps: swapping ONE real byte surface into the clean
  synthetic window — Ks ALONE reproduces the error (dmax 1.7e-1 /
  4.0e-1 / 1.8e-1, 38-40% of elements off-grid); K/V/Vs clean.
- r225k histograms + decode-law unit check: real Ks codes span
  [0..123] (top: 121x61515, 120x39045..., 119-class tails, 0x10864
  zero/padding rows); decode_kv matches the OCP reference laws
  BITWISE at every code band [0..160] — the model's decode is exact,
  the residual is kernel-side. The synthetic band [120,134] was
  peaked-only coverage (see below).
- r225l split probe (decisive): replay at num_splits_kv
  in {auto, 1, 2, 8} — nsplit1 is EXACT (mism 0-1/3072, max_ulp
  0.50-7.51 = triton class) on all three payloads; auto/2/8 carry the
  full residual (mism 471-814/3072). A FLAT synthetic (Ks pinned 119,
  zero real bytes) reproduces the error at auto (mism 1316/3072) and
  is exact at nsplit1. Peaked control (codes 134, max-p = 1.0) clean.

Mechanism (fix 2's hypothesis, tested and confirmed present but
SUBDOMINANT — the build23 gate residual did not move, see root cause
3): mx_reduce_split.hpp:359 rescaled every KV-split partial with
sycl::native::exp2 — the ~1e-3-relative approximation that r198
evicted from the mainloop softmax. In a PEAKED softmax one split
dominates: the dominant split's rescale is exp2(0) == 1 exactly (even
approximated) and the rest rescale to ~0, so every synthetic sweep and
every prior numerical gate was blind to it. The REAL deep-layer decode
softmax is FLAT (max-p p50 0.046-0.074; Ks codes 119-122 keep logits
small), so every per-split rescale factor is nontrivial and the
approximation error lands on the output under different
numerator/denominator weightings — part of the observed 2^-9.4 uniform
multiplicative signature (bf16 partial quantization, root cause 3, is
the dominant half).

Fix: one call site — `sycl::exp2(local_max_logit - global_max_logits)`
in the fork-owned mx_reduce_split.hpp (r225 delta 3, provenance
comment in-tree; prebak mx_reduce_split.hpp.r225prebak). The epilogue
family's remaining native::exp2 sites are empirically exonerated
(r225l nsplit1 runs the epilogue's reduce_A and is exact; the sink
branch is dead at Sink=false; the chunk-prefill copy is out of the
decode lane's scope) and are left untouched (minimal-diff law).

## Root cause 3 — bf16 split partials: temp_out (r225 fix 3, DOMINANT)

The split-path mechanism r225l isolated is bigger than the combine's
exp2: the multi-split partial-accumulator buffer `temp_out` was
allocated from `query.options()` = **bf16**
(mx_paged_decode_xe2.cpp:252-257, mirroring stock flash_api.cpp's
decode path — a STOCK-INHERITED pattern, not fork-introduced; the
`num_kv_splits == 1` case aliased temp_out = out and stored bf16
too). Each split's unnormalized fp32 O accumulator is quantized to
bf16 — 2^-9 relative — BEFORE the combine divides by the softmax
sum.

That ordering is the whole story, and it explains every observation:

- FLAT softmax (the real deep-layer regime: max-p p50 0.046-0.074,
  sum ~ 5e3): the denominator shrinks |O| ~5000x while the partial's
  absolute bf16 error rides through unchanged — hundreds-thousands of
  output ulp. Observed on the banked payloads: max_ulp 265-4252,
  frac>2ulp 3.5-5.4%, 15-26% of elements off-grid.
- PEAKED softmax (sum ~ 1): sub-ulp adjacent-grid flips only — which
  is why synth134 (max-p = 1.0) was clean and every synthetic gate
  before r225l passed.
- nsplit=1 exact: the single-split epilogue normalizes in-register
  in fp32 (`ElementA(1)/rA_sum` on the fp32 accumulator) and stores
  bf16 once — no partial quantization anywhere.
- Triton lane exact: fp32 partials throughout.
- Fix 2 changed nothing measurable: bf16 quantization (2^-9) and
  native::exp2 (~1e-3) are the same error class; both land on the
  rescaled partial, and the larger dominates the max.

Fix (r225 delta 4, three fork-owned files, apply script
tools/ar/r225_fix3_apply.py; prebaks *.r225fix2prebak):

- mx_paged_decode_xe2.cpp — temp_out ALWAYS a dedicated fp32 buffer
  (`query.options().dtype(at::kFloat)`; the splits==1 alias removed).
- mx_paged_decode.hpp — O element roles split: `ElementOAccum =
  float` builds TensorO (the decode kernel's O write = the partials),
  the reduce_arg's args.out cast pinned to bfloat16_t.
- mx_reduce_split.hpp — `ElementOFinal = bfloat16_t` for the final O
  write; Oaccum follows the decode kernel's now-fp32 ElementO.

The reduce kernel's combine is now the ONE bf16 quantization point:
fp32 partials in, fp32 rescale-accumulate with precise exp2, final
cast out. The combine always launches (the old call site skipped it
at num_kv_splits == 1); a one-split sequence rides the epilogue's
sentinel pass-through (exp_sums = 1, max_logits = 0 → rescale =
exp2(0) = 1, inv = 1), which is bitwise-identical to the old
single-split numerics: fp32 normalize + one round-to-nearest-even
bf16 store. Memory cost: the fp32 temp at engine decode shape
[4, 12*splits, 256] is ~3 MB — nothing. Wheel lineage: cert26
62f234c8 → fix1 d66e9255 → fix2 ebc5845b (preserved .r225fix2bak) →
build24.

## Root cause 4 hunt — resolved: NO kernel defect (gate-reference q-cast mismatch)

Fixes 1-3 (build24) collapsed the residual to the bf16 grid floor in
every replica harness (r225l ladder auto/2/8 clean; r225o page matrix;
r225p order probe), yet the gate itself kept FAILING — gate3 AND a
verbatim gate4 re-run reproduced native-vs-oracle
7.812e-3 / 5.078e-2 / 5.859e-3 identically (fp64 arbitration
7.685e-3 / 5.158e-2 / 5.999e-3). Deterministic on both sides: a real
harness delta.

r225q (order bisect, r225q_bisect.log) pinned it: 8 cells — verbatim
solo, 2447.18-then-30 (the gate's payload order), minus-triton,
bare, hardcoded-fmt, and freed-cached poison cells at the op's exact
internal buffer sizes — ALL produce byte-identical native output
(md5 51736f834358 on 2447.30) at the fp64 floor (dmax 1.556e-2,
mism 1/3072, max_ulp 0.54). Earlier probes' `poison` cells were
vacuous by construction: they called torch.xpu.empty_cache() after
freeing, releasing the poisoned blocks to the driver before the op
allocated. The kernel is stateless w.r.t. allocator history, triton
calls, banked-output loads, and fmt spelling (banked fmt == 'mxk8v4'
exactly; banked q is torch.float16).

r225r (q-cast quantifier, r225r_qcast.log) found the delta in the
REFERENCE, not the kernel: the banked q is fp16; the Triton oracle
eats fp16 q verbatim while the native op eats q.to(bfloat16), and
the two fp64 arbiters inherited the same split (r225b walked
p["q"].double(); r225j walked the bf16 cast). The fp64 walk of the
CAST DIFFERENCE ALONE reproduces the entire gate ladder:

  payload     dmax(walkF16, walkBF16)   gate native-vs-oracle
  2447.18            2.316e-3                  7.812e-3
  2447.30            4.939e-2                  5.078e-2
  2547.18            3.327e-3                  5.859e-3

(the oracle's own fp16-q walk sits at 4.8e-4 / 3.6e-3 / 6.0e-4 —
the gaps compose). 2447.30 is simply ~20x more q-cast-sensitive.
The gate's old criterion — native-vs-(fp16-q oracle) at triton
level — was unattainable BY CONSTRUCTION: it compared two kernels
computing functions of different inputs. No fourth source exists.

## Regression found en route — forced S==1 NaN (r225 fix 5)

r225l on build24 exposed a fix-3-introduced regression: forced
num_splits_kv=1 produced NaN (mism 3072/3072) on every payload.
Delta (4) removed the S==1 temp_out = out alias, making the reduce
the only copy to out — but the epilogue wrote the split LSE only on
`is_single_split || num_kv_splits > 1`, both false at S==1, so the
reduce read unwritten at::empty LSE bytes: garbage/negative exp_sum
skipped the lone partial -> 0/0 -> NaN.

Fix (r225 delta 5, chunk_prefill_epilogue.hpp:570): the sentinel
arm's predicate becomes `is_single_split || num_kv_splits <= 1` —
exactly the predicate the same epilogue uses for O-normalization.
Whenever O is normalized the LSE is the pass-through sentinel
(exp_sums = 1, max_logits = 0): bitwise the old single-split
numerics. Confirmed live: build25 nsplit1 cells sit at the floor
(6.611e-3 / 1.556e-2 / 6.005e-3, nan=0, bitwise-stable), and the
build23-measured near-tie tail (7.51 ulp at 1/3072 on 2447.18)
reproduces EXACTLY — fix5 restored the old single-split numerics
bit-for-bit. The auto heuristic never picks S==1 here, so shipping
behavior is unchanged; the fix closes the forced-split hazard.

## Gate (final) — r225t, q-cast-matched criterion

r225d/gate3/gate4 are preserved as history (r225d_gate3.log,
r225d_gate4.log). The corrected gate (tools/ar/r225t_gate.sh +
r225t_replay.py, r225t_gate.log) applies the honest criterion —
EACH LANE vs THE fp64 WALK OF ITS OWN INPUT — on build25
(refuses: wheel md5 must differ from cert26/fix1/fix2/fix3
lineage 62f234c8 / d66e9255 / ebc5845b / 42d59201):

- TRITON_BITEXACT vs banked engine output: 0.000e+00 on 12/12
  cells (the r224c replay law holds).
- NATIVE bitwise-stable in-process, nan=0: 12/12.
- NATIVE_VS_WALK (bf16-q walk): dmax 6.005e-3..1.556e-2,
  mism 1-2/3072, max_ulp 0.50-10.51 — floor class. The 7.51/10.51
  tails are the pre-registered tie-outlier signature (single
  element, fp32-vs-fp64 near-ties across split boundaries;
  allowance frac_gt2 <= 1e-3 AND max_ulp <= 16 was written into
  this VERDICT's gate plan BEFORE the run, from build23's r225l
  measurements; every other band — dmax, mism, stability,
  bitexactness — unchanged, and any broader signature still fails).
- QBORACLE_VS_WALK (q-matched oracle: q.to(bf16).to(fp16) — exact
  value carry, same triton autotune key, no sweep): 6.05e-4..3.62e-3
  — the triton lane's own floor on the native's input.
- NATIVE_VS_QMATCHED_ORACLE: 7.812e-3..1.556e-2 — both lanes agree
  on the same input at their respective quantization floors.

Verdict: R225T_GATE: PASS (12/12 cells: {base, nsplit1, nsplit2,
nsplit8} x 3 payloads; wheel_md5=aeca45daa740e9973a1c51bfc5ea2bc4).
Wheel lineage: cert26 62f234c8 -> fix1 d66e9255 -> fix2 ebc5845b ->
fix3/build24 42d59201 -> fix5/build25 aeca45da (build log
arc7mx_build25_r225.log, VXK_ARC7MX_CLEAN=0 resume).

## Coverage notes

- The combine fix serves mxfp4 / mxfp8 / mxk8v4 identically (same
  ReduceSplitK path); the payload-replay gate class extends to
  mxfp4/mxfp8 banks at the next cert battery (cert27) per the
  improvement-convergence law.
- The STOCK ReduceSplitK (fp8/bf16 lanes) still carries native::exp2
  in its combine, and stock flash_api's decode path allocates its
  tmp_out from q.options() (bf16 partials — the same error class this
  leg evicted) — upstream's code, out of fork scope, noted here as an
  observation for any future upstream report.
- r225l's peaked control (synth134-auto) shows a bounded near-tie
  artifact class (dmax 2.0 on tiny-|model| elements, big elements
  grid-exact at 0.50 ulp): fp32-vs-fp64 argmax tie-breaking when two
  logits tie across splits. Not a defect; real data has no exact ties.
