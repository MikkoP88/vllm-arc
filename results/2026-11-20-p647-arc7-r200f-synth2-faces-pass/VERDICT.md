# r200f — arc7 MX-native decode: correctness CLOSED on r200d (synth-v2 + 42/42 faces)

Date: 2026-11-20 · Leg: P6.47 Phase 8 (arc7_mx_attn) · Wheel: r200d
(`vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl`, md5
947ea4e2...) · Branch: feat/arc-p647-arc7-mx-native · Rig batch (W14 kept
down, R198_KEEP_DOWN / R200C_KEEP_DOWN).

## VERDICT: PASS — kernel correctness closed; perf frontier opens

1. **C-stage exonerated.** The r200e "non-affine tv coordinate law" was two
   instrumentation contaminants, not a kernel defect:
   - *Two-WG dump race*: HQ=2 launches two workgroups; both write the debug
     buffer, and head-1's v1 synth scales were +1 — a global 2x on every
     dumped value (`got == 2.0 * want` exactly).
   - *Mod-125 alphabet*: v1's payload byte `f(k,j) = (k*128+j) % 125`
     collides across rows (`byte(k,j) == byte(k-dk, j-3*dk)`) — the value ->
     (k,v) inversion fabricated strides and a bilinear cross-term.
   synth-v2 (byte injective per position j with 126/254 as the even
   complements of 0x7F/0xFF; scale 96+k*4+g injective per (row,group); both
   heads identical so the race is benign) returns **C 0/2048 on BOTH
   formats** (`p200f_synth2_*.json`): the assumed form-(ii) law
   `v = t1 + 8*v0 + 16*v2, k = t0 + 2*v1` is exactly right.
2. **Faces 42/42 PASS on r200d** (`r198_summary.txt`, `r200f_faces.log`):
   bitwise 0/1024 on all three formats (e2m1-V included — the r200b-era
   cos~0.87 failure is fixed by the r200d staging redesign), native-exact
   (fp64), native-v2, robust (adversarial/signed/len1/15/16/17), splits
   (forced8 + default).
3. **Probe matrix (stage attribution, 32K device ms)** — `p200c_probe_*.json`:

   | mode | mxfp8 | mxfp4 | mxk8v4 |
   |---|---|---|---|
   | 0 full | 6.72 | 5.64 | 6.43 |
   | 1 skip staging | 6.44 | 5.41 | 6.21 |
   | 2 const-V | 3.53 | 2.61 | 3.45 |
   | 3 floor | 3.31 | 2.49 | 3.36 |
   | 4 no barrier | 6.72 | 5.63 | 6.43 |
   | 5 no scales | 4.87 | 3.85 | 4.64 |

   Attribution: staging ~0.2-0.3; V gather+dequant ~3.0-3.2 of which the
   **scale path ~1.8-1.9** (4-entry cache thrashing at 8 keys/WI: rows
   k = t0+2*v1, group constant per VV); barrier ~0. v2 reference at 32K:
   0.768 / 0.660 / 1.572 ms — the floor (K lane + gemms + softmax, 3.3 ms)
   is itself ~4.3x v2, so the loop structure is the deep frontier after the
   scale fix (r200g: direct-indexed per-VV scale vector, no scan).

## Laws banked
- **Pitch law** (r200d correlator): 2D-block copy row pitch is encoded in
  16-byte units; pitch 8 mod 16 (odd HK x 264/200-byte slots) drifts +8
  bytes/row (row 0 exact, row k = cache[v_off + (slot+8)*k]). Production
  (even HK) and stock dtype slots always aligned; host guard lands in r200g.
- **Debug-dump attribution contamination** (memory
  debug-dump-attribution-contamination): injective-per-attribute alphabets +
  identical fills across racing WGs, or the inversion fabricates laws.
- r200c "width-16 loads row 0 only" — RETRACTED (width-independent pitch
  drift at HK=1; r200c/r200d A-images were byte-identical).

## Next (r200g, in flight)
dequant_vslm scale vector (sc[(i>>1)&7], 8 tv-derived rows, loads issued
together) + pitch-law TORCH_CHECK + rewritten law comments; then faces +
synth-v2 + probes re-run; then the 3.3 ms floor (K-lane/loop structure).
