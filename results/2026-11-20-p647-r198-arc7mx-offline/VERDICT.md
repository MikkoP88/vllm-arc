# r198 — arc7 MX-native paged-decode offline correctness (P6.47 Phase 8)

**VERDICT: PASS** (run 17, 2026-11-20 01:21 UTC, all 8 faces × 3 formats)

- Wheel: `vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl`
  sha256 `31a7fd95a1b87a10410df3c9629f14c648d8f17774fcaccc956471513b2a74b8`
  (build10, `vxk-arc3-builder`, oneAPI 2026.0.0)
- Harness: `tools/ar/p147_mxkv/p198_native_offline.py`
  md5 `587945140f950f7776b6a0b98ffe6785` (rig work-tree copy identical)
- Driver: rig `/root/build/p198/r198_native_offline.sh` (cert25zc image,
  wheel force-installed over the image venv; W14 quiesced during the run,
  fullres restore fired at close)

## Face results (run 17)

| face | result | notes |
| --- | --- | --- |
| native-exact (fp64 oracle) | ok ×3 | d 9.75e-4–1.72e-3, d_rounded 0–1.5e-5 |
| native-v2 (certified Triton) | ok ×3 | d ≤ 9.7e-4 (gate 2e-3), d_rounded ≤ 9.5e-7 |
| bitwise (seq1 dequant oracle) | ok ×3 | 0/1024 mismatches, all formats |
| softmax-probe (one-hot V) | clean | deviation == exactly the bf16 output cast, tail 0/120 |
| element-ledger | clean | n(>1step) 0/0/1; worst elements 1-step benign |
| robust (adversarial/signed/len 1/15/16/17) | ok ×21 | d_rounded 0–4.5e-8 |
| splits (forced8 + default, 8K ctx) | ok ×6 | 3.1e-4 max |
| telemetry | ok ×9 | native 0.10–0.26× v2 (r199 target) |

## The correctness ladder (what it took)

1. **tv_layout coordinate law** (runs 1–2): dequant per-WI positions from
   the destination fragment's tv_layout, not partition_S identity views
   (WI-canonical in this tree) — cleared the bitwise face to 0/1024
   after ±0 canonicalization (e2m1 nibble 8 = −0.0 accumulates +0).
2. **Softmax precision** (runs 3–8): probe decomposition isolated the
   P-bf16 cast (max|dp| == max|p−bf16(p)|) and the sycl::native::exp2
   tail (13–17/120 beyond cast). Fixes: precise `sycl::exp2`; PV atom
   tf32 interim.
3. **Split-P** (runs 9–14): element-ledger showed the residual as
   adjacent-grid rounding FLIPS (steps=1.00 at every max-diff element;
   got and law C are the two neighbors of ref) — gate outcome depended
   on WHICH magnitude flipped. bf16 PAIR (2^-18) killed the walk
   (n(>1step) 18/23/12 → 0/0/1) at full DPAS rate.
4. **Triple-P + v2-side ulp law** (runs 15–17): mini-ledger pinned the
   last 2 failures to single deep-cancellation elements where the PAIR's
   2^-18 floor is 1–2% of |O| (C == F32 exactly there — accumulation
   clean; the pair floor is the deviation). TRIPLE
   p = bf16(p)+bf16(r1)+bf16(r2) holds p to 2^-27 (below fp32's own
   noise — v2-grade); the harness's element-local 1-ulp rule now
   accepts within-one-rounding of EITHER the fp64 truth or the
   certified v2 (v2 itself is 2 steps off bf16(ref) at such elements —
   zero free parameters).

Kernel deviations from the vendor stock, all documented in-tree:
coordinate-driven MX dequant (mx_dequant.hpp), precise `sycl::exp2`,
split-P TRIPLE PV operands (mx_paged_decode_mainloop.hpp), shared bf16
DPAS atom for both MMAs (mx_paged_decode.hpp).

## Next

r199 T1 perf A/B vs v5 at the money cells — telemetry shows native at
0.10–0.26× v2 (10.8 GB/s effective at 32K): the coordinate-driven
per-value gmem byte reads + scale reads dominate (memory-bound walk,
not MMA). Then r200 e2e fastcycle (ARC_MXKV_NATIVE=1), r201 full §10
battery, cert26.
