# r200g — arc7 MX-native decode: direct-indexed V-scale vector (perf step 1)

Date: 2026-11-20 · Leg: P6.47 Phase 8 (arc7_mx_attn) · Wheel: r200g
(`vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl`, md5
15b98a29, archived `fix/...+arc7mx-r200g.whl`) · Branch:
feat/arc-p647-arc7-mx-native · Rig batch (W14 kept down).

## VERDICT: PASS — 32K decode -1.2..-1.9 ms; scale path solved; floor next

Change: `dequant_vslm` scale handling replaced the 4-entry scanned cache
(permanent ~50% miss at 8 keys/WI, r200f probe-5 cost 1.8-1.9 ms) with a
direct-indexed per-VV scale VECTOR: 8 rows read through `tv_coord` at even
registers (`kr[r] == k_loc(2r)` under the pinned form-(ii) law), scale codes
loaded together upfront (`sc[r]`, group constant within a VV), values applied
as `val * sc[(i>>1)&7]` — zero scan, zero eviction, no per-value scale
arithmetic. Same wheel carries the pitch-law TORCH_CHECK (k/v seq strides
% 16) and the rewritten r200c/r200d law comments (retraction recorded).

Gates:
- synth-v2 both formats: **ALL_STAGES_OK, C 0/2048** (`p200g_synth2_*.json`)
  — the direct index is validated by per-row-unique scale codes.
- faces: **42/42 PASS**, bitwise 0/1024 ×3 (`r200g_faces.log`,
  `r198_summary.txt`).
- probe matrix (`p200c_probe_*.json`, 32K device ms):

  | mode | mxfp8 | mxfp4 | mxk8v4 |
  |---|---|---|---|
  | full | 4.76 | 3.67 | 4.39 |
  | skip staging | 4.38 | 3.44 | 4.17 |
  | const-V | 3.46 | 2.49 | 3.32 |
  | floor | 3.16 | 2.38 | 3.20 |
  | no barrier | 4.75 | 3.67 | 4.39 |
  | no scales | 4.73 | 3.65 | 4.37 |

  Scale path: 1.8-1.9 → ~0.03 ms. Gather+dequant: 3.0-3.2 → 1.1-1.3
  (decode-only now). Telemetry face (harness geometry): 32K native
  5.548/3.764/4.573 vs v2 0.770/0.660/1.577 = 0.14x/0.18x/0.34x (was
  0.11x/0.11x/0.23x on r200d). mxfp4 and mxk8v4 now beat the old r199b
  numbers outright; mxfp8 is at r199b parity with correctness closed.

## Next bottleneck (always-loop)
The FLOOR (3.16/2.38/3.20 = 62-72% of total): K lane complete (staging +
reg dequant) + QK/PV gemms + softmax + epilogue over 2048 k-tiles — no probe
modes exist for its internals yet. Ladder: (a) K/softmax/gemm probe modes
(extend vlane_probe), (b) double-buffered staging across k-tiles (r200c plan
item — collapses the 1K-8K latency plateau too), (c) e4m3 decode slice
(~1.3 ms): cutlass float_e4m3_t -> float is software per-byte; a bitcast
u32 law or LUT could halve it. Stop condition: superiority vs v2, then
r200-e2e fastcycle (ARC_MXKV_NATIVE=1) -> full battery -> cert26.
