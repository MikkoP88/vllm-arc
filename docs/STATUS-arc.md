# vllm-arc — status & release ledger

Detail holder for the README front page (moved there 2026-11-06, P6.24d):
the certified-release detail, the full phase narrative (P0–P6.26), and the
improvement-lane disposition. The README keeps the headline + a link; this
file keeps the story. Nothing here is separate evidence — every claim
traces to the banked `results/` legs and the two ledger docs
([`TESTING-arc.md`](TESTING-arc.md) — validation record,
[`MIGRATION-arc.md`](MIGRATION-arc.md) — evidence table).

**Currency: `rel/v0.30.0-b70-cert23` (`vllm-arc:v0.30.0-b70-cert23`).**
This file rides the main-currency law: at every certification close-out
the README anchors repoint (checklist in `tools/check_currency.sh`) and
this ledger's release section is rewritten to the new cert in the same
change.

## Certified release — `rel/v0.30.0-b70-cert23`

Every gate GREEN with banked evidence. cert23 = **cert22 + the P6.35
graph-exec double-buffer** (r135 five-leg gate + r135b control; one
patch module, no wheel change, no serve-line var — the engine bytes
elsewhere are cert22's):

- **The defect it removes** — torch 2.13 `XPUGraph.replay()` on the
  SAME exec object blocks host-side until that exec's prior launch
  retires (p135i microbench: 13.96 ms same-graph in-flight vs 8–12 µs
  for two execs sharing one capture pool alternating). The fix
  (`vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`) captures every
  graph ×2 into the manager's shared pool and alternates replays per
  desc (swap-technique wrappers; second-capture failure degrades to
  stock semantics with a WARN — never a boot death). Leg J: replay max
  24 → 0.37 ms, engine serial work 40.8 → 12.7 ms/step, alternation
  proven (12 XRP ids × 59 over 119 steps).
- **E2E-honest position** — the gate A/B reads **NEUTRAL at every
  measured posture**: fp8-k3 94.5 (db on) vs 94.3 tok/s (db off),
  ITL p50 40.49 both; mx4k5 121.0 vs 123.9 (sweep). Leg-F's spiky
  81.6 "stock baseline" was ~3.8 ms/step instrumentation tax — the
  clean engine's step was never replay-limited (the block overlaps
  the device-floor work the step waits out anyway; leg K: the
  copy_event wait waits out real device work). Ships as a **costless
  driver-level hygiene fix**: +29 s boot (the ×2 captures), zero
  extra memory (shared pool), byte-identical semantics.
- **Knob** — `VLLM_ARC_GRAPH_EXEC_DB` (default `1` = ACTIVE; `=0` =
  stock single-exec replays — kill-switch proven in the gate: dboff2
  boot shows zero db markers and the stock face).
- **The P6.35 performance deliverable is POSTURE, not engine** (all on
  stock cert22-class bytes; serve-line choices on cert23): spec depth
  **k5 = +41% decode** (115.2 tok/s solo U8k vs 81.6 at k3); **mxfp4
  W4A16 + k5 = 123.9 tok/s (+52%) with TTFT 3.98 s < the 4.55 s
  baseline** — the recommended decode posture (`ARC_QUANT=mxfp4
  ARC_MXFP4_MODE=w4a16 ARC_QUANTCFG='{"ignore":["*visual*",
  "*linear_attn*","mtp.fc"]}' ARC_GMU=0.85 ARC_SPEC=5 ARC_FP8=0`, BASE
  checkpoint). mxfp8 W8A16 is a decode dud (83.8 ≈ baseline — no
  weight savings, bf16-activation tax). Rate accounting corrected:
  spec-decode SSE streams multi-token chunks (600 tokens/127 chunks at
  fp8k4) — honest numerator is max_tokens. Evidence:
  `results/2026-11-10-p635-telemetry/` (legs A–K + `sweep/` +
  `cert23/`).
- **P6.35-V full validation matrix (cert23, r136+r136b)** — 11 cells,
  both models, all online quants × spec depths, uniform protocol, 0
  tracebacks, db markers correct at every depth. Qwen: mx4-W4A16 wins
  every spec column — **k5 re-read 128.3 tok/s (+36% vs the clean
  fp8-k3 base 94.5, project best; 5.93 tok/chunk acceptance)**;
  spec0 column fp8 32.8 / mx8 28.2 / mx4 36.1. Gemma (spec0): **mx4
  37.1 tok/s = +43% vs mxfp8 26.0** (ITL 27.0 vs 38.5 ms — the W4A16
  weight-streaming win is model-agnostic); **online fp8 boots + serves
  (26.6) — new capability**; mxfp8 26.0 repeatable. mxfp8 deprecated
  for decode (below fp8 and mx4 at every depth, both models). TTFT:
  fp8-k5 3.48 s best; mx4 prefill cost bounded (worst 5.11 s at k5).
  Winner postures: qwen `ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 QCFG-ignore
  GMU0.85 SPEC5 FP8=0` (BASE ckpt); gemma same quant + `GMU0.8 LMC=0
  RPC=1200 MAXLEN=40960`. Evidence: `results/2026-11-10-p635-telemetry/
  vmatrix/` (incident note: 27B container rm >30 s caused a
  stale-record cascade — data banked before every marker; watchdog +
  hardened r136b teardown re-ran the 3 affected cells clean).
- History: cert22 (P6.33 skinny-M, banked OFF), cert21 (P6.31), cert20
  (P6.30), cert19 (P6.29), cert18 (P6.28 FA2), cert17 (P6.28 gemma-4
  chain), cert11 (P6.26), cert10 (P6.24), cert9 (P6.20), cert8
  (P6.19) remain fully valid.

Prior release (the kernel family, banked OFF) — cert22 = **cert21 + the
P6.33 skinny-M kernel family, banked OFF** (r125→r127 + r127g2;
behaviorally ≡ cert21 at serve — the route ships default OFF after the
e2e battery falsified the reroute):

- **The kernels** — the arc6 wheel (`vllm-xpu-kernels 0.1.14.1+arc6`,
  sha256 `484e06c1…`) adds three GEMV-regime M≤8 weight-only GEMM ops
  (`_xpu_C.skinny_fp8_gemm_w8a16` / `skinny_fp8_gemm_w8a16_mxfp8` /
  `skinny_fp4_gemm_w4a16`): one work-item streams TN weight columns
  K-contiguously, dequantizes in registers (f8 bit-identity re-bias,
  e2m1 value set, e8m0 exp2 group scales), accumulates fp32,
  deterministic fixed-order split-K (split law v2.7) — plus W4A8
  dispatch case arms and the v0.1.17 Option-B backports
  #611/#645/#650/#600. Correctness green (r126 GATE-1 rel < 2e-2 on
  every gate cell; GATE-2 identical-input A/B; GATE-3 dispatch).
  Sources `external_patches/vllm-xpu-kernels/arc6_skinny_v017bp/`.
- **The falsification (r127, five legs)** — the eager-regime kbench
  wins (w16pc/g16 6/6 at M=1, 0.80–0.92× stock) did NOT survive the
  serve path: the certified decode replays FULL_DECODE_ONLY XPU
  graphs, and under graph replay stock oneDNN runs its tensor-core
  gemmstone kernels at **420–680 GB/s device-side** (r127f 50-call
  amortized graph probe) — the ~197 GB/s "stock level" that motivated
  the kernel was an eager-launch-bound artifact (host dispatch
  ~0.15–0.3 ms/call caps every op eagerly; the round-interleaved
  kbench compared host rates). The skinny scalar-FMA streaming design
  (ceiling ~300 GB/s) loses 1.6–2.6× per call recorded into the
  replayed graph, and the size-1 decode graph pads the token dim to 2
  — so even M=1-capped routing executes as M=2 in-graph. Gemma solo
  decode regressed **−38% (m816, 17.1 vs 27.7) / −42% (g16, 23.3 vs
  40.5)**; qwen legs inert by construction (MTP k3 target decode =
  always M=4·B verify batches, over every cap); agg8/TTFT/spec
  exactly on ref everywhere. Causality proven by r127c EXP-1
  (`ARC_SKINNY_M=off` on the same image → 27.7/192.3 exact = the
  r124b ref).
- **Posture** — `ARC_SKINNY_M` unset/off = stock routes = the
  **certified default**; `=on` opts in (per-lane M caps w16 M=1,
  m816/g16 M≤2; eager-regime wins only). Serve behavior byte-identical
  to cert21; the kernels stay in the tree, correct and available. A
  tensor-core (joint-matrix) skinny family is the one open lane — the
  r127f probe is the required gate; eager-loop wins are host-rate
  illusions on this stack.
- **Image validation (r127g2)** — the cert leg on the cert22 content
  image (FORK_SHA e2457389): default posture, gemma m816 → solo
  **27.7 / agg8 192.9 tok/s** = the exact r124b reference values, zero
  route markers in the serve log; the installed
  `patches/skinny_m.py` md5-verified (`61ff347a…` = repo HEAD). (The
  first r127g run caught a stale-flat-file ctx22 build — the flat-COPY
  trap is banked in the plan's P6.33 execution log.) Evidence:
  `results/2026-11-08-p633-skinny-m/`.
- History: cert21 (P6.31 W8A16 route), cert20 (P6.30 W4A16 route),
  cert19 (P6.29 slide route), cert18 (P6.28 FA2 route), cert17
  (P6.28 gemma-4 validation chain), cert11 (P6.26 MXFP4 lane), cert10
  (P6.24 oneCCL fp8-wire overlay), cert9 (P6.20 prefill host-path),
  cert8 (P6.19 TQ fix) remain fully valid.

Prior release (the 8-bit online lane) — cert21 = **cert20 + the P6.31
online-MXFP8 W8A16 route** (r122→r125; the online-mxfp8 lane given the
same act-quant-elimination fix the mxfp4 lane got at P6.30):

- **The route** — online `ARC_QUANT=mxfp8` on a BASE bf16 checkpoint
  keeps its 8-bit e4m3 weights but serves GEMMs as **W8A16**: bf16
  activations × f8 weights, group-32 e8m0 weight scales, zero per-call
  activation quant, via the arc5 wheel's
  `_xpu_C.fp8_gemm_w8a16_mxfp8`
  (`external_patches/vllm-xpu-kernels/arc5_mxfp8_lanes/` build kit +
  `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` + the platform
  `_XPU_PATCH_MODULES` entry, via `tools/cert21.Dockerfile`; content
  `c8fdf32a`). Stock online mxfp8 is W8A8 — per-call activation quant
  ≈ the GEMM at decode Ms AND the (f8,f8) joint lane measured 2.05–2.64×
  slower than bf16 at prefill (r122: every scale flavor; the joint
  rate, not the flavor, is the wall — W8A8-grpw designed then dropped
  as strictly dominated). Knob `ARC_MXFP8_MODE` (unset/`w8a16` =
  route — default; `w8a8` = stock control).
- **The scorecard (same-image A/B, r124/r124b, both target models)** —
  **qwen: U8k 3.61 s / U32k 17.07 s / decode 53.8–55.6 solo / 265.9
  agg8 tok/s = 1.37–1.55× over stock W8A8 on every face**, spec pos1
  0.9804 vs 0.9758 (control) — spec-decode healthy, no numerics
  regression. **gemma: 3.87 / 20.94 s, 27.7 / 192.4 tok/s =
  1.57–1.85× over stock** (r124b at `ARC_MAXLEN=40960`).
- **Adjudication (pre-registered)** — the r123 kernel matrix predicted
  the op lands BETWEEN stock and the checkpoint class: m816 GEMM
  1.74×/2.68× faster than stock mxfp8 GEMM at decode/prefill medians,
  1.50×/1.06× behind the w16pc checkpoint-fp8 class (rel 2e-3). e2e
  confirms: **checkpoint-fp8 stays the perf default** (qwen solo 84.5
  vs 55.6; gemma 28.4 vs 27.7), **`ARC_MXFP8_MODE=w8a16` is the
  online-mxfp8 lane route**, and **mxfp4-w4a16 (cert20) stays the
  superior online lane** where 4-bit quality is acceptable (qwen solo
  62.5–65.2 vs 55.6; gemma 40.5 vs 27.7). The **W4A8 (e4m3×f4) probe
  is a banked closed-negative**: the arc5 op exists but oneDNN
  primitive-create refuses the joint dtype ("Only support int4 and fp8
  gemm", r123 all shapes) — the arc5test mis-attribution resolved: the
  wrapper's own `default:` threw before oneDNN was asked.
- **Gemma capacity law** — online-mxfp8 weights on the 31B model are
  ~29 GiB under TP2 (2× mxfp4): at gmu 0.8 only 5.31 GiB KV remains —
  short of one 262144-token seq (8.33 GiB needed; estimated max len
  103296), so both r124 gemma legs BOOT_FAILED at the capacity gate
  (NOT a lane defect; tracebacks banked). r124b re-ran at
  `ARC_MAXLEN=40960` (the battery's longest probe = the U32k face at
  32 800 tokens; admission-only — faces/decode comparable with the
  banked 262144 refs). Serve-script knob `ARC_MAXLEN` documented in
  CONFIGURATION-arc.
- **Image validation (r125, three-lane gate)** — build gate green
  (Dockerfile asserts + `CERT21_W8A16_LIVE`, manifest rows incl.
  `cert21_fork_sha=c8fdf32a…`/`cert21_vxk_wheel_sha256`); leg A qwen
  fp8 discriminator in-band = the champion lane untouched by the wheel
  swap (spec pos1 0.9258 byte-exact incumbent); leg B the w8a16 lane —
  `reroute ACTIVE` marker, dispatch
  proofs, battery + spec green vs the r124 band; leg C the w4a16 carry
  (updated patch on the arc5 wheel) green vs the r121 refs. The
  original verdict print FAILED VACUOUSLY (verdict python never cd $G —
  inherited the launch CWD, every leg read missed); the identical
  block re-run over the untouched banked artifacts reads PASS (r105b
  precedent; recompute + both runner fixes disclosed in the results
  README). Evidence:
  `results/2026-11-08-p631-mxfp8-w8a16/cert21_l1/`.
- History: cert20 (P6.30 W4A16 route), cert19 (P6.29 slide route),
  cert18 (P6.28 FA2 route), cert17 (P6.28 gemma-4 validation chain),
  cert11 (P6.26 MXFP4 lane), cert10 (P6.24 oneCCL fp8-wire overlay),
  cert9 (P6.20 prefill host-path), cert8 (P6.19 TQ fix) remain fully
  valid.

Prior release (the online-quant fix layer) — cert20 = **cert19 + the P6.30
online-MXFP4 W4A16 route** (r120c→r121; the online-quant
campaign + the mxfp4-perf MAJOR ISSUE FIXED):

- **The fix** — online `ARC_QUANT=mxfp4` on a BASE bf16 checkpoint
  reroutes every standard linear to the arc4 wheel's
  `_xpu_C.fp4_gemm_w4a16` (bf16 activations × f4 weights, group-32
  e8m0 weight scales → oneDNN `jit:gemm` DPAS via weight decomposition;
  `external_patches/vllm-xpu-kernels/` build kit +
  `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` + the platform
  `_XPU_PATCH_MODULES` entry, via `tools/cert20.Dockerfile`; content
  `a918bee9`). The r113→r120d diagnostic chain root-caused stock W4A4
  twice over: per-call activation quant ≈ the GEMM cost at decode Ms,
  and the (f4,f4) joint dtype can never pass oneDNN gemmstone's JIT
  gate (the counter-operand list has no f4) → `matmul:ref`,
  structurally. W4A16 drops both; verbose-proven `jit:gemm`, numerics
  rel 1.66e-3.
- **The scorecard (same-image A/B, r120c, qwen3.8-27b certified
  posture)** — **U8k 3.66 s / U32k 17.12 s / decode 62.5–65.2 solo /
  301–311 agg8 tok/s: SUPERIOR to online mxfp8 on every axis
  (1.47–1.75×)** — the mxfp4-vs-fp8 superiority requirement met — plus
  1.74–1.87× TTFT / 1.43–1.45× decode over the W4A4 control and spec
  pos1 1.0000. Gemma (r120g): 3.94 / 21.16 s, 40.5 / 262 tok/s =
  1.90–2.21× over W4A4; faster than gemma's plain-fp8 online record on
  3 of 4 axes (U32k the honest exception — the bf16-DPAS-par physics
  ceiling at prefill M=32768, where Xe2 has no fp4 MMA).
- **Kernel-level closure (r120d)** — w4a16 is the FASTEST lane at every
  decode M (0.17–0.25 ms — 2.53× faster than w4a4-apply, ~2× the
  e8m0-fp8 GEMM, 3–4× bf16); at prefill 2.1–2.7× faster than the
  e8m0-fp8 GEMM and 3.0–3.8× faster than w4a4 (ref kernel 108 ms at
  gemma.up). The v3.13 online-fp8 question closed mechanistically: its
  GEMM rides `jit:gemm` with bidirectional e8m0 group scales (NOT a ref
  fallback) yet stays 2.6–2.8× slower than bf16 — group-scale handling
  inside the DPAS kernel; mxfp8-online stays the slow lane. The qwen
  plain-fp8 online lane is an upstream interface dead-end (no `fp8`
  online shorthand; QCFG needs online targets — r120h traceback +
  r120h2 `fp8_per_channel` probe banked).
- **Engineering laws banked** — (1) `import vllm_xpu_kernels` does NOT
  dlopen `_xpu_C` (explicit submodule import before op-presence probes;
  `vllm/_xpu_ops.py` order-sensitive corollary); (2) ≤24 icpx jobs when
  a W14 window is live; (3) pip validates wheel FILENAMES — install
  under the real name in image Dockerfiles; (4) gemma legs: `ARC_LMC=0`
  + `ARC_RPC_TIMEOUT=1200` + dir `/models/gemma4-31b` (no dash — a
  wrong path auto-creates an EMPTY host dir).
- **Image validation (r121)** — build gate green (Dockerfile asserts +
  `CERT20_W4A16_LIVE`, manifest rows incl.
  `cert20_fork_sha=a918bee9…`/`cert20_vxk_wheel_sha256`); leg A qwen
  fp8 discriminator (default posture) in-band = incumbent lanes
  untouched; leg B the w4a16 lane — `reroute ACTIVE` marker, dispatch
  proofs, G1 + spot battery green, erraudit 0/0. Evidence:
  `results/2026-11-08-p630-mxfp4-perf/`.
- History: cert19 (P6.29 slide route), cert18 (P6.28 FA2 route),
  cert17 (P6.28 gemma-4 validation chain), cert11 (P6.26 MXFP4 lane),
  cert10 (P6.24 oneCCL fp8-wire overlay), cert9 (P6.20 prefill
  host-path), cert8 (P6.19 TQ fix) remain fully valid.

Prior release (layer 3 fix) — cert19 = **cert18 + the P6.29 sliding-window FA2 route** (r111; the remaining-bottleneck campaign — the gemma-4 prefill/decode wall layer 3 FIXED):

- **The fix** — sliding-window head-256 calls (gemma-4's 50/60 local layers, window (1023,0)) dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel WINDOWED in the same triton unified-attention wrapper (`external_patches/vllm/patch_arc_xpu_fa2_slide_route.py` via `tools/cert19.Dockerfile`; content `904a3de8`; the patcher asserts the cert18 v3 state byte-exact). r109a/r109a3 measured the class at production geometry: triton 0.147–0.158 s/call at q8192 fp8-KV and **k-INDEPENDENT** (0.1466 k8192 / 0.1580 k32k / 0.1576 k98k — the kernel never amortizes its window pruning across K) vs FA2 windowed 0.006–0.009 s = **18–25×**, decode S8×32k 4.38→0.3–0.4 ms = 11–15×.
- **Numeric guards (the ref-bug closure)** — kernels agree 1.2e-4…9.8e-4; vs global-softmax table-faithful window-masked GQA torch refs both kernels match at 1.3e-4…1.1e-3 prefill / 8e-5 decode (r109a6). The r108ad…r109a4 "~4 ref misses" (full class too, all conventions) were a **per-chunk-softmax bug in the TEST refs** — not a kernel issue; the softcap theory was eliminated independently (the triton wrapper cannot receive `None` — TypeError, r109a5; production passes 0.0 and `USE_SOFTCAP=(0.0>0)` skips it).
- **Image validation (r111)** — build gate green (slide marks 2/2, fa2 marks 3/3, cert12 table marks 3/3 still live, manifest rows incl. `cert19_fork_sha=904a3de8`); gemma L1: warm gate 32/32, BOTH markers live per rank (`[ARC-P628-FA2]` + `[ARC-P629-SLIDE]`), probe98 5 s clean, G1 3/3, faces **4.9 / 17.2 / 110.9 / 0.9 s** (face32k = **qwen-parity**, 1.016× of the U32k 16.93 record; the face120k saving 70.6 s matches the 7.4 s × ~9.5 chunks projection exactly), decode **28.4 solo / 195 agg8 tok/s** (+45 % / 2.64× over cert18), erraudit 0/0, JIT 0.
- **Spec parity dispositioned (r110, closed)** — gemma-4 ships no MTP/draft head (nothing to draft from), and the draft-free ngram lane dies at first request in the upstream v1 spec path (`_get_mamba_bufs()` → `AssertionError: no mamba layers in the model` on pure-attention hybrids — qwen3.8's GDN layers satisfy it, gemma-4 does not; banked `results/2026-11-07-p629/cert19_l1/ngram_crash_excerpt.txt`). Spec parity is not reachable by any lawful lane; certified decode posture stays `ARC_SPEC=0`.
- **qwen3.8 discriminator** — U8k 3.83 / U32k 17.58 s vs the 3.76/16.93 cert11-era records (inside the ±5 % cross-boot band), erraudit 0/0; fa2 marker 0; the slide marker's 2 lines = the guarded sliding-hd256 class also occurs in qwen traffic/tune-warmup — in-band, no regression.
- History: cert18 (P6.28 FA2 route, the first fix layer), cert17 (P6.28 gemma-4 validation chain), cert11 (P6.26 MXFP4 lane), cert10 (P6.24 oneCCL fp8-wire overlay), cert9 (P6.20 prefill host-path), cert8 (P6.19 TQ fix) remain fully valid.

Prior release (first fix layer) — cert18 = **cert17 + the P6.28 FA2 route** (r108ae; the prefill/decode/TTFT MAJOR ISSUE FIXED):

- **The fix** — full-attention head-512 calls (gemma-4's 10 global layers) dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel in the triton unified-attention wrapper (`external_patches/vllm/patch_arc_xpu_fa2_route.py` v3 via `tools/cert18.Dockerfile`; content `3136adfe`). r108ac/r108ad measured the route at production geometry: 0.3197 s vs 10.81 s per k32k fp8-KV call = **34×** (k98k ~30×; decode 1.5 ms/call), numeric guards green vs a block-table-faithful GQA torch reference. r108ac's fp16-KV control (10.28 s at 2× bytes) also refuted the earlier L2-bandwidth-wall reading — the triton kernel is slow per K-tile iteration; the engine shipped the fast kernel all along, the triton backend just never dispatched to it.
- **Image validation (r108ae)** — build gate green (fa2 marks 3/3, cert12 table marks 3/3 still live, manifest rows); gemma L1: warm gate 32/32, `[ARC-P628-FA2]` marker live on BOTH TP ranks, probe98 13 s, G1 3/3, faces 14.2 / 41.5 / 181.5 / 4.7 s, decode solo 19.6 / agg8 73.8 tok/s, erraudit 0/0. The v2 boot crash (the warm gate's full-storage descale form vs vxk's 1-element-storage assert) was root-caused and fixed in v3 (`_arc_fa2_scalar`) — forensics banked `phaseC/cert18_l1/r108ae_crash_descale.txt`.
- **qwen3.8 discriminator** — U8k 3.77 / U32k 17.49 s vs the 3.76/16.93 cert11-era records (inside the ±5 % cross-boot band), route marker 0, erraudit 0/0: the incumbent qwen lane is untouched (its full-attn goes through its own FA2 backend, not this wrapper).
- History: cert17 (P6.28 gemma-4 validation chain, the pre-fix baseline below), cert11 (P6.26 MXFP4 lane), cert10 (P6.24 oneCCL fp8-wire overlay), cert9 (P6.20 prefill host-path), cert8 (P6.19 TQ fix) remain fully valid.

Prior release (pre-fix baseline) — cert17 = the **cert11 engine + the P6.28 gemma-4 validation chain**:

- **cert12** (commit `a964e047`) — the B70 prefill launch-config table in the triton unified-attention launcher (upstream PR#60418 pattern; head-256/512 keyed points; r108k sweep values — head-512 M64/T32/w8 = 5.53× stock at the engine chunk band) + the engine warmup gate (cert13 `28187a9d`, rank-serialized pre-load of the table points — kills the r108o first-load SpirivUtils wedge).
- **cert16** (commit `5d895fef`) — warm-gate v4: the r108v root cause was a triton cache-key scalar-type split (the warm gate passed softcap as float 0.0 → key `('fp32', None)` while the engine passes int 0 → `('i32', 'D')` — every warm kernel lived in a key space the engine never looked up; the first prefill fresh-compiled ~288 s inside the 300 s execute-model watchdog → EngineCore death → HTTP 500). v4 passes softcap as int 0: warm gate 32/32 load-lines covering the engine key space.
- **cert17** (commit `e5e725b4`) — lmcint kit hardening: the KV-format detector BS floor 256→16 (small-block engines take the byte-true TWO_HS store path) + the fail-loud `[ARC-P628-LMC-KVGATE]` uniform-KV-row boot gate — a two-head-size hybrid like gemma-4 (KV rows `[512, 1024]`) cannot be addressed by single-hidden-size KV_2LTD memory objects; boots now refuse loudly in ~2.5 min instead of wedging at first request (gate-fire proven live on both TP ranks, r108z `gate_fire.txt`).
- **G2 long-prefill freeze: SOLVED in all three layers** (L1 warm-gate cache-key split; L2 the residual probe-time `kernel_unified_attention` compile, ~31 s once, acceptable; L3 the LMCache store wedge — KVGATE + certified posture `ARC_LMC=0`). Layer-2b, the face120k 300.5 s cliff, is the stock `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300` RPC watchdog killing the first late chunk — `ARC_RPC_TIMEOUT=1200` (serve-line knob → the env) is part of the certified gemma posture; face120k 2950.3 s completed with ZERO jit events in 49 min (compute, not compile — r108z).
- **Validation** (r108w→r108z L1 fp8; r108x L2 mxfp8; r108v2/v2b decode): full battery green on both legs (faces + decode + quality probes, erraudit 0/0 everywhere; tables in PERFORMANCE-arc). One flag: the L2 mxfp8 reason-probe arithmetic miss (single probe; the identical prompt answers correctly on L1 fp8) — a broader L2 quality battery is the listed follow-up before any quality-parity claim.
- **Prefill root cause (MAJOR ISSUE — diagnosed r108aa/r108ab, FIXED at cert18)**: gemma-4's 10/60 full-attention hd512 layers ride a structural **L2-bandwidth wall** in the shared triton kernel on B70 — each q-block re-walks its full causal K/V span (~37 GB L2-side reads per 10.8 s call at q=8k/k=32k = 3.4 TB/s effective); BLOCK_M is SLM-capped at 64 by the 128 KB Q tile (M128 launch-fails), and measured time ∝ 1/BLOCK_Q (M16 58.5 s → M32 21.1 s → M64 11.0 s) while TILE/stages/fp8-vs-fp16-KV/nkvh are all noise; upstream HEAD pins the same `(64,32,8,1)` — no lawful newer-fixed shared component exists, and the certified table is measured AT the ceiling (winner 1.01×, numeric guard max_abs 0.00000). The 50/60 sliding-window layers are healthy (0.15 s/call, pruning engaged, cost flat in k). The 8–17× face gap vs the qwen3.8 incumbent is architectural on top of the wall (GDN linear attention vs 10 quadratic hd512 layers). Evidence: `results/2026-11-06-p628-gemma4/phaseC/micro_realshape/` + `micro_sweep/`. Kernel-restructure (SMEM-acc / k-outer q-inner tiling, ~2–4× potential) was documented as research-class — then r108ac's fp16-KV control refuted the wall reading and found the real fix: the engine's own ESVD FA2 kernel at 34× (no kernel rewrite needed — a dispatch route). cert18 ships it; the wall story stands as the diagnosis of WHY the triton path was slow, the FA2 route as the fix. Listed follow-ups: the L2 mxfp8 cert18 re-run + broader L2 quality battery.

History: cert11 (P6.26 MXFP4 lane), cert10 (P6.24 oneCCL fp8-wire overlay — 1.65× at the dominant AR opt-in), cert9 (P6.20 prefill host-path), cert8 (P6.19 TQ fix) remain fully valid.

## Project status — P0–P6.33 complete

All phases closed with evidence — engine hardening (P2), cache lane (P3), certification battery (P4), audits + soak (P5), the P6.1–P6.15 measurement/analysis phases, **P6.16/P6.17 the draft-sampling posture** (probabilistic drafting = the mathematically-correct sampler at temperature, the certified serve-script default since P6.17 — every record face re-measured on that lane), **P6.18 the TurboQuant corruption campaign** (reproduction battery split the corruption into Class B/Class A; interim fix = config-time refusal of every `turboquant_*` dtype, certified as cert7), **P6.19 the definitive TurboQuant FIX (cert8)**: the corruption was root-caused to **three stacked spec-verify defects — none of them the quantizer's precision** — captured verify graphs baking the raw-KV flash fast-path (RC1), a replay-broken multi-row synthetic-decode dispatch (RC2), and the wheel's two TQ ops trees (RC3 — the first MQ-kernel draft targeted the never-used SoA tree and read the live cache as zeros). Fix = wheel-side hook `tq_spec_fix.py` (donor-v19b capture forcing + **split-row dispatch**, default-ON) + `tq_mq_aos.py` (MQ verify kernel on the engine's AoS layout, opt-in `VLLM_ARC_TQ_VERIFY_MODE=mq`, +21 % agg8, experimental); P6.18 refusal lifted; **all-dtype gate: zero garble on all four TQ dtypes AND the fp8 control** (identity faces dispositioned with the W14 incumbent control — engine-class probabilistic-drafting faces, not corruption), **P6.20 the prefill host-path lane (cert9)**: end-to-end step telemetry (`ARC_STEP_TRACE`) decomposed the prefill step, the LMCache wait_for_save tail was moved off the critical path (L3) + reordered against idle workers (L6), and the residual U32k/U64k gap was **proven device-compute-bound** (host path clean), then **P6.21/P6.22 the device-compute residue attacked head-to-head and every lane closed NEGATIVE** — ar_quant donor A/B (quality green, perf −4…−9 %, unadopted), the FA headroom ladder (stock vxk at 25 % of the measured 147 TF/s peak but every patch-level route falsified by measurement), the ESIMD dense-FA v2 kernel (numerics-green at **0.33× stock**, ablation-decomposed ceiling bounds best case ~0.83× — under the 2× bar), and the TP2 allreduce one-shot route (P6.22: **stock oneCCL ring 2.06× FASTER than the achievable one-shot ceiling** at the dominant 73.1 MiB prefill AR — closing only the one-shot lane), **P6.23 the docs-currency audit** (12 findings fixed, all six living docs on cert9 truth), and **P6.24 the quantized-AR route taken end-to-end (cert10)** — the one P6.22 route that remained open: same-version oneCCL 2022.0.0 rebuild (ABI/perf-identical, measured) + the e4m3-wire ring patch (u32-packed stores per the sector law, branchless converters, min-count threshold 2M elems, silent-off env trio) gated r96 fidelity → r97 1.65× → r98 task-level quality parity → r99 cert10 release gate (every p65 face beats cert9) → r100 bottleneck scan closing every remaining door with wire-rate measurement; the post-release **P6.24b patch audit** (2026-11-06) reviewed the whole landed lane for in-line-best-practice — three findings, all dispositioned (probe-patch sha chain reconciled to the as-built bytes; the res-scale invariant documented precisely + the frozen-artifact law banked; build-script polish), and **P6.25 the overlap-door audit (2026-11-06)**: the one prefill door no leg had measured — cross-stream AR∥GEMM co-tenancy (async-TP class) — closed MEASURED with the r101 microbench on real shapes (overlap slower than serial at every k on both wires: best 0.969× stock / 0.966× fp8; concurrent AR inflates GEMM latency 1.97×/1.58× — the device time-slices AR and GEMM kernels, no idle-EU lane; census bonus: layer GEMMs at ~99 % of the 147–150 TF/s per-card dense ceiling) — no engine delta, **cert10 stands** (`results/2026-11-06-p625-overlap-door/`), and **P6.26 the MXFP4 validation campaign (2026-11-06)**: both pre-quantized MXFP4 checkpoints tested end-to-end — the quark leg failed config validation exactly as the pre-boot wheel recon predicted (platform list omission, an upstream format-origin gap, not a hardware limit), fixed start-to-end with a minimal HpuPlatform-pattern two-override patch (commit 7ea0f27b), validated phase-A (boot) + phase-B (full certified posture: coherent generation, spec gates ABOVE the fp8 incumbent 0.9853/0.9951 vs 0.9258, decode penalty 1.21×/1.29× under the 2× trigger) and closed as **cert11** with a full release-gate chain (r105 engine battery at provenance-corrected anchors — a disclosed anchor-provenance defect in the original bars, corrected against the banked r99 artifacts and recomputed PASS; r105t/t2 same-day discriminator + thermal telemetry; r105b final-image gate on the tag-built overlay) — mxfp4 = certified weight-constrained lane, fp8 performance default unchanged (`results/2026-11-06-p626-mxfp4/`), and **P6.27 the MXFP8 AutoRound validation (2026-11-06)**: `intel-ai/Qwen3.8-27B-MXFP8-MXFP8MTP-CT-AutoRound` tested end-to-end on the CERTIFIED cert11 image — every pre-registered prediction held (no platform rejection, native `XPUMxFp8LinearKernel` dispatch on oneDNN `fp8_gemm`, quantized MTP drafter loads and drafts at spec pos1 0.9626 / acceptance 0.9174): phase-A isolation boot + phase-B full posture both green (health 285 s incumbent-identical, graph FULL_DECODE_ONLY captured, G1 coherent, 0/0 erraudit) — **ZERO fixes needed, no cert12**: the lane is a config-driven model route on cert11. G2 REPORT matrix: every TTFT face below both mxfp4 lanes (U32k 26.67 s), decode p50 81.8 ms = 1.53× (above both mxfp4 lanes — the decode-reversal finding, explained: W8 weight bytes + MX-scale path vs the fused block-scaled incumbent lane); all under the 2× trigger (`results/2026-11-06-p627-mxfp8/`), and **P6.28 the gemma-4 campaign (2026-11-06/07)**: full testing/validation of gemma-4-31B on both weight lanes with the G2 long-prefill freeze "solved fully"; the freeze was decomposed telemetry-end-to-end and closed in three layers (warm-gate v4 triton cache-key split — cert16; probe-time compile residue — accepted; LMCache store wedge — KVGATE fail-loud gate + `ARC_LMC=0` posture, cert17), the face120k watchdog cliff resolved as the execute-model RPC timeout (`ARC_RPC_TIMEOUT=1200`), both legs validated end-to-end (faces + decode, erraudit 0/0), and the prefill MAJOR ISSUE root-caused with bare-kernel benches (r108aa/r108ab: the hd512 full-attention wall — kernel-structural; sliding layers healthy; gap vs qwen3.8 architectural) — closed as **cert17**; then documented-only closure was rejected in favor of the FIX: r108ac/r108ad found the engine's own ESVD FA2 kernel does the same hd512 full-attention call 34× faster (the fp16-KV control refuted the bandwidth-wall reading — the triton kernel is slow per K-tile iteration), cert18 ships the wrapper-level dispatch and r108ae validated it end-to-end (gemma faces 2.2–16.3× faster — face120k 2950.3→181.5 s — decode +6 %, qwen discriminator marker-0 green) (`results/2026-11-06-p628-gemma4/`), and **P6.29 the remaining-bottleneck campaign (2026-11-07)**: eliminate all remaining gemma-4 decode/prefill/TTFT bottlenecks to qwen3.8-27b parity; r109a/r109a3 measured the 50/60 sliding hd256 layers as the remaining wall (triton 0.147–0.158 s/call at q8192 fp8-KV, k-INDEPENDENT — vs the vxk FA2 kernel windowed at 0.006–0.009 s = 18–25× prefill, 11–15× decode), r109a4–r109a6 eliminated the ref-miss suspects and closed them as a per-chunk-softmax bug in the TEST refs (both kernels correct — global-softmax refs green 1.3e-4…1.1e-3), cert19 ships the second wrapper-level dispatch (asserts cert18 v3 byte-exact; `ARC_ATTN_FA2=0` disables both) and r111 validated it end-to-end (faces 4.9/17.2/110.9/0.9 s — face32k at qwen-parity 1.016× of the 16.93 record; decode 28.4 solo / 195 agg8 tok/s = +45 %/2.64×; warm 32/32; G1 3/3; erraudit 0/0; both markers live per rank; qwen discriminator in-band 3.83/17.58 s); spec parity dispositioned closed (no MTP head in the checkpoint; ngram structurally refused by the upstream v1 spec path on pure-attention hybrids — banked r110/r111) (`results/2026-11-07-p629/`), and **P6.30 the online-quant campaign (2026-11-07/08)**: online MXFP4/MXFP8 quantization of base checkpoints (validated on gemma-4-31B + qwen3.8-27B) with mxfp4 perf SUPERIOR to fp8; the r113 1.9× penalty was root-caused through the full chain (r116b/c microbench + the half-geometry correction at r120; oneDNN v3.13 gemmstone gate math — the (f4,f4) joint dtype can never pass the counter-operand list → `matmul:ref`; per-call act quant ≈ GEMM cost at decode Ms) and FIXED with the W4A16 route (arc4 wheel `_xpu_C.fp4_gemm_w4a16` — bf16 acts × f4 weights → `jit:gemm` DPAS via weight decomposition, verbose-proven, rel 1.66e-3); same-image r120c A/B: **qwen online mxfp4 superior to online mxfp8 on every axis 1.47–1.75×** (U8k 3.66 / U32k 17.12 s, 62.5–65.2 solo / 301–311 agg8, spec 1.0000) + 1.74–1.87× TTFT over the W4A4 control; r120g gemma 1.90–2.21× over W4A4; r120d kernel matrix — w4a16 fastest at every decode M, the e8m0-fp8 GEMM pinned slow by group-scale handling inside `jit:gemm` (v3.13 verdict: not a ref fallback); plain qwen fp8-online closed as an upstream interface gap (r120h traceback + r120h2 `fp8_per_channel` probe banked); certified as **cert20** (r121 gate: fp8 discriminator in-band + the w4a16 lane green; 4 engineering laws incl. the `vllm_xpu_kernels._xpu_C` dlopen law) (`results/2026-11-08-p630-mxfp4-perf/`), and **P6.31 the online-mxfp8 W8A16 route (2026-11-08)**: the P6.30 act-quant-elimination fix applied to the 8-bit lane — the r122 kernel matrix first (every f8xf8 scale flavor 2.05–2.64× slower than bf16 at prefill; W8A8-grpw designed then dropped as strictly dominated), then the arc5 wheel op `fp8_gemm_w8a16_mxfp8` + `mxfp8_w8a16.py` reroute behind `ARC_MXFP8_MODE` (+ the w4a8 probe branch in `mxfp4_w4a16.py`); r123 kernel matrix — m816 1.74×/2.68× faster than stock mxfp8 GEMM at decode/prefill, 1.50×/1.06× behind the w16pc class (rel 2e-3), W4A8 joint oneDNN-refused at primitive-create (closed-negative, wrapper-`default:` mis-attribution resolved); r124/r124b e2e on BOTH target models: **qwen 1.37–1.55× over stock W8A8 (U8k 3.61 / U32k 17.07 s, 53.8–55.6 / 265.9 tok/s, spec 0.9804), gemma 1.57–1.85× (3.87 / 20.94 s, 27.7 / 192.4)** — checkpoint-fp8 retains the perf crown by adjudication, mxfp4-w4a16 the online-lane crown; the gemma online-mxfp8 capacity law banked (29 GiB TP2 weights → `ARC_MAXLEN=40960` accommodation, a capacity gate not a lane defect); certified as **cert21** (r125 three-lane gate: fp8 discriminator + w8a16 lane + w4a16 carry all green) (`results/2026-11-08-p631-mxfp8-w8a16/`), and **P6.32 the Xe2 FP8-nativeity verdict + kernel-feasibility map (2026-11-08)**: the ISA/oneDNN-source analysis that opened the custom-kernel door — no native FP8 dot on Xe2 (counter-operand conversion only; B70 = BMG-G31, 256 XMX), the W4A8 refusal re-read as our dispatch default, and the five pre-registered kernel lanes with the Option-B backport policy — and **P6.33 the skinny-M kernel campaign (2026-11-08/09)**: the arc6 wheel's three GEMV-regime M≤8 weight-only kernels (f8/f4 weight streaming, register dequant, deterministic split-K; split law v2.7) built correct (r126 GATE-1/2/3) and winning the eager kbench regime (w16pc/g16 6/6 at M=1) — then the r127 five-leg e2e battery FALSIFIED the reroute: under FULL_DECODE_ONLY XPU-graph replay stock oneDNN runs 420–680 GB/s device-side (r127f 50-call amortized probe; the motivating ~197 GB/s was an eager-launch host-rate artifact, and the size-1 decode graph pads M→2), so the scalar streaming family (ceiling ~300 GB/s) regressed gemma solo −38/−42% while every over-cap path sat on ref; route banked **default OFF** (`ARC_SKINNY_M`), cert22 behaviorally ≡ cert21 at serve (r127g2 cert leg 27.7/192.9 exact, zero route markers), kernels + route remain in-tree with the TC-redesign lane open behind the r127f gate (`results/2026-11-08-p633-skinny-m/`). Full phase ledger with every per-phase record: [`TESTING-arc.md`](TESTING-arc.md) (validation record) and [`MIGRATION-arc.md`](MIGRATION-arc.md) (evidence table).

## Improvement-lane status — the in-scope ledger is CLOSED (P6.24; audited exhaustive P6.25)

Every prefill device slice is dispositioned by measurement — GEMM 39.7 % (fastest on-image route already deployed, r40/r41; r101 census: ~99 % of the per-card dense ceiling — no headroom), allreduce ~30 % (**quantized e4m3 wire deployed on cert10** — 1.65× at the dominant AR opt-in, 82 % of the measured wire floor; r100 proved the fabric itself the binding limit: writes 9.85 GB/s flat across store widths, reads 2.2× slower, overlap collapses the link), FA ~20 % (ESIMD v2 ceiling-bound at 0.33× stock — P6.21c), GDN ~5 % (marginal), norms/act ~5 % (small), host path (clean — P6.20). The stack sits at the practical in-scope floor. The two P6.22 candidate routes are now: quantized AR **DEPLOYED** (cert10, default-off env `CCL_SYCL_ALLREDUCE_FP8=1`), and AR/compute overlap CLOSED at every level: r83 same-stream (the device stream is compute-saturated at prefill), r100 fabric (concurrency collapses the link), and **r101 cross-stream co-tenancy (P6.25)** — the async-TP-class door: enqueue is async (0.18 ms) but the device time-slices AR and GEMM kernels (concurrent AR inflates GEMM latency 1.97× stock / 1.58× fp8), so k-sub-batch overlap is slower than serial at every k on both wires (best 0.969×/0.966×) — closed schedule- and wire-independent. Residual headroom is bounded ≤1–2 % class (AR-internal pipelining r100 ~1 %; fp8 quant-pass fusion into GEMMs — a novel cross-library contract that only relocates the same EU work). Transport was proven clean earlier (r71: direct P2P, 10.23 GB/s D2D, all collectives on the `topo sycl` device path; `tools/pcie/README.md`).
