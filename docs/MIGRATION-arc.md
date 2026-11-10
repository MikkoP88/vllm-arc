# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
law; identities re-pointed at each certification close-out — most
recently **cert22 (P6.33, 2026-11-09; the skinny-M kernel family
banked OFF on the cert21 engine — the arc6 wheel
`vllm_xpu_kernels 0.1.14.1+arc6` adds three GEMV-regime M≤8
weight-only GEMM ops (`_xpu_C.skinny_fp8_gemm_w8a16` /
`skinny_fp8_gemm_w8a16_mxfp8` / `skinny_fp4_gemm_w4a16`: one
work-item streams TN weight columns K-contiguously, dequantizes in
registers, fp32 accumulate, deterministic fixed-order split-K) plus
W4A8 dispatch case arms + the v0.1.17 Option-B backports
#611/#645/#650/#600, via `tools/cert22.Dockerfile`; the wheel-side
route `vllm_gaudi/xpu/patches/skinny_m.py` ships behind
`ARC_SKINNY_M` — **unset/off = stock routes, the certified
default**: the r127 five-leg e2e battery falsified the reroute
(under FULL_DECODE_ONLY XPU-graph replay stock oneDNN runs
420–680 GB/s device-side — the r127f 50-call amortized probe — while
the motivating ~197 GB/s stock level was an eager-launch host-rate
artifact, and the size-1 decode graph pads M→2, so the scalar-FMA
streaming kernels regressed gemma solo −38/−42%; every over-cap
path sat exactly on ref; `ARC_SKINNY_M=off` on the same image
reproduced the r124b ref exactly — causality + wheel-stock-path
validation); kernels correct (r126 GATE-1/2/3) and retained in-tree,
`=on` opts in with per-lane M caps; **serve behavior byte-identical
to cert21** — r127g2 cert leg: gemma default posture solo 27.7 /
agg8 192.9 tok/s = the exact r124b reference values, zero route
markers; rig `serve_arc_b70.sh` gained the `ARC_SKINNY_M`
pass-through;
`results/2026-11-08-p633-skinny-m/`)**, after **cert21 (P6.31, 2026-11-08; the online-MXFP8 W8A16 route on
the cert20 engine — online `ARC_QUANT=mxfp8` on a BASE bf16 checkpoint
serves GEMMs as W8A16: bf16 activations × f8 weights, group-32 e8m0
weight scales, zero per-call activation quant, via the arc5 wheel's
`_xpu_C.fp8_gemm_w8a16_mxfp8` (stock online mxfp8 is W8A8 — the
per-call act quant plus the measured (f8,f8) joint-rate penalty
2.05–2.64× vs bf16 at prefill, r122; the r123 kernel matrix
pre-registered the e2e result: m816 1.74×/2.68× faster than stock mx8
at decode/prefill medians, 1.50×/1.06× behind w16pc;
`external_patches/vllm-xpu-kernels/arc5_mxfp8_lanes/` build kit +
`vllm_gaudi/xpu/patches/mxfp8_w8a16.py` + the platform
`_XPU_PATCH_MODULES` entry, via `tools/cert21.Dockerfile`; knob
`ARC_MXFP8_MODE` — unset/`w8a16` = route (default), `w8a8` = stock
control; same-image A/B r124: qwen U8k 3.61 / U32k 17.07 s, decode
53.8–55.6 / 265.9 agg8 tok/s = 1.37–1.55× over stock W8A8 on every
face, spec pos1 0.9804 (control 0.9758); gemma r124b at
`ARC_MAXLEN=40960` (the 8-bit-weights capacity law): 3.87 / 20.94 s,
27.7 / 192.4 = 1.57–1.85×; adjudication: checkpoint-fp8 stays the
perf default, mxfp4-w4a16 stays the superior online lane; W4A8
(e4m3×f4) banked oneDNN-refused closed-negative; qwen fp8
discriminator + mxfp4 w4a16 carry legs in-band at the r125 gate =
incumbent lanes untouched;
`results/2026-11-08-p631-mxfp8-w8a16/`)**, after cert20 (P6.30, 2026-11-08; the online-MXFP4 W4A16 route on
the cert19 engine — online `ARC_QUANT=mxfp4` on a BASE bf16 checkpoint
reroutes every standard linear to the arc4 wheel's
`_xpu_C.fp4_gemm_w4a16` (bf16 activations × f4 weights, group-32 e8m0
weight scales → oneDNN `jit:gemm` DPAS via weight decomposition; stock
W4A4's per-call act quant ≈ GEMM cost at decode AND its (f4,f4) joint
dtype can never pass the gemmstone JIT gate → `matmul:ref`, r113→r120d
diagnostic chain; `external_patches/vllm-xpu-kernels/` build kit +
`vllm_gaudi/xpu/patches/mxfp4_w4a16.py` + the platform
`_XPU_PATCH_MODULES` entry, via `tools/cert20.Dockerfile`; knob
`ARC_MXFP4_MODE` — unset/`w4a16` = route (default), `w4a4` = stock
control; same-image A/B r120c: qwen U8k 3.66 / U32k 17.12 s, decode
62.5–65.2 / ~305 agg8 tok/s — superior to online mxfp8 on every axis
1.47–1.75×, 1.74–1.87× TTFT over W4A4, spec 1.0000; gemma r120g:
1.90–2.21× over W4A4; numerics rel 1.66e-3; qwen fp8 discriminator
in-band at the r121 gate = incumbent lanes untouched;
`results/2026-11-08-p630-mxfp4-perf/`)**, after cert19 (P6.29, 2026-11-07; the sliding-window FA2 route on the
cert18 engine — sliding-window head-256 calls (gemma-4's 50/60 local
layers, window (1023,0)) dispatch to the `vllm_xpu_kernels` ESVD FA2
kernel windowed inside the same triton unified-attention wrapper
(`external_patches/vllm/patch_arc_xpu_fa2_slide_route.py` via
`tools/cert19.Dockerfile`; r109a/r109a3 microbenches 18–25x prefill
k-independent + 11–15x decode; guards green kernel-vs-kernel AND vs
global-softmax GQA torch refs (r109a6 — the prior ~4 "ref misses" were a
per-chunk-softmax bug in the test refs, both kernels correct); r111
image validation: face8k 14.2→4.9 s, face32k 41.5→17.2 s (qwen-parity —
1.016x of the 16.93 record), face120k 181.5→110.9 s, face32k_2
4.7→0.9 s, decode 19.6→28.4 solo / 73.8→195 agg8 tok/s, G1 3/3,
erraudit 0/0, BOTH route markers live per rank; qwen discriminator
U8k 3.83 / U32k 17.58 s in-band vs the 3.76/16.93 records; ngram spec
for gemma-4 closed negative — the upstream v1 spec path asserts "no
mamba layers" on pure-attention hybrids (banked r111 traceback);
`results/2026-11-07-p629/`)**, after cert18 (P6.28, 2026-11-07; the FA2 route on the cert17
engine — full-attention head-512 calls (gemma-4's 10 global layers)
dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel inside the triton
unified-attention wrapper (`external_patches/vllm/
patch_arc_xpu_fa2_route.py` via `tools/cert18.Dockerfile`; r108ac/r108ad
microbenches 34× at production geometry, guards green; r108ae image
validation: gemma faces 2.2–16.3× faster — face32k 283.4→41.5 s,
face120k 2950.3→181.5 s — decode +6 %, G1 3/3, erraudit 0/0, route
marker live on both ranks; the qwen3.8 discriminator U8k 3.77 / U32k
17.49 s vs the 3.76/16.93 records with marker 0 = the incumbent lane
untouched; `ARC_ATTN_FA2=0` escape hatch; `results/2026-11-06-p628-gemma4/
phaseC/cert18_l1/`)**, after cert17 (P6.28, 2026-11-07; the gemma-4
validation chain on
the cert11 engine — cert12 B70 prefill launch-config table for the
triton unified-attention launcher + engine warmup gate, cert16 warm-gate
v4 (the engine-exact int-softcap triton cache-key fix — the G2
long-prefill freeze layer 1, root-caused r108v), cert17 lmcint kit
hardening (detector BS floor 16 + the fail-loud two-head-size
`[ARC-P628-LMC-KVGATE]` boot gate — freeze layer 3, gate-fire proven
live r108z); gemma-4-31B fp8 + mxfp8 validated end-to-end (faces,
decode, quality probes; erraudit 0/0); the gemma prefill root cause
diagnosed with bare-kernel benches (r108aa/r108ab first read it as an
L2-bandwidth wall; r108ac's fp16-KV control refuted that — the triton
kernel is slow per K-tile iteration — and the fix is the cert18 FA2
route above);
certified gemma posture `ARC_LMC=0` + `ARC_RPC_TIMEOUT=1200` (both carry
into cert18 unchanged);
`results/2026-11-06-p628-gemma4/`)**, after cert16/
cert12 (the P6.28 interim certs), after cert11 (P6.26, 2026-11-06; the MXFP4/quark lane opened —
`vllm_gaudi/xpu/platform.py` gains `quark` in `supported_quantization`
+ `supports_mx() → True`, commit 7ea0f27b; two pre-quantized MXFP4
checkpoints validated end-to-end under the full posture with the
native `XPUMxFp4LinearKernel`; overlay on cert10 — every byte carries
except platform.py + its pyc (r105b walk 294→294, freeze diff 0), so
the fp8 default posture is byte-identical: r105 engine battery PASS at
cert10 refs + r105t/t2 same-day face-for-face discriminator;
weight-constrained lane, fp8 stays the performance default)**, after
cert10 (P6.24, 2026-11-05; the oneCCL fp8-wire overlay —
same-version 2022.0.0 rebuild + e4m3-wire large-ring allreduce,
default OFF env `CCL_SYCL_ALLREDUCE_FP8`; opt-in 1.65× at the
dominant AR, every p65 TTFT face beats cert9: U32k −3.8 % / U64k
−8.2 % / U8k −13.0 % / C32k −4.6 %; default posture = cert9-identical,
proven at the gate; its overlay rows carry into cert11), after cert9
(P6.20, 2026-11-04; the prefill
host-path patch — L3
off-thread LMCache store default-ON + L6 dispatch-late + step_trace
diagnostics; U32k/U64k/U8k TTFT −0.74/−1.64/−0.08 s, spec/identity
faces byte-exact cert8), cert8 (P6.19, 2026-11-04; the
TurboQuant spec-verify corruption root-caused and FIXED — all four
`turboquant_*` kv-cache dtypes servable, `VLLM_ARC_TQ_SPEC_FIX=1`
default-ON, fp8 bytes unchanged), cert7 (P6.18, 2026-11-04 — its dtype
refusal was the interim mitigation, lifted at P6.19), cert6 (P6.14,
2026-11-02) and the P6.17 serve-layer certification (2026-11-03: the
certified line is `ARC_SPEC_METHOD=probabilistic` — the serve script
default flipped). Target: serve **qwen3.8-27b-fp8**
(dense hybrid-GDN, **MTP k3 recommended** — r46t real-world campaign) on
`vllm-arc:v0.30.0-b70-cert23` (the P6.35 release: graph-exec
double-buffer patch, posture gains are serve-line choices), replacing
the W14 incumbent
(`radiance-test`, flash-next k3) as the rig's serving stack. Everything
below is rig-proven; the full phase-by-phase evidence table is §6.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert22` (cert21 + the P6.33 skinny-M kernel family banked OFF — arc6 wheel swap: `vllm_xpu_kernels 0.1.14.1+arc6` adds `_xpu_C.skinny_fp8_gemm_w8a16` / `skinny_fp8_gemm_w8a16_mxfp8` / `skinny_fp4_gemm_w4a16` (GEMV-regime M≤8 weight-only GEMMs — TN weight streaming, register dequant, deterministic split-K; sha256 `484e06c18d0c8ea1cf6dc36dda24828e37b5c800688e9d4087d779cb6dfb7156`, ~352 MB — rig-only per the 100 MB artifact law, sha banked) plus W4A8 dispatch case arms + the v0.1.17 Option-B backports #611/#645/#650/#600, and carries every arc5 op (`fp8_gemm_w8a16_mxfp8`, `fp4_gemm_w4a16`, the refused `fp4_gemm_w4a8` probe op) + `vllm_gaudi/xpu/patches/skinny_m.py` (the `ARC_SKINNY_M`-gated M≤8 reroute — unset/off = stock routes, the certified default after the r127 e2e falsification; `on` = opt-in, per-lane M caps w16 M=1 / m816,g16 M≤2 + permanent per-lane fallback) + the platform `_XPU_PATCH_MODULES` entry (last in the chain) + the envs `ARC_SKINNY_M` registry entry, via `tools/cert22.Dockerfile` (sha-pinned wheel, build-time GPU-free asserts — version + op-presence incl. the carried arc5/arc4 ops + platform/patch/envs content, IMAGE_MANIFEST rows incl. `cert22_fork_sha`/`cert22_built`/`cert22_vxk_wheel_sha256`); gate evidence `results/2026-11-08-p633-skinny-m/` (r127g2: gemma default posture 27.7/192.9 = the exact r124b reference values, zero route markers; installed `patches/skinny_m.py` md5-verified = repo HEAD e2457389)). Prior: `vllm-arc:v0.30.0-b70-cert21` (cert20 + the P6.31 online-MXFP8 W8A16 route — arc5 wheel swap: `vllm_xpu_kernels 0.1.14.1+arc5` adds `_xpu_C.fp8_gemm_w8a16_mxfp8` (bf16 acts × f8 weights, group-32 e8m0, zero per-call act quant; sha256 `219268cf7d24ebc7a99e5afec5097edd63d88ef26255511c589da71f3615399a`, 352 MB — rig-only per the 100 MB artifact law, sha banked) and carries `_xpu_C.fp4_gemm_w4a16` + `_xpu_C.fp4_gemm_w4a8` (the oneDNN-refused W4A8 probe op) + `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` (the ARC_MXFP8_MODE-gated reroute) + the `mxfp4_w4a16.py` w4a8 probe branch + the platform `_XPU_PATCH_MODULES` entry, via `tools/cert21.Dockerfile` (sha-pinned wheel, build-time GPU-free asserts — version + op-presence + platform/patch content, IMAGE_MANIFEST rows incl. `cert21_fork_sha`/`cert21_built`/`cert21_vxk_wheel_sha256`); gate evidence `results/2026-11-08-p631-mxfp8-w8a16/cert21_l1/` (r125: qwen fp8 discriminator in-band + the w8a16 leg markers/dispatch/battery green + the mxfp4 w4a16 carry leg green)). Prior: `vllm-arc:v0.30.0-b70-cert20` (cert19 + the P6.30 online-MXFP4 W4A16 route — arc4 wheel swap: `vllm_xpu_kernels 0.1.14.1+arc4` adds `_xpu_C.fp4_gemm_w4a16` (sha256 `c675f1a34619302edb118abbe72b9fd928157beb3508193c2afa8a0f58d14c08`, 352 MB — rig-only per the 100 MB artifact law, sha banked) + `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` (md5 `33e54cd3…`, the ARC_MXFP4_MODE-gated reroute) + the platform `_XPU_PATCH_MODULES` entry (md5 `ecfea664…`), via `tools/cert20.Dockerfile` (sha-pinned wheel, build-time GPU-free asserts — version + op-presence + platform/patch content, IMAGE_MANIFEST rows incl. `cert20_fork_sha`/`cert20_built`/`cert20_vxk_wheel_sha256`); gate evidence `results/2026-11-08-p630-mxfp4-perf/cert20_l1/` (r121: qwen fp8 discriminator in-band + the w4a16 leg markers/dispatch/battery green)). Prior: `vllm-arc:v0.30.0-b70-cert19` (cert18 + the P6.29 sliding-window FA2 route — `external_patches/vllm/patch_arc_xpu_fa2_slide_route.py` via `tools/cert19.Dockerfile`: sliding-window head-256 calls (window left ≥ 0 right ≤ 0, the (w-1,0) causal form) dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel windowed in the same triton unified-attention wrapper, inserted right after the cert18 hd512-full dispatch which the patcher asserts byte-exact; exactly one wheel file patched (same file/surface as cert12/cert18) + pyc purge; IMAGE_MANIFEST carries all prior rows forward + the cert19 description row + `cert19_fork_sha`/`cert19_built` stamps; `ARC_ATTN_FA2=0` disables BOTH routes; gate evidence `results/2026-11-07-p629/cert19_l1/`). Prior: `vllm-arc:v0.30.0-b70-cert18` (cert17 + the P6.28 FA2 route — `external_patches/vllm/patch_arc_xpu_fa2_route.py` v3 via `tools/cert18.Dockerfile`: full-attention head-512 calls dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel in the triton unified-attention wrapper; exactly one wheel file patched (`vllm/v1/attention/ops/triton_unified_attention.py`, same surface as the cert12 table) + pyc purge; IMAGE_MANIFEST carries the cert12/cert16/cert17 rows forward + the cert18 description row + `cert18_fork_sha`/`cert18_built` stamps; `ARC_ATTN_FA2=0` restores the pure-triton path; gate evidence `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/`). Prior: the cert11 engine + the P6.28 chain applied at build via the external_patches kit — cert12 `patch_arc_xpu_attn_launcher.py` (commit `a964e047`), cert16 warm-gate v4 (`5d895fef`, `tools/cert16.Dockerfile`), cert17 lmcint kit hardening (`e5e725b4`, `tools/cert17.Dockerfile` = cert16 + kit with code-pattern absence gates); IMAGE_MANIFEST carries the cert12 row + cert16/17 description rows (their fork_sha/built stamps ship empty — cosmetic build-script wart, content proven by the banked build logs + the r108z gate-fire boot); gate evidence `results/2026-11-06-p628-gemma4/`. Prior: cert11 — overlay on cert10 built from the `rel/v0.30.0-b70-cert11` tag archive via `tools/cert11.Dockerfile` — the ONLY byte changes vs cert10 are `vllm_gaudi/xpu/platform.py` (md5 `e48abe9cd9de0027736f962504e84aa9`, the P6.26 quark+supports_mx patch) + its regenerated pyc; r105b exhaustive walk 294→294 files with changed set exactly those two, pip freeze diff 0; cert10's engine + oneCCL overlay rows (`oneccl=2022.0.0+arc1`, fat-lib shas at both SONAME paths — FP8 opt-in `CCL_SYCL_ALLREDUCE_FP8=1` unchanged) carry byte-identical; gate evidence `results/2026-11-06-p626-mxfp4/` r105/r105b + phase A/B) |
| fork SHA baked | cert22 content = `e2457389` (arc6 build kit + skinny_m.py route default-OFF + platform entry + envs entry + cert22.Dockerfile — the branch content commit the r127g2 certification leg ran against; the squash-merged main close-out commit carries the docs/results outside the image) on cert21 content = `c8fdf32a` (arc5 build kit + mxfp8_w8a16.py + the w4a8 probe branch + platform entry + cert21.Dockerfile — the branch content commit at build time; the squash-merged main close-out commit carries the docs/results outside the image) on cert20 content = `a918bee9` (arc4 build kit + mxfp4_w4a16.py + platform entry + cert20.Dockerfile — the branch tip at build time; the squash-merged main close-out commit carries the docs/results outside the image) on cert19 content = `904a3de8` (slide-route patcher + cert19.Dockerfile) on cert18 content = `3136adfe` (FA2 route v3; the branch tip `247cd234` differs only by the kit-README doc hunk, outside the image) on cert17 content = `e5e725b4` (lmcint kit) on cert16 content `5d895fef` (warm-gate v4) on cert12 content `a964e047` (launcher table), all on the cert11 engine (`c987c2cd`); annotated tag `rel/v0.30.0-b70-cert22` points at the main close-out commit carrying the chain; the IMAGE_MANIFEST base rows carry from cert9/10 (`fork_sha=aa2ee17d…` + the oneCCL overlay rows) and `cert12_fork_sha=a964e047`, plus `cert18_fork_sha=3136adfe`, `cert19_fork_sha=904a3de8`, `cert20_fork_sha=a918bee91ced4ad826ebe617bba665b1e7fa9f0b`, `cert21_fork_sha=c8fdf32a8209f4992fbac063068948b6322a1790`, and `cert22_fork_sha=e2457389bba361252291a3b196ac4807ab9ac05f`; cert16/17 fork_sha stamps ship empty (wart, see image row). cert11: IMAGE_MANIFEST `cert11_fork_sha=c987c2cd57edf19f0f7a704a29de70d47a2c1264` == annotated tag `rel/v0.30.0-b70-cert11` on main; the base `fork_sha=aa2ee17d…` row preserved from cert9/cert10 — the vllm tree is byte-identical; image manifest `sha256:928a6dfe…`) |
| validation image | `vllm-arc:v0.30.0-b70-cert22` content image `vllm-arc:v0.30.0-b70-arc6test` (FORK_SHA e2457389; the tag-built cert22 image = the same validated layers retagged) — r127/r127g2 gate: build gate green (Dockerfile asserts incl. `CERT22_SKINNY_LIVE` — wheel version `0.1.14.1+arc6`, skinny op presence ×3 + the carried arc5/arc4 ops, platform/patch/envs content), installed `patches/skinny_m.py` md5-verified = repo HEAD (`61ff347a…`, contains the default-off gate; the first r127g run caught a stale-flat-file ctx22 build — the flat-COPY trap, banked in the plan's P6.33 execution log); r127 five-leg e2e battery (qwen fp8/mxfp8/mxfp4 + gemma mxfp8/mxfp4): route markers live per leg, qwen legs inert by construction (MTP k3 verify batches over every cap), gemma solo regressed −38/−42% with agg8/TTFT/spec exactly on ref — the falsification; r127c EXP-1 `ARC_SKINNY_M=off` same-image → 27.7/192.3 exact = the r124b ref (causality + wheel stock-path validation); r127g2 cert leg, default posture: solo **27.7 / agg8 192.9** = the exact r124b reference values, zero `skinny-M wrap installed`/`route FIRST` markers in the serve log; `results/2026-11-08-p633-skinny-m/` (incl. `rig_raw/r127g2_certleg.log`)). Prior: `vllm-arc:v0.30.0-b70-cert21` itself (r125 three-lane gate: build gate green — Dockerfile asserts incl. `CERT21_W8A16_LIVE` (wheel version `0.1.14.1+arc5`, op presence `fp8_gemm_w8a16_mxfp8` + `fp4_gemm_w4a8` + `fp4_gemm_w4a16` carried, platform/patch content), manifest rows incl. `cert21_fork_sha=c8fdf32a…` + `cert21_vxk_wheel_sha256=219268cf…`; leg A qwen fp8 discriminator — default posture, marker audit (fa2 0 / slide 2 / reroute lines inert), faces + decode + spec in-band vs the banked refs = incumbent lanes untouched; leg B qwen online mxfp8 W8A16 — `reroute ACTIVE` marker, kernel-select proofs, spec PASS, faces/decode in-band vs the r124 legs; leg C qwen online mxfp4 W4A16 carry — `reroute ACTIVE`, in-band vs the r120c refs; erraudit 0/0 every leg; `results/2026-11-08-p631-mxfp8-w8a16/cert21_l1/`). Prior: `vllm-arc:v0.30.0-b70-cert20` itself (r121 gate: build gate green — Dockerfile asserts incl. `CERT20_W4A16_LIVE` (wheel version `0.1.14.1+arc4`, op presence, platform/patch content), manifest rows incl. `cert20_fork_sha=a918bee9…`; leg A qwen fp8 discriminator — default posture, marker audit (fa2 0 / slide 2 / w4a16 reroute lines inert), G1 3/3, U8k/U32k in-band vs the 3.76/16.93 records, erraudit 0/0 = incumbent lanes untouched; leg B qwen online mxfp4 W4A16 — `reroute ACTIVE` marker, online-dispatch proofs, G1 3/3, spot battery vs the r120c legs (identical bytes already validated there: U8k 3.66 / U32k 17.12 s, decode 62.5–65.2 / 301–311 agg8, spec 1.0000), erraudit 0/0; `results/2026-11-08-p630-mxfp4-perf/cert20_l1/`). Prior: `vllm-arc:v0.30.0-b70-cert19` itself (r111: build gate green — slide marks 2/2, fa2 marks 3/3, tune marks 3/3, manifest rows incl. `cert19_fork_sha=904a3de8…`; gemma L1 leg — warm gate 32/32, BOTH markers live (`[ARC-P628-FA2]` + `[ARC-P629-SLIDE]`, one line per TP rank each), probe98 clean, G1 3/3, face8k 4.9 / face32k 17.2 / face120k 110.9 / face32k_2 0.9 s, decode solo 28.4–28.5 / agg8 194.5–195.5 tok/s, erraudit 0/0, JIT-during-inference 0; r110 ngram leg — boot + markers fine but EngineCore dies on first request (upstream `_get_mamba_bufs` assert "no mamba layers" on pure-attention hybrids; ngram CLOSED NEGATIVE for gemma-4, decode posture stays `ARC_SPEC=0`); qwen discriminator — U8k 3.83 / U32k 17.58 s vs the 3.76/16.93 records (inside the ±5 % cross-boot band), fa2 marker 0, slide marker 2 (the guarded sliding-hd256 class also occurs in qwen traffic/tune-warmup — in-band + erraudit 0/0 = no regression), erraudit 0/0; `results/2026-11-07-p629/cert19_l1/`). Prior: `vllm-arc:v0.30.0-b70-cert18` itself (r108ae: build gate green — fa2 marks 3/3, tune marks 3/3 (cert12 table still live), manifest rows; gemma L1 leg — warm gate 32/32, `[ARC-P628-FA2]` marker live on BOTH TP ranks, probe98 13 s, G1 3/3, face8k 14.2 / face32k 41.5 / face120k 181.5 / face32k_2 4.7 s, decode solo 19.6 / agg8 73.8 tok/s, erraudit 0/0, one benign first-request triton compile; qwen discriminator — U8k 3.77 / U32k 17.49 s vs the 3.76/16.93 records (inside the ±5 % cross-boot band), marker 0, erraudit 0/0; the v2 descale-crash forensics banked alongside; `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/`). Prior: itself (r108z cert17 build gate OK + KVGATE gate-fire boot refusal ("got [512, 1024]", both ranks) + full L1 fp8 battery green with `ARC_RPC_TIMEOUT=1200` — probe98 35 s, G1 3/3, face8k 31.5 / face32k 283.4 / face120k 2950.3 / face32k_2 9.8 s, erraudit 0/0, zero jit events in the 49-min face120k; r108x L2 mxfp8 leg green — face8k 35.4 / face32k 297.4 / face90k 1819.5 / face32k_2 10.3 s, G1 reason-probe miss flagged; r108v2/v2b decode — L1 18.4–18.5 solo / 69.9 agg8 tok/s, L2 11.9 / 54.6; r108aa/r108ab perf diagnostic closing the hd512 wall; `results/2026-11-06-p628-gemma4/`. Prior: cert11 itself (r105b final-image gate PASS from the banked artifacts: manifest + oneccl rows, tree walk changed-set proof, overlay shas ×2, thin-shim identity, freeze 0, HEALTH 295 s, flip-first 9.1 s clean, SPEC_GATE pos1 0.9217, ERROR=0; r105 engine battery on cert11c run-2 clean PASS at provenance-corrected r99 anchors — t0 300.48 / 84.55–84.56, TTFT 17.37/1.09/3.77/38.82 at-or-better, LONGRETURN PASS; phase-B mxfp4 legs coherent + spec 0.9853/0.9951 — `results/2026-11-06-p626-mxfp4/`; lineage: cert10 `results/2026-11-05-p624-fp8-ar/`, cert9 `results/2026-11-04-p620-prefill-opt/`, cert8 `results/2026-11-04-p619-tqverify/`, p617 prob-lane `results/2026-11-03-p617a-probcert/`, r70 comprehensive `results/2026-10-31-p70-r70-cert4-comprehensive/`, default faces cert4-native `results/2026-10-31-p65-cert4/`) |
| canonical serve script | rig `/root/build/serve_arc_b70.sh` (the uniform boot interface; carries the **25-var chain** incl. `LMCACHE_TRACK_USAGE=false` — P6.3; run with `ARC_IMAGE=vllm-arc:v0.30.0-b70-cert22 ARC_GRAPH=1 ARC_SPEC=3`; P6.33 skinny-M opt-in lane: engine env `ARC_SKINNY_M` (unset/`off` = stock routes — the certified default; `on` = the M≤8 skinny kernels with per-lane caps; the script passes it through to the container like `ARC_MXFP8_MODE`); P6.31 online-mxfp8 lane: `ARC_QUANT=mxfp8` on a BASE bf16 model + engine env `ARC_MXFP8_MODE` (unset/`w8a16` = the W8A16 route — default; `w8a8` = stock control; any other value WARNs and keeps stock) + **`ARC_MAXLEN`** for the 8-bit-weights capacity law (default 262144; gemma online-mxfp8 legs need 40960 — 29 GiB TP2 weights leave 5.31 GiB KV, short of one 262144 seq; admission-only cap); P6.30 online-quant lane: `ARC_QUANT=mxfp4` on a BASE bf16 model + engine env `ARC_MXFP4_MODE` (unset/`w4a16` = the W4A16 route — default; `w4a4` = stock control; `w4a8` = the banked oneDNN-refused probe; any other value WARNs and keeps stock) + `ARC_QUANTCFG` ignore list for hybrids (`{"ignore":["*visual*","*linear_attn*","mtp.fc"]}` on qwen3.8; gemma needs none — its GDN layers auto-skip); P6.28+P6.29 FA2 routes: engine env `ARC_ATTN_FA2` (default 1 = the hd512 full-attention route AND the hd256 sliding-window route ON; `=0` restores the pure-triton path — a serve-env line, not a 26th script var, see envs.md); P6.28 gemma-4 lane knobs: `ARC_MODEL` / `ARC_SERVED` / `ARC_QUANT` (empty = config-driven) + `ARC_TEMPLATE` (gemma4 chat template) + **`ARC_LMC=0`** (the KVGATE posture — REQUIRED for two-head-size hybrids) + **`ARC_RPC_TIMEOUT=1200`** (→ `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`; REQUIRED for gemma-4 face90k+/120k — the stock 300 s watchdog kills late prefill chunks) + **`ARC_SPEC=0` for gemma-4 decode** (P6.29: no MTP head in the checkpoint; the ngram lane is structurally refused by the upstream v1 spec path on pure-attention hybrids — r111 banked); P6.24 opt-in = add ONE env line `export CCL_SYCL_ALLREDUCE_FP8=1` to the serve env — the r99 [3] pattern, everything else unchanged; P6.26 model-lane knobs `ARC_MODEL` / `ARC_SERVED` / `ARC_QUANT` (empty = config-driven quant — the mxfp4 lane, CONFIGURATION-arc "MXFP4 weight lane")); since **P6.17 the certified default is the probabilistic draft-sampling posture — unset or `ARC_SPEC_METHOD=probabilistic` appends the upstream spec key `draft_sample_method`**, `ARC_SPEC_METHOD=greedy` = the opt-out historical lane (pre-P6.17 bytes), `ARC_SPEC=0` = no spec; flip validated end-to-end — flag probe 5/5 + default-path boot engine bytes identical to the record legs — `results/2026-11-03-p617a-probcert/`; the pre-flip script is preserved on the rig as `/root/build/serve_arc_b70_p616.sh` (provenance backup) |
| long-context profile | rig `/root/build/serve_arc_b70_defer2.sh` (the canonical chain + `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` — P6.5; opt-in, see CONFIGURATION-arc "fp8 stall levers") |
| A/B fallback | same script with `ARC_SPEC=4` (MTP k4 — documented alternative for near-deterministic workloads) |
| golden reference | rig `/root/build/p25/golden_ref.json` (md5 `e02cbb07…`; covers k4 AND k5 — r42/r43 proved byte-identical) |
| regression matrix | rig `/root/build/matrix_arc/leg-2026-10-29-r44-certified/` (12 cells) |
| container name | `arc-test` |

The image is **self-contained**: no cap `PYTHONPATH` overlay, no runtime pip
installs (lmcache 0.5.6+arc1 closure vendored; in-build import gate +
`lmcache_closure=` manifest line — r42 Fix A). Wheel-module patches load via
the in-image `_WHEEL_HOOKS` registry (r45 wiring + `_load_wheel_patch`
standalone loader — r42 Fix B).

**cert11 migration note (P6.26, 2026-11-06)**: for an existing cert10
deployment there is **no action and no behavior change** — the only
byte deltas are `vllm_gaudi/xpu/platform.py` (+pyc), and the patch is
config-validation-surface only (kernel selection is table-keyed and
never consults the changed paths; r105 battery + r105t/t2 same-day
discriminator prove the fp8 posture face-for-face identical). What
cert11 adds: quark-format checkpoints (`amd/Qwen3.8-27B-Quark-AWQ
-MXFP4`) boot instead of failing config validation, and the false
"Simulated weight dequantization" warning is gone. Serving knobs are
unchanged; the mxfp4 lane is a model-route choice
(`ARC_MODEL`/`ARC_SERVED`/`ARC_QUANT=`, CONFIGURATION-arc "MXFP4
weight lane"). Evidence: `results/2026-11-06-p626-mxfp4/`.

## 2. Boot procedure (from W14-serving state)

```bash
# on the rig host (address lives in the untracked local layer, never in
#                  tracked files — publishing-hygiene law; every ssh:
#                  export SSH_ASKPASS=/tmp/askpass.sh
#                  SSH_ASKPASS_REQUIRE=force DISPLAY=:0)
docker stop radiance-test                      # frees port 8000 + both GPUs
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert11 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash /root/build/serve_arc_b70.sh            # boot + serve as one unit
# draft posture: unset/probabilistic = CERTIFIED (P6.17); the certified
# line can be written explicitly as ARC_SPEC_METHOD=probabilistic (same
# bytes); ARC_SPEC_METHOD=greedy = opt-out historical lane
# logs land in the container at /root/serve_full.log (host: docker exec)
# poll health (first-traffic law: batteries run on FIRST traffic after boot)
curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health
```

The script boots the certified container (`--privileged` REQUIRED —
pidfd IPC law, r30 defect #1; `--network host --ipc host`, the model
mount, cache mounts, and the 2 boot-layer `-e` vars) and execs the
serve line with the full 25-var chain — env + CLI as one unit by
construction. Boot takes ~5 min to health 200 (285–295 s at the
cert7–cert11 gates; 295 s measured at the cert11 gate, r105b). Verify in
`/root/serve_full.log`: wheel hooks install (zero `FAILED`, silent
defer is DEBUG), `fp8 reroute ACTIVE`, `kvfmt TWO_HS`, the p40c re-pin
line both workers (`alloc/free_pinned_ptr re-pinned to the pageable
torch baseline`), `glue-unavailable 0`, `[p41c] wrap live`, posture
lines (mode=none, FULL_DECODE_ONLY [1..320], MTP k3, align), and since
cert9 the P6.20 store-worker line after the FIRST request
(`LMC store worker ACTIVE` — the off-thread L3 store, default ON).

## 3. Post-boot verification (any migration or image bump)

§10 continuous row, in order (harness tools unchanged, host-side python3):

1. **survival flip-first**: 832-token flip payload as FIRST traffic → 200.
2. **spec gate** (`arc_spec_gate.py`): pos1 ≥ 0.85 target (0.5 hard).
3. **golden** (`p25/p32_golden.py`): 5/5 shas byte-identical to
   `golden_ref.json` (0-drift law).
4. **run7n ×2**: conservative floors solo ≥ 79.74, agg#2 ≥ 274.41;
   certified steady family at k3, prob lane (p617 + release gates):
   solo **83.7–84.9** / agg **291.4–303.1** (cert11 gate: 300.48 /
   84.55–84.56, r105 run-2; cert10 gate: 300.39 /
   84.52–84.51; cert9 gate: 296.91 /
   84.54–84.48; cert8 gate: 300.65 / 84.53–84.57; cert4-native greedy
   family 294.7–300.5 in-band — the
   argmax path is posture-invariant) — **comparator law (r47c)**:
   a fresh-boot FIRST battery on the same stack reads lower (~206/64.5
   warmup face); gate a fresh boot against a same-posture A/B, not the
   campaign table.
5. **matrix 12 cells** (`r44_matrix.sh`): no face −5% vs
   `leg-2026-10-29-r44-certified` (regression block).
6. **integrity/determinism** (`r42/r42_integrity.py`): 5/5 + 3/3.

Any block → do not migrate; capture per E-law, fix on a branch, re-certify.

## 4. Rollback (arc → W14)

```bash
docker stop arc-test                            # artifacts stay in the container fs
docker start radiance-test
nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_rXX.log 2>&1 &
# poll health: 200 within ~5–10 min (first polls may be 000 — model load)
```

Keep `arc-test` (Exited) on the rig — it carries the leg's serve logs
(`/root/serve_full.log`). Delete only after the leg's artifacts are banked
to `results/`.

## 5. Rig laws that govern the migration window

- NEVER leave the rig on an arc image overnight; W14 restores after every
  arc block (plan §9 / App F etiquette).
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` on all graph legs (set inside the serve
  script).
- Verify interleave/BW matrix after any rig reboot.
- Port 8000 is single-tenant: stop radiance-test before arc legs.
- `--mamba-cache-mode=align` is pinned in the serve line (posture-pin
  law: never ship its removal).

## 6. Evidence table (phase ledger)

One row per phase/leg-group; the "what it proved" details live in
PERFORMANCE-arc / TESTING-arc. All paths relative to `results/`.

| Phase | Evidence | What it proved / banked |
|---|---|---|
| P4 r42 | `2026-10-30-p4-r42-certified-image/` | certified-image boot + battery; self-contained image (Fix A/B) |
| P4 r43 | `2026-10-30-p4-r43-k5-adoption/` | k5 adoption (later demoted by r46t) |
| P4 r44 | `2026-10-30-p4-r44-matrix-certified/` | 12-cell matrix — the regression baseline |
| P5 r45 | `2026-10-30-p5-r45-env-log-audit/` | env/log audit + this addendum |
| P5 r46 | `2026-10-30-p5-r46-soak/` | 8 h soak GREEN + mypy delta=0 + 64k face (P5 gate of record) |
| P5 r46t | `2026-10-30-p5-r46t-temps/` | real-world campaign: **k3 recommended**, k5 dropped, garbage suite v2 |
| P6.1 r47a | `2026-10-30-p6-r47a-ext-hitrate/` | external-hit-rate 0.0 % is a steady-state artifact, not a broken cache |
| P6.2+6.3 r47c/p48 | `2026-10-31-p6-r47c-p48-056gates/` | lmcache 0.5.6+arc1 gates; zero fork-added WARN/ERROR; comparator law |
| P6 r47e | `2026-10-31-p6-r47e-cert3/` | cert3 tag-built image (manifest A/B + image-native 28/28) |
| P6.5 r48a–l | `2026-10-31-p65-r48a-stall-capture/` | stall attribution + length-gated chunk-store deferral (the p65 profile) |
| P6.5 cert4 | `2026-10-31-p65-cert4/` | cert4 gates (candidate battery + p65 face on the shipped image) |
| P6.6 p66 | `2026-10-31-p66-r66a-prefill-capture/`, `…r66b-step-decomposition/`, `…r66c-mnbt16k/` | prefill anatomy: device-bound ~34–40 % MFU, host glue empty, MNBT-16384 A/B not adopted — analysis only, no engine change |
| P6.7 p67 | `2026-10-31-p67-r67a-compile-ab/` | compile REVAL rejected (acceptance −35 % gate); GDN capture-cap law; op-class trace — rungs closed/re-scoped, no engine change |
| P6.10 p70 | `2026-10-31-p70-r70-cert4-comprehensive/` | every current-record face re-measured cert4-native (3 boots): tier attribution (warm re-hits = engine APC), r46t parity no-regression, no-spec A/B 1.5–2.5×; leg c 4 h soak GREEN (56/56 cycles, zero failures, no drift — `c/` mirror) — no engine change, no cert5 |
| P2P p71 | `2026-10-31-p71-r71-p2p-analysis/` | TP2 allreduce path: **direct P2P confirmed** (r71b D2D 10.23 GB/s both directions; r71a all collectives `algo topo sycl`, pidfd IPC live, zero host-staged) — all lanes closed, no fix needed, AR gains are algorithmic |
| P6.5 annex p73 | `2026-11-01-p65-end-stall-analysis/` | END-stall five-mechanism analysis (end CLEAN at 32k; M1 store tail / M2 backlog hangover / M4 admission aliasing) — analysis only |
| P6.13 p73 | `2026-11-01-p65-rung3-slice/` | **sliced final-span store** (`LMC_STORE_SLICE_TOKENS`, default-OFF): rig A/B gate RUN + verdict — all safety/regression faces GREEN (integrity 5/5, run7n 300.48/84.4, families unchanged, 0 errors), mechanism landed exactly (63232-class = 0; 38 = 2×19 windows; U64k TTFT −1.5…−3.5 s); pre-registered ≤600 ms max-gap bar mechanically missed (1598) and recorded as mis-set (r48e nolmc floor 2347) — **code merged default-OFF, enabling profile experimental/uncertified** |
| P6.13 cert5 | `2026-11-01-p65-cert5/` | **cert5 release gates** (tag-built image, single flow): manifest fork_sha == tag, p48 28/28 + p73 slice-hunk 11/11 image-native, SLICE_ENV_ABSENT=0, run7n t0 297.19/84.48 + t1.0 189.41/62.35 in-family, integrity 5/5 + det 3/3, WARN class-set identical to cert4, slice face on the SAME image — U64k TTFT 50.77 (cert4 51.95), W_U64k_start max 1602 vs 1997, 63232-class 0 / 3328×38 exact — superseded by cert6 |
| P6.14 cert6 | `2026-11-02-p614-cert6/` (+ candidate legs `2026-11-02-p614-{l1v2,l2,l3}/`) | **cert6 release gates** (tag-built image, single flow): manifest fork_sha `986e97d6` == tag, p48 28/28 + p73 slice-hunk 11/11 image-native, SLICE_ENV_ABSENT=0, run7n t0 296.25/84.46 + t1.0 186.48/61.79 in-family, integrity 5/5 + det 3/3, WARN class-set identical to cert5 (44 == 44), slice face on the SAME image (3328×38 exact, 63232-class 0, W_U64k_start max 1604) + the first certified opt-in TQ face `turboquant_4bit_nc` on the same image (0 errors, kit D2H live @262, spec 0.9714/0.9381, KV 757,641 = 2.89×, U32k/U64k TTFT 15.85/33.98 s) — superseded by cert7 (P6.18): fp8 behavior byte-identical, but its TQ dtype faces degenerate on this model (P6.18 Class B) |
| P6.15 dtype/stall probes + collapse mechanism | `2026-11-02-p615-{stall-analysis,b6,b7,b8}-*/`, `2026-11-02-p615-probes/`, `2026-11-03-p615c-t10collapse/` | bystander-stall taxonomy + chunk-granularity Pareto matrix (all dtypes, MNBT 4096 halves stalls on every preset); 12-face solo/concurrent decode/prefill/TTFT master matrix on the fork's own harnesses; t1.0-xhigh sampled-collapse mechanism pinned (per-request MTP acceptance death from round 1, engine cadence constant, fp8 immune, no config fix) — analysis only, no engine change |
| P6.16 draft-sampling posture | `2026-11-03-p616a-probsmoke/`, `2026-11-03-p616b-probmatrix/`, `2026-11-03-p616c-spechook/` | **opt-in `ARC_SPEC_METHOD=probabilistic`** (serve hook in `tools/serve_arc_b70.sh`): upstream-config-native `draft_sample_method` key on the pinned 0.30.0 wheel; t1.0-xhigh agg fp8 +29 % / tq4 +44 % / k3v4 +54 % / 3bit +91 % (k8v4 EV-neutral), uniform solo pairs (zero 26-tok/s legs in measured faces), t0 provably + measurably identical, det 3/3 everywhere, 11/11 hook-deployment gates PASS — **merged serve-layer: default stays greedy (superseded at P6.17 — the P6.17 default flip, next row), wheel/image unchanged, cert6 remains the certified image (no cert7 — the serve script is not in the image)** |
| P6.17 prob-lane certification | `2026-11-03-p617a-probcert/` | **the certified line is `ARC_SPEC_METHOD=probabilistic`; every record value re-measured natively on that lane** (2026-11-03): 5-boot comprehensive campaign on cert6 — t0 agg 291.4–303.1 / solo 83.7–84.9, t1.0 steady family 222.7–241.3 + uniform 64–71 solo (zero collapse pairs in 20), TTFT 3.77/7.90/17.38/40.32, warm t2 0.99–1.08 (fresh-seed warm2 probe; t1 colds 3.89/8.39/17.44) + warm-hit 0.80–0.85 s, temp sweep 214.9–224.7, alt 271.5–281.3, MTP-over-no-spec 2.35–2.53× by face, per-face acceptance windows 0.57–0.61 sampled / 0.859 t0 / 0.91 alt; integrity 5/5 + det 3/3 every measured boot, golden 5/5, garbage 15/15, 0 ERROR/Traceback ×5; serve-default FLIP validated (flag probe 5/5 on the real script via PATH-stubbed docker; default-path boot engine bytes == record legs; greedy opt-out boot carries the pre-P6.17 bytes) — the every-dtype ≥ +10 % default-flip rule was overridden (recorded: k8v4 EV-neutral, fp8/tq4/k3v4/3bit all clear); no new image — cert6 remained certified through P6.17 (serve-layer default, wheel/engine byte-unchanged; pre-flip script preserved `serve_arc_b70_p616.sh`; the image line caught up at cert7/P6.18) |
| P6.18a/b repro + triage | `2026-11-03-p618b-repro/` | **the turboquant corruption campaign**: the k8v4 corrupt-returns report reproduced and split — **Class B** deterministic degenerate generation (REP-LOOP attractors) on ALL FOUR `turboquant_*` presets, cold-included, LMC-independent, cross-dtype byte-identical attractors; **Class A** t0 warm-resume trajectory flips (dtype+LMC-independent, settled as inherent warm-resume-vs-cold numerical divergence — diag/diag2 canaries; disposition documentation, not code); fp8 legs clean throughout; 14 MB tree with byte-exact redaction (+1872) + W14 restore incident record |
| P6.18c roundtest + fix | `2026-11-03-p618c-roundtest/` (+ `BEST-PRACTICES-AUDIT.md` in the cert7 tree) | **root cause closed + the fix**: the live SoA store→decode Triton pair is numerically correct on XPU (decode self-coherence rel_err ≤ 6e-6, store spec-conformant [V ≤ 1 tie-step, K value-exact], fully deterministic, all four presets with engine tables + model geometry) — the degeneracy is the 4-bit-class value-precision quality ceiling (16 affine levels per token-head, 8 on 3bit; upstream's own tables +2.71…+20.59 % PPL on gpt-oss) → fix = `ArcXpuPlatform.check_and_update_config` refuses any `cache_dtype` startswith `turboquant` (upstream-patterened: rocm's prefix-match fail-loud idiom, cuda's config-time ValueError+remedy); upstream-pattern audit `BEST-PRACTICES-AUDIT.md` |
| P6.18d cert7 gates | `2026-11-03-p618-cert7/` | **cert7 release gates** (tag-built image `a8cd765b0380`, single flow; tag moved once — the first build read the CLI flag name `kv_cache_dtype` instead of the pydantic field `cache_dtype` and was caught by G0): manifest fork_sha `5bf4192d` == tag, p48 28/28 + p73 11/11 image-native, **G0 guard proof** k8v4 + tq4nc (loud ValueError ~65 s, startup 0, health never), default fp8 posture (health ~295 s, SLICE_ENV_ABSENT=0, KV pool 400,739 == p618b cert6-fp8 legs, flip 9.2 s stop), run7n t0 299.73/84.50/84.44 in-family, integrity 5/5 + det 3/3 (×2), spec pos1 0.9258 PASS, 0 ERROR/Traceback, WARN class-set zero fork-added, **LONG-RETURN GATE 100 % CLEAN** (4 classes single-hash across cold/rehit/rehit2 + 3 churn cycles; zero REP-LOOP/JUNK/BLEED/EMPTY; t1-seed annex: one sporadic flip characterized 16/16 identical — Class-A RNG-path divergence at sampled temperature, not corruption) + **REPRODUCIBILITY-AUDIT.md** (image ⊆ repo 144/144 md5; every image input tracked or pinned) — **`rel/v0.30.0-b70-cert7` = certified release** — superseded by cert8 (P6.19, next row): the TQ corruption root-caused and fixed, the dtype refusal lifted |
| P6.19 TQ spec-verify fix + cert8 | `2026-11-04-p619-tqverify/` (+ `rig/` gate evidence) | **corruption root-caused and FIXED — all dtypes servable**: 3 root causes from the t2i telemetry campaign (verify-capture baked raw-KV fast-path graphs / replay-broken multi-row synthetic-decode dispatch / SoA-vs-AoS MQ addressing on the first draft read), fix = `patches/tq_spec_fix.py` default-ON (split dispatch certified; `VLLM_ARC_TQ_VERIFY_MODE=mq` opt-in experimental, agg8 +21 %, MQVAL5 4/4 exact); cert8 gates ALL GREEN on the tag-built image (`b212fbca59ec`): manifest == tag `e47739b0`, p48 28/28 + p73 11/11, fp8 byte-family cert7 (KV 400,739, t0 300.65/84.53-84.57, spec pos1 0.9258, integrity 5/5 + det 3/3), fp8 + all-TQ long-return zero garble on the t0 lanes (identity faces dispositioned: Class-A warm-resume variants, 3-bit precision floor, and the t1-seed REP-LOOP flag = a temp-1.0 model-lane property proven firing on the zero-fork W14 incumbent too), tps k8v4 54.42/54.59/235.97, mq 284.81; 0 ERROR/Traceback ×7 boots; best-practices audit banked — **`rel/v0.30.0-b70-cert8` = certified release** — superseded by cert9 (P6.20, next row): the prefill host-path store patch, strictly faster TTFT, byte-exact spec/identity |
| P6.20 prefill host-path + cert9 | `2026-11-04-p620-prefill-opt/` (+ `rig/` r80–r84b evidence) | **prefill host path eliminated as a bottleneck; U32k/U64k proven device-bound**: step telemetry first (`patches/step_trace.py` 9-segment per-step timeline, silent-off) → L1 evsync NEUTRAL (r80, in-order stream), **L3 off-thread store ADOPTED** (r81 interleave A/B/A/B: U32k −0.74 s / U64k −1.64 s / U8k −0.08 s; `arc-lmc-store` worker, device ops stay on the engine stream; default ON, `VLLM_ARC_LMC_ASYNC_STORE=0` kill switch), r82 battery GREEN + source-override instrument lesson (wrap the ENGINE's module), **L6 dispatch-late** (r83 FULL GREEN; spec byte-exact cert8 pos1 0.9258/acc 0.837), r83 window trace = **lane closure** (wfs waits = device-queue drain behind chunk forwards, worker bodies ≤108 ms, ~89 % of U64k TTFT device-side; L6a/L6b dispositioned NOT-built; remaining lever = kernel compute AR 30.4 %/FA 20.3 %); cert9 gates ALL GREEN on the tag-built image (`a9f8252451b6` @ `aa2ee17d`, 4/4 file md5 == tag tar): manifest + p48 ALL_GREEN + p73 11/11, **default fp8 with ZERO env overrides** (STORE_WORKER_ACTIVE=2, LMC_ENV_SET=0, trace silent-off), integrity 5/5 + det 3/3, t0 296.91/84.54-84.48 in-family, t1.0 240.06 steady, spec byte-exact cert8, fp8 long-return PASS zero t1-seed misses + WARN diff 0/0, **p65 image-native U32k 16.93 / U64k 38.93 / U8k 3.76 / C32k 1.09**, all-TQ zero garble (k8v4 tps 54.64/54.85/236.15; k3v4/q4nc identity faces annexed with **cross-cert dominant-hash proof** — variant families byte-identical to the cert8 gate boots; q3nc = its P6.14 record character), mq annex (cert8-precedent fixed sed), 0 ERROR/Traceback every boot — **`rel/v0.30.0-b70-cert9` certified; superseded by cert10 (next row) — fully valid, its engine bytes carry unchanged into cert10** |
| P6.21 device-compute lanes (ar_quant + FA) | `2026-11-05-p621-arquant/`, `2026-11-05-p621b-fa-headroom/`, `2026-11-05-p621c-fa-esimd-v2/` | **both lanes closed NEGATIVE, no engine change** — ar_quant donor port A/B: quality green, perf −4…−9 % on every U-bar, unadopted; FA headroom ladder r87–r91: stock vxk at 25 % of the measured 147 TF/s peak but every patch-level route falsified; ESIMD dense-FA v2 kernel: numerics-green at **0.33× stock**, ablation ceiling bounds best case ~0.83× (2× bar fails) — kernel laws banked in-source |
| P6.22 TP2 allreduce lane | `2026-11-05-p622-ar-lane/` (+ REDACTION note) | **closed NEGATIVE, no engine change at this phase, cert9 stayed current (no cert10 from P6.22 — nothing serve-affecting)** — census instrument `ar_trace.py` (silent-off) + r94 message-size census/trace (dominant AR 76,677,120 B = 7488×5120 fp16 ×1,584/worker = 88 % of family; in-situ 7.85 ms) + r95 ceiling bench: **stock oneCCL ring 7.78 ms = 2.06× faster than the achievable one-shot torch-primitive ceiling 16.03 ms** (ring 9.85 GB/rank/s concurrent vs copy engines 4.92 GB/s/dir); backend-name law banked (serve dist backend = `xccl`); rig-power-cut durability proven; the one-shot lane stayed closed — the quantized-wire route it left open was taken by P6.24 |
| P6.24 oneCCL fp8-wire allreduce + cert10 | `2026-11-05-p624-fp8-ar/` (r96–r100 evidence + VERDICT + REDACTION notes; build kit `external_patches/oneccl/` + `tools/build_oneccl_arc.sh` + `tools/ar/` runners) | **quantized-AR route deployed as a default-OFF image overlay; cert10 released** (`vllm-arc:v0.30.0-b70-cert10`, image `2f86fc2411fb`): r96 same-version oneCCL 2022.0.0 rebuild FIDELITY PASS (ABI/perf-identical measured — 7.782 vs 7.780 ms at the dominant size; canonical staging + pristine-tree laws banked); r97 `arc-fp8-allreduce.patch` v4 GATE PASS **1.65×** at 76,677,120 B (7.801→4.731 ms; FP8=0 control == anchor; numerics rel-rms 0.02285 == offline bit-exact floor; sector law: u32-packed peer stores, byte-granular remote stores ran 0.41 GB/s); r98 quality ladder task-level parity on one container one-env-line A/B (tps/spec/integrity/long-return green; min-count law live in serve — sub-2M-elem prefills never touch the wire); r99 cert10 release gate ALL GREEN (default-posture battery with FP8 env proven absent; k8v4 carry byte-identical to stock cert9; FP8 knob leg + in-situ STATS wire proof on the shipped image; A/B/A/B perf **beats cert9 on every p65 face**: U32k 16.29/C32k 1.04/U8k 3.27/U64k 35.73 s — the t1 REP-LOOP dispositioned boot-stochastic by the r99c annex); r100 bottleneck scan closed every remaining door with wire-rate measurement (writes 9.85 GB/s fabric ceiling flat in width; reads 2.2× slower; overlap collapses; v4 = 82 % of the wire-bound floor — final); the probe build is an env-gated diagnostic that never enters a cert image — **`rel/v0.30.0-b70-cert10` = current certified release** |
| P6.24b patch audit (2026-11-06) | `external_patches/oneccl/README.md` + `results/2026-11-05-p624-fp8-ar/VERDICT.md` (patch-audit section) | docs/tooling-only — the certified image is UNCHANGED (tag stays at the cert10 release commit): probe patch reconciled to as-built bytes (sha chain reproducible), res-scale invariant documented precisely + frozen-artifact law banked, build-script polish; no re-gate required |
| P6.24d README status-ledger move (2026-11-06) | `docs/STATUS-arc.md` (new detail holder) | docs-only — the certified image is UNCHANGED (tag stays at the cert10 release commit): the three README front-page narrative paragraphs (certified-release detail, project status, improvement-lane status) moved content-faithful to `docs/STATUS-arc.md` (lead-ins became the section headings, docs-internal links made relative — zero factual edits, diff-audited); README keeps the headline + links, currency anchors preserved (check_currency clean); no re-gate required |
| P6.24e best-performance-postures audit (2026-11-06: docs + serve-interface pass on cert10 truth) | CONFIGURATION "Best-performance postures" (new) + envs.md overlay table + `tools/serve_arc_b70.sh` | docs/tooling-only — the certified image is UNCHANGED (tag stays at the cert10 release commit; the FP8 env itself is the r96–r100-certified opt-in): consolidated objective→recipe table (best TTFT = the one-line `CCL_SYCL_ALLREDUCE_FP8=1` opt-in w/ exact placement, best throughput = Step 2 as-is, stall levers, KV-capacity trade); stale "Max TTFT = certified defaults" menu row fixed (cert10 opt-in beats it on every TTFT face); envs.md gains the oneCCL overlay-env table (trio previously pointer-only); uniform boot interface gains `ARC_FP8` pass-through (default unset emits NO env line — served-process env byte-identical to the pre-knob script; `=1` appends the certified opt-in env); README optional-postures + quick-start hint; no re-gate required (default-path posture unchanged) |
| P6.25 overlap-door audit (2026-11-06: r101 microbench — tools + docs only) | `results/2026-11-06-p625-overlap-door/` + `tools/ar/ar_overlap_bench.py`, `tools/ar/r101_overlap_bench.sh`, `tools/ar/r101b_overlap_bench.sh` | **closed NEGATIVE measured — the certified image is UNCHANGED (tag stays at the cert10 release commit; no re-gate required)**: the last unmeasured prefill door (cross-stream AR∥GEMM co-tenancy, async-TP class) closed by the pre-registered gate reading FAIL on both wires — mock best-k 0.969× stock / 0.966× fp8 (overlap slower than serial at every k; coten gemm_inflation 1.97×/1.58× = the device time-slices AR vs GEMM, no idle-EU lane; enqueue async 0.18/0.26 ms); GEMM census closes the GEMM lane at ~99 % of the 147–150 TF/s per-card dense ceiling; every prefill door now measured-closed, remaining headroom ≤1–2 % class; no cert11 — a zero-delta re-release would certify no change; docs updated (README/STATUS/PERFORMANCE/TESTING ledger rows) |
| P6.26 MXFP4 validation + fix + cert11 (2026-11-06) | `results/2026-11-06-p626-mxfp4/` (phase A/B + r105/r105t/r105t2/r105b + REDACTION note; fix commit 7ea0f27b; runners `tools/ar/r10[2-5]*.sh`) | **quark/mxfp4 lane opened; cert11 released** (`vllm-arc:v0.30.0-b70-cert11`, overlay on cert10, tag `rel/v0.30.0-b70-cert11` = c987c2cd): phase-A quark boot failure predicted byte-exact by r102 wheel recon (platform `supported_quantization` omits `quark` → config-validation ValueError) → fix = two `ArcXpuPlatform` overrides (HpuPlatform pattern, wheel-free); phase-B full posture both pre-quantized checkpoints coherent, spec 0.9853/0.9951 (above the fp8 incumbent's 0.9258), decode penalty 1.21×/1.29× < the 2× trigger — **weight-constrained lane; fp8 stays the performance default**; r105 engine battery run-2 PASS at provenance-corrected r99 anchors (run-1 voided: concurrent host download; disclosed in-tree) + r105t/t2 same-day cert10 discriminator (thermal hypothesis refuted with xpu-smi telemetry) + r105b final-image gate PASS (walk 294→294, changed set exactly platform.py+pyc, freeze 0); docs currency pass README/CONFIGURATION/PERFORMANCE/MIGRATION/envs/STATUS/TESTING |
| P6.27 MXFP8 AutoRound validation (2026-11-06) | `results/2026-11-06-p627-mxfp8/` (+ REDACTION note; runners `tools/ar/r106*.sh`) | **mxfp8 lane validated zero-change — NO new cert (cert11 stays current; no engine/wheel byte changed)**: the intel-ai MXFP8-MTP checkpoint boots/serves/speculates on the certified cert11 image via the same `ARC_MODEL`/`ARC_SERVED`/`ARC_QUANT=` knobs (native `XPUMxFp8LinearKernel`, health 285 s incumbent-identical, spec pos1 0.9626 on the quantized drafter, 0/0 erraudit, TTFT below both mxfp4 lanes / decode 1.53× above them — explained); existing deployments: nothing to do |
| P6.31 online MXFP8 W8A16 route + cert21 (2026-11-08: the act-quant-elimination pattern of P6.30 applied to the mxfp8 lane; validated on Qwen3.8-27B + gemma-4-31B; kernel matrix → e2e → adjudication → certified image) | `results/2026-11-08-p631-mxfp8-w8a16/` (r122/r123 kernel matrices + r124/r124b four-leg battery + capacity-law evidence + REDACTION note; `cert21_l1/` = the r125 gate) + `external_patches/vllm-xpu-kernels/arc5_mxfp8_lanes/` (arc5 csrc + build kit) + `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` + `tools/cert21.Dockerfile` | **online mxfp8 made a real lane (cert21, content `c8fdf32a`)**: stock W8A8's per-call act quant + the (f8,f8) joint rate (2.05–2.64× slower than bf16 at prefill, r122) eliminated by the W8A16 route — the arc5 wheel's `_xpu_C.fp8_gemm_w8a16_mxfp8` serves bf16 acts × f8 weights with group-32 e8m0 weight scales (r123: 1.74×/2.68× over stock mx8 at decode/prefill medians, rel ≤ 2e-3); **W4A8 closed-negative** (oneDNN refuses the e4m3×f4 joint dtype — "Only support int4 and fp8 gemm"); r124 qwen A/B: **U8k 3.61 s / U32k 17.07 s / 53.8–55.6 solo / 265.9 agg8 — 1.37–1.55× over stock W8A8 on every face**, spec pos1 0.9804 (control 0.9758); r124b gemma at `ARC_MAXLEN=40960` (the 8-bit-weights capacity law: ~29 GiB TP2 weights → KV gate refusal at 262144 — admission-only cap, faces comparable): **3.87 / 20.94 s / 27.7 / 192.4 = 1.57–1.85×**; adjudication (pre-registered): **checkpoint-fp8 stays the perf default** (qwen 84.5 vs 55.6), **`ARC_MXFP8_MODE=w8a16` = the online-mxfp8 lane route (default)**, mxfp4-w4a16 stays the superior online lane; r125 cert21 gate: fp8 discriminator + w8a16 + w4a16-carry legs all green; existing deployments: nothing to do (default posture unchanged) |
| P6.33 skinny-M kernel family, route banked OFF + cert22 (2026-11-08/09: the P6.32 feasibility map's first execution lane — GEMV-regime M≤8 weight-only kernels built on the arc6 wheel, e2e-falsified, banked default OFF; the transferable finding: eager-loop kbench wins are host-rate artifacts under this stack) | `results/2026-11-08-p633-skinny-m/` (r126 GATE-1/2/3 + ten kernel windows + the r127 five-leg e2e battery + the r127b/c/d/e/f attribution chain + the r127g/g2 cert legs + REDACTION note; `external_patches/vllm-xpu-kernels/arc6_skinny_v017bp/` = the arc6 build kit; runners `tools/ar/r126_*`, `tools/ar/r127*`) + `tools/cert22.Dockerfile` | **arc6 wheel (`0.1.14.1+arc6`, three `skinny_*` M≤8 ops + W4A8 dispatch arms + v0.1.17 Option-B backports 611/645/650/600) + `patches/skinny_m.py` route shipped default OFF behind `ARC_SKINNY_M` (cert22 content `e2457389`)**: kernels correct (r126 GATE-1 rel < 2e-2 every cell; GATE-2 identical-input A/B; GATE-3 W4A8 dispatch) and winning the eager kbench regime (w16pc/g16 6/6 at M=1, 0.80–0.92×), but the r127 e2e battery falsified the reroute — under FULL_DECODE_ONLY XPU-graph replay stock oneDNN runs 420–680 GB/s device-side (r127f 50-call amortized probe) vs the skinny streaming ceiling ~300 GB/s, and the size-1 decode graph pads M→2, so gemma solo regressed −38% (m816) / −42% (g16) while every over-cap path sat exactly on ref; r127c EXP-1 `ARC_SKINNY_M=off` reproduced the r124b ref exactly (27.7/192.3 — causality + arc6-wheel stock-path validation); cert22 behaviorally ≡ cert21 at serve (r127g2: 27.7/192.9 exact, zero route markers); the TC (joint-matrix) skinny redesign is the open future lane — the r127f probe is its required gate; existing deployments: nothing to do (default posture unchanged) |
| P6.30 online-quant W4A16 route + cert20 (2026-11-07/08: online MXFP4/MXFP8 quantization of base checkpoints, validated on gemma-4-31B + qwen3.8-27B; mxfp4 prefill/decode/TTFT made SUPERIOR to fp8 — root issue fixed) | `results/2026-11-08-p630-mxfp4-perf/` (r113→r120h2 full chain: verdict matrices + kernel matrix + jit:gemm proof + REDACTION note) + `external_patches/vllm-xpu-kernels/` (arc4 csrc + build kit + README laws) + `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` + `tools/cert20.Dockerfile` | **online mxfp4 implemented, root-caused, FIXED, and certified superior to online fp8 (cert20, content `a918bee9`)**: the r113 1.9× penalty decomposed to (a) per-call mxfp4 activation quant ≈ GEMM cost at decode Ms and (b) the (f4,f4) joint dtype failing oneDNN gemmstone's JIT gate (counter-operand list has no f4) → `matmul:ref` — both eliminated by the W4A16 route (bf16 acts × f4 weights → `jit:gemm` DPAS via weight decomposition, verbose-proven, rel 1.66e-3); same-image r120c qwen A/B: **U8k 3.66 s / U32k 17.12 s / 62.5–65.2 solo / 301–311 agg8 — superior to online mxfp8 on EVERY axis 1.47–1.75×** + 1.74–1.87× TTFT over the W4A4 control, spec 1.0000; r120g gemma: 1.90–2.21× over W4A4; r120d kernel matrix: w4a16 fastest at every decode M, 2.1–2.7× faster than the e8m0-fp8 GEMM at prefill; v3.13 fp8 question closed (jit:gemm + bidirectional e8m0 scales, still 2.6× slower — group-scale handling, mxfp8 stays the slow lane); plain qwen fp8-online closed as an upstream interface gap (no `fp8` shorthand; QCFG needs online targets — r120h traceback + r120h2 fp8_per_channel probe banked); 4 engineering laws incl. the `_xpu_C` dlopen law; r121 cert20 gate: fp8 discriminator in-band + w4a16 leg green |
| P6.29 sliding-window FA2 route + cert19 (2026-11-07: remaining gemma-4 decode/prefill/TTFT bottlenecks to qwen3.8-27b parity) | `results/2026-11-07-p629/phaseA/r109a/` (r109a/r109a2/r109a3 A/B + guards + triton k-independence anchors; r109a4/r109a5 convention/softcap eliminations; r109a6 global-softmax ref closure — the ~4 prefill-ref misses were a per-chunk-softmax bug in the TEST refs, both kernels correct) + `…/cert19_l1/` (r111 ladder: gemma L1 + r110 ngram refusal + qwen discriminator + `r111_summary.txt`) + `external_patches/vllm/patch_arc_xpu_fa2_slide_route.py` + `tools/cert19.Dockerfile` | **the sliding hd256 wall FIXED and validated; cert19 released** (`vllm-arc:v0.30.0-b70-cert19` = cert18 + the slide route, content `904a3de8`): the triton path on gemma-4's 50/60 sliding layers costs 0.147–0.158 s/call at q8192 fp8-KV k-INDEPENDENT (0.1466 k8192 / 0.1580 k32k / 0.1576 k98k) while the same vxk ESVD FA2 kernel windowed runs 0.006–0.009 s = 18–25x, decode S8x32k 4.38→0.3–0.4 ms = 11–15x; the second wrapper-level dispatch (hd256 + causal sliding window + every cert18 exclusion; window passed as [left, 0]) validated on the image: gemma faces face8k 14.2→4.9 / face32k 41.5→17.2 (qwen-parity — 1.016x of the 16.93 record) / face120k 181.5→110.9 / face32k_2 4.7→0.9 s; decode 19.6→28.4 solo / 73.8→195 agg8 tok/s; G1 3/3; erraudit 0/0; both markers live per rank; qwen discriminator U8k 3.83 / U32k 17.58 s in-band; spec parity dispositioned — MTP impossible (no draft head), ngram structurally refused (upstream `_get_mamba_bufs` assert on pure-attention hybrids, banked r111); `ARC_ATTN_FA2=0` disables both routes |
| P6.28 FA2 route fix + cert18 (2026-11-07: gemma-4 prefill/decode/TTFT root-cause fix) | `results/2026-11-06-p628-gemma4/phaseC/micro_fa2ab/` (r108ac/r108ad 34× A/B + guards) + `…/phaseC/cert18_l1/` (r108ae image validation incl. `r108ae_crash_descale.txt` v2-crash forensics) + `external_patches/vllm/patch_arc_xpu_fa2_route.py` + `tools/cert18.Dockerfile` | **the hd512 full-attention prefill defect FIXED and validated; cert18 released** (`vllm-arc:v0.30.0-b70-cert18` = cert17 + the FA2 route, content `3136adfe`): r108ac's fp16-KV control refuted the r108ab L2-bandwidth-wall reading (10.28 s at 2× bytes — the triton kernel is slow per K-tile iteration); the engine's own `vllm_xpu_kernels` ESVD FA2 kernel does the same call in 0.3197 s = 34× (k98k ~30×, decode 1.5 ms/call; guards green vs a block-table-faithful GQA reference); the wrapper-level dispatch (hd512 + full-attention + no-triton-only-features guards; descale storage normalized — the warm gate's `torch.ones(nkvh).expand` form crashed v2 at boot, fixed v3) validated end-to-end on cert18: gemma faces probe98 35→13 s, face8k 31.5→14.2, face32k 283.4→41.5, face120k 2950.3→181.5, face32k_2 9.8→4.7 s; decode 18.4→19.6 solo / 69.9→73.8 agg8 tok/s; G1 3/3; erraudit 0/0; marker live both ranks; qwen discriminator U8k 3.77 / U32k 17.49 s vs 3.76/16.93 records with marker 0 = incumbent lane untouched; `ARC_ATTN_FA2=0` escape hatch; L2 mxfp8 cert18 re-run + broader L2 quality battery = listed follow-ups |
| env reference | `docs/envs.md` | complete control surface + naming map |
