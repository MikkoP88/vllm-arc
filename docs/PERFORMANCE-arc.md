# Performance — vllm-arc, qwen3.8-27b-fp8 on dual Intel Arc Pro B70

Measured performance of the certified posture (standalone numbers; test
definitions and hardware in [`TESTING-arc.md`](TESTING-arc.md)).

**How to read this file** — three layers, strictest-first:

1. **Measurement methodology** — defines every face, the protocol laws,
   the two prefix-cache tiers, and the provenance rule. Read it first.
2. **Current record** — the **probabilistic draft lane on
   `vllm-arc:v0.30.0-b70-cert23`** (the certified posture since P6.17 —
   the serve-script default; every record
   face measured natively on that lane by the p617 campaign, and the
   cert7–cert11 release gates re-verified the key faces on the tag-built
   image — same engine under `fp8_e4m3`; cert9 adds the P6.20 off-thread
   store, so the default-posture prefill TTFT faces carry the cert9-gate
   values; **cert10 adds the P6.24 oneCCL fp8-wire overlay — default OFF,
   so the default-posture numbers are cert9-identical (proven at the
   gate), and the opt-in `CCL_SYCL_ALLREDUCE_FP8=1` prefill faces are the
   new best on every p65 bar**; **cert11 adds the P6.26 MXFP4 lane on
   top, a config-validation-surface patch only — the fp8 default-posture
   numbers are cert10-identical, proven face-for-face same-day on both
   images at the r105 discriminator legs**), plus the
   posture matrices (the P6.16 prob matrix; the TurboQuant dtype faces,
   P6.14, **servable again since cert8, re-gated on cert9, k8v4 carry
   proven on cert10**; the MXFP4 checkpoint faces, P6.26, **certified on
   cert11**; the MXFP8 AutoRound faces, P6.27, ride the same cert11
   image zero-change; **cert12–cert18 (P6.28) add the gemma-4 validation
   chain — the prefill launch-config table, the warm-gate v4, the
   lmcint kit behind the default-off ARC_LMC posture, and the cert18
   FA2 route (gemma's hd512 full-attention layers → the ESVD FA2 kernel:
   2.2–16.3× gemma prefill, +6 % decode; qwen3.8 never hits it — same-day
   cert18 discriminator re-ran U8k/U32k green at 3.77/17.49 s with the
   route marker 0, so the qwen rows below carry both the cert11-era
   records and the cert18 discriminator proof that the incumbent lane is
   untouched)**; **cert19 (P6.29) adds the sliding-window FA2 route —
   gemma's 50/60 sliding hd256 layers → the same ESVD FA2 kernel
   windowed: 2.4–2.9× more gemma prefill on top of cert18 (face32k at
   qwen-parity), decode +45 % solo / 2.64× agg8; the qwen discriminator
   re-ran green in-band at 3.83/17.58 s (the guarded sliding-hd256 class
   also occurs in qwen traffic — no regression, erraudit 0/0)**);
   **cert20 (P6.30) adds the online-quant W4A16 lane — the arc4
   `vllm_xpu_kernels` wheel + the `mxfp4_w4a16` reroute behind
   `ARC_MXFP4_MODE`; the fp8/checkpoint lanes are byte-untouched (qwen
   fp8 discriminator re-ran green at the r121 gate), and the online
   mxfp4/mxfp8 faces are a new P6.30 table below**);
   **cert21 (P6.31) adds the online-MXFP8 W8A16 route — the arc5
   wheel + the `mxfp8_w8a16` reroute behind `ARC_MXFP8_MODE`
   (1.37–1.85× over stock W8A8 on both models; checkpoint-fp8 stays
   the perf default, mxfp4-w4a16 the superior online lane; the
   fp8/checkpoint/mxfp4 lanes are byte-untouched — r125 three-lane
   gate: discriminator + w8a16 + w4a16-carry legs all green; the
   P6.31 online-mxfp8 table is in the online-quant section below**);
   **cert22 (P6.33) adds the skinny-M kernel family banked OFF — the
   arc6 wheel (`skinny_*` M≤8 GEMV-regime ops) + the `skinny_m` route
   patch ship behind `ARC_SKINNY_M` (unset/off = stock routes = the
   certified default after the r127 e2e falsification: under XPU-graph
   replay stock oneDNN runs 420–680 GB/s device-side and the recorded
   skinny kernels regressed gemma solo decode −38/−42%), so every
   default-posture number is cert21-identical — proven at the r127g2
   cert leg (gemma solo 27.7 / agg8 192.9, the exact r124b reference
   values, zero route markers)** and the
   current-era analyses
   (prefill anatomy, op-class trace). The pre-P6.17 greedy-lane record
   (r70, cert4-native) is retained as labeled history.
3. **History** — every pre-P6.17 measurement, labeled by image and
   posture, kept as the decision record (k-depth, load shape, soak).
   History numbers are never mixed into current-record tables
   (comparator law).

All certified faces: first-traffic fresh boot, timers off, XPU graphs
on, TP2. Temp-0 faces are posture-invariant (drafts are argmax at t0 —
proof in the draft-sampling section); sampled faces are measured on the
certified probabilistic lane. Opt-in faces (TQ dtypes) run the same
protocol laws with exactly one flipped variable; their tables are
labeled. **Recommended MTP depth = k3** (r46t campaign, reproduced
natively on cert4 by r70). **Certified draft posture = probabilistic**
(P6.17 — the serve-script default; `ARC_SPEC_METHOD=greedy` is the
opt-out historical lane; section below).

## Measurement methodology — how every number in this file is produced

**Provenance rule (2026-10-31; updated for P6.17, P6.18, P6.19):** the
current record cites **only faces measured natively on the certified
image line under the certified (probabilistic) draft posture** — the
p617 campaign on `vllm-arc:v0.30.0-b70-cert6` (every record face
re-measured on that lane) plus the P6.16 posture legs on cert6
(`results/2026-11-03-p616b-probmatrix/`, `…-p616c-spechook/`), with
the cert7 and cert8 release gates re-verifying the key faces on the
tag-built images (`results/2026-11-03-p618-cert7/`,
`results/2026-11-04-p619-tqverify/` — byte-identical engine under
`fp8_e4m3`; the P6.18 guard was config-time only and is lifted on
cert8). The TurboQuant dtype faces (P6.14) are cert6-era faces —
**servable again since cert8 (P6.19 fix)** — and carry their
own table below. Numbers measured on the pre-P6.17 greedy posture (r70
and earlier) and from earlier images (cert / cert2 / cert3 / p46) are
history — labeled by image and posture, never mixed into
current-record tables.

### Rig, image, and posture identity

| Layer | Identity |
|---|---|
| Hardware | dual Intel Arc Pro B70 (48 GB each), TP2, one card per socket (`ZE_AFFINITY_MASK=0,1`) |
| Engine image | **current**: `vllm-arc:v0.30.0-b70-cert23` — tag `rel/v0.30.0-b70-cert23` (cert22 + the P6.35 graph-exec double-buffer: `vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`, VLLM_ARC_GRAPH_EXEC_DB default 1 (=0 kill-switch proven); replay max 24 → 0.37 ms, engine serial 40.8 → 12.7 ms/step, **E2E-neutral at every measured posture** (fp8-k3 94.5 vs 94.3 tok/s on/off; mx4k5 121.0 vs 123.9) — costless hygiene fix, +29 s boot; gate r135+r135b, evidence `results/2026-11-10-p635-telemetry/`; **P6.35 posture gains (serve-line choices)**: k5 +41% (115.2 tok/s), mxfp4-W4A16+k5 123.9 (+52%) TTFT 3.98 s; **P6.35-V full matrix (r136+r136b, 11 cells)**: qwen mx4-W4A16 k5 re-read **128.3 tok/s = +36% vs clean base 94.5 (project best)**, gemma mx4-W4A16 **37.1 (+43% vs mxfp8)**, gemma online fp8 26.6 (new capability), mxfp8 deprecated for decode, best TTFT fp8-k5 3.48 s — evidence `results/2026-11-10-p635-telemetry/vmatrix/`. Prior: `vllm-arc:v0.30.0-b70-cert22` — tag `rel/v0.30.0-b70-cert22` (cert21 + the P6.33 skinny-M kernel family banked OFF: arc6 `vllm_xpu_kernels` wheel `0.1.14.1+arc6` (adds the three GEMV-regime M≤8 weight-only ops `_xpu_C.skinny_fp8_gemm_w8a16`/`skinny_fp8_gemm_w8a16_mxfp8`/`skinny_fp4_gemm_w4a16` — f8/f4 weight streaming, register dequant, deterministic fixed-order split-K; plus W4A8 dispatch case arms + the v0.1.17 Option-B backports #611/#645/#650/#600; sha256 `484e06c1…b7156`, ~352 MB — rig-only per the 100 MB artifact law) + `vllm_gaudi/xpu/patches/skinny_m.py` (the `ARC_SKINNY_M`-gated M≤8 reroute — **unset/off = stock routes, the certified default after the r127 e2e falsification**; `on` = opt-in with per-lane M caps) + the platform `_XPU_PATCH_MODULES` entry + the envs registry entry, via `tools/cert22.Dockerfile`; **serve behavior byte-identical to cert21** — r127g2 cert leg: gemma default posture 27.7 solo / 192.9 agg8, the exact r124b reference values, zero route markers; IMAGE_MANIFEST carries the cert22 row + `cert22_fork_sha`/`cert22_built`/`cert22_vxk_wheel_sha256`; gate evidence `results/2026-11-08-p633-skinny-m/`). Prior-cert context: cert21 — tag `rel/v0.30.0-b70-cert21` (cert20 + the P6.31 online-MXFP8 W8A16 route: arc5 `vllm_xpu_kernels` wheel `0.1.14.1+arc5` (adds `_xpu_C.fp8_gemm_w8a16_mxfp8` — bf16 activations × f8 weights, group-32 e8m0 weight scales, zero per-call act quant; also carries `_xpu_C.fp4_gemm_w4a8`, the oneDNN-refused W4A8 probe op; sha256 `219268cf…5399a`, 352 MB — rig-only per the 100 MB artifact law) + `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` reroute + the `mxfp4_w4a16.py` w4a8 probe branch + the platform `_XPU_PATCH_MODULES` entry, via `tools/cert21.Dockerfile`; knob `ARC_MXFP8_MODE` (unset/`w8a16` = route, `w8a8` = stock control); IMAGE_MANIFEST carries the cert21 row + `cert21_fork_sha`/`cert21_built`/`cert21_vxk_wheel_sha256`; gate evidence `results/2026-11-08-p631-mxfp8-w8a16/cert21_l1/` — r125 three-lane gate: qwen fp8 discriminator in-band + w8a16 route-markers/dispatch/battery green + the mxfp4 w4a16 carry leg green). Prior-cert context: cert20 — tag `rel/v0.30.0-b70-cert20` (cert19 + the P6.30 online-MXFP4 W4A16 route: arc4 `vllm_xpu_kernels` wheel `0.1.14.1+arc4` (adds `_xpu_C.fp4_gemm_w4a16` — bf16 activations × f4 weights, group-32 e8m0 weight scales → oneDNN `jit:gemm` DPAS via weight decomposition; sha256 `c675f1a3…d14c08`, 352 MB — rig-only per the 100 MB artifact law) + `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` reroute + the platform `_XPU_PATCH_MODULES` entry, via `tools/cert20.Dockerfile`; knob `ARC_MXFP4_MODE` (unset/`w4a16` = route, `w4a4` = stock control); IMAGE_MANIFEST carries the cert20 row + `cert20_fork_sha`/`cert20_built`/`cert20_vxk_wheel_sha256`; gate evidence `results/2026-11-08-p630-mxfp4-perf/cert20_l1/` — r121: qwen fp8 discriminator in-band + the w4a16 leg markers/dispatch/battery green). Prior-cert context: cert19 — tag `rel/v0.30.0-b70-cert19` (cert18 + the P6.29 sliding-window FA2 route: `external_patches/vllm/patch_arc_xpu_fa2_slide_route.py` via `tools/cert19.Dockerfile` — sliding-window head-256 calls (window (1023,0) class) dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel windowed in the same triton unified-attention wrapper; `ARC_ATTN_FA2=0` disables both routes; IMAGE_MANIFEST carries the cert19 row + `cert19_fork_sha`; gate evidence `results/2026-11-07-p629/cert19_l1/` — BOTH markers fired on both TP ranks in production serving, gemma faces 2.4–2.9× faster than cert18 (face32k at qwen-parity), decode +45 % solo / 2.64× agg8, qwen discriminator in-band). Prior-cert context: cert18 — tag `rel/v0.30.0-b70-cert18` (cert17 + the P6.28 FA2 route: `external_patches/vllm/patch_arc_xpu_fa2_route.py` via `tools/cert18.Dockerfile` — full-attention head-512 calls dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel in the triton unified-attention wrapper; `ARC_ATTN_FA2=0` escape hatch; IMAGE_MANIFEST carries the cert18 row + `cert18_fork_sha`; gate evidence `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/` — the FA2 marker fired on both TP ranks in production serving, gemma faces 2.2–16.3× faster, qwen discriminator marker 0). Prior-cert context: cert17 — tag `rel/v0.30.0-b70-cert17` (the cert11 engine + the P6.28 chain: cert12 launcher table `a964e047` + cert16 warm-gate v4 `5d895fef` + cert17 lmcint kit `e5e725b4`; built via the `tools/cert16/17.Dockerfile` lineage with the external_patches kit; IMAGE_MANIFEST carries the cert12 row + cert16/17 description rows — their fork_sha/built stamps are empty in the shipped manifest, a cosmetic build-script wart proven around by the banked build logs + the r108z gate-fire boot). Prior-cert context: cert11 — tag `rel/v0.30.0-b70-cert11` (commit `c987c2cd`), overlay on cert10 from the tag archive (`tools/cert11.Dockerfile`, platform.py md5 `e48abe9cd9de`), image manifest `sha256:928a6dfe…`; `IMAGE_MANIFEST` rows: `fork_sha == aa2ee17d…` + `oneccl=2022.0.0+arc1` overlay rows (both carried from cert10) + `cert11_fork_sha == c987c2cd…`; **exhaustive `vllm_gaudi/` md5 walk vs cert10: 294→294 files, changed set EXACTLY `vllm_gaudi/xpu/platform.py` + its regenerated pyc** (r105b gate, `results/2026-11-06-p626-mxfp4/`); pip freeze diff 0. Prior-cert context: **cert10** (tag `rel/v0.30.0-b70-cert10`, image `2f86fc2411fb`, `IMAGE_MANIFEST.fork_sha == aa2ee17d…` preserved from cert9 + the `oneccl=2022.0.0+arc1` overlay rows; **exhaustive `vllm_gaudi/` md5 walk vs cert9: diff_lines = 0** — cert9's vllm-layer certifications carried; only the comm lib changed), two-file fat-lib overlay sha-pinned at both SONAME paths (cert10 gate, `results/2026-11-05-p624-fp8-ar/`). The current-record numbers below were measured natively on cert6 under the certified probabilistic draft posture (p617 campaign) and the key faces re-verified at the cert7–cert10 release gates (t0 in-family — cert10 default-posture gate 300.39 steady / 84.52–84.51 solo with the FP8 env proven absent — integrity/det/spec green — cert10 spec pos1 0.9258 byte-exact on BOTH the default and FP8=1 legs — long-return t0 lanes 100 % clean on every dtype; the one t1-seed window flag dispositioned benign with the W14 incumbent control at cert8; the cert9-gate TQ identity faces carry the cross-cert dominant-hash proof — same variant families as the cert8 gate). cert10 = cert9 engine + the **P6.24 oneCCL fp8-wire overlay** (same-version 2022.0.0 rebuild + e4m3-wire large-ring allreduce, **default OFF** — `CCL_SYCL_ALLREDUCE_FP8=1` opts in; opt-in faces below). cert9 = cert8 engine + the **P6.20 prefill host-path patch** (L3 off-thread LMCache store, default ON via `VLLM_ARC_LMC_ASYNC_STORE` — device ops stay on the engine's in-order stream, only the host wait moves to the `arc-lmc-store` worker; L6 dispatch-late handoff; `patches/step_trace.py` diagnostics, silent-off). Under `fp8_e4m3` the compute bytes are identical to cert6–cert9 — the P6.20 delta is WHERE the store body's D2H gathers + CPU insert run (off-thread), worth **−0.74 s U32k / −1.64 s U64k / −0.08 s U8k TTFT**; KV pool 400,739 == cert7 byte-value, run7n t0 296.91/84.54/84.48 in-family. cert8 (`e47739b0`, image `b212fbca59ec`) remains fully valid |
| Wheels | md5-pinned at build (`tools/wheel_guard.sh`, 4 pins: vllm/triton/vxk/lmcache 0.5.6+arc1) |
| Serve line | the certified 25-var env chain + CLI ([`CONFIGURATION-arc.md`](CONFIGURATION-arc.md)): MTP k3 **+ probabilistic drafting (the P6.17 certified default)**, `mode:none`, FULL_DECODE_ONLY graphs (85 capture sizes, 1–320), MNBT 8192, LMCacheConnectorV1 `kv_both` |
| p617 campaign (the comprehensive prob-lane certification) | 5 fresh boots on cert6 — prob comprehensive leg (a), prob family leg (b), no-spec A/B leg (c), default-path validation boot (d), greedy opt-out boot (e) — evidence `results/2026-11-03-p617a-probcert/` |
| r70 campaign (the comprehensive greedy-lane measurement; history since P6.17) | 3 fresh boots on cert4 — greedy+sweep leg (a), no-spec leg (b), soak leg (c) — evidence `results/2026-10-31-p70-r70-cert4-comprehensive/` |

### Face definitions (exact)

| Face | Definition |
|---|---|
| agg4x256 | 4 concurrent chat requests, `max_tokens=256`, prompt "Count from 1 to 1000.", thinking off; throughput = Σtokens/wall across the 4-thread join. Two runs per battery; **run #1 after boot or an idle window is warmup-contaminated, run #2 is the steady face** |
| solo200 | one chat request, `max_tokens=200`, same prompt; tps = tokens/wall **including TTFT** |
| xhigh sweep | the agg/solo battery at temperature 0.2–1.0 with the real-world sampling set (top_p 0.95 / top_k 20 / min_p 0 / presence_penalty 0 / repetition_penalty 1.0) and `enable_thinking: true, reasoning_effort: "xhigh"` |
| alt face | temperature 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5, non-thinking (near-deterministic continuation regime) |
| U-bars (U8k/U16k/U32k/U64k) | SOLO streaming completion on a uniform synthetic prompt, sized via `/tokenize` to the face target, per-face seed (fresh prompts every boot — prefix-cache hit must be ≈0 or the face is invalid), `max_tokens=32`; TTFT = send → first content chunk; prefill rate = prompt_tokens/TTFT |
| warm t1/t2 | the U-bar request fired twice identically: t1 = cold (cached ≈ 0), t2 = the same request seconds later; plus 3× warm-hit re-fires (U8k) for a stable warm-hit number |
| no-spec A/B | the same battery faces on a second fresh boot with `--speculative-config` removed; spec counters verified absent (`--expect-off`) before measuring |
| soak | fresh boot; cycles of (agg battery + 24-prompt integrity probe + health check + counter snapshot) every ~5 min for the soak window |

### Protocol laws (apply to every certified face)

- **Fresh-boot, first-traffic**: every battery/probe runs on a freshly
  booted engine; the first request after boot exercises the p12b warmup
  interlock (the survival face) and is never scored.
- **Timers off** (`ARC_TRACE_FLIP=1` interlock) and **XPU graphs on**
  (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, FULL_DECODE_ONLY) — standing rig laws.
- **Same-face comparator law**: only same-face, same-session numbers
  compare (thinking on/off and battery warmth are confounders —
  `results/2026-10-31-p6-r47c-p48-056gates/VERDICT.md`). Cross-boot
  same-face compares are reported as families (ranges), not point
  deltas. Sampling params at temperature 0 are inert (argmax survives
  top_k/top_p), so greedy faces stay comparable across batteries.

### The two prefix-reuse tiers — which cache serves a warm hit

The certified line runs two tiers at once:

1. **Engine APC** (`--enable-prefix-caching`, block 64): GPU-resident
   prefix blocks, hit-checked by the scheduler; engine telemetry =
   "Prefix cache hit rate" in the serve log.
2. **LMCacheConnectorV1** (`kv_both`; CPU 100 GiB local tier, 1024-tok
   chunks): serves prefixes no longer GPU-resident (eviction recovery,
   long-horizon cross-request reuse); engine telemetry = "External
   prefix cache hit rate" + LMCache lookup/store log lines.

**Measured attribution (r70)**: during the warm t2 faces the engine log
shows prefix-cache hit rate climbing 41→46 % with **External prefix
cache hit rate 0.0 %**, and the window holds 62 asynchronous LMCache
*store* events (after the cold requests) and **zero retrievals** — a
same-boot warm re-hit is served entirely by **engine APC**; LMCache's
activity in that window is store-only. LMCache retrieval is the tier
that matters on longer horizons — its faces are the p65/p48 long-context
evidence, not the warm-hit face.

### Acceptance telemetry (speculative decoding)

Upstream-0.30 `/metrics` counters: drafts, draft tokens, accepted
tokens, per-position acceptance. Definitions used throughout:
**acceptance % = accepted/draft tokens**; **mean accepted length =
accepted/drafts + 1** (the +1 bonus token). Windows are counter deltas
snapshotted before/after each battery, so each face's acceptance covers
that face's traffic only.

## Current certified image — cert23 (`rel/v0.30.0-b70-cert23`) + probabilistic draft lane (P6.17): the current record

**Current release = cert22 (= cert21 + the P6.33 skinny-M kernel
family banked OFF: the arc6 wheel swap + the `skinny_m` route patch
behind `ARC_SKINNY_M` — unset/off = stock routes, the certified
default, so serve behavior is cert21-identical, proven at the r127g2
cert leg; every cert21 byte carries — the incumbent-lane record
numbers below are the cert6–cert11-era measurements (re-verified
through the cert11 gate, discriminated in-band at cert19/cert20),
re-discriminated green at the r125 cert21 gate (fp8 + w8a16 + w4a16
carry legs) and re-verified in-band at the r127 P6.33 legs on the
arc6 wheel (qwen legs A/B/C within ±2.6% noise, agg8/TTFT/spec on
ref); the P6.30/P6.31 online-quant lane tables are
further down;
`results/2026-11-08-p630-mxfp4-perf/` +
`results/2026-11-08-p631-mxfp8-w8a16/` +
`results/2026-11-08-p633-skinny-m/`). Current certified posture =
probabilistic
drafting (P6.17, the serve-script default).** Every record number below
was measured natively on the cert6 image under
`ARC_SPEC_METHOD=probabilistic` (p617 campaign, 2026-11-03, fresh boot
per leg; the serve script's hook appends the upstream-config-native
`draft_sample_method` key — P6.16). cert8 is the same engine under
`fp8_e4m3` — the deltas are the P6.19 TQ spec-verify fix (wheel-side
hook, TQ dtypes only; the hooked module never imports under fp8) and
the lifted P6.18 refusal — and the cert8 release gate re-verified the
record faces on the tag-built image: t0 agg#2 300.65 in-family (solo
84.53/84.57; cert7 gate 299.73 / 84.50/84.44), integrity 5/5 +
determinism 3/3, spec pos1 0.9258 PASS, 0 ERROR / 0 Traceback, and the
long-return gate on every corruption/looping face across 3 churn cycles
(t1.0-xhigh 221.36 solo 70.98/68.14 — the face's highest-variance
posture, in-family with cert7's banked 218.29; KV pool 400,739 tokens
== the cert7/p618b byte-value) —
`results/2026-11-04-p619-tqverify/` (gate VERDICT) +
`results/2026-11-03-p618-cert7/` (cert7 record).
**cert9 = cert8 + the P6.20 prefill host-path patch** (L3 off-thread
LMCache store, default ON, `VLLM_ARC_LMC_ASYNC_STORE=0` kill switch; L6
dispatch-late store handoff; `step_trace` diagnostics, silent-off) —
the store body's D2H gathers + CPU insert leave the engine thread
(device ops stay on the engine's in-order stream), and the cert9 gate
re-verified everything image-native with **no env overrides**: t0
agg#2 296.91 (solo 84.54/84.48; agg#1 276.88 first-run warm-up),
t1.0-xhigh agg 240.06 steady (133.47 first-sampled-face transient, the
documented warm-up face; solo 67.04/64.15 in-family), integrity 5/5 +
determinism 3/3, **spec gate at exactly the cert8 banked values (pos1
0.9258, acceptance 0.837, mean accepted length 3.51)**, long-return
bar PASS with zero t1-seed misses on fp8, 0 ERROR / 0 Traceback, WARN
class-set identical to cert8 (+0 −0), KV pool 400,739 tokens — and the
P6.20 TTFT wins on the same gate: **U32k 16.93 s / U64k 38.93 s / U8k
3.76 s / C32k re-hit 1.09 s** vs the cert8-posture faces 17.9–18.1 /
39.6–41.5 / 3.84 (r81 interleave A/B/A/B: −0.74 / −1.64 / −0.08 s) —
`results/2026-11-04-p620-prefill-opt/`.
**cert10 = cert9 engine + the P6.24 oneCCL fp8-wire overlay** (default
OFF — the certified default posture serves the stock path; proven
`FP8_ENV_SET=0` at the gate) — the cert10 release gate re-verified the
battery on both postures: default leg t0 agg 300.39 steady (268.89
first-run warm-up; solo 84.52/84.51), integrity 5/5 + determinism 3/3,
spec pos1 0.9258, tps 61.18/61.27/328.44 in-band, long-return t0
single-hash + 3 churn cycles clean, erraudit 0/0; **FP8=1 leg** flip
8.7 s, t0 agg 299.90 steady / solo 84.57/84.51, spec pos1 0.9258
byte-exact, KV pool 398,363 (the −0.6 % overlay init footprint,
quality-neutral — measured), in-situ wire proof on the shipped image
(40 `CCL fp8` STATS lines, count 7,459,840, both ranks agreeing on the
res scale); **k8v4 TQ carry** essay/code/simple byte-identical to the
stock cert9 k8v4 leg — `results/2026-11-05-p624-fp8-ar/` (r96→r100
phase ladder + the r99c t1 annex; the t1 REP-LOOP disposition:
boot-stochastic, non-reproducing on fresh boots, t0 green everywhere).
**cert11 = cert10 + the P6.26 platform patch (overlay on cert10 —
every byte carries except `vllm_gaudi/xpu/platform.py` + its pyc;
oneCCL overlay rows preserved)**: `quark` in
`ArcXpuPlatform.supported_quantization` + `supports_mx() → True`,
opening the MXFP4 checkpoint lane with **zero dispatch change for
existing lanes** (kernel selection is table-keyed; the patch is
config-validation-surface only — runtime-inertness for the fp8 lane
grep-proven, `supports_mx` consumers are all mxfp4/mxfp8/quark/MoE
paths). The r105 engine battery (cert11c full-rebuild vehicle): t0
agg 300.48 steady (271.94 warm-up; solo 84.56/84.55), tps agg8
326.43, spec pos1 0.9258, LONGRETURN PASS + zero garble across churn,
ERROR=0 — every face green against the provenance-corrected r99
same-posture anchors (TTFT −0.3…−9.9% vs the r99 off-leg means;
the originally quoted anchors were memory-recalled and wrong — the
anchor correction + the full discriminator chain are disclosed in
`results/2026-11-06-p626-mxfp4/README.md`). r105b final-image gate
PASS (294→294 file walk, changed set exactly platform.py + pyc;
overlay shas carried; freeze 0; spot boot + spec PASS pos1 0.9217).
**MXFP4 lane faces (phase B, full posture)**: rhmxfp4 (mixed CT —
fp8-block attn + native mxfp4 MLP) TTFT U32k 29.01 / C32k 1.68 /
U8k 7.37 / U64k 62.46 s, bystander decode p50 64.7 ms, spec pos1
0.9853; qurk (all-layer W4A4) 32.28 / 3.84 / 7.73 / 70.61 s, p50
68.8 ms, spec pos1 0.9951; both decode penalties 1.21×/1.29× vs the
fp8 incumbent — under the 2× collapse trigger. Disposition: mxfp4 =
certified weight-constrained lane (weights 9.03 GiB/rank on qurk vs
~13.6 fp8); **fp8 stays the performance default** —
`results/2026-11-06-p626-mxfp4/`.
**MXFP8 lane faces (P6.27, r106b, full posture on cert11 — zero
engine change)**: intel-ai AutoRound checkpoint (W8A8-dynamic,
group-32 E8M0, quantized MTP drafter) on the native
`XPUMxFp8LinearKernel` (oneDNN `fp8_gemm`): TTFT U32k 26.67 / C32k
1.52 / U8k 6.36 / U64k 59.52 s — **below both mxfp4 lanes on every
TTFT face**; bystander decode p50 81.8 ms (1.53× vs fp8 — above both
mxfp4 lanes: decode is weight-byte-bound and W8 reads 2× the W4
bytes, plus per-linear `xpu_mxfp8_quantize` + E8M0 scale handling vs
the incumbent's fused block-scaled lane; prefill amortizes the quant
pass); spec pos1 0.9626 / acceptance 0.9174 on the QUANTIZED drafter;
health 285 s incumbent-identical; 0/0 erraudit. Under the 2× trigger
everywhere → REPORT lane: the AutoRound-import route (weights 14.74
GiB/rank ≈ fp8, no capacity win, no quality loss) —
`results/2026-11-06-p627-mxfp8/`.
The opt-in prefill faces are in the TTFT section below. Nothing validated
beats the certified default (the two faster-on-paper postures are
rejected: k5-depth dropped — its greedy edge inverts under sampling;
compile — Appendix H law #1 acceptance gate, p67). The opt-in slice
profile (experimental, P6.13) is NOT part of the certified default; its
cert5-native face: U64k TTFT 50.77 s (cert4 51.95), W_U64k_start max
1602 vs 1997 ms. The TurboQuant dtype faces are **servable again since
cert8** (P6.19 spec-verify fix — zero garble all four presets; their
table is below; re-gated clean on cert9 — dominant identity hashes
byte-identical across the two gates). How to run: the 25-var chain in
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md).

### Gemma-4-31B validation faces (P6.28+P6.29; cert19 L1 current — r111 — with the cert18/cert17-native columns kept as labeled history)

Second-model lane (2026-11-06/07): gemma-4-31B (60-layer
two-head-size hybrid — 50 sliding-window hd256 layers + 10 full-attention
hd512 layers per rank 16Q/8KV hd256 win1024 and 16Q/2KV hd512) on the
certified posture `graph=1 spec=0 ARC_LMC=0 ARC_RPC_TIMEOUT=1200`, TP2,
chat template gemma4. Client walls, temperature 0, solo, finish=length.

| face | **L1 fp8 cert19 (slide route)** | L1 fp8 cert18 (history) | L1 fp8 cert17 (history) | L2 mxfp8 cert17 (maxlen 95000) |
|---|---|---|---|---|
| probe98 (98-token gen probe) | **5 s** OK coherent | 13 s OK coherent | 35 s OK coherent | 35 s OK coherent |
| G1 factual / code / reason | 3/3 | 3/3 (reason "4:40 pm") | 3/3 | 2/3 — **reason MISS** (single arithmetic probe, W8 quant; broader L2 quality battery = follow-up) |
| face8k (8,028 ptok / 16 ctok) | **4.9 s** | 14.2 s | 31.5 s | 35.4 s |
| face32k (32,028 ptok) | **17.2 s** — **qwen-parity** (1.016× of the U32k 16.93 record) | 41.5 s (6.8×) | 283.4 s | 297.4 s |
| face120k / face90k (maxlen-bound) | **110.9 s** (face120k) | 181.5 s (face120k, 16.3×) | 2950.3 s (face120k) | 1819.5 s (face90k) |
| face32k_2 (identical prompt, prefix re-hit) | **0.9 s** | 4.7 s | 9.8 s | 10.3 s |
| decode solo 1×256 (×3 reps) | **28.4–28.5 tok/s** | 19.6 tok/s | 18.4–18.5 tok/s | 11.9 tok/s |
| decode agg8×256 (steady rep) | **194.5–195.5 tok/s** | 73.8 tok/s | 69.9 tok/s | 54.6 tok/s |
| erraudit (whole battery boot) | 0 ERROR / 0 Traceback, JIT 0 | 0 ERROR / 0 Traceback | 0 ERROR / 0 Traceback | 0 ERROR / 0 Traceback |

cert19 parity analysis vs the qwen3.8-27b fp8 incumbent records:
face32k is AT parity (17.2 vs 16.93 s = 1.016×); face8k is 1.30× of U8k
(4.9 vs 3.76 s — the remaining delta is the 10 full-attention hd512
layers' k-dependence + GEMM floor); face120k runs 1.008 ms/prompt-token
vs qwen U64k's 0.594 (110,028 vs 65,536 ptok faces — the hd512 quadratic
share grows with k); face32k_2 re-hit 0.9 s beats the qwen C32k 1.09 s
record. Decode: cert19 agg8 195 tok/s is 1.57× qwen's own NO-SPEC agg
(124–127); vs qwen's MTP-k3 spec lane (solo 83.7–84.9 / agg8 326–328)
the gap is 2.9×/1.7× and is **structurally closed to parity work**:
gemma-4's checkpoint ships no MTP/draft head (nothing to draft from),
and the draft-free ngram lane is refused by the upstream v1 spec path —
`_get_mamba_bufs()` asserts "no mamba layers in the model" on
pure-attention hybrids (both ranks, first request; banked
`results/2026-11-07-p629/cert19_l1/ngram_crash_excerpt.txt`). Spec
parity for gemma-4 on this stack is therefore not reachable by any
lawful lane; the certified decode posture is `ARC_SPEC=0`.

L2 mxfp8 was not re-run on cert18/cert19 (listed follow-up with the
broader L2 quality battery); both FA2 routes are engine-level and
model-agnostic for their classes, so L2 is expected to benefit
identically — unmeasured until that leg runs.

**Prefill root cause, layers 1–2 (MAJOR ISSUE — FIXED at cert18;
history of the close).** The first prefill cost was the 10/60
full-attention hd512 layers: the shared triton unified-attention 2D path
runs them at 10.81 s/call (0.71 TF/s) at q=8192/k=32768 fp8-KV.
r108aa/r108ab first read this as a structural L2-bandwidth wall (~37 GB
of L2-side reads per call = 3.4 TB/s effective; BLOCK_M SLM-capped at
64 — M128 launch-fails; time ∝ 1/BLOCK_Q; TILE/stages/dtype/nkvh noise;
the cert12 table measured AT the ceiling, sweep winner 1.01×, numeric
guard 0.00000). r108ac's fp16-KV control then REFUTED the bandwidth
story — 10.28 s at 2× the bytes: the kernel is slow PER K-TILE
ITERATION (~10.3 µs vs 2.4 µs windowed), a kernel defect, not a
memory-system wall. The engine already ships the fast kernel for exactly
this shape class — the `vllm_xpu_kernels` ESVD FA2 kernel
(`_vllm_fa2_C.abi3.so`) — but the triton backend never dispatches to it.
r108ac/r108ad measured the route at production geometry: **FA2 0.3197
s/call = 24.1 TF/s = 34× (k98k ~30×; decode S8×32k 1.5 ms/call)**,
numeric guards green. cert18 = the wrapper-level dispatch (hd512 +
full-attention guards; descale storage normalized — r108ae v3),
validated end-to-end (faces 14.2/41.5/181.5/4.7 s, decode 19.6/73.8,
qwen discriminator marker-0 green).

**Prefill root cause, layer 3 — the remaining wall (P6.29; FIXED at
cert19).** After cert18, the dominant residual was the 50/60
SLIDING hd256 layers: the triton path costs 0.147–0.158 s/call at
q8192 fp8-KV and is **k-INDEPENDENT** (0.1466 k8192 / 0.1580 k32k /
0.1576 k98k, r109a3) — i.e. the kernel never amortizes its window
pruning across K, while the same vxk ESVD FA2 kernel WINDOWED runs
0.006–0.009 s (window-bounded) = 18–25×, decode S8×32k 4.38→0.3–0.4 ms
= 11–15× (r109a/r109a3). 50 layers × ~0.148 s ≈ 7.4 s per q8192 chunk —
the bulk of cert18's face8k 14.2 s. cert19 = the second wrapper-level
dispatch (hd256 + causal sliding window (left ≥ 0, right ≤ 0) + every
cert18 triton-only exclusion; window passed as `[left, 0]`;
`external_patches/vllm/patch_arc_xpu_fa2_slide_route.py` — asserts the
cert18 v3 state byte-exact before patching). Numeric guards: kernels
agree 1.2e-4…9.8e-4 (fp16/fp8/bf16-q, prefill+decode) and BOTH match
global-softmax table-faithful window-masked GQA torch refs at
1.3e-4…1.1e-3 prefill / 8e-5 decode (r109a6) — note the r108ad…r109a4
"ref misses ~4" (full class too) were a harness bug in the TEST refs
(per-chunk softmax normalization), not a kernel issue; the softcap
theory was eliminated separately (the triton wrapper cannot receive
`None` — TypeError at r109a5; production passes 0.0 and
`USE_SOFTCAP=(0.0>0)` skips it). Validated end-to-end on cert19 (r111):
faces 4.9/17.2/110.9/0.9 s (the 70.6 s face120k saving matches the 7.4 s
× ~9.5 chunks projection exactly), decode 28.4 solo / 195 agg8 tok/s,
G1 3/3, erraudit 0/0, JIT 0, BOTH route markers live per rank, qwen
discriminator U8k 3.83 / U32k 17.58 s in-band (±5 % of 3.76/16.93; fa2
marker 0; the slide marker's 2 lines on the qwen leg = the guarded
sliding-hd256 class occurs in qwen traffic/tune-warmup too — in-band +
erraudit 0/0, no regression). Decode notes: the slide route lifted L1
solo 19.6→28.4 and agg8 73.8→195 tok/s (50 layers' decode attention now
runs FA2's windowed paged path under the graphs). Listed follow-ups:
the L2 mxfp8 cert19 re-run + broader L2 quality battery. Evidence:
`results/2026-11-07-p629/` (`phaseA/r109a/` = the A/B + guards +
k-independence anchors + the ref-bug closure legs, `cert19_l1/` = the
r111 ladder incl. `r111_summary.txt` and the ngram refusal excerpt);
cert18-era evidence `results/2026-11-06-p628-gemma4/phaseC/`
(`micro_fa2ab/`, `cert18_l1/`).

### Online-quantization lanes (P6.30/P6.31; cert20/cert21-native — r120c/r120g/r120h2 + r124/r124b)

The P6.30 objective: make online MXFP4/MXFP8 quantization of BASE
bf16 checkpoints work, and make mxfp4 perf SUPERIOR to fp8; P6.31
gives mxfp8 the same act-quant-elimination fix. The lanes
run `ARC_QUANT=mxfp4|mxfp8` on the base model (qwen ignores
`*visual*`/`*linear_attn*`/`mtp.fc` via `ARC_QUANTCFG`; gemma
auto-skips its GDN layers, no ignore list); mxfp4 rides the W4A16
reroute (`ARC_MXFP4_MODE`, cert20) and mxfp8 the W8A16 reroute
(`ARC_MXFP8_MODE`, cert21). These lane records carry to cert22
unchanged (the P6.33 skinny route ships default OFF; the r127 legs
A/B/C re-verified the qwen faces within ±2.6% on the arc6 wheel).
Same-image A/B legs, certified
postures (qwen GMU 0.85 / MTP k3 / graph / LMC; gemma GMU 0.8 /
spec 0 / LMC 0 / RPC 1200; gemma cert21 legs at maxlen 40960 —
the 8-bit-weights capacity law, see CONFIGURATION):

| qwen3.8-27b online lane | U8k TTFT | U32k TTFT | decode solo | decode agg8 | spec pos1 |
|---|---|---|---|---|---|
| mxfp4 stock (W4A4) | 6.84 s | 29.74 s | 43.3–43.6 | 204–218 | 0.8813 |
| **mxfp4 W4A16 (cert20 lane)** | **3.66 s** | **17.12 s** | **62.5–65.2** | **301–311** | **1.0000** |
| mxfp8 stock (W8A8, e8m0) | 5.59 s | 25.20 s | 35.6–36.3 | 195–199 | 0.9597 |
| **mxfp8 W8A16 (cert21 lane)** | 3.61 s | **17.07 s** | 53.8–55.6 | 265.9 | 0.9804 |
| native-fp8 checkpoint (champion) | **3.76 s** | **16.93 s** | **84.5** | **~300** | 0.9258 |

**W4A16 is superior to every online lane on every axis (1.47–1.75×
over stock mxfp8; 1.2–1.5× over W8A16)** and
1.74–1.87× TTFT / 1.43–1.45× decode over the W4A4 control. W8A16
(P6.31) turns stock online mxfp8 from the slow lane into a real one:
1.37–1.55× over the W8A8 stock on every qwen face. Vs the
native-fp8 champion (a pre-quantized checkpoint — a different
comparison class): U8k + agg8 superior, U32k par (17.12 vs 16.93), solo
74 % — kernel-level proof the residual is NOT in the GEMM (w4a16 M=1
0.17–0.25 ms beats even the e8m0-fp8 GEMM's 0.33–0.41); the delta
lives in per-step overheads across ~200 linear calls/token (honest
ledger, full analysis in the results README; the named next lane
W4A8 was probed and closed: oneDNN refuses the joint dtype — "Only
support int4 and fp8 gemm", r123 banked).

| gemma-4-31b online lane | U8k TTFT | U32k TTFT | decode solo | decode agg8 |
|---|---|---|---|---|
| mxfp4 stock (W4A4) | 8.70 s | 40.25 s | 18.6 | 135.1–135.7 |
| **mxfp4 W4A16 (cert20 lane)** | **3.94 s** | 21.16 s | **40.5** | **261.8–262.1** |
| mxfp8 stock (W8A8, r124b) | 6.88 s | 32.86 s | 15.0 | 111.0 |
| mxfp8 W8A16 (cert21 lane, ml40k) | **3.87 s** | **20.94 s** | 27.7 | 192.4 |
| online fp8 (classic dynamic, P6.28 record) | 4.9 s | **17.2 s** | **28.4** | **195.5** |

Gemma W4A16: 1.90–2.21× over the same-window W4A4 control and faster
than gemma's plain-fp8 online lane on 3 of 4 axes — the U32k exception
is the W4A16 prefill physics ceiling (bf16-DPAS-par; Xe2 has no fp4
MMA) against fp8's static-class lane. The qwen plain-fp8 online lane is
an upstream interface dead-end on this vLLM (`fp8` is not an online
shorthand; `--quantization-config` is consumed by the online framework,
which requires linear/moe/targets — gemma's QCFG-unset classic lane has
no ignore channel for qwen's hybrid layers) — banked r120h traceback +
the r120h2 `fp8_per_channel`+QCFG probe in the results README.

Kernel-level matrix (r120d kbench, wheel-exact layouts): decode Ms —
w4a16 the FASTEST lane on every shape (0.17–0.25 ms; 2.53× faster than
w4a4-apply, ~2× than the e8m0-fp8 GEMM, 3–4× than bf16); prefill
M=32768 — par to 1.24× slower than bf16 (the physics ceiling) but
2.1–2.7× faster than the e8m0-fp8 GEMM and 3.0–3.8× faster than w4a4
(ref kernel: 108 ms at gemma.up). The v3.13 verdict on online-fp8: its
GEMM rides `jit:gemm` with bidirectional e8m0 group scales — NOT a ref
fallback — yet still 2.6–2.8× slower than bf16 at prefill: the penalty
is group-scale handling inside the DPAS kernel (plus the per-call act
quant on top). P6.31 (r122/r123 kernel matrix) closed exactly that:
eliminating the per-call act quant and serving bf16×f8 (the
`fp8_gemm_w8a16_mxfp8` op, m816) recovers 1.74×/2.68× at decode/prefill
medians vs stock mx8 (rel ≤ 2e-3), and the e2e tables above bank the
model-level result. Evidence:
`results/2026-11-08-p630-mxfp4-perf/` (verdict tables + kernel matrix
+ the jit:gemm verbose proof + four engineering laws incl. the
`vllm_xpu_kernels._xpu_C` dlopen law) +
`results/2026-11-08-p631-mxfp8-w8a16/` (r122/r123 kbench + r124/r124b
four-leg battery + the adjudication).

### Throughput — t0, thinking off (chat battery; posture-invariant face)

| Face | cert6 prob-lane value | Boots | Evidence |
|---|---|---|---|
| agg4x256 steady | **291.4–303.1 tok/s** (p617 298.29/298.19 · 303.13/291.37; greedy-lane family r70/p65/r67 294.7–300.5 in-band — argmax path identical) | 2+ | `…-p617a-probcert/`, `…-p70-r70-cert4-comprehensive/` |
| solo200 | **83.7–84.9 tok/s** (p617 83.73–84.85; one degraded pair 42.0/44.9 on one leg-b battery — disclosed below) | 2+ | same |
| survival (first request after boot) | HTTP 200, 6.2–8.2 s (p12b interlock warmup included) | 5 | p617, p70 |
| first-battery warmup face | agg run #1 261–270 vs steady #2 ≈291–303 — first-traffic cost, gone by run #2 | p617 | p617a |

### Sampled / thinking faces (real-world regimes, cert6 prob lane)

| Face | agg steady (tok/s) | solo (tok/s) | acceptance (window) |
|---|---|---|---|
| xhigh t0.2 | 214.9 (run #1 133.6 — first-sampled-face transient, disclosed below) | 62.3–64.2 | 0.58 (mean-len 1.75) |
| xhigh t0.4 | 215.0 | 63.4–69.1 | 0.57 (mean-len 1.71) |
| xhigh t0.6 | 222.3 | 68.4–71.0 | 0.61 (mean-len 1.83) |
| xhigh t0.8 | 224.7 | 64.9–66.3 | 0.61 (mean-len 1.82) |
| xhigh t1.0 | **232.7** (steady family 218–241.3 cross-boot: p616b/p616c/p617 222.7–241.3 + the cert7/cert8 release gates 218.29/221.36 — the face's highest-variance posture; first-sampled-face transient disclosed below) | 64.1–71.0 — uniform, zero collapse legs | 0.61 (mean-len 1.84) |
| alt (t0.7/tp0.80/pp1.5, non-thinking) | 281.3 (other face 271.5) | 82.7–92.1 | 0.91 (mean-len 2.73) |

Acceptance windows are engine-counter deltas per face (leg-a `metrics.txt`
snapshots; acceptance = accepted ÷ drafted tokens). The sampled-temp band
0.57–0.61 is the expected cost of the correct sampler: probabilistic
drafts sample the target distribution (rejection-ratio verify), so fewer
draft tokens verify than a greedy draft chasing argmax — the uniform
64–71 solo floor and zero collapse legs show what that acceptance buys.

The greedy-lane sweep (r70, cert4-native — history): 205.6 / 221.7 /
202.4 / 204.8 / 194.4 agg steady t0.2→t1.0, alt 264.0–280.9 / solo
72.5–73.1 — parity at cool temperatures, the prob lane pulls ahead as
temperature rises (t1.0 +19…+29 %) and clearly on alt solo (82.7–92.1
vs 72.5–73.1). The p46 campaign's k3 table (the depth decision) is
reproduced on cert4 within ~1 pp acceptance / ±6 % throughput per cell
— the **k3** law stands on cert4-native data. Full r46t→cert4 parity
table (every face, Δ column, verdict):
`results/2026-10-31-p70-r70-cert4-comprehensive/R46T_PARITY.md`
— **no regression**.

Per-KV-dtype t1.0-xhigh (solo pair + 4-way, all five quantizations,
prob lane) is banked in the draft-sampling matrix below.

### Prefill / TTFT — solo U-bars, fresh prompts (posture-invariant face)

| Face | prompt tok | TTFT (cert6 prob lane) | Family (r66a/r67/r70 bars) | prefill rate |
|---|---|---|---|---|
| U8k | 8,158 | **3.77 s** | 3.74–3.86 s | 2,164 t/s |
| U16k | 16,354 | **7.90 s** | 7.87 (r70) | 2,069 t/s |
| U32k | 32,553 | **17.38 s** | 17.08–17.36 s | 1,873 t/s |
| U64k | 65,303 | **40.32 s** | 39.19–40.26 s | 1,620 t/s |

**cert9 (P6.20 L3 off-thread store) — the certified-current DEFAULT-posture prefill
faces, image-native at the cert9 gate**: **U8k 3.76 s / U32k 16.93 s /
U64k 38.93 s / C32k re-hit 1.09 s** (16k not re-measured at the gate;
p617 record 7.90 s stands). The deltas vs the p617 record (−0.08 /
−0.45 / −1.39 s) are the store body leaving the engine thread — the
r81 interleave A/B/A/B on the same boot pair measured the pure L3 win
as **−0.74 s U32k / −1.64 s U64k / −0.08 s U8k** vs the cert8 inline
posture (17.9–18.1 / 39.6–41.5 / 3.84 s). P6.20's trace-driven closure:
the big-prefill face is **device-compute-bound** — ~89 % of U64k TTFT
is the wait_for_save device-queue drain behind just-submitted chunk
forwards (worker bodies ≤108 ms, submit walls ~400 ms); the residual
U32k/U64k gap is kernel compute (r67b: AR 30.4 % / FA 20.3 % of rank0
device time at U32k) — the lever P6.21/P6.22 then attacked and **closed
NEGATIVE on both slices**: FA (P6.21c ESIMD v2 kernel 0.33× stock, best
ablation-decomposed case ~0.83× — `results/2026-11-05-p621c-fa-esimd-v2/`,
ladder `results/2026-11-05-p621b-fa-headroom/`) and AR (P6.22 r94/r95:
stock oneCCL ring 2.06× FASTER than the achievable one-shot ceiling at
the dominant 73.1 MiB prefill AR — `results/2026-11-05-p622-ar-lane/`).
The prefill device ledger is fully dispositioned; see the P6.20 record in
`results/2026-11-04-p620-prefill-opt/`.

**cert10 + `CCL_SYCL_ALLREDUCE_FP8=1` (P6.24 e4m3-wire allreduce) — the
BEST prefill faces, opt-in, image-native at the cert10 gate (r99
A/B/A/B interleave, on/off ×2)**:

| Face | cert9 banked | cert10 default (off) | **cert10 FP8=1** | on/off | vs cert9 |
|---|---:|---:|---:|---:|---:|
| U8k | 3.76 s | 3.78 s | **3.27 s** | 0.865 | −13.0 % |
| U32k | 16.93 s | 17.71 s | **16.29 s** | 0.920 | −3.8 % |
| U64k | 38.93 s | 39.03 s | **35.73 s** | 0.916 | −8.2 % |
| C32k re-hit | 1.09 s | 1.21 s | **1.04 s** | 0.855 | −4.6 % |

The off phases ran ~4 % above r84's day (day drift — the interleave
design exists for this); every face beats the BANKED cert9 baseline.
Microbench root: **1.65× at the dominant 76,677,120 B AR** (7.801→4.731
ms, 16.2 GB/rank/s effective; 1.57–1.69× across all large sizes) with
numerics at the measured e4m3 round-trip floor (rel-rms 0.02285 == the
offline bit-exact simulation 0.022861; bar 0.035, corruption regime
≥0.2) and task-level quality parity (r98: tps/spec/long-return/integrity
all inside bars, the min-count law confirmed live — sub-2M-element
prefills never touch the wire, `simple` class byte-identical across
legs). r100 closed every remaining optimization door with wire-rate
measurement over the real peer IPC mapping: remote writes 9.85 GB/s
flat across 4/8/16-byte store widths (fabric ceiling — wider stores
dead), remote reads 4.54 GB/s (2.2× slower — read-reduce dead),
concurrent rd+wr collapses to 1.49 GB/s (overlap/pipelining dead), local
write 380 GB/s (38× headroom — local kernels not binding); v4 sits at
**82 % of the wire-bound floor** (38.4 MB/rank minimum e4m3 traffic at
the measured ceiling = 3.90 ms) with the remaining 0.83 ms in
measured-floor local stages — `results/2026-11-05-p624-fp8-ar/` (r97
gate + r100 scan). How to enable: one env line
(`CCL_SYCL_ALLREDUCE_FP8=1`) — knob table in
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md); certified default OFF.

### P6.25 — the co-tenancy audit (r101, 2026-11-06): the last door closed MEASURED

The one door no leg had measured: whether a side **stream** can hide the AR
behind GEMM EUs (async-TP class — r83 closed same-stream overlap, r100
closed fabric-level concurrency, neither EU-vs-wire co-tenancy). Microbench
on the serve engine's own stack (cert10 image, torchrun TP2, backend xccl,
real per-rank shapes at M=7488 incl. the GLU; `tools/ar/ar_overlap_bench.py`):

- **API permits it**: `dist.all_reduce` is async-enqueue — 0.182 ms stock /
  0.263 ms fp8.
- **The device does not**: a concurrent AR stream inflates GEMM latency
  **1.974× stock / 1.580× fp8** (AR side 1.09×/1.16×) — AR and GEMM
  kernels time-slice the same execution resources; there is no idle-EU
  lane to hide the wire in.
- **Mock-layer A/B** (L=8 layers, real GEMM+GLU+AR structure, serial vs
  k=2/3/4 sub-batch interleaved on a side stream): best-k **0.969× stock
  (k=2) / 0.966× fp8 (k=3)** — overlap is SLOWER than serial at every k on
  both wires. Pre-registered gate (≥1.08×) FAIL; the k-sweep is flat and
  both wires fail → closed as a **class**, schedule- and wire-independent.
  No r102; no engine change; cert10 stands.
- **Census bonus**: the four per-rank linears take 17.173 ms/layer =
  148.6 TF/s rank-local — ~99 % of the p66 devprobe's 147–150 TF/s
  per-card practical dense fp16 ceiling (the audit draft's 73.5 TF/s/rank
  reference was wrong — that figure was per-card already halved once).
  The GEMM lane closes as already-optimal.
- Anchors re-validated in-window: stock AR 7.809 ms (== the r94/r100
  ceiling), fp8 AR 4.597 ms (1.70×; in-family with r97's 4.731/1.65×).

Residue after r101: every prefill critical-path door is measured-closed
(GEMM ~99 % of ceiling, AR at the fabric ceiling + fp8 opt-in, same-stream
r83, fabric concurrency r100, AR-internal pipelining ~1 % r100,
co-tenancy r101, FA P6.21, host P6.20); remaining headroom bounded
≤1–2 % class. Evidence: `results/2026-11-06-p625-overlap-door/`.

### Prefix reuse — warm faces with tier attribution

| Face | cold t1 | warm t2 | served by |
|---|---|---|---|
| U8k | 3.89 s | **0.99 s** | engine APC (external tier 0.0 %, store-only LMCache window — same attribution as r70) |
| U16k | 8.39 s | **1.08 s** | same |
| U32k | 17.44 s | **1.07 s** | same |
| U8k warm-hit re-fires (n=6 across p617 legs a+f) | — | **0.80–0.85 s** (r70 family 0.80–0.91) | same |

Cold/warm provenance (p617): the t1 faces are the **fresh-seed**
`p70_warm_rehit2` probe (seeds 6157/6159/6161, disjoint from every
prior corpus — the v1 probe's t1 rows shared the ladder corpus and were
quarantined); t1 3.89/8.39/17.44 s is in-family with the solo ladder
TTFT faces 3.77/7.90/17.38 s (independent corpus, same boot family) and
t2 is 3.6–3.9× faster than t1 on every face. The engine's usage payload
does not emit `cached_tokens` (field None on every v2 row), so tier-1
attribution rests on the behavioral face + the counter window (warm
re-hit window prefix-hit 121,472/138,354 queries — leg-a `metrics.txt`).

(The cert-era warm faces used a different prompt corpus — prewarm-v7 —
so they stay in history and do not compare; see the history soak block.)

### Speculative decoding (MTP k3) — cert6 prob lane (+ greedy-lane rows, history)

| Condition | acceptance | mean accepted length |
|---|---|---|
| t0 strict-gate window (p617 boots a/b/d — identical to r70's greedy gate window: posture-invariant argmax path) | **83.7 %** (pos1 0.9258) | 3.51 |
| t0 battery window (p617 leg a; identical to r70's greedy-lane 85.9 % — posture-invariant at t0) | **85.9 %** | 2.58 |
| xhigh sweep t0.2 → t1.0 (p617 leg-a counter windows) | 0.58 / 0.57 / 0.61 / 0.61 / 0.61 | 1.75 / 1.71 / 1.83 / 1.82 / 1.84 |
| alt face (p617 leg-a counter window) | 0.91 | 2.73 |
| r67a compile posture (rejected) | — | 2.12–2.32 (−35 %, the gate that keeps `mode:none`) |

**MTP value over no-spec — p617 prob-lane A/B (fresh boot per side,
legs a/c):**

| Face | MTP k3 prob | no-spec | ratio |
|---|---|---|---|
| agg t0 | 298.2 / 303.1 | 124.5 / 127.1 | **2.35–2.44×** |
| solo t0 | 83.7–84.9 | 34.3 | **2.44–2.47×** |
| agg xhigh t1.0 | 232.7 (cross-boot family 218–241, incl. the cert7/cert8 gates) | 115.3–121.8 | **1.84–2.09×** |
| solo xhigh t1.0 | 64.1–71.0 | 33.6 | 1.91–2.11× |
| agg alt | 271.5 / 281.3 | 107.1 / 116.0 | **2.42–2.53×** |
| solo alt | 82.7–92.1 | 33.6 | 2.46–2.74× |

(Greedy-lane A/B, r70 legs a/b — history: t0 2.34×/2.46×, t1.0
1.54×/1.74×, alt 2.28×/2.16×. The prob lane widens every ratio.)

### Stability

- p617 campaign (prob lane, 5 fresh boots incl. default-path and
  greedy opt-out validation boots): **zero boot failures, survival 200
  on every boot (6.2–8.2 s), 0 ERROR / 0 Traceback in every full
  engine log**, integrity 5/5 + determinism 3/3 on every measured boot
  (a/b/d), golden 5/5 zero-drift (b/d), garbage suite 15/15 (a), spec
  gates hard-floor PASS everywhere (acceptance 0.837 gate window,
  identical to the greedy-lane gate window). W14 restored + health-200
  verified after the campaign AND after the validation phase.
- r70 soak (greedy lane, t0 battery — posture-invariant faces; 4 h, 56
  cycles of battery+integrity+health, 10:46–14:46 UTC): **GREEN** —
  battery failures 0 (agg 4/4 streams ×112, solo ×112), integrity 56/56
  `bad=0`, health 200 every cycle, 0 ERROR / 0 Traceback in the full
  engine log (141 WARNINGs: 98 LMCache pin_monitor self-resolving from
  the warm-v2 store window + 43 boot-accepted; neither class recurs in
  the driver's vllm-logger warn counter — 0). Throughput drift none:
  steady-state agg 260.53–301.25 tok/s (mean 283.12), solo 83.65–84.80
  (mean 84.32). Evidence: `results/2026-10-31-p70-r70-cert4-comprehensive/c/`.
- **Disclosed transients (p617, honest record):** (1) **first-sampled-
  face-after-t0 transient** — the first sampled battery/face after
  sustained t0 traffic can measure uniform-slow ~134–137 agg (leg b
  t1.0 battery 1: 134.13/134.76; boot d t1.0 face 1: 136.92; leg a
  t0.2 face 1: 133.56), recovering within 1–2 faces (boot d face 2:
  241.29). POSTURE-INVARIANT: the greedy-lane r70 sweep shows the
  identical class on its first sampled face (t0.2 #1 = 134.39 → #2
  205.63) — a sampled-path warm-up, not a posture defect; all faces
  4/4 streams, 0 errors. (2) one leg-b t0 solo pair measured
  42.0/44.9 tok/s (battery 2) — unreproduced across the 12 other t0
  solo faces that day (83.7–84.9 incl. boot d), the same one-off
  uniform-slow class as the p615 master-matrix † footnote.
- run7n ×2 per leg agree within 1.5 % (steady-state, outside disclosed
  windows); prefix battery phases byte-identical (deterministic corpus,
  temp 0).

(The 8 h soak of record — r46, 46/46 cycles — is cert-era: history
block below.)

### Posture laws in force on cert4 (do not re-litigate without new evidence)

- **`mode: none`** — the 0.30 compile REVAL FIRED the Appendix H law #1
  acceptance gate: `vllm_compile` collapses acceptance length
  3.40/3.57 → 2.21/2.32 (−35 %), decode −39 %, prefill flat (p67);
  r70's accidental first launch re-replicated it exactly (182/52 t/s,
  mean length 2.12 — see `a_invalid_compile_posture/` in the p70
  evidence).
- **MNBT stays 8192** — 16384 A/B: U32k −9.7 % but U8k +9.9 % and KV
  pool −18 %; documented, not adopted (p66). Never combine with the
  p65 trio.
- Any `FULL_AND_PIECEWISE` repeat must cap the capture ladder at 256
  (`max_num_seqs × (k+1)`; the `gdn_attn.py:120` law, p67).

### Prefill anatomy (p66 — analysis of record)

Solo TTFT sweep, fresh boot each, nolmc, graph, k3, greedy, unique seeds
(r66a; three faces × four boots — `--long-prefill-token-threshold`
none/8192/4096/2048):

| Face | prompt tok | thr=none | thr8192 | thr4096 | thr2048 | thr2048 Δ |
|---|---|---|---|---|---|---|
| U8k | 8158 | 3.74 s (2178 t/s) | 3.75 s | 4.04 s | 7.94 s | **+112%** |
| U32k | 32553 | 17.34 s (1878 t/s) | 17.35 s | 18.24 s | 21.49 s | +23.9% |
| U64k | 65303 | 39.20 s (1666 t/s) | 39.23 s | 42.38 s | 49.75 s | +26.9% |

thr=none and thr8192 are the same path (≤ 0.03 s apart — at MNBT 8192
the threshold IS the default chunked size). The rate falls with context
(2178 → 1666 t/s, attention KV growth) even before any chunking effect.
thr2048's cost is two mechanisms stacked: more chunk boundaries (8158 tok
= 5 × 1664-token chunks instead of 2 × 7488) **and** small-M device
inefficiency (U8k rate halves: 2178 → 1028 t/s).

**Where the time goes (r66b, per-chunked-step decomposition on the live
V2 runner)**: scheduler 0.3 ms, host prep ≤ 10 ms, TP2 allreduce ~28 ms,
sampling ≤ 107 ms — and the rest is the **device execution of the chunk
itself**, observed as a queue-drain wait inside the next step's
mamba/GDN `preprocess_state` launch window (780–1018 ms of an ~0.9 s
step at 1664-token chunks; XPU eager forward is async, so the enqueue
looks flat at ~127 ms and the device time surfaces one step later).
Prefill is **device-bound at ~34 % MFU** (1664-token chunks) to ~40 %
(7488-token chunks, 2.2–2.4 tok/ms) against the measured 294 TFLOPS
fleet fp16 GEMM ceiling (147–150 TFLOPS/card, 256 CUs, devprobe r66a).
The gap is attention over growing KV + W8A16 dequant + non-GEMM ops +
per-layer launch/dispatch — kernel territory, not host glue.

**Improvement ladder, closed out**:

- thr table (knob) — documented above; the p65 long-context profile
  (`--long-prefill-token-threshold 2048` + deferral) is the one place
  thr2048 is correct, for LMC-side reasons (see CONFIGURATION-arc).
- **MNBT 16384 (knob) — A/B'd, NOT adopted** (r66c): U32k −9.7 %
  (17.33→15.65 s), U64k −4.1 %, confirming the small-M diagnosis — but
  U8k +9.9 % (3.74→4.11 s) and the KV pool pays 18 % (405,491→332,629
  tokens; max concurrency 1.55x→1.27x) from the doubled prefill
  activation workspace. Never combine with thr2048 (the threshold pins
  chunk size — only the KV loss remains). Conditional long-prompt-only
  lever; the certified default stays 8192.
- **Host glue — closed as empty** (r66b): nothing material to cut.
- **Structural (raise device MFU)** — the only remaining large lever;
  P6.7 ran it — next section.

Evidence: `results/2026-10-31-p66-r66a-prefill-capture/` (curve + device
probe), `results/2026-10-31-p66-r66b-step-decomposition/` (step
decomposition + the engine-map correction + probe-instrumentation laws),
`results/2026-10-31-p66-r66c-mnbt16k/` (MNBT A/B).

### Prefill structural rungs (p67 — compile REVAL + op-class capture)

Zero engine-code change; cert4 remains the certified image (no cert5).
Evidence: `results/2026-10-31-p67-r67a-compile-ab/` (VERDICT.md,
r67b_capture_analysis.md, P67_LADDER_VERDICT.md, raw rank traces).

**Compile REVAL (r67a, Appendix H law #1 on 0.30) — REJECTED.**
`mode=vllm_compile` × `cudagraph_mode=FULL_AND_PIECEWISE` (the only legal
combo per the capture law), same-session A/B on the cert4 image: MTP mean
acceptance **3.40/3.57 → 2.21/2.32 (−35 %)**, agg4x256 steady 300.48 →
182.04 t/s and solo200 84.49 → 51.66 t/s (both −39 % — pure arithmetic of
fewer accepted tokens per verify step), prefill faces FLAT (U8k 3.75 vs
3.74 s, U32k 17.08 vs 17.33, U64k 39.20 vs 39.19). KV pool +5.3 %
(426,874 tokens) does not compensate. Boot cost 460 s vs 265 s stock;
3 fork-added jit_monitor WARNs (inference-time Triton JIT of spec kernels).
`mode` stays NONE. Config law banked en route: with MTP k3 and
max_num_seqs 64, every captured size must satisfy the GDN decode-only cap
`max_num_seqs×(k+1)` = 256 tokens (`gdn_attn.py:120`) — capture ladders
must top at 256 under FULL_AND_PIECEWISE (stock ladder headroom to 320
boot-dies at the first piecewise capture).

**Op-class device anatomy (r67b, torch-profiler XPU trace of the U32k
face, rank0, 15.38 s kernel time, 71.9 % busy):**

| Slice | Share | Verified launch site |
|---|---|---|
| GEMM | 39.7 % | `_xpu_C::fp8_gemm_w8a16` (the p44 W8A16 route — fastest on-image per r40) |
| TP2 allreduce (ring+ReduceCopy+barrier) | ~30.4 % | `c10d::allreduce_` |
| FA-layer attention (16 layers, DENSE) | 20.3 % | `_vllm_fa2_C::varlen_fwd` (~31 ms × 102) |
| GDN chunk-scan (48 layers) | 4.7 % | `gdn::ChunkFwdO/tiled/WU/A/InverseOpt` ×336 |
| norms + act + rope + elementwise | ~5 % | — |
| indexer/gather (sparse machinery) | 0.1 % | — |

Consequences: the radiance QSA-splitk whale does not exist on stock 0.30
(the model runs dense FA — no sparse indexer); **direct-fp8 GEMM rung
closed negative** (on-image `XPUFp8BlockScaledMMKernel`/`_xpu_C.fp8_gemm`
exists but r40 measured it 2.2–2.4× slower than the deployed w8a16 at
prefill Ms — the GEMM whale already runs the fastest on-image kernel);
**ESIMD FA-kernel rung re-scoped** to the dense varlen site (GQA 24q/4kv,
head_dim 256, fp16, paged fp8 KV PAGE 1664; est. 5–7 % compute MFU today;
ceiling −8…−12 % U32k TTFT if 1.5–2×) — opened as the P6.21 kernel
project and **closed negative** (P6.21c, 2026-11-05: v2 kernel
numerics-green at 0.33× stock; the ablation-decomposed ceiling bounds the
best case at ~0.83× — under the 2× bar —
`results/2026-11-05-p621c-fa-esimd-v2/`, ladder
`results/2026-11-05-p621b-fa-headroom/`); and the
**TP2 allreduce family (~30 %) was then the largest unattacked slice** — the
2026-10-31 P2P analysis (r71, `results/2026-10-31-p71-r71-p2p-analysis/`)
proved the data path is **already direct P2P** (D2D 10.23 GB/s both
directions; every collective `algo topo sycl` over pidfd ze-IPC handles,
zero host-staged selections), so the slice is algorithmic/latency-bound,
not link-bound — any gain there is a kernel/algorithm project. Attacked
as P6.22 and **closed negative** (2026-11-05, r94/r95): the message-size
census found the dominant 76,677,120 B (7488×5120 fp16) prefill chunk AR
= 88 % of the family (27.3 % of the r94 rank0 device window), and the
ceiling bench measured stock oneCCL ring at 7.78 ms isolated / 7.85 ms
in-situ vs the 16.03 ms one-shot torch-primitive ceiling — stock is
**2.06× faster than the achievable ceiling** (concurrent ring 9.85
GB/rank/s vs the torch copy engines' 4.92 GB/s/dir) —
`results/2026-11-05-p622-ar-lane/`.

### TurboQuant KV-cache dtype faces (P6.14 measurements — SERVABLE again since cert8 / P6.19 fix)

> **Reopened and FIXED (2026-11-04, P6.19 / cert8).** The P6.18
> deprecation (refusal at boot) is lifted: the corruption reported under
> `turboquant_k8v4` was root-caused to **three stacked spec-verify
> defects** — captured verify graphs baking the raw-KV flash fast-path
> (#TQ-blind-verify), a replay-broken multi-row synthetic-decode
> dispatch, and (for the first MQ-kernel draft) the wheel's two TQ ops
> trees (AoS engine-default vs FlyDSL-gated SoA — the draft read the
> live cache as zeros) — all fixed wheel-side by `tq_spec_fix.py`
> (split-row dispatch ships; donor v19b capture forcing; MQ kernel on
> the AoS layout opt-in-experimental, +20 % agg8). **Zero
> REP-LOOP/JUNK/BLEED/EMPTY on all four presets**, all-dtype gate with
> the fp8 certified control (which fires the same identity-face class —
> engine probabilistic-drafting faces, not corruption) —
> `results/2026-11-04-p619-tqverify/` (root-cause legs, MQVAL suite,
> t8/t9 batteries, best-practices audit). The P6.18 Class-B/Class-A
> split stands as the reproduction record; its precision-ceiling
> root-cause conclusion was corrected by P6.19. The P6.14–P6.16 faces
> below were measured with the pre-fix engine — their
> capacity/TTFT/throughput numbers remain valid as measurements of what
> the presets trade; the corruption they could not see is now gated by
> the long-return probe (in every release gate since cert7).

The matrix below was the candidate-image battery
(`results/2026-11-02-p614-l3/`, `…-l1v2/`; image p614c2, engine
identical to cert6) — the cert6 release gate re-proved the 4bit face on
the tag-built cert6 image itself (`results/2026-11-02-p614-cert6/`).
Every preset passed that era's battery: 0 ERROR / 0 Traceback,
integrity 5/5 + determinism 3/3, spec gate ≥ target, flip survival, kit
transfer live, external tier participating (hit-rate max 5.9 / 6.2 /
7.6 %) — none of which bounded long-form generation quality; the
capacity/TTFT/throughput numbers remain valid as measurements of what
the presets traded.

| Preset | KV capacity | U32k / U64k TTFT | t0 agg#2/solo | t1.0 xhigh agg#2 / solo pair (probabilistic lane — certified) | gap p95 U32k/U64k |
|---|---|---|---|---|---|
| `fp8_e4m3` (default)¹ | 405,491 tok (1.55×) | 23.16 / 50.77 s | 297.19 / 84.48 | **241.10** / **67.46 / 64.10 — uniform** | 1299 / 1569 ms¹ |
| `turboquant_k8v4` | 526,445 tok (2.01×) | **15.29 / 33.03 s** | 302.58 / 88.05 | **172.95 (−28 % vs fp8)** / 54.77 + 51.62 uniform | 3690 / 4385 ms |
| `turboquant_4bit_nc` | 757,641 tok (2.89×) | 15.91 / 34.02 s | 300.06 / 86.31 | 134.42 (−44 %) / 54.27 + 50.86 uniform | 3218 / 3785 ms |
| `turboquant_k3v4_nc` | 853,220 tok (3.25×) | 15.48 / 33.78 s | 294.43 / 83.44 | 137.11 (−43 %) / 52.37 + 62.32 uniform | 3736 / 4186 ms |
| `turboquant_3bit_nc` | **975,329 tok (3.72×)** | 16.02 / 35.74 s | 302.77 / 86.25 | 171.32 (−29 %) / 53.11 + 58.90 uniform | 3188 / 3729 ms |

(t1.0 greedy-lane history, for the affected-numbers comparison:
fp8 186.48–189.41 clean pair 61.73/61.79; k8v4 173.39 with 77.31 +
26.08 ragged; 4bit 93.05; k3v4 88.95; 3bit 89.58 — all ragged.)

t1.0 xhigh reading (solo + concurrent per quantization,
2026-11-03): under the certified probabilistic lane every dtype's solo
pair is uniform and every TQ aggregate is +44…+91 % over its greedy
face — concurrent sampled throughput is the honest dtype comparator:
k8v4 −28 %, sub-4-bit presets −43…−44 % vs fp8's 241.10. The greedy
lane (history) showed the ragged pairs. Mechanism (pinned by the
P6.15c diagnostic, `results/2026-11-03-p615c-t10collapse/`):
the collapse leg is NOT a stall and NOT mid-run — the engine step stays
41.7 ms and **wall = drafts × 41.7 ms** in every request; a collapsed
request is BORN collapsed (MTP draft acceptance dead from round 1,
sticky for the request's lifetime — pos0 rejections on 169/181 rounds in
the reproduced 26-tps request), so 200 tokens simply take 3.4× more
verify rounds. Trigger = 4-bit-V KV noise breaking draft/target match on
low-margin sampled-trajectory regions: fp8 KV immune at the same
posture, greedy t0 immune on every dtype, healthy TQ legs 77–86 t/s.
Per-request lottery observed at 10 % (1/10 full + 2/10 partial, this
leg) to 50 % (banked solo pairs) — n small. Fix menu: no config/stall
fix (nothing stalled; spec depth doesn't change wall); cool the sampling
(acceptance recovers as temp falls — intermediate temps unmeasured) or
stay on fp8 (immune) when uniform sampled latency matters;
engine-side acceptance-aware gating is out of scope (upstream
direction). **Update (P6.16→P6.17, 2026-11-03): the serve-layer fix is
measured, certified, and the default — probabilistic drafting
(see "Draft-sampling posture" below) removes the graded damage modes
and lifts every dtype's t1.0-xhigh aggregate (fp8 +29 %, sub-4-bit
+44…+91 %); the rare total-collapse trajectory remains (true draft/
target divergence); engine-side acceptance-aware gating stays upstream
direction.**

¹ fp8 comparator row as run in the A/B (thr2048-profile face —
hence its 23.16/50.77 TTFT and 1.3/1.6 s gaps). P6.15 re-measured
TRUE default-posture fp8: TTFT 17.25/39.77 s, gaps p95 **3986/5295 ms
— the WORST bystander-stall face of the family** (the TQ rows above,
at equal posture, IMPROVE bystander stalls; the old "TQ gaps ~2.5× vs
fp8" reading was this profile mismatch). See "Bystander-stall tuning"
below.

**cert8-gate re-measurement (2026-11-04, same day, all five dtypes —
the certified-image values):**

| dtype | KV pool (cert8 gate) | pool × | cert8-native faces |
|---|---|---|---|
| `fp8_e4m3` (default) | 400,739 (== cert7 byte-value; the P6.14 405,491 was that day's free-memory state) | — (baseline dtype) | run7n t0 agg#2 300.65 / solo 84.53/84.57; t1.0 221.36 / 70.98/68.14 |
| `turboquant_k8v4` (split) | 519,972 (== t6p dev leg, cross-day byte-value) | 1.98× | spec tps solo 54.42/54.59, agg8 235.97; long-return bar PASS full + 3 churn cycles |
| `turboquant_k8v4` (**mq**, opt-in) | 519,972 | 1.98× | spec tps solo 53.83/54.31, **agg8 284.81 = 1.21× split** (t8g family 289.23) |
| `turboquant_4bit_nc` | 748,549 | 2.86× | long-return fully clean incl. the t1 lane |
| `turboquant_k3v4_nc` | 843,202 | 3.22× | cold + rehit2 byte-match the t9 primaries on all four classes |
| `turboquant_3bit_nc` | 963,764 | 3.68× | precision-floor coherent variants (essay/code), multi/simple stable |

KV capacity tracks the boot-day free-memory state (same posture,
different days → 405,491 vs 400,739 on fp8) — the P6.14 columns in the
matrix above were that campaign's state and stay valid as its labeled
record. Gate source: `results/2026-11-04-p619-tqverify/` (VERDICT.md
performance table + `rig/gate/`, `rig/mqannex2/`).

### MQ experimental mode (`VLLM_ARC_TQ_VERIFY_MODE=mq`, opt-in)

The single-pass **multi-query verify kernel** for TurboQuant
spec-decode — one kernel scores all q_len+1 draft rows against the
quantized cache, replacing split's q_len sequential single-row decode
calls. Implementation `vllm_gaudi/xpu/patches/tq_mq_aos.py` re-ports
the wheel's own MQ kernel onto the engine's slot-interleaved **AoS**
ops layout (the wheel kernel targets the SoA tree, never dispatched on
XPU — the addressing mismatch was root cause RC3); validated offline
bit-exact vs the engine's own store/decode pair on all four presets
(MQVAL5 — LAW: offline validation imports the pair the runtime
dispatches).

Measured (cert8 gate annex, certified k8v4 posture, mode env injected
into the serve chain): **agg8 284.81 tok/s = 1.21× split's 235.97**
(in-family with the t8g dev leg's 289.23), solo unchanged
(53.83/54.31 vs split's 54.42/54.59) — the gain is concurrency, not
latency. **Experimental, not the certified default**: its identity
faces are wider than split's (multi/simple classes drift across
coherent variants between rounds — same text quality, different
hashes); split keeps the tightest t0 identity and backs every
certified TQ number. Serve `mq` when +20 % aggregate matters more than
variant stability; `VLLM_ARC_TQ_MQ_SPLITS=32` sets the KV split count.
Evidence: `results/2026-11-04-p619-tqverify/rig/mqannex2/` + the t8/t9
dev legs in the same tree.

**What improves (dtype-wide):** KV capacity 1.55× → 2.0–3.7× max
concurrency at the same 6.92 GiB pool; U32k/U64k TTFT −30…−35 % vs the
thr2048-profile fp8 comparator (¹) — vs **true-default fp8 solo U-bars
the gain is −10…−16 %** (P6.15b probes; the true-default fp8 solo
prefill is the slowest of the family, 1623 t/s U64k); t0 greedy
throughput in-family or better (294–303 agg across presets); at equal
posture, bystander prefill stalls (see ¹).

**TQ-inherent costs (per preset, probabilistic
lane):** the sampled t1.0 xhigh agg drops 28 % (k8v4) to 43–44 %
(sub-4-bit presets) vs fp8's 241.10 — with uniform solo pairs
(51–62 tok/s) and the rare divergent-trajectory collapse residual
(~26 tok/s) still possible. Leg-average spec acceptance stays ≥ 0.86
on every preset at t0; integrity/determinism hold.

**Dispositions (updated by P6.19):** the per-preset positioning below
is guidance again — since cert8 every `turboquant_*` preset serves
clean (P6.19 spec-verify fix; zero garble all-dtype gate) while
`fp8_e4m3` remains the certified default (fastest sampled throughput,
the quality reference face). Choose a TQ preset for the KV-capacity
lane (1.98×–3.68× at the cert8 gate; 2.01×–3.72× at the P6.14
campaign's memory state) or its U-bar TTFT edge; choose fp8 for maximum
aggregate quality/speed.
For bystander-stall relief use the fp8 levers (`ARC_MNBT=4096` first
choice — see "Bystander-stall tuning" below); true-default fp8's
higher stall face is the accepted cost of the only certified dtype.

### Draft-sampling posture — probabilistic drafting (P6.16 measured; P6.17: the CERTIFIED DEFAULT)

Serve-layer, **no image change** (the knob lives in
`tools/serve_arc_b70.sh`): the spec config carries the
upstream-config-native key `draft_sample_method` (pinned 0.30.0 wheel,
`config/speculative.py:588`) — MTP drafts are **sampled from the draft
distribution and verified by the rejection-ratio test**, instead of
argmax-drafted. **Since P6.17 the default (unset) posture IS
probabilistic** (the certified line is
`ARC_SPEC_METHOD=probabilistic`, every record value from that lane —
the superior sampled lane); `ARC_SPEC_METHOD=greedy` is the **opt-out
historical lane** (the pre-P6.17 certified bytes). Validated flip
(p617b): flag probe 5/5 on the real script via a PATH-stubbed docker
(unset/probabilistic/junk → the exact prob spec bytes; `greedy` → the
exact old certified bytes, no method key; `ARC_SPEC=0` → no spec flag),
and live-boot proof: the default-path boot's engine cmdline bytes are
IDENTICAL to the env-form record legs (`ps` audit, boot d vs leg a),
while the `greedy` opt-out boot carries the old certified bytes with
zero `draft_sample_method` occurrences (boot e, t0 battery in-family
with the greedy record).

**Why it matters**: greedy drafts + sampled verify is a **biased
sampler at temperature** (only argmax-drafted continuations can be
accepted); probabilistic drafts + the ratio test provably sample the
target distribution. At t0 the two are provably identical (drafts are
plain argmax at temp 0, `gumbel.py:102,117`; verify accepts iff
`target_argmax == draft_sampled`, `rejection_sampler_utils.py:571-585`)
— and measure in-band on every dtype (p616b; p617: t0 family
291.4–303.1 / 83.7–84.9, gate window 0.837 identical to the greedy
gate window). Engine cadence is unchanged (41–42 ms; the gumbel/triton
draft sampler runs inside the FULL draft graphs at zero measurable
cost).

The matrix (p616b — one boot per dtype on cert6; greedy column =
pre-P6.17 history):

| dtype | t1.0-xhigh agg greedy → prob | solo pair greedy → prob |
|---|---|---|
| `fp8_e4m3` | 186.48 → **241.10 (+29 %)** | 61.73/61.79 clean → **67.46/64.10 clean** |
| `turboquant_k8v4` | 173.39 → 172.95 (EV-neutral) | 77.31 **+ 26.08 ragged** → 54.77/51.62 |
| `turboquant_4bit_nc` | 93.05 → **134.42 (+44 %)** | 77.60 **+ 25.94 ragged** → 54.27/50.86 |
| `turboquant_k3v4_nc` | 88.95 → **137.11 (+54 %)** | 77.37 **+ 25.88 ragged** → 52.37/62.32 |
| `turboquant_3bit_nc` | 89.58 → **171.32 (+91 %)** | **25.99** + 85.57 ragged → 53.11/58.90 |

Every probabilistic solo pair is **uniform — zero 26-tok/s legs in any
measured face** (the greedy banks' defining defect; a same-day greedy
control boot sampled a live 25.96 + 81.44 pair). fp8's sampled solo
BEATS its greedy bank (67.46 vs 61.73) — the ~49 %-greedy-draft
acceptance ceiling on the certified dtype is broken. t0 faces unchanged
on every dtype (±1.5 %; the tq4 −3 % dip proven environmental by the
greedy control boot), TTFT within −3 % except one flagged face (3bit
U64k +7.2 %), det 3/3 everywhere, spec gates PASS.

**Collapse status (honest reading)**: the **graded damage modes are
gone** — 0 depth-dead legs in 40 traced requests vs 2/10 in the banked
greedy traces (p615c's 47/66 t/s legs), and the argmax-mismatch
lottery with them. But one deterministic **divergent-trajectory
request still collapses (~26 tok/s) on every TQ dtype**: on that
trajectory the draft distribution genuinely diverges from the target
(Σ min(p,q) ≈ 0.08 at pos0) — no verify scheme fixes true divergence.
**fp8 is immune-and-improved**: no such trajectory in the traced corpus
and its sampled face gains +29 %. Menu: sampled serving (t ≥ 0.5) →
the default probabilistic posture, any dtype (fp8 included);
worst-case-latency-sensitive fleets on TQ dtypes → fp8 or cooler
sampling.

**Default-flip ledger (recorded):** P6.16 shipped the
posture opt-in under a pre-registered adoption rule (default flip needs
EVERY dtype ≥ +10 % t1.0 agg — k8v4 is EV-neutral, fails). P6.17
flipped the default anyway (2026-11-03: probabilistic ruled the
superior lane on every dtype, EV-neutral on k8v4) — the certified dtype stays fp8
(gains +29 %, collapse-immune); k8v4 users get an EV-neutral posture
change with uniform solo streams (still strictly better than its
ragged greedy face).

Evidence: `results/2026-11-03-p616a-probsmoke/` (boot+smoke, cadence
law), `results/2026-11-03-p616b-probmatrix/` (5-dtype matrix + collapse
traces + the b2 integrity discriminator: the two t0 integrity flakes
need posture × full-battery churn together, not reproducible ×4,
opt-in-posture-only caveat), `results/2026-11-03-p616c-spechook/` (the
P6.16 hook deployment: flag probe + default/opt-in boots),
`results/2026-11-03-p617a-probcert/` (the P6.17 prob-lane certification
campaign + flipped-default validation). Config:
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md) "Draft-sampling posture".

### Bystander-stall tuning — chunk granularity (P6.15, 2026-11-02)

A bystander inter-token gap = exactly ONE chunked-prefill engine step
(decode does not piggyback into prefill steps on this stack; stall
≈ tokens-per-chunk / prefill-rate). The dtype KV-page size quantizes
the chunk ladder (TQ 1664 → 1664/3328/6656; fp8 832 → 1664/3328/7488);
smaller chunks cut stall duration but add prefix re-reads, so stalls
and TTFT trade off along a Pareto frontier — there is no free knob.
All faces on the cert6 image, same p65-timeline harness, integrity
5/5 + determinism 3/3, 0 ERROR / 0 Traceback
(`results/2026-11-02-p615-stall-analysis/` + per-leg VERDICTs):

| Face (opt-in) | Chunk | TTFT U32k / U64k | gap p95 U32k / U64k | KV capacity |
|---|---|---|---|---|
| fp8 default (certified) | 7488 | 17.25 / 39.77 s | 3986 / 5295 ms | 405,491 (1.53×) |
| fp8 + `ARC_MNBT=4096` | 3328 | 19.49 / 42.83 s | 2030 / 2587 ms | 433,210 (1.65×) |
| TQ4 (`turboquant_4bit_nc`) default | 6656 | 15.85 / 33.98 s | 3227 / 3780 ms | 757,641 (2.89×) |
| TQ4 + `ARC_MNBT=4096` | 3328 | 17.59 / 37.03 s | 2003 / 2049 ms | 809,161 (3.09×) |
| TQ4 + `ARC_MNBT=2048` | 1664 | 22.14 / 45.38 s | 1325 / 1286 ms | 821,283 (3.13×) |
| TQ4 + thr2048 (p65 profile knob) | 1664 | 21.60 / 47.78 s | 1159 / 1329 ms | 757,641 (2.89×) |
| k8v4 (`turboquant_k8v4`) + `ARC_MNBT=4096` | 3456 | 17.41 / 37.58 s | 1936 / 2225 ms | 562,045 (2.14×) |
| k3v4_nc (`turboquant_k3v4_nc`) + `ARC_MNBT=4096` | 3712 | **16.99 / 36.90 s** | 2011 / 2309 ms | 911,660 (3.48×) |
| 3bit_nc (`turboquant_3bit_nc`) + `ARC_MNBT=4096` | 2176 | 19.41 / 42.09 s | 1932 / 1637 ms | **1,042,793 (3.98×)** |

Guidance: stall-sensitive mixed fleets → **TQ4 + MNBT 4096** (stalls
−38/−46 %, capacity +6.8 %, TTFT ≤ fp8-default); sub-1.5 s stalls →
TQ4 + MNBT 2048 (largest TQ4 pool; dominates the equivalent thr2048
face) at +14–40 % TTFT; fp8-fixed fleets → MNBT 4096 (stalls halved,
+6.8 % capacity, +8–13 % TTFT — the first-choice fp8 lever; the deeper
fp8 stall floor is the p65
long-context profile — [`CONFIGURATION-arc.md`](CONFIGURATION-arc.md)
"fp8 stall levers" — p95/max 1580/1599 ms for ≥48k fleets
at U64k TTFT 51.16 s). The 2026-11-03 all-presets completion
set (B6–B8) generalizes the balanced face: **MNBT 4096 halves stalls on
every preset** at TTFT ≤ fp8-default family (k3v4_nc is under family on
both bars), with per-preset chunk = KV page × n; 3bit_nc + MNBT 4096 is
the max-capacity near-low-stall face (3.98× KV, U64k gaps 1637 ms — the
closest large-budget face to the 1.5 s bar) at its known dtype quality
cost. Certified defaults unchanged — this menu is opt-in.

Mechanism + fix detail, pre-fix failure capture, and the per-preset leg
narratives: `results/2026-11-02-p614-{dbg,preflight,l1v1,l1v2,l2,l3}/`
VERDICTs. Config guidance: [`CONFIGURATION-arc.md`](CONFIGURATION-arc.md)
"KV-cache dtype: TurboQuant presets".

### Master matrix — every face × decode / prefill / TTFT / stalls / capacity (P6.15b, 2026-11-03)

The comprehensive battery: 12 one-variable legs on the cert6 image, each
face re-run with the fork's own harnesses (`bench_run7n_t5.py` t0 +
t1.0-xhigh batteries, `p70_solo_ttft.py` U-bar ladder) so every number
reads directly against the banked records — all faces land in-family
with their older banks (fp8-def U-bar 17.34/40.24 s vs p70-cert4's
17.36/40.26; tq4-def t0 298.38 vs G7's 299.67). Full evidence:
`results/2026-11-02-p615-probes/VERDICT.md` (t1.0 from
`…-p614-{l1v2,l3,cert6}/`). Two views, same 12 faces:

**Decode** (t0 = thinking off — posture-invariant face; t1.0 = xhigh
sampled face on the **certified probabilistic lane** (P6.16 matrix,
P6.17 default; greedy-lane t1.0 history in the TurboQuant table);
solo = max-200 single stream ×2; concurrent = 4×256 agg#2 steady):

| Face | Chunk | t0 solo ×2 | t0 4-way | t1.0 solo pair (prob) | t1.0 4-way (prob) |
|---|---|---|---|---|---|
| fp8 default (certified) | 7488 | 84.51 / 84.41 | 296.25 † | **67.46 / 64.10 uniform** | **241.10** |
| fp8 + MNBT 4096 | 3328 | 84.47 / 84.47 | 296.88 | = dtype default (knob decode-neutral at t0; not separately measured) | — |
| TQ4 default | 6656 | 86.24 / 85.98 | 298.38 | 54.27 / 50.86 uniform | 134.42 |
| TQ4 + MNBT 4096 | 3328 | 83.06 / 83.23 | 290.94 | = dtype default (as above) | — |
| TQ4 + MNBT 2048 | 1664 | 86.19 / 85.75 | 299.34 | = dtype default (as above) | — |
| TQ4 + thr2048 | 1664 | 85.65 / 85.80 | 299.16 | = dtype default (as above) | — |
| k8v4 default | 8064 | **88.49 / 88.61** | 301.97 | 54.77 / 51.62 uniform | 172.95 (EV-neutral) |
| k8v4 + MNBT 4096 | 3456 | 88.47 / 88.27 | **302.38** | = dtype default (as above) | — |
| k3v4_nc default | 7424 | 80.83 / 80.45 | 285.85 | 52.37 / 62.32 uniform | 137.11 |
| k3v4_nc + MNBT 4096 | 3712 | 83.29 / 83.05 | 293.98 | = dtype default (as above) | — |
| 3bit_nc default | 6528 | 85.83 / 85.55 | 301.13 | 53.11 / 58.90 uniform | 171.32 |
| 3bit_nc + MNBT 4096 | 2176 | 83.45 / 83.11 | 294.33 | = dtype default (as above) | — |

† fp8-default t0 4-way: the probe's second repeat hit a one-off uniform
4-stream slowdown (153.44, 0 errors; repeat #1 270.67 in-band) — the
certified cert6 gate bank on the identical posture (**296.25**) stands.

**Prefill / TTFT / stalls / capacity** (solo U-bars = probe legs; wave =
banked 12-stream timeline legs; gaps = bystander inter-token p95):

| Face | Prefill solo U32k / U64k (t/s) | TTFT solo U32k / U64k (s) | TTFT wave U32k / U64k (s) | gap p95 U32k / U64k (ms) | KV capacity |
|---|---|---|---|---|---|
| fp8 default (certified) | 1877 / 1623 | 17.34 / 40.24 | 17.25 / 39.77 | **3986 / 5295 (worst)** | 405,491 (1.55×) |
| fp8 + MNBT 4096 | 1729 / 1516 | 18.83 / 43.07 | 19.49 / 42.83 | 2030 / 2587 | 433,210 (1.65×) |
| TQ4 default | 2077 / 1921 | 15.67 / 33.99 | 15.85 / 33.98 | 3227 / 3780 | 757,641 (2.89×) |
| TQ4 + MNBT 4096 | 1888 / 1791 | 17.24 / 36.47 | 17.59 / 37.03 | 2003 / 2049 | 809,161 (3.09×) |
| TQ4 + MNBT 2048 | 1589 / 1459 | 20.48 / 44.76 | 22.14 / 45.38 | 1325 / **1286** | 821,283 (3.13×) |
| TQ4 + thr2048 | 1595 / 1445 | 20.41 / 45.20 | 21.60 / 47.78 | **1159** / 1329 | 757,641 (2.89×) |
| k8v4 default | **2097 / 1945** | **15.52** / 33.58 | **15.29 / 33.03** | 3690 / 4385 | 526,445 (2.01×) |
| k8v4 + MNBT 4096 | 1885 / 1716 | 17.27 / 38.06 | 17.41 / 37.58 | 1936 / 2225 | 562,045 (2.14×) |
| k3v4_nc default | 2076 / 1945 | 15.68 / **33.58** | 15.48 / 33.78 | 3736 / 4186 | 853,220 (3.25×) |
| k3v4_nc + MNBT 4096 | 1930 / 1760 | **16.86** / 37.10 | **16.99** / 36.90 | 2011 / 2309 | 911,660 (3.48×) |
| 3bit_nc default | 2062 / 1893 | 15.79 / 34.51 | 16.02 / 35.74 | 3188 / 3729 | 975,329 (3.72×) |
| 3bit_nc + MNBT 4096 | 1696 / 1549 | 19.19 / 42.15 | 19.41 / 42.09 | 1932 / 1637 | **1,042,793 (3.98×)** |

**cert7-native re-verification (p618e, 2026-11-03, same harness —
`results/2026-11-03-p618-cert7/PERF.md`):** the two live fp8 rows
re-measured on the certified cert7 image — default: TTFT wave
18.27/39.91 s, gap p95 **4001/5273 ms**, KV 400,739 (the cert7-gate
byte-value); MNBT 4096: TTFT 19.51/42.34 s, gap p95 **2052/2523 ms**,
KV 427,666 — every face in-family with the table (the guard changes
nothing under `fp8_e4m3`). The TQ rows are the banked P6.15 campaign
record — **servable again since cert8** (P6.19); their faces were not
re-measured on cert7/cert8 (the cert8-gate dtype table above carries
the current capacity/tps values).

Findings:

- **Decode (t0) is knob-free and dtype-light.** Chunk-size knobs (thr2048 /
  MNBT 2048 / 4096) move solo decode <1.5 % vs the same dtype's default
  (tq4 family 85.7–86.2 solo across 1664/3328/6656-token chunks); dtype
  spread k8v4 88.3–88.6 > tq4/3bit 83.1–86.2 > fp8 84.4–84.5 >
  k3v4_nc 80.5–83.3 tok/s solo; 4-way concurrent 285.9–302.4 agg.
  Per-stream under 6-way load ≈ **70–73 tok/s clean-window** (one SSE
  chunk = one engine step = 3.93–3.95 accepted MTP tokens; lifetime
  45.9 tok/s for a stall-exposed bystander; engine-wide ≈ 440 tok/s) —
  solo 84.5 vs concurrent ~73 = −13 % per-stream for 6× parallelism.
- **Decode (t1.0 xhigh, probabilistic lane) is where the dtypes split.**
  fp8: uniform solo pair 67.46/64.10, agg **241.10** — the certified
  dtype+posture face. TQ presets: uniform solo pairs (51–62 tok/s — no
  collapse legs under probabilistic drafting) with aggregates k8v4
  172.95 (−28 % vs fp8), 4bit 134.42, k3v4 137.11, 3bit 171.32; every
  preset +44…+91 % over its own greedy face. Residual: the rare
  divergent-trajectory collapse (~26 tok/s) remains possible on TQ
  dtypes (see the TurboQuant t1.0 reading and
  `results/2026-11-03-p615c-t10collapse/`; fp8 immune). On the greedy
  lane (history) every TQ preset showed the ragged pair and −51…−53 %
  aggregates.
- **Solo prefill is a chunk-size story per dtype** (U64k t/s): tq4
  1664→1459, 3328→1791, 6656→1921; k8v4 3456→1716, 8064→1945; k3v4_nc
  3712→1760, 7424→1945; 3bit_nc 2176→1549, 6528→1893; fp8 3328→1516,
  7488→1623 — bigger chunks, fewer steps, less cumulative prefix
  re-read (the B5 mechanism), monotone within every dtype.
- **k8v4 is the decode+prefill sweet spot**: fastest t0 decode AND fastest
  default prefill (2097/1945 t/s), mildest sampled-throughput cost
  (−8.5 %), and P6.14's quality-safest preset; its MNBT-4096 face keeps
  top decode (88.3–88.5) with stalls halved.

## History — pre-cert4 images (labeled by image; never mixed into the record)

Everything in this section predates cert4. It is retained because it is
the **decision record** — the k-depth law, the load-shape matrix, the
8 h soak of record — and each block names the image it ran on. Where a
face below has a cert4-native counterpart, the current record above
governs.

### Image lineage

| Image | Tag | Carried | Status |
|---|---|---|---|
| `v0.30.0-b70-cert` | `rel/v0.30.0-b70-cert` | first certified image (r42 line) | **superseded — predates the r46u glue fix; never build from it** |
| `v0.30.0-b70-cert2` | `rel/v0.30.0-b70-cert2` | 0.5.5-era lmcache wheel line | superseded — valid, older lmcache line |
| `v0.30.0-b70-p46` / `-p47` / `-p48` | — (campaign tags) | r46t depth campaign; r46v build validation; r47c 0.5.6+arc1 gates | campaign lineage, not releases |
| `v0.30.0-b70-cert3` | `rel/v0.30.0-b70-cert3` | P6 close-out engine (identical default behavior to cert4) | superseded by cert4 |
| `v0.30.0-b70-cert4` | `rel/v0.30.0-b70-cert4` | cert3 engine + P6.5 opt-in long-context deferral | superseded by cert5 — **the default record above is cert4-native** (default posture byte-identical on cert5/cert6) |
| `v0.30.0-b70-cert5` | `rel/v0.30.0-b70-cert5` | cert4 engine + P6.13 default-OFF sliced-store hunk (`LMC_STORE_SLICE_TOKENS`, experimental opt-in profile) | superseded by cert6 (identical default behavior, lacks the P6.14 TQ-aware store hunk) |
| `v0.30.0-b70-cert6` | `rel/v0.30.0-b70-cert6` | cert5 engine + P6.14 TurboQuant-aware lagged store (inert under `fp8_e4m3`) + the four then-validated opt-in `turboquant_*` dtype faces | superseded — fp8 behavior byte-identical; its TQ dtypes carry the pre-P6.19 corrupted spec-verify path (P6.18 Class B), so **do not serve TQ dtypes on cert6** |
| `v0.30.0-b70-cert7` | `rel/v0.30.0-b70-cert7` | cert6 engine + the P6.18 `ArcXpuPlatform` config-time guard refusing every `turboquant_*` kv-cache dtype (fp8 serving bytes byte-identical to cert6) | superseded by cert8 — fp8 behavior byte-identical; the guard superseded by the P6.19 fix |
| `v0.30.0-b70-cert8` | `rel/v0.30.0-b70-cert8` | cert7 engine with the P6.18 refusal lifted + the **P6.19 TQ spec-verify fix** (`patches/tq_spec_fix.py` wheel-side hook: v19b verify-capture forcing + split-row dispatch, default-active on TQ dtypes; `patches/tq_mq_aos.py` MQ kernel opt-in-experimental; fp8 serving bytes identical — hooked module never imports) — all four TQ presets serve clean | superseded by cert9 — fully valid; its fp8 TTFT faces carry the pre-P6.20 inline store path |
| `v0.30.0-b70-cert9` | `rel/v0.30.0-b70-cert9` | cert8 engine + the **P6.20 prefill host-path patch** (L3 off-thread LMCache store default-ON `VLLM_ARC_LMC_ASYNC_STORE`; L6 dispatch-late handoff; `patches/step_trace.py` silent-off diagnostics): store D2H gathers + CPU insert leave the engine thread, device ops stay on the in-order stream — U32k −0.74 s / U64k −1.64 s / U8k −0.08 s TTFT; spec gate byte-exact cert8 values; TQ dtypes re-gated clean (dominant identity hashes identical to the cert8 gate) | superseded by cert10 — fully valid (cert10's engine is byte-identical; its default-posture numbers are this image's) |
| `v0.30.0-b70-cert10` | `rel/v0.30.0-b70-cert10` | cert9 engine (exhaustive `vllm_gaudi/` md5 walk diff_lines=0 — cert9's vllm-layer certs carry) + the **P6.24 oneCCL fp8-wire overlay**: same-version oneCCL 2022.0.0 rebuild (ABI/perf-identical, r96-measured) + `arc-fp8-allreduce.patch` v4 (e4m3 wire for the BMG TP2 fp16 large ring; u32-packed peer stores per the sector law; branchless converters; min-count 2M elems; env trio `CCL_SYCL_ALLREDUCE_FP8/_MIN_COUNT/_STATS`, defaults 0/2097152/0 — **silent-off, stock path unless opted in**); manifest row `oneccl=2022.0.0+arc1`, fat-lib sha pinned at both SONAME paths; opt-in **1.65× at the dominant AR**, every p65 TTFT face beats cert9 (U8k −13 % / U64k −8.2 % / U32k −3.8 % / C32k −4.6 %); r98 quality parity + r99 full battery + r100 wire-law door-closes (`results/2026-11-05-p624-fp8-ar/`) | superseded by cert11 — fully valid; its oneCCL overlay rows ride every later cert |
| `v0.30.0-b70-cert11` | `rel/v0.30.0-b70-cert11` | cert10 overlay (294→294 file walk; changed set exactly `vllm_gaudi/xpu/platform.py` + pyc) + the **P6.26 quark/MX platform patch** (`quark` in `supported_quantization`, `supports_mx() → True`) — opens the MXFP4 checkpoint lane, zero dispatch change for existing lanes; mxfp4 lane faces: rhmxfp4 U32k 29.01 s / qurk 32.28 s, decode penalties 1.21×/1.29× — **fp8 stays the performance default** (`results/2026-11-06-p626-mxfp4/`) | superseded by cert17 — fully valid; every byte carries except platform.py (+pyc). The MXFP4 checkpoint lane does NOT boot on cert10 and earlier |
| `v0.30.0-b70-cert17` | `rel/v0.30.0-b70-cert17` | cert11 engine + the **P6.28 gemma-4 chain**: cert12 B70 prefill launch-config table (head-256/512 keyed points) + cert16 warm-gate v4 (engine-exact int-softcap triton cache-key fix, G2 freeze) + cert17 lmcint kit hardening (detector BS floor 16 + the fail-loud two-head-size `[ARC-P628-LMC-KVGATE]` boot gate); **certified gemma posture: `ARC_LMC=0` + `ARC_RPC_TIMEOUT=1200`**; qwen3.8 fp8 engine bytes carry from cert11 (`results/2026-11-06-p628-gemma4/`) | superseded by cert18 — fully valid; the gemma-4 prefill/decode records are the pre-FA2 baseline |
| `v0.30.0-b70-cert18` | `rel/v0.30.0-b70-cert18` | cert17 + the **P6.28 FA2 route**: full-attention head-512 calls dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel in the triton unified-attention wrapper (`ARC_ATTN_FA2=0` escape hatch); 34× at production geometry in microbench; gemma faces 2.2–16.3× faster, decode +6 %, qwen discriminator green with route marker 0 (`results/2026-11-06-p628-gemma4/phaseC/cert18_l1/`) | superseded by cert19 — fully valid; gemma-4 prefill/decode records are the pre-slide baseline |
| `v0.30.0-b70-cert19` | `rel/v0.30.0-b70-cert19` | cert18 + the **P6.29 sliding-window FA2 route**: sliding-window head-256 calls (window (1023,0) class) dispatch to the same ESVD FA2 kernel windowed; gemma faces 2.4–2.9× faster (face32k at qwen-parity), decode +45 % solo / 2.64× agg8; qwen discriminator in-band; ngram spec for gemma closed negative (`results/2026-11-07-p629/`) | superseded by cert20 — fully valid; its gemma-4 records stand |
| `v0.30.0-b70-cert20` | `rel/v0.30.0-b70-cert20` | cert19 + the **P6.30 online-MXFP4 W4A16 route**: arc4 wheel (`fp4_gemm_w4a16` — bf16 acts × f4 weights, group-32 e8m0) + `mxfp4_w4a16.py` reroute behind `ARC_MXFP4_MODE`; qwen online mxfp4 3.66/17.12 s + 62.5–65.2/301–311 — superior to online mxfp8 on every axis; gemma 3.94/21.16 + 40.5/262; qwen fp8 discriminator in-band (`results/2026-11-08-p630-mxfp4-perf/`) | superseded by cert21 — fully valid; its online-mxfp4 records stand |
| `v0.30.0-b70-cert21` | `rel/v0.30.0-b70-cert21` | cert20 + the **P6.31 online-MXFP8 W8A16 route**: arc5 wheel (`fp8_gemm_w8a16_mxfp8` — bf16 acts × f8 weights, group-32 e8m0, zero per-call act quant; carries the oneDNN-refused `fp4_gemm_w4a8` probe op) + `mxfp8_w8a16.py` reroute behind `ARC_MXFP8_MODE` + the platform entry; qwen 3.61/17.07 s + 53.8–55.6/265.9 (1.37–1.55× over stock W8A8, spec 0.9804), gemma 3.87/20.94 + 27.7/192.4 at maxlen 40960 (1.57–1.85×); adjudication: checkpoint-fp8 stays the perf default, mxfp4-w4a16 the superior online lane, W4A8 banked closed-negative; r125 three-lane gate green (`results/2026-11-08-p631-mxfp8-w8a16/`) | superseded by cert22 — fully valid; its online-mxfp8 records stand |
| `v0.30.0-b70-cert22` | `rel/v0.30.0-b70-cert22` | cert21 + the **P6.33 skinny-M kernel family, banked OFF**: arc6 wheel (`skinny_fp8_gemm_w8a16`/`skinny_fp8_gemm_w8a16_mxfp8`/`skinny_fp4_gemm_w4a16` — GEMV-regime M≤8 weight-only GEMMs, register dequant, deterministic split-K; + W4A8 dispatch arms + v0.1.17 Option-B backports) + `skinny_m.py` reroute behind `ARC_SKINNY_M` — **unset/off = stock routes (certified default)**: the r127 e2e battery falsified the reroute (under XPU-graph replay stock oneDNN runs 420–680 GB/s device-side vs the skinny streaming ceiling ~300 GB/s; gemma solo −38/−42%; size-1 decode graph pads M→2); kernels correct (r126 GATE-1/2/3) and retained in-tree; r127g2 cert leg: gemma 27.7/192.9 exact ref, zero route markers (`results/2026-11-08-p633-skinny-m/`) | **current release** |

### Depth-selection faces (cert-era images — the k3/k4/k5 greedy record)

Measured on the original-cert/cert2-era engines (r42 k4 leg, r43 k5
leg, r46t k3 on the p46 image). The current-image reproduction of this
family is the cert4 record above (294.7–300.5 / 83.7–84.9 across five
cert4 boots; sweep acceptance within ~1 pp of the p46 campaign —
engine unchanged through cert2→cert4 default postures). Depth law
unchanged: **k3**.

| Face | k3 (recommended, r46t) | k4 (r42) | k5 (dropped, r43) |
|---|---|---|---|
| agg4x256 steady tokens/s | 299.86 | 297.65 / 297.94 | 303.48 / 305.42 |
| solo200 tokens/s (incl. TTFT) | 84.46 / 84.53 | 85.01 / 85.16 | 86.39–87.13 (×4) |
| greedy acceptance | 85.6% | ~87% | 89.2% (pos-1 0.8919) |

k3 and k4 tie within ±0.7 % on the certified face; k5 is faster there
(+1.9–2.5 % agg, +1.5–2.4 % solo, on 89.2 % greedy acceptance) — but
that greedy-era edge inverts under real-world sampling (r46t, below),
which is why k5 was dropped and k3 recommended.

Cert-era posture-level faces from the same legs (r42/r43/r46, superseded
by the cert4-native U-bar and warm tables above):

| Face (cert-era) | Value |
|---|---|
| TTFT t1 8k / 16k / 32k (ms) | 3987 / 7238 / 15534 (k5 leg r43); 3445 / 7252 / 15552 (k4 leg r42 — 8k steady face) |
| TTFT t1 64k (ms) | 40,926 (concurrent-load run — conservative upper bound) |
| prefill rate at 8k face | 2,308 tok/s (7,950 tok / 3.445 s) |
| warm t2 8k / 16k / 32k (ms) | 848 / 675 / 1178 (prewarm-v7 corpus) |
| warm-hit (prefix re-hit) | 740 ms (face limit 750 ms) |
| survival flip (first request after boot) | HTTP 200, 6.09 s |
| solo step time @ c1 (200 tokens / 2.30 s) | ~48.7 ms/step (k5 leg) |
| agg4x256 first run after boot (k3) | warmup cost on run #1 (93.7 vs 299.9 steady) — first-traffic face, gone by run #2 |

Notes: the 8k t1 TTFT is first-cell-of-the-ladder (warmup-sensitive);
the k4 leg's 3445 ms shows the steady 8k face. The 64k cell was taken
while the 8 h soak battery was cycling (conservative — a quiet-window
run would be equal or faster).

### Real-world faces — the k3 decision (r46t campaign, p46 image)

**A p46-image campaign** (2026-10-30, the certified engine of that
day). Its job was depth selection under real-world sampling; it picked
**k3**. The r70 campaign re-measured the k3 sweep **natively on
cert4** — acceptance within ~1 pp per cell, throughput ±6 % (current
record above), so the depth law now stands on cert4-native data.

Full methodology and evidence: `results/2026-10-30-p5-r46t-temps/`
(VERDICT + raw batteries + acceptance counters). Two request regimes,
both on the run7n battery (agg4x256 ×2 + solo200 ×2):

**A. xhigh thinking + real-world sampling** (temperature sweep, top_p
0.95 / top_k 20 / min_p 0 / presence_penalty 0 / repetition_penalty 1,
`enable_thinking: true, reasoning_effort: "xhigh"`). agg tok/s = 2-run
mean; steady = run-2 mean (run #1 after a fresh boot carries first-
battery warmup, heaviest on k3):

| temp | k3 agg | k4 agg | k3 solo | k4 solo | acceptance k3 / k4 |
|---|---|---|---|---|---|
| 0.2 | 174.9 * | 201.7 | 63.9 | 64.6 | 57.3 % / 50.2 % |
| 0.4 | 208.8 | 184.4 | 63.0 | 60.4 | 57.1 % / 45.9 % |
| 0.6 | 202.8 | 193.6 | 61.3 | 62.3 | 53.1 % / 46.6 % |
| 0.8 | 195.8 | 184.9 | 61.4 | 59.1 | 52.3 % / 43.5 % |
| 1.0 | 186.2 | 182.6 | 61.1 | 54.3 | 49.5 % / 43.2 % |
| mean | **193.7** (steady 205.4) | 189.4 (steady 194.2) | **62.1** | 60.2 | — |

\* run #1 = 137.2 (warmup), run #2 = 212.6. k5 measured 185.8 agg
mean before being dropped (evidence retained in the results dir).

**B. alt sampling, non-thinking** (temperature 0.7 / top_p 0.80 /
top_k 20 / min_p 0 / presence_penalty 1.5 / repetition_penalty 1.0):

| leg | agg #1 / #2 | solo #1 / #2 | acceptance |
|---|---|---|---|
| k3 | 170.6 * / 254.9 | 76.9 / 92.2 | 88.5 % |
| k4 | 236.9 / 252.6 | 85.3 / 104.3 | 82.6 % |

\* warmup-contaminated run #1. Near-deterministic continuation keeps
acceptance high — the one regime where k4's deeper draft pays (solo
104.3 vs 92.2); steady agg is a tie.

**Non-MTP baseline** (serve line minus `--speculative-config`): t1.0
xhigh thinking agg 118.8 / solo 33.5; alt face steady agg 124.2 /
solo 33.4. **MTP value on measured faces: k3 = 1.57× (xhigh t1.0) and
2.05× (alt steady) aggregate over no-spec.**

Mechanism (measured across the sweep): speculative acceptance falls as
temperature rises, and falls faster with deeper drafts (k3 49.5 % vs
k5 33.5 % at t1.0 xhigh; k3 85.6 % at greedy t0). Depth pays only at
high acceptance — real-world sampled thinking sits below the
crossover, which is why k3 is the recommendation and k5's greedy-era
win inverts under sampling.

### 12-cell load matrix (r44 — cert-era image; the load-shape record)

`vllm bench serve`, random dataset, output 512, ignore-eos, 12/12 cells
zero-error. TTFT/TPOT/ITL in ms:

| Cell | TTFT | TPOT | ITL | req/s | out tok/s | Acc % | Acc len |
|---|---|---|---|---|---|---|---|
| i1024-c1 | 583.67 | 8.60 | 46.77 | 0.20 | 102.79 | 89.15 | 5.46 |
| i1024-c4 | 1985.32 | 11.07 | 64.47 | 0.52 | 264.69 | 96.92 | 5.85 |
| i1024-c16 | 6237.01 | 28.97 | 126.85 | 0.64 | 325.55 | 67.96 | 4.40 |
| i16384-c1 | 7380.80 | 9.13 | 49.11 | 0.08 | 42.50 | 88.00 | 5.40 |
| i16384-c4 | 13263.11 | 31.51 | 160.63 | 0.13 | 66.43 | 82.44 | 5.12 |
| i16384-c16 | 46421.90 | 111.19 | 523.98 | 0.14 | 72.38 | 74.70 | 4.73 |
| i32768-c1 | 16083.60 | 10.83 | 52.19 | 0.05 | 23.68 | 76.42 | 4.82 |
| i32768-c4 | 34286.64 | 44.64 | 210.74 | 0.07 | 34.47 | 74.69 | 4.73 |
| i32768-c16 | 97008.14 | 161.02 | 708.16 | 0.07 | 34.84 | 68.28 | 4.41 |
| i131072-c1 | 101094.58 | 20.71 | 66.15 | 0.01 | 4.58 | 43.73 | 3.19 |
| i131072-c4 | 162169.64 | 162.68 | 854.78 | 0.01 | 6.24 | 85.76 | 5.29 |
| i131072-c16 | 903979.67 | 184.16 | 843.54 | 0.01 | 4.78 | 71.91 | 4.60 |

Raw cells: `results/2026-10-30-p4-r44-matrix-certified/cell-*.txt`.

### Speculative decoding detail (64k cell, r46 — cert-era)

Acceptance figures split by sampling condition: the cert-era headline
faces (pos-1 0.8919, spec-gate counters) are **greedy (temperature 0)**
— k5-leg values; k3's greedy acceptance is 85.6 % (r46t). Matrix-cell
and 64k-cell acceptance below are **server-default sampling**
(`vllm bench serve` 0.30 does not force greedy — see
TESTING-arc.md suite 5). TTFT/t1 faces are sampling-insensitive.

Acceptance 87.43%, mean length 5.37 tokens/step, 218 drafts / 1090
draft tokens / 953 accepted. Per-position acceptance:
96.79 / 96.79 / 96.79 / 96.33 / 50.46 (positions 0–4) — position 4 is
the k5 draft tail; see the 12-cell table above for per-cell positions.

### r46 8 h soak (cert-era — the long-soak record)

**GREEN**: 46 cycles over a 29,156 s window (each cycle: run7n →
longctx → prefix/churn → integrity probe), **zero error blocks,
integrity probes 46/46 PASS, final golden 5/5 byte-identical (zero
drift), speculative pos-1 rate 0.8919 steady to the last cycle, health
200 end-state**. Evidence: `results/2026-10-30-p5-r46-soak/` (VERDICT +
soak log + summary). The cert4-native soak face is the r70 leg above.
