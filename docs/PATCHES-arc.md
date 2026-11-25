# vllm-arc — the complete patch inventory

Every code delta this fork layers over upstream `vllm-project/vllm-gaudi`
@ `v0.30.0`, one entry per patch module, each with purpose, mechanism,
provenance, knobs, evidence, cert of adoption, and status. Nothing ships
unevidenced (law 6): every row cites its banked validating leg under
`results/`. The engine image in currency is
**`vllm-arc:v0.30.0-b70-cert28`** (tag `rel/v0.30.0-b70-cert28`):
manifest `cert28_fork_sha=43f84cf977f9444900ff21387715ca5865db4c1a`,
baked wheel `vllm_xpu_kernels-0.1.14.1+arc7mx` with in-image-asserted
`.so` pins — `libattn_kernels_xe_2`
`61d2f5a365f05084b3dd72c4ae1da3e0`, `_vllm_fa2_C.abi3.so`
`6e6c48508d40cb4c5007f27f79df97cf`, `libmqa_logits`
`190e261edc79b3b497d8d8e801e65757`.

Companion docs: [`CONFIGURATION-arc.md`](CONFIGURATION-arc.md) (every knob
and posture), [`PERFORMANCE-arc.md`](PERFORMANCE-arc.md) (numbers),
[`MIGRATION-arc.md`](MIGRATION-arc.md) (evidence table),
[`docs/reports/`](reports/README.md) (per-config performance reports).

## How patches load — four layers

1. **In-place patch modules** (`vllm_gaudi/xpu/patches/`, registered in the
   `_XPU_PATCH_MODULES` tuple of [`vllm_gaudi/xpu/platform.py`](../vllm_gaudi/xpu/platform.py)):
   imported at platform-register time in every vLLM process, each module
   patches core classes at runtime — idempotent, per-process, and echoed by
   a serve-log marker when active.
2. **Exec-wrap finder hooks** (same platform file): a meta-path finder
   rebinds symbols inside `vllm.*` / `lmcache.*` modules at their first
   import — used where an in-place patch cannot reach (engine-core
   scheduler delta, worker-side mamba/LMCache glue, the kv-connector
   module, wheel-side hooks, the MX KV admission modules). Loading order
   is the finder's contract. The same finder also carries **source
   transforms** (`_SOURCE_XFORMS`, P6.47): callables that edit a wheel
   module's source text before Python compiles it — used where even
   post-exec rebinding cannot reach (pydantic-decorated dataclass
   schemas, closed enums: the CacheDType literal widening and the MXKV
   backend-enum member).
3. **Image overlay patches** (`external_patches/vllm/`, applied at image
   build via `tools/cert*.Dockerfile`): whole-file or AST patches to
   upstream files that no runtime hook can replace cleanly (triton
   attention launcher table, FA2 dispatch routes, warmup gate). The
   Dockerfile asserts the base bytes before patching; the image manifest
   banks `cert<N>_fork_sha` + wheel sha256 rows.
4. **Custom wheels** (`external_patches/vllm-xpu-kernels/` build kits,
   `external_patches/oneccl/`, `external_patches/lmcache/`): fork-built
   binaries installed into the image under their real package names
   (pip validates filenames — engineering law 3 of P6.30); pins recorded
   in `external_patches/wheels.pins.md5`.

## Layer 1 — in-place patch modules (`_XPU_PATCH_MODULES`, load order)

### `graph_safe_output.py` — graph-safe worker output copies
- **Purpose**: vLLM 0.30's `AsyncOutput` D2H copies ride a side stream
  whose events the XPU-graph capture pool cannot own — torn outputs under
  graph replay.
- **Provenance**: in-house (P2).
- **Knob**: always-on (safety class, no opt-out).
- **Evidence**: `results/2026-10-28-p2-*` (defect-b reproduction + fix A/B).
- **Cert**: cert5-era; in every image since. **Status: ACTIVE, safety-critical.**

### `speculator_full_capture.py` — PIECEWISE→FULL graph coercion
- **Purpose**: v1's default `FULL_AND_PIECEWISE` compilation mode hands the
  speculator PIECEWISE-family graphs, which the MTP draft loop cannot
  replay; coerces the speculator's captures to FULL.
- **Provenance**: in-house (defect-a family, P2).
- **Knob**: always-on.
- **Evidence**: `results/2026-10-28-p2-*` (defect-a reproduction).
- **Cert**: cert5-era. **Status: ACTIVE, safety-critical.**

### `first_request_warmup.py` — first-request warmup interlock
- **Purpose**: with XPU graphs the boot reports healthy without ever
  running a real-size eager TP allreduce; the first real request then pays
  a multi-second oneCCL lazy-init stall (TTFT spike). The interlock runs
  one full-size warm collective before `/health` turns green.
- **Provenance**: in-house (lane W, r16/r17).
- **Knob**: always-on.
- **Evidence**: `results/2026-10-28-p2-graph-p12b-r16-kwcapture/`.
- **Cert**: cert5-era. **Status: ACTIVE.**

### `gdn_metadata_fastpath.py` — GDN attention-metadata fast path
- **Purpose**: steady decode/spec-decode GDN metadata builds without
  CPU-mask indexing of device tensors (≤5 ms/step law); removes a host
  stall from the graphed decode step.
- **Provenance**: in-house (p10, r19–r22a).
- **Knob**: always-on.
- **Evidence**: `results/2026-10-28-p2-*` (r19–r21b step decomposition,
  r22a A/B).
- **Cert**: cert5-era. **Status: ACTIVE.**

### `fp8_block_w8a16.py` — block-fp8 linear → `fp8_gemm_w8a16`
- **Purpose**: the fp8-checkpoint lane's block-quantized linears serve as
  W8A16 (fp16 activations, block-scale native oneDNN GEMM, per-call input
  quant skipped) — 3–8×/GEMM at decode Ms vs the stock W8A8 path.
- **Provenance**: in-house (p44, r40).
- **Knob**: always-on for the fp8-checkpoint lane.
- **Evidence**: `results/2026-10-29-p4-r40-forward-gemm-capture/`
  (same-silicon two-image microbench at the six per-rank TP2 projections).
- **Cert**: cert6-era. **Status: ACTIVE (the certified incumbent lane).**

### `step_trace.py` — per-step host timeline (diagnostic)
- **Purpose**: engine-side per-step host timeline for the prefill
  decomposition lanes (P6.20).
- **Provenance**: in-house (p620).
- **Knob**: `ARC_STEP_TRACE` — silent-off diagnostic (envs.md diagnostics
  class, deliberately not in the envs.py registry).
- **Evidence**: `results/2026-11-05-p620-*`.
- **Cert**: — (probe only, never ships in a serve posture).
  **Status: ACTIVE tooling, silent-off.**

### `ar_trace.py` — per-call collective census (diagnostic)
- **Purpose**: message-size telemetry for the TP2 allreduce family
  (~30.4 % of rank0 device time at U32k — r67b sizing).
- **Provenance**: in-house (p622).
- **Knob**: `ARC_AR_TRACE` — silent-off diagnostic.
- **Evidence**: `results/2026-11-05-p622-*`.
- **Cert**: —. **Status: ACTIVE tooling, silent-off.**

### `mxfp4_w4a16.py` — online MXFP4 → `fp4_gemm_w4a16` (cert20)
- **Purpose**: online `--quantization mxfp4` on a BASE bf16 checkpoint
  reroutes standard linears to the arc4 wheel's W4A16 op (bf16 acts × f4
  weights, group-32 e8m0 scales, oneDNN weight-decomposition GEMM) —
  kills the two stock W4A4 defects (per-call act quant ≈ GEMM cost at
  decode Ms; the (f4,f4) joint dtype can never pass oneDNN gemmstone's
  counter-operand gate). **The decode champion lane** (P6.35-V
  lineage; certified read: qwen mxfp4-W4A16 + MTP k3 = 103.3 tok/s @
  8K).
- **Provenance**: in-house (P6.30 FIX-P; wheel op from the
  `vllm-xpu-kernels` arc4 build kit).
- **Knob**: `ARC_MXFP4_MODE` = `w4a16` (route, default) / stock w4a4
  control; enabled by the `ARC_QUANT=mxfp4` posture.
- **Evidence**: `results/2026-11-08-p630-mxfp4-perf/` (r116–r120d chain
  + r121 gate).
- **Cert**: **cert20**. **Status: ACTIVE — the superior online lane.**

### `mxfp8_w8a16.py` — online MXFP8 → `fp8_gemm_w8a16_mxfp8` (cert21)
- **Purpose**: the same act-quant elimination for the 8-bit online lane
  (bf16 acts × f8 weights, weight-side group scales). 1.37–1.85× over
  stock W8A8 — but below fp8-checkpoint and mxfp4 on every axis; kept as
  the lawful mxfp8 route (mxfp8 deprecated for decode, P6.35-V).
- **Provenance**: in-house (P6.31; arc5 wheel).
- **Knob**: `ARC_MXFP8_MODE` = `w8a16` (default) / `w8a8` control.
- **Evidence**: `results/2026-11-08-p631-mxfp8-w8a16/` (r122–r125).
- **Cert**: **cert21**. **Status: ACTIVE, deprecated for decode.**

### `skinny_m.py` — skinny-M decode GEMM reroute (cert22, banked OFF)
- **Purpose**: M≤8 decode GEMMs to the arc6 wheel's GEMV-regime kernels
  (register dequant, fixed-order split-K). Eager-kbench wins did not
  survive graph replay (stock oneDNN runs 420–680 GB/s device-side
  under replay — the motivating 197 GB/s was a host-rate artifact);
  gemma regressed −38/−42%. Route ships **default OFF**; kernels stay
  in-tree behind the r127f tensor-core gate.
- **Provenance**: in-house (P6.33; arc6 wheel).
- **Knob**: `ARC_SKINNY_M` = `on` (opt-in per-lane M caps) / unset-off.
- **Evidence**: `results/2026-11-08-p633-skinny-m/` (r126 correctness,
  r127 five-leg falsification, r127g2 cert leg).
- **Cert**: **cert22** (behaviorally ≡ cert21 at serve).
  **Status: IN-TREE, DEFAULT OFF.**

### `graph_exec_doublebuffer.py` — XPU graph-exec double-buffer (cert23)
- **Purpose**: torch 2.13 `XPUGraph.replay()` on the SAME exec object
  blocks the host until that exec's prior launch retires (13.96 ms
  same-graph in-flight vs 8–12 µs alternating). Captures every graph ×2
  into the shared pool and alternates replays per desc; second-capture
  failure degrades to stock semantics with a WARN (never a boot death).
- **Provenance**: in-house (P6.35 D1).
- **Knob**: `VLLM_ARC_GRAPH_EXEC_DB` = `1` (ACTIVE, default) / `0`
  kill-switch (dboff proven in-gate).
- **Evidence**: `results/2026-11-10-p635-telemetry/` (leg J replay max
  24 → 0.37 ms; r135b control).
- **Cert**: **cert23**. **Status: ACTIVE — costless hygiene fix.**

## The MX KV lane — `vllm_gaudi/xpu/patches/mx_kv/` (cert24 → cert28)

The MX KV decode lane (P6.47, 2026-11-11+): MX (OCP
microscaling) `--kv-cache-dtype {mxfp4, mxfp8, mxk8v4}` KV caches at
FP8-class decode. From-scratch in-house — no donor carries MX KV
(survey `results/2026-11-11-p647-mxkv-research/`); the engine
integration surface follows the wheel's TurboQuant backend (subclass,
don't fork: uint8 paged allocation via `customize_spec`, page-size
unification, TurboQuantMetadata, XPU-graph capture, and the P6.19
`tq_spec_fix` hook all ride along). Format-discovery law: every kernel
infers the format from the cache slot width, so an MxConfig mismatch
against the actual allocated cache is impossible by construction.
Not `_XPU_PATCH_MODULES` installs — selection-time routing (the backend
is chosen per posture through the platform's `get_attn_backend_cls`).

### `mx_kv/engine.py` — `MxAttentionBackend` / `MxAttentionImpl`
- **Purpose**: the attention backend the three MX `--kv-cache-dtype`
  strings route to (platform routing → the MXKV enum member → this
  class); TQ-subclass design, the split-row continuation law
  re-implemented 1:1, `active_kv_dtype` stamped at selection time (the
  one process-wide format selector — spec customization needs the
  format before any spec carries it).
- **Knob**: `ARC_KVDTYPE={mxfp4,mxfp8,mxk8v4}` (serve-script level);
  `ARC_MXKV_NATIVE=1` routes verify-batch rows to the wheel-side
  native decode.
- **Evidence**: `results/2026-11-11-p647-r148-mxkv-admission/` (first
  end-to-end admission), `results/2026-11-12-p647-cert24-gate/`.
- **Cert**: **cert24**. **Status: ACTIVE — the certified MX KV lane,
  current form cert28 (r231d clamp wheel baked).**

### `mx_kv/mx_layout.py` — slot layout + pool arithmetic
- **Purpose**: single source of truth for the byte geometry every
  layer agrees on (kernels, engine cache-shape integration, offline
  harness, report capacity tables): mxfp8 264 B, mxk8v4 200 B, mxfp4
  136 B per (token, head) at D=128 — vs 512 B fp16 and 256 B
  fp8_e4m3 (mxk8v4 = 0.78× fp8 bytes; mxfp4 = 1.88× fp8 token
  capacity). A layout constant lives here, never duplicated at a call
  site.
- **Status: ACTIVE (the layout law).**

### `mx_kv/mx_quant.py` — bit-exact quantization reference
- **Purpose**: the PyTorch oracle every Triton kernel in the family
  must round-trip exactly (E8M0 group scales 1 B per 32 elements,
  e4m3 hardware cast / e2m1 7-midpoint chain, TQ-AoS slot interleave
  — even element low nibble); enforced offline by
  `tools/ar/p147_mxkv/p147_mxkv_offline.py` on XPU before any engine
  leg. Capacity doctrine: mxfp4's 4-bit K degrades long-return
  retrieval under churn (gate run1 essay REP-LOOP) — mxk8v4 is the
  quality-doctrine mix (8-bit keys, 4-bit values).
- **Evidence**: `results/2026-11-11-p647-r147-mxkv-lega/`,
  `results/2026-11-12-p647-cert24-gate/`.

### `mx_kv/triton_mx_store.py` — fused quantize+pack+scatter store
- **Purpose**: one kernel writes a token's whole K+V slot contiguously
  (K, Ks, V, Vs — no separate scale tensors, no gather). K and V load
  through SEPARATE token+head strides (the P6.47 vstride law: the two
  tensors arrive with independent layouts; per-tensor strides or
  42–80 % V corruption hides behind offline passes). The E8M0
  exponent is derived exactly from the fp32 divide result's bits —
  no log2/ceil ulp divergence can desync a scale byte between torch
  and Triton.
- **Evidence**: `results/2026-11-12-p647-mxkv-vstride-fix/`.
  **Status: ACTIVE.**

### `mx_kv/triton_mx_decode.py` — paged flash decode (v2 split-KV)
- **Purpose**: single-query-token decode with inline MX dequant
  (scale metadata 1 B per 32 elements — 8× lighter than TQ's fp16
  scale+zero; e2m1 is pure shift/compare arithmetic). v2
  (decode-perf): split-KV stage 1 grid (B, Hq, 32) + log-sum-exp
  combine — each program walks one contiguous split, surplus programs
  park explicit empty markers (never stale mid bytes), and the
  per-batch effective split count is derived in-kernel from the
  device-resident seq_len (no host reads).
- **Evidence**: `results/2026-11-12-p647-decode-perf-research/` (v1
  latency-bound diagnosis + v2), the S1/B1 ladder
  `results/2026-11-14-p647-s1-acceptknee/` +
  `results/2026-11-18-p647-b1-v5flip/` (the certified v5 lane —
  `ARC_MXKV_V5_*` dev knobs).
- **Status: ACTIVE — the certified gemma MX lane; the qwen MX
  postures route native since cert27.**

### `mx_kv/mx_dequant.py` — whole-cache dequant (continuation prefill)
- **Purpose**: bulk-dequant the cached prefix into dense fp16 K/V then
  flash-attn for large continuations (the TQ
  `_tq_full_dequant_kv` analogue — chunks above the continuation
  threshold dequant once instead of rescanning the compressed context
  per query row). Safety: tokens at or past `cached_len` write zeros
  (a wild gather could fault); pure device reads, shape-static,
  replay-safe.
- **Status: ACTIVE.**

### `mx_kv/native_decode.py` — arc7mx native-decode bridge
- **Purpose**: routes the MTP verify-batch rows to the wheel's
  `mx_paged_decode` (vendor-class CUTLASS/Xe2 kernel) inside XPU-graph
  capture — the split-row plan already holds the kernel's contract
  (one row per (request, draft-token) with its own causal limit and
  gathered block table; device ops only, no pageable H2D). A
  missing/old wheel fails LOUD (no silent fallback in A/B legs);
  capture-time telemetry buckets (fmt, rows, max_seqlen_k, bt width).
- **Knob**: `ARC_MXKV_NATIVE=1` (default-off; the certified qwen MX
  posture sets it — cert27 battery: all three lanes native PASS).
- **Evidence**: `results/2026-11-20-p647-r198-arc7mx-offline/`
  (offline correctness), the r212 e2e chain from
  `results/2026-11-21-p647-arc7-r212a-dispatch-e2e/` onward,
  `results/2026-11-23-p647-cert27-battery/`,
  `results/2026-11-24-p647-cert28-battery/` (the r231d split clamp).
- **Cert**: **cert27** (qwen postures). **Status: ACTIVE.**

### Admission modules + platform routing (P6.47 leg B)
- **`patches/mx_dtype_map.py`** (exec-hook on `vllm.utils.torch_utils`):
  the three MX dtypes map to the packed-uint8 allocation class
  (`STR_DTYPE_TO_TORCH_DTYPE`). **`patches/mx_kv_mode.py`** (exec-hook
  on `vllm.v1.kv_cache_interface`): bounded element-class alias onto
  the TURBOQUANT `KVQuantMode` members (the mode conveys only
  packed-uint8-class semantics; format identity travels on the dtype
  string + `active_kv_dtype`).
- **Platform source transforms** (`_SOURCE_XFORMS` +
  register-time injection in `vllm_gaudi/xpu/platform.py`): the
  CacheDType Literal widening (`vllm/config/cache.py` — a TRUE schema
  widening: CLI admission, help text, and config round-trip all see
  the real members; pydantic dataclass schemas cannot be rebound
  post-exec) and the MXKV member of `AttentionBackendEnum`
  (`vllm/v1/attention/backends/registry.py` — anchor-count asserts
  fail loud on wheel drift).
- **Evidence**: `results/2026-11-11-p647-r148-mxkv-admission/`.
  **Status: ACTIVE.**

### Posture notes (serve-script level)
- MX lane postures auto-off LMCache: e2m1-V double-quant garble (r168)
  + decode-parity LMC-free (r176) — the serve script prints the law
  and WARNs on a forced override.
- Dev knobs, never in a certified posture: `ARC_MXKV_OVERLAY` /
  `ARC_MXKV_WHEEL` (live swap for A/B legs), `ARC_MXKV_V5_*`,
  `ARC_MXKV_VERIFY_DECODER`, `ARC_MXKV_K2` / `ARC_MXKV_K3`.

### The lane's evidence spine (cert ladder)
cert24 gate → cert25zb certification → S1/B1 decode-perf ladder →
cert25zc / cert26 batteries → arc7mx offline + the r212 e2e chain →
cert27 battery (qwen native) → **cert28** (r231d clamp wheel baked +
the record legs r234–r238: n1 ITL, agg8, n8 steady-state long-ctx,
acceptance mining). Current record carriers:
[`PERFORMANCE-arc.md`](PERFORMANCE-arc.md).

## Layer 2 — exec-wrap finder patches

### `sched_tail_split.py` — scheduler tail-split (r35)
- **Purpose**: when eagle backoff would leave nothing bankable, plan the
  mamba-aligned chunk end at the deepest full block (engine-core
  scheduler delta).
- **Evidence**: `results/2026-10-30-*` (r35 chain).
- **Status: ACTIVE.**

### `mamba_lmcache_glue.py` + `mamba_hybrid_snap_glue.py` — hybrid-GDN
  LMCache store/load glue (p41/p41c)
- **Purpose**: worker-side mamba-state store/load for the hybrid-GDN
  target under the LMCache connector (p41), plus the carry-branch snap
  fix for the MTP+align posture (p41c, r31).
- **Provenance**: ported per the fork law from certified donor lineages
  (`external_patches/lmcache/README.md`, patch-table rows e02/e03).
- **Evidence**: `results/2026-10-30-p3-*` (r31/r42 chains).
- **Status: ACTIVE on every `ARC_LMC=1` posture (gemma runs LMC=0).**

### `lmcache_hma_connector.py` — LMCacheConnectorV1 HMA + 0.5.5 port
  (p41b/p42)
- **Purpose**: vllm-side connector glue — HMA allocator compat +
  lmcache-0.5.5 API port + lookup-elide/lookup-memo (D v139 extyield2).
  Loaded via sitecustomize exec_module hook on the kv_connector module.
- **Provenance**: donor port (line D v139; `external_patches/lmcache/`).
- **Evidence**: `results/2026-10-30-p3-*` (r42 boot chain).
- **Status: ACTIVE on `ARC_LMC=1` postures.**

### `tq_spec_fix.py` — TurboQuant spec-verify fix (cert8)
- **Purpose**: the P6.18 corruption root-caused to three stacked
  spec-verify defects (raw-KV fast-path baked into captured verify
  graphs; replay-broken multi-row synthetic-decode dispatch; the wheel's
  MQ kernel reading the wrong tree). Wheel-side hook forces donor-v19b
  capture + split-row dispatch. Default-ON.
- **Evidence**: `results/2026-11-04-p619-tq-fix/` (all-dtype gate: zero
  garble on all four TQ dtypes + the fp8 control).
- **Cert**: **cert8**. **Status: ACTIVE.**

### `tq_mq_aos.py` — TQ multi-query verify kernel (opt-in)
- **Purpose**: MQ spec-verify decode attention on the wheel's AoS layout
  (+21 % agg8, experimental).
- **Knob**: `VLLM_ARC_TQ_VERIFY_MODE=mq` (opt-in).
- **Evidence**: `results/2026-11-04-p619-tq-fix/`.
- **Cert**: cert8 (opt-in lane). **Status: ACTIVE, opt-in.**

### `patches/lmcint/` — LMCache integration kit (4 files)
- `lmc_kvfmt_detector.py` — KV-format detection (BS floor 16, cert17
  hardening: small-block engines take the byte-true TWO_HS store path).
- `lmc_twohs_transfer.py` — two-head-size KV transfer primitives.
- `vllm_v1_adapter.py` — v1 adapter (circular-import-safe lazy import).
- `utils.py` — shared helpers.
- **Provenance**: donor components (line D; `external_patches/lmcache/`).
- **Evidence**: `results/2026-10-30-p3-*`, `results/2026-11-06-p628-gemma4/`
  (KVGATE hardening).
- **Status: ACTIVE on `ARC_LMC=1` postures.**

## Layer 3 — image overlay patches (`external_patches/vllm/`)

### `patch_arc_xpu_attn_launcher.py` — B70 prefill launch-config table (cert12)
- **Purpose**: head-256/512 keyed launch-config points in the triton
  unified-attention launcher (upstream PR#60418 pattern; head-512
  M64/T32/w8 = 5.53× stock at the engine chunk band).
- **Evidence**: `results/2026-11-06-p628-gemma4/` (r108k sweep).
- **Cert**: cert12+. **Status: ACTIVE in every image since.**

### `patch_arc_xpu_warmup_gate.py` — engine warmup gate (cert13/cert16)
- **Purpose**: rank-serialized pre-load of the table points (kills the
  first-load SpirivUtils wedge); v4 passes softcap as int 0 so warm
  kernels live in the engine's triton cache-key space (the r108v
  cache-key split fix — 32/32 warm gate).
- **Evidence**: `results/2026-11-06-p628-gemma4/`.
- **Cert**: cert16+. **Status: ACTIVE.**

### `patch_arc_xpu_fa2_route.py` — full-attention FA2 dispatch (cert18)
- **Purpose**: head-512 full-attention calls (gemma-4's 10 global layers)
  dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel — 34× at k32k
  fp8-KV production geometry. `ARC_ATTN_FA2=0` disables.
- **Evidence**: `results/2026-11-06-p628-gemma4/phaseC/` (r108ac/ad/ae).
- **Cert**: **cert18**. **Status: ACTIVE.**

### `patch_arc_xpu_fa2_slide_route.py` — sliding-window FA2 dispatch (cert19)
- **Purpose**: sliding-window head-256 calls (gemma-4's 50 local layers)
  dispatch WINDOWED to the same FA2 kernel — 18–25× prefill, 11–15×
  decode over the triton path; asserts cert18 v3 bytes.
- **Evidence**: `results/2026-11-07-p629/` (r109a–r111).
- **Cert**: **cert19**. **Status: ACTIVE.**

### `patch_arc_xpu_attn_trace.py` / `patch_arc_xpu_jitkey_trace.py`
- Diagnostic probe patches (attention call census; triton JIT key trace).
  Never in a certified posture. **Status: tooling.**

## Layer 4 — custom wheels and donors

- **`vllm-xpu-kernels` arc4/arc5/arc6/arc7mx wheels**
  (`external_patches/vllm-xpu-kernels/` build kits): the W4A16 op (arc4,
  cert20), the W8A16-mxfp8 op (arc5, cert21), the skinny-M family (arc6,
  cert22 OFF), and the **arc7mx build28 wheel**
  (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
  `e6d8d337e43b907504537061414128f8`, baked in cert28): the MX-native
  paged-decode kernel (`csrc/xpu/attn/xe_2/mx/` — the vendor-class
  CUTLASS/Xe2 walk with MX dequant fused at the load path) plus the
  r231d seqused-tightened split clamp
  (`seq_num_kv_splits = min(num_kv_splits, max(1, windowed_k_blocks/4))`,
  mirrored in `mx_paged_decode_kernel.hpp` + `mx_reduce_split.hpp`).
  Installed under the real package name; pinned in the image manifest +
  `external_patches/wheels.pins.md5`; `.so` md5s asserted in-image.
  Kit `arc7_mx_attn/`: rig-side patcher on the certified arc6 base —
  exact anchors, refuses on miss/double-apply/drift.
- **`arc7_jm_gemm` kit (P6.34, research-only)**: the joint-matrix DPAS
  skinny-GEMM redesign (dpas.hf.hf.8.8 fragment geometry, packed pair
  decoders, group-scale accumulator flush). The offline wins (the
  fragment-op serialization + SLM-staging laws) never survived e2e
  certification — lane closed research-only, NOT in the certified
  wheel lineage (the arc7mx base assert requires jm-absent).
  Evidence: `results/2026-11-09-p634-jm-ladder/`.
- **oneCCL 2022.0.0 fork rebuild** (`external_patches/oneccl/`): the
  cert10 quantized-AR layer — same-version ABI-identical rebuild + the
  e4m3-wire ring patch (u32-packed stores, branchless converters,
  min-count 2M elems; `CCL_SYCL_ALLREDUCE_FP8=1` opt-in env). 1.65× at
  the dominant AR opt-in. Swap law: same-version rebuild only
  (oneccl-2022-battlemage-ar-cliff memory).
- **lmcache donor tree** (`external_patches/lmcache/`): the certified
  donor lineages + patch table for the Layer-2 glue ports (fork law:
  files/patches with provenance, never git parents).
- **`vllm_gaudi/xpu/esimd_fa/`** — ESIMD dense-FA v2 research kernel
  (P6.21c): numerics-green at 0.33× stock; ceiling-bounded ~0.83×.
  **Status: research kit, not routed.**
- **`vllm_gaudi/xpu/platform.py`** — the platform itself: patch
  registration (layers 1–2), the free-memory-report fix (early-birth
  reporting exactly 0 pre-import), BMG device gate (0xE200), GMU-relax
  (`VLLM_ARC_GMU_RELAX`), XPU-graph boot gate. In-house, every cert.

## Serve-script layer (`tools/serve_arc_b70.sh`)

Not an engine patch — the audited single entry point that assembles the
certified posture: container boot block (validated r42/r46 bytes; close-out
F 2026-11-10: `--shm-size 16g` + 16 GB LMCache CPU tier for 64 GB-RAM
hosts), the 25-var env chain, and the vLLM serve line (TP2, MNBT 8192,
`--kv-cache-dtype ${ARC_KVDTYPE:-fp8_e4m3}` — the MX KV lane rides the
same knob: `mxfp4` / `mxfp8` / `mxk8v4` postures with the LMC auto-off
law and, for the qwen MX postures, `ARC_MXKV_NATIVE=1`; speculative
config, per-model posture knobs). Every knob documented in
[`CONFIGURATION-arc.md`](CONFIGURATION-arc.md).
