# vllm — B70 prefill launch-config table + engine warmup gate for the triton unified-attention launcher

Kit of two wheel patches (both mark-tagged, idempotent, `--audit` mode,
`.arcxpubak` backup, stock wheel files — no source rebuild, staged pins
stay pristine):

- **patch #1 `patch_arc_xpu_attn_launcher.py`** → cert12: the per-platform
  prefill launch-config table (this file's original content).
- **patch #2 `patch_arc_xpu_warmup_gate.py`** → cert14: rank-serialized
  boot warmup for every kernel point the table can produce, replicating
  the engine's exact call signature — fixes the engine wedge the table
  exposed (below). cert13 carried the same patch in a first revision
  whose warm called a variant the engine never produces; it is
  interim/superseded (manifest chain cert12 → cert14).
- **patch #3 `patch_arc_xpu_attn_trace.py`** → cert14dbg (DIAGNOSTIC
  overlay, never a cert pin): env-gated per-call launch-key trace for
  the wrapper — every triton cache-key-relevant value plus the
  JITFunction object id, logged before each kernel launch. Exists
  because cert14's warm STILL left one fresh compile at the first
  request (r108r) while a static diff of the wrapper's full launch
  list finds no differing dimension; the trace settles it empirically
  (warm key-set and engine request-time key land in one serve log).

## Patch 1 — launch-config table (cert12)

### What

`patch_arc_xpu_attn_launcher.py` inserts a per-platform prefill
launch-config table into the vllm wheel's
`vllm/v1/attention/ops/triton_unified_attention.py` (stock wheel file —
no source rebuild, staged pins stay pristine). Three insertions, mark
`[ARC-P628-XPU-TUNE]`, backup `.arcxpubak`, idempotent, `--audit` mode:

1. module table + band selector after `float8_info`
   (`_ARC_XPU_PREFILL_CONFIGS` keyed by padded head size →
   (≤64, 65–4095, ≥4096) bands → `(BLOCK_M, TILE, warps, stages)` or None)
2. launcher branch after the `tuned_large_head` BLOCK_M block and BEFORE
   `total_num_q_blocks` (the 2D grid upper bound shrinks with BLOCK_Q),
   guarded `current_platform.is_xpu()` + `num_queries_per_kv <= 16` +
   `max_seqlen_q > 64` + `not is_batch_invariant` + `q.element_size() <= 2`
3. tile override after the `tuned_large_head` TILE block:
   `TILE_SIZE_PREFILL = max(32, tile)` (fp8-KV legality, PR#60418 guard;
   the stock `use_td` clamp still applies downstream)

Decode / 3D split-KV path untouched (band guard keeps every
decode-shaped call on stock configs). Only launch parameters change —
kernel source and numerics untouched (r108k numeric dumps allclose(1e-2)
vs stock).

B70 table (rig sweep r108k, 2026-11-06):

| padded head | 65–4095 band | ≥4096 band |
|---|---|---|
| 256 | (64, 64, 8, None) | (64, 64, 8, None) |
| 512 | (64, 32, 8, None) | (64, 32, 8, None) |

≤64 band = None (stock) for both. `num_stages=None` = triton default,
exactly what the sweep measured. TILE=64 at head 512 is infeasible on
B70 (all T64@512 sweep points die pre-print — SM-side memory wall, the
same LDS boundary class vllm#52585 documented for TILE=128 on gfx1100).

## Patch 2 — engine warmup gate (cert14)

### What

`patch_arc_xpu_warmup_gate.py` patches
`vllm/v1/worker/gpu_worker.py` with two insertions, mark
`[ARC-P628-XPU-WARMUP]`:

1. two module-level helpers after the `kernel_warmup` import
   (`_arc_xpu_warm_tuned_prefill_kernels`: table presence check, model
   geometry walk, TP rank serialization; `_arc_xpu_warm_points`: the
   dummy-tensor load calls)
2. one guarded call at the top of the stock warmup section of
   `Worker.compile_or_warm_up_model`, immediately before the wheel's own
   `kernel_warmup(self)` ("Warmup and tune the kernels used during model
   execution before cuda graph capture") — i.e. before cuda-graph
   capture and before any dummy run can reach a tuned config.

The warmup loads one kernel module per (geometry, band, divisibility,
window-style) combination a serving step can produce, **replicating
`TritonAttentionImpl.forward`'s call signature exactly** (the r108q
lesson — a warm that deviates leaves a fresh specialization for the
first request to load, and that load wedges):

- `softmax_segm_output` `(16, nqh, 16, head_pow2)` fp32 and
  `softmax_segm_max`/`softmax_segm_expsum` `(16, nqh, 16)` fp32 — the
  backend allocates these unconditionally and passes them on every
  call (never None); `seq_threshold_3D=16`,
  `num_par_softmax_segments=16` = `triton_attn.NUM_PAR_SOFTMAX_SEGMENTS`;
- `use_td=True` — the XPU default of the tri-state
  (`envs.VLLM_TRITON_USE_TD` unset);
- `k_descale`/`v_descale` = `ones(nkvh).expand((1, nkvh))` — the
  engine's stride-0 expanded views (fp8-KV per-tensor:
  `kv_quant_mode=1`, `q_descale=None`);
- `mm_prefix_clamp_sliding_window` paired with the window style —
  sliding layers `(1023, 0)` + clamp=True, full-attention layers
  `(-1, -1)` + clamp=False (Gemma-4 `is_sliding`);
- `softcap=0.0` (Gemma-4 ships no `attn_logit_softcapping` — the
  config's `final_logit_softcapping` is the logits head only).

Geometries are collected from the loaded model's attention modules
(head sizes covered by the table only — no model-name hardcoding);
points are the max_num_batched_tokens chunk (high band, div16) and the
98-class chunk remainder (mid band, non-div16), each against div16 and
non-div16 KV lengths. Loads run rank-by-rank behind
`torch.distributed.barrier()` so no two TP ranks module-create
concurrently; the helper frees its tensors and is fail-loud (a failed
warmup aborts boot rather than serving an engine whose tuned kernels
would wedge on first prefill). No-op when the table is absent (stock
wheel) or the model has no covered geometry.

### Why

cert12's table makes each tuned config a kernel specialization the
engine has never loaded. r108o (py-spy, banked
`results/2026-11-06-p628-gemma4/phaseC/hang_capture_c12/`) proved the
first in-engine module creation of such a config
(`CompiledKernel._init_handles` → `SpirvUtils.load_binary` →
`zeModuleCreate` + IGC compile) wedges BOTH TP ranks — identical stacks,
GIL held, ~166% CPU each, indefinitely:

- certified graph posture: wedge at the first real prefill request
  (98-token warmup) → engine rpc watchdog kills the step — the P6.28
  "engine freeze" G2 row;
- default eager posture: wedge inside `compile_or_warm_up_model`'s own
  dummy run ("Compile and warming up model for size 8192") — the engine
  never reaches health.

Single-process loads of the exact same binaries are clean (r108n micro
at the exact shapes; r108p micro for the concurrency split), and vLLM's
own jit-monitor names the fix class when it fires at request time:
*"consider extending warmup to cover this shape/config"*
(`vllm/utils/jit_monitor.py`). Moving every first-load into
`compile_or_warm_up_model` is precisely that.

cert13 proved the warm mechanics in-engine — boot reached health, the
rank-serialized warm ran to completion on both ranks (214.9 s) — yet
the first 98-token request STILL wedged (r108q, banked
`phaseC/cert13_boot/`): the jit-monitor re-fired at request time and
py-spy showed the identical `SpirvUtils.load_binary` frame, because the
warm had passed `None` segment tensors, `use_td` unset and plain
descales — each a distinct triton specialization from what the engine
actually calls. The wedge requires only that the request-time load be
FRESH; the warm must be engine-exact, not merely shape-covering
(unified theory across r108o/p/q: in the oneCCL-resident engine, two
ranks doing fresh module-create concurrently wedge; solo fresh loads
and cache-hit loads are clean).

### Provenance

Placement follows the wheel's own structure (the stock
`kernel_warmup(self)` section exists for exactly this purpose and the
jit-monitor expects warmup coverage after
`compile_or_warm_up_model`); rank serialization mirrors vLLM's
existing rank-ordered boot collectives. No upstream file exists with
this exact logic because no upstream platform ships the tuned table —
the two patches are one fix: a tuning table is only servable if the
engine pre-loads what it adds.

## Patch 1 — why the table exists

P6.28 gemma-4 serving "froze" on any real prompt. Root cause (r108j
bare-kernel discriminator, banked micro_disc/): NOT a deadlock — the 2D
prefill kernel runs at ~0.3% of achievable peak (1.1 TFLOP in 8.265 s
steady vs 43.2 TFLOPS fp16 matmul reference on the same device) at the
stock launch config (BLOCK_M=16 / TILE=32 / 4 warps). Gemma-4's 60-layer
5:1 mix of head-512 full / head-256 sliding prefill burns ~20 min device
time per 32k-token face → the 300 s `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
rpc kill reads as a hang. Exonerated along the way: tensor-parallel
deadlock, fp8-KV path, device asymmetry, block_table ordering, tile
pruning (already present in the wheel's helpers).

Headline sweep numbers (per-rank GQA 16Q:8KV, fp8-KV, seqlen 8192):
head-512 ntok=1024: stock 2.002 s → (64, 32, w8) **0.428 s = 4.67×**
(warps=8 is load-bearing: M64w4 = 2.02× only); head-256 sliding ntok=1024:
stock 0.0461 s → (64, 64, w8) **0.0222 s = 2.08×**.

## Provenance / upstream lineage (patch 1)

The fix pattern is upstream's own, applied to the platform upstream
doesn't tune yet:

- **vLLM PR#60418** `[ROCm][RDNA][Perf] Add tuned configs for TRITON_ATTN
  kernel` (merged 2026-11-03, 4d2fa5637): module-level per-platform
  config tables in this exact launcher, keyed by padded head size ×
  max_seqlen_q band, prefill branch before grid computation, decode
  branch after 2D/3D choice, "only launch parameters change".
- **vLLM#52585** (gfx1100, 2026-08-17): the same defect class — stock
  BLOCK_M=16 under-tiles prefill ~2×; tuned BLOCK_M/warps = 1.1–1.3×
  TTFT bitwise-identical; TILE=128 blows the LDS budget.
- **vLLM#38887**: Gemma-4 slow prefill on NVIDIA — still open, no fix;
  this is the first landed fix in this defect line, on XPU.
- **intel-xpu-backend-for-triton#8090**: Intel's offline-autotune
  direction — swept configs into static dicts + heuristics (citing
  #60418), explicitly NOT `@triton.autotune` (Triton maintainers: not
  production-intended).
- **vllm.ai Triton backend deep dive** (2026-03-04, IBM/Red Hat/AMD — the
  kernel's authors): "autotuning selects appropriate block sizes for each
  platform"; KV-side tiles are page-size-constrained so "optimization
  focuses on the query side" — i.e. a per-platform Q-block launch-config
  table IS the designed tuning surface of this kernel.

Scope law (plan §3/App I): generic shared-component fix (whole launcher
file, platform-gated, model-agnostic config table), no donor code —
pattern is upstream vLLM's, values are our own rig measurements.

## Validation (§10 gates before cert14 pin)

**Patch 1 (cert12, all green r108l):**

1. patcher: `--audit` PATCHABLE → patch → `ALREADY_PATCHED` on re-run →
   `py_compile` clean (local, pristine wheel file extracted from cert11).
2. rig in-container: patched launcher imports, MARK present, table
   entries readable; bare-kernel full-shape (ntok=8192) timing matches
   sweep winners (grid now shrinks with BLOCK_Q — same or better).
3. numeric sanity: same-seed output dumps vs stock — bitwise identical
   at head-512 (launch params only), max_abs 4.9e-4 allclose(1e-2) at
   head-256 (tile change); per-step prefill time @kv16k 24.85 s → 4.51 s
   = 5.51× (freeze budget math: worst chunk-step ~53 s at kv16k, ~98 s
   at kv32k — no timeout knob needed).

**Patch 2 (cert13 interim → cert14):**

1. patcher: `--audit` → patch → idempotent re-run → `py_compile` clean
   (local, pristine cert12 wheel file; cert14 gate additionally asserts
   the engine-exact warm signature — segments, use_td, clamp pairing).
2. rig engine boot, certified graph posture: warmup log lines present
   ("ARC XPU tune warmup: load …", "… complete on rank 0/2"), boot
   reaches health, the 98-token first request passes (the r108o g1
   wedge repro — cert13's remaining failure), jit-monitor silent on
   unified_attention during G1/G2 traffic.
3. eager-posture boot (ARC_GRAPH=0) reaches health (the r108o e0
   counterfactual).
4. engine G2: both gemma legs re-measured vs the fp8 incumbent row
   (17.715 s e2e / 39.025 s 32k prefill face @ 53.5 ms decode); decode
   unchanged (kernel/config path untouched for max_seqlen_q ≤ 64).

Evidence: `results/2026-11-06-p628-gemma4/` (r108j micro_disc/, r108k
tile_sweep/, r108o hang_capture_c12/, r108p micro_concurrency/, r108q
cert13_boot/, G2 legs, write-up).
