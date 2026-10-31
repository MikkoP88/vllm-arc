# external_patches/lmcache — lmcache-arc-0.5.6+arc1 (P6.2 / p62; base moved
off 0.5.5+arc1 = P3 / p40)

Pinned base + patches + build law for the internal lmcache wheel. Fork law:
everything enters as FILES with provenance; no donor is a git source.

**Framing (plan §6): C-base + D-deltas.** The base is
line C's (0.5.5, closest to upstream, better warm-hit); the delta slots
e01–e04 are line D's improvements rebased onto that base — the plan's own
superiority move over either parent. The 2026-10-28 dual-donor check
(r30-prep) verified the two lines CONVERGED on the same delta generation,
so the rebased materialization is file-identical for e02/e03 either way:

| check | C champion (llm-scaler-exp:v1.2.35) | D champion (vllm-iexperimental:v1.0.0 = v1.2.0, trees identical) |
|---|---|---|
| lmcache base | **0.5.5**, RECORD drift = 3 files | **0.4.7**, RECORD drift = 2 files |
| lmcache-side glue generation | v136 mamba (`mamba_engine`, `mamba_pool_ratio`, `m-*` deepest-PRESENT lookup) + v141 unpin ledger | **same markers present** (v136/v141 backported into 0.4.7) |
| vllm-side `mamba_utils.py` | 0.30, md5 `1a6ae0ec…` — v136 glue + v135 fp16 repair + v138c sanitize | 0.26, md5 `24e734b0…` — same v136 glue, D's "0.26 re-anchor" (fused-kernel snapshot strategy) |
| XPU ops backend | upstream 0.5.5 `csrc/sycl` build + `torch_ops.py` ptr path | prebuilt 0.4.x `xpu_ops.so` binary (superseded) |

D is therefore NOT an alternative implementation source — it is the same
implementation one base-generation older (binary XPU ops, vllm-0.26 glue).
Its contributions to our stack: the certified env set + kv-transfer form
(serve line), the p41 re-anchor STRATEGY (below), and the REVAL ledger of
9 runtime patches (below).

## P6.2 retarget to 0.5.6 (2026-10-30 — owner directive "lmcache-0.5.6
implementation", plan §12 item 7 / §12.6 backlog ACTIVATED)

The wheel base moved 0.5.5 → **upstream 0.5.6** (owner P6 directive). The
0.5.5 sections below stand as the provenance record of the arc1 graft
lineage; this section is the CURRENT truth for what the kit builds.

**0.5.6 platform restructure (why e01 changed shape):** the flat
`v1/platform/torch_ops.py` is gone — `v1/platform/torch_ops/` package
(`mem_kernels`, `mem_alloc`, `mp_mem_kernels`, `_tensor_from_ptr`) +
`v1/platform/devices/{cpu,cuda,xpu,musa,…}/device_ops.py` +
`base/device_ops.py`; `setup.py` became policy-driven
(`setup_extensions/build_profiles/sycl.py`, same `BUILD_WITH_SYCL` env).
The SYCL `xpu_ops` module still exports the transfer kernels +
`xpu_host_register` + `alloc_pinned_ptr` (no `tensor_from_ptr` — the
base raises by default), so XPU python pointer-wrap still needs e01.

**Current patch set** (all `patches/`; apply order = glob order, each on
pristine 0.5.6 — verified 4/4 clean + py_compile on the rig 2026-10-30):

| patch | replaces / lineage | content |
|---|---|---|
| `e01-xpu-platform-ops.patch` | `e01-xpu-torch-ops.patch` (deleted) | TWO parts on the new module layout: (i) `mem_alloc.py` honors `LMCACHE_DISABLE_PINNED_MEMORY` in the `_use_pinned` gate (+INFO when disabled); (ii) `devices/xpu/device_ops.py` `ensure_native` — **absorbs p40c + p40d wheel-side**: unconditional re-pin of the 5 KV-transfer ops (`multi_layer_kv_transfer{,_unilateral}`, `single_layer_kv_transfer{,_sgl}`, `multi_layer_block_kv_transfer`) to the fully-vectorized torch baseline (native SYCL kernels are MLA-shaped; GQA raises — p40d law), env-gated re-pin of `alloc_pinned_ptr`/`free_pinned_ptr` (p40c: native = `sycl::malloc_host`, the >16–32 GiB/proc segfault class) + the `tensor_from_ptr` signed-wrap override (ex-e01 `torch_ops.py` ptr path, unchanged semantics). The sitecustomize wrap `lmc_xpu_pinned_env.py` is now DEAD (its target module `lmcache.v1.platform.xpu.device_ops` no longer exists) — file deleted, registration removed from `platform.py` `_WHEEL_HOOKS`. |
| `e02e03-cache-engine-c-delta.patch` | unchanged | applies clean on 0.5.6 (cache_engine.py byte-identical between 0.5.5/0.5.6; identity-guarded). |
| `e-adapter-vllm-056-graft.patch` | `e-adapter-vllm-030-graft.patch` (deleted) | the arc1 adapter graft rebased onto the 0.5.6 adapter. Method (graft-truth law): start from **pristine 0.5.6 adapter**, apply the arc1 graft's semantic set (sync lookup replacing upstream async `lookup_cache`, v136 joint clamp, one-step store lag `_lmc_pending_stores`/`_lmc_flush_pending`, mamba page maps, v133 `get_finished`/`get_block_ids`, deleted `_probe_decoder_cache`/`shutdown`), then accept upstream's 0.5.6 delta where it does not collide: `lookup_unpin` after non-async retrieve, `mock_req` guard, degraded-mode `lookup_client is None` guard, %-style logging, layerwise `Retrieved` INFO line. Divergence from upstream 0.5.6 = 3615 diff lines (163,822 bytes). |
| `e04-usage-tlm-gate.patch` | NEW (P6.3 telemetry law) | `observability.py:1965` `ContinuousUsageContext.GetOrCreate` runs even when usage tracking is disabled — LMCache telemetry is DEFAULT-ON (`usage_telemetry/identity.py:25`; opt-outs `LMCACHE_TRACK_USAGE=false` / `DO_NOT_TRACK=1`), and the ungated per-request GetOrCreate with dual-engine metadata is the source of the fork-log `different metadata` ERROR ×2 per boot. Patch gates creation+use behind `is_usage_tracking_enabled()`; the certified chain ALSO sets `LMCACHE_TRACK_USAGE=false` (docs + serve_arc_b70.sh). |

**Kit survival audit (r43 wraps, 2026-10-30):** `lmc_twohs_transfer.py`
(0.5.6 bilateral signature byte-identical; the NEW 0.5.6
`has_interleaved_kv_blocks` branch is (NB,2,BS,HS) block-interleave ≠ our
per-token (NB,BS,2,HS) — wrap stays NECESSARY) and `lmc_kvfmt_detector.py`
(`VLLM_Detector.discover` still present) survive unchanged;
`vllm_v1_adapter.py` (lmcint live adapter) rebased with the same 4 upstream
guards as the graft. Composition order verified: the sitecustomize hook
fires at `torch_ops` first import → e01's `getattr(torch_ops, op)` at
`ensure_native` binds the WRAPPED function.

**Identity reference:** OFFICIAL lmcache 0.5.6 cp312 wheel
(md5 `1f96765db1d7b02ecdd3852794e400a1`, tree at rig /root/build/p56/wheel_tree)
— all 5 patched files verified byte-identical sdist-vs-wheel; the build
guard md5s every one of the 5 against it before patching.

## Base pin (0.5.6 current; 0.5.5 historical)

| artifact | md5 | source |
|---|---|---|
| `lmcache-0.5.5.tar.gz` (sdist) | see `wheels.pins.md5` entry added at build time | PyPI (rig /root/build/p40/) |
| `lmcache-0.5.5-cp312-...whl` (upstream CUDA wheel, reference only) | `ba1aa37fcbcdff058780431659fb3699` | PyPI |

Base = upstream lmcache **0.5.5** (commit g05a013b2) — the version the C
champion (llm-scaler-exp:v1.2.35 image) runs. Verified 2026-10-28 (r30-prep
recon): the C image's installed tree hashes CLEAN against its wheel RECORD
for every file except exactly three (below) — i.e. C's certified lmcache =
pristine 0.5.5 + 3-file delta + env set; C's caching strength lives in the
env set + vllm-side glue, not in a forked lmcache.

## Patches — HISTORICAL 0.5.5 set (superseded by the P6.2 table above;
e01-xpu-torch-ops.patch and e-adapter-vllm-030-graft.patch are DELETED)
(e01–e04: D-lineage deltas, rebased onto the C base — extracted
from the C image tree where the rebase already exists, provenance per file)

| patch | file | e-slot (plan §6) | provenance | content |
|---|---|---|---|---|
| `patches/e01-xpu-torch-ops.patch` | `lmcache/v1/platform/torch_ops.py` | e01 (D xpu_ops backend, rebased) | C image tree (md5 vs wheel diverges); D's 0.4.7-era backend superseded by upstream 0.5.5 `csrc/sycl` (same backend lineage, buildable — no binary vendoring) | `_tensor_from_xpu_ptr`: non-owning tensor from raw SYCL USM device pointer (unsigned→signed 64-bit wrap; `torch._C._construct_storage_from_data_pointer`); wires `device.type == "xpu"` into `tensor_from_ptr`. Python half of XPU enablement — the compiled half is the SYCL extension below. |
| `patches/e02e03-cache-engine-c-delta.patch` | `lmcache/v1/cache_engine.py` | e02 (D mamba-hybrid state) + e03 (D double-unpin) | C image tree (rebased form); markers verified IDENTICAL in D's 0.4.7 tree (v136 `m-*` + v141 ledger) | TWO deltas: (i) **e02 engine half** — `m-*` mamba boundary-state lookups (deepest PRESENT 512-token chunk end + single-object pin; stock consecutive-prefix semantics would return 0 forever for boundary-banked states); (ii) **e03 epoch-ledger double-unpin guard** — `_lookup_released` set prevents the async event-cleanup branch from re-releasing lookup-pin objects (pin_count −1 warnings). Worker-side half of e02 is p41 (vllm `mamba_utils` v136 port — `vllm_gaudi/xpu/patches/mamba_lmcache_glue.py`). |
| `patches/e-adapter-vllm-030-graft.patch` | `lmcache/integration/vllm/vllm_v1_adapter.py` | (e02/e03 carrier — the connector surface both halves hang off) | C image tree | Graft of the newer (lmcache-main, unreleased-at-0.5.5) vLLM v1 adapter: GPU connectors (`VLLMPagedMemGPUConnectorV2` etc.), lookup client, offload/ZMQ server, blend builder — the proven vllm-0.30 ↔ lmcache-0.5.5 connector compat surface. Hard-imports `_LMC_SEED_OVERRIDE` / `_LMC_BOUNDARY_REGISTRY` from `vllm.v1.worker.mamba_utils` → **p41 is a boot prerequisite**, not optional (without it the store path ImportErrors on the first finished request and the joint lookup clamps all hits to 0). |

The SYCL XPU ops backend (`lmcache.xpu_ops`, plan slot e01 compiled half)
needs NO patch: upstream 0.5.5 ships `csrc/sycl/*` + `SyclProfile`
(`BUILD_WITH_SYCL=1`, icpx, `torch.utils.cpp_extension.SyclExtension`) and
`v1/platform/xpu/device_ops.py` binds the built extension with a torch
fallback. We build it with upstream's own recipe.

## vllm-side connector glue (r30 leg, 2026-10-28 — sitecustomize patches,
`vllm_gaudi/xpu/patches/`, NOT wheel patches)

The r30 first leg exposed three vllm-side incompatibilities between
upstream-0.30 and the lmcache-0.5.5+arc1 stack; all cured with C's own
files/deltas as the byte-preserved reference (full evidence:
`results/2026-10-28-p3-r30-lmcache-first-leg/`):

| patch | target | provenance | content |
|---|---|---|---|
| `lmcache_hma_connector.py` (p41b) | `vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector` (exec_module, subclass-rebind) | C v1.2.35 file md5 `cee1a393…` (same reference D's resume13 byte-preserved; D adjudication: flag-flip REJECTED, HMA-on bar) | (a) SupportsHMA mixin + `request_finished_all_groups` (cures factory.py:58 HMA refusal); (b) v133 async-gate fix — real `load_kv_async` from `get_num_new_matched_tokens` + 3-arg `update_state_after_alloc` (hardcoded False made loads land a step late, "byte-perfect but useless"); (c) KV-events guard (0.5.5 has no get_kv_events). Knob `VLLM_ARC_LMC_HMA=0`. |
| `lmcint/` kit (p41b) | `…v1.lmcache_integration{,.vllm_v1_adapter,.utils}` (source-override exec at import) | C image files, md5 `1f3e97d3`/`dcb23fbe`/`b5300ee0` — C's 0.5.5-compat patch of vllm-0.30's vendored integration | C's `__init__` drops `multi_process_adapter` ("requires lmcache main (CudaIPCWrapper)… NOT importable here" — cures the boot ImportError); the vendored adapter is the LIVE engine under `use_native:true` (mamba-hybrid: `MambaPageConnector`, `_init_mamba_engine`, per-group block_ids, late-imports p41's `_LMC_*` at its lines 1741/2049). |
| `lmc_xpu_pinned_env.py` (p40c) — **DELETED at P6.2** | `lmcache.v1.platform.xpu.device_ops` (exec_module wrap) — **module deleted by the 0.5.6 restructure** | C serve_lmc_v133.sh law (pinned USM >16–32 GiB/pro process segfaults, probed 2026-10-08 → pools pageable) | our built `lmcache.xpu_ops` exports native `alloc_pinned_ptr`/`free_pinned_ptr` that `bind_native` layers OVER the env-honoring torch baseline (C's 0.4.7 binary lacked the pair); with `LMCACHE_DISABLE_PINNED_MEMORY=1`, drop the two instance bindings post-bind → pageable baseline serves them, all other native ops stay native. Cured the 100 GiB `sycl::malloc_host` boot death. **P6.2: absorbed wheel-side into e01-xpu-platform-ops.patch (env-gated re-pin in `ensure_native`); registration removed from platform.py `_WHEEL_HOOKS`.** |

Also discovered en route: lmcache-0.5.5 runtime deps `aiofile` + `redis`
missed by the p40 `--no-deps` set (upstream Requires-Dist; eager import in
`storage_backend/connector/fs_connector.py`) — installed in r30_boot.sh,
Dockerfile dep line gains both at next image build. And the env set needs
`PYTHONHASHSEED=0` (C CRITICAL: chunk keys chain from builtin hash;
without a fixed seed scheduler/workers derive different hashes → 100%
lookup misses despite clean stores) — already in r30_serve.sh line 8.

## r31 fork deltas on the C adapter (2026-10-29 — breaks byte-preservation,
provenance per site; live evidence `results/2026-10-29-p3-r31-mamba-live/`)

The C-adapter port proved the mamba half LIVE under our posture (p41c snaps
fire on the V2 model-states site; `LMC mamba store` ×2 ranks; joint lookups
`attn_hit>0 ∧ mamba_hit>0` → `LMCache hit tokens` > 0 for the first time).
Getting there required fixing a C-v136 arithmetic assumption our spec breaks:

- **The 256-byte floor bug family (4 sites, floor→ceil).** C v136's object
  sizing computes `page_size_bytes × num_layers // chunk_size`-family floors;
  our true mamba page is 858,112 B/layer (conv 71,680 + ssm 786,432 fp16),
  ×48 layers = 41,189,376 = 896×45,970 + **256** — the floor truncated the
  deepest layer's last 256 state bytes (store side logged
  `Mamba store seed size mismatch: 41189376 vs obj 41189120` ×56; the
  initial ceil edits at `get_shape`/`_init_mamba_engine` did NOT cure it
  because the LIVE site is the adapter's post-init "Mamba dtype page
  override" `metadata.kv_shape` rewrite — instrumentation showed the same
  connector id mutate psb 917,504→858,112 after init). Fixed at all four
  sites (`-(-a*b//c)`): `MambaPageConnector.get_shape`,
  `_init_mamba_engine` kv_shape, the post-init dtype-page-override
  `metadata.kv_shape` (live), scheduler-side mamba lookup-client metadata.
  Ceil is REQUIRED, not cosmetic: floor would corrupt the deepest layer's
  last 256 state bytes on every load.
- **Prefix-copy + zero-tail in `batched_from_gpu`.** Ceil-sized dst vs exact
  src: copy `dst[:src.numel()] = src`, zero the pad tail (guard still raises
  on true overflow `src > dst`).
- **Load-side seed slice.** Retrieved ceil-sized seeds (41,190,016) park
  sliced to exactly `page_size_bytes × num_layers` (41,189,376); undersized
  stays a HARD error (real truncation). Without this the engine 500s
  (`Mamba seed size 41190016 != expected 41189376`) on the first mamba load.
- Probe-gated r31 instrumentation (`ARC_SNAP_PROBE`): connector/engine init
  lines, `bfg` copy lines, snap/ncc prints — silent when the env is unset.

All four deltas are commented `arc r31 fork delta (2026-10-29)` in
`vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py` (deployed md5 lineage
diverges from C's `b5300ee0…` by exactly this set).

**r31 leg outcome** (full evidence + root-cause analysis:
`results/2026-10-29-p3-r31-mamba-live/VERDICT.md`): the mamba chain is LIVE
end-to-end (snaps→stores→m-keys→joint lookups→seed parks→seed writes), but
integrity FAILED (5×8k re-hit 0/5 sha, golden len=4096 drift). Both root
causes are now FIXED (r32, evidence
`results/2026-10-29-p3-r32-integrity-fixed/VERDICT.md`; integrity **5/5**):

- **#1 upstream `mamba_hybrid.py:121`** seeds the running-state column as
  `(num_computed−1)//cache_config.block_size` with the ATTENTION block size
  (64) while column units are mamba blocks (896) — a loaded resume lands at
  column 97 instead of 6 and the pre-copy carries a foreign page (p41c
  wrapper corrects the column for seed rows; probe print `p41c-sidx`).
- **#2 mid-chunk boundary crossings never materialize in the block table**
  (p41c snap defect, fixed 2026-10-29): a boundary crossed inside a chunk
  (8k prompts: the final `[7168,8192)` 1024-token chunk crosses 8064) is
  skipped by the pre-copy (`bt[7]→bt[9]` directly) and never migrated
  (postprocess fires only on chunk-end-aligned boundaries) — `bt[j]` is a
  stale/foreign page, and banking it as the boundary state poisoned every
  warm resume (the E3 "byte-exact roundtrip" was delivering garbage).
  Fix: the p41c snap banks ONLY exact chunk-end boundaries
  (`ncc % bs == 0`); the joint then books the true deepest banked boundary
  and the warm resume recomputes the identical chunk shape → bit-identical
  outputs.

Ruled out en route: fp16 seed-repair asymmetry (E1); seed byte accounting;
conv-ring `token_bias` rotation semantics (bias-0 copies are verbatim —
proven by kernel source + E4 md5s); attn external-load port (integrity 5/5
with attn loads live). Interim safe knob retained:
`VLLM_ARC_LMC_MAMBA_SNAP=0` (mamba keys never bank → joint 0 → APC-only
r30-equivalent posture).

**r32 clean-boot battery** (full evidence
`results/2026-10-29-p3-r32-integrity-fixed/`): ALL correctness gates green —
survival 200/200, spec pos1 0.872, golden 5/5 zero-drift vs the P2 product
ref (the len=4096 face is a context-sensitive knife-edge: matches the ref in
the certified gate order; flips deterministically to 99955622 after heavier
prior traffic or with the snap off — documented, not a defect), integrity
5/5. Warm-hit 1047.1 ms vs 750 bar FAIL → p43 lane with named mechanisms
(final-boundary-at-finish snap killing the −1024-token recompute, async
pinned snaps, seed-load overlap, store-lag-on-immediate-rehit).

## r33 attn-KV format fix (2026-10-29 — per-token K/V interleaved pool;
byte-true `NL_X_NB_BS_NH_TWO_HS` path; `vllm_gaudi/xpu/patches/lmcint/`,
NOT wheel patches; full evidence `results/2026-10-29-p3-r33-attn-kv-twohs/`)

The N2/N4-class warm re-hit integrity break root-caused to the attn pool's
GEOMETRY, not the mamba half (r31/r32): the FA-XPU pool is ONE stacked
tensor per layer `(NB, 2, BS, kvdim)` whose strides make the PHYSICAL order
`(NB, BS tokens, K/V, kvdim)` — per-token K/V interleaving; a K-plane
tensor view of it is impossible. lmcache could neither DETECT nor EXECUTE
that layout: the raw rank-4 pool lands on the fused-CS branch whose slot
model folds every page P onto P//2 (verified 8/8 aliases), and both XPU
transfer backends lack the interleaved branch — the native SYCL
`multi_layer_kv_transfer` implements 3 GQA formats + MLA only (CUDA has the
`NL_X_NB_BS_NH_TWO_HS` branch at mem_kernels.cu:310; the SYCL port never
got it — raises `Unsupported non-MLA`), and the torch fallback (the LIVE
path under p40d) folds it again via its generic `(2, page_buffer_size,
hidden)` half-plane view. Three kit files close the chain (all
sitecustomize source-overrides; installed tree untouched):

| patch | target | content |
|---|---|---|
| `lmcint/vllm_v1_adapter.py` `_arc_attn_pool_view` + seam transforms (r43) | every site handing attn pools to lmcache (register, load, store ×6) | contiguous ZERO-COPY 5-D view `pool.permute(0,2,1,3).reshape(NB, BS, 1, TWO, kvdim)` (permute of the non-contiguous pool IS contiguous in (NB,BS,K/V,kvdim) order; same storage/data_ptr). |
| `lmcint/lmc_kvfmt_detector.py` (r43) | `lmcache.v1.gpu_connector.kv_format.detectors.vllm` (exec_module wrap of `VLLM_Detector.discover`) | maps the 5-D view to `NL_X_NB_BS_NH_TWO_HS` (NH=1, HS=kvdim) — get_block_size=BS, num_blocks=NB correct; canonicalizes the singleton axis wherever the contiguity recovery's stride sort puts it. |
| `lmcint/lmc_twohs_transfer.py` (r43) | `lmcache.v1.platform.torch_ops` (exec_module wrap of `multi_layer_kv_transfer`) | TWO_HS-only branch: pool viewed physically `(NB, BS, 2, HS)`, vectorized gather/scatter per layer — `D2H: kv[:, l, valid, :] = pool[bidx, boff, :, :].T`; H2D mirror — byte-true per token, one slot per token, both K and V per slot. All other formats pass through to the original (p40d baseline posture preserved). |

Result: N4 8k cold+3 warms ALL `9c1623f0d38e28e9` with `Retrieved 7168`×6
(real loads), 0 store failures. REVAL adds: native SYCL TWO_HS branch
(CUDA parity) = perf-only follow-up — the torch-baseline branch is C's
posture and is correct.

## Build law

`build_lmcache_arc.sh` (rig host): fresh sdist extract → md5 guard vs the
upstream wheel tree → apply patches → version-tag `0.5.5+arc1` → build in
the persistent omix builder (`intel/omix:0.2.0-devel-ubuntu24.04`, oneAPI
2026.0.0 = the arc venv runtime generation — the vxk arc3 law; torch
2.13.0+xpu venv) via `BUILD_WITH_SYCL=1 setup.py bdist_wheel` →
`auditwheel repair` (upstream exclude set: torch + oneAPI/SYCL runtime libs
bind to the host image) → SYCL offload-bundle check → wheel to
`/root/build/wheels-arc-lmc/`, md5 recorded into `wheels.pins.md5`.

Inner script: `lmc_arc1_inner.sh` (copied to the rig at
/root/build/, driven by the outer script). Never patched in-place at boot.

## REVAL ledger (plan §6 disposition — decisions recorded with gate evidence)

- `unpin` (D cache_engine v141) → **e03 present in the 0.5.5 delta** (both
  donor trees carry the identical v141 guard); applied with e02e03 patch.
- `extyield2` → p42 (vllm-side lookup-elide knob), NOT an lmcache patch.
- `extyield`, `flushmove` → DROP (superseded / falsified, plan law).
- `kvev`, `xfer`, `btg`, `seedfix-v5`, `storededup` → REVAL on 0.5.5 at P3:
  port as e05+ ONLY if 0.5.5 lacks the fix AND the P3 gates expose the
  defect it fixed. Status after r30 (2026-10-28): store/load/integrity all
  PROVEN clean (golden zero-drift, 5/5 byte-identical re-hits, warm
  755 ms vs cold 5556 ms) — none of the five defects surfaced at the
  gate level; keep pending p43 (warm-hit closure may stress the store
  path harder). New telemetry from r30: ~80 "Lagged attn store failed
  (cached chunk absent; later lookups recompute)" events ×2 ranks during
  the agg phases — C's graceful-degrade path, non-fatal; re-examine
  after p43 changes store timing (candidate `storededup` trigger).
- usage telemetry (plan e04 slot) → 0.5.5 ships `LMCStatsMonitor` +
  `InitializeUsageContext` + observability.py natively. r30 finding: the
  engine-stat "External prefix cache hit rate" counter reads 0.0% even
  while sha-verified external hits land (7.4× warm speedup) — NOT usable
  as gate evidence; gate truth stays sha-identity + warm/cold timing.
  LMC INFO lines (hybrid-mode init, hash init) do log and are banked.
- **v138c sanitize-carry** (D + C both carry it, env-gated default ON in
  both donors; NaN/Inf→0 healing on FALLBACK-path carried pages) → REVAL:
  our live posture is align-mode FUSED copies, which no donor hooks it
  into; enable only on LMCBOUND telemetry nf>0 evidence (the p41 snap
  logs a NONFINITE warning as the trigger).
- **v135 fp16 seed repair** → carried IN p41 (both donors run it; our
  `--mamba-ssm-cache-dtype float16` makes it live). Disable knob:
  `VLLM_MAMBA_FP16_SEED_REPAIR=0`.
- W14's 0.4.7-identity tree (with its own compiled `xpu_ops.so`) is NOT our
  base — superseded by building upstream's own SYCL backend generation-
  matched (no binary vendoring).

## Certified env set (p40 serve config — D champion serve line verbatim)

```
PYTHONHASHSEED=0  # C CRITICAL law (serve_lmc_v133.sh): chunk keys chain
                  # from builtin hash; unset seed -> per-process hashes
                  # -> 100% lookup misses despite clean stores
LMCACHE_USE_EXPERIMENTAL_API=True LMCACHE_MAX_LOCAL_CPU_SIZE=100
LMCACHE_LOCAL_CPU=True LMCACHE_EVICT_POLICY=lr_u LMCACHE_CHUNK_SIZE=1024
LMCACHE_SAVE_DECODE_CACHE=True LMCACHE_MIN_RETRIEVE_TOKENS=1024
LMCACHE_LOG_LEVEL=INFO LMCACHE_DISABLE_PINNED_MEMORY=1
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both",
  "kv_connector_extra_config":{"use_native":true,"lmcache.extra_config":
  {"mamba_pool_ratio":2}},"engine_id":"arc-p40"}'
```
(`mamba_pool_ratio:2` = the D-certified value; note the §6 vxk exposure note
— rows past 4096 — already covered by our arc3 int64 wheel. First leg runs
this form verbatim; if the mamba half needs p41 glue not yet present, the
fallback is the KV-only form and the decision is recorded in the leg
VERDICT.)
