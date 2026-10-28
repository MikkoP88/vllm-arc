# external_patches/lmcache — lmcache-arc-0.5.5+arc1 (P3 / p40)

Pinned base + patches + build law for the internal lmcache wheel. Fork law:
everything enters as FILES with provenance; no donor is a git source.

**Framing (plan §6, owner-confirmed): C-base + D-deltas.** The base is
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

## Base pin

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

## Patches (e01–e04: D-lineage deltas, rebased onto the C base — extracted
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
| `lmc_xpu_pinned_env.py` (p40c) | `lmcache.v1.platform.xpu.device_ops` (exec_module wrap) | C serve_lmc_v133.sh law (pinned USM >16–32 GiB/pro process segfaults, probed 2026-10-08 → pools pageable) | our built `lmcache.xpu_ops` exports native `alloc_pinned_ptr`/`free_pinned_ptr` that `bind_native` layers OVER the env-honoring torch baseline (C's 0.4.7 binary lacked the pair); with `LMCACHE_DISABLE_PINNED_MEMORY=1`, drop the two instance bindings post-bind → pageable baseline serves them, all other native ops stay native. Cured the 100 GiB `sycl::malloc_host` boot death. |

Also discovered en route: lmcache-0.5.5 runtime deps `aiofile` + `redis`
missed by the p40 `--no-deps` set (upstream Requires-Dist; eager import in
`storage_backend/connector/fs_connector.py`) — installed in r30_boot.sh,
Dockerfile dep line gains both at next image build. And the env set needs
`PYTHONHASHSEED=0` (C CRITICAL: chunk keys chain from builtin hash;
without a fixed seed scheduler/workers derive different hashes → 100%
lookup misses despite clean stores) — already in r30_serve.sh line 8.

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
