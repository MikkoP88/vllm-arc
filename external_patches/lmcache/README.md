# external_patches/lmcache — lmcache-arc-0.5.5+arc1 (P3 / p40)

Pinned base + patches + build law for the internal lmcache wheel. Fork law:
everything enters as FILES with provenance; no donor is a git source.

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

## Patches (all extracted from the C image tree, provenance per file)

| patch | file | provenance | content |
|---|---|---|---|
| `patches/e01-xpu-torch-ops.patch` | `lmcache/v1/platform/torch_ops.py` | C image tree (md5 vs wheel diverges) | `_tensor_from_xpu_ptr`: non-owning tensor from raw SYCL USM device pointer (unsigned→signed 64-bit wrap; `torch._C._construct_storage_from_data_pointer`); wires `device.type == "xpu"` into `tensor_from_ptr`. Python half of XPU enablement — the compiled half is the SYCL extension below. |
| `patches/e02e03-cache-engine-c-delta.patch` | `lmcache/v1/cache_engine.py` | C image tree | TWO C deltas: (i) **e02 engine half** — `m-*` mamba boundary-state lookups (deepest PRESENT 512-token chunk end + single-object pin; stock consecutive-prefix semantics would return 0 forever for boundary-banked states); (ii) **e03 epoch-ledger double-unpin guard** — `_lookup_released` set prevents the async event-cleanup branch from re-releasing lookup-pin objects (pin_count −1 warnings). Worker-side half of e02 is p41 (vllm `mamba_utils` v136 port). |
| `patches/e-adapter-vllm-030-graft.patch` | `lmcache/integration/vllm/vllm_v1_adapter.py` | C image tree | Graft of the newer (lmcache-main, unreleased-at-0.5.5) vLLM v1 adapter: GPU connectors (`VLLMPagedMemGPUConnectorV2` etc.), lookup client, offload/ZMQ server, blend builder — the proven vllm-0.30 ↔ lmcache-0.5.5 connector compat surface. |

The SYCL XPU ops backend (`lmcache.xpu_ops`, plan slot e01 compiled half)
needs NO patch: upstream 0.5.5 ships `csrc/sycl/*` + `SyclProfile`
(`BUILD_WITH_SYCL=1`, icpx, `torch.utils.cpp_extension.SyclExtension`) and
`v1/platform/xpu/device_ops.py` binds the built extension with a torch
fallback. We build it with upstream's own recipe.

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

- `unpin` (D cache_engine v141) → **e03 present in C's 0.5.5 delta** (the
  C tree carries the guard); applied with e02e03 patch.
- `extyield2` → p42 (vllm-side lookup-elide knob), NOT an lmcache patch.
- `extyield`, `flushmove` → DROP (superseded / falsified, plan law).
- `kvev`, `xfer`, `btg`, `seedfix-v5`, `storededup` → REVAL on 0.5.5 at P3:
  port as e05+ ONLY if 0.5.5 lacks the fix AND the P3 gates expose the
  defect it fixed. Status: pending first p40 leg ( Stored/Retrieved + churn
  + integrity gates ).
- usage telemetry (plan e04 slot) → 0.5.5 ships `LMCStatsMonitor` +
  `InitializeUsageContext` + observability.py natively; p40 leg verifies the
  counters suffice for the "prewarm counts = usage-truth" gate before any
  port is considered.
- W14's 0.4.7-identity tree (with its own compiled `xpu_ops.so`) is NOT our
  base — superseded by building upstream's own SYCL backend generation-
  matched (no binary vendoring).

## Certified env set (p40 serve config — D champion serve line verbatim)

```
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
