# p40 — LMCache connector (vllm-side glue + certified env set)

## What

External caching on the target model via `--kv-transfer-config
LMCacheConnectorV1 use_native + mamba_pool_ratio 2, engine_id arc-p40`.

- **Wheel** `lmcache-0.5.5+arc1` — kit, build law, md5 pins, patch table:
  [`external_patches/lmcache/README.md`](../../../external_patches/lmcache/README.md)
  (base = upstream 0.5.5 per Appendix E; C's three deltas e01–e03 + adapter
  graft e04; built with upstream's own sycl recipe, no vendored binaries).
- **vllm-side glue** `vllm_gaudi/xpu/patches/lmcache_hma_connector.py`
  (p41b): upstream-0.30's v1 wrapper lacks the `SupportsHMA` mixin the
  factory demands (factory.py:58 refuses HMA without it) — byte-preserved
  port from C's reference (md5 cee1a393 lineage) adding the mixin +
  `request_finished_all_groups` + the v133 async-gate fix (real
  `load_kv_async`, 3-arg `update_state_after_alloc` riding our scheduler's
  native WAITING_FOR_REMOTE_KVS) + KV-events guard. Knob
  `VLLM_ARC_LMC_HMA=0` disables.
- **lmcint kit** `vllm_gaudi/xpu/patches/lmcint/` — the 0.5.5-compat patch
  of vllm's vendored `lmcache_integration` adapter (C carries the same;
  byte-pinned md5s in p41b commit). `lmcache_hma_connector` redirects the
  adapter import onto our copy (stock `__init__` imports the absent
  `multi_process_adapter`). The adapter is the mamba-hybrid engine
  (MambaPageConnector, per-group block_ids); it hard-imports p41's registry
  symbols — p41 is a boot prerequisite.
- **Pinned-env device ops** `lmc_xpu_pinned_env.py` (p40d): when
  `LMCACHE_DISABLE_PINNED_MEMORY` is set, drops the two instance-bound
  native pinned allocs (pinned USM >16–32 GiB/process segfaults — C law);
  all other native ops stay native.
- **Certified env set** (Appendix E, D champion verbatim): the 9
  `LMCACHE_*` vars in the serve line — see `docs/envs.md`.
- **Knobs** (default-ON, see docs/envs.md): `VLLM_ARC_LMC_HMA` (p41b HMA
  mixin, `=0` = A/B only — boot dies at the factory gate) ·
  `VLLM_ARC_LMC_TORCH_KV` (p40d KV-transfer ops on the torch baseline,
  `=0` forces the native instance-bound ops that segfault >16–32
  GiB/process) · `LMCACHE_DISABLE_PINNED_MEMORY` (p40d law, certified line
  sets `=1`).
- **Image self-containment** (r42 Fix A): Dockerfile vendors the
  `--no-deps`-missed closure (`aiofile`, `redis` — `gds_backend.py:19`
  module-level import) + in-build import gate + `IMAGE_MANIFEST
  lmcache_closure=` line. Runtime pip installs into the container are
  BANNED (Appendix H law; legs r30–r41 had masked the gap that way).

## Test

Rig battery (first-traffic, timers-off): Stored/Retrieved > 0 on churn;
warm-hit face ≤ 0.75 s class (r36 face law); LMCache re-hit integrity 5/5
byte-identical (`r42_integrity.py`); hooks `lmcint src-override OK` in
every process, zero FAILED. Legs: r30–r36, r42 (certified image),
r43 (k5), r44 (matrix).

## Validation

- r30 first leg: boot chain of 5 defects found+fixed (privileged/pidfd,
  HMA mixin, vendored adapter, aiofile/redis, pinned USM) — commit 77824b03.
- r36: warm gate closed (beat-D-adder MET 1.03–1.05 vs D-SPEC1 1.14–1.28;
  absolute 0.75 at 8k structural — composition-coherence law).
- r42: all faces green on the standalone certified image (solo 85.0–85.2,
  warm-hit 740 ms bar MET first time).
