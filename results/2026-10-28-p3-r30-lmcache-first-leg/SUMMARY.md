# P3 r30 — first LMCache leg (p40 wheel + p41 glue + p41b/p40c boot-chain fixes)

**Date:** 2026-10-28 · **Branch:** feat/arc-p41-mamba-lmcache-glue · **Image:**
vllm-arc:v0.30.0-b70-p41 (ef8aa0da6558, fork_sha p41-5729b344) · **Posture:**
product verbatim (mode=none + FULL_DECODE_ONLY 1..320, MTP k4, fp8, kv
fp8_e4m3, prefix caching, HMA ON) + LMCacheConnectorV1 use_native +
9-var LMCACHE env set (D champion verbatim) + PYTHONHASHSEED=0 (C law).

## VERDICT: PASS with follow-ups — LMCache store/load PROVEN on the arc
stack; warm-hit gate 5 ms over (p43 scoped to close it); churn-eviction
cell needs pool pressure (design fix, not a cache defect).

## Gate results (timers off)

| gate | result | evidence |
|---|---|---|
| survival (flip payload first traffic) | **PASS** 200/200 | survival.txt |
| spec acceptance pos1 | **PASS** hard (0.8985 ≥ 0.5); target-miss recorded (0.85) — in P2 band 0.877–0.912 | spec_gate.json |
| golden vs banked product ref | **PASS 5/5 zero drift** | golden_compare.txt |
| LMCache integrity (cold-sha == re-hit-sha, 5 seeds) | **PASS 5/5 byte-identical** | r30_lmc.json |
| LMCache warm-hit ≤ 750 ms | **FAIL by ~5 ms** — single 778.2; WARM5 re-measure: 800.9/757.5/755.1/750.7/752.4 → **median 755.1, min 750.7**, sha-identical hits | r30_lmc.txt, scripts/r30_warm5.py |
| LMCache warm vs cold (8k) | cold 5556.1 ms → warm 755 ms = **7.4×** | r30_lmc.json |
| churn: re-hit most-recent | **PASS** 776.8 ms (warm class) | r30_lmc.txt |
| churn: oldest seed evicted → cold class | **N/A — no eviction pressure** (24×8k ≪ 100 GiB pool; both re-hits warm-class by design of the pool) | r30_lmc.txt |
| run7n regression | **PASS** solo 59.15/59.20 tok/s, agg 196.2/202.2 (P2-consistent) | run7n_1.txt |
| rig restore | **PASS** W14_HEALTH_200 21:39:51 | teardown log |

## Boot chain — five defects found and fixed en route (all evidence in leg)

1. **oneCCL `init_device_fds` opendir failure** (both TP ranks died at
   `all_reduce` init): r30 docker run omitted `--privileged` (r18
   precedent — oneCCL pidfd IPC needs /proc fd access). Fix: `--privileged`
   added to r30_boot.sh docker run.
2. **HMA refusal** (`factory.py:58`: "Connector LMCacheConnectorV1 does
   not support HMA but HMA is enabled"): upstream-0.30's own
   v1/lmcache_connector.py wrapper lacks the SupportsHMA mixin. D-line
   resume13 ledger already adjudicated this exact gate (flag-flip
   REJECTED: "certified memory-manager semantics vs CL-1 HMA-on bar";
   fix = byte-preserved port from v1.2.35, md5 cee1a393…). **Fix = p41b**
   (`vllm_gaudi/xpu/patches/lmcache_hma_connector.py`): SupportsHMA mixin
   + request_finished_all_groups + v133 async-gate fix (real
   load_kv_async + 3-arg update_state_after_alloc; rides our scheduler's
   native WAITING_FOR_REMOTE_KVS) + KV-events guard (0.5.5 has no
   get_kv_events). Subclass-rebind via sitecustomize hook; knob
   VLLM_ARC_LMC_HMA=0 for A/B.
3. **CudaIPCWrapper ImportError**: under `use_native:true` the connector
   instantiates VLLM'S VENDORED lmcache_integration adapter, but upstream
   0.30's vendored copy targets lmcache-main (its __init__ imports
   multi_process_adapter → hard-imports CudaIPCWrapper, absent in 0.5.5).
   Full-tree diff vs the C image proved C carries a 0.5.5-compat patch of
   the vendored package (its __init__ header: "multi_process_adapter
   requires lmcache main (CudaIPCWrapper) and is NOT importable here") —
   the compat surface extends beyond the 3-file lmcache wheel delta.
   **Fix = p41b lmcint kit** (`patches/lmcint/{__init__,vllm_v1_adapter,
   utils}.py`, byte-preserved C files md5 1f3e97d3/b5300ee0/dcb23fbe,
   source-override exec at import; no installed-tree edits). The vendored
   adapter is the mamba-hybrid engine (MambaPageConnector,
   _init_mamba_engine, per-group block_ids) and late-imports p41's
   _LMC_SEED_OVERRIDE/_LMC_BOUNDARY_REGISTRY at its lines 1741/2049.
4. **`No module named 'aiofile'`**: lmcache-0.5.5 runtime deps missed by
   the p40 `--no-deps` install set (upstream Requires-Dist lists aiofile
   + redis; storage_backend fs_connector imports aiofile eagerly).
   Fix: `pip install aiofile redis` in r30_boot.sh (container is
   recreated each boot — the earlier in-container install was wiped);
   Dockerfile dep line gains both at next image build.
5. **`sycl::malloc_host failed for 107374182400 bytes (XPU pinned
   host)`**: our 0.5.5-built lmcache.xpu_ops exports NATIVE
   alloc_pinned_ptr/free_pinned_ptr, and XpuDeviceOps.bind_native
   instance-binds them OVER the class-level torch baseline that honors
   LMCACHE_DISABLE_PINNED_MEMORY (C's 0.4.7-era binary lacked the pair,
   so C always ran the env-honoring baseline — C serve_lmc_v133.sh: "SYCL
   USM pinned host alloc segfaults above ~16-32GiB per process (probed
   2026-10-08); run the CPU pools pageable"). **Fix = p40c**
   (`patches/lmc_xpu_pinned_env.py`): with the env set, drop the two
   instance bindings post-bind so the pageable torch baseline serves
   them; all other native ops stay native.

## Hybrid-mode evidence (serve log)

- `Hybrid model: forcing lmcache chunk_size 1024 -> 896 (mamba
  tokens_per_page)`; `LMCache hybrid mode: attn layers=17 mamba layers=48
  (fa_gidx=3 mamba_gidxs=[0,1,2] page_bytes=917504 tpp=896)`
- `Initialized NONE_HASH=5313149338359954478 from vLLM (>= PR#20511)` +
  `hash algorithm: builtin` under PYTHONHASHSEED=0 (C's CRITICAL
  cross-process hash law, serve_lmc_v133.sh)
- p41 glue `_install OK` in all 13 processes; p41b markers in
  APIServer/EngineCore/Worker; p40c pinned-drop marker present.

## Follow-ups (next legs)

- **p42**: lookup-elide (extyield2 port, VLLM_ARC_LOOKUP_ELIDE knob).
- **p43 warm-t2 adder** (plan §12: "warm-t2 adder closure = beat C's
  hit-cost with D's correctness"): close the 755→≤750 ms warm gate.
- **Churn cell redesign**: eviction needs pool pressure — lower
  LMCACHE_MAX_LOCAL_CPU_SIZE for the churn leg (e.g. 2 GiB) or scale
  seed count; re-run at p43 leg.
- **Lagged attn store telemetry**: 160 log lines (≈80 events ×2 ranks),
  21:26:46–21:33:59 (LMCache-gate + run7n phases), C's graceful-degrade
  path ("cached chunk absent; later lookups recompute") — non-fatal,
  integrity/warm/churn gates unaffected; re-examine after p43 changes
  store timing.
- **Dockerfile**: add aiofile+redis to the lmcache runtime dep set.
- **Stepprobe cell**: battery referenced stale path /root/r22a_stepprobe.py
  (not deployed) — redeploy from r22a stage at next leg.

## Files

- `legs/arc-p3-r30/` — battery artifacts + full serve log
- `scripts/` — r30_boot.sh, r30_serve.sh, r30_battery.sh,
  r30_teardown.sh, r30_sitecustomize.py, r30_warm5.py, battery.log
