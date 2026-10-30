# vllm-arc environment reference (generated)

Generated at P5 hardening (r45, 2026-10-30) from `vllm_gaudi/envs.py` arc-lane
markers + the certified serve line (`r43_serve_k5.sh`, canonical k5 posture on
image `vllm-arc:v0.30.0-b70-p45b`). Every entry cites its provenance
(no-unevidenced law, plan Appendix H #6). Audit of record:
`results/2026-10-30-p5-r45-env-log-audit/`.

## Certified serve-line env chain (24 vars)

Runtime/system:

| var | value | provenance |
|---|---|---|
| `PYTHONHASHSEED` | `0` | every leg since r22a; NONE_HASH init determinism (r30 law) |
| `ZE_AFFINITY_MASK` | `0,1` | p02 dual-B70 law; champion boot `boot_iexp_v1237.sh` |
| `LD_LIBRARY_PATH` | `/opt/venv/lib` + torch lib | Dockerfile runtime; P1 boot interface |
| `CCL_ENABLE_SYCL_KERNELS` | `1` | §7 ENV map CCL row — D champion set verbatim (oneCCL runtime) |
| `CCL_ZE_IPC_EXCHANGE` | `pidfd` | §7 CCL row + r18/r30 pidfd IPC law (`--privileged` pair) |
| `CCL_TOPO_P2P_ACCESS` | `1` | §7 CCL row (champion boot script) |
| `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK` | `0` | §7 CCL row |
| `CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD` | `131072` | §7 CCL row |
| `CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD` | `1048576` | §7 CCL row |
| `CCL_SYCL_ALLGATHERV_TMP_BUF` | `0` | §7 CCL row |
| `CCL_SYCL_ALLREDUCE_TMP_BUF` | `0` | §7 CCL row |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | `1` | standing rig law (plan App I #11) |
| `ARC_TRACE_FLIP` | `1` | p12b flip-payload interlock (r11/r12 tracer); gates every leg r18+ |

LMCache certified set (D champion verbatim, plan Appendix E — keep as a set):

| var | value |
|---|---|
| `LMCACHE_USE_EXPERIMENTAL_API` | `True` |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | `100` |
| `LMCACHE_LOCAL_CPU` | `True` |
| `LMCACHE_EVICT_POLICY` | `lr_u` |
| `LMCACHE_CHUNK_SIZE` | `1024` |
| `LMCACHE_SAVE_DECODE_CACHE` | `True` |
| `LMCACHE_MIN_RETRIEVE_TOKENS` | `1024` |
| `LMCACHE_LOG_LEVEL` | `INFO` |
| `LMCACHE_DISABLE_PINNED_MEMORY` | `1` (p40d law: pinned USM >16–32 GiB/process segfaults) |

Container flags (docker run, P1 boot interface):

| var | value | why |
|---|---|---|
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | `--max-model-len 262144` exceeds derived config max |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | TP2 worker process law (P1) |

## vllm-arc knobs (`vllm_gaudi/envs.py` arc lane)

| knob | default | consumer | status |
|---|---|---|---|
| `VLLM_ARC_LOOKUP_ELIDE` | `true` | `patches/lmcache_hma_connector.py` (p42) | live |
| `VLLM_ARC_GMU_RELAX` | `0.0` | `platform.py` (p05; 0.8 sanctioned on B70) | live |
| `VLLM_ARC_XG_COMPACT` | `false` | `tools/serve_arc_b70.sh` (p06) | live |
| `VLLM_ARC_ASYNC` | `None` (core 0.30 default) | — | inert (p50 closed at P2; drop at p61 — r45 F2) |
| `VLLM_ARC_YIELD_TOKENS` | `4096` | — | inert (p51 never built; drop at p61 — r45 F1) |
| `VLLM_ARC_COMPACT_GDN` | `false` | — | dead twin of live `VLLM_COMPACT_GDN`; drop at p61 (r45 F3) |

Live non-ARC fork knobs (auto-detected, not operator-set):

| knob | set by | consumed by |
|---|---|---|
| `VLLM_COMPACT_GDN` | `hpu_model_runner.py` auto-detect (:1410) | `platform.py:501/539`, `hpu_worker.py` (GDN layout law) |

## Kill-switches (patch escape hatches; all default-OFF = patch active)

| switch | off-value | patch |
|---|---|---|
| `VLLM_ARC_FP8_W8A16_DISABLE` | unset → reroute ACTIVE | p44 fp8 block W8A16 route |
| `VLLM_ARC_LMC_HMA` | `0` → mixin off | p41b HMA connector |
| `VLLM_MAMBA_FP16_SEED_REPAIR` | `0` → repair off | p41 mamba glue |

Regenerate: this file is a snapshot; on any envs.py or serve-line change,
update alongside the leg's VERDICT (the audit leg re-runs the §A–E sweeps).
