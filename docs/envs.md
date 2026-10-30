# vllm-arc environment reference (generated)

Generated at P5 hardening (r45, 2026-10-30; serve-line identity updated
at the r46t close-out) from `vllm_gaudi/envs.py` arc-lane markers + the
certified serve line (`r43_serve_k5.sh`, spec depth 3 per the r46t
recommendation — a serve-time CLI change — on image
`vllm-arc:v0.30.0-b70-cert`). Every entry cites its provenance
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

## vllm-arc knobs — complete control surface (r45g deep-audit truth table)

Every env consumed by fork code, with its **true code default** and class.
The certified serve line exports **none** of these — all ride code defaults
(verified against `r43_serve_k5.sh`). The `envs.py` registry documents this
surface (r45g law: registry documents, consumers read `os.environ` directly
at the cited line — defaults must mirror each other).

Module cites are shorthand, shared with the `envs.py` registry:
`patches/<m>.py` = `vllm_gaudi/xpu/patches/<m>.py` (the root `patches/`
tree holds the pNN patch-item docs, not the modules), `platform.py` =
`vllm_gaudi/xpu/platform.py`, `hpu_model_runner.py` =
`vllm_gaudi/v1/worker/hpu_model_runner.py`.

**Default-ON patches** (active in the certified posture without any env;
kill-switch to disable):

| knob | default | consumer (site) | patch | disable-by |
|---|---|---|---|---|
| `VLLM_ARC_WARMUP_DISABLE` | `0` → interlock ACTIVE | `patches/first_request_warmup.py:79` | p12b | `=1` |
| `VLLM_ARC_GDN_FASTPATH_DISABLE` | `""` → fastpath ACTIVE | `patches/gdn_metadata_fastpath.py:275` | p10/r23 | any non-`0` |
| `VLLM_ARC_LOOKUP_MEMO` | `1` → memo ACTIVE (async only) | `patches/lmcache_hma_connector.py:260` | p42 | `=0` |
| `VLLM_ARC_LMC_HMA` | `1` → HMA mixin ACTIVE | `patches/lmcache_hma_connector.py:170` | p41b | `=0` (boot then dies at factory gate — A/B only) |
| `VLLM_ARC_LMC_TORCH_KV` | `1` → torch-baseline KV ops ACTIVE | `patches/lmc_xpu_pinned_env.py:85` | p40d | `=0` (native pinned ops segfault >16–32 GiB/proc) |
| `VLLM_ARC_LMC_MAMBA_SNAP` | `1` → boundary snap ACTIVE | `patches/mamba_hybrid_snap_glue.py:94` | p41c | `=0` (breaks integrity) |
| `VLLM_MAMBA_FP16_SEED_REPAIR` | `1` → repair ACTIVE | `patches/mamba_lmcache_glue.py:69` | p41 | `=0` |
| `VLLM_ARC_R35_TAIL_SPLIT` | `1` → tail-split v4 ACTIVE | `patches/sched_tail_split.py:59` | p43/r35 | `=0` |
| `VLLM_ARC_FP8_W8A16_DISABLE` | `0` → W8A16 reroute ACTIVE | `patches/fp8_block_w8a16.py:65` | p44 | `=1` (stock W8A8) |

**Default-OFF / opt-in** (inactive unless set):

| knob | default | semantics | consumer |
|---|---|---|---|
| `VLLM_ARC_LOOKUP_ELIDE` | `0` = OFF | **INT token count, not a bool** — skips both ZMQ lookups only when engine prefix held AND residual ≤ N tokens; r31 disposition: opt-in only (p40d/p41c made real joint loads exist; elide would forfeit them) | `patches/lmcache_hma_connector.py:235` |
| `VLLM_ARC_XG_COMPACT` | `false` | compact-JSON structured-output posture | `tools/serve_arc_b70.sh` (p06) |
| `VLLM_ARC_GMU_RELAX` | `0.0` | GMU conditional-relax fraction (0.8 = sanctioned B70 posture, set via serve env) | `platform.py` (p05) |

**Tuning ints** (default-ON logging/caps, donor v136 lineage names):

| knob | default | meaning |
|---|---|---|
| `LMC_BOUNDARY_KEEP` | `8` | per-req mamba boundary-registry cap |
| `LMC_BOUNDARY_CAP` | `48` | global boundary FIFO cap |
| `LMC_BOUND_STATS` | `1` | boundary stats logging |
| `LMC_BOUND_MD5` | `1` | boundary md5 logging |

**Auto-detected (not operator-set):**

| knob | set by | consumed by |
|---|---|---|
| `VLLM_COMPACT_GDN` | `hpu_model_runner.py` auto-detect (:1410) | `platform.py:501/539`, `hpu_worker.py` (GDN layout law) |

**Diagnostics (silent-off, never in the certified line):** `ARC_SNAP_PROBE`
(r31/r34/r35 leg telemetry, ~12 sites, comment-marked "silent off").
**Donor-internal (byte-pinned lmcint kit, not ours):** `LMCACHE_CONFIG_FILE`,
`LMCACHE_FORCE_SKIP_SAVE`, `LMC_STORE_LAG`.

Dropped at r45d (no-unevidenced law — declared, never consumed; audit r45
findings F1–F3; re-add only with a consumer leg): `VLLM_ARC_ASYNC` (p50
closed at P2 — core 0.30 auto-enables async scheduling),
`VLLM_ARC_YIELD_TOKENS` (p51 never built), `VLLM_ARC_COMPACT_GDN` (dead
twin of the live `VLLM_COMPACT_GDN` above).

## Naming map (r45g naming audit; completed r45h/r45i full sweep)

In-line with best practice: `VLLM_ARC_*` engine-env namespace (**11 live
vars** + 3 dropped at r45d, below), `NN-category/pNN-slug` patch dirs,
`feat/arc-pNN-<slug>` branches, `vllm-arc:v0.30.0-b70[-pNN]` image tags,
`ARC_*` **host-side script vars** (9, `tools/serve_arc_b70.sh` — a
different layer from engine envs), kebab-case docs, and the evidence-dir
convention `results/YYYY-MM-DD-<phase/item>-r<leg>-<slug>/` (internal
only, never upstreamed). **`pNN` is TWO sequences** — patch dirs/branches
carry PLAN-ITEM numbers (§5: p01..p61), image tags carry BUILD-sequence
numbers (p44, p45b, p46, …); they coincide at p44/p45 and diverge after
(image p46 exists with no item p46; item p60 branch has no image p60).
Sanctioned deviations (kept for evidence-chain continuity; rename is
an upstream-PR-lane item per §11.7 rule 8 — "format chosen to upstream
without rewriting"):

| name | deviation | why kept |
|---|---|---|
| `ARC_TRACE_FLIP` | engine env, no `VLLM_ARC_` prefix | §7-cited p12b interlock name; certified line + every evidence leg since r18 uses it |
| `ARC_SNAP_PROBE` | engine diagnostic env, `ARC_` host-layer prefix | r31/r34/r35 leg telemetry (~12 read sites: `sched_tail_split.py:93`, `mamba_lmcache_glue.py`, `mamba_hybrid_snap_glue.py`, lmcint adapter); silent-off default, deliberately NOT in the envs.py registry (envs.py:156 header law: diagnostics documented here only); rename candidate `VLLM_ARC_SNAP_PROBE` at the upstream lane |
| `VLLM_MAMBA_FP16_SEED_REPAIR` | `VLLM_` without `ARC` | upstream-style name — an asset for the upstream PR lane |
| `LMC_BOUND*` (4) | donor v136 lineage names; intra-family abbreviation split (`LMC_BOUNDARY_KEEP`/`_CAP` vs `LMC_BOUND_STATS`/`_MD5`) | ported verbatim for provenance (p41 dual-donor convergence evidence) — the BOUND/BOUNDARY split is the donor's own; unify only at the upstream lane |
| `VLLM_ARC_XG_COMPACT` | donor-lineage name (C/D `V1227_XGCOMPACT` port) worn in the engine namespace, but consumed HOST-side only (`serve_arc_b70.sh:23`) | p06 provenance + certified-line spelling; upstream-lane candidate: fold into the host layer (`ARC_XG_COMPACT`) or spell out (`VLLM_ARC_XGRAMMAR_COMPACT`) |
| `VLLM_ARC_R35_TAIL_SPLIT` | embeds leg number `r35` | named for the leg that banked tail-split **v4** (ab81fc11) after r34 A/B-falsified v3 (c4468339); `R35` ties the knob to its falsification evidence — same evidence-chain class as `ARC_TRACE_FLIP`; internal marker attrs `_arc_r35_tail_split`/`_arc_r35_probe` (`sched_tail_split.py:65/94`) share the stem (internal-only, never upstreamed); rename candidate `VLLM_ARC_TAIL_SPLIT` at the upstream lane |
| `VLLM_ARC_LOOKUP_ELIDE` | name reads bool, value is int tokens | documented above; rename candidate `..._ELIDE_TOKENS` at the upstream lane |
| module prefix `lmc_` vs `lmcache_` | our own patch modules use both (`lmcache_hma_connector.py` vs `lmc_xpu_pinned_env.py`, `lmc_twohs_transfer.py`, `lmc_kvfmt_detector.py`) | historical per-patch coinage; `patches/lmcint/` dir is the donor kit name (kept for provenance); unify candidate `lmcache_*` at the upstream lane |

Regenerate: this file is a snapshot; on any envs.py or serve-line change,
update alongside the leg's VERDICT (the audit leg re-runs the §A–E sweeps).
