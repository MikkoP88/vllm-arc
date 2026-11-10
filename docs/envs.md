# vllm-arc environment reference (generated)

Generated at P5 hardening (r45, 2026-10-30; serve-line identity updated
at the r46t close-out) from `vllm_gaudi/envs.py` arc-lane markers + the
certified serve line (`r43_serve_k5.sh`, spec depth 3 per the r46t
recommendation — a serve-time CLI change — on image
`vllm-arc:v0.30.0-b70-cert`; unchanged since — the qwen fp8 chain is identical on
the current certified image `vllm-arc:v0.30.0-b70-cert22`, where
`LMCACHE_TRACK_USAGE=false` (P6.3) made it 25; cert22 = cert21 + the
P6.33 skinny-M kernel lane (arc6 `vllm_xpu_kernels` wheel swap +
the `skinny_m` route patch — adds NO serve-line var; the lane rides
the engine env `ARC_SKINNY_M` (unset/`off` = stock routes — the
CERTIFIED default after the r127 e2e falsification; `on` = the M≤8
kernels opt-in; a serve-ENV knob passed through by the script, like
`ARC_ATTN_FA2`/`ARC_MXFP8_MODE`); serve behavior byte-identical to
cert21 — r127g2 cert leg 27.7/192.9 exact ref, zero route markers);
cert21 = cert20 + the
P6.31 online-MXFP8 W8A16 lane (arc5 `vllm_xpu_kernels` wheel swap +
the `mxfp8_w8a16` patch — adds NO serve-line var; the lane rides the
model-lane env `ARC_QUANT=mxfp8` + the engine env `ARC_MXFP8_MODE`
(unset/`w8a16` = the W8A16 route — default; `w8a8` = stock control; a
serve-ENV knob, not part of the 25-var chain, like `ARC_ATTN_FA2`) +
the script arg `ARC_MAXLEN` (default 262144; 40960 on gemma
online-mxfp8 legs — the 8-bit-weights KV capacity law); the
fp8/checkpoint/mxfp4 lanes are byte-untouched — r125 three-lane gate);
cert20 = cert19 + the
P6.30 online-quant W4A16 lane (arc4 `vllm_xpu_kernels` wheel swap +
the `mxfp4_w4a16` patch — adds NO serve-line var; the lane rides the
existing model-lane envs `ARC_QUANT=mxfp4` + `ARC_QUANTCFG` (the
hybrid ignore list) + the engine env `ARC_MXFP4_MODE` (unset/`w4a16` =
the W4A16 route — default; `w4a4` = stock control; a serve-ENV knob,
not part of the 25-var chain, like `ARC_ATTN_FA2`); the fp8/checkpoint
lanes are byte-untouched — r121 discriminator); cert19 = cert18 + the
P6.29 sliding-window FA2 route (the same wheel file,
`triton_unified_attention.py` — adds NO serve-line var; the shared
engine env `ARC_ATTN_FA2` defaults 1 = the hd512 full-attention route
AND the hd256 sliding-window route ON, `=0` = the pure-triton escape
hatch — a serve-ENV knob, not part of the 25-var chain; gemma-4 hits
both routes, qwen3.8's full-attn never hits the hd512 route
marker-proven at r108ae/r111); cert18 = cert17 + the
P6.28 FA2 route (one wheel file — adds
NO serve-line var); cert17 = the cert11
engine + the P6.28 chain (cert12 launcher-table points, cert16 warm-gate
v4, cert17 lmcint kit — none adds a qwen serve-line var; the gemma-4
lane rides the same script with `ARC_LMC=0` + `ARC_RPC_TIMEOUT=1200`
(→ `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`; REQUIRED for gemma-4
long-prefill faces — the stock 300 s execute-model watchdog kills late
chunks) + `ARC_TEMPLATE`); cert11 = cert10 + the
P6.26 platform patch ONLY (`vllm_gaudi/xpu/platform.py` — quark/mxfp4
lane, config-validation surface; r105b walk-proven byte-identical
everywhere else, and the fp8 posture proven env- and face-identical at
the r105 gate); cert10 = cert9's engine
byte-identical (exhaustive md5 walk at the cert10 gate) + the P6.24
oneCCL fp8-wire overlay, which is **default-OFF** and adds NO certified
serve-line var — opting into the e4m3-wire allreduce is the single
uncertified-line addition `CCL_SYCL_ALLREDUCE_FP8=1` (rows in the
overlay table below + `CONFIGURATION-arc.md`'s knob table; proven
env-absent at the cert10 gate with
FP8_ENV_SET=0 and default-posture numbers cert9-identical; the overlay
rows carry into cert11);
cert9 = cert6-era
compute bytes under the certified fp8 line — the P6.19 TQ fix never
engages there (control-proven t7/t9 + cert8 gate) and the P6.20
off-thread store (L3, default ON) adds no serve-line var — it rides the
code default, proven env-free at the cert9 gate [STORE_WORKER_ACTIVE
with LMC_ENV_SET=0]). Every entry cites its
provenance (no-unevidenced law, plan Appendix H #6). Audit of record:
`results/2026-10-30-p5-r45-env-log-audit/`.

## Certified serve-line env chain (25 vars: 13 runtime/system + 10 LMCache + 2 boot-layer)

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
| `LMCACHE_TRACK_USAGE` | `false` (P6.3: LMCache usage telemetry is default-ON — `usage_telemetry/identity.py:25`; this rig serves offline. Pairs with the e04 wheel patch that gates the per-request usage-context creation, the source of the dual-engine "different metadata" ERROR ×2/boot) |

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
| `VLLM_ARC_LMC_MAMBA_SNAP` | `1` → boundary snap ACTIVE | `patches/mamba_hybrid_snap_glue.py:94` | p41c | `=0` (breaks integrity) |
| `VLLM_MAMBA_FP16_SEED_REPAIR` | `1` → repair ACTIVE | `patches/mamba_lmcache_glue.py:69` | p41 | `=0` |
| `VLLM_ARC_R35_TAIL_SPLIT` | `1` → tail-split v4 ACTIVE | `patches/sched_tail_split.py:59` | p43/r35 | `=0` |
| `VLLM_ARC_FP8_W8A16_DISABLE` | `0` → W8A16 reroute ACTIVE | `patches/fp8_block_w8a16.py:65` | p44 | `=1` (stock W8A8) |
| `ARC_MXFP4_MODE` | `w4a16` → certified route (bf16 acts → `fp4_gemm_w4a16`, zero per-call act quant; cert20) | `patches/mxfp4_w4a16.py:65` | p630 | `=w4a8` P6.31 probe lane (per-token e4m3 acts); `=w4a4` stock control |
| `ARC_MXFP8_MODE` | `w8a16` → new route (bf16 acts → `fp8_gemm_w8a16_mxfp8`, group-32 e8m0 weight scales, zero runtime act quant; P6.31) | `patches/mxfp8_w8a16.py:57` | p631 | `=w8a8` stock control |
| `ARC_SKINNY_M` | unset/`off` → stock routes at all Ms (**certified default**: r127 e2e falsified the reroute — under XPU-graph replay stock oneDNN runs 420–680 GB/s device-side and the recorded skinny kernels regressed gemma solo −38/−42%; results/2026-11-08-p633-skinny-m/) | `patches/skinny_m.py:167` | p633 | `=on` opt-in (per-lane caps w16 M=1, m816/g16 M≤2; eager-regime wins only — graph replay loses 1.6–2.6×; serve needs the rig `serve_arc_b70.sh` ARC_SKINNY_M pass-through) |
| `VLLM_ARC_TQ_SPEC_FIX` | `1` → spec-verify fix ACTIVE on every `turboquant_*` dtype (fp8 never imports the hooked module — control proof t7/t9) | `patches/tq_spec_fix.py:99` | p619 | `=0` (pre-fix TQ spec-verify — corrupt under graphs, diagnosis only) |
| `VLLM_ARC_GRAPH_EXEC_DB` | `1` → graph-exec double-buffer ACTIVE: every XPU graph captured ×2 into the manager's shared pool, replays alternate execs — torch 2.13 same-exec `XPUGraph.replay()` blocks until that exec's prior launch retires (p135i microbench: 13.96 ms same-graph in-flight vs 8–12 µs alternating; leg J: replay max 24 → 0.37 ms, engine serial 40.8 → 12.7 ms/step). **E2E-honest note (cert23 gate A/B): end-to-end step wall is NEUTRAL at every measured posture (fp8-k3 94.5 vs 94.3 tok/s; mx4k5 121.0 vs 123.9) — the clean engine's step was never replay-limited (the block overlaps the device-floor work; leg-F's 81.6 "baseline" was probe tax ~3.8 ms/step). Ships as a costless driver-level hygiene fix (+29 s boot, 0 extra memory, byte-identical semantics); XPU-only — CUDA launches are already async. results/2026-11-10-p635-telemetry/) | `patches/graph_exec_doublebuffer.py:76` | p635 | `=0` stock single-exec replays (cert23 gate: boots with 0 db markers = kill-switch proven) |

Dead since r47b (kept OUT of the live tables — no-unevidenced law):
`VLLM_ARC_LMC_TORCH_KV` — its consumer module
`patches/lmc_xpu_pinned_env.py` was DELETED when p40c/p40d were
absorbed into the lmcache wheel patch
(`external_patches/lmcache/patches/e01-xpu-platform-ops.patch`: the
5-op torch-baseline transfer re-pin is UNCONDITIONAL in
`XpuDeviceOps.ensure_native`; the live gate is
`LMCACHE_DISABLE_PINNED_MEMORY`, part of the certified 25-var chain).
Setting the var is a no-op on the certified image. (The stale registry
row in `vllm_gaudi/envs.py:196-198` is engine-layer and stays
untouched — the cert6 engine is byte-frozen; cleanup belongs to the
next engine-touching item.)

**Default-OFF / opt-in** (inactive unless set):

| knob | default | semantics | consumer |
|---|---|---|---|
| `VLLM_ARC_TQ_VERIFY_MODE` | `"split"` = the certified dispatch (q_len single-row stock decode calls, graph-safe — t5/t9-proven all dtypes, cert8 gate) | `=mq` single-pass MQ kernel on the wheel's AoS layout (`patches/tq_mq_aos.py`; MQVAL5 4/4 presets exact vs the engine's own pair; cert8 gate annex: agg8 284.81 = 1.21× split's 235.97, solo unchanged, t8g family 289.23 — **experimental**: multi/simple identity faces wider than split); `=synth` the pre-fix multi-row synthetic-decode dispatch (t4f-proven replay-broken — differential diagnosis only) | `patches/tq_spec_fix.py:100` |
| `VLLM_ARC_TQ_MQ_SPLITS` | `32` | KV split count for the MQ kernel (`mq` mode only) | `patches/tq_mq_aos.py` |
| `VLLM_ARC_LOOKUP_ELIDE` | `0` = OFF | **INT token count, not a bool** — skips both ZMQ lookups only when engine prefix held AND residual ≤ N tokens; r31 disposition: opt-in only (p40d/p41c made real joint loads exist; elide would forfeit them) | `patches/lmcache_hma_connector.py:235` |
| `LMC_DEFER_CHUNK_STORE` | `"0"` = OFF | P6.5 chunk-set store deferral — under `--long-prefill-token-threshold`, intermediate prefill-chunk rows carry `can_save=False` and leave `num_saved_tokens` un-advanced; the final chunk stores the whole span in ONE event. Opt-in long-context profile only (with `LMC_DEFER_MIN_TOKENS=40960`); default posture byte-identical to cert3 | lmcint `vllm_v1_adapter.py` `from_request_tracker` (p65/r48l) |
| `LMC_DEFER_MIN_TOKENS` | `"0"` = defer every chunked prefill | length gate: defer only prompts with `prompt_len >= N`. Certified value **40960** — per-chunk events are free at 32k (LMC ≡ no-LMC) while the ungated one-shot store regresses that face +4.2 s (r48k); the win only exists at 64k (+24.6 s attribution @12 bys) | lmcint `vllm_v1_adapter.py` `from_request_tracker` (p65/r48l) |
| `LMC_STORE_SLICE_TOKENS` | `"0"` = OFF (certified default; ships since cert5) | **experimental** — P6.13: slices the deferred one-shot final-span store into ~19 bounded windows (3328 tok runtime each), one per step, same chunk-row store call shape. `=4096` opt-in on top of the defer profile: U64k TTFT −1.2 s, `63232`-class → 0; NOT adopted into the certified profile (pre-registered max-gap bar mis-set — rung3 VERDICT §2) | lmcint `vllm_v1_adapter.py` `_lmc_flush_pending` (p73, cert5) |
| `VLLM_ARC_XG_COMPACT` | `false` | compact-JSON structured-output posture | `tools/serve_arc_b70.sh` (p06) |
| `VLLM_ARC_GMU_RELAX` | `0.0` | GMU conditional-relax fraction (0.8 = sanctioned B70 posture, set via serve env) | `platform.py` (p05) |
| `VLLM_ARC_LMC_ASYNC_STORE` | `1` = ON (**cert9 certified default**, p620) | L3 off-thread store: the per-step lagged-flush dispatcher hands (pending, engine stream) to the `arc-lmc-store` daemon — device ops stay on the engine stream (in-order law), only the HOST wait (sync + D2H gathers + store insert) moves off the engine thread. r81 A/B/A/B (cert8 base): U32k TTFT −0.74 s mean, U64k −1.64 s, U8k −0.08 s (both interleave pairs), zero error delta; r82/r83 full gate batteries green incl. LONG-RETURN (zero-KV proof under async store). `=0` = the cert≤8 inline posture (kill switch) | lmcint `vllm_v1_adapter.py` `_lmc_flush_pending`/`_lmc_store_machinery` (p620 L3; L6 dispatch-late = same flush handed at wfs end) |
| `VLLM_ARC_LMC_EVSYNC` | `1` | L1 flush-scoped event sync inside the store body (vs `torch.xpu.synchronize()`); r80: neutral ALONE — on the in-order XPU stream the event sits behind the just-submitted chunk forward; kept as the cheap default inside the L3 worker body | lmcint `vllm_v1_adapter.py` `_lmc_flush_body` (p620 L1) |

**oneCCL overlay envs (cert10+; carried into cert11; consumed by the patched `libccl`, NOT by
engine code — registered by `arc-fp8-allreduce.patch`, deliberately
outside every namespace above; full knob rows with measured semantics in
CONFIGURATION-arc.md's knob table, posture guide in its
"Best-performance postures" section):**

| var | default | semantics |
|---|---|---|
| `CCL_SYCL_ALLREDUCE_FP8` | `0` = OFF (**the certified default** — proven env-absent at the cert10 gate, FP8_ENV_SET=0) | `=1` opts the TP2 fp16 large ring into the e4m3-wire allreduce (1.65× at the dominant 76,677,120 B AR; every p65 TTFT face beats cert9 — r99 A/B/A/B; decode/spec neutral, r98 task-level quality parity; sub-`_MIN_COUNT` allreduces take the stock path — the `simple`-class byte-identity proof) |
| `CCL_SYCL_ALLREDUCE_FP8_MIN_COUNT` | `2097152` elements/rank | only messages ≥ this count use the fp8 wire (measured crossover — fp8 is slower below ~2M elements: 0.353 vs 0.327 ms at 2.6 MB); leave at default |
| `CCL_SYCL_ALLREDUCE_FP8_STATS` | `0` = silent-off | `=1` per-stage kernel telemetry to stderr — diagnostic only, declared perturbation (drains the pipeline between stages; numbers attribute cost, they do not race) |

Host-script pass-through: `ARC_FP8=1` on `tools/serve_arc_b70.sh` appends
the opt-in env to the exec block (default emits no line — the served
process env is byte-identical to the pre-knob script).

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
(r31/r34/r35 leg telemetry, ~12 sites, comment-marked "silent off"),
`ARC_STEP_TRACE` (p620 per-step host timeline — engine/worker step JSONL
decomposition: schedule/prep_inputs/execute_model/sample_tokens/wfs/
wfs_flush/wfs_body/wfs_total/get_finished; `ARC_STEP_TRACE_FILE` override;
banked r80–r83 legs; same registry-exempt class as `ARC_SNAP_PROBE`),
`ARC_AR_TRACE` (p622 per-call collective census — wraps
`torch.distributed.{all_reduce, all_gather_into_tensor,
reduce_scatter_tensor, barrier, all_gather}` in every platform-hooked
process, JSONL {ts, rank, op, caller, numel, dtype, bytes, shape,
enqueue dt_ms}; `ARC_AR_TRACE_FILE` override; the P6.22 TP2-AR lane's
message-size instrument; same registry-exempt class).
**Donor-internal (byte-pinned lmcint kit, not ours):** `LMCACHE_CONFIG_FILE`,
`LMCACHE_FORCE_SKIP_SAVE`, `LMC_STORE_LAG`.

Dropped at r45d (no-unevidenced law — declared, never consumed; audit r45
findings F1–F3; re-add only with a consumer leg): `VLLM_ARC_ASYNC` (p50
closed at P2 — core 0.30 auto-enables async scheduling),
`VLLM_ARC_YIELD_TOKENS` (p51 never built), `VLLM_ARC_COMPACT_GDN` (dead
twin of the live `VLLM_COMPACT_GDN` above).

## Naming map (r45g naming audit; completed r45h/r45i full sweep)

In-line with best practice: `VLLM_ARC_*` engine-env namespace (**18 live
vars** — the r45g audit's 10 + `VLLM_ARC_TQ_SPEC_FIX`/`VLLM_ARC_TQ_VERIFY_MODE`/`VLLM_ARC_TQ_MQ_SPLITS` (p619) +
`VLLM_ARC_LMC_ASYNC_STORE`/`VLLM_ARC_LMC_EVSYNC` (p620) +
`ARC_MXFP4_MODE` (p630) + `ARC_MXFP8_MODE` (p631) + `ARC_SKINNY_M`
(p633); grep-verified
against code consumers at the P6.23 docs audit + the 2026-11-08 P6.31
registry catch-up (the p619 trio and the p630 gate had gone live
docs-only; the `vllm_gaudi/envs.py` registry now mirrors the docs) — + 3 dropped at r45d +
1 dead since r47b, below), the P6.24 `CCL_SYCL_ALLREDUCE_FP8`/
`_MIN_COUNT`/`_STATS` trio (cert10 oneCCL overlay — oneCCL-runtime envs,
deliberately OUTSIDE the `VLLM_ARC_` namespace: they configure the comm
library, not the engine; registered by `arc-fp8-allreduce.patch`, all
default-off, documented in `CONFIGURATION-arc.md`'s knob table),
`NN-category/pNN-slug` patch dirs,
`feat/arc-pNN-<slug>` branches, `vllm-arc:v0.30.0-b70[-pNN]` image tags,
`ARC_*` **host-side script vars** (23 — recounted at the P6.33
close-out: the base 11 plus the P6.28–P6.33 lane knobs
`ARC_MODEL`/`ARC_SERVED`/`ARC_QUANT`/`ARC_QUANTCFG`/`ARC_TEMPLATE`/
`ARC_RPARSER`/`ARC_TPARSER`/`ARC_RPC_TIMEOUT`/`ARC_MXFP4_MODE`/
`ARC_MXFP8_MODE`/`ARC_MAXLEN`/`ARC_SKINNY_M`, most as set-only passthroughs (default
emits no line) or defaulted-overridable script args;
`tools/serve_arc_b70.sh` — a
different layer from engine envs; the 11th is `ARC_FP8`, the P6.24
cert10 overlay pass-through: `1` appends the oneCCL opt-in env above,
default emits no line; the 10th is `ARC_SPEC_METHOD`, the
draft-sampling posture knob: P6.17 — unset = `probabilistic` = the
CERTIFIED default (appends the upstream
spec-config key `draft_sample_method`), `greedy` = the opt-out
historical lane — flip validated
`results/2026-11-03-p617a-probcert/`), kebab-case docs, and the
evidence-dir
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
| module prefix `lmc_` vs `lmcache_` | our own patch modules use both (`lmcache_hma_connector.py` vs the `patches/lmcint/` kit's `lmc_twohs_transfer.py`, `lmc_kvfmt_detector.py`) | historical per-patch coinage; `patches/lmcint/` dir is the donor kit name (kept for provenance); unify candidate `lmcache_*` at the upstream lane |

Regenerate: this file is a snapshot; on any envs.py or serve-line change,
update alongside the leg's VERDICT (the audit leg re-runs the §A–E sweeps).
