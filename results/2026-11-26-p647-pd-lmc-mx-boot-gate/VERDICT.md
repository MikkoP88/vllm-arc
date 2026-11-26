# P6.47 P-D — LMC x MX boot gate: VERDICT

**PASS — the r240b posture confound is now structural.** The engine
refuses an LMCache-named connector over an MX KV dtype at connector
construction (`ARC_LMC_MX_REFUSED` ValueError), beside the stock
factory's own HMA refusal — same method, same shape. The refusal is
proven on the real factory wiring (offline probe, 11/11) and on the
real serve path (two-cell e2e: confound posture dies at construction,
lawful posture boots and serves).

- gate: `vllm_gaudi/xpu/patches/lmc_mx_boot_gate.py` (rev 2, md5
  `ffb1ef11b17bf7b99987c84940c16360`) — exec-hook on
  `vllm.distributed.kv_transfer.kv_connector.factory`, registered in
  `_WHEEL_HOOKS` (platform.py) with the `VLLM_ARC_LMC_MX_GATE=0`
  bypass knob (envs.py, A/B evidence only).
- image lane: cert28 (`vllm-arc:v0.30.0-b70-cert28`) with the three
  P-D files mounted over site-packages (serve script `ARC_PD_*`
  mounts). The gate rides cert29 from the build6 lineage.

## Final census

### Offline probe (r246 run 4, 13:43:59Z) — 11/11 PASS

| face | result |
|---|---|
| A_fired | hook fired on the real factory create_connector |
| B_refuse | mxfp4 refused over the real 0.30 surface |
| B_mxfp8 | mxfp8 refused |
| B_mxk8v4 | mxk8v4 refused |
| B_legacy | kv_cache_dtype surface refused (fallback chain pinned) |
| D_resolve | real registry resolves LMCacheConnectorV1 LMC-named |
| C_scope | non-LMC connector over MX passes through |
| D_lawful | LMC over checkpoint_fp8 passes through (certified lane) |
| E_bypass | VLLM_ARC_LMC_MX_GATE=0 logs gate=bypassed |
| F_stock | stock None-config ValueError intact |
| G_idem | second _install is a no-op |

### e2e (r246b run 3, R246B_DONE failed=[] 13:53:22Z)

| cell | posture | outcome |
|---|---|---|
| pd_refuse | ARC_LMC=1 over mxfp4 (the exact r240b confound) | GATE_FIRED — 5x ARC_LMC_MX_REFUSED in serve.log (worker + scheduler construction sites), 5 tracebacks, warn_line=1, boot died at construction |
| pd_clean | ARC_LMC=0 over mxfp4 | HEALTH_OK 13:52:41, gate_hook_installed=4, tracebacks=0, lmc_lines=0, CONC_OK n_ok=1 (p139 face ctx 8192 n1 mt 64) |

## Run history — every FAIL class named, none a gate-core defect

- r246 runs 1-3 (probe revs 1-3): 6/10, 7/10, 10/10 — all FAIL
  classes were probe-stub gaps (scheduler_config,
  kv_connector_module_path, engine_id), banked in the probe headers.
- r246b run 1 (13:19:34Z): pd_refuse NO_REFUSAL was a RUNNER defect
  — the marker poll broke at iteration 1 on the not-yet-created
  container and pkilled a healthy 14-second boot (banked:
  e2e/run1/, r246b_run1_launch.log). Runner rev 2 waits for the
  container, then polls marker/health/EngineCore liveness.
- r246b run 2 (13:30:43Z): pd_clean PASS; pd_refuse GENUINELY LEAKED
  — the gate fired (posture lines at worker AND scheduler sites)
  but read `kv_dtype=None`: vllm 0.30 renamed
  `CacheConfig.kv_cache_dtype` -> `cache_dtype` (config/cache.py:111,
  no legacy alias; the CLI kwarg and the AttentionSelectorConfig
  tuple field kept the old name). The probe's stub had invented the
  pre-rename attr and masked the drift (r246b_run2_launch.log holds
  the verdict summary; run 3's cells overwrote run 2's serve.log).
  Gate rev 2 reads both surfaces (legacy first, then the 0.30 field);
  probe rev 4 models the real surface and pins the chain (B_legacy).
- r246 run 4 + r246b run 3 (this bank): all green.

## Bank layout

- `probe/` — r246_summary.txt (11-face census), r246.out (full probe
  output), remote_SHA256SUMS (pre-redaction provenance).
- `e2e/pd_refuse/` — boot.txt (WARN posture line + ARC_SERVE_LAUNCHED),
  serve.log (5x refusal ValueError tracebacks), posture.txt,
  outcome.txt (GATE_FIRED).
- `e2e/pd_clean/` — serve.log, posture.txt, warmup.json/code,
  p139_conc_pd_clean.json (n_ok=1), outcome.txt (CONC_OK).
- `e2e/run1/` — the run-1 runner-defect snapshot (banked as the
  poll-fix evidence).
- `e2e/r246b_run{1,2,3}_launch.log` — the three e2e run summaries.
- `SHA256SUMS` — over the redacted bytes (host addresses byte-exact
  redacted per law 8; sweep clean).

Evidence pointers: r240b posture confound
(results/2026-11-25-p647-r240b-mxkv-posture, r240b_confound);
r241 re-measure (results/2026-11-25-p647-r241-mxkv-lmcfree, 68/68);
the 0.30 rename (config/cache.py:111 in cert28's site-packages).
