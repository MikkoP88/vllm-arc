# r253 — P-H4 posture-aware LMC x MX gate flip (P6.47 fix phase)

**VERDICT: PASS — 15/15 probe faces green (rev 2, 2026-11-26T21:45:18Z,
rc=0).** The P-D LMC x MX boot gate is now posture-aware in the
cert28 + overlay probe class: the r249 packed-row interop (capability
marker `ARC_LMC_MX_INTEROP` on the fork-owned `lmcint` adapter) admits
the LMCache-over-MX lane with `gate=interop-live` telemetry, and its
absence keeps the P-D refusal (`ARC_LMC_MX_REFUSED`) — pre-r249
installs stay protected. This leg is the validating evidence for the
P-H4 delta that rides the cert29 content commit.

## Leg definition

- Image `vllm-arc:v0.30.0-b70-cert28`, no engine boot — the gate fires
  at connector construction, so the probe (p246 pattern) drives the
  REAL wiring: platform `_install_wheel_hooks` -> meta-path finder ->
  factory import -> exec-wrap -> `KVConnectorFactory.create_connector`.
- Overlay: the four patch files over site-packages (stage md5s in
  `stage_provenance.txt`), probe mounted read-only.
- Runner `tools/ar/r253_ph4_gate_flip_offline.sh`, probe
  `tools/ar/p147_mxkv/p253_ph4_gate_flip_probe.py` (both banked as
  `.shipped`).

## The 15 faces

| face | result | proof |
| --- | --- | --- |
| A_fired | ok | hook fired on the real factory `create_connector` |
| R_bootstrap | ok | real registry resolution ran first — the p41b exec-wrap installed the lmcint src-override finder; the overlaid kit adapter then imported clean, carrying the marker |
| H_live_mxfp4 / _mxfp8 / _mxk8v4 | ok x3 | interop PRESENT + LMCache over each MX dtype -> pass-through with `gate=interop-live` — never refused |
| B_stock_mxfp4 / _mxfp8 / _mxk8v4 | ok x3 | interop ABSENT (marker deleted on the live module) -> `ARC_LMC_MX_REFUSED` ValueError per dtype |
| B_legacy_stock | ok | same refusal over the pre-rename `kv_cache_dtype` surface — the fallback chain stays pinned under the new posture logic |
| D_resolve | ok | the REAL registry entry resolves `LMCacheConnectorV1 -> LMCacheConnectorV1` (detection basis intact) |
| C_scope | ok | non-LMC connector over MX passes through, `gate=on` — refusal scope is LMCache-named connectors only |
| D_lawful | ok | LMCache over `checkpoint_fp8` passes through and constructs — the certified LMC lane untouched |
| E_bypass_stock | ok | knob `VLLM_ARC_LMC_MX_GATE=0` + interop absent -> `gate=bypassed` posture line noting `interop=absent` (A/B evidence only) |
| F_stock | ok | stock None-config ValueError intact |
| G_idem | ok | second gate `_install()` is a no-op (exactly one wrapped constructor) |

Full per-face lines: `r253_summary.txt`; complete probe log
`r253_run2.out`.

## Run-1 lesson: the kit import is resolution-ordered (probe law banked)

Run 1 (rev 1, 18:39:54Z) died rc=1 right after face A. The staging
import — `import vllm_gaudi.xpu.patches.lmcint.vllm_v1_adapter` — was
the first thing to touch the kit, and the kit adapter's line-55 import
of the vendored `lmcache_integration.utils` executed the STOCK
`lmcache_integration/__init__.py`, which hard-imports
`multi_process_adapter` -> `CudaIPCWrapper` from
`lmcache.v1.multiprocess.custom_types` — a symbol that exists only on
lmcache main; cert28 pins lmcache 0.5.5 (the kit `__init__.py` header
documents exactly this skew):

```
ImportError: cannot import name 'CudaIPCWrapper' from
'lmcache.v1.multiprocess.custom_types'
```

Root cause: import ORDER, not bytes. The engine never imports the kit
path bare — the registry RESOLUTION first imports
`vllm...v1.lmcache_connector`, whose exec-wrap hook
(`lmcache_hma_connector`, p41b) installs the `_SrcOverrideFinder`
that maps the three vendored `lmcache_integration` names onto the kit
files (verified empirically in cert28: the three
`[...]@lmcint src-override OK` lines fire during resolution, and the
kit path then imports clean). The P-H4 gate is safe by construction:
its `create_connector` wrap resolves the connector class BEFORE the
posture branch, so by the time `_interop_live()` imports the kit
path the redirect is live on every construction path that can reach
the posture branch at all (a failed resolution means the stock
constructor would die on the same defect immediately after).

Rev 2 adds the R_bootstrap face — the real
`factory.get_connector_class` resolution before any kit-path import —
making the probe exercise the engine's own import order instead of a
probe-invented one. Run 2: 15/15 green.

## Shipped semantics (the P-H4 delta)

- `vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py`:
  `ARC_LMC_MX_INTEROP = "packed-row/r249"` capability marker (module
  attribute; the r249 INTEROP_PASS evidence: re-hit A/A byte-equal at
  6.18 -> 1.89 s wall, the exact r240b kill sequence clean, zero
  transfer errors).
- `vllm_gaudi/xpu/patches/lmc_mx_boot_gate.py`: `_interop_live()`
  reads the live adapter module (import failure and missing marker
  both read as absent); the three-way branch — knob bypass /
  interop-live allow with telemetry / refusal with the updated
  message (cert29+ image or `ARC_LMC=0` for MX lanes; knob for A/B
  evidence only).
- `tools/serve_arc_b70.sh` Phase-6 auto-off review rides the same
  content commit: the default `ARC_LMC=0` on MX lanes now rests on the
  r176 perf posture only (ITL unchanged, 26/259 ms TTFT store tax
  saved; worker + blob freed) — the r168 correctness reason is
  REPEALED at P-H2/r249, and explicit `ARC_LMC=1` is legal (r249
  re-hit win 6.18 -> 1.89 s wall on warm faces). The WARN became a
  NOTE; the `NOTE: ARC_LMC auto-off` prefix is preserved (the cert
  gate runners grep it).

## Banked files

| file | content |
| --- | --- |
| VERDICT.md | this verdict |
| r253_summary.txt | the 15-face summary (rig pull) |
| r253_run2.out | full probe log, run 2 (rig pull) |
| launch2.txt | runner launch log with R253_STAGE_OK / R253_DONE rc=0 / W14_RESTORED |
| SHA256SUMS.rig | rig-side sha256 of summary + full log |
| stage_provenance.txt | the five staged md5s + roles + runner description |
| r253_ph4_gate_flip_offline.sh.shipped | the runner bytes |
| p253_ph4_gate_flip_probe.py.shipped | the probe bytes (rev 2) |
| SHA256SUMS | local sha256 over the bank |

Run-1 artifacts were overwritten by run 2 at the same rig paths
(`/root/build/p253/logs/r253.out`); the run-1 traceback is quoted
verbatim above from the session transcript.

## Disposition

P-H4 CLOSED. The posture-aware gate + the Phase-6 auto-off review are
the cert29 content commit; the probe face census pins both postures
against future drift (interop-live must never refuse; absent-interop
must never pass silently). Posture telemetry (`ARC_LMC_MX_POSTURE
gate=...`) is the cert29-gate grep surface for LMC x MX boots.
