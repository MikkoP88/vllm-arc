# P6.47 r148 — MX KV in-engine admission gate (leg B1)

**Date**: 2026-11-11 · **Rig**: dual Arc Pro B70 (TP2) · **Image**:
`vllm-arc:v0.30.0-b70-cert23` + `/work` PYTHONPATH shadow (P6.47 tree)
· **Probe**: `tools/ar/p148_mxkv/p148_admission.py` (one-shot, no serve)

## VERDICT: PASS — 14/14 faces

| # | Face | Result | Evidence |
|---|------|--------|----------|
| 1 | platform | ok | `current_platform=ArcXpuPlatform` |
| 2 | literal | ok | `mxfp4`/`mxfp8` true members of `CacheDType` **(source transform)** |
| 3 | admit | ok | `CacheConfig(cache_dtype=mxfp4/mxfp8)` admitted + round-trip |
| 4 | dtype_map | ok | `STR_DTYPE_TO_TORCH_DTYPE`: both MX dtypes → `uint8` |
| 5 | mode_alias | ok | mxfp8→`TURBOQUANT_K8V4`, mxfp4→`TURBOQUANT_4BIT_NC` (bounded element-class alias) |
| 6 | enum_member | ok | `AttentionBackendEnum.MXKV` == backend qualname, both `.value` and `get_path()` **(register-time injection)** |
| 7 | routing | ok | `get_attn_backend_cls` → `vllm_gaudi.xpu.patches.mx_kv.engine.MxAttentionBackend` |
| 8 | stamp | ok | `active_kv_dtype=mxfp8` |
| 9 | slot (mxfp8) | ok | `state_content_bytes=264` (2D + D/16 @D=128) |
| 10 | page (mxfp8) | ok | `page_size_bytes=135168` = 64 block × 8 heads × 264 B |
| 11 | slot (mxfp4) | ok | `state_content_bytes=136` (D + D/16) |
| 12 | page (mxfp4) | ok | `page_size_bytes=69632` = 64 × 8 × 136 |
| 13 | tq_spec_fix | ok | P6.19 impl-forward patch carries over (`impl_forward_patched=True`) |
| 14 | engine_import | ok | `mx_kv.engine` + `mx_dequant` import clean |

## Naming law (owner directive 2026-11-11)

MXFP4/MXFP8 are their **own backend family**, not TurboQuant. The lane
now carries a true `AttentionBackendEnum.MXKV` member whose value is the
MX backend qualname (enum-native `get_class()` lands on the real backend;
`get_name()` consumers are log/error strings only — selector.py:238,
composite.py:179/181/490, cp_utils.py:38, speculator.py:112; nothing
iterates the enum; nothing branches on `== TURBOQUANT` for compute).

Two-path implementation:
- **Source transform** (`_add_mxkv_backend_member`) — registry.py compiles
  with the member; late-import case.
- **Register-time injection** (`_install_mxkv_enum_member`) — the primary
  path: the registry module is ALWAYS in `sys.modules` before hooks
  install (the platform's module-scope `from vllm.platforms.xpu import
  XPUPlatform` pulls it — rig-proven). Recipe: register in
  `_member_map_` + `_value2member_map_` only; `EnumType.__getattr__`
  serves attribute access and `["MXKV"]`; plain `setattr` on the class is
  blocked ("cannot reassign member") and unnecessary.

## Regression found + fixed (run 2, face evidence in raw/)

Adding the registry entry to `_SOURCE_XFORMS` broke the cache Literal
transform (literal+admit FAIL):
1. Platform registers mid-import of `vllm.utils.torch_utils` (plugin-load
   circular window).
2. The fallback trigger `any(t in sys.modules for t in _SOURCE_XFORMS)`
   became True via the pre-imported registry → premature
   `_install_mx_dtype_admission_fallback`.
3. Its `import vllm.config.cache` executed cache.py fresh inside the
   window; cache.py's own from-import died on the partial torch_utils.
4. The finder's find_spec-time `pending.discard` (xform-only targets)
   un-claimed cache.py → retry imported it **pristine** → no widening,
   raw pydantic ValidationError.

Fix set (all landed): (A) fallback trigger scoped to exactly
`"vllm.config.cache"`; (B) retry-safe pending — discard moved into an
exec_module wrapper for transform-only targets; (C) the register-time
enum injection. Run 3 = 14/14.

## Files

- `raw/legB1_summary.txt` — face summary (run 3)
- `raw/legB1_run3.out` — full probe stdout incl. platform install lines
  (`source-transform targets already imported at install: ['vllm.v1.attention.backends.registry']`,
  `injected AttentionBackendEnum.MXKV -> ...`)
- `raw/r148_run3.log` — r148 script run log
- `raw/r148_run2_regression.log` — run-2 failure summary (regression)
- `raw/rig_SHA256SUMS.txt` — rig-side checksums
