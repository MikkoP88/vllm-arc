# p01 — fork bootstrap: dormant HPU + xpu/ tree + ArcXpuPlatform

## What

- `vllm_gaudi/__init__.py`: `_HPU_AVAILABLE` runtime gate
  (`importlib.util.find_spec("habana_frameworks")`). `register()`,
  `register_utils()`, `register_ops()` return early on hosts without a Gaudi
  runtime — `register()` returns `None` so vLLM's
  `resolve_current_platform_cls_qualname()` keeps resolving.
- `vllm_gaudi/models/__init__.py`: the HPU-override registrations
  (habana-importing, eager) move into `_register_hpu_overrides()` behind the
  same gate; `register_model()` keeps the device-neutral imports
  (`models.utils`, `interfaces`, `bert`). No XPU model override at this
  layer — the target model (qwen3.8-27b-fp8, Qwen3_5 hybrid family) is
  served by core's own implementation on XPU (radiance leg-B-proven on the
  0.30 stack, `vllm_gaudi/models/qwen3_5.py` has zero delta since fork
  point).
- `vllm_gaudi/xpu/` (NEW tree): device-wide XPU home
  (`platform.py` = `ArcXpuPlatform` + `is_xe2_arch()` probe).
- `setup.py`: `arc = vllm_gaudi.xpu:register` entry point in the
  `vllm.platform_plugins` group alongside upstream's `hpu`.

## Why

The fork must install and load on Intel Arc (XPU) hosts where
`habana_frameworks` does not exist, while staying a first-class vllm-gaudi
fork on Gaudi hosts. Core's plugin loader calls general-plugin entry points
without exception handling, so any ungated `import habana_frameworks` at
registration time kills `import vllm`. The `None`-return mechanism is core's
own "not activated" path for platform plugins.

## Provenance

- Dormant-HPU pattern + gated `_register_hpu_overrides`: radiance fork
  `vllm_gaudi/__init__.py` / `vllm_gaudi/models/__init__.py` at branch
  `radiance` HEAD c24eeed (donor, entered as files with provenance — fork
  model law §13). Proven serving vLLM 0.30.0 on the dual-B70 rig
  (`vllm-gaudi-radiance:v0.30.0` image).
- Platform resolution mechanics: core 0.30 `vllm/platforms/__init__.py`
  `resolve_current_platform_cls_qualname()` (OOT plugin wins over builtin
  when exactly one activates).

## Validation

- P0/P1 rig leg: `results/2026-10-27-p1-platform/` (see TEST.md).
