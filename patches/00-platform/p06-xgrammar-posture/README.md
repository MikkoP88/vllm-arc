# p06 — xgrammar posture (VLLM_ARC_XG_COMPACT)

## What

`tools/serve_arc_b70.sh`: `VLLM_ARC_XG_COMPACT=1` appends
`--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'`
to the serve command. Default off. The env var itself is registered in
`vllm_gaudi/envs.py` (p01 registry block).

## Why

Port of D-champion's `V1227_XGCOMPACT` serve append (v128 WS-D fix): guided
JSON grammars compiled whitespace-free — kills the whitespace-attractor
degeneration class by construction; JSON output becomes compact. Core 0.30
carries the identical surface natively (`vllm/config/structured_outputs.py`:
`StructuredOutputsConfig.backend` + `disable_any_whitespace`, verified on the
rig venv), so the knob is pure serve-interface wiring — no platform code.

## Validation

results/2026-10-27-p1-platform/ — XG leg boot log carries the structured
outputs config line; a guided-JSON probe returns compact output.
