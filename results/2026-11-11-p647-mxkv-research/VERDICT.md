# P6.47 research bank — MXFP4/MXFP8 KV-cache-dtype deep survey

**Date:** 2026-11-11 · **Phase:** P6.47 (owner directive: add MXFP4 and
MXFP8 supports on `--kv-cache-dtype`, full testing/validation on
google/gemma-4-31B + Qwen/Qwen3.8-27B, telemetry-driven, cert image at
the end, all docs updated, corruption/garbage gates end-to-end)

## VERDICT: lane justified as FROM-SCRATCH IN-HOUSE (negative-result survey + template identified)

- **No lawful source carries MX KV-cache support.** The pinned cert23
  wheel's `kv-cache-dtype` Literal has no mxfp4/mxfp8 (survey doc has
  the full menu); upstream vllm-gaudi v0.30.0, line C donors
  (llm-scaler-exp), line D donors (iexperimental + twin), and radiance
  are all negative on MX KV (radiance's newer-fixed shared components
  cover QSA/GDN/graph laws/mem-info only — scope law respected).
- **Template identified:** the wheel's own sub-byte per-token-head KV
  modes — `turboquant_*` AoS store/decode Triton kernels +
  `int4_per_token_head` — define the integration surface (paged uint8
  cache `[num_blocks, block_size, Hk, slot]`, in-kernel pack/unpack,
  arg_utils branch, backend dispatch). 7 reference source copies in
  `wheelref/` (extracted via docker create + export; no container
  started, W14 untouched).
- **Design derived** (OCP MX, per-32-group E8M0 scale; mxfp8 slot 264 B
  and mxfp4 slot 136 B at D=128 vs 256 B certified fp8 KV; key path =
  direct scaled float, no rotation/centroids/zero-point): documented in
  `survey/WHEEL-KVDTYPE-SURVEY.md` §design and the module headers of
  `vllm_gaudi/xpu/patches/mx_kv/`.

Follow-on: leg A (offline XPU validation of the implemented primitives)
banked in `../2026-11-11-p647-r147-mxkv-lega/` — **PASS**.
