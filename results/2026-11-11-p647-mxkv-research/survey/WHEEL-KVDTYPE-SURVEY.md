# P6.47 wheel KV-dtype survey — cert23 image, 2026-11-11

Method: `docker create vllm-arc:v0.30.0-b70-cert23` → `docker export`
(no container started; W14 untouched) → extraction of the pinned wheel
(`opt/venv/lib/python3.12/site-packages/vllm`) + the in-image wheel
source tree (`/llm-scaler/vllm`). Reference copies: `../wheelref/`.

## The kv-cache-dtype menu in the pinned wheel (config/cache.py Literal)

```
auto, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla, nvfp4_ds_mla,
turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc,
int4_per_token_head, int8_per_token_head, fp8_per_token_head,
nvfp4, nvfp4_4over6
```

**No `mxfp4` / `mxfp8` anywhere in the wheel.** `nvfp4`/`nvfp4_ds_mla`
are Literal entries with ZERO implementation surface in
`v1/attention/` or `model_executor/layers/quantization/` (dead entries
in this wheel build). `int4_per_token_head` is a complete
self-contained Triton mode (nibble pack, RHT, write + read kernels)
dispatched from `triton_unified_attention.py` — the wheel's own
template for a sub-byte per-token-head KV mode.

## Donor / upstream survey (all READ-ONLY, 2026-11-11)

| source | mxfp hits | KV-cache relevance |
|---|---|---|
| line C `C:/Projects/llm-scaler` (+ `llm-scaler-final`) | gpt-oss MoE weight-quant backports (`vllm-xpu.patch` #34), ESIMD W8A16 | none — no KV-dtype mxfp code |
| line D `C:/Projects/vllm-iexperimental` | none | none |
| radiance `C:/Projects/vllm-gaudi-radiance` | inherited HPU gpt-oss files + qwen4_exp flash-next lane (out of scope) | none |
| upstream vllm-gaudi v0.30.0 (pristine mirror) | `ops/hpu_mxfp4.py` (gpt-oss MoE, HPU) | HPU platform admits only `fp8_inc` KV (`platform.py:573`) |

**Verdict: no lawful donor or upstream source carries MXFP4/MXFP8
KV-cache support. The lane is a from-scratch in-house implementation
(owner directive 2026-11-11), on the wheel's own sub-byte KV-mode
template (TQ AoS kernels + int4_per_token_head), injected via the
fork's `_WHEEL_HOOKS` exec-wrap (pins untouched).**

## The TQ machinery the MX lane models on (wheelref/)

- `ops/triton_turboquant_store.py` (447 ln): per-(token,head) slot in a
  uint8 paged cache `[num_blocks, block_size, Hk, padded_slot]`; key =
  fp8-e4nv cast or MSE centroid path; value = uniform min/max VQB-bit
  with fp16 scale+zero inline in the slot.
- `ops/triton_turboquant_decode.py` (630 ln): paged decode, inline
  unpack/dequant, online softmax, split-K combine.
- `backends/turboquant_attn.py` (1243 ln): the attention backend; arg_utils
  imports `TurboQuantConfig.get_boundary_skip_layers` (dense: n=2 ends;
  hybrids: disabled).
- Fork patches already prove the pattern on XPU: `tq_spec_fix` (graph
  capture safety) + `tq_mq_aos` (MQ verify kernel, AoS layout).

## MX design mapping (OCP MX, from-scratch)

- MXFP8 KV: e4m3 elements + E8M0 scale per 32-element group;
  slot = 2·D + D/16 bytes (D=128 → 264 B ≈ fp8's 256 B; the win is
  per-group vs per-tensor scaling = accuracy).
- MXFP4 KV: e2m1 elements + E8M0/32; slot = D + D/16 bytes
  (D=128 → 136 B = 1.88× fp8 capacity, 3.76× fp16 — TQ-3bit capacity
  class with standardized float dynamics, no zero point, no rotation,
  no centroid tables — simpler than the TQ-MSE key path).
- Key path = direct scaled float (the TQ fp8-key template, no RHT);
  dequant = `2^(e8m0−127) · e2m1/e4m3(bytes)`, e2m1 decode arithmetic
  (exp/man bits), e4m3 via Triton fp8 bitcast.
