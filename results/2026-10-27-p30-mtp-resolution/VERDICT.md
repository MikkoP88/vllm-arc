# VERDICT — p30 MTP config-resolution probe (pre-leg, CPU-only)

**LEG**: 2026-10-27-p30-mtp-resolution (config dry-run, no engine boot)
**WHERE**: arc-test container (vllm-arc:v0.30.0-b70-p12), /opt/venv python

## Probe

`SpeculativeConfig(method="mtp", num_speculative_tokens=4,
model="/models/target", target_model_config=ModelConfig("/models/target",
trust_remote_code=True), target_parallel_config=ParallelConfig(tp=2))`
— i.e. exactly what the p30 `ARC_SPEC=1` CLI flag produces at engine
config time.

## Result — PASS

```
method: mtp k: 4
draft arch: ['Qwen3_5MTP']
draft model_type: qwen3_5_mtp
n_predict: 1
```

Upstream 0.30 resolves the checkpoint's embedded MTP head exactly as the
p30 README audit predicted (config/speculative.py:908-925: `qwen3_5` →
`qwen3_5_mtp`, `n_predict` from `text_config.mtp_num_hidden_layers=1`,
arch `Qwen3_5MTP` — dense branch). No draft-model path, no registration
gap, nothing to port. First GPU proof lands with the P2 spec legs
(boot + first request + p31 acceptance gate).
