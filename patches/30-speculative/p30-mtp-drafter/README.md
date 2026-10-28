# p30 — MTP k4 drafter posture (qwen3.8-27b-fp8): wiring, NOT a port

## Outcome first

The plan item (#23 / p30: "qwen3.8 MTP head k4 + spec config parity (CLI, not
env)") was scoped as a PORT of "the MTP k4 head + its registration" from the
D lane. Audit on the v0.30.0 tag (2026-10-27, arc-test venv) shows **there is
nothing to port** — the head, its registration, and the speculative-config
resolution are all upstream-native on this tag. p30 therefore lands as:

1. `tools/serve_arc_b70.sh`: the `ARC_SPEC=1` posture — appends the
   D-champion spec CLI flag (`--speculative-config
   '{"method":"mtp","num_speculative_tokens":4}'`, verbatim from rig
   `/root/serve_lmc_v133.sh:53`); spec-refusal relaxed (validation IS the
   §10 P2 spec-leg row); spec-eager A/B combo note.
2. This assessment (evidence below), mirroring the plan's own naming-audit
   correction for item #23's model side.

## Evidence: upstream v0.30.0 already carries the whole MTP surface

All verified in the arc-test venv image `vllm-arc:v0.30.0-b70-p12`
(`/opt/venv/lib/python3.12/site-packages/vllm`):

- **Head implementation**: `model_executor/models/qwen3_5_mtp.py` —
  `Qwen3_5MultiTokenPredictor` (:67), `Qwen3_5MTP` (:217, LocalArgmaxMixin +
  SupportsMultiModal + SupportsPP), `Qwen3_5MoeMTP` (:326).
- **Registration**: `model_executor/models/registry.py:699-700` —
  `"Qwen3_5MTP": ("qwen3_5_mtp", "Qwen3_5MTP")`, `"Qwen3_5MoeMTP"`.
- **Spec-config resolution**: `config/speculative.py:908-925` — model_type in
  (`qwen3_5`, `qwen3_5_moe`, `qwen3_5_text`, `qwen3_5_moe_text`) →
  draft model_type `qwen3_5_mtp`, `n_predict = mtp_num_hidden_layers`
  (top-level, falling back to `text_config`), architectures
  `["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"]`. `"mtp"` and
  `"qwen3_5_mtp"` are both legal method values (`MTPModelTypes`,
  `config/speculative.py:37-64`).
- **Checkpoint shape matches the resolution exactly**
  (`/models/target/config.json`): `model_type: qwen3_5`,
  `architectures: [Qwen3_5ForConditionalGeneration]`, top-level
  `mtp_num_hidden_layers: null` → `text_config.model_type: qwen3_5_text`,
  `text_config.mtp_num_hidden_layers: 1`; head weights embedded in the same
  checkpoint (22 `mtp.*` entries in the safetensors index, e.g.
  `mtp.fc.weight`, `mtp.layers.0.*`). No draft-model path needed.

Expected spec-boot log line: draft architecture `Qwen3_5MTP`, `n_predict=1`
draft layers, k=4 speculative tokens (the head loops).

## D-lane MTP artifacts: disposition (scope law §3)

| Donor artifact | Disposition |
|---|---|
| `flashnext/wave_fn6_mtp/` (`qwen4_exp/xpu/mtp.py`, `patch_fn6_spec_mtp_register_026.py`) | **OUT of scope** — flash-next MoE lane (qwen4_exp family) |
| `waveA_model_agnostic/patch_mtp_mm_mask_026.py` (iexp-a5) | **Not needed** — multimodal placeholder-mask fix; qwen3.8-27b legs are text-only (no MM embeds → the IndexError path is unreachable) |
| `waveA_model_agnostic/patch_mtp_loopbreak_026.py` (iexp-a6) | **Not needed** — inert needle (`getattr(self,"_iexp_stop",False)`), no controller exists on this stack; lands ahead of a consumer that never arrives |
| `waveE_moe_quant/patch_quark_bf16_mtp_026.py` | **OUT of scope** — MoE quant lane |

## Validation plan (P2 spec legs, with p32 in the image)

1. `ARC_GRAPH=1 ARC_SPEC=1` boot — expect: `Qwen3_5MTP` draft resolution in
   the log, coercion log from p32 if any manager path receives a
   PIECEWISE-family mode, boot survives to healthy, first request completes.
2. Acceptance telemetry per p31 (pos1 hard ≥ 0.5, target ≥ 0.85) — the D
   champion acceptance reference for this model+k is the superiority-contract
   acceptance ≥ 0.85 row.
3. compile×spec A/B (TORCH_COMPILE_DISABLE) — falsified-path #1 REVAL.
