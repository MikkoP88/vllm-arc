# p10 — GDN attention path for qwen3_5 dense hybrids: core-native, no port

## Outcome

Plan p10 ("GDN attention path for qwen3_5 hybrid archs (dense)") closed as
**nothing to port** — the same verdict class as item #23's naming audit and
p11's architecture audit, with the evidence carried by the p12 part-2
assessment (patches/10-attention/p12-gdn-spec-state-fix/README.md, "Scope"):

- Core 0.30 `gdn_attn.py` (548 L) is an evolved SUPERSET of the donor
  reference (`src_v126_gdn_attn.py`, 475 L): same persistent-buffer
  spec-state machinery (`spec/non_spec_state_indices_tensor`, `token_indx`,
  `query_start_loc` pre-sized by `decode_cudagraph_max_bs`, non_blocking
  `copy_` fill, `NULL_BLOCK_ID` padding) plus newer surface the donor lacks
  (chunk-prefill metadata: `prefill_query_start_loc` /
  `prefill_state_indices` / `prefill_has_initial_state`;
  `gdn_prefill_backend` resolution triton|flashinfer|cutedsl). The donor is
  the ancestor lineage — porting backwards would drop functionality.
- Side-by-side mirrored at `results/2026-10-27-p2r3e-zombie-noshow/ref/`.

## Residual work (moved to the P2 spec legs)

At the first MTP leg: verify `spec_state_indices` reach the kernel with
correct geometry under k4 spec (k1k spec cells + golden-logit spot check);
any defect found gets its own patch item. The target-attention defects that
DID exist on this stack are tracked as p12 (defect-b capture-zombie family)
and p32 (defect-a speculator piecewise boot death) — both fixed on this
train.
