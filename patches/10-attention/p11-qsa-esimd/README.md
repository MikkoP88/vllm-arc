# p11 — QSA ESIMD kernel: VOID for the target model (architecture audit)

## Outcome

Plan p11 ("QSA ESIMD kernel — width-2051/PAGE-1664/lsc≤64 laws", item #16,
§10 P2 row "QSA ESIMD engaged: ≥ 7× vs Triton") assumed qwen3.8-27b-fp8
exercises the QSA sparse-attention path ("same shared GDN-path op, different
model"). **It does not.** The target model's architecture carries no QSA
layers; the op is exclusive to the qwen4_exp flash-next family — the lane
§3 excludes. Nothing to port; the P2-row QSA cell is void for this model.

This is the same class of correction as the plan's own item-#23 naming
audit ("CORRECTED 2026-10-27: NOTHING to port"): a plan-time belief about
which shared component the target model exercises, amended by checkpoint
ground truth.

## Evidence chain (all 2026-10-27)

1. **Checkpoint ground truth** (`/models/target/config.json` text_config,
   probed in arc-test): `layer_types = {linear_attention: 48,
   full_attention: 16}` — dense hybrid-GDN. No `indexer`, no sparse/topk
   field, no `output_width`. Geometry: hidden 5120, 64 layers, 24 q-heads /
   4 kv-heads, head_dim 256, linear key/value head dims 128.
2. **Core 0.30 model**: `model_executor/models/qwen3_5.py` and
   `qwen3_5_mtp.py` contain zero indexer/qsa/topk symbols — the dense
   hybrid has no code path that could call a QSA op.
3. **D's own layer map** (iexperimental MIGRATION_026.md:2619):
   "GDN 36, QSA 12 (3,7,…,47)" — measured on `/models/flash-next-mxfp4-gptq`
   (the qwen4_exp flash-next model, D wave0 lane). The 12 QSA layers and
   the 2051-width indexer belong to THAT model.
4. **Radiance donor placement**: the QSA implementation lives at
   `vllm_gaudi/models/qwen4_exp/xpu/{qsa.py, indexer_qsa.py, ops/qsa.py}`
   — the flash-next family tree (plan §638/§842 source map; §3 scope law
   excludes the flash-next MoE lane).
5. **D champion never used QSA for qwen3.8**: serve_lmc_v133/v134 (the
   274.41 agg champion posture) set no QSA env/flag — grep of the D serve
   surface is empty. The champion numbers stand without any QSA
   engagement.
6. **Where the 134 µs law was measured**: W14 build10 on the radiance-test
   incumbent serving the flash-next model with `IEXP_QSA_ESIMD=1`
   (probe26: rows=4 134 µs vs Triton 1001-1387 µs, PAGE=1664). Valid law —
   for a layer the target model does not have.

## Consequences

- **p11**: closed as VOID-for-target (this assessment). The ESIMD kernel
  laws corpus (512B q-head, GRF spill <4KB, uniform-token blindness,
  grow-only scratch, graph-capturable lsc-gather recipe — Appendix D)
  remains banked donor knowledge for any future shared-op work; nothing
  enters this fork.
- **§10 P2 row interpretation**: the operative cells are defects (a)+(b)
  fixed; FULL graph + MTP boot/first-request survival; acceptance
  telemetry live (p31, wired); TPOT c8 ≤ 60 ms. The "QSA ESIMD ≥ 7×" cell
  is struck for qwen3.8-27b-fp8 — it cannot bind on a path the model never
  executes. (TPOT ≤ 60 ms is carried by the graph+MTP work: p12/p32 + the
  P2 spec legs.)
- **p10** remains as resolved by the p12 part-2 assessment: the GDN
  attention path for qwen3_5 dense hybrids is core-0.30-native (superset
  of the donor GDN code; no port).
