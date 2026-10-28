# VERDICT — r19–r21b: c8 step decomposition. QSA row VOID on the target model; TPOT blocker = per-step CPU serialization (~100 ms), not kernels.

Postures: r19/r21/r21b = r18 product posture (graphs + MTP k4) with torch
profiler; r20 = graphs OFF (diagnostic eager). Loads: 8 streams, pybench
prompt construction; r19 8k-ctx (window caught prefill), r20/r21 1k-ctx
steady decode. Analyzers banked verbatim (raw traces stay on the rig —
inventory in trace_inventory.txt).

## Finding 1 — QSA has no call site in qwen3.8-27b-fp8 (plan P2 row amendment)

- `config.json` text_config: layer_types = 48 linear_attention + 16
  full_attention; NO index_topk_str / line_size / index_type keys.
- `vllm/model_executor/models/qwen3_5.py` (Qwen3Next lineage): zero
  indexer/qsa references. The boot's indexer_topk warning is module-import
  noise from the generic vllm tree, not this model's forward.
- The Appendix-B "Triton QSA = 58% of the c8 step" law was measured on the
  W14 flash-next posture (qwen4_exp MoE + indexer) — a different model.
  Porting `qsa_esimd.so` (p11) would ship dead code. The plan's own hedge
  ("must be re-proven on qwen3.8-27b-fp8") resolves to: VOID.
- What remains valid from that appendix row: the ESIMD toolchain laws
  (p90/p91, Appendix D) as capability, and the Triton-vs-native principle —
  but no target-model op needs it now.

## Finding 2 — the step decomposition (c8, 1k ctx)

| leg | mode | step wall | GPU busy | where the rest goes |
|---|---|---|---|---|
| r20 | eager | ~540 ms | ~135 ms | ~1870 launches/step × 0.05–0.2 ms tax (diffuse; only 15 gaps >1 ms) |
| r21 | graphed | ~87 ms | ~4 ms visible (forward inside graph) | 96% idle: 3×~21 ms pre-`VectorizedGatherKernel` stalls + 3.8 ms pre-`_prepare_decode_inputs` |
| r21b | graphed+stacks | — | — | stalls named: **~54 ms `Event.synchronize`** at `graph_safe_output.py:1075 get_output` (our defect-b hygiene patch) + **~50 ms `gdn_attn.py:209 build` / `build_attn_metadata` / `mamba_hybrid.prepare_attn`** CPU build, serialized per step |

r19 (prefill window): eager prefill gemms 3–24 ms each, ~343/pass,
~2.1 s GPU per 8192-token pass — the TTFT face (ours 5794 ms @8k-c1 vs
champion 3944); separate from TPOT.

## Implications

1. **TPOT c8 ≤ 60 ms gate**: pure CPU-serialization problem. Remove the
   ~100 ms serial CPU section → step ~30–40 ms GPU-bound → gate met at any
   acceptance ≥ 1. Levers: (a) **p10** — port radiance's newer-fixed GDN
   attention path (plan §12 step 2 already orders it; the 50 ms/step
   `build_attn_metadata` is exactly the shared GDN-path code radiance
   newer-fixed); (b) overlap the output-copy sync (get_output lane).
2. **Solo/agg contract bars**: at step 30–40 ms and acceptance ~4.9
   (run7n counting prompt), per-token ≈ 6–8 ms → solo ≈ 125–160 tok/s —
   the 79.74/274.41 bars are reachable WITHOUT any kernel lane.
3. Champion step×acc law (104–109 ms @ acc≈1.8) ⇒ champion step ≈ 58 ms:
   our 87 ms gap ≈ the CPU serial section. Our acceptance telemetry is
   healthy (r18: pos1 0.8756) — acceptance is not the gap driver at fixed
   prompts.
4. Eager posture remains launch-bound (r20) — FULL_DECODE_ONLY graphs stay
   load-bearing (defect-a/b fixes + launch collapse).

## Disposition

- P2 row amended by evidence: QSA ESIMD → VOID (no call site); the TPOT c8
  row stays OPEN with the CPU-serialization lever (p10 + output-overlap).
- p11/p90/p91: parked (capability only, no target-model surface).
- arc4 vxk wheel (P3 precondition) unaffected by this finding.
- Next legs: p10 port → r22 re-run of the r18 gate cells (TPOT c8, spec
  gate, run7n) on the ported build.

## Artifacts (legs_pull/)

r19_prefill_table.txt + r19_prefill_analyze.txt (prefill window);
r20_eager_analyze.txt + r20_eager_gaps.txt (eager decode: launch-bound);
r21_graphed_gaps.txt (96% idle, gather stalls); r21_cpu_in_stalls.txt
(aten::to/copy_/index chain); r21b_python_stalls.txt (the named frames);
r21b_graphed_breakdown.txt; trace_inventory.txt (rig trace paths);
analyzer + serve + driver scripts (r19_analyze.py, r19_gaps.py,
r21_cpu.py, r21b_stacks.py, r19/r20/r21b_serve.sh, r19_load.py,
r19/r20/r21_prof.sh).
