# P6.27 — MXFP8 validation: intel-ai AutoRound checkpoint (2026-11-06)

Directive (owner, 2026-11-06): full MXFP8 testing/validation on
`intel-ai/Qwen3.8-27B-MXFP8-MXFP8MTP-CT-AutoRound`; deep-analyze any
issue, fix start-to-end if needed, certify, update docs. Scope-law
basis (plan §P6.27): owner-named model route; same arch family as the
target (`Qwen3_5ForConditionalGeneration` text config, dense hybrid-GDN,
MTP drafter in-checkpoint and quantized).

Harnesses: `r106_mxfp8_boot.sh` (phase A, isolation posture),
`r106b_mxfp8_phaseB.sh` (phase B, full certified posture). Both legs on
the **certified cert11 image** (`vllm-arc:v0.30.0-b70-cert11` — the
overlay; no engine change was needed at any point). Model staged at
`/models/qwen3.8-27b-mxfp8` (18 shards, 29.46 GiB checkpoint).

## Outcome: ALL GATES GREEN, zero fixes needed — lane rides cert11

The r106 pre-boot recon (plan §P6.27) predicted no platform rejection
and the native kernel path; the boots confirmed everything on the
first try. **No engine/wheel byte changed → no cert12** (per plan G3:
config-only lanes ride the incumbent cert, P6.26 precedent).

### Phase A (isolation: eager, spec 0, LMC off) — G0 PASS

- `quantization=compressed-tensors` resolved config-driven (no flag).
- **`[Worker_TP0] Using XPUMxFp8LinearKernel for MXFP8 GEMM`**
  (`kernels/linear/__init__.py:923`) — the exact pre-registered
  dispatch-line proof; native oneDNN `fp8_gemm` lane live.
- Weights 14.74 GiB/rank in 21.1 s (18 shards); health at ~365 s
  (isolation posture, no graph).
- Generation probe coherent (reasoning-channel format identical to the
  fp8 incumbent). ERROR=0 / Traceback=0. Prediction **P-1 refuted**
  (no param mismatch at load).

### Phase B (full certified posture: graph=1 FULL_DECODE_ONLY, MTP k3
probabilistic, LMCache on) — G0/G1/G2 PASS

| telemetry | value |
|---|---|
| health | **285 s** (incumbent-family identical) |
| kernel dispatch | `XPUMxFp8LinearKernel` (both postures) |
| graph posture | FULL_DECODE_ONLY, 71 sizes captured — **P-3 refuted** (no eager fallback; `xpu_mxfp8_quantize` dynamic op replays in-graph) |
| MTP drafter | `Qwen3_5MTP` resolved, **quantized drafter loads + drafts** |
| KV pool | 6.25 GiB (weights 14.74 GiB/rank leave less headroom than fp8's 13.6) |
| erraudit | 0 ERROR / 0 Traceback |

**G1 quality (t0, 3 probes)**: factual/reason/code all coherent; the
reason probe computes the train-catch time correctly (6pm) inside the
reasoning channel; same finish=length-on-short-budget behavior as the
incumbent (`phaseB/mxfp8/g1_probes.txt`).

**G1 spec gate (k3 probabilistic, quantized drafter — prediction P-2)**:
**PASS** — pos1 **0.9626**, acceptance **0.9174** (target 0.85, hard
0.5), mean acceptance length 3.75. vs fp8 incumbent 0.9258 pos1 and
mxfp4 lanes 0.9853/0.9951: the quantized MTP drafter is comfortably
above target, slightly below the mxfp4 lanes' acceptance.

**G2 perf faces (p65 timeline, full posture)** — REPORT matrix:

| face | fp8 incumbent (r105-corrected anchors) | mxfp8 | ratio | rhmxfp4 | qurk |
|---|---|---|---|---|---|
| U32k TTFT | 17.715 s | 26.67 s | 1.50× | 29.01 (1.64×) | 32.28 (1.82×) |
| C32k TTFT | 1.21 s | 1.52 s | 1.26× | 1.68 (1.39×) | 3.84 (3.17×) |
| U8k TTFT | 3.78 s | 6.36 s | 1.68× | 7.37 (1.95×) | 7.73 (2.04×) |
| U64k TTFT | 39.025 s | 59.52 s | 1.53× | 62.46 (1.60×) | 70.61 (1.81×) |
| bystander decode p50 | 53.5 ms | **81.8 ms** | **1.53×** | 64.7 (1.21×) | 68.8 (1.29×) |

All faces under the pre-registered 2× decode-collapse trigger → G2
PASS (REPORT). **The decode-side reversal vs the pre-registered
expectation is the finding**: the plan expected MXFP8 to sit below
mxfp4's 1.2–1.3× decode penalty (it rides `fp8_gemm`); measured 1.53×
— *above* both mxfp4 lanes, while every TTFT face sits *below* them.
Mechanism (weights-bytes analysis, decode is weight-bound):
- decode GEMMs are memory-bound on **weight bytes** — W8 MXFP8 reads
  2× the weight bytes of the W4 mxfp4 lanes per step, so mxfp8 decode
  sits above both mxfp4 lanes (81.8 vs 64.7/68.8 ms);
- vs the fp8 incumbent (also W8 weights) the residual 1.53× is the MX
  scale path: per-token `xpu_mxfp8_quantize(x)` as a separate op per
  quantized linear + group-32 E8M0 scale handling, where the incumbent
  rides the fused p44 block-scaled lane with coarser scales;
- prefill amortizes the quant-pass overhead over large GEMMs → mxfp8
  beats both mxfp4 lanes on every TTFT face (and W8 GEMMs are compute-
  dense vs W4's quant+gather overhead).

Disposition: **mxfp8 = supported lane on cert11** — no posture change,
no engine change; weight footprint 14.74 GiB/rank (≈ fp8's 13.6; no
capacity win vs fp8, unlike mxfp4's 9.03). Practical read: for this
checkpoint family fp8 (native) remains the performance default, mxfp4
the capacity lane, mxfp8 the AutoRound-import lane (runs as-imported
with no quality loss at t0/spec — coherent, acceptance 0.9174).

### Known-warning disposition (not an mxfp8 finding)

`Speculative decoding (method=mtp) is enabled but no KV cache group
could be identified as the draft model's` (kv_cache_utils.py:2206):
present 1× on this leg AND 1× on BOTH P6.26 mxfp4 phase-B legs,
0× on the fp8 incumbent (r105 def boot). Pre-existing CT-quant-class
behavior; spec gates pass on every leg that shows it (0.9853 / 0.9951
/ 0.9626 pos1) → dispositioned benign, tracked as a known class.

## Evidence tree

- `p627/mxfp8_phaseA.log` — phase-A serve log (kernel line, weights,
  load telemetry).
- `p627/phaseB/mxfp8/` — serve.log (boot telemetry: dispatch line,
  graph posture, MTP resolution, KV pool), serve_end.log, g1_probes.txt,
  spec_gate.txt, timeline.json + timeline_stdout.txt (G2 faces +
  bystander windows), erraudit.txt (0/0), host_load.txt (quiescence).
- `p627/stage/download.log` — model staging record.
- `r106_mxfp8_boot.sh`, `r106b_mxfp8_phaseB.sh`, `r106_run.log`,
  `r106b_run.log` — runners + raw console (tracked copies of
  `tools/ar/`).
- Host-address redaction: `results/REDACTION-2026-11-06-p627.md`
  (17 occurrences, residual 0).

## Verdict

The owner-named MXFP8 checkpoint boots, serves, speculates (k3, on the
quantized MTP drafter), and measures clean under the full certified
posture on the unmodified cert11 image — the r106 recon's predictions
held byte-for-byte (no platform rejection, native `XPUMxFp8LinearKernel`
dispatch, W8A8-dynamic on `fp8_gemm`). **No fix required; cert11
remains the certified image** (no engine-byte change; the lane is a
config-driven model route). Perf recorded with the decode-reversal
finding explained (W8 weight bytes + MX-scale overhead vs the fused
block-scaled incumbent lane).
