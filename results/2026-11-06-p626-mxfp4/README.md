# P6.26 — MXFP4 support: validation + fix (2026-11-06)

Directive (owner, 2026-11-06): full MXFP4 testing/validation on
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` and `RedHatAI/Qwen3.8-27B-MXFP4xFP8`;
deep-analyze any issue, fix start-to-end upstream-style, certify image,
update docs. Scope-law basis (plan §3/App I): both checkpoints are
owner-named model routes — line-C/D "target model only" law applies to
the *engine*, not to owner-directed probe models (P6.26 directive
recorded plan §P6.26).

Harnesses: `tools/ar/r102_mxfp4_recon*.sh` (wheel recon, 8 parts),
`tools/ar/r103_mxfp4_boot.sh` + `r103b_qurk_retry.sh` (phase A),
`tools/ar/r104_mxfp4_phaseB.sh` (phase B). Model staging on rig:
`/models/qwen3.8-27b-mxfp4xfp8`, `/models/qwen3.8-27b-quark-awq-mxfp4`.

## The issue (predicted, then confirmed byte-exact)

r102 recon of the shipped wheel (cert10) predicted the exact failure
before any boot (plan §P6.26 P-1): core's `XPUPlatform
.supported_quantization` omits `"quark"`, and `Platform
.verify_quantization` (platforms/interface.py:988) hard-enforces that
list at ModelConfig-validation time — so `amd/Qwen3.8-27B-Quark-AWQ
-MXFP4` (quant_method `quark`) dies with

```
Value error, quark quantization is currently not supported in xpu.
```

before any scheme/kernel code runs. This is an upstream format-origin
omission (AMD/ROCm export format), not a hardware limit: the wheel's
QuarkConfig routes its OCP-MX W4A4 scheme through
`init_mxfp4_linear_kernel` — the same platform-neutral entry
CompressedTensorsW4A4Mxfp4 uses — which resolves on Xe2 to the native
`XPUMxFp4LinearKernel` (vllm-xpu-kernels fp4_gemm).

Phase A confirmed: rhmxfp4 booted UNMODIFIED on cert10; qurk failed
with exactly the predicted error (phaseA/qurk_phaseA.log).

## The fix (commit 7ea0f27b, `vllm_gaudi/xpu/platform.py`)

Two overrides on `ArcXpuPlatform`, HpuPlatform-pattern, zero wheel edits:

1. `supported_quantization = [*XPUPlatform.supported_quantization, "quark"]`
   — passes the interface gate for quark checkpoints.
2. `supports_mx() → True` — truthful Xe2 capability statement (BMG
   computes MX types natively; the class is BMG-gated by device-id).
   Kills the false "Simulated weight dequantization" warning
   (quark_ocp_mx.py:73). Kernel selection is table-keyed
   (`_POSSIBLE_MXFP4_KERNELS[XPU] = [XPUMxFp4LinearKernel]`) and never
   consults `supports_mx` — r102 recon part 7/8 — so this changes no
   dispatch, only log honesty.

After the docker-cp retry (phaseA/qurk_phaseA_retry.log): weights
9.03 GiB/rank in 2.51 s, KV 13.25 GiB, native kernel line present, no
Simulated warning, coherent generation.

## Phase B — full certified posture (cert11c image)

Posture: `ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1` (= incumbent cert10's exact
served posture: XPU graph FULL_DECODE_ONLY, MTP k3 probabilistic,
LMCache on), config-driven quant (no `--quantization` flag — checkpoint
quant_config drives the scheme). Image `vllm-arc:v0.30.0-b70-cert11c`
(cert10 + 7ea0f27b). Runner r104; W14 stopped for the window, restored
by trap + generation-probed after (rig law).

### Telemetry (both legs)

| leg | kernel dispatch | KV pool | graph posture |
|---|---|---|---|
| rhmxfp4 | `XPUFp8BlockScaledMMKernel` for CompressedTensorsW8A8Fp8 (attn, CT-mixed) **+** `XPUMxFp4LinearKernel for MXFP4 GEMM` (MLP) | 10.38 GiB | FULL_DECODE_ONLY, 71 sizes |
| qurk | `XPUMxFp4LinearKernel for MXFP4 GEMM` (all quantized linears) | 12.16 GiB | FULL_DECODE_ONLY, 71 sizes |

The mixed-precision CT checkpoint keeps its fp8 attention lane on the
proven block-scaled path and runs MLP on the native fp4 kernel — exactly
the split its config declares.

### G1 quality (3-probe, t0) — both legs coherent

factual/reason/code probes produce well-formed reasoning + correct
content (phaseB/*/g1_probes.txt); reasoning-format behavior identical to
the fp8 incumbent (finish=length on the short budgets, answers inside
the reasoning channel — same as cert10 target).

### Spec gate (MTP k3 probabilistic) — PASS both, above incumbent

| leg | pos1 | acceptance | mean acc len | verdict |
|---|---|---|---|---|
| rhmxfp4 | 0.9853 | 0.9820 | 3.95 | PASS (hard 0.5, target 0.85) |
| qurk | 0.9951 | 0.9805 | 3.94 | PASS |
| cert10 fp8 incumbent (ref) | 0.9258 | — | — | — |

### G2 long-stream faces (TTFT, s) vs cert10 incumbent (r99 perf_on)

| face | fp8 incumbent | rhmxfp4 | ratio | qurk | ratio |
|---|---|---|---|---|---|
| U32k | 16.13–16.45 | 29.01 | 1.79× | 32.28 | 1.97× |
| C32k (cached) | 1.02–1.05 | 1.68 | 1.62× | 3.84 | 3.71× |
| U8k | 3.26–3.28 | 7.37 | 2.25× | 7.73 | 2.36× |
| U64k | 35.72–35.75 | 62.46 | 1.75× | 70.61 | 1.98× |
| bystander decode p50 | 53.5 ms | 64.7 ms | 1.21× | 68.8 ms | 1.29× |

G2 was pre-registered REPORT-only (trigger = decode collapse > 2×):
decode-side latency penalty is 21–29 % — well under the trigger.
Prefill/TTFT pays 1.6–2.4× (fp4_gemm W4A4-dynamic includes per-call
activation quant and is not as tuned as the p44 fp8 block lane; qurk's
all-layer W4A4 pays more than rhmxfp4's MLP-only split, and its
uncached C32k shows the weight-dequant cache-build too). In exchange
the 27 B weights drop to 9.03 GiB/rank (quark, vs ~13.6 fp8) and KV
headroom rises ~2 GiB/rank. Disposition: **mxfp4 = supported, certified
lane for weight-constrained deployments; fp8 stays the certified
performance default** (no posture change; incumbent regression gates in
r105 guard the default lane).

## Evidence tree

- `phaseA/` — r103 boot logs both legs (qurk failure + patched retry),
  driver logs.
- `phaseB/{rhmxfp4,qurk}/` — serve.log (boot telemetry), serve_end.log
  (end-of-leg), g1_probes.txt, spec_gate.txt, timeline.json +
  timeline_stdout.txt (p65 faces + bystander gaps).
- Host-address redaction: `results/REDACTION-2026-11-06-p626.md`.

## Verdict

Both owner-named MXFP4 checkpoints boot, serve, speculate (k3), and
measure clean under the full certified posture on the patched platform.
Fix is minimal, upstream-patterned (HpuPlatform precedent), wheel-free,
and carries no dispatch change for existing lanes (table-keyed
selection). Certification: r105 tree-identity + incumbent regression
battery on cert11c → `rel/v0.30.0-b70-cert11` (see plan §P6.26 outcome).
