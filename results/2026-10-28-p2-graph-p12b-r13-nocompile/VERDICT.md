# VERDICT — r13 REFUTES the compile mechanism: with CompilationMode.NONE (fully eager, zero inductor artifacts) and FULL decode graphs captured cleanly, the stream STILL flips at the first request — and the eager onset stack names the GDN linear-attention prefill gemm

## The leg

arc-graph-nocompile-r13-20261028-073651, image c966fcf89a3f (fork
cbb6937e), new serve knob ARC_NOCOMPILE=1 → `--compilation-config
{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY", sizes 1..320}`,
ARC_TRACE=1 (watchdog + hygiene state logging armed).

Posture verified in the engine config dump:
- `'mode': <CompilationMode.NONE: 0>` — the piecewise backend was NEVER
  constructed ("Inductor compilation was disabled by user settings");
  boot 2.5 min (no ~199 s compile).
- `cudagraph_mode FULL_DECODE_ONLY`, FULL decode graphs captured
  cleanly: "Capturing CUDA graphs (FULL): 27/27", 8 s — **mode=NONE is
  compatible with FULL decode capture on 0.30** (relevant if the
  productized force survives).

## Result — fix candidate #1 dead, mechanism narrowed

- First request **200 OK** (log line 285, 07:39:44).
- **07:39:46 ARC_GRAPH_ZOMBIE_ONSET**: device-default stream reads
  RECORDING with no capture open — same signature as r3-r12.
- Every request after: 500; integrity guard raised "flipped mid-forward
  twice in one step" and killed the worker (fail-loud; no garbage).
- Zero "FLIP at triton-pre" lines: the r12c tracer wraps inductor's
  CachingAutotuner, which eager GDN kernels never touch — the 10 ms
  watchdog caught the onset instead.

**Compilation is exonerated.** The r12/r12c attribution (loose
compiled-prefill launch) was circumstantial: with compile on, prefill
dominates the first-request timeline, so the watchdog landed inside the
compiled callable. Cofactor law updates to: **flip = first real GDN
prefill × live graph machinery**, compile on or off.

## The eager onset stack (stack_extract.txt) — no inductor noise

```
MainThread:
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:841 forward
  → forward_xpu:992
  → self.in_proj_ba(hidden_states)            # GDN ba projection gemm
vllm/model_executor/layers/linear.py:238 apply
  → utils.py:90 default_unquantized_gemm
  → torch.nn.functional.linear(x, weight, bias)
```

This is the SAME GEMM r12 caught in compiled form (`extern_kernels.mm(
buf1, reinterpret_tensor(arg6_1, (5120, 48), ...))` — the in_proj_ba
weight view). Two independent legs (compiled r12, eager r13) land in the
same projection of the same layer family at the flip.

## Model v2 (defect-b)

The stock **Qwen GDN XPU prefill path** (qwen_gdn_linear_attn
forward_xpu: in_proj_ba / chunked delta-rule / fla triton mix) poisons
the device-default stream when XPU graph machinery is live in the
process. Both clean controls avoid exactly this path:

- **W14 incumbent** (core 0.30, clean, TPOT c8 33 ms): GDN runs on the
  ported radiance **ESIMD QSA/GDN kernels** — plan-sanctioned
  shared-component lane — not this stock path; graph posture
  FULL_AND_PIECEWISE, sizes [4,8,16,32,64], max 64.
- **0.26 D champion** (clean, agg 274.41): older GDN code + torch 2.12,
  no compile, FULL graphs.

## r14 (next): W14-graph-posture leg — decisive between the two suspects

All-CLI-knob leg on our stack: `cudagraph_mode=FULL_AND_PIECEWISE`,
`sizes [4,8,16,32,64]`, `max_cudagraph_capture_size 64`, compile ON
(PIECEWISE needs the compiled pieces — W14's exact serve line).
- Clean → graph posture (mode/sizes/count) is the mechanism; the W14
  posture becomes the P2 candidate posture (owner gates: defect-a block,
  P4 prefill-parity bars).
- Flips → posture insufficient; the GDN kernel path is the poison and
  the fix lane is the p10 ESIMD QSA/GDN import (radiance sanctioned,
  plan §3/Appendix I) — or the torch/triton regression pin.

Productization note: the drafted ArcXpuPlatform mode=NONE force is ON
HOLD (uncommitted) — r13 shows it does not fix the flip, and it would
block the FULL_AND_PIECEWISE posture if that becomes the fix.

Raws: serve_full.log (onset line 286, stacks 287-399, 200 at 285, 500s
from 513), stack_extract.txt, boot.txt, gates.txt, k1k_sanity.txt,
spec_gate.txt, matrix.log (rig leg dir
arc-graph-nocompile-r13-20261028-073651).
