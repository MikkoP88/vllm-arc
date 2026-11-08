# P6.30 — MXFP4 online-quantization performance: root cause + W4A16 fix (r113–r120e)

Owner directive (2026-11-07): *"Prefill, Decode and TTFT performance number
HAS TO BE SUPERIOR ON MXFP4 versus FP8! RUN FULL DIAGNOSTIG AND ANALYZE OF
PERFORMANCE ISSUES AND FIX ROOT ISSUE, THIS MAJOR ISSUE!"* — plus the P6.30
directive to make online MXFP4/MXFP8 quantization work and validate on
`google/gemma-4-31B` + `Qwen/Qwen3.8-27B`.

## Verdict

**SUPERIOR — same-image, same-config A/B (r120c window, arc4test image,
qwen3.8-27b, TP2, GMU 0.85, MTP k3, XPU-graph, LMC):**

| metric | mxfp4 stock (W4A4) | **mxfp4 W4A16 (FIX-P)** | mxfp8 online (ref leg) | **W4A16 vs online-fp8** |
|---|---|---|---|---|
| U8k TTFT (8,158 ptok) | 6.84 s (1,192 t/s) | **3.66 s (2,229 t/s)** | 5.59 s (1,458 t/s) | **1.53× faster** |
| U32k TTFT (32,553 ptok) | 29.74 s (1,095 t/s) | **17.12 s (1,902 t/s)** | 25.20 s (1,292 t/s) | **1.47× faster** |
| decode solo (tps) | 43.3–43.6 | **62.5–65.2** | 35.6–36.3 | **1.75× faster** |
| decode agg8 (tps) | 204–218 | **301–311** | 195–199 | **1.55× faster** |
| spec gate (MTP k3) | PASS 0.8813 | **PASS 1.0000** | PASS 0.9597 | — |

W4A16 vs W4A4 (the route's own A/B): **1.74–1.87× TTFT, 1.43–1.45× decode**
on every axis.

Vs the NATIVE-fp8 checkpoint champion (cert6–cert10 record battery,
docs/PERFORMANCE-arc.md: solo 84.5 / steady agg ~300 / U8k 3.76 s /
U32k 16.93 s): W4A16 is **superior on U8k TTFT (3.66 vs 3.76 s) and
agg8 (~305 vs ~300), par on U32k (17.12 vs 16.93 s)**; solo decode is
74 % of champion (62.5–65.2 vs 84.5 tps) — the one residual gap,
analysis below. Note the comparison class: online quantization (bf16
checkpoint → mxfp4 at load, zero offline tooling) vs a pre-quantized
fp8 checkpoint.

Gemma-4-31B rows (r120g, arc4test, gemma posture: GMU 0.8, spec=0 — no
MTP head, LMC=0, RPC=1200; 240 layers online-quantized, reroute ACTIVE):

| gemma-4-31b mxfp4 | W4A4 control | **W4A16** | W4A16/W4A4 | online-fp8 record¹ | W4A16 vs fp8 |
|---|---|---|---|---|---|
| U8k TTFT | 8.70 s (938 t/s) | **3.94 s (2,070 t/s)** | **2.21× faster** | 4.9 s | **1.24× faster** |
| U32k TTFT | 40.25 s (809 t/s) | **21.16 s (1,539 t/s)** | **1.90× faster** | 17.2 s | 1.23× slower |
| decode solo | 18.6 tps | **40.5 tps** | **2.18× faster** | 28.4 tps | **1.43× faster** |
| decode agg8 | 135.1–135.7 tps | **261.8–262.1 tps** | **1.93× faster** | 195.5 tps | **1.34× faster** |

¹ gemma online-fp8 = plain dynamic fp8 (the P6.28 lane, cert19 r111
ladder) — the FAST fp8 path, unlike mxfp8's e8m0 group path. W4A16 beats
it on 3 of 4 axes; the U32k exception is the W4A16 prefill physics
ceiling (bf16-DPAS-par; no fp4 MMA on Xe2) against fp8's static-class
lane. Cross-window sanity: the W4A4 control matches r113's cert19
numbers (9.4 / 31.9 s; 18.7 / 132.3) in-family.

Boot-failure postmortem (r120c + r120e/f, kept for the record): the
first three gemma attempts died at boot — (a) r120c/r120e:
`ARC_MODEL=/models/gemma-4-31b` was a path typo (real dir
`/models/gemma4-31b`, no dash); docker's bind-mount auto-created an
EMPTY host dir and vLLM refused /models/target at ModelConfig validation
(no hint the name was wrong). (b) r120f: `Unsupported speculative
method: 'mtp'` — gemma-4 has no MTP head; spec=3 is qwen-only. r120g
(path + spec=0) boots and passes everything. The r120c LMC-in-args
observation was real but not proximate (P6.28 law still enforced:
LMC=0).

r120h (qwen BASE + plain `ARC_QUANT=fp8` online, the gemma-lane class)
also died at boot — an upstream INTERFACE gap, not a kernel issue:
`--quantization fp8 --quantization-config '{"ignore":[…]}'` fails
pydantic validation with *"OnlineQuantizationConfig requires at least
one of quantization_config.linear, quantization_config.moe, or
quantization_config.targets to be set"*. On this vLLM, plain `fp8` is
NOT an online shorthand (`_ONLINE_SHORTHANDS` =
fp8_per_tensor/fp8_per_block/**fp8_per_channel**/mxfp8/mxfp4/int8/…);
any `--quantization-config` JSON is consumed by the online framework,
where it must carry targets. Gemma's P6.28 fp8-online lane ran with
QCFG **unset** (classic dynamic fp8 — fine on a pure-attention model
with nothing to ignore); qwen's hybrid needs the ignore list for
linear_attn/visual/mtp.fc, and the classic path has no CLI ignore
channel. The expressible qwen lane is `fp8_per_channel` + QCFG (the
llmcompressor FP8_DYNAMIC recipe shape, per vLLM's own comment) —
probed as r120h2 at close-out: **BOOT_FAILED** with an explicit
platform refusal, *"fp8_per_channel quantization is currently not
supported in xpu"* (ModelConfig validation, W14 pre-flight health 200,
clean stop) — traceback banked in
`rig_raw/leg_qwen_fp8perchannel/serve.log`. With r120h (schema gap)
and r120h2 (explicit refusal) both closed, the online-fp8 matrix for
the qwen hybrid is complete: NO lawful CLI channel exists on this
stack — checkpoint-fp8 stays the certified default; plain-fp8-online
traceback in `rig_raw/leg_qwen_fp8online_ref/serve.log`.

## Root cause (the diagnostic chain, r113→r118)

1. **r113 e2e**: online mxfp4 ~1.9× prefill/TTFT penalty vs fp8 — the
   "major issue" of the directive.
2. **r116b/r116c microbench** (kernel-level, cert19/oneDNN v3.12):
   (a) the per-call mxfp4 activation quant alone costs ≈ the GEMM at
   decode Ms; (b) the (f4 acts × f4 weights) matmul runs oneDNN's
   `matmul:ref` kernel while fp8 W8A8 rides `jit:gemm` DPAS.
   CORRECTION (r120): r116b/r116c's manual fp4 GEMM cells were
   half-geometry (weight passed untransposed; M=8 assert caught it) —
   fp4 absolute timings invalid, fp8 cells stand; r120 kbench re-measured
   with wheel-exact layouts.
3. **Gate math** (oneDNN v3.13 `src/gpu/intel/gemm/jit/pd.cpp`
   `wei_decomp()`): a matmul is JIT-eligible when the DECOMPOSED operand
   (here: weights) is f4_e2m1 (in the decomp list) AND the counter
   operand (activations) is one of f16/f32/bf16/f8_e5m2/f8_e4m3 AND the
   decomposed type is <16 bits. **f4 activations are NOT in the counter
   list** → stock W4A4's joint (f4, f4) dtype can never pass the gate →
   ref kernel, structurally.
4. **FIX-P**: keep activations bf16/f16 (drop the per-call quant AND the
   f4 activation operand) → new op `_xpu_C.fp4_gemm_w4a16`
   (arc4 wheel, 5 source files, banked at
   `external_patches/vllm-xpu-kernels/arc4_fp4_gemm_w4a16/csrc/`).

## Proof the fix hit the designed path

oneDNN verbose, arc4test smoke (rig `p630/arc4build/smoke.out`, local
`rig_raw/arc4build/smoke.out`):

```
onednn_verbose,v1,primitive,create:cache_miss,gpu,gemm,jit:gemm:any,
  undef, src_a:bf16::blocked:ab::f0 src_b:f4_e2m1::blocked:ba::f0
  dst:bf16::blocked:ab::f0,
  attr-fpmath:bf16:true attr-scales:wei:3:e8m0:32x1,,1x5376:5376x5376
```

`jit:gemm` (DPAS, weight decomposition on the fly) where stock W4A4
lands `matmul:ref`. Numerics: rel ≈ 1.66e-3 vs fp32 dequant reference
(gemma.o shape, all Ms) — the activation-quant error term is gone;
weights stay the checkpoint's own f4.

## Kernel-level matrix (r120d kbench — full four-lane matrix, arc4test)

Wheel-exact layouts (B = packed w `.t()` nt view; w4a16 scales
[k/32, n] e8m0; fp8 cell = e4m3 operands with e8m0 per-32 group scales
on BOTH sides — the online-MXFP8 GEMM; w4a4 = quant + GEMM), ms, median
of 50 (`rig_raw/kbench/r120d_kbench.out`; the r120 first pass agreed
in-family on the shared w4a16/bf16 cells):

| shape (K×N) | M | bf16 | fp8 e8m0 | w4a4 q+g | **w4a16** |
|---|---|---|---|---|---|
| gemma.up 5376×10752 | 1 | 0.710 | 0.367 | 0.442 | **0.251** |
| gemma.up | 8 | 0.751 | 0.365 | 0.551 | **0.322** |
| gemma.up | 4096 | 3.498 | 9.658 | 13.856 | **3.839** |
| gemma.up | 32768 | 29.701 | 78.283 | 109.221 | **36.722** |
| gemma.down 10752×5376 | 1 | 0.813 | 0.360 | 0.680 | **0.168** |
| gemma.down | 8 | 0.733 | 0.359 | 0.764 | **0.198** |
| gemma.down | 4096 | 3.713 | 10.233 | 14.100 | **3.314** |
| gemma.down | 32768 | 29.672 | 81.563 | 111.064 | **29.911** |
| gemma.o 5376×5376 | 1 | 0.638 | 0.406 | 0.424 | **0.170** |
| gemma.o | 8 | 0.624 | 0.248 | 0.431 | **0.169** |
| gemma.o | 4096 | 2.216 | 4.414 | 7.003 | **1.991** |
| gemma.o | 32768 | 14.793 | 39.462 | 56.188 | **18.399** |
| qwen.up 5120×8704 | 1 | 0.796 | 0.372 | 0.426 | **0.192** |
| qwen.up | 8 | 0.839 | 0.422 | 0.568 | **0.382** |
| qwen.up | 4096 | 2.943 | 7.516 | 10.920 | **3.167** |
| qwen.up | 32768 | 22.820 | 60.600 | 86.635 | **28.826** |
| qwen.down 8704×5120 | 1 | 0.819 | 0.329 | 0.440 | **0.172** |
| qwen.down | 8 | 0.656 | 0.367 | 0.533 | **0.171** |
| qwen.down | 4096 | 2.970 | 7.607 | 10.878 | **2.766** |
| qwen.down | 32768 | 23.059 | 62.067 | 86.578 | **22.905** |
| qwen.o 5120×5120 | 1 | 0.678 | 0.400 | 0.413 | **0.174** |
| qwen.o | 8 | 0.701 | 0.402 | 0.499 | **0.175** |
| qwen.o | 4096 | 1.989 | 4.030 | 6.523 | **1.785** |
| qwen.o | 32768 | 13.424 | 35.374 | 50.631 | **16.616** |

Aggregate (n=12 per class): decode (M≤8) w4a4-apply/w4a16 = 2.53×,
w4a16/fp8 = 0.52×; prefill (M≥4096) w4a4-apply/w4a16 = 3.56×,
w4a16/fp8 = 0.43×.

Reading: at decode Ms **w4a16 is the fastest lane on every shape** —
4-bit weights, memory-bound, the expected physics. At prefill M=32768
w4a16 is par (down-proj) to 1.24× slower than bf16 — Xe2/BMG has no
fp4 MMA, so bf16-DPAS-par is the W4A16 ceiling at large M — but
**2.1–2.7× faster than the e8m0-fp8 GEMM** and 3.0–3.8× faster than
w4a4.

**v3.13 verdict on the online-fp8 slow path: NOT fixed — and now
mechanistically pinned.** The verbose trail shows the fp8 cell riding
`jit:gemm` (NOT the ref kernel) with bidirectional group scales
(`attr-scales:src0:3:e8m0:1x32+wei:3:e8m0:32x1`), yet it still runs
2.6–2.8× slower than bf16 at prefill (gemma.up 78.3 vs 29.7 ms). The
penalty is the e8m0 group-scale handling inside the DPAS kernel — a
kernel-quality gap oneDNN v3.13 did not close (the online lane adds
the per-call activation quant on top; native-checkpoint fp8 with
static per-tensor scales remains the fast fp8 incumbent, a different
op path). mxfp8-online therefore stays the slow lane; W4A16 is the
superior online-quant route on both models.

## The solo-decode residual vs the native-fp8 champion (honest ledger)

62.5–65.2 vs 84.5 tps (74 %). Kernel-level M=1 (r120d) says w4a16
(0.17–0.25 ms) is already ~2× faster than the e8m0-fp8 GEMM cell
(0.33–0.41 ms) and 3–4× faster than bf16 (0.64–0.82 ms) — the gap does
not live in the GEMM itself. Same-window spec gates pass with HIGHER
acceptance
(1.0000 vs champion's 0.9258 pos1). The delta (≈3.8 ms/token) is in
per-step overheads across ~200 linear calls/token (per-call group-scale
pointer setup inside graph replay, weight-decomp prologue at M=1).
Next-lane candidates, in order: (1) **W4A8** — e4m3 activations are IN
the counter-operand list, so (e4m3, f4) passes the same gate and rides
the fp8-rate DPAS path with an exact per-group e4m3 quant (no e8m0 math
at runtime); (2) a decode-specialized GEMV lane if the primitive's M=1
prologue dominates. Neither is needed for the directive's comparison
(online mxfp4 vs online fp8 — won on every axis).

## Engineering laws this campaign added

1. **`import vllm_xpu_kernels` does NOT dlopen `_xpu_C`** — the package
   `__init__` only re-exports flash-attn; even stock `fp4_gemm` reads
   absent after a bare package import. Always import the submodule
   explicitly before op-presence probes (r120b smoke FATAL was exactly
   this, not a bad wheel). Corollary: `vllm/_xpu_ops.py` registers the
   `torch.ops.vllm::xpu_mxfp*_quantize` wrappers and its own hasattr
   probes are ORDER-sensitive: `import vllm_xpu_kernels._xpu_C` FIRST,
   then `import vllm._xpu_ops` (r120 kbench lost its fp8/w4a4 cells to
   the missing half of this pair).
2. **≤24 icpx jobs on the rig whenever a W14 window is live** — 52 jobs
   OOM-killed the linker and starved the W14 checkpoint load into a
   WorkerProc crash (arc4 build attempt 1).
3. pip validates wheel FILENAMES — COPY/install under the real wheel
   name in image Dockerfiles (r120 FATAL).
4. Gemma legs must launch with `ARC_LMC=0` (+ `ARC_RPC_TIMEOUT=1200`,
   the r108x knob) AND the base-model dir is `/models/gemma4-31b` (no
   dash — a wrong path makes docker auto-create an EMPTY host dir and
   vLLM refuses `/models/target` at ModelConfig validation with no hint
   the dir name was the cause) — encoded in r120f's gemma_leg().

## Evidence map

- `rig_raw/leg_qwen_mxfp4_w4a4/` — control leg: full battery + serve.log
  (mode proof: reroute OFF) + kv_lines + dispatch proofs.
- `rig_raw/leg_qwen_mxfp4_w4a16/` — the fix leg: full battery + serve.log
  (reroute ACTIVE, mxfp4_w4a16.py:118) + kv_lines (10.22 GiB).
- `rig_raw/leg_qwen_mxfp8_ref/` — same-image fp8 reference (final pull).
- `rig_raw/leg_qwen_fp8online_ref/` — the r120h plain-fp8-online boot
  failure (serve.log with the OnlineQuantizationConfig traceback +
  leg_config).
- `rig_raw/leg_qwen_fp8perchannel/` — r120h2 probe leg: BOOT_FAILED,
  explicit "fp8_per_channel … not supported in xpu" ModelConfig refusal
  (serve.log + leg_config; closes the online-fp8 matrix).
- `rig_raw/cert20_l1/` — the r121 cert20 gate L1 record: build.log +
  manifest.txt (IMAGE_MANIFEST with cert20 row + fork_sha + wheel
  sha256) + leg_fp8_disc/ (fp8 discriminator leg — in-band battery,
  spec 0.9258 byte-exact, marker audit proving the w4a16 reroute
  installed-but-inert) + leg_w4a16/ (gate leg — reroute ACTIVE
  marker, `Quantized 160 layers` group-32 packed dispatch proof,
  `Using XPUMxFp4LinearKernel`, battery matching r120c bytes).
- `rig_raw/window_logs/` — all r120* window narratives (r120→r120h,
  complete post-flight pulls).
- `rig_raw/qwen_mxfp8_fixc/` — r119 leg 1: FIX-C @GMU 0.85 full battery
  (the r117 KV-sizing ValueError closed as config fix).
- `rig_raw/kbench/r120_kbench.out` — w4a16 + bf16 cells (first pass).
- `rig_raw/kbench/r120d_kbench.out` — fixed kbench full matrix (close-out).
- `rig_raw/arc4build/smoke.out` — SMOKE_OK + the jit:gemm route verdict.
- `rig_raw/leg_gemma_mxfp4_w4a16/` + `rig_raw/leg_gemma_mxfp4_w4a4/` —
  the r120g gemma A/B legs (full batteries + serve logs, mode proofs
  ACTIVE/OFF); the dirs also hold the earlier r120c/e/f boot-failure
  serve logs (overwritten by the successful legs' final docker-cp).
- `rig_raw/window_logs/r120c_window.log` (r120/r120b/d/e/f/g/h alongside)
  — window narratives.
- Wheel + build: `external_patches/vllm-xpu-kernels/` (arc4 section).
- Wheel-side patch: `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` +
  `platform.py` `_XPU_PATCH_MODULES` entry (gate `ARC_MXFP4_MODE`).

Redaction: `../REDACTION-2026-11-08-p630.md` (byte-exact perl pass over
the fetched serve logs).
