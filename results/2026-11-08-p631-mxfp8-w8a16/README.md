# P6.31 — online MXFP8 W8A16 route (cert21 content)

Date: 2026-11-08 · Branch: `feat/arc-p631-mxfp8-w8a16` · Image lineage:
`vllm-arc:v0.30.0-b70-cert21` (= cert20 + arc5 overlay) · Evidence root:
`rig_raw/` (this directory).

## What landed

`ARC_MXFP8_MODE` (default `w8a16`): stock online-MXFP8 quantization keeps
its 8-bit e4m3 weights but serves GEMMs as **W8A16** — bf16 activations
against f8 weights with group-32 e8m0 weight scales — via the new
vllm-xpu-kernels 0.1.14.1+arc5 op `_xpu_C.fp8_gemm_w8a16_mxfp8`
(sources: `external_patches/vllm-xpu-kernels/arc5_mxfp8_lanes/csrc/`,
wheel sha256
`219268cf7d24ebc7a99e5afec5097edd63d88ef26255511c589da71f3615399a`).
Python route: `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` (registered in
platform.py `_XPU_PATCH_MODULES`; `ARC_MXFP8_MODE=w8a8` = stock control,
byte-identical dispatch). The arc5 wheel also carries
`_xpu_C.fp4_gemm_w4a8` — the W4A8 joint (e4m3 act, f4 weight) probe op.

Why: stock online-MXFP8 is W8A8 — per-call activation quant at every GEMM
plus the (f8,f8) dtype pair lands the slow oneDNN lane. The r123 kernel
matrix (`rig_raw/kbench/r123_kbench.out`) measured the op class:
m816 GEMM medians 0.327 ms decode-M / 5.944 ms prefill-M vs stock mx8
0.569 / 15.911 (**1.74× / 2.68× faster**, correctness rel ≤ 2e-3 on every
shape) — but w16pc (the checkpoint-fp8 champion class) is still 1.50× /
1.06× faster than m816. The e2e batteries below bank the model-level
truth of exactly that prediction.

## e2e battery (r124 four-leg + r124b gemma re-run)

Postures: qwen GMU=0.85 graph=1 spec=3 lmc=1 (certified, QCFG ignore
list); gemma GMU=0.8 graph=1 spec=0 lmc=0 rpc=1200 (P6.28 law) +
`ARC_MAXLEN=40960` on the r124b legs (see capacity law below). Faces =
solo TTFT probes (p66, U8k/U32k); decode = r113 battery (solo ×3,
agg8 ×3); spec gate on qwen legs only (gemma has no MTP head).

### Qwen3.8-27B (r124, arc5test image)

| face | w8a16 (new) | w8a8 stock | ratio | champion fp8 (r105 ref) | mxfp4 w4a16 (cert20) |
|---|---|---|---|---|---|
| U8k TTFT | **3.61 s** | 5.59 s | 1.55× | 3.77 s | 3.66 s |
| U32k TTFT | **17.07 s** | 25.05 s | 1.47× | 17.37 s | 17.12 s |
| solo tps | 53.8–55.6 | 35.6–36.3 | 1.53× | **84.5** | 62.5–65.2 |
| agg8 tps | **265.9** | 194.2 | 1.37× | **300.5** | 301–311 |
| spec gate | PASS pos1=0.9804 | PASS pos1=0.9758 | — | — | — |

Route proofs: `mxfp8 w8a16 reroute ACTIVE` + 240 online-mxfp8 layers +
`Using XPUMxFp8LinearKernel for MXFP8 GEMM` (w8a16 leg) vs `reroute OFF`
(w8a8 control) — `rig_raw/leg_qwen_mxfp8_w8a16/` /
`leg_qwen_mxfp8_w8a8/`.

### Gemma-4-31B (r124b, ARC_MAXLEN=40960)

| face | w8a16 (new) | w8a8 stock | ratio | fp8 cert19 (r111) | mxfp4 w4a16 (r120g) |
|---|---|---|---|---|---|
| U8k TTFT | **3.87 s** | 6.88 s | 1.78× | 4.9 s | 3.94 s |
| U32k TTFT | **20.94 s** | 32.86 s | 1.57× | **17.2 s** | 21.16 s |
| solo tps | **27.7** | 15.0 | 1.85× | **28.4** | **40.5** |
| agg8 tps | **192.4** | 111.0 | 1.73× | **195** | **262** |

Route proofs identical (`leg_gemma_mxfp8_w8a16_ml40k/` /
`leg_gemma_mxfp8_w8a8_ml40k/`: reroute ACTIVE/OFF, 240 layers, kernel
select, ERROR=0 / Traceback=0 both legs).

### Gemma capacity law (why r124b exists)

Both r124 gemma legs BOOT_FAILED at the KV gate:
`To serve at least one request with the model's max seq len (262144),
8.33 GiB KV cache needed > available 5.31 GiB (estimated max len 103296)`.
Online-MXFP8 weights on the 31B model are ~29 GiB under TP2 — 2× the
MXFP4 legs that previously fit — leaving 5.31 GiB KV at gmu=0.8, short
of one 256K sequence. This is a **capacity gate, not a lane defect**
(`rig_raw/leg_gemma_mxfp8_w8a16*/serve.log` r124 dirs hold the refusal
tracebacks). r124b re-ran both gemma legs at `ARC_MAXLEN=40960`
(longest probe = U32k face at 32 800 tokens; g1/decode probes ≈ 200
tokens): max_model_len gates admission only, so faces/decode stay
comparable with the banked 26 2144-len refs.

## Adjudication (cert21 decision)

1. **checkpoint-fp8 stays the perf default** — w8a16 does not catch it
   (qwen solo 55.6 vs 84.5; gemma par-at-best 27.7 vs 28.4).
2. **`ARC_MXFP8_MODE=w8a16` is the online-MXFP8 lane route** (the
   default when the env is unset): 1.37–1.85× over stock w8a8 on every
   face of both models, spec-decode healthy (acceptance 0.9804 vs
   0.9758 — no numerics regression).
3. On the online-quant ladder, **MXFP4 W4A16 (cert20) remains the
   superior online lane** where 4-bit quality is acceptable (qwen solo
   62.5–65.2 vs 55.6; gemma 40.5 vs 27.7); w8a16 is the 8-bit-weights
   online option and now a real one.
4. **W4A8 (e4m3 act × f4 weight) is a banked closed-negative**: the
   arc5 op exists but oneDNN primitive-create refuses the joint dtype —
   `"Only support int4 and fp8 gemm"` (r123, all shapes `-1.00 (r-1e+00)`
   in the kbench). Documented, not pursued.

## cert21 image validation (r125 three-lane gate)

Built on the rig from `/root/build/p631/cert21ctx` (fork_sha.txt =
`c8fdf32a…`, wheel sha256 `219268cf…`); `tools/cert21.Dockerfile`
asserts passed + `CERT21_W8A16_LIVE` at build 16:01:28 (evidence
`rig_raw/cert21_l1/build.log` + `manifest.txt`). Three legs, qwen
posture (GMU 0.85, graph, MTP k3), bars pre-registered from the r121
cert20_l1 + r124 refs:

| leg | U8k | U32k | solo min | agg8 min | spec | verdict |
|---|---|---|---|---|---|---|
| A fp8 discriminator | 3.78 s (−5.3%) | 17.47 s (−5.2%) | 60.10 (bar ≥56.5) | 302.50 (bar ≥286.7) | PASS pos1 **0.9258** (byte-exact incumbent) | GREEN |
| B mxfp8 w8a16 route | 3.61 s (−4.7%) | 17.09 s (−4.6%) | 54.30 (bar ≥51.1) | 266.80 (bar ≥252.6) | PASS pos1 0.9804 | GREEN |
| C mxfp4 w4a16 carry | 3.66 s (−3.9%) | 17.10 s (−4.4%) | 61.80 (bar ≥58.4) | 299.60 (bar ≥286.0) | PASS pos1 1.0000 | GREEN |

Route proofs: leg B `mxfp8 w8a16 reroute ACTIVE` (APIServer + Worker) +
`Using XPUMxFp8LinearKernel` + 160-layer mxfp8 online dispatch; leg C
`mxfp4 w4a16 reroute ACTIVE` + `XPUMxFp4LinearKernel` + 160-layer
dispatch; leg A marker audit fa2=0 / slide=2 (the expected guarded
class). erraudit recomputed from serve.log: ERROR=0 / Traceback=0 on
every leg. W14 restored by the exit trap (radiance-test up).

**Verdict-defect disclosure (the r105b pattern)**: the original run
printed `R125_CERT21: FAIL ['A U8k face missing', …]` — a VACUOUS
fail. The gate script's verdict block reads leg files by RELATIVE
path but never `cd $G`, so it inherited the launch CWD
(`/root/build`): every read missed, all mins read 0.00, and the
verdict.txt landed outside the evidence dir (preserved as
`cert21_l1/verdict_vacuousFAIL_original.txt`). The verdict python's
regexes and bars are correct — re-running the identical block with
`cd $G` over the UNTOUCHED banked leg artifacts reads
`R125_CERT21: PASS` (`cert21_l1/verdict.txt`; every face/decode/spec
value above is parsed from those files). Two runner defects fixed in
`tools/ar/r125_cert21_gate.sh` (both commented at the fix sites):
(1) the missing `cd "$G"`; (2) the erraudit grep-absence trap —
zero-match `grep -c` exits 1, failing docker exec and firing the
`|| echo "?"` clobber (the banked `?` erraudit files; counts
recomputed leg-side). Runner md5 after fix `3fc5db3c7aa4814d8e6f5294280eda34`.

## Reproduce

```
# lane leg (default route)
ARC_QUANT=mxfp8 ARC_GRAPH=1 ... tools/serve_arc_b70.sh   # unset = w8a16
# stock control
ARC_MXFP8_MODE=w8a8 ...
# gemma online-mxfp8 legs need the capacity accommodation:
ARC_MAXLEN=40960   # 31B model at 8-bit weights: 5.31 GiB KV at gmu .8
```

Battery harnesses: `rig_raw/window_logs/r124_window.sh` +
`r124b_gemma_window.sh` (leg orchestration), rig `/root/build/r113_battery.py`
(g1 + decode), rig `/root/build/p66/p66_solo_ttft.py` (faces),
rig `/root/build/arc_spec_gate.py` (spec). Kernel matrix:
`rig_raw/kbench/r123_kbench.py` on the arc5 image. Cert gate:
`tools/ar/r125_cert21_gate.sh` (banked run log
`rig_raw/window_logs/`, evidence `rig_raw/cert21_l1/`).

Banked per publishing-hygiene law: see `../REDACTION-2026-11-08-p631.md`.
