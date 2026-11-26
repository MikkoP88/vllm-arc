# P6.47 P-E — r247 pinned-stack MX surface census: VERDICT

**CENSUS COMPLETE (rc=0, 27-line bank, W14 untouched).** The P-E
existence-proof base for every MX lever disposition: what the pinned
stack actually ships, measured inside cert28 (`vllm-arc:v0.30.0-b70-
cert28`), CPU-safe (no device init, no engine boot).

## Facts banked

- `torch.version=2.13.0+xpu` (the oneCCL-pinned build; no-lawless-
  upgrade law).
- MX dtype surface COMPLETE: `torch.float4_e2m1fn_x2` (packed e2m1),
  `float8_e4m3fn`, `float8_e4m3fnuz`, `float8_e5m2`,
  **`float8_e8m0fnu`** (the MX scale dtype) — all present.
- Public compute API: **`torch.nn.functional.scaled_mm`** with the
  recipe surface (`_ScalingType` | list, `_SwizzleType` | list,
  per-operand scale tensors) + `scaled_grouped_mm`; private
  `torch._scaled_mm` / `torch._scaled_grouped_mm` (builtins).
- Dispatch evidence: `aten::_scaled_mm` carries **4 XPU entries** —
  XPU kernels are registered (the dot is counter-operand conversion
  on Xe2 per the xe2-xmx-fp8-nativeity verdict — no native FP8/MX
  dot on BMG-G31).
- **ZERO aten-level MX format ops**: no `float4`/`e2m1`/`e8m0`/`mx`
  op names in `torch.ops.aten` — MX packing/conversion is not an aten
  surface in this build; the de-facto MX compute lanes are the fork's
  own C-ext ops (`fp4_gemm_w4a16`, `fp4_gemm_w4a8` in the onednn
  family) and the mx_kv wheel kernels.
- **torchao is NOT INSTALLED** — the torchao `mx_formats`/`to_mx`
  branch of the MX design space is closed on this stack (adding it
  would be a new dependency for a surface the fork owns more
  directly; scope law).
- `intel_extension_for_pytorch` is not a separate importable module
  in this image (the xpu extension is built into `torch.*+xpu`; the
  fork's patches call the C-ext ops through the deferred kernel-module
  import).

The lever dispositions built on these facts live in
`docs/MX-RESEARCH-arc.md` (the P-E deliverable). Bank files:
`r247.out` (the census), `SHA256SUMS` (redacted-byte sums; host sweep
clean).
