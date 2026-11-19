# arc7_mx_attn — MX-native paged-decode kernel (P6.47 Phase 8)

Wheel section for `vllm-xpu-kernels` 0.1.14.1+arc7: the MX KV lanes
(mxfp4 / mxfp8 / mxk8v4) get the same vendor-class fused decode kernel
the fp8 ctrl lane rides, with MX dequant fused into the load path.

## Why (research banked 2026-11-19, master plan §P6.47 Phase 8)

The ctrl lane's kernel is `cutlass_paged_decode_xe2` (CUTLASS-3/CuTe on
SYCL, `csrc/xpu/attn/xe_2/`): LSC 2D-block descriptor loads with
prefetch off the paged pool, register-resident online softmax (no
mid-buffer round-trip), split-KV + `ReduceSplitK`. FP8 is not the MAC
dtype anywhere on Xe2 — the vendor loads fp8 bytes and casts in-register
(`reorder()`), folding per-tensor scales by linearity. The MX lanes lose
~22-25% agg8 decode to the Triton split-dot structure (r197 attribution:
mid-buffer traffic, no LSC descriptors, coarse grid geometry). This
module ports the vendor structure to MX formats: identical walk, MX
dequant inserted at the `reorder()` cast site.

## What ships here

```
csrc/xpu/attn/xe_2/mx/
  mx_types.hpp               format traits + dequant bit-math primitives
  mx_dequant.hpp             dequantize_block_wise (mirror of upstream
                             cute::quantize_block_wise, cutlass-sycl)
  mx_paged_decode_mainloop.hpp  fork of DecodeFwdMainloop, MX load path
  mx_paged_decode_kernel.hpp fork of XeFMHAFwdSplitKVKernel (four
                             surface tensors; scheduler/split/epilogue
                             plumbing stock)
  mx_paged_decode.hpp        config/launcher (fork of paged_decode.hpp)
  mx_paged_decode_xe2.cpp    entry + runtime dtype/fmt dispatch
patches/
  attn_interface.{h,cpp}.patch   dispatch entry (pure addition)
  torch_bindings.patch           pybind registration
  CMakeLists.patch               source-list entries
apply_arc7.py               rig-side patcher (arc6 discipline: exact
                            anchors, refuses on miss/double-apply)
build_vxk_wheel_arc7.sh     host-swap wheel build (arc6 pattern)
vxk_arc7_inner.sh           in-builder build script
vllm_xpu_kernels/mx_attn_interface.py   python API (mirrors
                            flash_attn_interface decode branch incl.
                            build_decode_split_plan semantics)
```

## Layout contract (zero re-layout: our TQ-AoS slots ARE the surfaces)

Our cache (`vllm_gaudi/xpu/patches/mx_kv/mx_layout.py`) is one uint8
pool `[num_blocks, block, Hk, slot_bytes]`; per slot
`[K bytes][Ks e8m0 x D/32][V bytes][Vs e8m0 x D/32]`. The vendor kernel
already accepts arbitrary paged strides (MLA/cross-layer support), so
the four byte-surfaces are the slot sub-ranges at the mx_layout offsets:

- K bytes: base `+0`, width `D` (mxfp8/mxk8v4-K) or `D/2` (mxfp4)
- K scales: base `+ks`, width `D/32 = 4`
- V bytes: base `+v`, width as K (per-format)
- V scales: base `+vs`, width 4

with `seq stride = Hk*slot_bytes`, `head stride = slot_bytes`,
`page stride = block*Hk*slot_bytes` — our strides `(1),(2),(0)` verbatim.
D=128 throughout this lane. Nibble law (mx_quant.py): element `2j` =
LOW nibble of byte `j`, `2j+1` = HIGH; `value = e2m1(nibble) *
2^(code-127)`.

## Status (2026-11-19, second authoring pass)

- `mx_types.hpp`, `mx_dequant.hpp` — complete first implementations.
- `mx_paged_decode_kernel.hpp` — first pass: the stock kernel fork
  (four-surface tensor construction, mainloop invocation; scheduler/
  split/epilogue plumbing stock). Unverified against compiler.
- `mx_paged_decode.hpp` — first pass: args/launcher/config + the
  self-contained policy dispatch (3 formats x qgroup {_8,_16} x head
  _128 x page _16). Decode laws mirrored from vendor: `is_causal`
  always false (seqused_k bounds KV; causal adds +q_sg_tile
  pollution), split default = the flash_api `get_num_splits` rule.
- `mx_paged_decode_xe2.{h,cpp}` — first pass entry: slot-offset table
  (mirrors mx_layout.py), C++-side decode-buffer allocation (vendor
  shapes), fmt x qgroup runtime dispatch.
- `mx_paged_decode_mainloop.hpp` — SKELETON: stock control flow
  (paging, surface split, softmax, loop structure) complete; the MX
  dequant insertion points are designed and marked, but the
  fragment/coordinate plumbing (`identity_*_partition` shorthands,
  `TileShapeVS`, the dequant-fragment TV layouts feeding `reorder()`)
  must be resolved against the actual cute XPU copy/MMA APIs during
  rig-side compile iteration (r198's build loop) — the cutlass-sycl
  headers only exist in the builder's `.deps`, so this iterates there.
- `mx_attn_interface.py` — first pass python surface (split-plan
  mirror; seqused_k rides the cu_seqlens_k slot per the vendor decode
  flow).
- Patches (attn_interface/torch_bindings/CMake), apply + build
  scripts, vllm-side wiring: pending (post-cert25zc-release).

## Validation ladder (master plan §P6.47 Phase 8)

r198 offline correctness (oracle: identical quantized inputs, bitwise
vs Triton dequant; gate d<=2e-3 vs v2 reference) -> r199 T1 perf A/B vs
v5 at the money cells -> r200 e2e fastcycle (ARC_MXKV_NATIVE=1) ->
r201 full §10 battery -> cert26. Kernel ships dormant behind
`ARC_MXKV_NATIVE` until certified; Triton v2/v5 stay the default.

## Build laws

oneAPI 2026.0.0 toolchain, `vxk-arc3-builder` host-swap (arc6 tree kept
at `/root/build/vxk-arc6-kept`), `.deps` carried, wheel version
`0.1.14.1+arc7`. Same ABI laws as arc3-arc6 (`external_patches/
vllm-xpu-kernels/README.md`).
