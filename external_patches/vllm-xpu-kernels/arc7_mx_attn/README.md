# arc7_mx_attn — MX-native paged-decode kernel (P6.47 Phase 8)

Wheel section for `vllm-xpu-kernels` 0.1.14.1+arc7mx: the MX KV lanes
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
  mx_dequant.hpp             dequant_dtile: in-register u8->bf16 for one
                            d-tile/v-step, coordinate-driven scale
                            reads from the gmem e8m0 surface (4-slot
                            per-WI cache; L1 absorbs group reuse)
  mx_paged_decode_mainloop.hpp  fork of DecodeFwdMainloop, MX load path
  mx_paged_decode_kernel.hpp fork of XeFMHAFwdSplitKVKernel (four
                             surface tensors; scheduler/split/epilogue
                             plumbing stock)
  mx_paged_decode.hpp        config/launcher (fork of paged_decode.hpp)
  mx_paged_decode_xe2.cpp    entry + runtime dtype/fmt dispatch
apply_arc7mx.py             rig-side patcher (arc6 discipline: exact
                            anchors, refuses on miss/double-apply; base
                            = the certified arc6 tarball, provenance
                            asserts skinny-present/jm-absent; edits
                            embedded per the arc6/arc7 kit law)
vxk_arc7mx_bootstrap.sh     one-time builder-venv reconstruction (the
                            jm-era builder container is gone; arc3
                            postmortem recipe)
build_vxk_wheel_arc7mx.sh   host orchestrator: builder (re)creation +
                            host-swap wheel build (arc6 pattern)
vxk_arc7mx_inner.sh         in-builder build script (+arc7mx, resume
                            mode VXK_ARC7MX_CLEAN)
vllm_xpu_kernels/mx_attn_interface.py   python API (mirrors
                            flash_attn_interface decode branch incl.
                            build_decode_split_plan semantics)
_vendor_anchors/            (untracked) pulled vendor ground truth:
                            cutlass-sycl cute/cutlass headers, the fmha
                            app sources, paged_decode_kernel.hpp,
                            chunk_prefill_mainloop.hpp (the fork
                            parent) — enables local structural
                            verification without rig round-trips
```

## Integration surface (verified against the tree's bytes 2026-11-19)

The xe_2 kernel library GLOBs `*.cpp` recursively
(`cmake/utils.cmake::add_xe2_kernel_library`), so the module joins the
build with ZERO CMake changes. `apply_arc7mx.py` makes exactly these
edits (all pure insertions, anchor-verified in a local sandbox run):

- `csrc/xpu/attn/attn_interface.h`: `cutlass_mx_paged_decode_interface`
  declaration after the stock decode decl.
- `csrc/xpu/attn/attn_interface.cpp`: mx include inside the XE2 ifdef +
  the arch-guarded forwarder (vendor layering: fa2 op -> attn_interface
  -> xe2 entry).
- `csrc/flash_attn/flash_api.cpp`: `mx_paged_decode_op` in
  FLASH_NAMESPACE (queue via `vllmGetQueue`; decode buffers undefined --
  the xe2 entry allocates per the vendor shapes) + the
  `mx_paged_decode` TORCH_LIBRARY def/impl via `make_pytorch_shim`
  (same int/float/optional conversions as `varlen_fwd`).

Wheel version `+arc7mx` (the `+arc7` label belongs to the P6.34 jm lab
line): arc6-certified content + the mx module only. Base tarball
`/root/build/vxk-arc6-cert22-src.tgz` (taken pre-jm == exactly the
certified wheel's source, .deps warm cache carried).

## vllm side (fork tree)

`vllm_gaudi/xpu/patches/mx_kv/native_decode.py` bridges the engine's
split-row plan to the kernel's contract (one row per (request, draft
token) = its own one-query "sequence": cu_seqlens_q = arange(R+1),
seqused_k = the plan's per-row limits, block_table = the per-row
gathered tables; all device ops -- capture law; max_seqlen_k from the
metadata's host-known field). `engine.py` gates it behind
`ARC_MXKV_NATIVE=1` (DORMANT until certified; Triton v2/v5 stay the
default; G>16 refuses with a warning) on the `_split_verify_decode`
path -- the MTP verify hot shape. The per-row non-spec decode path
stays Triton for the first cut (r199's money cells are the verify
shapes); it adopts native after r198 proves correctness.

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

## Status (2026-11-19, sixth pass — tv_layout coordinate law)

- First r198 run (fragment-lookup wheel) failed every face: the
  custom-extent byte copies and the stock dequant-destination copies
  distribute the same tile across work-items on different sub-grids,
  so a work-item-local coordinate match only hit grid coincidences.
- Second r198 run (coordinate-read redesign) still failed — but with a
  decisive signature: bitwise probe showed exactly the elements at
  `d % 8 == 0` correct, rest garbage. Root cause (see mx_dequant.hpp
  COORDINATE LAW): copy `partition_S` of an identity tensor returns
  WI-CANONICAL coordinates in this tree (the SG-scoped copy atom
  exposes the subgroup, not the work-item, as its thread mode — the
  vendor's own remainder mask compensates with `+ local_id`).
- Fix: per-WI positions come from the DESTINATION fragment's
  `tv_layout()` — `partition_sg_fragment_D` builds it over the
  tidfrg_D (thr, val) mapping whose codomain IS the tile's 2D
  coordinate space (ScaledBasis strides; instantiated K-lane evidence:
  row = t1(wi) + 8*v0(dv), elem = t0(wi) + 2*v1(dv) + 16*v2(dv),
  injective over the tile). `tv_coord` evaluates it at flat
  (work-item, register) via `idx2crd` against the rank-3
  (Thr, Val, Rest) nesting; the tile origin is anchored to the
  identity slot-0 tuple minus its own tv position (self-correcting).
  QK B-tile frame mode 0 is the row (k); PV B-tile mode 0 is the
  element (v) — `kFrame0IsRow` selects.
- mx TU compiles clean through the redesign (preflight RC=0).

## Softmax-path precision law (2026-11-19/20, probe + ledger decomposition)

The tv_layout fix cleared bitwise to 0/1024 on ALL formats (after
canonicalizing +-0: a -0.0 e2m1 value (nibble 8, 1/16 of elements)
accumulates as (+0)+(-0)=+0 inside the PV DPAS — IEEE RN sums
opposite-signed zeros to +0; sign-of-zero is an accumulation
property, not a dequant property). The residual exact-face failures
(4-9e-3 on boundary-lens payloads, d_v2 ~ 3e-7) decomposed via the
ONE-HOT-V PROBE (v_t = e_t quantizes exactly on every lane, so
O == P bit-visible; len 15 = one k-tile, no rescale):

- max|dp| == max|p - bf16(p)| exactly (1.71e-3 mxfp8) — the P-bf16
  cast (the shared XE_DPAS_TT<..., float, bfloat16_t> PV atom) is the
  DOMINANT term;
- a 10-14% tail (n(dp > 1.5*cast) = 13-17/120) beyond the cast —
  sycl::native::exp2 carries ~1e-3 relative error (vendor stock uses
  it in both fmha mainloops); fixed by precise `sycl::exp2`;
- certified Triton v2 (precise exp, fp32 dot) matches the fp64 oracle
  at 3e-7 — the plan's r198 gate (d <= 2e-3) is v2's bar.

The tf32 PV atom (interim fix, run 9-12) shrank the walk but the
ELEMENT-LEDGER census exposed the surviving structure: every
max-diff element sits at steps=1.00 — got and the pure-cast law
C = bf16(p64@v64) are the two ADJACENT bf16 grid points around ref
(a rounding FLIP), while a 12-23 element tail per cell runs >1 step
at small magnitudes (the 2^-11 P-walk at cancellation elements, where
O nearly vanishes but the per-term errors do not). THE FLIP LAW:
whether the gate passes depends on WHICH magnitude flips —
random549's native-v2 9.6e-4 is exactly one step at [0.125,0.25),
passing by luck; the same flip at [1,2) is 7.8e-3 and fails. A
payload-independent gate needs pre-cast agreement with v2 to
SUB-HALF-ULP, which no P-cast short of fp32 provides.

Final fix (deviation from stock, documented in-tree): P rides the PV
gemm as a SPLIT bf16 TRIPLE — p = bf16(p) + bf16(r1) + bf16(r2),
r1 = p − bf16(p), r2 = r1 − bf16(r1) — holding p to 2^-27 relative
(below fp32's own accumulation noise, i.e. v2-grade; the flips and
the walk both vanish, and the PAIR's 2^-18 floor — which the
mini-ledger showed as 1-2% of |O| at single deep-cancellation
elements, e.g. ref 2.8e-6 from 0.3-magnitude terms — is gone). Every
gemm uses the full-rate bf16 atom, so the interim tf32 atom is
reverted and both MMAs again share one `XE_DPAS_TT<M, float,
bfloat16_t>`; no fp32-operand DPAS exists on Xe2 anyway. The softmax
keeps the precise `sycl::exp2`. Perf priced at r199.

Companion harness law (same run): the element-local 1-ulp rule
accepts being within one rounding of EITHER the fp64 truth or the
CERTIFIED v2 decoder's output — at deep-cancellation elements even
the certified fp32 pipeline diverges from fp64 by 2+ grid steps
(mini-ledger len17: v2 itself 2 steps off bf16(ref)); the v2 side
holds the kernel to v2's own rounding behavior, zero free parameters.

Harness laws hardened the same day: element-local 1-ulp rounded-match
(the global 2^-8*max|ref| proxy underestimates an element's own ulp
by up to 2x — every marginal cell sat at exactly 1 ulp of the diff
element's binade) and the v2-calibration chain (|native-ref| <=
2e-3 + |v2-ref|: composes the native-v2 face law with v2's own
deviation; zero free parameters).

- `mx_types.hpp`, `mx_dequant.hpp` — complete. `dequant_dtile` is the
  coordinate-driven design (see below): per-destination-value scale
  reads from the gmem e8m0 surface via the partitioned coordinate
  views, 4-entry per-WI register cache, byte lookup by coordinate
  match against the byte fragment.
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
  shapes; undefined buffers get the vendor shapes here, so the fa2 op
  passes undefined tensors), fmt x qgroup runtime dispatch.
- `mx_paged_decode_mainloop.hpp` — COMPLETE AUTHORING: the MX load
  path is fully plumbed and structurally verified 1:1 against the
  pulled fork parent (`_vendor_anchors/.../chunk_prefill_mainloop.hpp`
  DecodeFwdMainloop). Byte tiles (gK/gV over the u8 surfaces,
  custom-extent B-copies `make_block_2d_copy_bwidth/_bheight`) pair
  with element tiles (gKD/gVD, stock-verbatim shapes) per DB==DD;
  dequant-destination copies (copy_kd/copy_vd) provide the fragments
  + coordinate views `reorder()` feeds; the identity-tensor +
  data-constructed-TiledCopy pattern is stock's own. Remaining
  compile risks (r198 iterates): TiledCopy-from-recast-proxy ctor
  acceptance, narrow byte-tile atom coverage, tv_layout preservation
  through `partition_sg_fragment_D`.
- `mx_attn_interface.py` — python surface (split-plan mirror;
  seqused_k rides the cu_seqlens_k slot per the vendor decode flow;
  `_vllm_fa2_C` import guard per the arc4 dlopen law).
- Integration + build layer AUTHORED AND SANDBOX-VERIFIED
  (2026-11-19): `apply_arc7mx.py` (all six anchor edits applied
  byte-exact in a local sandbox against the tree's pulled bytes;
  idempotence-refuse verified), `vxk_arc7mx_bootstrap.sh` /
  `build_vxk_wheel_arc7mx.sh` / `vxk_arc7mx_inner.sh`.
- vllm side AUTHORED: `native_decode.py` bridge + `engine.py`
  `ARC_MXKV_NATIVE=1` wiring (verify-batch path; dormant by default;
  ruff-clean under the repo face). NOT yet compile-iterated: the first
  engine boot with the env set is r200's fastcycle face (r198 tests
  the kernel offline first).
- NEXT: rig compile iteration (r198) — ship the kit, run apply +
  build, resolve the remaining compile risks against real compiler
  errors, then the validation ladder below.

### r198 compile-iteration postmortem (rig, 6 rounds, ~50 s/round)

The `arc7mx_preflight_mx.sh` loop (compile ONLY the mx TU via its exact
compile_commands.json command, `-o` redirected to /tmp; the arc8
pre-flight-the-changed-TU law) drove six error classes to zero:

1. **CMake FindPython** — fresh omix builder lacked `Python.h`:
   `python3.12-dev` apt-installed in `vxk_arc7mx_bootstrap.sh`.
2. **Missing include** — `cutlass::device_memory` (workspace
   allocation) needed `cutlass/util/device_memory.h` (resolves from
   the `tools/util/include` extra root).
3. **Transcription parens** — 2 unclosed parens (SGPerWG / FragC
   sites) vs the vendor anchor.
4. **constexpr member-call law** — `Int<get<0>(select<1,2>(
   mma.tile_mnk()))>` is ill-formed (member call in a constant
   template arg); the stock pattern passes the member-call value
   through UNwrapped and only the custom extent as `Int<Width>{}`
   (4 sites: bwidth/bheight tile_nk + cB).
5. **cute recast is tensor-level** — no iterator recast; the
   dequant-destination copy proxies are raw-pointer views:
   `make_tensor(make_gmem_ptr(reinterpret_cast<ElementQ*>(
   raw_pointer_cast(K_2D_0.data()))), K_2D_0.layout())`
   (`static_cast` cannot cross `unsigned char*` -> `ElementQ*`).
   The tensor-taking TiledCopy ctor consumes only value_type +
   stride; these copies never issue loads (identity partitions
   supply coordinates).
6. **Tree-local member laws** — `SubgroupTensor::size()` is
   non-static (call `dst.size()`, never `DstFrag::size()`), and
   `bfloat16_t(float)` is explicit (element writes need
   `typename DstFrag::value_type(val * scale)`).

Debug law learned the hard way: a clang "candidate template
ignored: substitution failure" with NO reason text means an earlier
hard error was deduplicated — diagnose from the FIRST error in the
FULL output (`PREFLIGHT_TAIL=4000`), never the tail window.

### Design notes (fourth pass)

- **Scale access is coordinate-driven, not broadcast.** An earlier
  design staged the tile's e8m0 codes into a per-WI float fragment
  via ~tile_k*groups subgroup broadcasts per d-tile — ~8x the MMA
  cost. Instead each destination value computes (row, group) from its
  partitioned coordinate and reads `scales(row, group)` directly
  (L1-resident; 4-entry register cache hoists repeats). The same
  coordinate views drive the payload byte lookup, so dequant needs no
  scale copies, fragments, or cross-lane traffic.
- **bdpas (hardware blockscaled MX MMA) is Xe3-forward, not BMG.**
  cutlass-sycl carries `xe_mma_blockscaled_mxfp.hpp` (e2m1 + ue8m0
  zip tensors into `bdpas` DPAS), but the atoms are gated
  SYCL_INTEL_TARGET==35 with Xe35 reorder trees. On BMG/Xe2 the
  in-register dequant->bf16 MMA (this design) is correct; bdpas is
  the future-hardware rebase lane (banked, master plan Phase 8).
- **BMG lane follows the vendor fp8 law**: Xe2 has no native MX/fp8
  MAC — the vendor fp8 path loads bytes and casts in-register through
  `reorder()`; MX dequant inserts at exactly that site.

## Validation ladder (master plan §P6.47 Phase 8)

r198 offline correctness (oracle: identical quantized inputs, bitwise
vs Triton dequant; gate d<=2e-3 vs v2 reference) -> r199 T1 perf A/B vs
v5 at the money cells -> r200 e2e fastcycle (ARC_MXKV_NATIVE=1) ->
r201 full §10 battery -> cert26. Kernel ships dormant behind
`ARC_MXKV_NATIVE` until certified; Triton v2/v5 stay the default.

## Build laws

oneAPI 2026.0.0 toolchain, `vxk-arc3-builder` host-swap (live arc6+jm
tree kept at `/root/build/vxk-arc7jm-kept`), `.deps` carried from the
certified base tarball, wheel version `0.1.14.1+arc7mx`. Same ABI laws
as arc3-arc6 (`external_patches/vllm-xpu-kernels/README.md`).
