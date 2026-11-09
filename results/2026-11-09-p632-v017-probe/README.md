# P6.32 item 8 — vllm-xpu-kernels v0.1.17 Phase-2 build-only feasibility probe

Date: 2026-11-09 (build 04:41–07:00 UTC, ~2h20m, rig `vxk-arc3-builder`)

## Question

Does upstream vllm-xpu-kernels **v0.1.17** (the current release line,
newer than our pinned v0.1.14.1 base) build in our certified BMG builder
against torch 2.13.0+xpu — i.e. is a future Phase-3 rebase of the arc
wheel line mechanically feasible without a toolchain migration?

Probe script: `tools/ar/vxk_v017_inner.sh` (banked; v0.1.17 source tree
at `/root/build/vxk-v017-src` on the rig, built in-place through the
builder bind).

## Result: AMBER (core feasible; attention libs missing from wheel)

- Build **GREEN**: rc=0, wheel 389,042,809 B,
  `vllm_xpu_kernels-0.1.17+probe-cp312-cp312-linux_x86_64.whl`
- sha256 `813d211cfc3730c38c60a075e40e7b2c00fb72255f7d4a71c18da300b1b69d1a`
  (persisted at rig `/root/build/wheels-v017/`; >100 MB — never tracked,
  per the artifact law)
- Import smoke: version `0.1.17+probe`, torch `2.13.0+xpu`, `_xpu_C`
  loads; core GEMM ops present: `fp8_gemm_w8a16`, `fp4_gemm`,
  `int4_gemm_w4a16`, `fp8_gemm`
- **Gap**: `gdn_attn` / attention shared libs NOT in the wheel
  (`gdn_libs []`, `attn_libs []`). v0.1.17 ships attention as a separate
  build component our builder flow does not produce (new CMake deps or
  subproject flags in the 0.1.17 layout — to be identified at Phase-3
  time from its CMakeLists).

## Disposition

- Phase-3 rebase (as its own plan item, if/when pursued): feasible for
  the GEMM core — no toolchain migration needed, oneAPI 2026.0.0 +
  torch 2.13 suffice. Before any rebase, the missing attention-component
  build path must be identified and closed, else the rebase drops
  gdn_attn coverage vs our current 0.1.14.1+arcN line.
- Our line does NOT hard-depend on vxk attention libs today (GDN lane
  rides the radiance-derived components in `vllm_gaudi/xpu/`), so the
  gap is a rebase-time blocker, not an incumbent defect.
- Steady state unchanged: Phase-1 backport kit onto 0.1.14.1 remains the
  certified line; this probe closes P6.32 item 8 as AMBER/banked.

## Rig homes

- source tree: `/root/build/vxk-v017-src` (builder bind view
  `/src/vxk-v017`)
- build log: `vxk-arc3-builder:/root/logs/v017_build.log` (markers
  `VXK_V017_*`)
- wheel: `/root/build/wheels-v017/vllm_xpu_kernels-0.1.17+probe-...whl`
