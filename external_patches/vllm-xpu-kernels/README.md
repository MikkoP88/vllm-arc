# vllm-xpu-kernels — int64 pool-offset fix (wedgefix-v88 lineage)

## What

`patch_vxk_int64fix.py` widens the 6 bare `states_id * conv_states_stride_0`
int32 multiplies in the GDN conv-state kernels to int64, then
`build_vxk_wheel_arc2.sh` (+ in-container `vxk_arc2_inner.sh`) rebuilds the
wheel INSIDE the arc runtime image with the staged DPC++ 7.2.0 toolchain,
version `0.1.14.1+arc2`.

## arc1 postmortem (why arc2 exists — ABI lesson, banked)

arc1 was built in the omix devel image (oneAPI 2025.3): the wheel links
`libsycl.so.8`, while the arc image venv runs `intel_sycl_rt 2026.0.0`
(`libsycl.so.9`) → `ImportError: libsycl.so.8: cannot open shared object
file` at wheel load inside the gate container. **Never reuse a wheel across
SYCL runtime majors — build against the target runtime image.** (A so.8→so.9
symlink or dual-runtime mount was rejected as unevidenced customization.)

arc2 recipe (validated fixes over arc1's flow):

- builder `FROM vllm-arc:v0.30.0-b70-p12` (target runtime, torch 2.13.0+xpu
  and all pip build deps preinstalled — cmake 4.4.4, ninja, setuptools-scm).
- toolchain `/root/build/dpcpp72/dpcpp` (DPC++ 7.2.0 = clang 23.1.1,
  intel/llvm 5680988, docker-cp'd from esimd-w11 — the W14-recipe toolchain
  whose output is proven to run on the so.9 venv family). Symlinks
  `icpx→clang++`, `icx→clang` inside our private copy (setup.py probes
  `$SYCL_HOME/bin/icpx -v` and resolves `which icx/icpx` for CMake).
- `CMPLR_ROOT=/src/dpcpp` — vxk `cmake/toolchain.cmake` derives the CMake
  compilers as `${CMPLR_ROOT}/bin/{icx,icpx}`; unset, it produces `/bin/icpx`
  and dies.
- `XPU_SYCL_COMPILER=dpclang` + `PKG_CONFIG_PATH=/src/dpcpp/lib/pkgconfig` —
  torch's `FindSYCLToolkit.cmake` default icx parser expects an Intel(R)
  banner and CMake `math()`-dies on the raw clang banner; the dpclang parser
  matches this toolchain's "DPC++ compiler 7.2.0" banner (mapped 2019+7 =
  oneAPI 2026.2) and consumes the shipped `sycl-dpcpp-7.pc`.
- venv-first `LD_LIBRARY_PATH` (venv torch libs, then `/src/dpcpp/lib`);
  `TORCH_XPU_ARCH_LIST=bmg`, `MAX_JOBS=52 KERNELS_MAX_JOBS=52`,
  `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc2`, clean `build/`,
  `pip wheel --no-build-isolation --no-deps`.


## Why (audit evidence — results/2026-10-27-p1-vxk-wedge-audit/)

The §10 P1 gate row requires "vxk wedge-site audit clean (or v88 wheel
rebuilt + device-gated, §6)". The audit was missing from the P1 leg record;
run 2026-10-27 it returns **NOT CLEAN** for the pinned release:

- image pins `vllm-xpu-kernels==0.1.14.1` (PyPI, Dockerfile stack-swap step).
- PyPI ships only a binary wheel (no sdist), released 2026-08-28.
- the `v0.1.14` tag source carries 6 bare sites (2 per file: causal_conv1d.hpp
  :232/:452, chunk_causal_conv1d_xe2.hpp :203/:627,
  chunk_causal_conv1d_tiled_xe2.hpp :220/:585), zero `static_cast<int64_t>`.
- zero commits touched these headers 2026-08-01→2026-09-15 (GitHub API) —
  the 0.1.14.1 release sits inside that window, so its source is the tag's.
- `v0.1.15` is equally bare (fix never landed upstream through 0.1.16.1).

Overflow condition: stride(0)=524,288 × id in int32 wraps at id ≥ 4097 →
silent corruption (mapped) or DEVICE_LOST (unmapped). mamba_pool_ratio 2
pushes dense qwen3.8-27b pool rows past 4096 — the target model IS exposed
(plan §6). Every real prefill ≥ 8 tok takes the tiled path.

This is a SEPARATE defect from the p2r3 defect-b capture-zombie (command-
graph event errors at first request) — see the audit VERDICT for the
discrimination; the zombie is not the wedge.

## Provenance / scope law

Port of donor kit `C:\Projects\vllm-iexperimental\vllm\patches\prod\
external-deps\wedgefix-v88\` (read-only import, plan §6 names this exact
kit as the contingency mechanism). Differences from donor: source root is a
parameter (donor hardcoded /root/build/vxk at the 0.26 pin); expected site
count 6 (upstream refactored the 0.26 pin's 8 → 6 before v0.1.14; the
donor's own docstring records the 6→8 history in the other direction);
torch pin 2.13.0+xpu (donor built 2.11.0+xpu); wheel version
0.1.14.1+arc1. Backup mark `[VXKI64]`, backups `.vxki64bak`.

## Validation (§6 contingency gate — all required before image pin swap)

1. patcher: `TOTAL_SITES_PATCHED=6`, re-audit `--audit` → CLEAN.
2. standalone device repro (donor `repro_v88_int32.py`, cap727.pt, rig
   /root/build/lce1/): exit 0 CLEAN WHEEL — LO+HI+BND, noseed AND seeded.
3. engine gates on the rebuilt image: serialized-24 + burst_harsh ×3
   (donor `v88d_wedge_validation.sh` shape), zero new dmesg "Engine reset".
4. k1k sanity cells unchanged (no perf regression from the int64 widen).

Results in results/2026-10-27-p1-vxk-wedge-audit/ and the follow-up wheel
leg dir.
