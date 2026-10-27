# vllm-xpu-kernels — int64 pool-offset fix (wedgefix-v88 lineage)

## What

`patch_vxk_int64fix.py` widens the 6 bare `states_id * conv_states_stride_0`
int32 multiplies in the GDN conv-state kernels to int64, then
`build_vxk_wheel_arc3.sh` (+ in-container `vxk_arc3_inner.sh`) rebuilds the
wheel in `intel/omix:0.2.0-devel-ubuntu24.04` (oneAPI **2026.0.0**) against
a venv pinned to the target stack torch 2.13.0+xpu, version
`0.1.14.1+arc3`.

## Wheel ABI lessons (arc1/arc2 postmortems — banked)

The arc image venv runtime is `intel_sycl_rt 2026.0.0` + `intel-cmplr-lib-rt
2026.0.0` (PyPI release train 2026-04-24; `libsycl.so.9`). Two failed
builders bracket the requirement:

- **arc1** (omix 0.1.0-devel = oneAPI 2025.3): wheel linked `libsycl.so.8` →
  `ImportError` at load. **A wheel never crosses SYCL runtime sonames.**
- **arc2** (DPC++ 7.2.0 = oneAPI 2026.2, staged from esimd-w11): linked
  so.9 ✓ but emitted calls to `sycl::detail::reportAdjustedRange`, which the
  2026.0 runtime does not export → `undefined symbol` at import. **The
  COMPILER GENERATION must equal the RUNTIME GENERATION** — a newer compiler
  references helper symbols the older runtime never had. (Older compiler +
  newer runtime would be the compatible direction but is unreachable here:
  2025.3 emits so.8 references — arc1.)

arc3 = the D-lineage v78/v88 recipe shape (rig `build_vxk_v78.sh`: oneAPI
setvars + venv + `pip wheel --no-build-isolation --no-deps`) with two pin
changes: oneAPI 2025.3 → **2026.0** (omix 0.2.0 = "Intel(R) oneAPI DPC++/C++
Compiler 2026.0.0 (2026.0.0.20260331)", per Intel's OMIX support matrix:
0.1.0→2025.3, 0.2.0→2026.0, 0.3.0→2026.1) and torch 2.11.0+xpu →
2.13.0+xpu. The PyPI 0.1.14.1 wheel imports cleanly in the arc venv — Intel
CI built it with this same generation — so generation-matched linking is
proven sound by that precedent.

arc3 builder environment notes (all diagnosed live, 2026-10-27):

- `set -u` kills the SHELL inside `source setvars.sh` (compiler
  `env/vars.sh:258` references unset `OCL_ICD_FILENAMES`; an expansion error
  aborts before `|| true` can run) — guard with `set +u` around the source.
  This is why the v78 recipe used plain `set -eo pipefail`.
- omix ships a hollow `/opt/venv` (python symlinks, no pip) — use our own
  `/opt/arcvenv`.
- Ubuntu's ensurepip needs the `python3.12-venv` deb (not installed) —
  create with `python3 -m venv --without-pip` + `get-pip.py` bootstrap.
- astral.com (uv) is unreachable from the rig; pypi/pytorch indexes are not.
- torch `FindSYCLToolkit.cmake` default `icx` parser works with the real
  Intel banner (no `XPU_SYCL_COMPILER=dpclang` workaround — that was only
  needed for arc2's raw-clang toolchain).
- `rm -rf build .deps` between attempts: incremental reuse across compiler
  generations is an ABI hazard.



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
