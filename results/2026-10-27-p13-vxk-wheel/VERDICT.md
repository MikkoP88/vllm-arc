# p13 §6 contingency — vxk int64 wheel: BUILT + DEVICE-GATED CLEAN

Date: 2026-10-27 (build 18:37–22:27 UTC gate window ~22:30–23:30 UTC)
Rig: 10.20.3.REDACTED, GPUs drained (arc-test engine killed for the gate; W14
restore pending session end).

## Wheel

`vllm_xpu_kernels-0.1.14.1+arc3-cp312-cp312-linux_x86_64.whl`
(352,236,146 B, md5 `a51df187aae8efa3871b0cb9b7d3f3c7`)

- source: `/root/build/vxk-arc-src` = PyPI 0.1.14.1 source (v0.1.14 tag
  lineage, audit in results/2026-10-27-p1-vxk-wedge-audit/) +
  `patch_vxk_int64fix.py` (6 sites → int64, re-audit CLEAN).
- builder: `intel/omix:0.2.0-devel-ubuntu24.04` = oneAPI **2026.0.0**
  ("Intel(R) oneAPI DPC++/C++ Compiler 2026.0.0 (2026.0.0.20260331)") —
  the exact compiler generation of the arc venv runtime
  (`intel_sycl_rt` / `intel-cmplr-lib-rt` 2026.0.0, PyPI train 2026-04-24) —
  against `/opt/arcvenv` torch 2.13.0+xpu (pytorch XPU index).
- Build: v78/v88 D-recipe shape (`build_vxk_wheel_arc3.sh` +
  `vxk_arc3_inner.sh`), KERNELS_MAX_JOBS=52, clean build/.deps,
  1 h 30 m, rc=0.

## §6 device-gate triplet (all three required rows)

| run | wheel | verdict | exit | new resets |
|---|---|---|---|---|
| seeded | arc3 | `{'LO': True, 'HI': True, 'BND': True} -> CLEAN WHEEL` | 0 | 0 |
| noseed | arc3 | CLEAN WHEEL (REPRO_EXIT=0) | 0 | 0 |
| bare control | PyPI 0.1.14.1 | `{'LO': True, 'HI': False} -> BUGGY WHEEL` — `UR_RESULT_ERROR_DEVICE_LOST`, "DEVICE FAULT (the serve-wedge mechanism)" at HI row 4343 | 1 | +1 (device self-recovered, DEV_OK 31.7 GiB) |

The HI row (4343 ≥ 4097 = the int32 wrap region with stride 524,288) is the
discriminating cell: bare faults the device there; arc3 writes exactly the 7
expected rows, finite, conv+ssm. BND (row 4096) confirms the boundary.
Positive control captured LAST per §6 with full artifact preservation
(`gate_bare_control.log`, `repro_bare.log`).

## ABI trail (arc1/arc2 rejects — lessons banked in the p13 README)

- arc1 (omix 0.1.0 = oneAPI 2025.3): linked `libsycl.so.8` → ImportError
  against the venv so.9. Wheel never crosses SYCL sonames.
- arc2 (DPC++ 7.2.0 = oneAPI 2026.2, esimd-w11 toolchain): linked so.9 but
  emitted `sycl::detail::reportAdjustedRange` — absent from the 2026.0
  runtime → undefined symbol at import. Compiler generation must equal
  runtime generation.
- arc3: generation-matched (2026.0 ↔ 2026.0). Imports, loads, and passes.

## Image wiring (this branch)

- `external_patches/wheels.pins.md5`: arc3 md5 pin added (wheel_guard law).
- `Dockerfile`: stack swap installs the arc3 wheel; new in-build gate
  `import vllm_xpu_kernels` + `+arc3` version assert (no GPU needed —
  catches arc1/arc2-class ABI failures at image build time); manifest line
  with md5.
- `tools/build_image.sh`: copies the arc3 wheel by exact name (arc1/arc2
  rejects excluded).

**OUTCOME: PASS** — §6 contingency closed; the P1 gate row "v88 wheel
rebuilt + device-gated, §6" is satisfied by arc3. Banking authorized;
image build from integration/phase-2-attention follows, then P2 legs
(engine serialized-24 + burst×3 on the rebuilt image remain as P2 leg
rows, per the p13 README validation plan).
