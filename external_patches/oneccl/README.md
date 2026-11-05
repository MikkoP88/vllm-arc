<!-- SPDX-License-Identifier: Apache-2.0 -->
# external_patches/oneccl — oneccl 2022.0.0+arc1 (P6.24 / p624)

Pinned base + patch + build law for the fp8-wire allreduce rebuild of the
oneCCL pip runtime that torch 2.13.0+xpu pins EXACTLY (`oneccl==2022.0.0`;
torch↔oneCCL pairing law — see memory `oneccl-2022-battlemage-ar-cliff`).
Same-version rebuild ONLY: identical tag, identical C API, identical
sonames, identical sycl-rt generation (2026.0.0) — ABI-identical by
construction; never cross-version swap under torch.

## Why (evidence chain)

- P6.22 r94/r95: the dominant prefill AR is 76,677,120 B
  (7488×5120 fp16 = 38,338,560 elems) ×1,584/worker = 88% of the 27.3%
  AR-family slice of the 51.82 s r94 prefill window; stock ring = 7.78 ms
  isolated / 7.85 ms in-situ at 9.85 GB/rank/s — pure wire-bound, 2 fp16
  wire writes per chunk (`allreduce_large_su_ring_write_multi_kernel` +
  `ReduceCopyKernel`, the #215-matrix PASS row).
- One-shot lane CLOSED NEGATIVE (16.03 ms ceiling); engine-level ar_quant
  CLOSED NEGATIVE (P6.21a, +4–9% slower — overhead-only). The remaining
  lawful route is the wire itself: e4m3 halves the bytes.

## Base pin

| artifact | value | verify |
|---|---|---|
| superproject | https://github.com/oneapi-src/oneCCL @ tag **2022.0.0** | commit `1318c3aaf8a67536ff538331c418c8cea3114f33` |
| `deps/libccl` submodule | gitlink | `4ceafd15c03ce46f11eeaf91781a92afebd3cecf` |
| builder image | `intel/deep-learning-essentials:2026.0.0-devel-ubuntu22.04` | DPC++ 2026.0.0 = cert9's sycl-rt generation |

The build guard `tools/build_oneccl_arc.sh` hard-fails unless both commit
hashes match the pin.

## Patch set (apply order = series order)

| patch | content |
|---|---|
| `arc-fp8-allreduce.patch` | 5 files, ~500 lines: NEW `allreduce_large_sycl_ring_fp8.hpp` (e4m3 wire for the BMG TP2 fp16 SUM large-ring: K1 amax → per-chunk [K2 quantize → p2p barrier → K4 reduce+quant] → K5 dequant; slot first half = input, tail half = result, 16-byte scale area at slot-0 tail; `res_scale = 2·max(s_local, s_peer)`; both ranks' outputs carry exactly one e4m3 round trip; stock slot rotation and barrier count preserved; graph-capture safe — no host reads unless STATS); dispatch guard in `allreduce_large_sycl.cpp` BMG no-tmp branch (`fp16 ∧ SUM ∧ N==2 ∧ count/2 ≥ min-count`, `if constexpr` half-only); env trio registration (`CCL_SYCL_ALLREDUCE_FP8`, `..._FP8_MIN_COUNT`, `..._FP8_STATS`) in `vars_experimental.hpp`/`env.hpp`/`env.cpp` (defaults 0 / 2097152 / 0 = silent off). |

Patch applies with `patch -p1` at the SUPERPROJECT root (submodule files
are plain files in the work tree).

## Runtime knobs (silent-off; serve chain sets nothing by default)

```
CCL_SYCL_ALLREDUCE_FP8=1          # enable (0 default)
CCL_SYCL_ALLREDUCE_FP8_MIN_COUNT=2097152  # per-rank elems below -> stock
CCL_SYCL_ALLREDUCE_FP8_STATS=1    # per-call diagnostics (PERTURBS: syncs)
```

The min-count default also keeps every decode-sized and graph-replayed AR
on the stock path (decode ARs ≈ 14k elems ≪ 2M), so XPU-graph capture
posture is unchanged.

## Build law

`tools/build_oneccl_arc.sh` (rig host): pinned fetch of tag 2022.0.0 +
submodule + hash verify → `patch -p1` (mode `stock` skips; mode `arc`
applies) → cmake+build inside the DLE 2026.0.0-devel-ubuntu22.04 builder
(icpx; `ENABLE_MPI=OFF`, tests/examples off) → staging overlay of the
BUILT fat `libccl.so.1` (the fp8 implementation lives there) + unchanged
thin `libccl.so.2` C-API shim → verify pass (SONAMEs, C-API symbols,
kernel symbols, version metadata `2022.0.0+arc1`) → sha256 manifest.

Overlay target on the certified image: `/opt/venv/lib/libccl.so.1(.0)`
(the 210 MB fat library; sole consumer = `libtorch_xpu.so` via DT_NEEDED
on both sonames — recon 2026-11-05). The thin `.so.2` shim and headers
are byte-identical and untouched.

## Telemetry (three layers)

1. engine — existing ARC_AR_TRACE census + ARC_STEP_TRACE (unchanged);
2. patch — `CCL_SYCL_ALLREDUCE_FP8_STATS=1` per-call line (count, chunks,
   amax, peer scale, res scale); declared perturbation, diagnostics only;
3. fidelity — phase-0 stock-rebuild race vs the 7.78 ms r95 anchor
   (`tools/ar/ar_ceiling.py --role dist --guard randn --label ...`).

## Gates (pre-registered, plan P6.24)

- r96 phase 0: stock rebuild races the 7.78 ms anchor within ±3%, else STOP;
- r97 phase 1: FP8=0 control == phase-0; FP8=1 gate **≥1.3×** (≤6.0 ms at
  76,677,120 B), numerics recorded (cos ≥ 0.9999, rel-rms ≤ 2e-2);
- r98 phase 2: quality ladder (t5/t9-class prompts, agg8 + LONG-RETURN,
  parity bars vs cert9);
- r99 phase 3: serve A/B (U8k/U32k/U64k), full §10 battery, cert10 image
  (metadata `2022.0.0+arc1`, md5 into IMAGE_MANIFEST, DEFAULT-OFF in the
  certified chain), docs.
