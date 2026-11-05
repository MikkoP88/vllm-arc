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
| `arc-fp8-allreduce.patch` | 5 files, ~600 lines: NEW `allreduce_large_sycl_ring_fp8.hpp` (e4m3 wire for the BMG TP2 fp16 SUM large-ring: K1 amax → per-chunk [K2 quantize → p2p barrier → K4 reduce+quant] → K5 dequant; slot first half = input, tail half = result, 16-byte scale area at slot-0 tail; `res_scale = 2·max(s_local, s_peer)`; both ranks' outputs carry exactly one e4m3 round trip; stock slot rotation and barrier count preserved; graph-capture safe — no host reads unless STATS). Converters are branchless fp32-native (~8 ops, exact rare path below 2⁻⁶; zeros encode to 0 for sparse activations) — bit-exact vs an exhaustive RTNE grid (`results/2026-11-05-p624-fp8-ar/fp8_noise_sim.py`, 630k values, 0 mismatches). **Peer-memory stores are u32-packed (4 codes/store)** — the r97v3 sector law: byte-granular stores to the peer's IPC-mapped memory pay a sub-sector RMW (0.41 GB/s measured); packed restores 9.6 GB/s. K1 runs a 16384-thread register-max grid (≤64 CAS). STATS adds per-stage timing. Dispatch guard in `allreduce_large_sycl.cpp` BMG no-tmp branch (`fp16 ∧ SUM ∧ N==2 ∧ count/2 ≥ min-count`, `if constexpr` half-only); env trio registration (`CCL_SYCL_ALLREDUCE_FP8`, `..._FP8_MIN_COUNT`, `..._FP8_STATS`) in `vars_experimental.hpp`/`env.hpp`/`env.cpp` (defaults 0 / 2097152 / 0 = silent off). |

**r97 verdict (2026-11-05): GATE PASS — 1.65× at the dominant
76,677,120 B (7.801 → 4.731 ms, 16.2 GB/rank/s), 1.57–1.69× across all
large sizes; FP8=0 control identical to the r95 anchor; numerics at the
bit-exact e4m3 noise floor (rel-rms 0.02285 vs simulated 0.022861).**
r98 quality ladder PASS (task-level parity, one env line A/B);
r99 cert10 release gate PASS (default posture cert9-identical with the
env proven absent; FP8=1 leg beats the cert9 baseline on EVERY p65 face:
U8k −13.0 % / U64k −8.2 % / U32k −3.8 % / C32k −4.6 %); r100 bottleneck
scan closed every remaining door with wire-rate measurement — **v4 is
FINAL** (writes 9.85 GB/s flat across store widths = the fabric ceiling;
reads 4.54 GB/s = 2.2× slower, read-reduce dead; concurrent rd+wr
collapses to 1.49 GB/s, overlap dead; v4 = 82 % of the 3.90 ms
wire-bound floor with the remainder in measured-floor local stages).
Full chain: `results/2026-11-05-p624-fp8-ar/` (VERDICT.md — r96→r100
sections, r97_debug_ledger.md — the v1→v4 capture-first iteration
history, r98/r99/r100 evidence dirs).

Patch applies with `patch -p1` at the SUPERPROJECT root (submodule files
are plain files in the work tree).

| `arc-fp8-probe.patch` | **Diagnostic only — NEVER enters a cert image.** 4 files: the one-shot IPC wire-rate probe (`CCL_SYCL_ALLREDUCE_FP8_PROBE=1`) inside `allreduce_large_fp8_ring` before the first algorithm run — remote stores at 4/8/16 B widths, remote loads at the same widths, concurrent rd+wr, local write/read references; 5 warm + 20 timed reps over the REAL peer IPC mapping; loads accumulate into device scratch with a never-taken store (elision-proof). The r100 decision instrument. |

## Runtime knobs (silent-off; serve chain sets nothing by default)

```
CCL_SYCL_ALLREDUCE_FP8=1          # enable (0 default)
CCL_SYCL_ALLREDUCE_FP8_MIN_COUNT=2097152  # per-rank elems below -> stock
CCL_SYCL_ALLREDUCE_FP8_STATS=1    # per-call diagnostics (PERTURBS: syncs)
```

The min-count default also keeps every decode-sized and graph-replayed AR
on the stock path (decode ARs ≈ 14k elems ≪ 2M), so XPU-graph capture
posture is unchanged.

## Build law (three modes: `stock | arc | probe`)

`tools/build_oneccl_arc.sh <mode>` (rig host): pinned fetch of tag
2022.0.0 + submodule + hash verify → per-mode patch state
(marker-proven, E-law: mode `stock` = pristine tracked tree; mode `arc`
= v4 patch; mode `probe` = v4 + probe patch — and going `probe`→`arc`
REVERSE-applies the probe patch whole and verifies absence) → cmake+
build inside the DLE 2026.0.0-devel-ubuntu22.04 builder
(icpx; `ENABLE_MPI=OFF`, tests/examples off; canonical artifact paths —
the fat SYCL lib is `./libccl/src/libccl.so.1.0`, the 6 MB CPU-fallback
under `./libccl_cpu` must never shadow it; STABLE `/p624` root mount —
container mounts freeze at creation) → staging to `stage-<mode>/` +
sha256 manifest (the mode's patch rows included).

Overlay target on the certified image: `/opt/venv/lib/libccl.so.1(.0)`
(the 210 MB fat library; sole consumer = `libtorch_xpu.so` via DT_NEEDED
on both sonames — recon 2026-11-05). The thin `.so.2` shim and headers
are byte-identical and untouched. **cert10 ships the mode-`arc`
product as a two-file overlay (fat lib at BOTH `.so.1` and `.so.1.0`)
on cert9's engine — manifest row `oneccl=2022.0.0+arc1`** (recipe:
`results/2026-11-05-p624-fp8-ar/r99/cert10ctx/cert10.Dockerfile` =
cert9 + overlay + manifest rows; the exhaustive `vllm_gaudi/` md5 walk
proving engine identity is banked in the same evidence dir). **The
210 MB fat-lib binary itself is NOT tracked** — GitHub's pre-receive
hook hard-rejects >100 MB files (GH001, push declined 2026-11-05);
identity = the sha256 pinned in
`results/2026-11-05-p624-fp8-ar/r99/cert10ctx/ARTIFACT-libccl.md`
(durable copies: rig `stage-arc/` + inside the certified image, both
sha-verified).

## Telemetry (three layers)

1. engine — existing ARC_AR_TRACE census + ARC_STEP_TRACE (unchanged);
2. patch — `CCL_SYCL_ALLREDUCE_FP8_STATS=1`: per-call line (count, chunks,
   amax, peer scale, res scale) + per-stage timing (k1/k2/barrier/k3/k4/k5
   via q.wait() chrono); declared perturbation, diagnostics only;
3. fidelity — phase-0 stock-rebuild race vs the 7.78 ms r95 anchor
   (`tools/ar/ar_ceiling.py --role dist --guard randn --label ...`).

## Gates (pre-registered, plan P6.24) — ALL PASS, lane closed

- r96 phase 0: stock rebuild races the 7.78 ms anchor within ±3%, else STOP — **PASS** (0e7270cc evidence);
- r97 phase 1: FP8=0 control == phase-0; FP8=1 gate **≥1.3×** (≤6.0 ms at
  76,677,120 B), numerics green — **PASS 1.65×** (rel-rms bar
  re-registered 0.035 from the bit-exact noise floor 0.0229, 5.7× under
  the corruption regime; evidence fp8_noise_sim.py);
- r98 phase 2: quality ladder (t5/t9-class prompts, agg8 + LONG-RETURN,
  parity bars vs cert9) — **PASS** (c5f08ba4);
- r99 phase 3: serve A/B (U8k/U32k/U64k), full §10 battery, cert10 image
  (metadata `2022.0.0+arc1`, md5 into IMAGE_MANIFEST, DEFAULT-OFF in the
  certified chain), docs — **PASS, cert10 = the certified release**
  (image `2f86fc2411fb`; every p65 face beats cert9; the one t1 flag
  dispositioned boot-stochastic by the r99c fresh-boot annex);
- r100 bottleneck scan: wire-rate probe + v4-rate control — **ALL DOORS
  CLOSED** (probe build control reproduced r97 at +0.0 % drift — the
  probe patch is purely additive; v4 final).
