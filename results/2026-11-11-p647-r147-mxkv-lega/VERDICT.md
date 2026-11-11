# P6.47 leg A — MXFP4/MXFP8 KV-cache offline XPU validation (r147)

**Date:** 2026-11-11 · **Phase:** P6.47 (MX KV-cache-dtype lane) ·
**Gate:** `tools/ar/r147_mxkv_legA.sh` → `tools/ar/p147_mxkv/p147_mxkv_offline.py`
· **Image:** vllm-arc:v0.30.0-b70-cert23 (offline docker run, XPU devices,
no engine boot; W14 quiesced for the window and restored after — verified
`W14_RESTORED` in every gate log)

## VERDICT: **PASS** (rc=0, all faces green, both formats)

## What was validated

The from-scratch MX (OCP microscaling) KV-cache primitives
(`vllm_gaudi/xpu/patches/mx_kv/`: mx_layout, mx_quant reference,
triton_mx_store, triton_mx_decode) on the rig's B70 XPU device:

| Face | Gate | Result |
|---|---|---|
| ref | reference self-test: round-trip cos/rel-RMSE, zero rows → zero payload, ceil_log2_exact property oracle `2^e >= r > 2^(e-1)` on 1M logspace pts | PASS (mxfp8 cos 0.99966 / 0.0263; mxfp4 cos 0.99326 / 0.1160; naive log2 disagrees on exactly 1/1M pts — ulp edge, property holds) |
| bytes | **store-kernel BYTE LAW**: kernel bytes == reference bytes over 512 slots × adversarial rows (denormal group, zero group, single outlier) | PASS — **0 byte mismatches, both formats** |
| decode | kernel attention vs fp64 exact on the SAME dequantized K/V (isolates kernel correctness) | PASS — maxdiff ≤ 2.7e-7, cos = 1.0000000 |
| accuracy | vs fp64 attention on unquantized K/V (format fidelity, empirically-calibrated floors 0.9985/0.985) | PASS — mxfp8 cos 0.99928, mxfp4 cos 0.98830 |
| robust | denormal / outlier / zeroed / signed tensors; boundary lens 1/63/64/65; fragmented shuffled block tables; zero-len NaN hardening | PASS — all vs-exact c=1.0, no non-finite |
| telemetry | store/decode timings + effective GB/s at 1K/8K/32K | banked (below) |

## Leg-A findings (fixed in-flight, kept for the record)

1. **Annotation-form constexpr globals rejected** by this Triton
   (`_EMAX_FP8: tl.constexpr = 448.0` at module scope) — constexpr
   params with call-site literals are the supported pattern.
2. **Oracle correction:** `torch.ceil(torch.log2(r))` crosses a ulp
   edge on ~1/1M points; the bit-derived `ceil_log2_exact` is exact —
   the defining property is the gate, not naive agreement.
3. **Nibble-pack weight bug:** `arange(0,2)*16` = `[0, 16]` annihilated
   the low nibble (89% byte mismatch); shift semantics `[1, 16]`
   (`arange*15+1`) is the correct pack. Found by the one-slot
   divergence probe (`p147_dbg.py`), fixed, re-run green.
4. **Harness:** zero-length sequences are not an engine case —
   build_paged guarantees len ≥ 1; the kernel additionally hardens
   `acc/l_i` against 0/0 → defined zeros.

## Telemetry (first always-loop datapoint)

| fmt | ctx | store ms | decode ms | effective GB/s |
|---|---|---|---|---|
| mxfp8 | 1024 | 0.070 | 0.333 | 6.5 |
| mxfp8 | 8192 | 0.067 | 2.405 | 7.2 |
| mxfp8 | 32768 | 0.219 | 9.594 | 7.2 |
| mxfp4 | 1024 | 0.068 | 0.409 | 2.7 |
| mxfp4 | 8192 | 0.071 | 2.874 | 3.1 |
| mxfp4 | 32768 | 0.175 | 11.379 | 3.1 |

**Loop observation (queued, not blocking):** at the harness's tiny grid
(B×Hq = 2×8 = 16 programs) the decode kernel is occupancy-bound, not
bandwidth-bound (7.2 GB/s effective vs ~230 GB/s class HBM) — the
engine leg must size the grid against real B×Hq and queue a split-KV
variant if small-batch decode shows the same ceiling. Store cost is
negligible (prefill path).

## Standing laws applied on this lane (owner directives 2026-11-11)

- Corruption/garbage validation is END-TO-END (live engine, both
  models, suite-13 + suite-14 class) — offline faces here prove the
  kernels, not the release bar.
- All performance reports run comprehensively (full context ladder per
  the P6.46 standing directive, both models, every posture).
- Always-loop: telemetry above feeds the next-bottleneck analysis.

Artifacts: `rig/p147_offline.json` (machine verdict),
`rig/legA_summary.txt`, `rig/logs/legA.out` (full harness output),
gate logs attempts 1–6 (the failed attempts are the finding record).
