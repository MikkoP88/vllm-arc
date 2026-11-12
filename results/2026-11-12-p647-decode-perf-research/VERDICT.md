# P6.47 decode-path research — can MXFP4/MXFP8 be tuned to superiority vs the native baseline?

**Date:** 2026-11-12 · **Leg:** decode-perf research (pre-A4) · **Owner directive:** "do
realy deep research and analyze is possible tune --kv-cache-dtype MXFP4 and MXFP8 types
superior versus baseline? removing all bottlenecks, improve and make more effiency
end-to-end path."

## Verdict (headline)

| Format | Can it beat native fp8_e4m3 KV? | Where | Mechanism |
|---|---|---|---|
| **mxfp4** | **YES — speed AND capacity** | long ctx (≥8K), decode-bound | 136 B vs 256 B slot → 1.88× fewer bytes per token; once the decode kernel is not latency-bound, bandwidth ratio converts ~1:1 to decode speed; capacity 1.81× more tokens (banked r154: 721,994 vs 398,363) |
| **mxk8v4** | **YES — capacity + quality-fixed speed** | long ctx | 200 B slot → 1.28× byte cut at m8q-quality keys (the m4q REP-LOOP fix); same kernel levers apply |
| **mxfp8** | **NO — parity at best, and only after tuning** | nowhere dominant | 264 B vs 256 B native slot (E8M0 scale overhead) — it carries MORE bytes than the baseline; tuning removes its self-inflicted gap but cannot pass the byte law |

The measured gap today (m4q 34.6–37.3 solo tps vs ctrl 59.9–61.6; m8q 42–44) is
**mostly schedulable, not format-intrinsic**. Evidence below.

## 1. Measured baseline (banked)

Engine (r154 gate legs, qwen champion chain @ maxlen 262144, cert24x):

| leg | KV dtype | pool tok | conc | solo tps | agg8 tps |
|---|---|---|---|---|---|
| ctrl | native fp8_e4m3 | 398,363 | 1.52× | 59.9–61.6 | 298.7–308.2 |
| m8q | mxfp8 | 386,484 | 1.47× | 42.2–44.5 | 88.0–90.6 |
| m4q | mxfp4 | 721,994 | 2.75× | 34.6–37.3 | 69.0–70.8 |

Kernel telemetry (p147 leg A, D=128, HK=2, Hq=8, decode only):

| fmt | 1K | 8K | 32K | eff. GB/s @32K |
|---|---|---|---|---|
| mxfp8 | 0.33 ms | 2.41 ms | 9.59 ms | 6.5–7.2 (unique bytes) |
| mxfp4 | 0.41 ms | 2.87 ms | 11.38 ms | 2.7–3.1 (unique bytes) |
| store | 0.07 ms | — | — | not a bottleneck |

Key anomaly: **mxfp4 decode is SLOWER than mxfp8 despite 0.52× the bytes.** A
bandwidth-bound kernel cannot behave this way — the v1 kernel is bound elsewhere.

## 2. Bottleneck analysis — v1 structure vs the proven template

Surveyed template: the wheel's own sub-byte KV decode
(`results/2026-11-11-p647-mxkv-research/wheelref/v1/attention/ops/triton_turboquant_decode.py`,
installed in our fork's wheel) — the scheduling used by the incumbent TQ fp8
lane itself.

Defects in our v1 `triton_mx_decode.py`, ranked:

- **D1 — no split-KV (the dominant defect).** v1 grid is `(bsz, Hq)`: at 32K ctx,
  256 serial BLOCK_N=128 tiles per program, only 32 programs on a ~20-Xe-core
  part → latency-bound. 9.59 ms for ~69 MB unique (mxfp8) is ~7 GB/s — an order
  of magnitude under DRAM. The wheel template runs **fixed `NUM_KV_SPLITS=32`**
  (grid `(B, Hq, S)`) with per-split partial `(acc/l, lse)` rows and a stage-2
  LSE reduction; the native TQ baseline has had this all along. This alone
  explains most of the m4q/m8q-vs-ctrl engine gap.
- **D2 — GQA traffic amplification.** One program per q-HEAD: the 4 q-heads of
  each GQA group (Hq/Hk=4 on both certified models) re-walk identical K/V bytes
  → 4× read amplification (L2-mitigated at small ctx, real DRAM traffic at
  32K+ where the working set far exceeds L2).
- **D3 — e2m1 join/reshape interleave.** The mxfp4/mxk8v4 V dequant materializes
  lo/hi nibble tensors then `tl.join`+`tl.reshape` — a cross-lane layout op per
  tile. This is why mxfp4 loses to mxfp8 per-byte (an e4m3 payload dequants via
  one hardware bitcast). A duplicated-load + dynamic-shift formulation (`byte =
  d//2, nibble = (d%2)*4`) is pure vector ALU with no cross-lane op.
- **D4 — no tuning surface.** BLOCK_N=128/num_warps=4 hardwired; no per-shape
  sweep evidence.

## 3. Levers, ranked by expected gain (from the analysis)

1. **L1 split-KV scheduling** (D1): ~8–32× more resident CTAs at long ctx; the
   proven wheel structure ported onto MX geometry. Expected: recovers most of
   the decode-speed gap; the m4q 11.38 ms → byte-law bound would be ~5.6 ms
   (2×) or better.
2. **L2 e2m1 shift-unpack** (D3): removes the per-tile layout op on the V path
   (mxfp4, mxk8v4). Expected: mxfp4 per-byte cost ≤ mxfp8 per-byte cost.
3. **L3 GQA-batched K/V reads** (D2): 4× traffic cut — needs either tl.dot
   (XMX, tf32/ieee precision questions on the XPU Triton fork) or 3D fused
   reductions (compile-risk). **Parked as v3** — only worth it if, after L1+L2,
   telemetry shows decode is bandwidth-bound AND still slower than the byte law
   predicts.
4. **L4 tile/warp tuning** (D4): BLOCK_N {64,128} × num_warps {4,8} sweep in the
   A/B harness once L1/L2 land.
5. **L0 already banked:** store path is fast (0.07 ms @1K); prefill continuations
   use bulk dequant + flash (not on the decode critical path for steady-state
   generation).

## 4. Why the tuned lanes are superior (and where mxfp8 lands)

- Steady decode reads the full KV once per token. Bytes per token per head:
  native fp8 256 B, mxfp8 264 B, mxk8v4 200 B, mxfp4 136 B. At equal kernel
  efficiency the decode-speed ceiling ratio vs native is 256/264 = 0.97 (mxfp8,
  it LOSES), 256/200 = 1.28 (mxk8v4), 256/136 = 1.88 (mxfp4).
- mxfp4 additionally doubles usable context/concurrency at equal memory
  (banked 721,994 vs 398,363 tokens) — the unique selling point; the m4q
  essay-class REP-LOOP is a **key-precision** limit, so mxk8v4 (e4m3 K) is the
  quality-fixed 4-bit-value lane: 200 B with m8q-class retrieval.
- End-to-end engine speed also includes prefill, weights traffic, and spec
  overhead — the decode win surfaces as tps especially at long ctx and high
  concurrency (the KV read dominates the step there).

## 5. v2 design (implemented this leg)

`_mx_decode_split` (stage 1) + `_mx_decode_combine` (stage 2) in
`vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py`:

- Stage 1 grid `(B, Hq, 32)`: wheel-structure split walk, v1's per-row math
  verbatim (fused fp32 reductions; no tl.dot → no precision-mode questions),
  partial rows `[:D]=acc/l, [D]=lse` into a cached `mid [B, Hq, 32, D+1]`
  buffer (module-level cache, allocation once per shape on the eager warmup —
  graph law).
- **In-kernel adaptive `s_eff`** (new vs the wheel): each program derives the
  per-batch effective split count from its device-resident `seq_len`
  (`s_eff = clamp(ceil(tiles/8), 1, 32)`); surplus programs park an empty
  marker (`lse=-inf`, weights 0 in combine). Short sequences don't pay 32
  micro-splits, and no host read of seq lens is ever needed (XPU-graph safe).
- Empty-split rows always written (`lse=-inf`) — stale bytes can never leak
  into a combine (the wheel's early-return leaves stale `mid` bytes; ours does
  not repeat that hazard).
- Stage 2: standard log-sum-exp weighted combine over splits, all-empty guard
  → zeros (v1 zero-seq semantics preserved).
- e2m1 unpack flavor: `SHIFT: tl.constexpr` on the shared dequant helper —
  v1 keeps the join path bit-frozen (oracle law); v2 selects via
  `ARC_MXKV_UNPACK=shift|join` (default join) so the A/B leg measures the
  delta before anything defaults.
- Decoder selection: `ARC_MXKV_DECODER=v1|v2` (default **v1**). The certified
  path and every banked r154 quality verdict stay tied to v1; v2 is opt-in
  until its own gates pass.

## 6. Validation gates (before any default flip or cert inclusion)

1. **A4 offline (p157_mxkv_decode_ab.py, r157 on rig):** for every
   (fmt ∈ {mxfp8, mxfp4, mxk8v4}) × (decoder ∈ {v1, v2-join, v2-shift}) ×
   (random, adversarial rows, boundary lens 1/63/64/65, fragmented tables):
   vs fp64-exact on the same dequantized cache `max|d| ≤ 2e-3` and
   `cos ≥ 0.99999` (p147 decode-face gates), and vs v1 output
   `max|d| ≤ 2e-3` (split reassociation bound). Telemetry
   fmt × ctx {1K, 8K, 32K, 131072} at qwen-like HK=8/Hq=32: decode ms +
   effective GB/s + speedup vs v1 — the L1/L2 evidence bank.
2. **Engine A/B:** r155-style cell with `ARC_MXKV_DECODER=v2` vs the banked
   v1 cells (same posture) — solo/agg8 tps + suite-13/14 quality on both
   models.
3. **Certification sequencing:** cert24 ships v1 (correctness-first law — the
   pending retag window certifies what the r154 legs already proved). v2
   enters as its own content commit + full re-gate (its numerics differ from
   v1 by split reassociation, so the quality legs re-run with v2 engaged)
   → cert25 or a cert24 additive re-cut only after every gate is green.

## 7. Explicit non-goals

- No tl.dot / XMX score path in v2 (precision-mode risk without a measured
  need — revisit only if post-v2 telemetry shows bandwidth-bound decode below
  the byte-law prediction, i.e. lever L3).
- No store-kernel work (banked fast), no prefill restructuring (not on the
  decode critical path), no mxfp8 rescue attempts (byte law is unpassable).
- No default flip of `ARC_MXKV_DECODER` in this leg.

## Artifacts

- Implementation: `vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py` (v2 added,
  v1 bit-frozen) — this branch, commit follows this bank.
- A/B harness: `tools/ar/p147_mxkv/p157_mxkv_decode_ab.py` +
  `tools/ar/r157_decode_ab.sh` (rig leg; runs after the run3 correctness
  window — GPU is occupied by the gate until then).
- Banked inputs: `results/2026-11-12-p647-cert24-gate/` (engine numbers),
  `results/2026-11-11-p647-mxkv-research/` (wheel template survey),
  `results/2026-11-12-p647-mxkv-offline/` (p147 kernel telemetry).
