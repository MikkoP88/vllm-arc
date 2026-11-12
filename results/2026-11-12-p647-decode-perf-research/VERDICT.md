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

---

# ADDENDUM 2026-11-12 (post-A8): the improvement loop ran to convergence — v3 is the decoder

The always-loop directive ("do not stop while there is room") drove four more
measured iterations after the bank above (legs A5-A8 + two probes, all banked
here). Outcome: **v3, a GQA-batched split decoder, is the shipping default**
(cert25x), with per-format launch tables. The non-goal list above is
superseded where the evidence moved; the tl.dot non-goal HELD (never needed).

## A5 — the (BLOCK_N, num_warps) axis refutes the "mxfp4 regresses" verdict

Register pressure was the mxfp4 v2 regression: (64,8) takes mxfp4 to 2.413 ms
@32K (x7.26 vs v1 — from 0.87x at (128,4)); mxk8v4 (64,8) 5.951 ms (x3.06);
mxfp8 keeps (128,4) 2.994 ms (x5.55). (128,8) maps the spill cliff at 17-20 ms.
Format order now matches the byte law (mxfp4 < mxfp8 < mxk8v4 @ equal tuning).
Adversarial-face A4 FAILs were a harness zero-V case bug (cos(0,0)=NaN); fixed.

## A6 — num_stages is a no-op: the walk is a data-dependent pointer chase

Full (BLOCK_N, warps) x stages {1,2,3} factorial: stages flat within noise on
every combo (mxfp4 n64w8 2.414/2.414/2.419). Software pipelining cannot cross
the block-table -> slot pointer chase. Winners unchanged; stages=2 baked.

## v3 — GQA batching answers the pointer chase structurally (leg A7/A7b)

`_mx_decode_split_gqa`: grid (B, Hk, 32); ONE program serves all N_REP q-heads
of a kv-head — K/V dequanted once per tile, the chase amortizes N_REP-fold.
Per-r (q, m, l, acc) state is unrolled under tl.static_range as TUPLES (the
tracer coerces loop-captured lists to tuples — append is illegal; annotations
evaluate in Triton's namespace — bare `tuple` is NameError). No tl.dot, no 3D
broadcast materialization; v1 per-row math verbatim; v3 == v2 BIT-EXACT at 549
tokens (d=0.000, both flavors, all formats — same splits, same op order).

## Flavor sweep — per-format e2m1 unpack (FLAVOR-SWEEP.md)

Under v3's shared walk mxk8v4's V-only e2m1 dequant becomes the hot op: shift
(no cross-lane join) wins (2.520 vs join 3.124 ms); double-e2m1 mxfp4 keeps
join (1.156 vs 2.446); mxfp8 flavor-flat. Baked as the 4th _MX_TUNE field.

## A8 — shipping-config battery on the exact cert25 bytes: PASS

| fmt | @32K | @131K | vs v1 @32K | vs v1 @131K |
|---|---|---|---|---|
| mxfp8 (64,4,s2,join) | 1.183 ms | 4.593 ms | x14.0 | x14.4 |
| mxfp4 (64,8,s2,join) | 0.844 ms | 3.309 ms | x20.7 | x21.1 |
| mxk8v4 (64,8,s2,shift) | 2.232 ms | 8.576 ms | x8.2 | x8.5 |

(v1 baseline 16.6-18.2 / 66.1-73.2 ms; v1delta <= 3e-8 throughout; tune-face
correctness spots vs fp64 d ~ 1e-7.)

## Superseded non-goals, held non-goals

- HELD: no tl.dot/XMX path anywhere (the pointer-chase diagnosis made it
  unnecessary); no store-kernel work; no prefill restructuring.
- SUPERSEDED: "no default flip this leg" — v3 default flipped on the strength
  of A4-A8 (offline gates green; the r159 engine battery on cert25x re-earns
  the serve-level quality verdicts per the convergence law).
- mxfp8 superiority note: the byte law (264 vs 256 B) still caps mxfp8 at
  kernel parity vs a perfect native kernel; x14 vs OUR v1 is the measured
  claim; the e2e ctrl comparison rides the r159/r155 cells.

## Certification path taken

Bake (commit 663a760f) -> cert25x image (r158, fork_sha 663a760f, live
assertions incl. `_MX_TUNE == {mxfp8: (64,4,2,join), mxfp4: (64,8,2,join),
mxk8v4: (64,8,2,shift)}`, `_DECODER == 'v3'`) -> r159 full 8-leg battery
(ctrl m8q m4q m8g m4g mkq mkg g0) on cert25x -> r155 three-axes ladder
(decode/prefill/TTFT vs ctrl) -> close-out.

## A9 (2026-11-12, run 2) — the true-shape rebake overturns the v3 default

A4-A8 tuned at HK=8/Hq=32/D=128 (N_REP=4). The REAL qwen3.8-27b-fp8 attention
shape is HK=4/Hq=24/D=256 (N_REP=6): at D=256 a BLOCK_N=64 K-tile holds 64KB
fp32 of live accumulator state — the spill cliff moves, and BLOCK_N=32 was
never swept. A9 (p157_v3_shortctx.py, cert24x dev image) swept the full
(decoder x {32,64} x {4,8} warps) grid at the true shape, serve cells
B=1 {1K,4K,32K} and B=8@2048:

- v2 + BLOCK_N=32 dominates EVERY serve cell. B=8@2048: mxfp8 1.414 ms
  (32,4), mxfp4 1.174 (32,8), mxk8v4 1.519 (32,8) — vs the baked v3 (64,8)
  2.478/1.516/2.478. v3 keeps exactly ONE niche: solo B=1@1K (0.331 vs 0.74).
- REBAKE: `_MX_TUNE` = mxfp8 (32,4,2,join), mxfp4 (32,8,2,join), mxk8v4
  (32,8,2,shift); `_DECODER` default v2; v3 + s_eff 1-tile floor stay
  reachable via ARC_MXKV_DECODER=v3.
- run-2 faces (bank `legA9_run2_summary.txt`, md5 b527c6f1): fp16-q native
  7.0e-05-1.2e-04 vs rounded-fp32 (NOT bit-equal — the kernel's intermediate
  dtype follows the q/out element type, i.e. one fp16 rounding; gate <= 2e-3,
  two orders inside the MX quant-noise envelope; out dtype follows q via
  empty_like) — evidence for the engine's native-q path; verify-batch face
  (R=32, BARE shipping-config calls) bit-equal d=0.000 with the batched call
  1.1-1.3x over the 32-row loop at every format. A9_PROBE_PASS rc=0.

## The engine decode-gap line (same leg) — batched MTP verify

Telemetry (A10 directive) attributed the MX-vs-ctrl decode gap primarily to
per-step kernel COUNT, not per-kernel speed: with spec=3 every decode step is
a q_len=4 verify batch routed to the split branch — one mx_decode PER ROW
plus a slot-inversion cascade (~8 kernels) per request, ~2k kernels/step at
agg8 x ~17 attn layers vs ctrl's ONE fused FA kernel per layer. Fixes banked
with the rebake: `_split_verify_decode` (ONE mx_decode for all split rows in
the step; plan cached one step on self keyed on metadata identity +
slot_mapping data_ptr/_version, strong metadata ref), native-q decode (drop
query.float() + out .to guard), LIMIT-LAW per-row limits retained verbatim.
v3-vs-v2 correctness gates and the fp8-path reference (FP8-PATH-REFERENCE.md)
explain the ctrl shape this line converges toward.

## Certification path (updated)

Rebake commit -> cert25y image (r163, two-file overlay on cert25x, live
assertions incl. the (32,...) table + `_DECODER == 'v2'`) -> r162 A10
telemetry ladder on cert25y -> r161 full 8-leg battery + gemma_compare
(gemma gates comparative per GEMMA-THINKSPAM-DISPOSITION.md) -> r155
three-axes ladder -> close-out.
