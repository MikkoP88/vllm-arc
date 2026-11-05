# P6.21c — dense-FA ESIMD v2: built, numerics-green, measured to a decomposed ceiling BELOW the bar (2026-11-05)

Item: the kernel lane opened by p621b (r88 pre-registered S_dual 3.96 →
"v2 economics" rung-B). This record is the full build-measure-close
story. **Verdict: lane closes NEGATIVE** — the ESIMD SLM-tile
architecture measured (and ceiling-bounded) below the stock vxk kernel
at every step; the pre-registered ≥ 2× bar is unreachable for this
kernel class on this part. All legs on throwaway cert9 containers, GPUs
quiet, W14 stop/restore per leg, FRANCE-verified restores, build
container `esimd-w11` (DPC++ /src/dpcpp, TORCH_XPU_ARCH_LIST=bmg).

## What was built

`vllm_gaudi/xpu/esimd_fa/` (default-off, NOT routed into serve):

- `csrc/fa_dense_v2.sycl` — launcher + torch binding. Contract: bs == 1
  (chunked-prefill), kvLen ≥ qLen (bottom-right causal), pageSize % 32
  == 0 (tile-never-straddles; 832 serve + 1664 padded probe pages),
  qHeads % kvHeads == 0 (GQA), hd 256 constexpr, paged e4m3 KV with
  block table, optional per-kv-head f32 descales, `ablate` perf
  instrument (level = `ablate & 7`, variant bits = `ablate & ~7`).
- `csrc/fa_dense_v2.fp8.h` — the kernel. Grid (qHead, 64-row slab), WG
  = 8 threads, 32-token KV tiles in SLM (QV 32 KB row-major q + KGVNNI
  16 KB + VVVNNI 16 KB + ACC 64 KB = 128 KB ≤ BMG 131072), online
  softmax in registers, XMX `dpas<8,8>` for QK' and PV, e4m3 dequant in
  registers, epilogue acc→fp16 out. v1 (`fa_dense.*`, p72) stays
  buildable for A/B.

## Legs

| leg | what | verdict |
|---|---|---|
| r92 build+defect | v2.0 first light (rows-as-M QK with 64-lane C) | numerics garbage at every shape; probe-isolated: `dpas<8,8>` with a **64-lane C makes BMG execute M4×K8×N16** (B read in N=16 word layout, only A rows 0..3 computed) — the N is selected by OPERAND WIDTHS, not template args (dpas_probe.sycl, dbg6/dbg7) |
| r92 v2.1 | QK' rewritten to N=16: A = q rows row-major [8][16] via 256-B QRM block stores; B = K-tile VNNI staged per-(token,kb) 8-lane slm_scatter; new [m][16] mask/fold/pf geometry; PV untouched (its 128-lane C was always N=16) | **numerics ladder PASS** — A-cases (exact softmax expectation `p = eᶜ/(eᶜ+edge)`; probe artifact, not kernel), L1 fro 2.55–2.88e-3 vs torch-exact incl. (100,833) partial page and (2048,8192), L2a vs stock 2.70e-3 (uniform descales — stock's per-kv-head descale semantics ≠ h//group, p72_descale_semantics.py), L2b vs exact 2.88e-3, L3 OOB canaries 0 corrupted at both shapes (v1 corrupted neighbors at big bs=1) |
| r92 perf | stock vs v2 across the U32k pattern | v2 = **0.34–0.35× stock** (12.1–13.0 TF/s useful; stock 76.95 ms @ 8192×32768 vs v2 223.3) |
| r92_v22 | staging fat loads: every token's K/V one single-message 256-B `lsc_block_load`, all 8 in flight before dequant | numerics green; **staging time unchanged** (140→142 ms) — staging is not load-message-bound |
| r92_v22b | ablation instrument fixed (bit values 8/16/24 satisfied the old `>=` level guards → the "variant" runs had silently executed the FULL pipeline; guards → `(ablate & 7) >= N`), full bucket decomposition | min-of-3 interleaved, rounds identical to 0.1 ms: L0 staging 159.6 = **89.2 floor + 46.7 K-stores + 23.9 V-stores** (additive to 0.2 ms); dequant free (L96≡L104); scatter *pattern* premium only 5 ms (L16 Kcontig); compute ladder +20.4 RMW / +32.9 QK / +9.4 softmax / +28.4 PV |
| r92_v23 | staging message-halving: K 4-token 32-lane scatter (one per kb), V merged-pair 128-B block stores | ladder PASS again; L4 250.7→233.5, L0 159.6→144.2 — but stores only 70.4→58.7: **a scatter costs its LANE count** (512 scattered 4-B lane-writes/thread-tile unchanged by wider descriptors), not its message count |
| r92_v23 floor | new probes: L224 (no loads, no stores) = **77.5 ms**; L128 (stores only) = 136.7; L96 (loads+dequant only) = 85.5 | **global loads ≈ 8 ms — already hidden** (inter-WG parallelism); staging cost is 96% SLM stores + the structural floor (barriers/loop/epilogue) |
| r93 | final clean stock-vs-v2.3 same-window probe (fixed aggregate; the early perf window's aggregate TF/s field had a stale ×16 — per-case rates were correct) | numerics all PASS incl. 3072×6656 serve shape, OOB clean; **0.33× stock across the pattern** (11.61/33.44/55.23/77.16 ms stock vs 35.59/101.40/167.38/233.62 v2.3; agg 12.3 TF/s useful) — probe 233.62 ≡ bench L4 233.5 (harnesses cross-validated) |

## The ceiling argument (why it closes)

Budget for ≥ 2× stock at U32k 8192×32768: 38.8 ms (stock 76.95). Measured v2.3: 233.5. Decomposed (same-harness min-of-3):

- structural floor 77.5 (2048 barriers/WG, tile-loop, q staging, epilogue) — T64 tiles halve barriers → best case ~40
- SLM stores ~59 (K scatter = 512 scattered 4-B lane-writes/thread-tile; contiguous-as-A swap → ~64 sectors → best case ~15)
- global loads 8 (already hidden)
- compute ~89 (RMW 20.2 + QK 31.3 + softmax 9.4 + PV 28.5) — the dpas chains run ~32 TF/s useful vs r88 bmm-class ceilings (QK 67.7 / PV 136.7 TF/s) → best case ~30 if chains pipelined perfectly

Every bucket at its optimistic bound simultaneously: ~93 ms ≈ 0.83×
stock — **still below 1.0×**, let alone the 2.0× bar. The SLM-tile
architecture cannot beat the stock closed kernel on this part even in
fantasy-land; the gap is structural (SLM round-trip + barrier
orchestration per 32-token tile vs the tuned closed-source kernel).

## What ships

- The kernel + probes + runners land on `feat/arc-p621c-fa-esimd-v2`
  (default-off tooling; no serve routing, no image delta — cert9 stays
  the certified currency). It is the reproducible evidence base and a
  working N=16/dpas-law reference for any future FA work.
- Banked laws (also in source comments): dpas N-selection by operand
  widths (64-lane C → broken M4×K8×N16 on BMG); VNNI B-word layout
  word(kp,n)@(kp*16+n)*4; scatter cost ≈ lane count; ESIMD pitfalls
  (simd_view non-deduction, deleted simd ternaries, const-select
  ref-qualification, bounded unrolls for IGC, 256-B max lsc block
  message).

## Reproduce

- build: `tools/esimd/fa_dense_v2_build.sh` (in esimd-w11 at /src/esimd-fa)
- numerics gate: `tools/esimd/numerics_v21.py` (+ `numerics_v21_run.sh` window)
- rates: `tools/esimd/probe_fa_dense_v2.py` (+ `perf_v21_run.sh`)
- ablation: `tools/esimd/bench_v21.py` (+ `abl_v21_run.sh` — numerics gate then bench)
- raw logs: `rig/` (numerics_v21.out, perf_v21.out, abl_v22b.out, abl_v23.out, abl_v23b.out, perf_v23.out, bench_v21.json, numerics_v21.json)
