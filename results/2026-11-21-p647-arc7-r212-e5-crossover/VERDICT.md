# r212 E5 — device-only crossover map: v2 Triton vs native CUTLASS

**Date**: 2026-11-21 · **Leg**: r212_e5 (arc7 Phase-8, E5) ·
**Wheel**: arc7mx r210 keeper `7247e83b` · **Image**:
`vllm-arc:v0.30.0-b70-cert25zc` · **Tool**:
`tools/ar/p147_mxkv/p204_graphtime.py` (shipped unchanged, md5
`cf49611f…`) + `r212_e5.sh` (5 ctx runs, splits=auto only — the E2''
ladder was flat)

## Verdict

**PASS — per-format device-only crossovers located; r212a thresholds
set.** Graph-replay (32×50) device times across ctx
1024/2048/4096/8192/32768, cells v2 + native_auto, all canaries 0.
The native lane's fixed barrier/walk term (E4-lite: 87-163 µs)
dominates until v2's unsaturated Triton walk overtakes it; each
format crosses at a different ctx because v2's slope is 3-10× the
native slope. mxk8v4 crosses EARLIEST (v2 pays 86-1558 µs for the K8
dequant at every ctx); mxfp8 crosses LAST (native's e4m3-V staging
protocol is the fattest fixed term).

## Results (device-only µs/call; native_auto/v2 ratio)

| ctx | mxfp8 v2 | native | ratio | mxfp4 v2 | native | ratio | mxk8v4 v2 | native | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 49.7 | 327.9 | 6.60 | 42.8 | 233.6 | 5.46 | 84.0 | 219.4 | 2.61 |
| 2048 | 65.2 | 202.8 | 3.11 | 62.6 | 146.6 | 2.34 | 126.1 | 138.3 | 1.10 |
| 4096 | 107.8 | 211.6 | 1.96 | 98.2 | 155.5 | 1.58 | 211.8 | 147.2 | 0.69 |
| 8192 | 208.6 | 228.9 | 1.10 | 184.5 | 172.4 | 0.93 | 417.1 | 164.1 | 0.39 |
| 32768 | 759.6 | 422.2 | 0.56 | 648.2 | 310.8 | 0.48 | 1558.5 | 307.5 | 0.20 |

Slopes (µs/token, 8K→32K segment): v2 0.0224 vs native 0.0079
(mxfp8); v2 0.0211 vs native 0.0041 (mxfp4, 4K→8K); v2 0.042 vs
native 0.0044 (mxk8v4, 2K→4K).

## Crossovers (linear interpolation between bracketing ctx)

| format | crossing ctx | r212a dispatch threshold |
|---|---|---|
| mxfp8 | ≈ 9.6K tokens | native iff max_kv_len ≥ 10240 |
| mxfp4 | ≈ 7.5K tokens | native iff max_kv_len ≥ 8192 |
| mxk8v4 | ≈ 2.4K tokens | native iff max_kv_len ≥ 3072 |

Sanity anchors: the 32K cells reproduce the r208 chain (native
422/311/308 vs r208 427/327/333; v2 760/648/1559 → 1.80×/2.09×/5.06×)
and the 1K cells reproduce E2'' (328/234/219 native; 50/43/84 v2).
mxk8v4's v2 at 32K (1558 µs) is the worst cell on the map — the
dispatch removes it entirely.

## Findings

1. **Dispatch is worth more than any kernel-side 1K fix**: taking
   min(v2, native) per cell vs pure v2 = 1K-4K parity (v2's own
   number) while keeping the 8K/32K native wins; vs pure native it
   removes the 2.3-6.6× short-ctx deficits. No single kernel can
   serve both regimes — the fixed term (E4-lite) vs slope tradeoff
   is structural.
2. **mxk8v4 is the dispatch's biggest beneficiary** (2.4K threshold;
   5× at 32K, 2.6× deficit removed at 1K).
3. **mxfp8 needs the most ctx before native wins** — consistent with
   E4-lite's +73 µs m3 penalty (2× V pitch staging barriers).
4. All 15 cells captured + replayed clean (canary 0) — both lanes
   are graph-capturable in their engine shapes; dispatch inside the
   captured region would need static branch selection per capture
   (ctx is static per capture — the threshold branches at capture
   time, not replay time).

## Next

- **r212a**: engine ctx-adaptive dispatch in
  `vllm_gaudi/xpu/patches/mx_kv/` — threshold on max KV len per
  format (default table above; env override
  `ARC_MXKV_NATIVE_MINLEN="mxfp8:10240,mxfp4:8192,mxk8v4:3072"`),
  short ctx → the Triton v2 lane, long ctx → the native kernel.
- **r212b**: native barrier reduction (always-loop native floor;
   raises the native ceiling everywhere right of threshold).
- Then the r210 convergence runway: r200-e2e fastcycle with dispatch
  on → new image → full §10 battery → cert26.

## Artifacts

`p204_graphtime_c{1024,2048,4096,8192,32768}.json`,
`r212_e5_c*.log`, `r212_e5_launch.log` (rig: `/root/build/p199/out`,
`/root/build/p199/logs`).
