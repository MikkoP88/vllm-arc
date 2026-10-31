# P6.6 r66a — prefill solo-throughput curve + device truth (capture leg 1)

**Date**: 2026-10-31 · **Phase**: P6.6 (plan §12 item 8, the [viii]
engine-side lane opened by P6.5's attribution matrix)
**Posture per point**: `vllm-arc:v0.30.0-b70-cert4`, fresh boot,
`ARC_LMC=0` (NO LMCache — pure engine), graph law, MTP k3, greedy,
**solo** (one request at a time, no bystanders), unique seeds per face.

## The solo prefill curve (p66_solo_ttft.py; TTFT s / rate tok/s)

| thr | U8k (8158 tok) | U32k (32553) | U64k (65303) |
|---|---|---|---|
| none | 3.74 / 2178 | 17.34 / 1878 | 39.20 / 1666 |
| 8192 | 3.75 / 2175 | 17.35 / 1876 | 39.23 / 1665 |
| 4096 | 4.04 / 2018 | 18.24 / 1785 | 42.38 / 1541 |
| 2048 | 7.94 / 1028 | 21.49 / 1515 | 49.75 / 1312 |

Findings:

1. **thr=8192 ≡ thr=none** (≤0.03 s on every face) — at MNBT 8192 the
   threshold's long-prefill path is the default chunked path; no separate
   branch to account for.
2. **Rate falls with context** (2178 → 1666 at thr none): the eager
   prefill cost per token grows with context — attention re-reads over a
   growing KV + fp8-e4m3 KV writes; not a fixed-rate engine.
3. **Small chunks cost real throughput**: thr2048 vs none at U64k =
   **+10.55 s (+27 %)**; at U32k +4.15 s (+24 %). This is the solo,
   zero-contention form of the [viii] share — and it **matches the +10.7 s
   engine-side attribution** from the P6.5 bystander matrix exactly.
4. **Per-step fixed overhead model**: TTFT ≈ N·r(L) + c·⌈N/thr⌉. Solving
   from the 8192↔2048 pairs: c ≈ 0.435 s (U64k), 0.343 s (U32k); the U8k
   pair is noise-dominated (3 chunks). A 2048-token step's token compute
   is ~0.9 s, so **~0.35–0.44 s of every chunked step (~30 % at thr2048)
   is NOT token compute**. Cross-checks: bystander ITT p50 during U32k
   under thr2048 measured 746–1242 ms (p65 legs) — consistent with
   2048-token compute + the same per-step overhead; and the engine's own
   10-s window stat showed 815.7 tok/s for the U8k/thr2048 window
   (client 1028; window dilution expected).
5. Reproducibility: two independent boots (none and 8192) agreed to
   ≤0.03 s on every face — the curve is tight.

## Device truth (p66_devprobe.py, gate1b; GPU idle, ZE_AFFINITY_MASK=0)

| Item | Measured |
|---|---|
| Device | Intel Graphics **[0xe223]**, **31.9 GiB** per card, max_compute_units=256 |
| fp16 GEMM | **147.0** (8192³), **150.4** (4096³), **132.6** (8192×4096×16384) TFLOPS/card |
| bf16 GEMM | 136.0 / 130.4 / 131.9 TFLOPS/card |
| Model | `model_type=qwen3_5`, **34 layers**, intermediate 17408 (hidden nested in text_config; nominal ~27 B params) |

**MFU**: fleet GEMM peak ≈ 294 TFLOPS (2 × 147). Prefill demand ≈ 2 ×
27 GFLOP/token → measured fleet compute 118 TFLOPS (U8k) to 90 TFLOPS
(U64k) = **~40 % → 31 % of achievable GEMM rate**. Prefill is NOT
GEMM-saturated: the gap is attention (fp16 score math over growing KV),
non-GEMM ops (norms/rope/GDN linear-attention + mamba), TP2 allreduces,
W8A16 dequant, and the per-step overhead above.

The fp8 randn probe line failed (`normal_kernel_xpu` unimplemented for
Float8_e4m3fn — a probe limitation, not an engine one); the W8A16
dequant+matmul combined cost remains unmeasured (leg-2 item).

**Doc discrepancy found (for P6.6 close-out docs)**:
CONFIGURATION-arc.md's hardware floor note says "dual B70 (16 G each)" —
the device reports **31.9 GiB/card**. The GMU/memory math should be
re-derived (0.8 × 32 GiB ≈ 25.6 GiB pools/card) when docs are next
touched; not load-bearing for this leg.

## Where the improvement levers are (ladder per plan §10)

| Rung | Lever | Expected face |
|---|---|---|
| 1 (knob) | threshold choice per workload — 2048 only when bystander protection is worth +24–27 % long-stream TTFT; 8192/none for solo-heavy fleets | documented trade-off table (like MTP depth); zero code |
| 1b (knob) | `--max-num-batched-tokens 16384` (bigger chunks, fewer steps) for solo-heavy fleets | U64k −N·c overhead; needs mamba-split + KV headroom check |
| 2 (glue) | cut the ~0.4 s per-step overhead: attribute it first (eager kernel-launch count, TP2 allreduce latency ×68/step, W8A16 dequant, GDN chunk-boundary state work, scheduler host time) | leg 2: step-loop instrumentation probe (r48j pattern) |
| 3 (structural) | raise the 31–40 % MFU: prefill compile (`mode:none` → piecewise) — large posture change, A/B gated | leg 3+ |

## Leg-2 design (next)

Instrumented probe image (docker-commit pattern, r48j): timers around
the engine step path for one solo U32k face at thr2048 — per-step wall,
forward vs pre/post split, allreduce count/time, kernel-launch count
(CPU-side), mamba/GDN boundary work. Output: the 0.4 s decomposed. No
fix without that capture.

## Artifacts

- `p66_solo_ttft.py` / `p66_gate1.sh` — solo probe + sweep gate
- `p66_solo_thr{none,8192,4096,2048}.{txt,json}` — the curve
- `p66_devprobe.py` / `p66_gate1b.sh` / `p66_devprobe.out` — device truth
- `serve_thr*.log` — full engine logs (stats lines, config dumps)
- `boot_thr*.out`, `gate1.log`, `gate1b.log` — leg logs
- rig: `/root/build/p66/` (all of the above)
