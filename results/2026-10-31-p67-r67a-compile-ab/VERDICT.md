# P6.7 r67a — prefill compile REVAL (Appendix H law #1) — NEGATIVE

**Date**: 2026-10-31 · **Phase**: P6.7 rung A (plan §12 item 10, ladder A)
**Posture**: `vllm-arc:v0.30.0-b70-cert4` image, CLI-only compile change,
fresh boot, `ARC_LMC=0`, graph law (`ARC_GRAPH=1`), MTP k3, greedy.

## Question

Appendix H law #1 (compile-off acceptance law): on 0.26, inductor piecewise ×
spec-decode produced wrong logits — acceptance collapse 22.5 → 79.7 (rejection
rate), recovered only by `TORCH_COMPILE_DISABLE`. Prefill was unaffected then.
The law ordered: REVAL on 0.30, keep the acceptance gate regardless of
outcome. Rung A = that REVAL: `mode=vllm_compile` × `cudagraph_mode=
FULL_AND_PIECEWISE` (the combo law, Appendix C b2 — captured, never loose).

## Config law discovered en route (gate1c → gate1d)

Pass 1 died at boot: stock `cudagraph_capture_sizes` ladder tops at **320**
(k4 headroom), but `gdn_attn.py:120` hard-caps captured tokens at
`max_num_seqs × (num_spec+1)` = 64 × 4 = **256** — the synthetic PIECEWISE
capture at (64 reqs, 320 tokens) tripped the GDN decode-only assert
(`gdn_attn.py:537`). Fix: ladder → 256 + `max_cudagraph_capture_size: 256`
(satisfies `compilation.py:1145`). Real k3 verify shape IS 64×4=256, so the
decode graph set is unchanged vs cert4. PIECEWISE then captured 72/72 clean;
boot-to-health ~460 s (stock same-rig ref: ~265 s). Full quote file:
`gate1c_pass1_death_quote.txt`.

## The A/B (same session, same image, fresh boots both arms)

| Face | PW compile arm | Stock ref | Δ |
|---|---|---|---|
| **Mean acceptance length** | 2.21 / 2.32 | 3.40 / 3.57 | **−35% collapse** |
| agg4x256 #1 (incl. warmup) | 176.26 t/s | 96.64 t/s | (polluted face both) |
| agg4x256 #2 (steady) | 182.04 t/s | 300.48 t/s | **−39%** |
| solo200 t0 | 51.33 / 51.66 t/s | 84.49 / 84.49 t/s | **−39%** |
| U8k TTFT (8158 tok) | 3.75 s (2178 t/s) | bar 3.74 s | flat |
| U32k TTFT (32553 tok) | 17.08 s (1906 t/s) | bar 17.33 s | −1.4% |
| U64k TTFT (65303 tok) | 39.20 s (1666 t/s) | bar 39.19 s | flat |
| XPU KV cache size | 426,874 tok (1.63x) | 405,491 tok (1.55x) | **+5.3%** |
| Boot to health | ~460 s | ~265 s | +74% |
| Fork-added WARN/ERROR | 3 jit_monitor (inference-time Triton JIT: `_resample_kernel`, `_rejection_kernel`, `_compute_local_logits_stats_kernel`) | 0 | hygiene break |

Speculator coercion (defect-a patch): LIVE first exercise under compile
posture — `speculator graph mode coerced` in serve log ✓.

## Verdict — rung A CLOSED REVAL-NEGATIVE

**Law #1 CONFIRMED on 0.30: the acceptance gate fired.** MTP mean acceptance
collapses 3.4-3.6 → 2.2-2.3 under `vllm_compile`; the −39% decode t/s is
arithmetic (fewer accepted tokens per verify step), not a separate defect.
Prefill is flat — compiled glue is not the prefill bottleneck, consistent
with r66b's rung-2 (glue) closed as empty: device-side attention/GDN/GEMM
time dominates and those are splitting ops / custom kernels outside inductor's
reach. The side-benefits (KV pool +5.3%, concurrency 1.63x) do not compensate
a −39% fleet decode face. `mode` stays **NONE** — cert4 remains the certified
posture. No cert5 from rung A.

P6.7 continues with rungs B (ESIMD FA-layer attention + GDN chunk-scan,
radiance-shared-component scope) and C (fp8 XMX GEMM inventory) — kernel-level
work on the stock eager path, independent of this result.

## Artifacts

- `gate1d.log` — the A/B leg (pass 2) exactly as run; `gate1c.log` — pass 1
- `p67_gate1.sh` / `p67_gate1d.sh` — both gate scripts (sed byte-exactness
  laws in comments)
- `serve_pwd.log` / `serve_pwd_after.log` — PW arm boot + battery-phase serve
  logs (config dump, PIECEWISE 72/72, KV line, acceptance telemetry)
- `serve_ref_d.log` — stock ref arm (acceptance 3.40/3.57)
- `p67_pwd_solo.{txt,json}` / `p67_pwd_run7n_t0.out` / `p67_refd_run7n_t0.out`
- `boot_pwd.out` / `boot_ref_d.out` — launch banners
- `flag_line_pwd.txt` — the transformed compilation flag (provenance)
- `gate1c_pass1_death_quote.txt` — pass-1 death, quoted with honest
  provenance (container removed before full log pull; lines captured live)
- rig: `/root/build/p67/` (same names, plus serve_pw* pass-1 names)
