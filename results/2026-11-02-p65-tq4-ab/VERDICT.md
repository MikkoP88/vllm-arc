# TQ4 A/B — temporary `--kv-cache-dtype turboquant_4bit_nc` test (2026-11-02)

**Owner directive:** run a temporary performance test changing ONLY
`--kv-cache-dtype fp8_e4m3` → `turboquant_4bit_nc`; return current vs
temporary numbers.

**Posture:** image `vllm-arc:v0.30.0-b70-cert5` (current certified),
certified 25-var chain + certified CLI, k3, graphs, TP2, GMU 0.8, MNBT
8192 — **one variable only** (substitution verified: exactly 1 token
changed, `serve_diff.txt` line 52; script `rig/tq4_leg.sh`). Baseline
("current") = the cert5 gate artifacts measured on the same image/rig
the day before (`../2026-11-01-p65-cert5/rig/`). TurboQuant is a
first-class dtype in the vllm-0.30.0+xpu wheel (`turboquant_4bit_nc`:
"4-bit MSE keys + 4-bit values + NC, 3.8x" per the wheel's config
docstring; the engine auto-forces FA2 and auto-adds hybrid boundary
skip-layers). W14 incumbent restored by the leg's EXIT trap after the
run (a pre-existing non-leg arc-test container from 11:02Z was
snapshotted to `preexisting_arc-test.json` and removed — it was NOT the
certified posture: GMU 0.9, model-len 299008).

## Current vs temporary

| Face (same harness, same image) | current `fp8_e4m3` | temp `turboquant_4bit_nc` | delta |
|---|---|---|---|
| **KV cache capacity** (6.92 GiB pool, per boot log) | 405,491 tokens (1.55× concurrency @262k) | **757,641 tokens** (2.89×) | **+1.87×** |
| run7n t0 agg4x256 #2 | 297.19 tok/s | **299.36 tok/s** | +0.7 % |
| run7n t0 solo200 ×2 | 84.48 / 84.47 | **85.97 / 85.57** | +1.8 % (above family top 84.9) |
| run7n t1.0 xhigh agg #2 | 189.41 tok/s | **88.38 tok/s** | **−53 %** (ragged streams: 53/193-tok early finishes inside 4/4 "ok") |
| run7n t1.0 xhigh solo200 ×2 | 62.35 / 61.85 | 77.34 / **25.85** | one stream +24 %, one **−58 %** |
| spec gate pos1 / acceptance | 0.9258 family (certified, ≥0.85 target) | **0.9623 / 0.9261 — PASS** | MTP verify path intact |
| r42 integrity (cold==rehit) | 5/5 | **3/5 — FAIL** (seeds 404+505: cold≠rehit; 404 also truncated 1024/989 chars) | **correctness regression** |
| determinism (repeat) | 3/3 | 3/3 | — |
| U32k TTFT (12 bystanders, W2MT 4000) | 23.16 s | **15.94 s** | **−31 %** |
| U64k TTFT (12 bystanders) | 50.77 s | **33.44 s** | **−34 %** |
| W_U32k_start bystander gap p95 / max | 1299 / 1310 ms | **3302 / 3474 ms** | +154 % / +165 % |
| W_U64k_start bystander gap p95 / max | 1569 / 1602 ms | **3490 / 3850 ms** | +122 % / +140 % |
| W_U32k_end / W_U64k_end | clean (0 stall rows) | clean (0 stall rows) | — |
| flip-first (832 tok) | stop 8.3 s | stop 9.7 s | in family |
| boot to health | ~315 s | ~300 s | in family |
| Engine ERROR lines | 0 | **112** (all one class) | LMCache lagged-attn-store failure |
| Tracebacks | 0 | **112** (one per ERROR) | see below |
| LMCache stores (1664-class) | 44 × (U32k family) | 52 × | stores run, lag path errors |

## What breaks: the LMCache lagged-store path vs the TQ packed layout

All 112 ERROR+Traceback pairs are ONE class, from our lmcint adapter's
P6.5/P6.13 lag machinery:
`LMCache ERROR: Lagged attn store failed for req … (cached chunk absent;
later lookups recompute)` (`vllm_v_adapter.py`). Under TQ the attention
KV lives in the packed TQ K|V layout (page size re-derived; attention
block size forced to 1664 to align with the mamba page, +2.84 % mamba
padding) — the deferred/lagged store's assumption that prior chunk rows
are retrievable for the windowed store does not hold, so ~112 store
attempts fail and later lookups RECOMPUTE. That recompute-mix is the
mechanism behind the two integrity FAILs (cold≠rehit divergence) and it
means the KV-reuse tier (the 0.99–1.11 s warm re-hit faces) is
unreliable under TQ4 as-is.

## Reading (owner decision material, nothing adopted)

- **Wins are real and large:** +87 % KV capacity, −31/−34 % long-prompt
  TTFT, greedy t0 slightly above family (small context, decode-bound —
  TQ barely touches decode; the TTFT win is the halved KV write/read
  bandwidth in prefill attention).
- **Costs are disqualifying for serving as-is:** sampled/thinking
  throughput collapses (−53 % agg; ragged early-finish streams — a
  generation-quality effect of 4-bit KV, matching the wheel's own
  "+2.71 %" quality-cost note at the extreme end), bystander prefill
  gaps more than double (3.4–3.9 s max — TQ quantize/dequant on the
  prefill path serializes worse under contention), and cache-replay
  integrity fails via the broken lagged-store path.
- **Disposition:** temporary test only — reverted immediately (the leg
  container is gone; W14 restored). Not adopted, not banked into any
  posture. If the KV-capacity or TTFT win is wanted, the prerequisites
  would be (a) making the lagged store TQ-aware (or disabling the defer
  profile under TQ and re-testing the plain store path), and (b) an
  owner call on the sampled-throughput quality cost. No engine change
  was made by this leg.

**Baseline provenance:** all "current" numbers re-derived from
`../2026-11-01-p65-cert5/rig/` (same image, same harnesses, 2026-11-01).
Artifacts: `rig/` (serve log redacted — `../REDACTION-2026-11-02.md`).
