# r37 — P4 BASELINE: first throughput + TTFT-by-ctx capture on the banked P3 product (certified D tools, unchanged)

Date: 2026-10-29 (boot 19:12, battery 19:30–19:52, integrity 19:55).
Posture: the r35/r36-banked product exactly (vllm-arc:v0.30.0-b70-p41 +
overlay: lmcint kit, TWO_HS, glues, sched_tail_split v4; align ON; timers
OFF; first-traffic fresh boot, health 200 before any request).

**Tools = D's certified bench harness, run UNCHANGED host-side**
(provenance: line-D rig artifacts, read-only): `bench_prefix.py`
(bfeaa4c4…), `bench_longctx.py` (3854f26f…), `bench_run7n.py`
(e5b8d594…) — the exact scripts behind the cl1b bars and D's certified
cells, same model id, same order as D's leg driver (prefix ×3 → longctx ×1
→ run7n ×2). Every number below is like-for-like with the donor ledgers.

## Faces vs bars

| face | bar (source) | r37 (steady cell) | status |
|---|---|---|---|
| solo200 tps (incl. TTFT) | ≥ **79.74** (contract/D-cert) | **59.07 / 59.13 / 60.02 / 59.12** | **FAIL −26%** |
| agg4x256 tok/s | ≥ **274.41** | **196.06 / 209.20 / 195.63 / 206.18** | **FAIL −24…−28%** |
| longctx t1 8k (7950 tok) | 3944 ms (D ladder parity) | **5726 ms** | **FAIL +45%** |
| longctx t1 16k (16237) | 8083 ms | **12074 ms** | **FAIL +49%** |
| longctx t1 32k (32081) | 17447 ms | **25288 ms** | **FAIL +45%** |
| longctx t2 8k / 16k / 32k | C 0.587 / 0.741 / 0.566 | 1.503 / 1.140 / 1.850 s | r36 posture class (documented) |
| prefix rehit (post-churn external) | cl1b 0.48 s | 1.56–1.57 s | r36 posture class |
| integrity after full battery | 0 drift | **5/5, ALL five canonical shas** (8898f039/28f2a474/8fa239e9/a1573902/138272250b2e) | **PASS** |

Steady-state evidence: prefix p2/p3 byte-identical phases (deterministic
corpus, temp 0); run7n two invocations agree within 1.4%; prefix p1 cold
ttft_med 3.89 s (true first-traffic cold; p2/p3 "cold" = engine-warm 1.39).

## Decomposition (metrics + arithmetic)

- **Spec counters** (vllm:spec_decode_*): drafts 1712, draft tokens 6848
  (= 4×1712, k4 full-width), accepted 5642 → **0.824 aggregate
  acceptance, 4.295 tokens emitted/step**. (Per-position pos1 re-measure
  owed in tuning legs; prior probe read 0.8985.)
- **solo**: decode window ≈ 200/(3.38−0.44) = 67.8 tps → step time =
  4.295/67.8·1000 = **63.4 ms/step @ c1**. D-P2r3: 83 tps @ 4.45/step =
  53.6 ms. → gap = **step time +18%** and **acceptance −3.5%**.
- **agg c4**: wall 4.97 s / ~59.6 steps = **83.4 ms/step @ c4** (batch
  cost +20 ms over c1; D's implied c4 step ≈ 64 ms, +10 ms) → the
  batch-scaling axis is a second, independent deficit.
- **prefill (cold)**: rate ≈ 1.34–1.39 k tok/s FLAT across 5.2k/5.7k/8k/
  16k/32k (churn 4.16 s, t1 ladder above). D-P2r3 ≈ 2.0–2.27 k tok/s
  (the 3944/8083/17447 bars), C ≈ 1.18–1.66 k → we are AT C-level at
  16k/32k, behind at 8k, and 69% of the D-bar rate overall.

## Reading

The P4 gaps are step-time, batch-scaling, and prefill-rate — NOT the cache
lane (integrity 5/5 canonical after 160+ bench requests; external chain
byte-true under load — a certification-grade cell). This matches the
plan's own P2→P4 lever ladder, now with numbers attached:

1. **p32/S1 compile posture** (FULL_AND_PIECEWISE; forward replay ≤45 ms
   vs current implied ~; W14 control 33 ms) — the c1 step-time lever.
2. **p33 draft-replay chain** (14 → ≤7 ms) + **p50 E2E residue** (≤15 ms)
   + AR-attribution leg — the rest of the 63.4 ms step.
3. **Batch scaling at c4** (+20 vs +10 ms) — forward batch shape axis.
4. **p14 prefill parity** (1.39 → 2.0 k tok/s at 8k) — suspects: align-mode
   896-chunk serialization, spec overhead on prefill chunks; E-law capture
   leg first.
5. **MTP acceptance tuning** (4.295 → 4.45+ tok/step; pos1 probe).

## Rig

arc-test stopped → radiance-test up → host-side w14_serve_only.sh →
**W14 HEALTH=200** (restored 20:0x). Raw cells: r37_*.out in this dir.
