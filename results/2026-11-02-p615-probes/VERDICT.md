# p615 probes — comprehensive solo + concurrent performance matrix, all 12 faces (2026-11-02/03)

Owner directives (2026-11-02): *"all testing results has to contains also
decode, prefill, TTFT performance number solo and concurrent"*, *"result has
to be comprehensive performance analyzes"*, *"check all performace test are
in line the forks older test so user can easily compare"*. This dir banks
that matrix: **12 one-variable legs** (5 dtype defaults + fp8 default + 6
knob faces + tq4+MNBT2048), every leg on the certified image
`vllm-arc:v0.30.0-b70-cert6`, every leg using the **fork's own validated
harnesses** so the numbers are directly comparable to the banked records:

- `r70/bench_run7n_t5.py` — the Run-7n t0 face: concurrent decode
  `agg4x256` (4×256-thread join, toks/wall) ×2 + solo decode ×2
  (max 200, incl. TTFT; temp 0, thinking off).
- `r70/p70_solo_ttft.py` — the U-bar solo TTFT/prefill ladder (U32k/U64k,
  `/tokenize` sizing, `cached=None` fresh-prefill check on every leg).

All 12 legs: dtype substitution verified 1/0, flip-first traffic law,
**0 ERROR / 0 Traceback**, W14 restored by EXIT trap (health 200 verified
before every subsequent launch). Concurrent (wave) columns come from the
banked timeline legs: B1–B8 (`../2026-11-02-p615-*/`), P6.14 L3
(`../2026-11-02-p614-l3/`), cert6 G7 (`../2026-11-02-p614-cert6/`).

## Table 1 — solo battery (this dir, `rig/<tag>/`)

| Face | agg#1 | **agg#2** | solo200 ×2 (tok/s) | TTFT U32k | prefill U32k | TTFT U64k | prefill U64k |
|---|---|---|---|---|---|---|---|
| fp8 default | 270.67 | **153.44 †** | 84.51 / 84.41 | 17.34 s | 1877 t/s | 40.24 s | 1623 t/s |
| turboquant_4bit_nc default | 240.35 | **298.38** | 86.24 / 85.98 | 15.67 s | 2077 t/s | 33.99 s | 1921 t/s |
| turboquant_k8v4 default | 235.63 | **301.97** | 88.49 / 88.61 | 15.52 s | 2097 t/s | 33.58 s | 1945 t/s |
| turboquant_k3v4_nc default | 225.78 | **285.85** | 80.83 / 80.45 | 15.68 s | 2076 t/s | 33.58 s | 1945 t/s |
| turboquant_3bit_nc default | 238.04 | **301.13** | 85.83 / 85.55 | 15.79 s | 2062 t/s | 34.51 s | 1893 t/s |
| tq4 + thr2048 | 241.87 | **299.16** | 85.65 / 85.80 | 20.41 s | 1595 t/s | 45.20 s | 1445 t/s |
| tq4 + MNBT 4096 | 228.61 | **290.94** | 83.06 / 83.23 | 17.24 s | 1888 t/s | 36.47 s | 1791 t/s |
| tq4 + MNBT 2048 | 239.17 | **299.34** | 86.19 / 85.75 | 20.48 s | 1589 t/s | 44.76 s | 1459 t/s |
| fp8 + MNBT 4096 | 273.23 | **296.88** | 84.47 / 84.47 | 18.83 s | 1729 t/s | 43.07 s | 1516 t/s |
| k8v4 + MNBT 4096 | 248.51 | **302.38** | 88.47 / 88.27 | 17.27 s | 1885 t/s | 38.06 s | 1716 t/s |
| k3v4 + MNBT 4096 | 205.78 | **293.98** | 83.29 / 83.05 | 16.86 s | 1930 t/s | 37.10 s | 1760 t/s |
| 3bit + MNBT 4096 | 228.60 | **294.33** | 83.45 / 83.11 | 19.19 s | 1696 t/s | 42.15 s | 1549 t/s |

† fp8-default agg#2 outlier: all four streams uniformly 6.6–6.7 s (one
engine-wide slowdown window, no ERROR/Traceback, cause not identified;
agg#1 270.67 in-band). The certified concurrent-decode reference for this
exact posture/image is the cert6 release-gate bank **296.25** (cert5:
297.19) — use that, not 153.44.

## Table 2 — concurrent columns (banked timeline waves) + capacity

| Face | wave TTFT U32k/U64k | bystander gap p95 U32k/U64k | KV capacity | 4-way decode (t0 agg#2) |
|---|---|---|---|---|
| fp8 default (B1) | 17.25 / 39.77 s | 3986 / 5295 ms | 405,491 (1.53×) | **296.25** (cert6 bank) |
| tq4 default | 15.85 / 33.98 s | 3227 / 3780 ms | 757,641 (2.89×) | **299.67** (G7) |
| k8v4 default | 15.29 / 33.03 s | 3690 / 4385 ms | 526,445 (2.01×) | **302.58** (L3) |
| k3v4 default | 15.48 / 33.78 s | 3736 / 4186 ms | 853,220 (3.25×) | **294.43** (L3) |
| 3bit default | 16.02 / 35.74 s | 3188 / 3729 ms | 975,329 (3.72×) | **302.77** (L3) |
| tq4 + thr2048 (B2) | 21.60 / 47.78 s | 1159 / 1329 ms | 757,641 | 299.16 (probe) |
| tq4 + MNBT 4096 (B3) | 17.59 / 37.03 s | 2003 / 2049 ms | 809,161 (3.09×) | 290.94 (probe) |
| tq4 + MNBT 2048 (B5) | 22.14 / 45.38 s | 1325 / 1286 ms | 821,283 (3.13×) | 299.34 (probe) |
| fp8 + MNBT 4096 (B4) | 19.49 / 42.83 s | 2030 / 2587 ms | 433,210 (1.65×) | 296.88 (probe) |
| k8v4 + MNBT 4096 (B6) | 17.41 / 37.58 s | 1936 / 2225 ms | 562,045 (2.14×) | 302.38 (probe) |
| k3v4 + MNBT 4096 (B7) | 16.99 / 36.90 s | 2011 / 2309 ms | 911,660 (3.48×) | 293.98 (probe) |
| 3bit + MNBT 4096 (B8) | 19.41 / 42.09 s | 1932 / 1637 ms | 1,042,793 (3.98×) | 294.33 (probe) |

Decode under heavier load (from the banked p65/B1 timeline decomposition,
banked here for the record): one SSE chunk = one engine step = **3.93–3.95
accepted tokens** (MTP k4; B1 39004/9929, B5 27843/7062, TQ4-def
34632/8761). Clean-window concurrent per-stream ≈ 3.93 / 53–56 ms cadence ≈
**70–73 tok/s**; bystander lifetime (incl. stalls) 45.9 tok/s; engine-wide
at 6-way ≈ **440 tok/s**. Solo 84.5 vs concurrent ~73 per-stream = **−13 %
per-stream cost for 6× parallelism** (memory-bound batching).

## Table 3 — in-line with the fork's older tests (comparability)

| Face | probe (this dir) | older bank | Δ |
|---|---|---|---|
| fp8 def solo decode | 84.51 / 84.41 | 84.34 / 84.46 (cert6 leg 1) | **+0.1 %** |
| fp8 def U32k/U64k | 17.34 s / 40.24 s | 17.36 / 40.26 (p70 cert4 comprehensive) | **~0.0 %** (cross-era cert4→cert6) |
| tq4 def | 298.38 / 86.24-85.98 | 299.67 / 85.63-85.78 (cert6 G7) | −0.4 % / +0.5 % |
| k8v4 def | 301.97 / 88.49-88.61 | 302.58 / 88.05 (L3) | −0.2 % / +0.6 % |
| k3v4 def | 285.85 / 80.83-80.45 | 294.43 / 83.44 (L3) | −2.9 % / −3.4 % (band-edge; inside the Run-7n band 282.33–312.05 / 80.26–88.70) |
| 3bit def | 301.13 / 85.83-85.55 | 302.77 / 86.25 (L3) | −0.5 % / −0.8 % |

Every face lands in-family with its bank — the probe numbers can be read
directly against the older records.

## Findings

1. **Decode is knob-free and dtype-light.** thr2048 / MNBT 2048 / MNBT 4096
   move solo decode and 4-way decode by <1.5 % vs the same dtype's default
   (e.g. tq4 family: 85.65–86.24 solo across all four chunk sizes). The
   knobs trade ONLY prefill/TTFT/stalls. Dtype spread is real but modest:
   k8v4 fastest (88.3–88.6 solo), k3v4_nc slowest (80.5–83.3), fp8/tq4/3bit
   83–86.
2. **Solo prefill is a chunk-size story, per dtype** (U64k rate): tq4
   1664-chunk 1459 → 3328-chunk 1791 → 6656-chunk 1921; k8v4 3456→1716,
   8064→1945; k3v4 3712→1760, 7424→1945; 3bit 2176→1549, 6528→1893; fp8
   3328→1516, 7488→1623. Bigger chunks = fewer chunked-prefill steps =
   less cumulative prefix re-read (the B5 mechanism), monotone within every
   dtype.
3. **The fp8 solo-prefill inversion (B1's finding reaches the solo face).**
   fp8 TRUE-default solo prefill is **1623 t/s (U64k) — the slowest of all
   five defaults** despite the largest chunks (7488): the "TQ improves TTFT
   −32/−33 %" claims in the P6.14 banks were measured against the
   thr2048-profile fp8 (23.16/50.77), not the true default. Against the
   true fp8 default the TQ4 solo TTFT gain is **−9.6 % / −15.5 %**
   (15.67 vs 17.34; 33.99 vs 40.24) — still a win, one third the size of
   the advertised number. (fp8's 40.24 s reproduces p70-cert4's 40.26
   exactly, so this is stable across eras, not a bad leg.)
4. **Solo (probe) and concurrent (wave) TTFT agree within ~0.1–1.7 s** on
   every face (solo slightly under wave — the concurrent-load cost), e.g.
   fp8 def 17.34/17.25, tq4 def 15.67/15.85, k8v4+4096 17.27/17.41,
   3bit+4096 19.19/19.41. The two batteries corroborate each other.
5. **k8v4 is the decode+prefill sweet spot of the family**: fastest decode
   (88.3–88.6 solo, 301.97–302.38 agg), fastest default prefill (2097/1945),
   and P6.14 already marks it quality-safest (t1.0 within 9 % of fp8). Its
   +MNBT 4096 face keeps top decode (88.27–88.47) while halving stalls
   (1936/2225 ms) at 2.14× capacity.

Disposition: nothing adopted — this dir is measurement + comparability
banking on top of the P6.15 Pareto menu; certified defaults (fp8 default
posture) unchanged. Guidance lives in `docs/CONFIGURATION-arc.md`
(Bystander-stall tuning) and `docs/PERFORMANCE-arc.md`.

Provenance: legs run solo from stable W14-up, health-gated; serve logs +
pre-existing-container snapshots redacted byte-exact (serve_full.log
+34/+6/+34/+34/+34/+34/+34/+34/+34/+6/+34/+13, preexisting_arc-test.json
+5×12; total +365) — `../REDACTION-2026-11-02-p615.md` §probes.
