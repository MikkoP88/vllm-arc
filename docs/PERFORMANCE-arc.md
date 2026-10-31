# Performance — vllm-arc, qwen3.8-27b-fp8 on dual Intel Arc Pro B70

Measured performance of the certified posture (standalone numbers; test
definitions and hardware in [`TESTING-arc.md`](TESTING-arc.md)).
**Read "Measurement methodology" first** — it defines every face, the
protocol laws, the two prefix-cache tiers, and the provenance rule: the
current record is cert4-native only; everything else is labeled
history. All certified faces: first-traffic fresh boot, timers off, XPU
graphs on, TP2, greedy temp 0. **Recommended MTP depth = k3** (r46t
campaign, reproduced natively on cert4 by r70 — below); the
certified-greedy history at k4/k5 is retained row-labeled — k5 is
dropped from the recommendation; its evidence is superseded, not
retracted.

## Measurement methodology — how every number in this file is produced

**Provenance rule (owner law, 2026-10-31):** the current record cites
**only faces measured natively on the current certified image**
(`vllm-arc:v0.30.0-b70-cert4`). Numbers from earlier images (cert /
cert2 / cert3 / p46) are history — labeled by image, never mixed into
current-record tables.

### Rig, image, and posture identity

| Layer | Identity |
|---|---|
| Hardware | dual Intel Arc Pro B70 (48 GB each), TP2, one card per socket (`ZE_AFFINITY_MASK=0,1`) |
| Engine image | `vllm-arc:v0.30.0-b70-cert4` — tag `rel/v0.30.0-b70-cert4`, commit `b5df28b5`, image digest `0c25a4f747af`, `IMAGE_MANIFEST.fork_sha` verified at build (p65 Gate B) |
| Wheels | md5-pinned at build (`tools/wheel_guard.sh`, 4 pins: vllm/triton/vxk/lmcache 0.5.6+arc1) |
| Serve line | the certified 25-var env chain + CLI ([`CONFIGURATION-arc.md`](CONFIGURATION-arc.md)): MTP k3, `mode:none`, FULL_DECODE_ONLY graphs (85 capture sizes, 1–320), MNBT 8192, LMCacheConnectorV1 `kv_both` |
| r70 campaign (the comprehensive cert4 measurement) | 3 fresh boots on the one image — greedy+sweep leg (a), no-spec leg (b), soak leg (c) — evidence `results/2026-10-31-p70-r70-cert4-comprehensive/` |

### Face definitions (exact)

| Face | Definition |
|---|---|
| agg4x256 | 4 concurrent chat requests, `max_tokens=256`, prompt "Count from 1 to 1000.", thinking off; throughput = Σtokens/wall across the 4-thread join. Two runs per battery; **run #1 after boot or an idle window is warmup-contaminated, run #2 is the steady face** |
| solo200 | one chat request, `max_tokens=200`, same prompt; tps = tokens/wall **including TTFT** |
| xhigh sweep | the agg/solo battery at temperature 0.2–1.0 with the real-world sampling set (top_p 0.95 / top_k 20 / min_p 0 / presence_penalty 0 / repetition_penalty 1.0) and `enable_thinking: true, reasoning_effort: "xhigh"` |
| alt face | temperature 0.7 / top_p 0.80 / top_k 20 / presence_penalty 1.5, non-thinking (near-deterministic continuation regime) |
| U-bars (U8k/U16k/U32k/U64k) | SOLO streaming completion on a uniform synthetic prompt, sized via `/tokenize` to the face target, per-face seed (fresh prompts every boot — prefix-cache hit must be ≈0 or the face is invalid), `max_tokens=32`; TTFT = send → first content chunk; prefill rate = prompt_tokens/TTFT |
| warm t1/t2 | the U-bar request fired twice identically: t1 = cold (cached ≈ 0), t2 = the same request seconds later; plus 3× warm-hit re-fires (U8k) for a stable warm-hit number |
| no-spec A/B | the same battery faces on a second fresh boot with `--speculative-config` removed; spec counters verified absent (`--expect-off`) before measuring |
| soak | fresh boot; cycles of (agg battery + 24-prompt integrity probe + health check + counter snapshot) every ~5 min for the soak window |

### Protocol laws (apply to every certified face)

- **Fresh-boot, first-traffic**: every battery/probe runs on a freshly
  booted engine; the first request after boot exercises the p12b warmup
  interlock (the survival face) and is never scored.
- **Timers off** (`ARC_TRACE_FLIP=1` interlock) and **XPU graphs on**
  (`VLLM_XPU_ENABLE_XPU_GRAPH=1`, FULL_DECODE_ONLY) — standing rig laws.
- **Same-face comparator law**: only same-face, same-session numbers
  compare (thinking on/off and battery warmth are confounders —
  `results/2026-10-31-p6-r47c-p48-056gates/VERDICT.md`). Cross-boot
  same-face compares are reported as families (ranges), not point
  deltas. Sampling params at temperature 0 are inert (argmax survives
  top_k/top_p), so greedy faces stay comparable across batteries.

### The two prefix-reuse tiers — which cache serves a warm hit

The certified line runs two tiers at once:

1. **Engine APC** (`--enable-prefix-caching`, block 64): GPU-resident
   prefix blocks, hit-checked by the scheduler; engine telemetry =
   "Prefix cache hit rate" in the serve log.
2. **LMCacheConnectorV1** (`kv_both`; CPU 100 GiB local tier, 1024-tok
   chunks): serves prefixes no longer GPU-resident (eviction recovery,
   long-horizon cross-request reuse); engine telemetry = "External
   prefix cache hit rate" + LMCache lookup/store log lines.

**Measured attribution (r70)**: during the warm t2 faces the engine log
shows prefix-cache hit rate climbing 41→46 % with **External prefix
cache hit rate 0.0 %**, and the window holds 62 asynchronous LMCache
*store* events (after the cold requests) and **zero retrievals** — a
same-boot warm re-hit is served entirely by **engine APC**; LMCache's
activity in that window is store-only. LMCache retrieval is the tier
that matters on longer horizons — its faces are the p65/p48 long-context
evidence, not the warm-hit face.

### Acceptance telemetry (speculative decoding)

Upstream-0.30 `/metrics` counters: drafts, draft tokens, accepted
tokens, per-position acceptance. Definitions used throughout:
**acceptance % = accepted/draft tokens**; **mean accepted length =
accepted/drafts + 1** (the +1 bonus token). Windows are counter deltas
snapshotted before/after each battery, so each face's acceptance covers
that face's traffic only.

## Current certified image — cert4 (`rel/v0.30.0-b70-cert4`): the current record

**Every number below was measured natively on the cert4 image** — the
r70 comprehensive campaign (2026-10-31, 3 fresh boots) plus the p65/r67
certification legs; per-face evidence in the right column. **cert4 is
both the LATEST and the FASTEST certified image** — nothing validated
beats it (the two faster-on-paper postures are rejected: k5-depth
owner-dropped — its greedy edge inverts under sampling; compile —
Appendix H law #1 acceptance gate, p67). Identity: tag commit
`b5df28b5`; default posture = cert3's engine byte-identical (the P6.5
deferral is opt-in and inert without its envs); how to run: the 25-var
chain in [`CONFIGURATION-arc.md`](CONFIGURATION-arc.md).

### Throughput — greedy t0, thinking off (chat battery)

| Face | cert4-native value | Boots | Evidence |
|---|---|---|---|
| agg4x256 steady | **294.7–300.5 tok/s** (r70 294.68/296.42 · p65 299.15/300.16 · r67 300.48) | 5 | `…-p70-r70-cert4-comprehensive/`, `…-p65-cert4/`, `…-p67-r67a-compile-ab/` |
| solo200 | **83.7–84.9 tok/s** (r70 83.66–84.48 · p65 84.53–84.86 · r67 84.49) | 5 | same |
| survival (first request after boot) | HTTP 200, 6.2–7.3 s (p12b interlock warmup included) | 3 | p70 |
| first-battery warmup face | agg run #1 147–178 vs steady #2 ≈295+ — first-traffic cost, gone by run #2 | r70 | p70 |

### Sampled / thinking faces (real-world regimes, cert4-native)

| Face | agg steady (tok/s) | solo (tok/s) | acceptance (window) |
|---|---|---|---|
| xhigh t0.2 | 205.6 | 62.5–65.8 | 56.6 % |
| xhigh t0.4 | 221.7 | 64.7–65.8 | 55.8 % |
| xhigh t0.6 | 202.4 | 65.7–68.4 | 54.5 % |
| xhigh t0.8 | 204.8 | 58.9–61.0 | 50.5 % |
| xhigh t1.0 | 194.4 | 55.1–62.2 | 49.0 % |
| alt (t0.7/tp0.80/pp1.5, non-thinking) | 264.0 (peak 280.9) | 72.5–73.1 | 87.7 % |

The p46 campaign's k3 table (the depth decision) is reproduced on
cert4 within ~1 pp acceptance / ±6 % throughput per cell — the **k3**
law now stands on cert4-native data. The p65 t1.0-xhigh battery
(190.62 / 56.90, 62.24) is the same face on an earlier cert4 boot —
inside the family above. Full r46t→cert4 parity table (every face, Δ
column, verdict): `results/2026-10-31-p70-r70-cert4-comprehensive/R46T_PARITY.md`
— **no regression**.

### Prefill / TTFT — solo U-bars, fresh prompts, greedy

| Face | prompt tok | TTFT (cert4-native) | Family (r66a/r67 bars) | prefill rate |
|---|---|---|---|---|
| U8k | 8,158 | **3.86 s** | 3.74–3.86 s | 2,078–2,178 t/s |
| U16k | 16,354 | **7.87 s** — first-ever 16k bar | — | 2,078 t/s |
| U32k | 32,553 | **17.36 s** | 17.08–17.36 s | 1,875 t/s |
| U64k | 65,303 | **40.26 s** | 39.19–40.26 s | 1,622–1,666 t/s |

### Prefix reuse — warm faces with tier attribution

| Face | cold t1 | warm t2 | served by |
|---|---|---|---|
| U8k | 3.87 s | **0.99 s** | engine APC (external tier 0.0 %, zero LMCache retrievals in window) |
| U16k | 8.35 s | **1.08 s** | same |
| U32k | 17.30 s | **1.11 s** | same |
| U8k warm-hit re-fires (n=6, two probes) | — | **0.80–0.91 s** | same |

The cert-era "warm-hit 740 ms / warm t2 848/675/1178 ms" cells used a
different prompt corpus (prewarm-v7 battery) on the original-cert image
— same mechanism (engine APC), different corpus, so those numbers stay
in history and do not compare (comparator law).

### Speculative decoding (MTP k3) — cert4-native

| Condition | acceptance | mean accepted length |
|---|---|---|
| greedy t0, battery window (r70) | **85.9 %** | **3.58** |
| spec-gate light-load windows | 83.1–83.7 % | 3.49–3.51 |
| xhigh t0.2 → t1.0 (sweep) | 56.6 % → 49.0 % | — |
| alt face | 87.7 % | — |
| r67a compile posture (rejected) | — | 2.12–2.32 (−35 %, the gate that keeps `mode:none`) |

**MTP value over no-spec — cert4-native A/B (fresh boot per side,
r70 legs a/b):**

| Face | MTP k3 | no-spec | ratio |
|---|---|---|---|
| agg greedy t0 | 296.4 | 126.5 | **2.34×** |
| solo greedy t0 | 84.5 | 34.3 | **2.46×** |
| agg xhigh t1.0 | 194.4 | 126.1 | **1.54×** |
| solo xhigh t1.0 | 58.6 | 33.6 | 1.74× |
| agg alt | 264.0 | 115.9 | **2.28×** |
| solo alt | 72.8 | 33.6 | 2.16× |

### Stability — cert4-native

- r70 soak (4 h, battery+integrity cycles): in flight at this commit —
  interim state clean (every cycle health 200, integrity bad=0, zero
  errors); the close-out row with final counts is appended by the
  follow-up commit (evidence: `results/2026-10-31-p70-r70-cert4-comprehensive/c/`).
- Zero boot failures across the cert4 image family: r70 3 boots, p65
  2 gate images, r66 7 probe boots, r67 compile-A/B boots.

### Posture laws in force on cert4 (do not re-litigate without new evidence)

- **`mode: none`** — the 0.30 compile REVAL FIRED the Appendix H law #1
  acceptance gate: `vllm_compile` collapses acceptance length
  3.40/3.57 → 2.21/2.32 (−35 %), decode −39 %, prefill flat (p67);
  r70's accidental first launch re-replicated it exactly (182/52 t/s,
  mean length 2.12 — see `a_invalid_compile_posture/` in the p70
  evidence).
- **MNBT stays 8192** — 16384 A/B: U32k −9.7 % but U8k +9.9 % and KV
  pool −18 %; documented, not adopted (p66). Never combine with the
  p65 trio.
- Any `FULL_AND_PIECEWISE` repeat must cap the capture ladder at 256
  (`max_num_seqs × (k+1)`; the `gdn_attn.py:120` law, p67).

Decode-step anatomy, allreduce share, and the prefill op-class table
(GEMM 39.7 % / AR ~30.4 % / FA 20.3 % / GDN 4.7 %): see "Prefill
anatomy (p66)" and "Prefill structural rungs (p67)" below.

## Image history — depth-selection faces (cert-era images; the k-depth decision record)

Measured on the original-cert/cert2-era engines (r42 k4 leg, r43 k5
leg, r46t k3 on the p46 image). The current-image reproduction of this
family is the cert4 record above (294.7–300.5 / 83.7–84.9 across five
cert4 boots; sweep acceptance within ~1 pp of the p46 campaign —
engine unchanged through cert2→cert4 default postures). Depth law
unchanged: **k3**.

| Face | k3 (recommended, r46t) | k4 (r42) | k5 (dropped, r43) |
|---|---|---|---|
| agg4x256 steady tokens/s | 299.86 | 297.65 / 297.94 | 303.48 / 305.42 |
| solo200 tokens/s (incl. TTFT) | 84.46 / 84.53 | 85.01 / 85.16 | 86.39–87.13 (×4) |
| greedy acceptance | 85.6% | ~87% | 89.2% (pos-1 0.8919) |

k3 and k4 tie within ±0.7 % on the certified face; k5 is faster there
(+1.9–2.5 % agg, +1.5–2.4 % solo, on 89.2 % greedy acceptance) — but
that greedy-era edge inverts under real-world sampling, and the faces
below decide the depth: k3.

| Face (posture-level) | Value |
|---|---|
| TTFT t1 8k / 16k / 32k (ms) | 3987 / 7238 / 15534 (k5 leg r43); 3445 / 7252 / 15552 (k4 leg r42 — 8k steady face) |
| **TTFT t1 64k (ms)** | **40,926** (concurrent-load run — conservative upper bound) |
| prefill rate at 8k face | 2,308 tok/s (7,950 tok / 3.445 s) |
| warm t2 8k / 16k / 32k (ms) | 848 / 675 / 1178 |
| **warm-hit (prefix re-hit)** | **740 ms** (face limit 750 ms) |
| survival flip (first request after boot) | HTTP 200, 6.09 s |
| solo step time @ c1 (200 tokens / 2.30 s) | ~48.7 ms/step (k5 leg) |
| agg4x256 first run after boot (k3) | warmup cost on run #1 (93.7 vs 299.9 steady) — first-traffic face, gone by run #2 |

Notes: the 8k t1 TTFT is first-cell-of-the-ladder (warmup-sensitive);
the k4 leg's 3445 ms shows the steady 8k face. The 64k cell was taken
while the 8 h soak battery was cycling (conservative — a quiet-window
run would be equal or faster). The 8k/16k/32k ladder and 64k face were
measured on the certified image (r42/r43 legs and r46 leg respectively).

## Real-world faces — the k3 decision (r46t campaign)

**A p46-image campaign** (2026-10-30, the certified engine of that
day). Its job was depth selection under real-world sampling; it picked
**k3**. The r70 campaign re-measured the k3 sweep **natively on
cert4** — acceptance within ~1 pp per cell, throughput ±6 % (see
"Sampled / thinking faces" in the current record above), so the depth
law now stands on cert4-native data. cert4 is the same engine line and
the **fastest certified image**; the only faster-ever-measured posture
(k5-depth, greedy face only) was dropped by owner directive — its edge
inverts under real-world sampling.

Full methodology and evidence: `results/2026-10-30-p5-r46t-temps/`
(VERDICT + raw batteries + acceptance counters). Two request regimes,
both on the run7n battery (agg4x256 ×2 + solo200 ×2):

**A. xhigh thinking + real-world sampling** (temperature sweep, top_p
0.95 / top_k 20 / min_p 0 / presence_penalty 0 / repetition_penalty 1,
`enable_thinking: true, reasoning_effort: "xhigh"`). agg tok/s = 2-run
mean; steady = run-2 mean (run #1 after a fresh boot carries first-
battery warmup, heaviest on k3):

| temp | k3 agg | k4 agg | k3 solo | k4 solo | acceptance k3 / k4 |
|---|---|---|---|---|---|
| 0.2 | 174.9 * | 201.7 | 63.9 | 64.6 | 57.3 % / 50.2 % |
| 0.4 | 208.8 | 184.4 | 63.0 | 60.4 | 57.1 % / 45.9 % |
| 0.6 | 202.8 | 193.6 | 61.3 | 62.3 | 53.1 % / 46.6 % |
| 0.8 | 195.8 | 184.9 | 61.4 | 59.1 | 52.3 % / 43.5 % |
| 1.0 | 186.2 | 182.6 | 61.1 | 54.3 | 49.5 % / 43.2 % |
| mean | **193.7** (steady 205.4) | 189.4 (steady 194.2) | **62.1** | 60.2 | — |

\* run #1 = 137.2 (warmup), run #2 = 212.6. k5 measured 185.8 agg
mean before being dropped (evidence retained in the results dir).

**B. alt sampling, non-thinking** (temperature 0.7 / top_p 0.80 /
top_k 20 / min_p 0 / presence_penalty 1.5 / repetition_penalty 1.0):

| leg | agg #1 / #2 | solo #1 / #2 | acceptance |
|---|---|---|---|
| k3 | 170.6 * / 254.9 | 76.9 / 92.2 | 88.5 % |
| k4 | 236.9 / 252.6 | 85.3 / 104.3 | 82.6 % |

\* warmup-contaminated run #1. Near-deterministic continuation keeps
acceptance high — the one regime where k4's deeper draft pays (solo
104.3 vs 92.2); steady agg is a tie.

**Non-MTP baseline** (serve line minus `--speculative-config`): t1.0
xhigh thinking agg 118.8 / solo 33.5; alt face steady agg 124.2 /
solo 33.4. **MTP value on measured faces: k3 = 1.57× (xhigh t1.0) and
2.05× (alt steady) aggregate over no-spec.**

Mechanism (measured across the sweep): speculative acceptance falls as
temperature rises, and falls faster with deeper drafts (k3 49.5 % vs
k5 33.5 % at t1.0 xhigh; k3 85.6 % at greedy t0). Depth pays only at
high acceptance — real-world sampled thinking sits below the
crossover, which is why k3 is the recommendation and k5's greedy-era
win inverts under sampling.

## 12-cell load matrix (r44 — cert-era image, the load-shape record)

`vllm bench serve`, random dataset, output 512, ignore-eos, 12/12 cells
zero-error. TTFT/TPOT/ITL in ms:

| Cell | TTFT | TPOT | ITL | req/s | out tok/s | Acc % | Acc len |
|---|---|---|---|---|---|---|---|
| i1024-c1 | 583.67 | 8.60 | 46.77 | 0.20 | 102.79 | 89.15 | 5.46 |
| i1024-c4 | 1985.32 | 11.07 | 64.47 | 0.52 | 264.69 | 96.92 | 5.85 |
| i1024-c16 | 6237.01 | 28.97 | 126.85 | 0.64 | 325.55 | 67.96 | 4.40 |
| i16384-c1 | 7380.80 | 9.13 | 49.11 | 0.08 | 42.50 | 88.00 | 5.40 |
| i16384-c4 | 13263.11 | 31.51 | 160.63 | 0.13 | 66.43 | 82.44 | 5.12 |
| i16384-c16 | 46421.90 | 111.19 | 523.98 | 0.14 | 72.38 | 74.70 | 4.73 |
| i32768-c1 | 16083.60 | 10.83 | 52.19 | 0.05 | 23.68 | 76.42 | 4.82 |
| i32768-c4 | 34286.64 | 44.64 | 210.74 | 0.07 | 34.47 | 74.69 | 4.73 |
| i32768-c16 | 97008.14 | 161.02 | 708.16 | 0.07 | 34.84 | 68.28 | 4.41 |
| i131072-c1 | 101094.58 | 20.71 | 66.15 | 0.01 | 4.58 | 43.73 | 3.19 |
| i131072-c4 | 162169.64 | 162.68 | 854.78 | 0.01 | 6.24 | 85.76 | 5.29 |
| i131072-c16 | 903979.67 | 184.16 | 843.54 | 0.01 | 4.78 | 71.91 | 4.60 |

Raw cells: `results/2026-10-30-p4-r44-matrix-certified/cell-*.txt`.

## Speculative decoding detail (64k cell, r46)

Acceptance figures split by sampling condition: the headline faces
above (pos-1 0.8919, spec-gate counters) are **greedy (temperature 0)**
— k5-leg values; k3's greedy acceptance is 85.6 % (r46t). Matrix-cell
and 64k-cell acceptance below are **server-default sampling**
(`vllm bench serve` 0.30 does not force greedy — see
TESTING-arc.md suite 5). TTFT/t1 faces are sampling-insensitive.
Real-world (sampled, xhigh-thinking) acceptance by depth and
temperature: see the r46t table above.

Acceptance 87.43%, mean length 5.37 tokens/step, 218 drafts / 1090
draft tokens / 953 accepted. Per-position acceptance:
96.79 / 96.79 / 96.79 / 96.33 / 50.46 (positions 0–4) — position 4 is
the k5 draft tail; see the 12-cell table above for per-cell positions.

## Prefill anatomy (p66 — analysis of record, 2026-10-31)

Solo TTFT sweep, fresh boot each, nolmc, graph, k3, greedy, unique seeds
(r66a; three faces × four boots — `--long-prefill-token-threshold`
none/8192/4096/2048):

| Face | prompt tok | thr=none | thr8192 | thr4096 | thr2048 | thr2048 Δ |
|---|---|---|---|---|---|---|
| U8k | 8158 | 3.74 s (2178 t/s) | 3.75 s | 4.04 s | 7.94 s | **+112%** |
| U32k | 32553 | 17.34 s (1878 t/s) | 17.35 s | 18.24 s | 21.49 s | +23.9% |
| U64k | 65303 | 39.20 s (1666 t/s) | 39.23 s | 42.38 s | 49.75 s | +26.9% |

thr=none and thr8192 are the same path (≤ 0.03 s apart — at MNBT 8192
the threshold IS the default chunked size). The rate falls with context
(2178 → 1666 t/s, attention KV growth) even before any chunking effect.
thr2048's cost is two mechanisms stacked: more chunk boundaries (8158 tok
= 5 × 1664-token chunks instead of 2 × 7488) **and** small-M device
inefficiency (U8k rate halves: 2178 → 1028 t/s).

**Where the time goes (r66b, per-chunked-step decomposition on the live
V2 runner)**: scheduler 0.3 ms, host prep ≤ 10 ms, TP2 allreduce ~28 ms,
sampling ≤ 107 ms — and the rest is the **device execution of the chunk
itself**, observed as a queue-drain wait inside the next step's
mamba/GDN `preprocess_state` launch window (780–1018 ms of an ~0.9 s
step at 1664-token chunks; XPU eager forward is async, so the enqueue
looks flat at ~127 ms and the device time surfaces one step later).
Prefill is **device-bound at ~34 % MFU** (1664-token chunks) to ~40 %
(7488-token chunks, 2.2–2.4 tok/ms) against the measured 294 TFLOPS
fleet fp16 GEMM ceiling (147–150 TFLOPS/card, 256 CUs, devprobe r66a).
The gap is attention over growing KV + W8A16 dequant + non-GEMM ops +
per-layer launch/dispatch — kernel territory, not host glue.

**Improvement ladder, closed out**:

- thr table (knob) — documented above; the p65 long-context profile
  (`--long-prefill-token-threshold 2048` + deferral) is the one place
  thr2048 is correct, for LMC-side reasons (see CONFIGURATION-arc).
- **MNBT 16384 (knob) — A/B'd, NOT adopted** (r66c): U32k −9.7 %
  (17.33→15.65 s), U64k −4.1 %, confirming the small-M diagnosis — but
  U8k +9.9 % (3.74→4.11 s) and the KV pool pays 18 % (405,491→332,629
  tokens; max concurrency 1.55x→1.27x) from the doubled prefill
  activation workspace. Never combine with thr2048 (the threshold pins
  chunk size — only the KV loss remains). Conditional long-prompt-only
  lever; the certified default stays 8192.
- **Host glue — closed as empty** (r66b): nothing material to cut.
- **Structural (raise device MFU)** — the only remaining large lever
  (prefill compile NONE→PIECEWISE, attention kernel work); deferred,
  A/B-gated, out of P6.6 scope. → **P6.7 ran it; see the next section.**

Evidence: `results/2026-10-31-p66-r66a-prefill-capture/` (curve + device
probe), `results/2026-10-31-p66-r66b-step-decomposition/` (step
decomposition + the engine-map correction + probe-instrumentation laws),
`results/2026-10-31-p66-r66c-mnbt16k/` (MNBT A/B).

## Prefill structural rungs (p67 — compile REVAL + op-class capture, 2026-10-31)

Zero engine-code change; cert4 remains the certified image (no cert5).
Evidence: `results/2026-10-31-p67-r67a-compile-ab/` (VERDICT.md,
r67b_capture_analysis.md, P67_LADDER_VERDICT.md, raw rank traces).

**Compile REVAL (r67a, Appendix H law #1 on 0.30) — REJECTED.**
`mode=vllm_compile` × `cudagraph_mode=FULL_AND_PIECEWISE` (the only legal
combo per the capture law), same-session A/B on the cert4 image: MTP mean
acceptance **3.40/3.57 → 2.21/2.32 (−35 %)**, agg4x256 steady 300.48 →
182.04 t/s and solo200 84.49 → 51.66 t/s (both −39 % — pure arithmetic of
fewer accepted tokens per verify step), prefill faces FLAT (U8k 3.75 vs
3.74 s, U32k 17.08 vs 17.33, U64k 39.20 vs 39.19). KV pool +5.3 %
(426,874 tokens) does not compensate. Boot cost 460 s vs 265 s stock;
3 fork-added jit_monitor WARNs (inference-time Triton JIT of spec kernels).
`mode` stays NONE. Config law banked en route: with MTP k3 and
max_num_seqs 64, every captured size must satisfy the GDN decode-only cap
`max_num_seqs×(k+1)` = 256 tokens (`gdn_attn.py:120`) — capture ladders
must top at 256 under FULL_AND_PIECEWISE (stock ladder headroom to 320
boot-dies at the first piecewise capture).

**Op-class device anatomy (r67b, torch-profiler XPU trace of the U32k
face, rank0, 15.38 s kernel time, 71.9 % busy):**

| Slice | Share | Verified launch site |
|---|---|---|
| GEMM | 39.7 % | `_xpu_C::fp8_gemm_w8a16` (the p44 W8A16 route — fastest on-image per r40) |
| TP2 allreduce (ring+ReduceCopy+barrier) | ~30.4 % | `c10d::allreduce_` |
| FA-layer attention (16 layers, DENSE) | 20.3 % | `_vllm_fa2_C::varlen_fwd` (~31 ms × 102) |
| GDN chunk-scan (48 layers) | 4.7 % | `gdn::ChunkFwdO/tiled/WU/A/InverseOpt` ×336 |
| norms + act + rope + elementwise | ~5 % | — |
| indexer/gather (sparse machinery) | 0.1 % | — |

Consequences: the radiance QSA-splitk whale does not exist on stock 0.30
(the model runs dense FA — no sparse indexer); **direct-fp8 GEMM rung
closed negative** (on-image `XPUFp8BlockScaledMMKernel`/`_xpu_C.fp8_gemm`
exists but r40 measured it 2.2–2.4× slower than the deployed w8a16 at
prefill Ms — the GEMM whale already runs the fastest on-image kernel);
**ESIMD FA-kernel rung re-scoped** to the dense varlen site (GQA 24q/4kv,
head_dim 256, fp16, paged fp8 KV PAGE 1664; est. 5–7 % compute MFU today;
ceiling −8…−12 % U32k TTFT if 1.5–2×) — open as a kernel project; and the
**TP2 allreduce family (~30 %) is the largest unattacked slice** (not in
the owner's ladder; recorded as the top recommendation beyond it).

## Stability

- run7n ×2 per leg agree within 1.5% (steady-state).
- Prefix battery phases byte-identical (deterministic corpus, temp 0).
- **8 h soak (r46) — GREEN**: 46 cycles over a 29,156 s window (each
  cycle: run7n → longctx → prefix/churn → integrity probe), **zero
  error blocks, integrity probes 46/46 PASS, final golden 5/5
  byte-identical (zero drift), speculative pos-1 rate 0.8919 steady to
  the last cycle, health 200 end-state**. Evidence:
  `results/2026-10-30-p5-r46-soak/` (VERDICT + soak log + summary).
