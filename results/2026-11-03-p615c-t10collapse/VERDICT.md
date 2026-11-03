# p615c — t1.0-xhigh collapse mechanism: WHEN / WHY / FIX (2026-11-03)

Owner ask (2026-11-03): *"when this collapse happened — k8v4 solo200
n=200 dt=7.67 s tps=26.08 ← collapse leg: 3× the wall time — and why, and
is a possible fix?"*

This dir banks the diagnostic leg that answers it. One k8v4
default-posture leg on the certified image
`vllm-arc:v0.30.0-b70-cert6` (same one-variable dtype gate as every
P6.15 leg; substitution verified 1/0), greedy flip-first, one throwaway
t1.0 request to absorb first-sampled-request JIT, then
`collapse_trace.py`: **10 streaming solo t1.0-xhigh requests** (byte-exact
banked t1.0 body: prompt "Count from 1 to 1000.", temp 1.0, top_p 0.95 /
top_k 20 / min_p 0 / pp 0 / rp 1.0, `enable_thinking` + xhigh), each
request timestamping every SSE chunk and snapshotting the
`/metrics` spec-decode counters before/after.

## The answer in one line

The engine never slows down. **Wall time = drafts × 41.7 ms** (verify-step
cadence, constant in every request), and the collapse leg simply needs
3.4× more verify rounds because **MTP draft acceptance is dead from
round 1 and stays dead for the whole request** — a per-request property
of the sampled trajectory under 4-bit-V KV noise, not a stall, not a
mid-run degradation, and not fixable by any engine/stall knob.

## Evidence table (all 10 requests, this leg)

`tok/round = (drafts + accepted) / drafts`; per-pos rates are
conditional (pos1 given pos0 accepted, pos2 given pos1 accepted);
`200 = drafts + accepted ± 1` (finish-boundary rounding) on every
request; `wall ≈ drafts × 41.7 ms + 0.01–0.15 s` on every request;
median chunk-dt = 41 ms in all 10.

| req | wall (s) | tok/s | drafts | accepted | tok/round | pos0 | pos1 | pos2 | mode |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.31 | 86.6 | 53 | 147 | 3.77 | 94 % | 98 % | 98 % | healthy |
| **2** | **7.56** | **26.5** | **181** | **18** | **1.11** | **6 %** | 36 %* | 75 %* | **total collapse — the 26-tps face** |
| 3 | 2.29 | 87.3 | 52 | 148 | 3.85 | 96 % | 98 % | 100 % | healthy |
| 4 | 2.37 | 84.4 | 54 | 147 | 3.72 | 94 % | 94 % | 100 % | healthy |
| 5 | 2.41 | 83.0 | 55 | 144 | 3.62 | 95 % | 94 % | 88 % | healthy |
| 6 | 2.73 | 73.3 | 63 | 136 | 3.16 | 83 % | 83 % | 95 % | mild, uniform |
| 7 | 3.02 | 66.2 | 70 | 129 | 2.84 | 93 % | 97 % | **1.6 %** | pos2 dead |
| 8 | 2.37 | 84.4 | 54 | 147 | 3.72 | 93 % | 98 % | 98 % | healthy |
| 9 | 4.28 | 46.7 | 101 | 99 | 1.98 | 88 % | **7.9 %** | 43 % | pos1+2 dead |
| 10 | 2.40 | 83.3 | 55 | 145 | 3.64 | 96 % | 89 % | 96 % | healthy |

\* req2 pos1/pos2 rates sit on n=11/n=4 — the meaningful number is pos0:
the FIRST draft token is rejected on 169 of 181 rounds.

- req2 reproduces the banked collapse leg exactly: 200/7.56 s = 26.5
  tok/s (banked: 26.08 / 7.67 s), 3.2× the healthy wall — and
  181 × 41.7 ms = 7.55 s = the whole wall. Nothing but extra rounds.
- Frequency on this leg (k8v4, this prompt): 6/10 healthy, 1/10 mild,
  2/10 depth-dead, 1/10 total. The banked L3 k8v4 solo pair hit the full
  collapse on 1 of 2 — the flip is per-request, stochastic, small-n on
  both measurements.
- TTFT is untouched: band-0 mean 48–50 ms in every request (tiny prompt)
  — the collapse is purely a decode-phase property.

## WHEN

**At request start (round 1), sticky for the request's lifetime.** req2's
acceptance is dead from the first round (11 accepts scattered across 181
rounds — no healthy prefix, no recovery); healthy requests never degrade
into it (req1/3/4/5/8/10 stay 88–100 % to the last round). It is NOT a
mid-run event — the earlier "collapses mid-run" wording in
`docs/PERFORMANCE-arc.md` is corrected by this leg. Which requests are
born collapsed is a coin flip at t1.0 on 4-bit-V KV; the same flip at
population scale is the ragged solo pair (~77 + ~26) and the ragged
4-way streams (3.7–11.5 s spreads) in every TQ t1.0 bank.

## WHY

Spec-decode verification under sampling requires each MTP draft token to
match the target's sampled continuation; both the draft head and the
target verify run against the SAME quantized KV. 4-bit-V quantization
noise perturbs target logits and draft features; on low-margin
(high-entropy) regions of the sampled xhigh-thinking trajectory the
perturbation breaks the match → rejection. Severity is graded by how
deep the death goes — pos2-dead → 2.84 tok/round (66 t/s); pos1+2 →
1.98 (47 t/s); pos0 too → 1.11 (26 t/s) — and depth maps monotonically
to throughput. Corroboration: **fp8 KV (finer V precision) never
collapses at the same posture** (61.73/61.79 clean pair, 186.48/189.41
agg); **greedy t0 never collapses on ANY dtype** (12-leg probe matrix,
all solo pairs uniform); healthy TQ t1.0 legs run 77–86 t/s — ABOVE
fp8's sampled 61.7 — so 4-bit V costs nothing when the trajectory stays
low-margin. The trigger is the interaction (4-bit V noise × sampled
trajectory), not either alone.

## Ruled out (this leg's negative results)

Engine step stall (median 41 ms in all 10 requests; wall = drafts ×
41.7 ms to 2 decimals); preemption (0 lines; Running: 1 / Waiting: 0;
KV 2.7 %); LMCache/GC interference (only boot-time config lines, trace
phase clean); JIT (kernel warmup 0.00 s at boot + t1.0 throwaway before
the trace); queueing (solo sequential, 1 s gaps); ERROR/Traceback 0/0;
serve-log 10-s windows during the trace 45.4/47.2/52.8/47.2 gen tok/s —
mixed-population average, no zero-throughput window.

Instrumentation notes (honest gaps): (a) req2 shows 33 SSE chunks for
181 rounds — the first ~30 rounds streamed live (48/41/41 ms bands),
then 151 rounds surfaced as 3 coalesced events in a ~6.3 s tail (the
"2086 ms band"). The ENGINE cadence during that tail is proven constant
(181 × 41.7 ms = wall); the coalescing locus (server-side output-queue
merge vs client HTTP buffering) is not pinned by this trace and does
not affect the mechanism. (b) `content_chars`/`reasoning_chars` parsed
0 on every request — the template streams under a field the parser
didn't match; token accounting uses usage + counter identities instead.

## Is a fix possible? (ranked, evidence-based)

1. **No config/stall fix — nothing stalled.** Graph, steps, KV
   allocation, cache all healthy. Spec depth doesn't help: with
   acceptance dead, wall = rounds × 41.7 ms regardless of k (drafts ride
   in the same graphed step); with acceptance healthy, lowering k only
   lowers tok/round. MNBT/thr are decode-neutral (probe matrix).
2. **Sampling side — the practical lever.** Acceptance recovers as
   temperature falls; t0 greedy = zero collapses on every dtype
   banked. Workloads that can run cooler keep the TQ capacity win with
   sampled serving. (Intermediate temps unmeasured on TQ presets —
   flagged gap; only the t0 and t1.0 endpoints are banked.)
3. **Precision side — the guarantee.** fp8 KV is immune (all banked
   t1.0 faces clean): the certified default posture already avoids the
   collapse. k8v4 is the mildest TQ face (−8.5 % agg) but NOT immune —
   this leg IS k8v4.
4. **Engine side — out of scope.** Acceptance-aware spec gating or
   verify-on-dequantized-V would be new engine work in the TQ verify
   path; the scope law (plan §3 + App I: radiance shared-components
   only) excludes it. Noted as the only route to "4-bit V + sampled +
   uniform" — left as an upstream-direction observation, not attempted.

## Disposition

Diagnosis only — nothing adopted; certified defaults (fp8 default
posture — the collapse-immune face) unchanged. TQ presets remain opt-in
per the `docs/CONFIGURATION-arc.md` menu, now with the explicit caveat:
sampled (t≥1.0) serving on 4-bit-V presets carries a per-request
collapse lottery (10–50 % observed, n small); use t0/cool sampling or
fp8 when uniform sampled latency matters.

## Provenance

Leg run solo from stable W14-up, health-gated, W14 restored by EXIT
trap. Harnesses: `p615c_collapse_probe.sh` + `collapse_trace.py`
(deployed `/root/build/p65/p615/`, copies in `rig/` are the exact
artifacts: serve log, boot, flip, warmup, posture, pre-existing
container snapshot, serve script). Serve log + pre-existing json
redacted byte-exact (serve_full.log 1 rig-addr ×6 = +6;
preexisting_arc-test.json 1 host ×5 = +5; total +11; raw-ref grep 0
after) — `../REDACTION-2026-11-03-p615c.md`.
