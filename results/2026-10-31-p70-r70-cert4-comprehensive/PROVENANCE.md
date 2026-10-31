# P6.10 provenance inventory — which current-record numbers were cert4-native before r70

**Owner directive (2026-10-31)**: "cert4 has to be measured comprehensively
and all information has to be exist based on last certificated image not
from other image." This table is the audit that scoped the r70 campaign:
every number in the current record (README "Certified performance" +
PERFORMANCE-arc "Current certified image"), with the image that actually
measured it. r70 = the campaign that closes every NO row natively on
`vllm-arc:v0.30.0-b70-cert4` (tag `b5df28b5`, the certified image).

## Already cert4-native (pre-r70)

| Claim | Value | Native evidence |
|---|---|---|
| agg4x256 steady t0 (thinking off) | 299.15 / 300.16 (p65 gates) · 300.48 (r67a baseline) | `results/2026-10-31-p65-cert4/` (Gate A rc + Gate B tag-built), `results/2026-10-31-p67-r67a-compile-ab/` |
| solo200 t0 | 84.53 / 83.65 · 84.86 / 84.54 · 84.49 (r67a) | same |
| t1.0 xhigh battery | agg 190.62 / solo 56.90, 62.24 | `results/2026-10-31-p65-cert4/` Gate A (cert4rc image) |
| prefill solo U-bars U8k/U32k/U64k | 3.75/17.08/39.20 s (r67 stock re-confirm of r66a 3.74/17.33/39.19) | `results/2026-10-31-p67-r67a-compile-ab/`, `results/2026-10-31-p66-r66a-prefill-capture/` |
| MTP mean accepted length (greedy) | 3.40–3.57 tok/step | r67a acceptance gate on cert4 boots |
| p65 opt-in long-context face | U64k 51.16 s; bystanders 1580/1599 ms | `results/2026-10-31-p65-cert4/` Gate B (tag-built image) |

## Inherited from other images (the r70 gap list)

| Claim | Cited value | Actual source image | r70 leg that re-measures |
|---|---|---|---|
| greedy acceptance % (85.6 %) | r46t counters | **p46** (original-cert engine) | leg A spec gate (t0 window) |
| TTFT t1 ladder 8k/16k/32k | 3.4/7.3/15.6 s | r42/r43 legs — **original cert** | leg A solo ladder U8k/**U16k**/U32k/U64k (U16k is new — no image ever measured it) |
| TTFT t1 64k | 40.9 s (concurrent-load) | r46 leg — **original cert** | leg A U64k solo (39.19 s already cert4-native via r67; r70 re-runs for same-session n) |
| warm t2 8k/16k/32k + warm-hit 740 ms | 848/675/1178 ms; 740 ms | cert-era prewarm battery — **original cert** | leg A warm re-hit probe, **with cache-tier attribution** (owner question: engine-APC `cached_tokens` vs LMCache connector — probe records both, engine-log window banked around it) |
| temp sweep xhigh 0.2–1.0 + alt face | r46t tables | **p46** | leg A sweep + alt battery (same session, cert4) |
| MTP-over-no-spec ratio 1.57×/2.05× | r46t baseline legs | **p46** | leg B no-spec boot (t0 + t1.0 xhigh + alt) |
| 8 h soak 46 cycles zero errors | r46 soak | **original cert** | leg C soak (4 h default — same-day rig window; labeled honestly, not as 8 h) |

## Mechanism notes for the tier-attribution question

The certified line runs TWO prefix-reuse tiers simultaneously:

1. **Engine APC** (`--enable-prefix-caching`, block 64): GPU-resident block
   hash table. A re-request whose blocks are still in the GPU pool is served
   here — surfaces as `usage.prompt_tokens_details.cached_tokens`.
2. **LMCacheConnectorV1** (`kv_both`, CPU 100 GiB local tier, 1024-tok
   chunks, `use_native`): serves prefixes no longer GPU-resident (eviction
   recovery, cross-request longer horizons); visible in engine-log
   lookup/store lines + connector counters.

The banked warm faces (740 ms class) were same-boot re-requests seconds
after the cold request — the expected serving tier is engine APC with
LMCache's store completing asynchronously post-cold-request; r70 measures
`cached_tokens` per warm request and banks the engine-log window so the
attribution is evidence, not architecture argument.

## Harness integrity (post-campaign)

The campaign kit is mirrored byte-exact in `kit/` (md5):

| File | md5 |
|---|---|
| `kit/serve_arc_b70_p70.sh` | `2c20cec58cd6d940cf4d8561ab181d28` — **byte-identical to the repo-canonical `tools/serve_arc_b70.sh`** (the serve line the certification docs specify; the accidental p67 variant was reverted to exactly this) |
| `kit/r70_campaign.sh` | `4a34758457c5092b9abe0d517fc3de6a` |
| `kit/p70_solo_ttft.py` | `a77f2732d2562af2052bb90bc8eec9c1` |
| `kit/p70_warm_rehit.py` | `f7917b7cce882115110a4cb0a0d26271` (v1 — seeds shared with ladder; t1 rows invalid, quarantined in analysis) |
| `kit/p70_warm_rehit2.py` | `a99af6535dcdac34c69556317bdca5e1` (v2 — fresh seeds; the warm table in VERDICT) |

## Resolution — every gap row closed natively (r70)

| Gap row (above) | r70 result (cert4-native) |
|---|---|
| greedy acceptance % | **85.9 %**, mean length 3.58 (battery window; light-load window 83.1–83.7 %) — leg a |
| TTFT ladder | U8k **3.86 s** · U16k **7.87 s** (first-ever 16k bar) · U32k **17.36 s** · U64k **40.26 s** — leg a |
| warm t2 + warm-hit | t2 **0.988 / 1.075 / 1.109 s** (8k/16k/32k), re-fires **0.802–0.906 s** — leg c warm v2 |
| sweep + alt | full table in VERDICT leg a / PERFORMANCE "Sampled / thinking faces" |
| MTP-over-no-spec | **2.34×/2.46× (greedy), 1.54×/1.74× (t1.0 xhigh), 2.28×/2.16× (alt)** — legs a/b |
| 8 h soak → cert4 soak | 4 h soak leg c (labeled 4 h, not 8) — close-out in VERDICT |

Tier-attribution method note (planned vs actual): the planned
`cached_tokens` channel is **absent** on this build (completions API
returns usage without `prompt_tokens_details`), so attribution rests on
the engine-log window around the warm faces ("Prefix cache hit rate"
41→46 % climbing, **"External prefix cache hit rate: 0.0 %"**, 62
LMCache stores / 0 retrievals) — engine APC serves the same-boot warm
re-hit; see VERDICT leg C.
