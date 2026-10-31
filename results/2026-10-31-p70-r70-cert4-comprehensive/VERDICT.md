# P6.10 — comprehensive cert4 measurement campaign (r70)

**Date**: 2026-10-31 · **Phase**: P6.10 (owner directives: "cert4 has to
be measured comprehensively and all information has to be exist based
on last certificated image not from other image" + "all feature,
performance and etc... informations has to be more detailed,
comprehensive and World-class professional grade")
**Image**: `vllm-arc:v0.30.0-b70-cert4` (digest `0c25a4f747af`, tag
commit `b5df28b5`) — the certified image, unchanged. **No engine-code
change; cert4 remains the certified release (no cert5).**

Scope: every current-record face re-measured natively on cert4 —
`PROVENANCE.md` (this dir) is the audit that scoped it;
`R46T_PARITY.md` answers "any regression vs r46t?" (no). Harness and
driver: `kit/` (byte-exact copies of what ran on the rig). Legs: `a/`
(greedy + sweep + ladder + warm + alt, one fresh boot), `b/` (no-spec
A/B, fresh boot), `c/` (4 h soak, fresh boot).

## Incident — first launch ran the compile posture (invalid, quarantined)

The first launch (10:11–10:23 UTC) copied the rig's
`serve_arc_b70_p67pwd.sh`, whose `ARC_GRAPH=1` branch carried the p67
compile-A/B flag (`vllm_compile` + `FULL_AND_PIECEWISE`, ladder capped
256). The tell was immediate: agg 171.7–186.2, solo 51.3–52.2, spec
gate pos1 0.4507 / acceptance 37.4 % / mean length 2.12 — **exactly the
r67a compile signature** (182.04 / 51.66 / 2.21–2.32). Killed; the EXIT
trap restored W14 cleanly; the flag line was restored byte-exact from
the repo canonical `tools/serve_arc_b70.sh` (one-line diff, verified);
relaunch booted the certified posture (engine log: FULL_DECODE_ONLY +
85-size ladder; boot 315 s). The invalid outputs are preserved in
`a_invalid_compile_posture/` — an accidental third replication of the
compile rejection, not certified-posture data.

## Leg A — cert4 default (k3), fresh boot 10:25–10:35 UTC

| Face | Result |
|---|---|
| boot → health 200 | 315 s; posture verified in-engine (FULL_DECODE_ONLY, ladder 1–320) |
| survival (first request) | HTTP 200, 7.33 s (p12b interlock warmup included) |
| agg4x256 greedy t0 | 147.07 (warmup #1) → **294.68 / 276.44 / 296.42** steady |
| solo200 greedy t0 | **83.66 / 83.68 / 84.43 / 84.48** |
| spec gate (light load) | PASS hard (pos1 0.9258); acceptance 83.7 % window, mean len 3.51; strict-target rc=4 on the light-load window (see battery-window number below — the 0.85 contract target is met on the full-battery window) |
| **greedy acceptance, battery window** (metrics deltas) | **85.9 %, mean accepted length 3.58** (Δ1382 drafts / Δ4146 draft tok / Δ3562 accepted) — cert4-native replacement for the p46-era 85.6 % |
| solo U-bars | U8k 3.86 s / 2,111 t/s · **U16k 7.87 s / 2,078 t/s (first-ever 16k bar)** · U32k 17.36 s / 1,875 t/s · U64k 40.26 s / 1,622 t/s (prompt cached≈0 — faces valid; family 3.74–3.86/17.08–17.36/39.19–40.26 with r66a/r67) |
| warm v1 (seeds shared with ladder — t1 rows INVALID, quarantined in analysis) | warm-hit re-fires 802/809/805 ms (valid); superseded by warm v2 |
| xhigh sweep (agg steady / solo / acceptance window) | t0.2 205.63 / 64.2 / 56.6 % · t0.4 221.69 / 65.2 / 55.8 % · t0.6 202.39 / 67.1 / 54.5 % · t0.8 204.76 / 60.0 / 50.5 % · t1.0 194.44 / 58.6 / 49.0 % |
| alt face | agg 280.94 / **264.00** steady; solo 73.10 / 72.45; acceptance 87.7 % |

## Leg B — cert4 no-spec (ARC_SPEC=0), fresh boot 10:35–10:41 UTC

Posture purity: `--expect-off` PASS (no spec counters). Survival 200 /
6.17 s.

| Face | no-spec result | MTP k3 (leg a) | ratio |
|---|---|---|---|
| agg greedy t0 | 124.41 / **126.46** | 296.42 | **2.34×** |
| solo greedy t0 | 34.30 / **34.31** | 84.48 | **2.46×** |
| agg xhigh t1.0 | 113.45 / **126.10** | 194.44 | **1.54×** |
| solo xhigh t1.0 | 33.61 / **33.62** | 58.62 | 1.74× |
| agg alt | 109.21 / **115.93** | 264.00 | **2.28×** |
| solo alt | 33.63 | 72.78 | 2.16× |

(Alt run-1 cells show early-stop streams at 132/168 tok — presence
penalty 1.5 pushes EOS; throughput counts tokens/wall honestly.)

## Leg C — cert4 soak (4 h) + warm-v2 tier attribution

Warm v2 (fresh seeds 6157/6159/6161, run on the soak boot after cycle 1
— `c/warm_rehit2.*`, engine-log windows `c/serve_{pre,post}_warm2.log`):

| Face | cold t1 | warm t2 |
|---|---|---|
| U8k (8,145 tok) | 3.871 s | **0.988 s** |
| U16k (16,341 tok) | 8.348 s | **1.075 s** |
| U32k (32,567 tok) | 17.297 s | **1.109 s** |
| U8k warm-hit re-fires | prime 3.048 s | **0.902 / 0.902 / 0.906 s** |

**Tier attribution (answers the owner's "LMCache hits or some else
hits?")**: engine log during the warm window — "Prefix cache hit rate"
41.2 → 46.2 % climbing, **"External prefix cache hit rate: 0.0 %"**
throughout; LMCache lines in the window: 62 async **stores**, **zero
retrievals**. A same-boot warm re-hit is served by **engine APC**
(`--enable-prefix-caching`, GPU-resident blocks); LMCache's activity in
that window is store-only. (The completions API does not return
`prompt_tokens_details.cached_tokens` on this build — usage-level
breakdown absent is why the log-window + external-rate counters are the
attribution evidence.)

Soak close-out: **IN FLIGHT at commit time** — leg c started 10:45:48
UTC (4 h window → ends ~14:50 UTC). Interim at 11:00 UTC: 4 cycles
complete, health 200 every cycle, integrity probe P1_SOAK_PASS bad=0,
zero ERROR/Traceback in the serve log, battery steady (agg 299.94,
solo 84.4). Final counts (cycles / battery failures / integrity
failures / final spec gate / full-engine ERROR+WARN audit + W14-restore
confirmation) are banked by the follow-up close-out commit together
with the mirrored `c/` evidence; this VERDICT is updated by that commit
and never before the data exists (E-law).

## Outcome

- Current record rebuilt cert4-native (PERFORMANCE-arc "Current
  certified image" + README table + CONFIGURATION/TESTING updates);
  history labeled by image and never mixed in.
- No engine change; `rel/v0.30.0-b70-cert4` remains the certified
  release. `tools/check_currency.sh` and `tools/leak_sweep.sh` pass.
