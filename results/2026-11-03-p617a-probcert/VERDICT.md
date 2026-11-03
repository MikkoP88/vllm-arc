# P6.17 — prob-lane certification campaign (VERDICT)

**Date:** 2026-11-03 · **Image:** `vllm-arc:v0.30.0-b70-cert6` (byte-unchanged
from P6.14 — no cert7; the serve script is not shipped in the image) ·
**Serve script:** rig `/root/build/serve_arc_b70.sh`, P6.16 hook form for legs
a/b/c (env `ARC_SPEC_METHOD=probabilistic`, sha256 fb631d35…), FLIPPED default
for the validation boots d/e (sha256 57ff1564…, identical to
`tools/serve_arc_b70.sh` on this branch; pre-flip bytes preserved on the rig
as `/root/build/serve_arc_b70_p616.sh`).

**Owner directives (2026-11-03):** the certified line is
`ARC_SPEC_METHOD=probabilistic`; ALL record values come from that lane —
missing faces measured; prob lanes are the superior lanes; old non-prob-lane
record values cleaned from the current docs (kept only as the labeled opt-out
alternative + banked history).

## What ran

| leg | boot | posture | purpose |
|---|---|---|---|
| a | 251 s | prob (env form) | comprehensive: survival, t0 ×2, strict spec gate, integrity + det, golden capture, solo TTFT ladder + solo pair, warm re-hit v1 + log windows, xhigh sweep 0.2→1.0, alt face, garbage suite, hook audit, per-phase metrics snapshots |
| b | 243 s | prob (env form) | family boot: t0 ×2, t1.0 ×2 (first-sampled-face window + recovery), strict gate, integrity + det, golden capture+compare |
| c | ~260 s | no-spec (`ARC_SPEC=0`) | MTP-over-no-spec denominators: t0, t1.0, alt (`--expect-off` verified first) |
| d | 296 s | **default-path** (flipped script, env unset) | certification link: engine cmdline must equal leg-a bytes; full battery + gates |
| e | 283 s | **greedy opt-out** (`ARC_SPEC_METHOD=greedy`) | opt-out carries the exact pre-P6.17 bytes (`draft_key_count=0`), t0 in-family, strict gate |
| f | 303 s | prob (explicit) | fresh-seed warm2 probe (`p70_warm_rehit2.py` — closes the v1 t1-row validity gap) |

Plus the offline **flag probe** (5/5, `probe/flag_probe.txt`): PATH-stubbed
docker on the real flipped script — unset / `probabilistic` / junk → prob
bytes; `greedy` → the exact pre-P6.17 bytes; `ARC_SPEC=0` → no spec flag.

## Record faces (probabilistic lane, all from this campaign unless noted)

- **t0 agg4x256 steady** (post-warmup faces): 291.37 / 295.05 / 298.19 /
  298.29 / 303.13 (warmup #1 faces 261.34–272.63). **Solo t0:** 83.73–84.85
  across a/b/d (one-off degraded pair 42.01/44.92 on b#2 — unreproduced in
  12 other solo faces, disclosed; same class as the p615 † footnote).
- **t1.0-xhigh agg steady:** 222.66 / 223.92 / 232.70 / 241.29 (this
  campaign) — in-family with p616b 241.10 and p616c 226.87 (cross-boot
  family 222.7–241.3). **Solo t1.0:** 64.03–71.02, uniform — zero
  collapse legs in 20 measured pairs (greedy lane bank: 61.73/61.79 with
  ragged TQ-dtype faces).
- **First-sampled-face-after-t0 transient (disclosed, posture-invariant):**
  the first sampled battery after sustained t0 measures ~134–137
  (b#1 134.13/134.76; d#1 136.92; a t0.2#1 133.56) and recovers within
  1–2 faces. r70 GREEDY t0.2#1 = 134.39 — identical class, i.e. NOT a
  posture property. The one-off b#2 t0 solo pair above is a separate
  (unreproduced) event.
- **Temp sweep steady (agg #2 faces):** 214.93 / 214.95 / 222.25 / 224.65
  (t0.2/0.4/0.6/0.8). Greedy r70 bank: 205.63 / 221.69 / 202.39 / 204.76 —
  greedy t0.4 BEATS prob there (parity at cool temps; prob pulls ahead with
  temperature). Honest record, not smoothed.
- **Alt face (t0.7/tp0.80/pp1.5):** 271.51 (#1) / 281.34 (#2), solo
  82.68 / 92.14 (greedy r70 bank 264.0–280.9 agg, 72.45–73.10 solo).
- **Solo TTFT ladder:** U8k 3.77 s (2164 tok/s prefill) / U16k 7.90 s
  (2069) / U32k 17.38 s (1873) / U64k 40.32 s (1620) — within noise of the
  r70 greedy bank (posture-independent prefill).
- **Warm faces (fresh-seed v2 probe, leg f):** t1 cold 3.89/8.39/17.44 s →
  t2 warm 0.99/1.08/1.07 s (3.6–3.9×); U8k warm-hit re-fires 0.845/0.839/
  0.852 s (leg-a family 0.80–0.81). v1's t1 rows were corpus-shared with
  the ladder (r70 quarantine) — v2 seeds 6157/6159/6161 are disjoint.
  Engine usage does not emit `cached_tokens` (None on all rows): tier-1
  attribution = behavioral face + counter window (leg-a warm window
  prefix-hit 121,472/138,354 queries).
- **Per-face acceptance (engine-counter deltas, leg a):** t0 battery 0.859 /
  mean-len 2.58 (≡ r70 greedy 85.9 % — posture-invariant at t0); sweep
  t0.2→t1.0: 0.583/0.571/0.611/0.606/0.613 (mean-len 1.75/1.71/1.83/1.82/
  1.84); alt 0.911 / 2.73; strict-gate window 0.837 / 3.51 / pos1 0.9258 —
  identical on boots a/b/d AND on the greedy opt-out boot e (gate workload
  is t0 → argmax path).
- **MTP over no-spec (same-day A/B, legs a/c):** t0 295.05+298.19-family vs
  124.46/127.06 → 2.35–2.44× (solo 84.85 vs 34.31 → 2.47×); t1.0
  222.66–241.29 vs 115.25/121.83 → 1.84–2.09×; alt 271.51–281.34 vs
  107.14/116.02 → 2.42–2.53×.

## Gates

| gate | result |
|---|---|
| first-request survival | 200 ×6 (dt 6.18–8.39 s; f=200 untimed) |
| integrity | 5/5 PASS ×3 measured boots (a/b/d) |
| determinism | 3/3 PASS ×3 (a/b/d) |
| golden (capture→compare vs `p25/golden_ref.json`) | 5/5 identical, zero drift (b/d; leg-a attempt hit the known `--run` invocation flaw — fixed on b) |
| garbage suite v2 | 15/15 (a) |
| strict spec gate | hard floor PASS everywhere (pos1 0.9258 ≥ 0.5); acceptance 0.837 vs the 0.85 greedy-calibrated target → rc=4 TARGET-MISS, identical class on the greedy lane (r70 measured the same 0.837; the 85.9 % battery-window face is the tabled acceptance) |
| engine log hygiene | 0 ERROR / 0 Traceback ×6 boots |
| flag probe (flipped script) | 5/5 |
| certification link (d==a) | boot-d default-path engine cmdline bytes == leg-a env-form bytes (ps audit `cmp` identical) |
| greedy opt-out bytes (e) | exactly the pre-P6.17 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`, `draft_key_count=0` |

## Disposition

- **Certified line:** `ARC_SPEC_METHOD=probabilistic` — now the serve
  script's DEFAULT (unset → prob bytes). `ARC_SPEC_METHOD=greedy` = the
  opt-out historical lane (pre-P6.17 bytes, byte-identical at t0).
- **Owner override recorded:** the P6.16 pre-registered default-flip rule
  (every dtype ≥ +10 % agg) would have kept greedy (k8v4 EV-neutral at
  −0.3 %); the owner directives of 2026-11-03 override it — prob is the
  certified default on the strength of fp8 +29 % / tq4 +44 % / k3v4 +54 % /
  3bit +91 %, uniform sampled solo streams, and the correct-sampler
  property. Ledger in PERFORMANCE-arc "Default-flip ledger".
- **Image:** cert6 remains THE certified image (serve-layer default change
  only; wheel + engine bytes untouched; no cert7).
- Docs rewritten to make the prob lane THE record (README headline +
  Quick start, PERFORMANCE tables, CONFIGURATION, envs, TESTING, MIGRATION)
  with the greedy lane as the labeled opt-out alternative and banked
  history. All values above are backed by files in this directory.

## Rig closure

W14 incumbent (`radiance-test`) stopped per leg, restored by EXIT trap after
every leg and after validation (12:15, 12:37, 12:50) — health 200 verified
post-campaign. No arc image left serving.
