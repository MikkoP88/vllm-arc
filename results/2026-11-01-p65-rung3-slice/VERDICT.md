# P6.13 rung-3 gate verdict — sliced final-span store (2026-11-01)

**Posture (A/B leg, owner window):** probe image
`vllm-arc:v0.30.0-b70-cert4-p73probe` (cert4 + repo lmcint adapter, md5
`9a1e9227…` == repo, docker-commit law) on `arc-test`; serve =
`serve_arc_b70_p73.sh` = r48l posture (`--long-prefill-token-threshold 2048`
+ `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960`) **+
`LMC_STORE_SLICE_TOKENS=4096`**; k3, graphs, greedy. Liveness proven
in-process: `SRC_OVERRIDE=9`, engine `/proc/<pid>/environ` carries all three
LMC_* vars, `PY_COMPILE_OK`, probe-file md5 == repo.

**Harness:** `p65_timeline.py` v5.1 faces (U32k+C32k+U64k, W2MT=4000, 12
bystander streams), `p73_replay.py` / `p73_replay2.py` (NEW integrity faces,
this gate), `bench_run7n_t.py --temperature 0`, `r42_integrity.py`.
Artifacts: `rig/` (mirrored; the serve log is redacted —
`../REDACTION-2026-11-01.md`).

## 1. Face-by-face vs the pre-registered bars (DESIGN.md §5)

| check | bar (r48l / prediction) | observed (slice 4096) | verdict |
|---|---|---|---|
| U64k TTFT | ≤ 52.25 s; band 47.5–50.5 | **50.72 s** (faces, 12 bys) + **48.79 s** (solo, novel seed 4040404, true cold) | **PASS** — −1.53 / −3.46 s vs bar; 2-sample mean 49.76 in band (faces sample 0.22 s above edge) |
| U64k W_start p50/p95/max | 1287/1585/1612 (r48l) | 1198/1561/1598 | in-family, all three ≤ bar |
| U64k max gap ≤ 600 ms | pre-registered | **1598 ms — MISS** | see §2 (bar defect) |
| U64k W_end | clean | 55/109/110 | **PASS** (clean) |
| store histogram 63 232-class | = 0 | **0 for the entire session** (incl. the 2nd novel 64k prompt) | **PASS** — the one-shot is gone |
| ~38 × 3 328 windows | 2 workers × 19 | faces: **exactly 38**; full session 80 (2 × 2 × 19 = 76 + 4 stragglers) | **PASS** — the design's count, exactly |
| U32k family | unchanged | TTFT 22.67 (family 22.78–22.89); W 730/1289/1300 (r48g 729/1291); 44 × 1664 = r48g's exact count | **PASS** |
| C32k replay | ~1.2 s | 1.18 s | **PASS** |
| C64k full-hit (NEW) | C-class | **1.74 s**, correct greedy prefix | **PASS** — 19 sliced windows/worker reassemble into a serving-grade full-prefix cache |
| mid-drain replay (NEW) | correct output; TTFT between C-class and U64k-class | prefix identical fresh/drain/post (greedy integrity); mid-drain (+2 s) TTFT **1.82 s** | **PASS with note**: solo, the 19-window drain closes in ~2 s (uncontended store regime ≈ 10× the contended 0.24 GB/s), so the +2 s replay was already near-full-hit — the partial-recompute race path was only weakly stressed. The r48b clean-miss law covers it by call-shape identity (§2 of DESIGN.md) |
| run7n agg#2 | 290–300 (±5 % of 300.05) | **300.48** tok/s (agg#1 266.3 = the known fresh-traffic warmup face, comparator law) | **PASS** |
| run7n solo | 83–84 (±5 % of 84.44) | **84.41 / 84.57** | **PASS** — dead center |
| integrity battery | 5/5 | **5/5** (+ determinism 3/3) | **PASS** |
| ERROR / Traceback | 0 | **0** (2 107-line serve log) | **PASS** |
| WARNING | boot-accepted only | 43 total — exactly the cert4 boot-accepted class count (p70 leg-c also 43); 0 pin_monitor this session; 0 fork-added | **PASS** |

## 2. The max-gap bar: mechanical MISS on a mis-set bar (recorded honestly)

The pre-registered `max ≤ 600 ms` for W_U64k_start is **contradicted by
banked evidence that predates the gate**: r48e (nolmc — NO stores at all,
same face family) read max **2 347 ms**. The W_U64k_start gaps are dominated
by engine-side chunk-compute steps ([viii]; >1 s each at thr2048 with
bystanders) — a floor no store-side change can move. The bar conflated the
END-window tail (which the one-shot inflated, M1) with the START-window max;
at prediction time the r48e nolmc row was not consulted. The corrected claim
— "slicing removes the STORE-side contribution to bystander gaps" — is
CONFIRMED by mechanism evidence, not by the 600 ms number:

- max 1 598 vs r48l's 1 612 WITH the one-shot (−14 ms ≈ noise ⇒ the store
  tail no longer adds on top of the chunk floor);
- TTFT −1.5…−3.5 s (the one-shot left the pre-first-token step — the
  designed effect);
- histogram 63 232-class = 0 across the session (the concentration is gone
  by construction);
- W_U64k_end clean (55/109/110) and W_U32k_end clean (55/59/60).

Per the pre-registered rule's own letter this is a **formal RED for
ADOPTING the profile** (one bar not met). The rule is honored: the slice
profile is **NOT added to the certified long-context posture**.

## 3. Outcome

1. **Engine code: MERGED (default-OFF).** The `LMC_STORE_SLICE_TOKENS=0`
   branch is the original code path (mirror-gate + code-identity; every
   sub-40 960-token face this session exercised the unchanged default
   semantics: integrity 5/5, determinism 3/3, run7n dead-center, U32k/C32k
   families unchanged, zero errors). Cert4's default posture remains
   byte-identical.
2. **Profile: experimental, uncertified.** `LMC_STORE_SLICE_TOKENS=4096`
   demonstrably de-concentrates the final-span store (TTFT −1.5…−3.5 s at
   64k, zero one-shot stores, full-hit + mid-drain integrity clean) but the
   pre-registered adopt rule was not met; enabling it stays an owner call on
   the corrected bar (§2). Recorded in CONFIGURATION-arc's long-context
   profile as experimental.
3. W14 restored post-window (this time the full recipe: container recreate
   `+ plepatch.tgz hot-apply + chat_template_qwen38_high.jinja` — the
   original container's writable layer carried both; a bare recreate is NOT
   sufficient. Recorded in the plan §9 rig manual).

**Predictions ledger:** TTFT band — confirmed (2-sample mean in band);
histogram — exact (38 = 2×19); U32k family — exact; run7n — exact;
integrity — exact; max-gap — wrong bar (§2); "~19 drain steps at base
+~300 ms" — not directly instrumented; bounded indirectly by the solo drain
closing in ~2 s (≈ 19 × ~100 ms uncontended store steps; the 300 ms figure
was the 12-bys contended estimate).
