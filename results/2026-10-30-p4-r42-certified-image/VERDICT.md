# r42 — certified image boot: vllm-arc:v0.30.0-b70-p45b — all §10 gates green, all superiority bars beaten, warm-hit bar met for the first time

First standalone-image boot of the full patch set (no PYTHONPATH cap, no
runtime pip installs). First boot of p45 died (two lmcint wheel-hook
circular imports + `ModuleNotFoundError: aiofile`); root cause + fixes
below. p45b = p45 + aiofile/redis vendored into the image + standalone
sub-package hook loading. Battery = first-traffic, TIMERS-OFF posture
(r42_serve.sh = r30 line verbatim minus the cap PYTHONPATH export — the
single delta), D-certified tools md5-pinned unchanged.

## Root causes of the p45 boot death (both banked as laws)
1. **Image not self-contained**: the lmcache 0.5.5+arc1 wheel installs
   `--no-deps` and its Requires-Dist closure was only partially vendored
   (`aiofile`, `redis` missing). `storage_backend/__init__:14` imports
   `gds_backend` unconditionally and `gds_backend.py:19` does
   `import aiofile` at module level — the first `lmcache_integration`
   import (connector `__init__` -> lmcint adapter -> `cache_engine:62`
   -> storage_manager) dies. Legs r30–r41 never saw it because
   `r30_boot.sh:27-29` pip-installed aiofile+redis into the RUNNING
   container at every boot — container state, never image content.
   **Fix A**: Dockerfile closure delta `pip install "aiofile" "redis"`
   (placed after the stack RUN so its layer stays cached) + in-build gate
   `import aiofile, caio, redis` + IMAGE_MANIFEST `lmcache_closure=` line.
2. **Sub-package wheel hooks executed the package `__init__`**:
   `import_module("...lmcint.lmc_kvfmt_detector")` first runs
   `lmcint/__init__`, which IS the lmcache_integration entry (imports the
   heavy `vllm_v1_adapter` -> `lmcache.v1.cache_engine` at module level);
   both hooked targets exec MID lmcache init (detectors.vllm finishes
   while cache_engine is still executing; torch_ops while
   `lmcache/__init__` is) → re-entry into partially initialized lmcache →
   ImportError in both workers. The r30 cap loaded these same files
   standalone via `spec_from_file_location` (files byte-identical cap ↔
   wheel, md5-checked; the only relative import in the kit is the
   `__init__` itself). **Fix B**: `platform.py::_load_wheel_patch()` —
   sub-package patches load standalone (cap semantics), top-level patches
   import normally. Commit 00af061a.

## Faces (vs champion bars, D-certified tools)

| face | bar | r41 (cap overlay) | r42 (certified image) |
|---|---|---|---|
| solo200 tps | **79.74** | 86.31–86.64 | **85.01/85.11/85.16** (+6.7% vs bar) |
| agg4x256#1 | 254.33 | 272.58/278.12 | **266.68/266.72** (+4.9%) |
| agg4x256#2 | **274.41** | 300.60/308.07 | **297.65/297.94** (+8.5%) |
| TTFT t1 8k | 3944 | 3729 | **3445** (−12.6%) |
| TTFT t1 16k | 8083 | 7788 | **7252** (−10.3%) |
| TTFT t1 32k | 17447 | 16630 | **15552** (−10.9%) |
| prefill rate | 1836 (D) | 2132 | **2308 tok/s** (7950/3.445s) |
| warm t2 8k/16k/32k | — | 1108/922/1426 | **845/665/1179** |
| warm-hit (prefix) | **≤ 750 ms** | 990 ms | **740 ms — BAR MET** (champion band 0.587–0.741) |

Solo/agg sit −1…−3% vs r41's cap leg (cross-boot variance class); every
throughput and TTFT bar still beaten with margin. c1 step ≈ 200/4.14
steps in 2.35 s ≈ 48.7 ms (≤ 60 gate; D 43).

## Gates
- **survival**: 832-token flip payload FIRST traffic HTTP 200 in 7.02 s;
  small-after 200.
- **spec gate: pos1 0.9020 ≥ 0.85 target MET** (hard 0.5); profile
  [0.902/0.804/0.647/0.588], acceptance 0.7353 — per-position decay
  IMPROVED vs r41 ([0.892/0.762/0.632/0.562]); the r38-open convergence
  item closes positively on the certified posture.
- **integrity 5/5** cold-vs-warm-re-hit byte-identical (LMCache D-law);
  **determinism 3/3**.
- **golden: 3/5 drift vs pre-p44 ref = expected numerics class** — and
  the run shas are byte-identical to r41's on ALL 5 cells (64/2048/4096
  drifted cells reproduce r41 exactly; 256/1024 match ref) → certified
  image numerics == r41 numerics. **golden_ref RE-BANKED**
  (golden_run.json → /root/build/p25/golden_ref.json, md5 e02cbb07…).
- **wheel hooks: 7/7 OK in every process** (mamba_utils ×2,
  mamba_hybrid ×2, lmcache_connector ×4, device_ops ×3, torch_ops ×3,
  kvfmt detector ×3, scheduler ×1), zero FAILED, zero aiofile lines;
  fp8 reroute ACTIVE ×4; `[r43kvfmt] TWO_HS` fired ×6 post-traffic.
- prefix posture: cold 2.24 / warm 0.74 / churn 112@2.46 — warm-hit bar
  met; rehit-after-churn fell to cold class (2.24 s = full eviction under
  churn this boot; informational face, not a gate).

## Owed next
1. r43: k5 A/B re-decision on the new forward (script staged,
   one-sed num_speculative_tokens 4→5) — re-canonicalizes sha batteries
   if adopted.
2. 18-cell matrix + run7n ×2 archive on the certified image.
3. P5: 8h soak, env/log audit (no markers, no IEXP_*), docs/envs.md,
   write-up + MIGRATION addendum, squash-merge on §10 gates.

## Rig end-state
arc-test up on p45b (r43 continues immediately); radiance-test stopped
for the arc block; W14 restore owed at block end. Artifacts:
rig /root/build/legs/arc-p4-r42/ + this dir.
