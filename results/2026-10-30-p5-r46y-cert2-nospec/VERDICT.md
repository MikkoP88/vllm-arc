# r46y — cert2 cut + tag-built image verified; non-spec performance tested

**Date:** 2026-10-30 · **Outcome: GREEN — `rel/v0.30.0-b70-cert2` is the valid certified release; non-spec faces measured on the tag-built image**

## Part 1 — cert2 re-certification (closes the "is cert valid?" gap)

The original `rel/v0.30.0-b70-cert` (61923191) predates the r46u
glue-alias fix — building from it reproduced the defect, which is why
the docs had moved to p47/main. Verified this session: the ONLY
engine-code delta cert..main is r46u (`git diff --stat` = 2 files,
+30/−4); everything else is docs/tools/evidence.

**cert2 cut**: annotated tag `rel/v0.30.0-b70-cert2` at the r46x docs
commit (SHA a78fe3e928e16efaa5eeb2bd6246c9bfc2661ecf), pushed to origin
+ github. Docs (README quick start, CONFIGURATION) build from and
reference cert2; the original cert tag is marked superseded.

**Tag→image reproducibility proven**: `git archive` of the TAG → rig
`build_image.sh` → `vllm-arc:v0.30.0-b70-cert2` (11.2 GB). Manifest
(evidence/cert2_manifest.txt): `fork_sha=a78fe3e9…` == tag SHA exactly
(the A/B check the docs teach), 4/4 wheel md5s = wheels.pins.md5,
wheel_guard PASS, in-build gates PASS. **r46u glue repro image-native on
the tag-built image: 13/13 PASS, ALL_GREEN** (evidence/cert2_repro.txt).
Build log: evidence/cert2_build.log (stack layers cached; ~7 min).

## Part 2 — non-spec performance testing on cert2 (owner-requested)

Per owner directive: non-spec PERFORMANCE TESTING only — non-spec stays
out of the recommended configurations (docs unchanged on that point;
`speculative_config=None` serves are a testing/baseline posture, run as
`ARC_SPEC=0` through the uniform tool or by omitting
`--speculative-config`).

Leg: fresh arc-test from cert2, certified posture minus the spec flag,
full 24-var env chain. `speculative_config=None` confirmed in the engine
config line; glue-unavailable 0; p41c wrap live both workers; health 200
in ~260 s (evidence/serve_keylines.txt). Battery = Run-7n
(agg4x256 ×2 + solo200 ×2), raw output in evidence/nospec_bench_cert2.txt:

| face (non-spec, cert2) | agg4x256 | solo200 |
|---|---|---|
| greedy t0 (non-thinking) | 72.5 warmup / **126.5 steady** | **34.3 / 34.3** |
| t1.0 xhigh thinking (real-world) | 121.1 / 115.2 (mean 118.2) | **33.6 / 33.6** |

Cross-check vs the banked r46t nospec legs (p46 image): t1.0 xhigh agg
mean 118.2 vs 118.8, solo 33.6 vs 33.5 — identical within noise: the
cert2 image's non-spec posture reproduces the r46t baseline. The greedy
non-spec face (not in r46t) lands at 126.5 agg / 34.3 solo vs the
certified k3 greedy face 299.9 agg / 84.5 solo — **MTP k3 value on the
greedy face: 2.37× agg / 2.46× solo**; on the t1.0 xhigh face 1.57×
(k3 186.2) — consistent with r46t's reported 1.5–2.1× real-world range.

## Rig state after the leg

arc-test left Up on cert2 with the serve STOPPED; W14 incumbent
restored (radiance-test started + w14_serve_only.sh, boot observed in
progress at 20:11Z — serve proc alive, health polled to 200 in the
session close-out). Operator's private swift-checkpoint testing: boot
arc-test (or a new container) from `vllm-arc:v0.30.0-b70-cert2` with
their own mounts — the image now carries the r46u fix natively; the
r46u hot-patch container is superseded and was recreated.
