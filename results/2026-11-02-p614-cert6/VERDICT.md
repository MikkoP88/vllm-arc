# p614 cert6 — release gate on the tag-built image `vllm-arc:v0.30.0-b70-cert6` (2026-11-02)

**Flow (single-flow law):** squash-merge `feat/arc-p614-tq-lagstore` →
main `986e97d6ac76df138347daa0bab2dc1546ea36e8` → annotated tag
`rel/v0.30.0-b70-cert6` → image built FROM THE TAG (`build_image.sh`,
log `rig/cert6_build.out`) → the two-leg gate below on the tag-built
image itself. Candidate-leg evidence (engine-identical p614c2 image):
`results/2026-11-02-p614-{dbg,preflight,l1v1,l1v2,l2,l3}/`.

## Leg 1 — certified DEFAULT posture + slice face (`rig/cert6_gate/`)

| Gate | Result | Detail |
|---|---|---|
| manifest fork_sha | **PASS** | `fork_sha=986e97d6ac76df138347daa0bab2dc1546ea36e8` == tag commit; lmcache 0.5.6+arc1 pin unchanged |
| image-native verify | **PASS** | p48 battery **28/28** + p73 slice-hunk **11/11** (in-container md5 path — adapter `33812d19…` == repo) |
| default boot | PASS | health ~305 s; `SLICE_ENV_ABSENT=0` (no p65/slice envs); flip-first 8.4 s → 200 |
| run7n t0 | **PASS (in-family)** | agg#2 **296.25**, solo 84.34/84.46 (cert5: 297.19/84.48) |
| run7n t1.0 xhigh | **PASS (in-family)** | agg#2 **186.48**, solo 61.73/61.79 (cert5: 189.41/62.35; family band) |
| integrity + det | **PASS** | 5/5 + 3/3 |
| WARN class-set | **PASS — identical** | 44 lines == cert5's 44, class-set diff **0** both ways; 0 ERROR / 0 Traceback |
| slice face (same image) | **PASS** | U32k/U64k TTFT 22.70/51.00 s (cert5: 22.66/50.77 family); W_U64k_start max **1604 ms** (cert5 1602 — one-shot tail stays gone); store classes 832×220 + 1664×44 + **3328×38 exact** (slice signature), **63232-class = 0**; end windows clean; 0 ERROR/Traceback |

## Leg 2 — first certified opt-in TQ face: `turboquant_4bit_nc` (`rig/cert6_tq4/`)

One-variable leg off the same image (sed substitution verified **1/0**,
`serve_diff.txt` line 52 = the dtype line only):

| Gate | Result | Detail |
|---|---|---|
| G1 errors | **PASS** | 0 ERROR / 0 Traceback (pre-fix A/B: 112 pairs) |
| G2 kit store path live | **PASS** | `[r43kvfmt] … HS=262` ×2 workers + `[r43xfer] … H=262 dir=D2H` ×2 (A/B: 0 kit lines) |
| G3 external tier | live | max External prefix cache hit rate 5.9 % (first time under TQ) |
| G4 MemoryObj GC | **PASS** | 0 GC-warning lines (A/B: 272) |
| G5 integrity + det | **PASS** | 5/5 + 3/3 (A/B: 3/5) |
| G6 spec gate | **PASS** | pos1 0.9714 ≥ 0.5 hard, acceptance 0.9381 ≥ 0.85 target (mean len 3.81) |
| G7 run7n t0 | **PASS** | agg#2 **299.67** (band 282.33–312.05), solo 85.63/85.78 (band 80.26–88.70) |
| capacity | **reproduced** | KV pool **757,641 tokens = 2.89×** (vs fp8 405,491 / 1.55×) at the same 6.92 GiB |
| TTFT | **reproduced** | U32k **15.85 s** / U64k **33.98 s** (fp8 same-rig: 23.16/50.77 → −32/−33 %) |
| t1.0 xhigh | banked | agg 93.76, solo 77.30/25.85 — the known TQ-inherent ragged pair (dtype-wide, owner-call) |
| bystander gaps | banked | W_U32k_start p95 **3227** / max 3329 ms; W_U64k_start p95 **3780** / max 3918 ms (fp8: 1273–1304 / 1574–1604); end windows clean (58.4/54.3 ms p95); baseline p50 55.8 ms |

## Verdict

**cert6 = the P6.14 release: every gate GREEN on the tag-built image.**
The default posture is cert5's engine byte-identical (WARN class-set
identical, all default faces in-family, slice signature exact) — the
P6.14 hunk is proven inert under `fp8_e4m3`. The 4bit TQ face is the
first **certified opt-in** kv-cache dtype: store path healthy (0
errors, kit D2H live both workers, external tier participating),
integrity/determinism/spec all hold, capacity +1.87× and U-bar TTFT
−32/−33 % reproduce exactly. TQ-inherent costs (ragged t1.0 solo pair,
bystander gaps ~2.5×) are dtype-wide, unchanged by the release, and
remain owner-call material (per-preset matrix:
`results/2026-11-02-p614-l3/VERDICT.md`).

`rel/v0.30.0-b70-cert6` = **current certified release**.

Redaction: 6 files, convention masks, deltas +15/+5/+55/+125/+5/+125
(total +330, arithmetic exact) — `results/REDACTION-2026-11-02-p614.md`
§cert6. W14 restored by the leg EXIT trap (radiance-test up, health
verified post-leg).
