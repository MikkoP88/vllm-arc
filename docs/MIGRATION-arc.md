# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
directive; identities re-pointed at each certification close-out — most
recently **cert10 (P6.24, 2026-11-05; the oneCCL fp8-wire overlay —
same-version 2022.0.0 rebuild + e4m3-wire large-ring allreduce,
default OFF env `CCL_SYCL_ALLREDUCE_FP8`; opt-in 1.65× at the
dominant AR, every p65 TTFT face beats cert9: U32k −3.8 % / U64k
−8.2 % / U8k −13.0 % / C32k −4.6 %; default posture = cert9-identical,
proven at the gate)**, after cert9 (P6.20, 2026-11-04; the prefill
host-path patch — L3
off-thread LMCache store default-ON + L6 dispatch-late + step_trace
diagnostics; U32k/U64k/U8k TTFT −0.74/−1.64/−0.08 s, spec/identity
faces byte-exact cert8), cert8 (P6.19, 2026-11-04; the
TurboQuant spec-verify corruption root-caused and FIXED — all four
`turboquant_*` kv-cache dtypes servable, `VLLM_ARC_TQ_SPEC_FIX=1`
default-ON, fp8 bytes unchanged), cert7 (P6.18, 2026-11-04 — its dtype
refusal was the interim mitigation, lifted at P6.19), cert6 (P6.14,
2026-11-02) and the P6.17 serve-layer certification (2026-11-03: the
certified line is `ARC_SPEC_METHOD=probabilistic` — the serve script
default flipped). Target: serve **qwen3.8-27b-fp8**
(dense hybrid-GDN, **MTP k3 recommended** — r46t real-world campaign) on
`vllm-arc:v0.30.0-b70-cert10`, replacing the W14 incumbent
(`radiance-test`, flash-next k3) as the rig's serving stack. Everything
below is rig-proven; the full phase-by-phase evidence table is §6.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert10` (tag-built engine from `rel/v0.30.0-b70-cert10` + the P6.24 two-file oneCCL overlay — cert9's engine (exhaustive `vllm_gaudi/` md5 walk diff_lines=0) + `external_patches/oneccl` build product [same-version 2022.0.0 rebuild + `arc-fp8-allreduce.patch` v4]; manifest row `oneccl=2022.0.0+arc1`, fat-lib sha pinned at both SONAME paths; overlay default-OFF — opt-in `CCL_SYCL_ALLREDUCE_FP8=1`; gate evidence `results/2026-11-05-p624-fp8-ar/`) |
| fork SHA baked | `aa2ee17d` (IMAGE_MANIFEST `fork_sha=aa2ee17d11612d9292684cbcbd2825bde4c7a65d` — preserved from cert9, the vllm tree is byte-identical; annotated tag `rel/v0.30.0-b70-cert10` on main; image id `2f86fc2411fb`) |
| validation image | `vllm-arc:v0.30.0-b70-cert10` itself (r99 cert10 gate: identity gates incl. the tree walk; default-posture battery with the FP8 env proven absent — t0 300.39 steady/84.52–84.51, integrity 5/5 + det 3/3, spec pos1 0.9258, long-return t0 clean + one measured t1 disposition (r99c annex); k8v4 TQ carry byte-identity; FP8=1 leg + in-situ wire proof; A/B/A/B perf interleave beating cert9 on every p65 face — `results/2026-11-05-p624-fp8-ar/`; lineage: cert9 `results/2026-11-04-p620-prefill-opt/`, cert8 `results/2026-11-04-p619-tqverify/`, p617 prob-lane `results/2026-11-03-p617a-probcert/`, r70 comprehensive `results/2026-10-31-p70-r70-cert4-comprehensive/`, default faces cert4-native `results/2026-10-31-p65-cert4/`) |
| canonical serve script | rig `/root/build/serve_arc_b70.sh` (the uniform boot interface; carries the **25-var chain** incl. `LMCACHE_TRACK_USAGE=false` — P6.3; run with `ARC_IMAGE=vllm-arc:v0.30.0-b70-cert10 ARC_GRAPH=1 ARC_SPEC=3`; P6.24 opt-in = add ONE env line `export CCL_SYCL_ALLREDUCE_FP8=1` to the serve env — the r99 [3] pattern, everything else unchanged); since **P6.17 the certified default is the probabilistic draft-sampling posture — unset or `ARC_SPEC_METHOD=probabilistic` appends the upstream spec key `draft_sample_method`**, `ARC_SPEC_METHOD=greedy` = the opt-out historical lane (pre-P6.17 bytes), `ARC_SPEC=0` = no spec; flip validated end-to-end — flag probe 5/5 + default-path boot engine bytes identical to the record legs — `results/2026-11-03-p617a-probcert/`; the pre-flip script is preserved on the rig as `/root/build/serve_arc_b70_p616.sh` (provenance backup) |
| long-context profile | rig `/root/build/serve_arc_b70_defer2.sh` (the canonical chain + `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` — P6.5; opt-in, see CONFIGURATION-arc "fp8 stall levers") |
| A/B fallback | same script with `ARC_SPEC=4` (MTP k4 — documented alternative for near-deterministic workloads) |
| golden reference | rig `/root/build/p25/golden_ref.json` (md5 `e02cbb07…`; covers k4 AND k5 — r42/r43 proved byte-identical) |
| regression matrix | rig `/root/build/matrix_arc/leg-2026-10-29-r44-certified/` (12 cells) |
| container name | `arc-test` |

The image is **self-contained**: no cap `PYTHONPATH` overlay, no runtime pip
installs (lmcache 0.5.6+arc1 closure vendored; in-build import gate +
`lmcache_closure=` manifest line — r42 Fix A). Wheel-module patches load via
the in-image `_WHEEL_HOOKS` registry (r45 wiring + `_load_wheel_patch`
standalone loader — r42 Fix B).

## 2. Boot procedure (from W14-serving state)

```bash
# on the rig host (address lives in the untracked local layer, never in
#                  tracked files — publishing-hygiene law; every ssh:
#                  export SSH_ASKPASS=/tmp/askpass.sh
#                  SSH_ASKPASS_REQUIRE=force DISPLAY=:0)
docker stop radiance-test                      # frees port 8000 + both GPUs
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert10 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash /root/build/serve_arc_b70.sh            # boot + serve as one unit
# draft posture: unset/probabilistic = CERTIFIED (P6.17); the certified
# line can be written explicitly as ARC_SPEC_METHOD=probabilistic (same
# bytes); ARC_SPEC_METHOD=greedy = opt-out historical lane
# logs land in the container at /root/serve_full.log (host: docker exec)
# poll health (first-traffic law: batteries run on FIRST traffic after boot)
curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health
```

The script boots the certified container (`--privileged` REQUIRED —
pidfd IPC law, r30 defect #1; `--network host --ipc host`, the model
mount, cache mounts, and the 2 boot-layer `-e` vars) and execs the
serve line with the full 25-var chain — env + CLI as one unit by
construction. Boot takes ~5 min to health 200 (285–295 s at the
cert7–cert10 gates; ~290 s measured at the cert10 gate). Verify in
`/root/serve_full.log`: wheel hooks install (zero `FAILED`, silent
defer is DEBUG), `fp8 reroute ACTIVE`, `kvfmt TWO_HS`, the p40c re-pin
line both workers (`alloc/free_pinned_ptr re-pinned to the pageable
torch baseline`), `glue-unavailable 0`, `[p41c] wrap live`, posture
lines (mode=none, FULL_DECODE_ONLY [1..320], MTP k3, align), and since
cert9 the P6.20 store-worker line after the FIRST request
(`LMC store worker ACTIVE` — the off-thread L3 store, default ON).

## 3. Post-boot verification (any migration or image bump)

§10 continuous row, in order (harness tools unchanged, host-side python3):

1. **survival flip-first**: 832-token flip payload as FIRST traffic → 200.
2. **spec gate** (`arc_spec_gate.py`): pos1 ≥ 0.85 target (0.5 hard).
3. **golden** (`p25/p32_golden.py`): 5/5 shas byte-identical to
   `golden_ref.json` (0-drift law).
4. **run7n ×2**: conservative floors solo ≥ 79.74, agg#2 ≥ 274.41;
   certified steady family at k3, prob lane (p617 + release gates):
   solo **83.7–84.9** / agg **291.4–303.1** (cert10 gate: 300.39 /
   84.52–84.51; cert9 gate: 296.91 /
   84.54–84.48; cert8 gate: 300.65 / 84.53–84.57; cert4-native greedy
   family 294.7–300.5 in-band — the
   argmax path is posture-invariant) — **comparator law (r47c)**:
   a fresh-boot FIRST battery on the same stack reads lower (~206/64.5
   warmup face); gate a fresh boot against a same-posture A/B, not the
   campaign table.
5. **matrix 12 cells** (`r44_matrix.sh`): no face −5% vs
   `leg-2026-10-29-r44-certified` (regression block).
6. **integrity/determinism** (`r42/r42_integrity.py`): 5/5 + 3/3.

Any block → do not migrate; capture per E-law, fix on a branch, re-certify.

## 4. Rollback (arc → W14)

```bash
docker stop arc-test                            # artifacts stay in the container fs
docker start radiance-test
nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_rXX.log 2>&1 &
# poll health: 200 within ~5–10 min (first polls may be 000 — model load)
```

Keep `arc-test` (Exited) on the rig — it carries the leg's serve logs
(`/root/serve_full.log`). Delete only after the leg's artifacts are banked
to `results/`.

## 5. Rig laws that govern the migration window

- NEVER leave the rig on an arc image overnight; W14 restores after every
  arc block (plan §9 / App F etiquette).
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` on all graph legs (set inside the serve
  script).
- Verify interleave/BW matrix after any rig reboot.
- Port 8000 is single-tenant: stop radiance-test before arc legs.
- `--mamba-cache-mode=align` is pinned in the serve line (posture-pin
  law: never ship its removal).

## 6. Evidence table (phase ledger)

One row per phase/leg-group; the "what it proved" details live in
PERFORMANCE-arc / TESTING-arc. All paths relative to `results/`.

| Phase | Evidence | What it proved / banked |
|---|---|---|
| P4 r42 | `2026-10-30-p4-r42-certified-image/` | certified-image boot + battery; self-contained image (Fix A/B) |
| P4 r43 | `2026-10-30-p4-r43-k5-adoption/` | k5 adoption (later demoted by r46t) |
| P4 r44 | `2026-10-30-p4-r44-matrix-certified/` | 12-cell matrix — the regression baseline |
| P5 r45 | `2026-10-30-p5-r45-env-log-audit/` | env/log audit + this addendum |
| P5 r46 | `2026-10-30-p5-r46-soak/` | 8 h soak GREEN + mypy delta=0 + 64k face (P5 gate of record) |
| P5 r46t | `2026-10-30-p5-r46t-temps/` | real-world campaign: **k3 recommended**, k5 dropped, garbage suite v2 |
| P6.1 r47a | `2026-10-30-p6-r47a-ext-hitrate/` | external-hit-rate 0.0 % is a steady-state artifact, not a broken cache |
| P6.2+6.3 r47c/p48 | `2026-10-31-p6-r47c-p48-056gates/` | lmcache 0.5.6+arc1 gates; zero fork-added WARN/ERROR; comparator law |
| P6 r47e | `2026-10-31-p6-r47e-cert3/` | cert3 tag-built image (manifest A/B + image-native 28/28) |
| P6.5 r48a–l | `2026-10-31-p65-r48a-stall-capture/` | stall attribution + length-gated chunk-store deferral (the p65 profile) |
| P6.5 cert4 | `2026-10-31-p65-cert4/` | cert4 gates (candidate battery + p65 face on the shipped image) |
| P6.6 p66 | `2026-10-31-p66-r66a-prefill-capture/`, `…r66b-step-decomposition/`, `…r66c-mnbt16k/` | prefill anatomy: device-bound ~34–40 % MFU, host glue empty, MNBT-16384 A/B not adopted — analysis only, no engine change |
| P6.7 p67 | `2026-10-31-p67-r67a-compile-ab/` | compile REVAL rejected (acceptance −35 % gate); GDN capture-cap law; op-class trace — rungs closed/re-scoped, no engine change |
| P6.10 p70 | `2026-10-31-p70-r70-cert4-comprehensive/` | every current-record face re-measured cert4-native (3 boots): tier attribution (warm re-hits = engine APC), r46t parity no-regression, no-spec A/B 1.5–2.5×; leg c 4 h soak GREEN (56/56 cycles, zero failures, no drift — `c/` mirror) — no engine change, no cert5 |
| P2P p71 | `2026-10-31-p71-r71-p2p-analysis/` | TP2 allreduce path: **direct P2P confirmed** (r71b D2D 10.23 GB/s both directions; r71a all collectives `algo topo sycl`, pidfd IPC live, zero host-staged) — all lanes closed, no fix needed, AR gains are algorithmic |
| P6.5 annex p73 | `2026-11-01-p65-end-stall-analysis/` | END-stall five-mechanism analysis (end CLEAN at 32k; M1 store tail / M2 backlog hangover / M4 admission aliasing) — analysis only |
| P6.13 p73 | `2026-11-01-p65-rung3-slice/` | **sliced final-span store** (`LMC_STORE_SLICE_TOKENS`, default-OFF): rig A/B gate RUN + verdict — all safety/regression faces GREEN (integrity 5/5, run7n 300.48/84.4, families unchanged, 0 errors), mechanism landed exactly (63232-class = 0; 38 = 2×19 windows; U64k TTFT −1.5…−3.5 s); pre-registered ≤600 ms max-gap bar mechanically missed (1598) and recorded as mis-set (r48e nolmc floor 2347) — **code merged default-OFF, enabling profile experimental/uncertified** |
| P6.13 cert5 | `2026-11-01-p65-cert5/` | **cert5 release gates** (tag-built image, single flow): manifest fork_sha == tag, p48 28/28 + p73 slice-hunk 11/11 image-native, SLICE_ENV_ABSENT=0, run7n t0 297.19/84.48 + t1.0 189.41/62.35 in-family, integrity 5/5 + det 3/3, WARN class-set identical to cert4, slice face on the SAME image — U64k TTFT 50.77 (cert4 51.95), W_U64k_start max 1602 vs 1997, 63232-class 0 / 3328×38 exact — superseded by cert6 |
| P6.14 cert6 | `2026-11-02-p614-cert6/` (+ candidate legs `2026-11-02-p614-{l1v2,l2,l3}/`) | **cert6 release gates** (tag-built image, single flow): manifest fork_sha `986e97d6` == tag, p48 28/28 + p73 slice-hunk 11/11 image-native, SLICE_ENV_ABSENT=0, run7n t0 296.25/84.46 + t1.0 186.48/61.79 in-family, integrity 5/5 + det 3/3, WARN class-set identical to cert5 (44 == 44), slice face on the SAME image (3328×38 exact, 63232-class 0, W_U64k_start max 1604) + the first certified opt-in TQ face `turboquant_4bit_nc` on the same image (0 errors, kit D2H live @262, spec 0.9714/0.9381, KV 757,641 = 2.89×, U32k/U64k TTFT 15.85/33.98 s) — superseded by cert7 (P6.18): fp8 behavior byte-identical, but its TQ dtype faces degenerate on this model (P6.18 Class B) |
| P6.15 dtype/stall probes + collapse mechanism | `2026-11-02-p615-{stall-analysis,b6,b7,b8}-*/`, `2026-11-02-p615-probes/`, `2026-11-03-p615c-t10collapse/` | bystander-stall taxonomy + chunk-granularity Pareto matrix (all dtypes, MNBT 4096 halves stalls on every preset); 12-face solo/concurrent decode/prefill/TTFT master matrix on the fork's own harnesses; t1.0-xhigh sampled-collapse mechanism pinned (per-request MTP acceptance death from round 1, engine cadence constant, fp8 immune, no config fix) — analysis only, no engine change |
| P6.16 draft-sampling posture | `2026-11-03-p616a-probsmoke/`, `2026-11-03-p616b-probmatrix/`, `2026-11-03-p616c-spechook/` | **opt-in `ARC_SPEC_METHOD=probabilistic`** (serve hook in `tools/serve_arc_b70.sh`): upstream-config-native `draft_sample_method` key on the pinned 0.30.0 wheel; t1.0-xhigh agg fp8 +29 % / tq4 +44 % / k3v4 +54 % / 3bit +91 % (k8v4 EV-neutral), uniform solo pairs (zero 26-tok/s legs in measured faces), t0 provably + measurably identical, det 3/3 everywhere, 11/11 hook-deployment gates PASS — **merged serve-layer: default stays greedy (superseded at P6.17 — the P6.17 default flip, next row), wheel/image unchanged, cert6 remains the certified image (no cert7 — the serve script is not in the image)** |
| P6.17 prob-lane certification | `2026-11-03-p617a-probcert/` | **the certified line is `ARC_SPEC_METHOD=probabilistic`; every record value re-measured natively on that lane** (2026-11-03): 5-boot comprehensive campaign on cert6 — t0 agg 291.4–303.1 / solo 83.7–84.9, t1.0 steady family 222.7–241.3 + uniform 64–71 solo (zero collapse pairs in 20), TTFT 3.77/7.90/17.38/40.32, warm t2 0.99–1.08 (fresh-seed warm2 probe; t1 colds 3.89/8.39/17.44) + warm-hit 0.80–0.85 s, temp sweep 214.9–224.7, alt 271.5–281.3, MTP-over-no-spec 2.35–2.53× by face, per-face acceptance windows 0.57–0.61 sampled / 0.859 t0 / 0.91 alt; integrity 5/5 + det 3/3 every measured boot, golden 5/5, garbage 15/15, 0 ERROR/Traceback ×5; serve-default FLIP validated (flag probe 5/5 on the real script via PATH-stubbed docker; default-path boot engine bytes == record legs; greedy opt-out boot carries the pre-P6.17 bytes) — the every-dtype ≥ +10 % default-flip rule was overridden (recorded: k8v4 EV-neutral, fp8/tq4/k3v4/3bit all clear); no new image — cert6 remained certified through P6.17 (serve-layer default, wheel/engine byte-unchanged; pre-flip script preserved `serve_arc_b70_p616.sh`; the image line caught up at cert7/P6.18) |
| P6.18a/b repro + triage | `2026-11-03-p618b-repro/` | **the turboquant corruption campaign**: owner report (k8v4 corrupt returns) reproduced and split — **Class B** deterministic degenerate generation (REP-LOOP attractors) on ALL FOUR `turboquant_*` presets, cold-included, LMC-independent, cross-dtype byte-identical attractors; **Class A** t0 warm-resume trajectory flips (dtype+LMC-independent, settled as inherent warm-resume-vs-cold numerical divergence — diag/diag2 canaries; disposition documentation, not code); fp8 legs clean throughout; 14 MB tree with byte-exact redaction (+1872) + W14 restore incident record |
| P6.18c roundtest + fix | `2026-11-03-p618c-roundtest/` (+ `BEST-PRACTICES-AUDIT.md` in the cert7 tree) | **root cause closed + the fix**: the live SoA store→decode Triton pair is numerically correct on XPU (decode self-coherence rel_err ≤ 6e-6, store spec-conformant [V ≤ 1 tie-step, K value-exact], fully deterministic, all four presets with engine tables + model geometry) — the degeneracy is the 4-bit-class value-precision quality ceiling (16 affine levels per token-head, 8 on 3bit; upstream's own tables +2.71…+20.59 % PPL on gpt-oss) → fix = `ArcXpuPlatform.check_and_update_config` refuses any `cache_dtype` startswith `turboquant` (upstream-patterened: rocm's prefix-match fail-loud idiom, cuda's config-time ValueError+remedy); upstream-pattern audit `BEST-PRACTICES-AUDIT.md` |
| P6.18d cert7 gates | `2026-11-03-p618-cert7/` | **cert7 release gates** (tag-built image `a8cd765b0380`, single flow; tag moved once — the first build read the CLI flag name `kv_cache_dtype` instead of the pydantic field `cache_dtype` and was caught by G0): manifest fork_sha `5bf4192d` == tag, p48 28/28 + p73 11/11 image-native, **G0 guard proof** k8v4 + tq4nc (loud ValueError ~65 s, startup 0, health never), default fp8 posture (health ~295 s, SLICE_ENV_ABSENT=0, KV pool 400,739 == p618b cert6-fp8 legs, flip 9.2 s stop), run7n t0 299.73/84.50/84.44 in-family, integrity 5/5 + det 3/3 (×2), spec pos1 0.9258 PASS, 0 ERROR/Traceback, WARN class-set zero fork-added, **LONG-RETURN GATE 100 % CLEAN** (4 classes single-hash across cold/rehit/rehit2 + 3 churn cycles; zero REP-LOOP/JUNK/BLEED/EMPTY; t1-seed annex: one sporadic flip characterized 16/16 identical — Class-A RNG-path divergence at sampled temperature, not corruption) + **REPRODUCIBILITY-AUDIT.md** (image ⊆ repo 144/144 md5; every image input tracked or pinned) — **`rel/v0.30.0-b70-cert7` = certified release** — superseded by cert8 (P6.19, next row): the TQ corruption root-caused and fixed, the dtype refusal lifted |
| P6.19 TQ spec-verify fix + cert8 | `2026-11-04-p619-tqverify/` (+ `rig/` gate evidence) | **corruption root-caused and FIXED — all dtypes servable**: 3 root causes from the t2i telemetry campaign (verify-capture baked raw-KV fast-path graphs / replay-broken multi-row synthetic-decode dispatch / SoA-vs-AoS MQ addressing on the first draft read), fix = `patches/tq_spec_fix.py` default-ON (split dispatch certified; `VLLM_ARC_TQ_VERIFY_MODE=mq` opt-in experimental, agg8 +21 %, MQVAL5 4/4 exact); cert8 gates ALL GREEN on the tag-built image (`b212fbca59ec`): manifest == tag `e47739b0`, p48 28/28 + p73 11/11, fp8 byte-family cert7 (KV 400,739, t0 300.65/84.53-84.57, spec pos1 0.9258, integrity 5/5 + det 3/3), fp8 + all-TQ long-return zero garble on the t0 lanes (identity faces dispositioned: Class-A warm-resume variants, 3-bit precision floor, and the t1-seed REP-LOOP flag = a temp-1.0 model-lane property proven firing on the zero-fork W14 incumbent too), tps k8v4 54.42/54.59/235.97, mq 284.81; 0 ERROR/Traceback ×7 boots; best-practices audit banked — **`rel/v0.30.0-b70-cert8` = certified release** — superseded by cert9 (P6.20, next row): the prefill host-path store patch, strictly faster TTFT, byte-exact spec/identity |
| P6.20 prefill host-path + cert9 | `2026-11-04-p620-prefill-opt/` (+ `rig/` r80–r84b evidence) | **prefill host path eliminated as a bottleneck; U32k/U64k proven device-bound**: step telemetry first (`patches/step_trace.py` 9-segment per-step timeline, silent-off) → L1 evsync NEUTRAL (r80, in-order stream), **L3 off-thread store ADOPTED** (r81 interleave A/B/A/B: U32k −0.74 s / U64k −1.64 s / U8k −0.08 s; `arc-lmc-store` worker, device ops stay on the engine stream; default ON, `VLLM_ARC_LMC_ASYNC_STORE=0` kill switch), r82 battery GREEN + source-override instrument lesson (wrap the ENGINE's module), **L6 dispatch-late** (r83 FULL GREEN; spec byte-exact cert8 pos1 0.9258/acc 0.837), r83 window trace = **lane closure** (wfs waits = device-queue drain behind chunk forwards, worker bodies ≤108 ms, ~89 % of U64k TTFT device-side; L6a/L6b dispositioned NOT-built; remaining lever = kernel compute AR 30.4 %/FA 20.3 %); cert9 gates ALL GREEN on the tag-built image (`a9f8252451b6` @ `aa2ee17d`, 4/4 file md5 == tag tar): manifest + p48 ALL_GREEN + p73 11/11, **default fp8 with ZERO env overrides** (STORE_WORKER_ACTIVE=2, LMC_ENV_SET=0, trace silent-off), integrity 5/5 + det 3/3, t0 296.91/84.54-84.48 in-family, t1.0 240.06 steady, spec byte-exact cert8, fp8 long-return PASS zero t1-seed misses + WARN diff 0/0, **p65 image-native U32k 16.93 / U64k 38.93 / U8k 3.76 / C32k 1.09**, all-TQ zero garble (k8v4 tps 54.64/54.85/236.15; k3v4/q4nc identity faces annexed with **cross-cert dominant-hash proof** — variant families byte-identical to the cert8 gate boots; q3nc = its P6.14 record character), mq annex (cert8-precedent fixed sed), 0 ERROR/Traceback every boot — **`rel/v0.30.0-b70-cert9` certified; superseded by cert10 (next row) — fully valid, its engine bytes carry unchanged into cert10** |
| P6.21 device-compute lanes (ar_quant + FA) | `2026-11-05-p621-arquant/`, `2026-11-05-p621b-fa-headroom/`, `2026-11-05-p621c-fa-esimd-v2/` | **both lanes closed NEGATIVE, no engine change** — ar_quant donor port A/B: quality green, perf −4…−9 % on every U-bar, unadopted; FA headroom ladder r87–r91: stock vxk at 25 % of the measured 147 TF/s peak but every patch-level route falsified; ESIMD dense-FA v2 kernel: numerics-green at **0.33× stock**, ablation ceiling bounds best case ~0.83× (2× bar fails) — kernel laws banked in-source |
| P6.22 TP2 allreduce lane | `2026-11-05-p622-ar-lane/` (+ REDACTION note) | **closed NEGATIVE, no engine change at this phase, cert9 stayed current (no cert10 from P6.22 — nothing serve-affecting)** — census instrument `ar_trace.py` (silent-off) + r94 message-size census/trace (dominant AR 76,677,120 B = 7488×5120 fp16 ×1,584/worker = 88 % of family; in-situ 7.85 ms) + r95 ceiling bench: **stock oneCCL ring 7.78 ms = 2.06× faster than the achievable one-shot torch-primitive ceiling 16.03 ms** (ring 9.85 GB/rank/s concurrent vs copy engines 4.92 GB/s/dir); backend-name law banked (serve dist backend = `xccl`); rig-power-cut durability proven; the one-shot lane stayed closed — the quantized-wire route it left open was taken by P6.24 |
| P6.24 oneCCL fp8-wire allreduce + cert10 | `2026-11-05-p624-fp8-ar/` (r96–r100 evidence + VERDICT + REDACTION notes; build kit `external_patches/oneccl/` + `tools/build_oneccl_arc.sh` + `tools/ar/` runners) | **quantized-AR route deployed as a default-OFF image overlay; cert10 released** (`vllm-arc:v0.30.0-b70-cert10`, image `2f86fc2411fb`): r96 same-version oneCCL 2022.0.0 rebuild FIDELITY PASS (ABI/perf-identical measured — 7.782 vs 7.780 ms at the dominant size; canonical staging + pristine-tree laws banked); r97 `arc-fp8-allreduce.patch` v4 GATE PASS **1.65×** at 76,677,120 B (7.801→4.731 ms; FP8=0 control == anchor; numerics rel-rms 0.02285 == offline bit-exact floor; sector law: u32-packed peer stores, byte-granular remote stores ran 0.41 GB/s); r98 quality ladder task-level parity on one container one-env-line A/B (tps/spec/integrity/long-return green; min-count law live in serve — sub-2M-elem prefills never touch the wire); r99 cert10 release gate ALL GREEN (default-posture battery with FP8 env proven absent; k8v4 carry byte-identical to stock cert9; FP8 knob leg + in-situ STATS wire proof on the shipped image; A/B/A/B perf **beats cert9 on every p65 face**: U32k 16.29/C32k 1.04/U8k 3.27/U64k 35.73 s — the t1 REP-LOOP dispositioned boot-stochastic by the r99c annex); r100 bottleneck scan closed every remaining door with wire-rate measurement (writes 9.85 GB/s fabric ceiling flat in width; reads 2.2× slower; overlap collapses; v4 = 82 % of the wire-bound floor — final); the probe build is an env-gated diagnostic that never enters a cert image — **`rel/v0.30.0-b70-cert10` = current certified release** |
| P6.24b patch audit (2026-11-06) | `external_patches/oneccl/README.md` + `results/2026-11-05-p624-fp8-ar/VERDICT.md` (patch-audit section) | docs/tooling-only — the certified image is UNCHANGED (tag stays at the cert10 release commit): probe patch reconciled to as-built bytes (sha chain reproducible), res-scale invariant documented precisely + frozen-artifact law banked, build-script polish; no re-gate required |
| P6.24d README status-ledger move (2026-11-06) | `docs/STATUS-arc.md` (new detail holder) | docs-only — the certified image is UNCHANGED (tag stays at the cert10 release commit): the three README front-page narrative paragraphs (certified-release detail, project status, improvement-lane status) moved content-faithful to `docs/STATUS-arc.md` (lead-ins became the section headings, docs-internal links made relative — zero factual edits, diff-audited); README keeps the headline + links, currency anchors preserved (check_currency clean); no re-gate required |
| P6.24e best-performance-postures audit (2026-11-06: docs + serve-interface pass on cert10 truth) | CONFIGURATION "Best-performance postures" (new) + envs.md overlay table + `tools/serve_arc_b70.sh` | docs/tooling-only — the certified image is UNCHANGED (tag stays at the cert10 release commit; the FP8 env itself is the r96–r100-certified opt-in): consolidated objective→recipe table (best TTFT = the one-line `CCL_SYCL_ALLREDUCE_FP8=1` opt-in w/ exact placement, best throughput = Step 2 as-is, stall levers, KV-capacity trade); stale "Max TTFT = certified defaults" menu row fixed (cert10 opt-in beats it on every TTFT face); envs.md gains the oneCCL overlay-env table (trio previously pointer-only); uniform boot interface gains `ARC_FP8` pass-through (default unset emits NO env line — served-process env byte-identical to the pre-knob script; `=1` appends the certified opt-in env); README optional-postures + quick-start hint; no re-gate required (default-path posture unchanged) |
| env reference | `docs/envs.md` | complete control surface + naming map |
