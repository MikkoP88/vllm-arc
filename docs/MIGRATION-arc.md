# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
directive; identities re-pointed at each certification close-out — most
recently cert6 (P6.14, 2026-11-02; same default posture as cert5, adds
the TurboQuant-aware lagged-store hunk + validated `turboquant_*` dtype
faces). Target: serve **qwen3.8-27b-fp8**
(dense hybrid-GDN, **MTP k3 recommended** — r46t real-world campaign) on
`vllm-arc:v0.30.0-b70-cert6`, replacing the W14 incumbent
(`radiance-test`, flash-next k3) as the rig's serving stack. Everything
below is rig-proven; the full phase-by-phase evidence table is §6.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert6` (tag-built from `rel/v0.30.0-b70-cert6` — cert6; default posture byte-identical to cert5, adds the P6.14 TQ-aware lagged-store hunk + the validated opt-in `turboquant_*` dtype faces) |
| fork SHA baked | `986e97d6` (IMAGE_MANIFEST `fork_sha=986e97d6ac76df138347daa0bab2dc1546ea36e8` == the annotated tag's commit; annotated tag `rel/v0.30.0-b70-cert6` on main; image-native adapter md5 `33812d19…` == repo) |
| validation image | `vllm-arc:v0.30.0-b70-cert6` itself (28/28 p48_verify + 11/11 p73_verify image-native, default-posture battery + slice face + TQ4 face — `results/2026-11-02-p614-cert6/`; candidate-image legs `results/2026-11-02-p614-{l1v2,l2,l3}/`; default faces first validated cert4-native `results/2026-10-31-p65-cert4/` + r70 comprehensive `results/2026-10-31-p70-r70-cert4-comprehensive/`) |
| canonical serve script | rig `/root/build/serve_arc_b70.sh` (the uniform boot interface; carries the **25-var chain** incl. `LMCACHE_TRACK_USAGE=false` — P6.3; run with `ARC_IMAGE=vllm-arc:v0.30.0-b70-cert6 ARC_GRAPH=1 ARC_SPEC=3`) |
| long-context profile | rig `/root/build/serve_arc_b70_defer2.sh` (the canonical chain + `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` — P6.5; opt-in, see CONFIGURATION-arc "Long-context serve profile") |
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
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert6 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash /root/build/serve_arc_b70.sh            # boot + serve as one unit
# logs land in the container at /root/serve_full.log (host: docker exec)
# poll health (first-traffic law: batteries run on FIRST traffic after boot)
curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health
```

The script boots the certified container (`--privileged` REQUIRED —
pidfd IPC law, r30 defect #1; `--network host --ipc host`, the model
mount, cache mounts, and the 2 boot-layer `-e` vars) and execs the
serve line with the full 25-var chain — env + CLI as one unit by
construction. Boot takes ~5–10 min to health 200. Verify in
`/root/serve_full.log`: wheel hooks install (zero `FAILED`, silent
defer is DEBUG), `fp8 reroute ACTIVE`, `kvfmt TWO_HS`, the p40c re-pin
line both workers (`alloc/free_pinned_ptr re-pinned to the pageable
torch baseline`), `glue-unavailable 0`, `[p41c] wrap live`, posture
lines (mode=none, FULL_DECODE_ONLY [1..320], MTP k3, align).

## 3. Post-boot verification (any migration or image bump)

§10 continuous row, in order (harness tools unchanged, host-side python3):

1. **survival flip-first**: 832-token flip payload as FIRST traffic → 200.
2. **spec gate** (`arc_spec_gate.py`): pos1 ≥ 0.85 target (0.5 hard).
3. **golden** (`p25/p32_golden.py`): 5/5 shas byte-identical to
   `golden_ref.json` (0-drift law).
4. **run7n ×2**: conservative floors solo ≥ 79.74, agg#2 ≥ 274.41;
   certified steady family at k3 (cert4-native, 5 boots): solo
   **83.7–84.9** / agg **294.7–300.5** — **comparator law (r47c)**:
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
| P6.14 cert6 | `2026-11-02-p614-cert6/` (+ candidate legs `2026-11-02-p614-{l1v2,l2,l3}/`) | **cert6 release gates** (tag-built image, single flow): manifest fork_sha `986e97d6` == tag, p48 28/28 + p73 slice-hunk 11/11 image-native, SLICE_ENV_ABSENT=0, run7n t0 296.25/84.46 + t1.0 186.48/61.79 in-family, integrity 5/5 + det 3/3, WARN class-set identical to cert5 (44 == 44), slice face on the SAME image (3328×38 exact, 63232-class 0, W_U64k_start max 1604) + the first certified opt-in TQ face `turboquant_4bit_nc` on the same image (0 errors, kit D2H live @262, spec 0.9714/0.9381, KV 757,641 = 2.89×, U32k/U64k TTFT 15.85/33.98 s) — **`rel/v0.30.0-b70-cert6` = current certified release** |
| env reference | `docs/envs.md` | complete control surface + naming map |
