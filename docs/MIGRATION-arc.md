# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
directive; identities re-pointed at the certified release image at P6
close-out (2026-10-31, cert3) and again at P6.5 close-out (same day,
cert4 — same default posture, adds the opt-in long-context profile).
Target: serve **qwen3.8-27b-fp8** (dense
hybrid-GDN, **MTP k3 recommended** — r46t real-world campaign; k5
dropped) on `vllm-arc:v0.30.0-b70-cert4`, replacing the W14 incumbent
(`radiance-test`, flash-next k3) as the rig's serving stack. Everything
below is rig-proven (r42–r48l); evidence links at the end.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert4` (tag-built from `rel/v0.30.0-b70-cert4` — cert4; default posture byte-identical to cert3, adds the P6.5 opt-in long-context profile) |
| fork SHA baked | `b5df28b5` (IMAGE_MANIFEST `fork_sha=` == the annotated tag's commit; annotated tag `rel/v0.30.0-b70-cert4` on main) |
| validation image | `vllm-arc:v0.30.0-b70-cert4` itself (28/28 p48_verify + 10/10 p65_verify image-native, certified default boot + run7n t0/t1.0 + zero fork-added WARN/ERROR + p65 face U32k/U64k — `results/2026-10-31-p65-cert4/`) |
| canonical serve script | rig `/root/build/serve_arc_b70.sh` (the uniform boot interface; carries the **25-var chain** incl. `LMCACHE_TRACK_USAGE=false` — P6.3; run with `ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_GRAPH=1 ARC_SPEC=3`) |
| long-context profile | rig `/root/build/serve_arc_b70_defer2.sh` (the canonical 25-var chain + `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` — P6.5; opt-in, see CONFIGURATION-arc "Long-context serve profile") |
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
# on rig 10.20.3.65 (every ssh: export SSH_ASKPASS=/tmp/askpass.sh
#                        SSH_ASKPASS_REQUIRE=force DISPLAY=:0)
docker stop radiance-test                      # frees port 8000 + both GPUs
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
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
4. **run7n ×2**: solo ≥ 79.74, agg#2 ≥ 274.41 (bars; certified class
   86.4–87.1 / 303.5–305.4 at k5) — **comparator law (r47c)**: these
   bars are WARM-FLEET numbers; a fresh-boot first battery on the SAME
   certified stack reads ~206/64.5. Gate a fresh boot against a
   same-posture A/B, not the campaign table.
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

## 6. Evidence

- r42 certified-image boot + battery: `results/2026-10-30-p4-r42-certified-image/`
- r43 k5 adoption (canonical posture): `results/2026-10-30-p4-r43-k5-adoption/`
- r44 12-cell matrix (regression baseline): `results/2026-10-30-p4-r44-matrix-certified/`
- r45 env/log audit + this addendum: `results/2026-10-30-p5-r45-env-log-audit/`
- r46 8 h soak + mypy parity + 64k face (P5 gate of record): `results/2026-10-30-p5-r46-soak/`
- r46t real-world campaign (k3 recommendation, k5 dropped, garbage suite): `results/2026-10-30-p5-r46t-temps/`
- r47a P6.1 external-hit-rate root cause (the 0.0 % metric is a steady-state artifact, not a broken cache): `results/2026-10-30-p6-r47a-ext-hitrate/`
- r47c P6.2+P6.3 gates on p48 (lmcache 0.5.6+arc1, zero fork-added WARN/ERROR, same-posture A/B = the comparator law): `results/2026-10-31-p6-r47c-p48-056gates/`
- r47e cert3 tag-built image (manifest A/B + image-native 28/28): `results/2026-10-31-p6-r47e-cert3/`
- r48a–r48l P6.5 stall attribution + length-gated chunk-store deferral (the wrong-file discovery, the attribution matrix, the fix legs): `results/2026-10-31-p65-r48a-stall-capture/`
- cert4 gates (P6.5 close-out — candidate battery + p65 face on the shipped image): `results/2026-10-31-p65-cert4/`
- p66 P6.6 prefill anatomy (r66a solo curve + device probe, r66b step
  decomposition — prefill device-bound ~34–40 % MFU, host glue closed as
  empty, the engine-map correction, r66c MNBT-16384 A/B — knob documented
  not adopted; **analysis only, zero engine-code change: cert4 remains
  the certified image, no cert5**):
  `results/2026-10-31-p66-r66a-prefill-capture/`,
  `results/2026-10-31-p66-r66b-step-decomposition/`,
  `results/2026-10-31-p66-r66c-mnbt16k/`
- p67 P6.7 prefill structural rungs (r67a compile REVAL — Appendix H law
  #1 acceptance gate FIRED on 0.30: acceptance 3.4→2.2, decode −39 %,
  prefill flat, mode stays NONE; plus the GDN capture-cap config law
  `max_num_seqs×(k+1)`=256 under FULL_AND_PIECEWISE; r67b op-class XPU
  trace — GEMM 39.7 % IS fp8_gemm_w8a16, allreduce ~30.4 %, dense FA
  varlen 20.3 %, GDN 4.7 %, sparse machinery 0.1 %; rung C closed
  negative on r40 evidence, rung B re-scoped to the dense varlen site as
  a kernel project; **again zero engine-code change: cert4 remains the
  certified image, no cert5**):
  `results/2026-10-31-p67-r67a-compile-ab/`
- env reference: `docs/envs.md`
