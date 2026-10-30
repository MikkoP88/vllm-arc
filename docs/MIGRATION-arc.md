# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
directive; identities re-pointed at the certified release image. Target:
serve **qwen3.8-27b-fp8** (dense hybrid-GDN, **MTP k3 recommended** —
r46t real-world campaign; k5 dropped) on
`vllm-arc:v0.30.0-b70-cert`, replacing the W14 incumbent
(`radiance-test`, flash-next k3) as the rig's serving stack. Everything
below is rig-proven (r42–r46t); evidence links at the end.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert` (id `sha256:1a5fed5e…`; content-identical to validated p46 by tree-exactness) |
| fork SHA baked | `rel-fb273adc` (IMAGE_MANIFEST `fork_sha=`; annotated tag `rel/v0.30.0-b70-cert` on main) |
| validation image | `vllm-arc:v0.30.0-b70-p46` (the image the r46 soak + r46t campaign ran on) |
| canonical serve script | rig `/root/build/r42/r43_serve_k5.sh` with `"num_speculative_tokens":5` → `3` (MTP k3 recommended, r46t; the depth is serve-time CLI only) |
| A/B fallback script | rig `/root/build/r42/r42_serve.sh` (MTP k4 — documented alternative for near-deterministic workloads) |
| golden reference | rig `/root/build/p25/golden_ref.json` (md5 `e02cbb07…`; covers k4 AND k5 — r42/r43 proved byte-identical) |
| regression matrix | rig `/root/build/matrix_arc/leg-2026-10-29-r44-certified/` (12 cells) |
| container name | `arc-test` |

The image is **self-contained**: no cap `PYTHONPATH` overlay, no runtime pip
installs (lmcache 0.5.5+arc1 closure vendored; in-build import gate +
`lmcache_closure=` manifest line — r42 Fix A). Wheel-module patches load via
the in-image `_WHEEL_HOOKS` registry (r45 wiring + `_load_wheel_patch`
standalone loader — r42 Fix B).

## 2. Boot procedure (from W14-serving state)

```bash
# on rig 10.20.3.65 (every ssh: export SSH_ASKPASS=/tmp/askpass.sh
#                        SSH_ASKPASS_REQUIRE=force DISPLAY=:0)
docker stop radiance-test                      # frees port 8000 + both GPUs
docker rm -f arc-test 2>/dev/null
docker run -d --name arc-test --privileged --network host --ipc host \
  --shm-size 64g \
  -v /models/qwen3.8-27b-fp8:/models/target \
  -v /root/build/v127_stage/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja \
  -v /root/build/neo_cache:/root/.neo_cache \
  -v /root/build/triton_cache:/root/.triton_cache \
  -v /root/build/inductor_cache:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint bash vllm-arc:v0.30.0-b70-p45b -c "sleep infinity"
docker cp /root/build/r42/r43_serve_k5.sh arc-test:/root/
docker exec -d arc-test bash /root/r43_serve_k5.sh   # logs -> /root/serve_full.log
# poll health (first-traffic law: batteries run on FIRST traffic after boot)
curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health
```

Boot takes ~10 min to health 200. Verify in `/root/serve_full.log`:
7/7 wheel hooks install (zero `FAILED`), `fp8 reroute ACTIVE`,
`kvfmt TWO_HS`, posture lines (mode=none, FULL_DECODE_ONLY [1..320], MTP k5,
align). `--privileged` is REQUIRED (pidfd IPC law — r30 defect #1).

## 3. Post-boot verification (any migration or image bump)

§10 continuous row, in order (harness tools unchanged, host-side python3):

1. **survival flip-first**: 832-token flip payload as FIRST traffic → 200.
2. **spec gate** (`arc_spec_gate.py`): pos1 ≥ 0.85 target (0.5 hard).
3. **golden** (`p25/p32_golden.py`): 5/5 shas byte-identical to
   `golden_ref.json` (0-drift law).
4. **run7n ×2**: solo ≥ 79.74, agg#2 ≥ 274.41 (bars; certified class
   86.4–87.1 / 303.5–305.4 at k5).
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
- `--mamba-cache-mode=align` is pinned in the serve line (owner directive:
  never ship its removal; App F posture-pin law).

## 6. Evidence

- r42 certified-image boot + battery: `results/2026-10-30-p4-r42-certified-image/`
- r43 k5 adoption (canonical posture): `results/2026-10-30-p4-r43-k5-adoption/`
- r44 12-cell matrix (regression baseline): `results/2026-10-30-p4-r44-matrix-certified/`
- r45 env/log audit + this addendum: `results/2026-10-30-p5-r45-env-log-audit/`
- r46 8 h soak + mypy parity + 64k face (P5 gate of record): `results/2026-10-30-p5-r46-soak/`
- r46t real-world campaign (k3 recommendation, k5 dropped, garbage suite): `results/2026-10-30-p5-r46t-temps/`
- env reference: `docs/envs.md`
