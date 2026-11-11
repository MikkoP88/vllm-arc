# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
law; identities re-pointed at each certification close-out — currently
**cert23 (2026-11-10, the graph-exec double-buffer on the cert22
engine)**. Every prior cert's content, wheels, fork SHAs, gate
evidence, and migration notes live in
[`../CHANGELOG.md`](../CHANGELOG.md). Target: serve
**qwen3.8-27b-fp8** (dense hybrid-GDN, **MTP k3 recommended** — the
real-world campaign of record) on `vllm-arc:v0.30.0-b70-cert23`,
replacing the W14 incumbent (`radiance-test`, flash-next k3) as the
rig's serving stack. Everything below is rig-proven.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert23` (cert22 engine + the graph-exec double-buffer — `vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`, every XPU graph captured ×2 into the manager's shared pool, replays alternate execs; `VLLM_ARC_GRAPH_EXEC_DB=0` kill-switch; built via `tools/cert23.Dockerfile` with build-time asserts + `IMAGE_MANIFEST` rows incl. `cert23_fork_sha`; gate evidence `results/2026-11-10-p635-telemetry/`). Prior images, wheel shas, and Dockerfile asserts: [`../CHANGELOG.md`](../CHANGELOG.md) |
| fork SHA baked | cert23 content commit per the image's `IMAGE_MANIFEST` (`cert23_fork_sha`; verify: `docker run --rm --entrypoint cat vllm-arc:v0.30.0-b70-cert23 /etc/vllm-arc/IMAGE_MANIFEST`) == annotated tag `rel/v0.30.0-b70-cert23` on main. Full per-cert SHA chain: [`../CHANGELOG.md`](../CHANGELOG.md) |
| validation image | `vllm-arc:v0.30.0-b70-cert23` itself — five-leg gate (build asserts + `CERT23_DB_LIVE`, db markers per rank, A/B on/off faces at every posture: fp8-k3 94.5 on / 94.3 off, mx4k5 121.0 / 123.9 — E2E-neutral; replay max 24 → 0.37 ms; engine serial 40.8 → 12.7 ms/step) + control boot (dboff2: zero db markers, stock face) + the 11-cell validation matrix (both models, 0 tracebacks); evidence `results/2026-11-10-p635-telemetry/` (`cert23/` + `vmatrix/`) |
| canonical serve script | rig `/root/build/serve_arc_b70.sh` (the uniform boot interface; carries the **25-var chain** incl. `LMCACHE_TRACK_USAGE=false`; run with `ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_GRAPH=1 ARC_SPEC=3`; lane knobs — **`ARC_KVDTYPE`** kv-cache dtype pass-through, default `fp8_e4m3` (`fp8` ≈ incumbent; `auto` (fp16 KV) refuses at maxlen 262144 but serves at 131072; `turboquant_*` presets servable; `fp8_e5m2` hard-refused on Xe2); `ARC_SKINNY_M` (unset/off = stock — certified default); `ARC_QUANT=mxfp8` + `ARC_MXFP8_MODE` + **`ARC_MAXLEN`** (gemma 8-bit weights need 40960); `ARC_QUANT=mxfp4` + `ARC_MXFP4_MODE` + `ARC_QUANTCFG` hybrid ignore list; `ARC_ATTN_FA2` (`=0` = pure triton); gemma-4 lane: `ARC_MODEL`/`ARC_SERVED`/`ARC_TEMPLATE` + **`ARC_LMC=0`** + **`ARC_RPC_TIMEOUT=1200`** + **`ARC_SPEC=0`**; opt-in `export CCL_SYCL_ALLREDUCE_FP8=1`; **the certified default is probabilistic drafting** — unset or `ARC_SPEC_METHOD=probabilistic` (same bytes), `greedy` = opt-out, `ARC_SPEC=0` = no spec; full per-knob semantics: `docs/envs.md` + CONFIGURATION-arc) |
| long-context profile | rig `/root/build/serve_arc_b70_defer2.sh` (the canonical chain + `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` — opt-in, see CONFIGURATION-arc "fp8 stall levers") |
| A/B fallback | same script with `ARC_SPEC=4` (MTP k4 — documented alternative for near-deterministic workloads) |
| golden reference | rig `/root/build/p25/golden_ref.json` (md5 `e02cbb07…`; covers k4 AND k5) |
| regression matrix | rig `/root/build/matrix_arc/leg-2026-10-29-r44-certified/` (12 cells) |
| container name | `arc-test` |

The image is **self-contained**: no cap `PYTHONPATH` overlay, no runtime pip
installs (lmcache 0.5.6+arc1 closure vendored; in-build import gate +
`lmcache_closure=` manifest line). Wheel-module patches load via
the in-image `_WHEEL_HOOKS` registry (`_load_wheel_patch`
standalone loader).

**cert23 migration note**: for an existing cert22 deployment
there is **no action and no behavior change** — cert23 adds one
default-ON patch module (graph-exec double-buffer, E2E-neutral at every
measured posture, `VLLM_ARC_GRAPH_EXEC_DB=0` kill-switch) and +29 s boot
(the ×2 graph captures). Per-prior-cert migration notes:
[`../CHANGELOG.md`](../CHANGELOG.md).

## 2. Boot procedure (from W14-serving state)

```bash
# on the rig host (address lives in the untracked local layer, never in
#                  tracked files — publishing-hygiene law; every ssh:
#                  export SSH_ASKPASS=/tmp/askpass.sh
#                  SSH_ASKPASS_REQUIRE=force DISPLAY=:0)
docker stop radiance-test                      # frees port 8000 + both GPUs
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash /root/build/serve_arc_b70.sh            # boot + serve as one unit
# draft posture: unset/probabilistic = CERTIFIED; the certified
# line can be written explicitly as ARC_SPEC_METHOD=probabilistic (same
# bytes); ARC_SPEC_METHOD=greedy = opt-out; ARC_SPEC=0 = no spec
# logs land in the container at /root/serve_full.log (host: docker exec)
# poll health (first-traffic law: batteries run on FIRST traffic after boot)
curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health
```

The script boots the certified container (`--privileged` REQUIRED —
pidfd IPC law; `--network host --ipc host`, the model
mount, cache mounts, and the 2 boot-layer `-e` vars) and execs the
serve line with the full 25-var chain — env + CLI as one unit by
construction. Boot takes ~5–6 min to health 200 on cert23 (the cert22
base booted 285–295 s; the cert23 ×2 graph captures add ~29 s). Verify
in `/root/serve_full.log`: wheel hooks install (zero `FAILED`, silent
defer is DEBUG), `fp8 reroute ACTIVE`, `kvfmt TWO_HS`, the re-pin
line both workers (`alloc/free_pinned_ptr re-pinned to the pageable
torch baseline`), `glue-unavailable 0`, `[p41c] wrap live`, posture
lines (mode=none, FULL_DECODE_ONLY [1..320], MTP k3, align), and the
store-worker line after the FIRST request
(`LMC store worker ACTIVE` — the off-thread store, default ON).

## 3. Post-boot verification (any migration or image bump)

The continuous-row battery, in order (harness tools unchanged,
host-side python3):

1. **survival flip-first**: 832-token flip payload as FIRST traffic → 200.
2. **spec gate** (`arc_spec_gate.py`): pos1 ≥ 0.85 target (0.5 hard).
3. **golden** (`p25/p32_golden.py`): 5/5 shas byte-identical to
   `golden_ref.json` (0-drift law).
4. **run7n ×2**: conservative floors solo ≥ 79.74, agg#2 ≥ 274.41;
   certified steady family at k3, prob lane: solo **83.7–84.9** / agg
   **291.4–303.1** — **comparator law**: a fresh-boot FIRST battery on
   the same stack reads lower (~206/64.5 warmup face); gate a fresh
   boot against a same-posture A/B, not the campaign table.
5. **matrix 12 cells** (`r44_matrix.sh`): no face −5% vs
   `leg-2026-10-29-r44-certified` (regression block).
6. **integrity/determinism** (`r42/r42_integrity.py`): 5/5 + 3/3.

Any block → do not migrate; capture per E-law, fix on a branch, re-certify.

## 4. Rollback (arc → W14)

```bash
docker stop arc-test                            # artifacts stay in the container fs
docker start radiance-test
nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore.log 2>&1 &
# poll health: 200 within ~5–10 min (first polls may be 000 — model load)
```

Keep `arc-test` (Exited) on the rig — it carries the leg's serve logs
(`/root/serve_full.log`). Delete only after the leg's artifacts are banked
to `results/`.

## 5. Rig laws that govern the migration window

- NEVER leave the rig on an arc image overnight; W14 restores after every
  arc block.
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` on all graph legs (set inside the serve
  script).
- Verify interleave/BW matrix after any rig reboot.
- Port 8000 is single-tenant: stop radiance-test before arc legs.
- `--mamba-cache-mode=align` is pinned in the serve line (posture-pin
  law: never ship its removal).

## 6. Evidence

Every phase's evidence bank (`results/` paths, one-line verdicts) is
archived in [`history/MIGRATION_HISTORY.md`](history/MIGRATION_HISTORY.md);
per-cert/per-phase changelog entries:
[`../CHANGELOG.md`](../CHANGELOG.md). The complete control surface
incl. the naming map: [`envs.md`](envs.md).
