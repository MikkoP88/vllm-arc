# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
law; identities re-pointed at each certification close-out — currently
**cert23 (P6.35, 2026-11-10: the graph-exec double-buffer on the cert22
engine; every prior cert's content, wheels, fork SHAs, and gate evidence
live in [`../CHANGELOG.md`](../CHANGELOG.md))**. Target: serve
**qwen3.8-27b-fp8** (dense hybrid-GDN, **MTP k3 recommended** — r46t
real-world campaign) on `vllm-arc:v0.30.0-b70-cert23`, replacing the W14
incumbent (`radiance-test`, flash-next k3) as the rig's serving stack.
Everything below is rig-proven; the phase-by-phase evidence table is §6.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert23` (cert22 engine + the P6.35 graph-exec double-buffer — `vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`, every XPU graph captured ×2 into the manager's shared pool, replays alternate execs; `VLLM_ARC_GRAPH_EXEC_DB=0` kill-switch; via `tools/cert23.Dockerfile` with build-time asserts + `IMAGE_MANIFEST` rows incl. `cert23_fork_sha`; gate evidence `results/2026-11-10-p635-telemetry/` — r135 five-leg gate + r135b control). Prior images (cert22 arc6 wheel, cert21 arc5, cert20 arc4, cert19/18 FA2 routes, cert17 chain, cert11 overlay, cert10 oneCCL overlay, …) with their wheel shas and Dockerfile asserts: [`../CHANGELOG.md`](../CHANGELOG.md) |
| fork SHA baked | cert23 content commit per the image's `IMAGE_MANIFEST` (`cert23_fork_sha`; verify: `docker run --rm --entrypoint cat vllm-arc:v0.30.0-b70-cert23 /etc/vllm-arc/IMAGE_MANIFEST`) == annotated tag `rel/v0.30.0-b70-cert23` on main; base rows carry from cert9/10 (`fork_sha=aa2ee17d…` + oneCCL overlay rows) with per-cert stamps `cert11=c987c2cd…`, `cert12=a964e047…`, `cert16=5d895fef…`, `cert17=e5e725b4…`, `cert18=3136adfe…`, `cert19=904a3de8…`, `cert20=a918bee9…`, `cert21=c8fdf32a…`, `cert22=e2457389…`; full per-cert SHA chain: [`../CHANGELOG.md`](../CHANGELOG.md) |
| validation image | `vllm-arc:v0.30.0-b70-cert23` itself — r135 five-leg gate (build asserts + `CERT23_DB_LIVE`, db markers per rank, A/B on/off faces at every posture: fp8-k3 94.5 on / 94.3 off, mx4k5 121.0 / 123.9 — E2E-neutral; replay max 24 → 0.37 ms; engine serial 40.8 → 12.7 ms/step) + r135b control (dboff2: zero db markers, stock face) + the P6.35-V 11-cell matrix (r136/r136b, both models, 0 tracebacks); evidence `results/2026-11-10-p635-telemetry/` (`cert23/` + `vmatrix/`). Prior gates (r127g2, r125, r121, r111, r108ae, r108z, r105b, r99…): [`../CHANGELOG.md`](../CHANGELOG.md) |
| canonical serve script | rig `/root/build/serve_arc_b70.sh` (the uniform boot interface; carries the **25-var chain** incl. `LMCACHE_TRACK_USAGE=false` — P6.3; run with `ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_GRAPH=1 ARC_SPEC=3`; lane knobs — **`ARC_KVDTYPE`** kv-cache dtype pass-through, default `fp8_e4m3` (P6.37: `fp8` ≈ incumbent; `auto` (fp16 KV) refuses at maxlen 262144 but serves at 131072; `turboquant_*` presets servable; `fp8_e5m2` hard-refused on Xe2); `ARC_SKINNY_M` (P6.33, unset/off = stock — certified default); `ARC_QUANT=mxfp8` + `ARC_MXFP8_MODE` + **`ARC_MAXLEN`** (P6.31 — gemma 8-bit weights need 40960); `ARC_QUANT=mxfp4` + `ARC_MXFP4_MODE` + `ARC_QUANTCFG` hybrid ignore list (P6.30); `ARC_ATTN_FA2` (P6.28/29 routes, `=0` = pure triton); gemma-4 lane: `ARC_MODEL`/`ARC_SERVED`/`ARC_TEMPLATE` + **`ARC_LMC=0`** + **`ARC_RPC_TIMEOUT=1200`** + **`ARC_SPEC=0`**; P6.24 opt-in `export CCL_SYCL_ALLREDUCE_FP8=1`; **since P6.17 the certified default is probabilistic drafting** — unset or `ARC_SPEC_METHOD=probabilistic` (same bytes), `greedy` = opt-out historical lane, `ARC_SPEC=0` = no spec; full per-knob semantics: `docs/envs.md` + CONFIGURATION-arc) |
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

**cert23 migration note (P6.35)**: for an existing cert22 deployment
there is **no action and no behavior change** — cert23 adds one
default-ON patch module (graph-exec double-buffer, E2E-neutral at every
measured posture, `VLLM_ARC_GRAPH_EXEC_DB=0` kill-switch) and +29 s boot
(the ×2 graph captures). Per-prior-cert migration notes (cert11, cert10,
…): [`../CHANGELOG.md`](../CHANGELOG.md).

## 2. Boot procedure (from W14-serving state)

```bash
# on the rig host (address lives in the untracked local layer, never in
#                  tracked files — publishing-hygiene law; every ssh:
#                  export SSH_ASKPASS=/tmp/askpass.sh
#                  SSH_ASKPASS_REQUIRE=force DISPLAY=:0)
docker stop radiance-test                      # frees port 8000 + both GPUs
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
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
construction. Boot takes ~5–6 min to health 200 on cert23 (the cert22
base booted 285–295 s; the cert23 ×2 graph captures add ~29 s). Verify
in `/root/serve_full.log`: wheel hooks install (zero `FAILED`, silent
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
   certified steady family at k3, prob lane: solo **83.7–84.9** / agg
   **291.4–303.1** (byte-stable across the cert8–cert11 release gates;
   per-gate values: CHANGELOG) — **comparator law (r47c)**:
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
  arc block.
- `VLLM_XPU_ENABLE_XPU_GRAPH=1` on all graph legs (set inside the serve
  script).
- Verify interleave/BW matrix after any rig reboot.
- Port 8000 is single-tenant: stop radiance-test before arc legs.
- `--mamba-cache-mode=align` is pinned in the serve line (posture-pin
  law: never ship its removal).

## 6. Evidence table (phase ledger)

One row per phase/leg-group; the "what it proved" details live in
PERFORMANCE-arc / TESTING-arc. All paths relative to `results/`.

| Phase | Evidence (under `results/`) | One-line verdict |
|---|---|---|
| P4 r42–r44 | `2026-10-30-p4-r42-certified-image/` … `-r44-matrix-certified/` | certified-image boot + battery; self-contained image (Fix A/B); 12-cell regression baseline |
| P5 r45/r46/r46t | `2026-10-30-p5-r45-env-log-audit/`, `-r46-soak/`, `-r46t-temps/` | env/log audit + this addendum; 8 h soak GREEN (gate of record); real-world campaign — **k3 recommended** |
| P6.1–P6.7 | `2026-10-30-p6-r47a-ext-hitrate/` … `2026-10-31-p67-r67a-compile-ab/` | external hit-rate artifact; lmcache 0.5.6+arc1 gates + comparator law; cert3 process; p65 stall capture + cert4; prefill anatomy (device-bound); compile REVAL rejected |
| P6.10 / P2P | `2026-10-31-p70-r70-cert4-comprehensive/`, `-p71-r71-p2p-analysis/` | cert4-native comprehensive re-measurement + 4 h soak; TP2 transport proven clean (direct P2P 10.23 GB/s) |
| P6.13 | `2026-11-01-p65-end-stall-analysis/`, `-p65-rung3-slice/`, `-p65-cert5/` | END-stall analysis; sliced final-span store default-OFF; **cert5** |
| P6.14/P6.15 | `2026-11-02-p614-*`, `2026-11-02-p615-*`, `2026-11-03-p615c-*` | **cert6** TQ presets + store fix; bystander-stall taxonomy; t1.0-xhigh collapse mechanism pinned |
| P6.16/P6.17 | `2026-11-03-p616a-c*/`, `-p617a-probcert/` | probabilistic drafting measured then certified as the serve default; every record face re-measured |
| P6.18 (cert7) | `2026-11-03-p618b-repro/`, `-p618c-roundtest/`, `-p618-cert7/` | TQ corruption reproduced/split; interim dtype refusal certified; reproducibility audit |
| P6.19 (cert8) | `2026-11-04-p619-tqverify/` | **TQ fixed** — three spec-verify defects; all-dtype long-return gate zero garble |
| P6.20 (cert9) | `2026-11-04-p620-prefill-opt/` | L3 off-thread store default-ON + L6; prefill proven device-bound |
| P6.21/P6.22 | `2026-11-05-p621*/`, `-p622-ar-lane/` | device-compute residue closed NEGATIVE (ar_quant, FA ladder, ESIMD v2, one-shot AR); AR census instrument banked |
| P6.24 (cert10) | `2026-11-05-p624-fp8-ar/` | oneCCL fp8-wire overlay certified default-OFF (1.65× dominant AR; every p65 TTFT face beats cert9 opt-in); + b/d/e audits |
| P6.25 | `2026-11-06-p625-overlap-door/` | cross-stream AR∥GEMM closed MEASURED — the in-scope improvement ledger is CLOSED |
| P6.26 (cert11) | `2026-11-06-p626-mxfp4/` | quark/MXFP4 lane certified; fp8 default byte-identical (same-day discriminator) |
| P6.27 | `2026-11-06-p627-mxfp8/` | MXFP8 AutoRound checkpoint validated zero-change (no cert) |
| P6.28 (cert12/16/17/18) | `2026-11-06-p628-gemma4/` | gemma-4 chain: G2 freeze closed 3 layers; hd512 FA2 route 34× — faces 2.2–16.3× faster |
| P6.29 (cert19) | `2026-11-07-p629/` | sliding-window FA2 route; gemma decode +45 %; spec parity closed for gemma |
| P6.30 (cert20) | `2026-11-08-p630-mxfp4-perf/` | online mxfp4 W4A16 route — superior online lane on every axis |
| P6.31 (cert21) | `2026-11-08-p631-mxfp8-w8a16/` | online mxfp8 W8A16 route 1.37–1.85×; W4A8 closed-negative; gemma 8-bit capacity law |
| P6.32 | (analysis, in-tree notes) | Xe2 FP8-nativeity verdict + kernel-feasibility map |
| P6.33 (cert22) | `2026-11-08-p633-skinny-m/` | skinny-M kernels correct but falsified e2e under graph replay — banked OFF, serve ≡ cert21 |
| P6.34 | `2026-11-09-p634-jm-ladder/` | joint-matrix ladder laws; research lane closed |
| P6.35 (cert23) | `2026-11-10-p635-telemetry/` | graph-exec double-buffer certified (E2E-neutral; posture gains: mx4+k5 123.9 tok/s) |
| P6.35-V | `2026-11-10-p635-telemetry/vmatrix/` | 11-cell matrix green; **project best 128.3 tok/s** (mx4 k5 re-read) |
| P6.36 | `2026-11-10-p636-legA/` | prefill = device drain; no remaining lawful lever ≥ 10 % — close-out triggered |
| P6.37 | `2026-11-10-p637-reports/` | per-config reports + 64 GB-RAM defaults + `ARC_KVDTYPE` knob; no engine bytes changed |

| env reference | `docs/envs.md` | complete control surface + naming map |
