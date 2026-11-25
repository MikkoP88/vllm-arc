# MIGRATION — W14 incumbent → vllm-arc certified stack (dual Arc Pro B70, TP2)

Migration addendum banked at P5 close-out (2026-10-30) per the banking
law; identities re-pointed at each certification close-out — currently
**cert28 (2026-11-24, cert27 content + the r231d seqused-tightened
split clamp wheel)**. Every prior cert's content, wheels, fork SHAs,
gate evidence, and migration notes live in
[`../CHANGELOG.md`](../CHANGELOG.md). Target: serve
**qwen3.8-27b-fp8** (dense hybrid-GDN, **MTP k3 recommended** — the
real-world campaign of record) on `vllm-arc:v0.30.0-b70-cert28`,
replacing the W14 incumbent (`radiance-test`, flash-next k3) as the
rig's serving stack. Everything below is rig-proven.

## 1. Identities of record

| item | value |
|---|---|
| image | `vllm-arc:v0.30.0-b70-cert28` (cert27 + the r231d seqused-tightened split clamp wheel — arc7mx build28 `vllm_xpu_kernels-0.1.14.1+arc7mx` md5 `e6d8d337e43b907504537061414128f8` BAKED over the cert27 wheel, installed `.so` md5s `61d2f5a365f05084b3dd72c4ae1da3e0`/`6e6c48508d40cb4c5007f27f79df97cf`/`190e261edc79b3b497d8d8e801e65757` asserted in-image; kernel delta: `seq_num_kv_splits = min(num_kv_splits, max(1, windowed_k_blocks/4))` legacy scalar path, mirrored `mx_paged_decode_kernel.hpp` + `mx_reduce_split.hpp`; overlay = cert27 overlay inherited byte-identical; certified native posture on all three MX lanes; MX lanes LMCache-free by Phase-6 auto-off, ctrl LMC-on; built via the cert28 bake (`tools/ar/r232_build_cert28.sh` — inline Dockerfile on the cert27 base) with build-time asserts + `IMAGE_MANIFEST` rows incl. `cert28_fork_sha`; gate evidence `results/2026-11-24-p647-cert28-battery/` + `results/2026-11-24-p647-r234-cert28-itl/`). Prior images, wheel shas, and Dockerfile asserts: [`../CHANGELOG.md`](../CHANGELOG.md) |
| fork SHA baked | cert28 build commit per the image's `IMAGE_MANIFEST` (`cert28_fork_sha=43f84cf977f9444900ff21387715ca5865db4c1a` = the r231d evidence merge, PR #123 — the commit the image was built and gated from; verify: `docker run --rm --entrypoint cat vllm-arc:v0.30.0-b70-cert28 /etc/vllm-arc/IMAGE_MANIFEST`). The annotated tag `rel/v0.30.0-b70-cert28` points at the cert28 close-out merge (docs + banks only, zero engine bytes — both SHAs documented in the CHANGELOG entry). Full per-cert SHA chain: [`../CHANGELOG.md`](../CHANGELOG.md) |
| validation image | `vllm-arc:v0.30.0-b70-cert28` itself — r233 full §10 battery, 8 legs + gemma compare (qwen absolute gates PASS: m8q/m4q/mkq — LONGRETURN + T1_SEED + garbage + erraudit clean, native route live at 108 capture-telemetry lines each; ctrl default-off + LMC-on postures intact; gemma comparative PASS, KNOWN-CLASS churn only), gated chain end-to-end (`R232C_LAUNCH_START 23:39:39 → R232C_LAUNCH_DONE 05:10:16` 2026-11-24); r234 record re-discrimination on the same image (every anchored face within +0.1%, first shallow rows m8q/mkq). Offline replay lineage (cert27 wheel, fixes carried through byte-identical): r225t 12/12 + r226a 8/8 (mxk8v4) + r229 mxfp8 8/8 + mxfp4 8/8; evidence `results/2026-11-24-p647-cert28-battery/` + `results/2026-11-24-p647-r234-cert28-itl/` |
| canonical serve script | rig `/root/build/serve_arc_b70.sh` (the uniform boot interface; carries the **25-var chain** incl. `LMCACHE_TRACK_USAGE=false`; run with `ARC_IMAGE=vllm-arc:v0.30.0-b70-cert28 ARC_GRAPH=1 ARC_SPEC=3`; lane knobs — **`ARC_KVDTYPE`** kv-cache dtype pass-through, default `fp8_e4m3` (`fp8` ≈ incumbent; `auto` (fp16 KV) refuses at maxlen 262144 but serves at 131072; `turboquant_*` presets servable; `fp8_e5m2` hard-refused on Xe2); `ARC_SKINNY_M` (unset/off = stock — certified default); `ARC_QUANT=mxfp8` + `ARC_MXFP8_MODE` + **`ARC_MAXLEN`** (gemma 8-bit weights need 40960); `ARC_QUANT=mxfp4` + `ARC_MXFP4_MODE` + `ARC_QUANTCFG` hybrid ignore list; `ARC_ATTN_FA2` (`=0` = pure triton); gemma-4 lane: `ARC_MODEL`/`ARC_SERVED`/`ARC_TEMPLATE` + **`ARC_LMC=0`** + **`ARC_RPC_TIMEOUT=1200`** + **`ARC_SPEC=0`**; opt-in `export CCL_SYCL_ALLREDUCE_FP8=1`; **the certified default is probabilistic drafting** — unset or `ARC_SPEC_METHOD=probabilistic` (same bytes), `greedy` = opt-out, `ARC_SPEC=0` = no spec; full per-knob semantics: `docs/envs.md` + CONFIGURATION-arc) |
| long-context profile | rig `/root/build/serve_arc_b70_defer2.sh` (the canonical chain + `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` — opt-in, see CONFIGURATION-arc "fp8 stall levers") |
| A/B fallback | same script with `ARC_SPEC=4` (MTP k4 — documented alternative for near-deterministic workloads) |
| golden reference | rig `/root/build/p25/golden_ref.json` (md5 `e02cbb07…`) |
| regression matrix | rig `/root/build/matrix_arc/leg-2026-10-29-r44-certified/` (12 cells) |
| container name | `arc-test` |

The image is **self-contained**: no cap `PYTHONPATH` overlay, no runtime pip
installs (lmcache 0.5.6+arc1 closure vendored; in-build import gate +
`lmcache_closure=` manifest line). Wheel-module patches load via
the in-image `_WHEEL_HOOKS` registry (`_load_wheel_patch`
standalone loader).

**cert28 migration note**: for an existing cert27 deployment there is
**no action beyond re-pulling the image** — the clamp lives inside the
arc7mx build28 wheel (no new env vars, no serve-line changes), every
certified posture is unchanged, and decode is certified-equivalent or
better on every lane (r234: mxfp4 2K ITL −7.6%, long faces parity
exact at worst +0.1%; the fp8 ctrl lane is untouched by the clamp's
windowed path). For a deployment still on the default `fp8_e4m3` KV,
nothing changes — the P6.47 lane activates only when
`--kv-cache-dtype` selects an MX format. To adopt the capacity lane,
add `ARC_KVDTYPE=mxfp4` (1.81× KV tokens at solo decode parity, now
with the full five-face ITL record 43.4→58.1 ms) or `mxk8v4`/
`mxfp8`; `ARC_MXKV_K2=0`/`ARC_MXKV_K3=0` revert the packed dequant
lanes. Per-prior-cert migration notes:
[`../CHANGELOG.md`](../CHANGELOG.md).

## 2. Boot procedure (from W14-serving state)

```bash
# on the rig host (address lives in the untracked local layer, never in
#                  tracked files — publishing-hygiene law; every ssh:
#                  export SSH_ASKPASS=/tmp/askpass.sh
#                  SSH_ASKPASS_REQUIRE=force DISPLAY=:0)
docker stop radiance-test                      # frees port 8000 + both GPUs
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert28 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
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
construction. Boot takes ~5–6 min to health 200 on cert26 qwen legs
(the cert23-lineage ×2 graph captures add ~29 s over the cert22 base's
285–295 s; gemma native-KV legs take ~13 min — the larger KV pool
profile, r184). Verify
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
