# vllm-arc

**Certified vLLM stack for `qwen3.8-27b-fp8` (dense hybrid-GDN + MTP) on dual Intel® Arc™ Pro B70 — TP2.**

A fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) at **v0.30.0** (package name `vllm_gaudi` preserved; the Arc/XPU work lives in `vllm_gaudi/xpu/`).

- Certified release: **`rel/v0.30.0-b70-cert7`** — every gate GREEN with banked evidence. cert7 = cert6's engine + the **P6.18 TurboQuant deprecation guard** (`vllm_gaudi/xpu/platform.py` refuses every `turboquant_*` kv-cache dtype at boot with a loud `ValueError` — 4-bit-class value quantization is a proven quality ceiling on this model; see the TurboQuant section below). Default `fp8_e4m3` posture byte-identical to cert6.
- **Project status: P0–P6.18 complete.** All phases closed with evidence — engine hardening (P2), cache lane (P3), certification battery (P4), audits + soak (P5), measurement/analysis phases P6.1–P6.15 (external-hit-rate truth, comparator law, telemetry off, publishing hygiene, main currency, the comprehensive cert4-native re-measurement, the P2P-allreduce closure r71, the docs overhaul, the END-stall mechanism analysis + sliced-store rung, the TurboQuant lagged-store fix + full preset validation, and the bystander-stall taxonomy + chunk-granularity Pareto matrix for ALL dtypes), **P6.16 the draft-sampling posture fix** (`ARC_SPEC_METHOD=probabilistic`: sampled drafts + rejection-ratio verify — the mathematically-correct sampler at temperature), **P6.17 the prob-lane certification (2026-11-03)**: the certified line is now **probabilistic drafting by default** — the serve script's default posture flipped (unset = probabilistic), and **every record face re-measured natively on the probabilistic lane** (comprehensive p617 campaign on cert6: t0/t1.0 batteries across fresh boots, TTFT ladder, warm re-hit, temp sweep, alt face, integrity + determinism + golden + garbage gates, no-spec A/B), and **P6.18 the TurboQuant deprecation (2026-11-03/04)**: corrupted generations reported under `turboquant_k8v4` were split by the p618b reproduction battery into Class B (deterministic degenerate generation on ALL FOUR presets, cold-included) and Class A (t0 warm-resume trajectory flips — inherent cached-serving numerics, dispositioned); the p618c roundtest proved the SoA store→decode Triton pair numerically exact on XPU, closing the root cause as the **4-bit-class value-precision quality ceiling** (upstream's own tables rate these presets +2.71…+20.59 % PPL on gpt-oss); fix = **platform-level refusal of every `turboquant_*` dtype at config time** (no override lever — new quant kernels exist in no admitted donor, scope law), certified as **cert7** with the full release-gate battery + the long-return gate 100 % CLEAN on every corruption/looping face (`results/2026-11-03-p618-cert7/`). The engine's serving code is byte-identical to cert6 under `fp8_e4m3` — the guard is the only delta. Full phase ledger: [`docs/TESTING-arc.md`](docs/TESTING-arc.md) (validation record) and [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) (evidence table).
- **Open improvement lanes** (all analysis-side, none certified yet): ESIMD dense-FA kernel project (r67b re-scope). The TP2 allreduce path question is **closed** — r71 proved direct P2P (10.23 GB/s D2D; all collectives on the `topo sycl` device path; `tools/pcie/README.md`).

## What this fork adds

| Area | What | Where |
|---|---|---|
| Platform | Arc XPU platform: device detection, mem-info patch, GMU conditional relax | `vllm_gaudi/xpu/platform.py` |
| Graphs | `FULL_DECODE_ONLY` XPU graphs (capture sizes 1–320), stream hygiene, first-request warmup interlock | `patches/`, `vllm_gaudi/xpu/patches/` |
| Speculative decoding | MTP (multi-token prediction) — **k3 recommended**, k4 documented alternative; acceptance gate instrument | `tools/arc_spec_gate.py` |
| KV cache | LMCache hybrid-KV connector: mamba/GDN boundary snap, TWO_HS transfer, lookup memo, pinned-memory safety law; TurboQuant dtypes refused at config time (P6.18 — proven 4-bit-class quality ceiling on this model) | `vllm_gaudi/xpu/patches/lmcache_hma_connector.py`, `vllm_gaudi/xpu/platform.py` |
| Quantization | FP8 weights with a block-quant **W8A16** route (vs stock per-token W8A8) | `vllm_gaudi/xpu/patches/fp8_block_w8a16.py` |
| Scheduler | Tail-split v4 for hybrid prefix-cache loads | `vllm_gaudi/xpu/patches/sched_tail_split.py` |
| Structured output | XGrammar backend, compact-JSON posture | `tools/serve_arc_b70.sh` |
| Distribution | Certified Docker image with `IMAGE_MANIFEST` (fork SHA, pinned wheels, vendored lmcache closure) | `Dockerfile`, `tools/build_image.sh` |

## Certified performance (dual Arc Pro B70, TP2, qwen3.8-27b-fp8)

**Certified posture (P6.17): probabilistic drafting — the serve script's default.** The certified line is `ARC_SPEC_METHOD=probabilistic` (the default when unset). **All record values below are measured natively on the probabilistic lane** (comprehensive p617 campaign on `vllm-arc:v0.30.0-b70-cert6` — every face re-measured on fresh boots; earlier prob-lane faces `results/2026-11-03-p616b-probmatrix/` + `results/2026-11-03-p616c-spechook/`. The cert7 release gate re-verified the key faces on `vllm-arc:v0.30.0-b70-cert7` — t0 in-family, integrity/determinism/spec green, long-return 100 % clean — `results/2026-11-03-p618-cert7/`. Methodology, face definitions, and per-face provenance in [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md)):

| Face | Value |
|---|---|
| t0 agg4x256 / solo200 | **291.4–303.1 / 83.7–84.9 tok/s** |
| t1.0-xhigh agg4x256 | **222.7–241.3 tok/s steady** |
| t1.0-xhigh solo | **64–71 tok/s, uniform (no collapse face in 20 measured pairs)** |
| Temperature sweep agg (t0.2/0.4/0.6/0.8 steady) | **214.9 / 215.0 / 222.3 / 224.7 tok/s** |
| Alt sampling face (t0.7/tp0.80/pp1.5) agg / solo | **271.5–281.3 / 82.7–92.1 tok/s** |
| TTFT solo 8k / 16k / 32k / 64k | **3.77 / 7.90 / 17.38 / 40.32 s** |
| Warm prefix re-hit 8k / 16k / 32k (t2) | **0.99 / 1.08 / 1.07 s** (warm-hit face 0.80–0.85 s) |
| MTP k3 acceptance, t0 | **83.7 %** strict-gate window / **85.9 %** battery window (mean accepted length 3.51 / 2.58; hard floor PASS) |
| MTP over no-spec (same-boot A/B) | **t0 2.35–2.47× · t1.0 1.84–2.09× · alt 2.42–2.53×** |

The dtype face behind these records is `fp8_e4m3` — the only supported
kv-cache dtype. The TurboQuant presets are **deprecated and refused at
boot** since cert7 (section below).

### TurboQuant KV-cache dtypes — DEPRECATED (refused at boot since cert7, P6.18)

> **DEPRECATED (closed 2026-11-04, P6.18): every `turboquant_*` kv-cache
> dtype is refused at config time by `ArcXpuPlatform` with a loud
> `ValueError` naming the certified default** — on every entry point (CLI,
> scripts, raw image), no override lever. Root cause is closed: the
> p618c roundtest proved the SoA store→decode Triton pair **numerically
> exact on XPU** (decode self-coherence rel_err ≤ 6e-6, deterministic, all
> four presets) — the corruption is the **4-bit-class value-precision
> quality ceiling** (16 affine levels per token-head; upstream's own tables
> rate these presets +2.71…+20.59 % PPL on gpt-oss), which degenerates the
> target model into deterministic repetition attractors on ALL FOUR
> presets, cold-included. Serve with `--kv-cache-dtype fp8_e4m3`
> (the default — unaffected, re-certified on cert7 with the long-return
> gate 100 % CLEAN).

What you see if you try (`--kv-cache-dtype turboquant_k8v4`): the boot dies
in ~65 s with `ValidationError … Value error, ArcXpuPlatform: kv_cache_dtype
'turboquant_k8v4' is deprecated on vllm-arc: … Serve with the certified
default 'fp8_e4m3'` — `Application startup complete` is never reached
(G0 guard-proof legs, both presets, `results/2026-11-03-p618-cert7/`).

Historical table (the pre-deprecation P6.14/P6.15/P6.16 capacity and speed
faces — **kept for the record, not servable on cert7+**; fp8 row is live):

| Preset | KV capacity | U32k / U64k TTFT | t1.0 agg (tok/s) | Status |
|---|---|---|---|---|
| `fp8_e4m3` (default) | 405,491 tok (1.55×) | 23.16 / 50.77 s | **241.10** | **the certified dtype — fastest aggregate, corruption-free** |
| `turboquant_k8v4` | 526,445 tok (2.01×) | 15.29 / 33.03 s | 172.95 | refused since cert7 |
| `turboquant_4bit_nc` | 757,641 tok (2.89×) | 15.91 / 34.02 s | 134.42 | refused since cert7 |
| `turboquant_k3v4_nc` | 853,220 tok (3.25×) | 15.48 / 33.78 s | 137.11 | refused since cert7 |
| `turboquant_3bit_nc` | **975,329 tok (3.72×)** | 16.02 / 35.74 s | 171.32 | refused since cert7 |

The TQ capacity/TTFT advantages are **not recoverable** on this stack: the
degeneracy is the quantizer's precision, not a fixable defect (no
quality-correcting kernel exists in any admitted donor — scope law). For
stall relief on the certified fp8 line use the measured levers instead:
`ARC_MNBT=4096` (first choice) or the p65 long-context profile — see
"Optional postures" below. Campaign evidence: reproduction battery
`results/2026-11-03-p618b-repro/`, kernel roundtest
`results/2026-11-03-p618c-roundtest/`, release gate + long-return
`results/2026-11-03-p618-cert7/`; the earlier per-preset validation legs
`results/2026-11-02-p614-l1v2/` + `-l3/` remain banked as history. Full
matrix + dispositions: [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md).

## Quick start

```bash
# 1. Build the certified image from the certified tag.
#    Runs ON THE RIG (or an equally prepared Linux host): needs the pinned-wheel
#    stashes, the rig-local base image, and internet on the FIRST build only
#    (stack layers cache in docker after). Every patch ships in the archive
#    (COPY src/ + pip install .); wheel pins enforced by tools/wheel_guard.sh.
#    Full provenance + the wheel-stash inventory: docs/CONFIGURATION-arc.md;
#    rebuildability audit (every image input mapped to a tracked artifact
#    or a pinned external prerequisite): results/2026-11-03-p618-cert7/
#    REPRODUCIBILITY-AUDIT.md.
RIG=<build-host>                                # <- set: your rig/build host
                                                 #    (never commit the real
                                                 #    address — publishing-
                                                 #    hygiene law, and run
                                                 #    tools/leak_sweep.sh
                                                 #    before pushing)
SHA=$(git rev-parse rel/v0.30.0-b70-cert7)      # the certified commit
git archive --format=tar -o /tmp/fork.tar rel/v0.30.0-b70-cert7
scp /tmp/fork.tar root@$RIG:/root/build/
ssh root@$RIG 'tar -xOf /root/build/fork.tar tools/build_image.sh > /root/build/build_image.sh'
ssh root@$RIG bash /root/build/build_image.sh /root/build/fork.tar $SHA vllm-arc:v0.30.0-b70-cert7
#    -> wheel_guard PASSes (4/4 md5s), IMAGE_MANIFEST records fork_sha == $SHA
#       (verify: docker run --rm --entrypoint cat
#        vllm-arc:v0.30.0-b70-cert7 /etc/vllm-arc/IMAGE_MANIFEST)
#       ~15 min cold; ~1 min when only the source changed (stack layers
#       cached — layer-order law in the Dockerfile).
#
#    Tag lineage (full table: docs/PERFORMANCE-arc.md "Image lineage"):
#    cert7 = current (the P6.18 TQ deprecation guard; fp8 default byte-
#    identical to cert6). Superseded: cert6 (TQ dtypes servable but
#    degenerate on this model — do NOT serve TQ on cert6), cert5
#    (identical default behavior, lacks the TQ-aware store hunk), cert4
#    (identical default behavior, lacks the default-OFF slice hunk),
#    cert3 (identical default behavior, lacks the deferral hunk), cert2
#    (0.5.5-era wheel line), cert (predates r46u — never build from it).

# 2. Boot — the validated posture (annotated semantics in
#    docs/CONFIGURATION-arc.md Step 1; key points: --privileged REQUIRED
#    with CCL_ZE_IPC_EXCHANGE=pidfd [TP2 init law]; --ipc host shares
#    host /dev/shm; --shm-size 64g is a ceiling, not a reservation;
#    model :ro — the checkpoint is an input; the -e pair are the 2
#    boot-layer vars of the 25-var chain).
docker run -d --name arc-test --privileged --network host --ipc host \
  --shm-size 64g \
  -v /path/to/qwen3.8-27b-fp8:/models/target:ro \
  -v ~/.cache/arc/neo:/root/.neo_cache \
  -v ~/.cache/arc/triton:/root/.triton_cache \
  -v ~/.cache/arc/inductor:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint bash vllm-arc:v0.30.0-b70-cert7 -c "sleep infinity"

# 3. Serve — the certified line (P6.17: probabilistic drafting = the serve
#    script's default posture), byte-identical to
#    docs/CONFIGURATION-arc.md Step 2 (that file is canonical; per-var
#    semantics there + docs/envs.md). The rig pattern: script + docker cp
#    + exec; logs land in /root/serve_full.log.
cat > serve_k3.sh <<'EOF'
#!/bin/bash
# --- serve environment: 23 of the certified 25-var chain (2 set at boot) ---
export PYTHONHASHSEED=0                 # init determinism
export ZE_AFFINITY_MASK=0,1             # one Arc Pro B70 per socket
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd        # --privileged pair (TP2 init law)
export CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
export CCL_SYCL_ALLGATHERV_TMP_BUF=0
export CCL_SYCL_ALLREDUCE_TMP_BUF=0
export VLLM_XPU_ENABLE_XPU_GRAPH=1      # graph posture law
export ARC_TRACE_FLIP=1                 # flip-payload interlock
# --- LMCache certified set (10 — keep as a set) ---
export LMCACHE_USE_EXPERIMENTAL_API=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=100
export LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_DISABLE_PINNED_MEMORY=1  # law: pinned USM >16-32 GiB/proc segfaults
export LMCACHE_TRACK_USAGE=false        # P6.3: LMCache telemetry is default-ON; rig serves offline
# --- the certified CLI: MTP k3 + probabilistic drafting (P6.17 — the CLI
#     form of the serve-script default posture; ARC_SPEC_METHOD=probabilistic
#     on the boot interface produces identical runtime bytes), LMCache ON,
#     mode:none (no torch.compile) + decode-only graphs, explicit capture
#     list. ---
exec /opt/venv/bin/vllm serve --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 64 \
  --dtype float16 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --mamba-cache-mode=align \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache.extra_config":{"mamba_pool_ratio":2}},"engine_id":"arc-p40"}' \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --port 8000 \
  > /root/serve_full.log 2>&1
EOF
docker cp serve_k3.sh arc-test:/root/
docker exec -d arc-test bash /root/serve_k3.sh

# 4. Health — first boot takes ~10 min to reach 200 (model load + graph capture)
curl -s -o /dev/null -w "%{http_code}\n" -m 5 http://127.0.0.1:8000/health
```

Optional postures (all documented in full in
[`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md)):

- **OpenAI `reasoning_effort` compatibility** — mount + pass
  `assets/chat_template_qwen38_high.jinja` (high/max→xhigh aliasing,
  none→thinking-off; default path identical either way).
- **Bystander-stall tuning — first choice (P6.15)** — `ARC_MNBT=4096`
  on fp8: bystander p95 2030/2587 ms (vs 3986/5295 default) at TTFT
  +8–13 %, KV +6.8 %, decode-neutral — re-verified natively on cert7
  (p95 2052/2523 vs 4001/5273, `results/2026-11-03-p618-cert7/PERF.md`);
  menu in `docs/CONFIGURATION-arc.md`.
- **Long-context serve profile (p65)** — the deeper-stall niche for
  ≥48k prompts into concurrent decoders:
  `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` +
  `LMC_DEFER_MIN_TOKENS=40960` (U64k TTFT 51.16 s, bystander p95/max
  1580/1599 ms). All three together; never lower the min-tokens gate
  below 40960 (32k-class prompts regress +4.2 s). Prefer
  `ARC_MNBT=4096` unless the absolute stall floor is the binding
  constraint.

The uniform boot interface with posture knobs (rig paths) is
`tools/serve_arc_b70.sh`.

## Documentation

- [`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md) — how to run the certified posture (the canonical serve block, knob table, posture laws)
- [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md) — measured numbers: **current certified record (P6.17 prob-lane campaign on cert6, re-verified at the cert7 release gate) first**, the dtype/posture matrices, then pre-P6.17 history (posture-labeled)
- [`docs/TESTING-arc.md`](docs/TESTING-arc.md) — 14 test suites, gates, validation record
- [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) — swap an incumbent stack for this one (boot, verify, rollback) + the evidence table
- [`docs/envs.md`](docs/envs.md) — complete env/knob control surface + naming map

## Repository model

- **`main` is linear and protected**: changes land as squash commits from per-item lanes (`feat/arc-pNN-*`); direct pushes, force pushes, and deletions are blocked; PRs are required.
- **Certified releases are annotated tags** pointing at the certified `main` commit — current: `rel/v0.30.0-b70-cert7`; superseded: `rel/v0.30.0-b70-cert6` (fp8 behavior byte-identical; its servable `turboquant_*` dtypes degenerate on this model — the reason cert7 refuses them), `rel/v0.30.0-b70-cert5` (identical default behavior, lacks the P6.14 TQ-aware store hunk), `rel/v0.30.0-b70-cert4` (identical default behavior, lacks the default-OFF P6.13 slice hunk), `rel/v0.30.0-b70-cert3` (identical default behavior, lacks the p65 deferral hunk), `rel/v0.30.0-b70-cert2` (0.5.5-era lmcache line — valid, older), `rel/v0.30.0-b70-cert` (predates the r46u glue-alias fix — never build from it). The Docker image embeds the fork SHA in `/etc/vllm-arc/IMAGE_MANIFEST` for A/B verification.
- **Evidence is in-tree**: every validation leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT + raw outputs) — **107 evidence dirs, 100 VERDICTs of record**, P0 through P6.
- **No secrets or real host addresses in-tree (publishing-hygiene law)**: tracked == pushable, so credentials, rig/host addresses, and private identities never enter tracked files; banked logs carry redacted host addresses (each banking-time scrub documented in its `results/REDACTION-*.md` note). `tools/leak_sweep.sh` must exit 0 before every push.
- The `feat/arc-*` branches are kept deliberately: squash-merging leaves the per-item development history nowhere else.

## Upstream & license

Fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) @ tag `v0.30.0` (the vLLM hardware plugin for Intel® Gaudi®). Upstream's README and docs for the base plugin: [vllm-gaudi docs](https://vllm-gaudi.readthedocs.io/en/latest/index.html). License: Apache-2.0 (see [LICENSE](LICENSE)).
