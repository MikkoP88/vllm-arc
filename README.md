# vllm-arc

**Certified vLLM stack for `qwen3.8-27b-fp8` (dense hybrid-GDN + MTP) on dual Intel® Arc™ Pro B70 — TP2.**

A fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) at **v0.30.0** (package name `vllm_gaudi` preserved; the Arc/XPU work lives in `vllm_gaudi/xpu/`).

- Certified release: **`rel/v0.30.0-b70-cert8`** — every gate GREEN with banked evidence. cert8 = cert7's engine with the P6.18 boot refusal lifted + the **P6.19 TurboQuant spec-verify fix** (`vllm_gaudi/xpu/patches/tq_spec_fix.py`: donor-v19b verify-capture forcing + graph-safe split-row continuation dispatch, active by default on every `turboquant_*` dtype): **all four TurboQuant presets now SERVE CLEAN** — zero REP-LOOP/JUNK/BLEED/EMPTY on every dtype, all-dtype gate with the fp8 certified posture as control (`results/2026-11-04-p619-tqverify/`). Default `fp8_e4m3` posture unchanged (fix absent by construction — the hooked module never imports on fp8).
- **Project status: P0–P6.19 complete.** All phases closed with evidence — engine hardening (P2), cache lane (P3), certification battery (P4), audits + soak (P5), the P6.1–P6.15 measurement/analysis phases, **P6.16/P6.17 the draft-sampling posture** (probabilistic drafting = the mathematically-correct sampler at temperature, the certified serve-script default since P6.17 — every record face re-measured on that lane), **P6.18 the TurboQuant corruption campaign** (reproduction battery split the corruption into Class B/Class A; interim fix = config-time refusal of every `turboquant_*` dtype, certified as cert7), and **P6.19 the definitive TurboQuant FIX (cert8)**: the corruption was root-caused to **three stacked spec-verify defects — none of them the quantizer's precision** — captured verify graphs baking the raw-KV flash fast-path (RC1), a replay-broken multi-row synthetic-decode dispatch (RC2), and the wheel's two TQ ops trees (RC3 — the first MQ-kernel draft targeted the never-used SoA tree and read the live cache as zeros). Fix = wheel-side hook `tq_spec_fix.py` (donor-v19b capture forcing + **split-row dispatch**, default-ON) + `tq_mq_aos.py` (MQ verify kernel on the engine's AoS layout, opt-in `VLLM_ARC_TQ_VERIFY_MODE=mq`, +21 % agg8, experimental); P6.18 refusal lifted; **all-dtype gate: zero garble on all four TQ dtypes AND the fp8 control** (identity faces dispositioned with the W14 incumbent control — engine-class probabilistic-drafting faces, not corruption). Full phase ledger with every per-phase record: [`docs/TESTING-arc.md`](docs/TESTING-arc.md) (validation record) and [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) (evidence table).
- **Open improvement lanes** (all analysis-side, none certified yet): ESIMD dense-FA kernel project (r67b re-scope). The TP2 allreduce path question is **closed** — r71 proved direct P2P (10.23 GB/s D2D; all collectives on the `topo sycl` device path; `tools/pcie/README.md`).

## What this fork adds

| Area | What | Where |
|---|---|---|
| Platform | Arc XPU platform: device detection, mem-info patch, GMU conditional relax | `vllm_gaudi/xpu/platform.py` |
| Graphs | `FULL_DECODE_ONLY` XPU graphs (capture sizes 1–320), stream hygiene, first-request warmup interlock | `patches/`, `vllm_gaudi/xpu/patches/` |
| Speculative decoding | MTP (multi-token prediction) — **k3 recommended**, k4 documented alternative; acceptance gate instrument | `tools/arc_spec_gate.py` |
| KV cache | LMCache hybrid-KV connector: mamba/GDN boundary snap, TWO_HS transfer, lookup memo, pinned-memory safety law; TurboQuant dtypes SERVE CLEAN since P6.19 (spec-verify fix active by default; see the TurboQuant section) | `vllm_gaudi/xpu/patches/lmcache_hma_connector.py`, `vllm_gaudi/xpu/patches/tq_spec_fix.py` |
| Quantization | FP8 weights with a block-quant **W8A16** route (vs stock per-token W8A8) | `vllm_gaudi/xpu/patches/fp8_block_w8a16.py` |
| Scheduler | Tail-split v4 for hybrid prefix-cache loads | `vllm_gaudi/xpu/patches/sched_tail_split.py` |
| Structured output | XGrammar backend, compact-JSON posture | `tools/serve_arc_b70.sh` |
| Distribution | Certified Docker image with `IMAGE_MANIFEST` (fork SHA, pinned wheels, vendored lmcache closure) | `Dockerfile`, `tools/build_image.sh` |

## Certified performance (dual Arc Pro B70, TP2, qwen3.8-27b-fp8)

**Certified posture (P6.17): probabilistic drafting — the serve script's default.** The certified line is `ARC_SPEC_METHOD=probabilistic` (the default when unset). **All record values below are measured natively on the probabilistic lane** (comprehensive p617 campaign on `vllm-arc:v0.30.0-b70-cert6` — every face re-measured on fresh boots; earlier prob-lane faces `results/2026-11-03-p616b-probmatrix/` + `results/2026-11-03-p616c-spechook/`. The cert7 and cert8 release gates re-verified the key faces on the tag-built images — t0 in-family (cert8: 300.65 agg / 84.53–84.57 solo), integrity/determinism/spec green, long-return t0 lanes 100 % clean — `results/2026-11-03-p618-cert7/`, `results/2026-11-04-p619-tqverify/`. Methodology, face definitions, and per-face provenance in [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md)):

| Face | Value |
|---|---|
| t0 agg4x256 / solo200 | **291.4–303.1 / 83.7–84.9 tok/s** (cert8 gate: 300.65 / 84.53–84.57) |
| t1.0-xhigh agg4x256 | **218–241 tok/s steady** cross-boot (p616/p617 boots + the cert7/cert8 gates 218.29/221.36 — the face's highest-variance posture) |
| t1.0-xhigh solo | **64–71 tok/s, uniform (no collapse face in 20 measured pairs)** |
| Temperature sweep agg (t0.2/0.4/0.6/0.8 steady) | **214.9 / 215.0 / 222.3 / 224.7 tok/s** |
| Alt sampling face (t0.7/tp0.80/pp1.5) agg / solo | **271.5–281.3 / 82.7–92.1 tok/s** |
| TTFT solo 8k / 16k / 32k / 64k | **3.77 / 7.90 / 17.38 / 40.32 s** |
| Warm prefix re-hit 8k / 16k / 32k (t2) | **0.99 / 1.08 / 1.07 s** (warm-hit face 0.80–0.85 s) |
| MTP k3 acceptance, t0 | **83.7 %** strict-gate window / **85.9 %** battery window (mean accepted length 3.51 / 2.58; hard floor PASS) |
| MTP over no-spec (same-boot A/B) | **t0 2.35–2.47× · t1.0 1.84–2.09× · alt 2.42–2.53×** |

The dtype face behind these records is `fp8_e4m3` — the certified default
and still the fastest aggregate. The TurboQuant presets are **servable
since cert8** (P6.19 spec-verify fix — section below): they trade
aggregate speed for KV capacity (up to 3.68× at the cert8 gate;
3.72× at the P6.14 campaign's memory state).

### TurboQuant KV-cache dtypes — SERVABLE since cert8 (P6.19 spec-verify fix)

> **FIXED (2026-11-04, P6.19 / cert8): every `turboquant_*` kv-cache dtype
> serves clean** — zero REP-LOOP / JUNK / BLEED / EMPTY on all four presets,
> verified by the all-dtype long-return gate with the certified fp8 posture
> as control (`results/2026-11-04-p619-tqverify/`). The P6.18 "4-bit-class
> precision ceiling" disposition was wrong: the corruption was three stacked
> spec-verify defects, all fixed wheel-side by `tq_spec_fix.py` (active by
> default on TQ dtypes; fp8 never imports the hooked module):
>
> 1. **#TQ-blind-verify (RC1)** — captured XPU-graph verify shapes baked the
>    raw-KV flash fast-path (`max_query_len == max_seq_len` at capture), so
>    verify replays never read the quantized cache. Fix: donor-v19b capture
>    forcing (`max_seq_len=q_len+1`, `seq_lens_cpu=q_len+1`,
>    `seq_lens=q_len`) — enforcing the stock builder's own contract ("real
>    seq_lens are filled at replay time", `turboquant_attn.py` capture
>    builder).
> 2. **Replay-broken multi-row dispatch (RC2)** — the stock continuation
>    branch's single multi-row synthetic-decode call (stride-0 expanded
>    block table + synthesized seq_lens + B=q_len grid) computes wrong under
>    graph REPLAY. Fix: **split-row dispatch** (ships) — q_len separate
>    single-row stock decode calls, per-row causal limit
>    `seq_lens[i]-(q_len-1-r)` as a recorded device op on the LIVE seq_lens
>    buffer.
> 3. **Two TQ ops trees (RC3)** — the wheel ships a slot-interleaved AoS
>    pair (`ops/triton_turboquant_{store,decode}.py` — the engine's XPU
>    default) AND a FlyDSL-gated SoA tree never used on XPU. The first
>    single-pass MQ kernel draft targeted SoA and read the live cache as
>    exact zeros (REP-LOOP on every class). Fix: `tq_mq_aos.py` re-ports
>    the kernel to the AoS layout — offline-exact vs the engine's own pair
>    on all four presets (MQVAL5). LAW: offline validation must import the
>    pair the runtime dispatches.
>
> Knobs: `VLLM_ARC_TQ_SPEC_FIX=1` (default), `VLLM_ARC_TQ_VERIFY_MODE=
> split|mq|synth` (split ships; mq = single-pass kernel, +20 % agg8,
> experimental), `VLLM_ARC_TQ_MQ_SPLITS=32` — see `docs/envs.md`.
>
> Identity faces (t0 variant flips / t1-seed sporadic divergence) remain on
> some classes/dtypes — dispositioned as the engine's probabilistic-drafting
> class (p618b Class-A semantics): the certified fp8 control fires the same
> class with the fix absent by construction, and the variant hashes are
> cross-boot-stable. t0 within-path byte-identity holds wherever t0 is
> identical; the coarsest preset (3bit) shows the widest t0 variant drift.

Servable preset table (KV capacity = **cert8-gate same-day values** —
capacity tracks boot-day free memory, so the P6.14-era numbers in
PERFORMANCE-arc stay as that campaign's labeled record; TTFT and
t1.0-agg columns are the P6.14 campaign faces, not re-measured since;
corruption gate re-proven on cert8; fp8 row is the certified default.
Comprehensive per-dtype guide: `docs/CONFIGURATION-arc.md` "KV-cache
dtype"):

| Preset | KV capacity (cert8) | U32k / U64k TTFT (P6.14) | t1.0 agg (P6.14) | Status |
|---|---|---|---|---|
| `fp8_e4m3` (default) | 400,739 tok | 23.16 / 50.77 s¹ | **241.10** | **the certified default — fastest aggregate; the TQ fix never engages (control-proven)** |
| `turboquant_k8v4` | 519,972 tok (1.98×) | 15.29 / 33.03 s | 172.95 | **servable (cert8)** — spec tps solo 54.4 / agg8 236.0 split; **284.8 MQ mode** (+21 %, experimental — `VLLM_ARC_TQ_VERIFY_MODE=mq`) |
| `turboquant_4bit_nc` | 748,549 tok (2.86×) | 15.91 / 34.02 s | 134.42 | **servable (cert8)** — cleanest identity record (zero flags anywhere in the gate) |
| `turboquant_k3v4_nc` | 843,202 tok (3.22×) | 15.48 / 33.78 s | 137.11 | **servable (cert8)** |
| `turboquant_3bit_nc` | **963,764 tok (3.68×)** | 16.02 / 35.74 s | 171.32 | **servable (cert8)** — widest t0 variant drift (8 V levels; coherent output, not corruption) |

¹ fp8 comparator as run in the P6.14 A/B (thr2048-profile face);
true-default fp8 TTFT is 17.25/39.77 s — the honest TQ TTFT edge is
−10…−16 %.

For stall relief on the certified fp8 line the measured levers remain first
choice: `ARC_MNBT=4096` or the p65 long-context profile — see "Optional
postures" below. Campaign evidence: P6.19 root-cause + fix + all-dtype
gate + best-practices audit `results/2026-11-04-p619-tqverify/` (this
closed the P6.18 disposition); the P6.18 record stays banked at
`results/2026-11-03-p618{b-repro,c-roundtest,-cert7}/`; the earlier
per-preset validation legs `results/2026-11-02-p614-l1v2/` + `-l3/` remain
banked as history. Full matrix + dispositions:
[`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md).

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
SHA=$(git rev-parse rel/v0.30.0-b70-cert8)      # the certified commit
git archive --format=tar -o /tmp/fork.tar rel/v0.30.0-b70-cert8
scp /tmp/fork.tar root@$RIG:/root/build/
ssh root@$RIG 'tar -xOf /root/build/fork.tar tools/build_image.sh > /root/build/build_image.sh'
ssh root@$RIG bash /root/build/build_image.sh /root/build/fork.tar $SHA vllm-arc:v0.30.0-b70-cert8
#    -> wheel_guard PASSes (4/4 md5s), IMAGE_MANIFEST records fork_sha == $SHA
#       (verify: docker run --rm --entrypoint cat
#        vllm-arc:v0.30.0-b70-cert8 /etc/vllm-arc/IMAGE_MANIFEST)
#       ~15 min cold; ~1 min when only the source changed (stack layers
#       cached — layer-order law in the Dockerfile).
#
#    Tag lineage (full table: docs/PERFORMANCE-arc.md "Image lineage"):
#    cert8 = current (P6.19 TQ spec-verify fix — all four turboquant_*
#    dtypes serve clean; refusal lifted; fp8 default unchanged). Superseded:
#    cert7 (its P6.18 refusal is superseded by the P6.19 fix), cert6
#    (its TQ dtypes carry the pre-fix corrupted spec-verify path — do NOT
#    serve TQ on cert6 or earlier), cert5 (identical default behavior,
#    lacks the TQ-aware store hunk), cert4 (identical default behavior,
#    lacks the default-OFF slice hunk), cert3 (identical default behavior,
#    lacks the deferral hunk), cert2 (0.5.5-era wheel line), cert (predates
#    r46u — never build from it).

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
  --entrypoint bash vllm-arc:v0.30.0-b70-cert8 -c "sleep infinity"

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

# 4. Health — first boot takes ~5 min to reach 200 (295 s at the cert7/
#    cert8 gates: model load + graph capture)
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
- [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md) — measured numbers: **current certified record (P6.17 prob-lane campaign on cert6, re-verified at the cert7/cert8 release gates) first**, the comprehensive dtype faces (fp8 + all four TurboQuant presets + MQ experimental mode), the posture matrices, then pre-P6.17 history (posture-labeled)
- [`docs/TESTING-arc.md`](docs/TESTING-arc.md) — 14 test suites, gates, validation record
- [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) — swap an incumbent stack for this one (boot, verify, rollback) + the evidence table
- [`docs/envs.md`](docs/envs.md) — complete env/knob control surface + naming map

## Repository model

- **`main` is linear and protected**: changes land as squash commits from per-item lanes (`feat/arc-pNN-*`); direct pushes, force pushes, and deletions are blocked; PRs are required.
- **Certified releases are annotated tags** pointing at the certified `main` commit — current: `rel/v0.30.0-b70-cert8`; superseded: `rel/v0.30.0-b70-cert7` (cert6 engine + the P6.18 TQ refusal — superseded by the P6.19 fix: its refusal is lifted on cert8 and the TQ dtypes serve clean), `rel/v0.30.0-b70-cert6` (fp8 behavior byte-identical; its servable `turboquant_*` dtypes carried the P6.18-corrupted spec-verify path), `rel/v0.30.0-b70-cert5` (identical default behavior, lacks the P6.14 TQ-aware store hunk), `rel/v0.30.0-b70-cert4` (identical default behavior, lacks the default-OFF P6.13 slice hunk), `rel/v0.30.0-b70-cert3` (identical default behavior, lacks the p65 deferral hunk), `rel/v0.30.0-b70-cert2` (0.5.5-era lmcache line — valid, older), `rel/v0.30.0-b70-cert` (predates the r46u glue-alias fix — never build from it). The Docker image embeds the fork SHA in `/etc/vllm-arc/IMAGE_MANIFEST` for A/B verification.
- **Evidence is in-tree**: every validation leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT + raw outputs) — **108 evidence dirs, 101 VERDICTs of record**, P0 through P6.
- **No secrets or real host addresses in-tree (publishing-hygiene law)**: tracked == pushable, so credentials, rig/host addresses, and private identities never enter tracked files; banked logs carry redacted host addresses (each banking-time scrub documented in its `results/REDACTION-*.md` note). `tools/leak_sweep.sh` must exit 0 before every push.
- The `feat/arc-*` branches are kept deliberately: squash-merging leaves the per-item development history nowhere else.

## Upstream & license

Fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) @ tag `v0.30.0` (the vLLM hardware plugin for Intel® Gaudi®). Upstream's README and docs for the base plugin: [vllm-gaudi docs](https://vllm-gaudi.readthedocs.io/en/latest/index.html). License: Apache-2.0 (see [LICENSE](LICENSE)).
