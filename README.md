# vllm-arc

**Certified vLLM stack for `qwen3.8-27b-fp8` (dense hybrid-GDN + MTP) on dual Intel® Arc™ Pro B70 — TP2.**

A fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) at **v0.30.0** (package name `vllm_gaudi` preserved; the Arc/XPU work lives in `vllm_gaudi/xpu/`).

- Certified release: **`rel/v0.30.0-b70-cert21`** — every gate GREEN with banked evidence. cert21 = cert20 + the **P6.31 online-MXFP8 W8A16 route**: online `ARC_QUANT=mxfp8` on a BASE bf16 checkpoint keeps its 8-bit e4m3 weights but serves GEMMs as **W8A16** — bf16 activations × f8 weights, group-32 e8m0 weight scales — via the arc5 wheel's `_xpu_C.fp8_gemm_w8a16_mxfp8` (stock online mxfp8 is W8A8: per-call activation quant ≈ the GEMM at decode Ms plus the slow (f8,f8) oneDNN lane). Same-image A/B (r124/r124b, both target models): **qwen U8k 3.61 s / U32k 17.07 s / decode 53.8–55.6 solo / 265.9 agg8 tok/s = 1.37–1.55× over stock W8A8 on every face, spec 0.9804**; **gemma 3.87 / 20.94 s, 27.7 / 192.4 tok/s = 1.57–1.85× over stock** (gemma online-mxfp8 legs need `ARC_MAXLEN=40960` — 8-bit weights double mxfp4's footprint on the 31B model and the KV pool at gmu 0.8 cannot hold a 256K seq; capacity law, not a lane defect). Kernel matrix (r123): the op is 1.74×/2.68× faster than stock mxfp8 GEMM at decode/prefill, 1.50×/1.06× behind the checkpoint-fp8 w16pc class — e2e adjudication: **checkpoint-fp8 stays the perf default; `ARC_MXFP8_MODE=w8a16` is the online-mxfp8 lane route; mxfp4-w4a16 (cert20) stays the superior online lane**; the W4A8 (e4m3×f4) probe is a banked oneDNN-refused closed-negative. Knob `ARC_MXFP8_MODE` (unset/`w8a16` = route — default; `w8a8` = stock control). Full release detail + per-cert lineage: [`docs/STATUS-arc.md`](docs/STATUS-arc.md).
- **Project status: P0–P6.31 complete** — all phases closed with evidence (engine hardening → cache lane → certification battery → audits/soak → the P6.1–P6.25 measurement, fix, and certification phases; the in-scope improvement ledger closed at P6.24 and audited exhaustive at P6.25; the P6.26/P6.27 quant-lane validations; **P6.28 the gemma-4 campaign** — the G2 long-prefill freeze solved in all three layers (cert16/cert17), gemma-4-31B fp8 + mxfp8 validated end-to-end, and the prefill perf root cause closed r108aa/r108ab then FIXED at cert18 — the FA2 route to the engine's own ESVD kernel (34x microbench, 2.2–16.3x on the faces, +6 % decode; r108ac/r108ad/r108ae); **P6.29 the remaining-bottleneck campaign** — the sliding hd256 wall measured k-independent (r109a/r109a3), routed to the same FA2 kernel at cert19 (2.4–2.9x on the faces, decode +45 %/2.64x; r109a6 ref-bug closure + r111 ladder), spec parity dispositioned (MTP: no draft weights; ngram: upstream structural refusal — banked), qwen discriminator green in-band; **P6.30 the online-quant campaign** — online MXFP4/MXFP8 of base checkpoints implemented and the mxfp4 perf MAJOR ISSUE root-caused end-to-end (per-call act quant ≈ GEMM cost at decode + the (f4,f4) gemmstone gate → `matmul:ref`, structurally) and FIXED with the W4A16 route (cert20: arc4 wheel op + reroute patch behind `ARC_MXFP4_MODE`), validated on both target models — superior to online mxfp8 on every qwen axis (1.47–1.75×), the e8m0 group-scale GEMM penalty pinned mechanistically on oneDNN v3.13 (jit:gemm with bidirectional e8m0 scales, still 2.6× slower than bf16 at prefill), three new engineering laws banked incl. the `vllm_xpu_kernels._xpu_C` dlopen law; **P6.31 the online-mxfp8 W8A16 route** — the same act-quant-elimination fix applied to the 8-bit lane (cert21: arc5 wheel op `fp8_gemm_w8a16_mxfp8` + `mxfp8_w8a16.py` reroute behind `ARC_MXFP8_MODE`), r123 kernel matrix → r124/r124b e2e on both models: 1.37–1.85× over stock W8A8 everywhere, checkpoint-fp8 retains the perf crown by adjudication, W4A8 closed oneDNN-refused, gemma online-mxfp8 capacity law banked). Full phase narrative + per-phase records: [`docs/STATUS-arc.md`](docs/STATUS-arc.md), [`docs/TESTING-arc.md`](docs/TESTING-arc.md) (validation record), [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) (evidence table).
- **Improvement-lane status: the in-scope ledger is CLOSED (P6.24; audited exhaustive P6.25)** — every prefill device slice is dispositioned by measurement (GEMM / allreduce / FA / GDN / norms-act; host path clean; the last unmeasured door — cross-stream AR∥GEMM co-tenancy — closed measured at P6.25), the stack sits at the practical in-scope floor, and the transport was proven clean. Per-slice table + closed-route ledger: [`docs/STATUS-arc.md`](docs/STATUS-arc.md).

## What this fork adds

| Area | What | Where |
|---|---|---|
| Platform | Arc XPU platform: device detection, mem-info patch, GMU conditional relax | `vllm_gaudi/xpu/platform.py` |
| Graphs | `FULL_DECODE_ONLY` XPU graphs (capture sizes 1–320), stream hygiene, first-request warmup interlock | `patches/`, `vllm_gaudi/xpu/patches/` |
| Speculative decoding | MTP (multi-token prediction) — **k3 recommended**, k4 documented alternative; acceptance gate instrument | `tools/arc_spec_gate.py` |
| KV cache | LMCache hybrid-KV connector: mamba/GDN boundary snap, TWO_HS transfer, lookup memo, pinned-memory safety law; **off-thread store since P6.20** (L3, default ON — engine thread never waits on the store body's D2H gathers); TurboQuant dtypes SERVE CLEAN since P6.19 (spec-verify fix active by default; see the TurboQuant section) | `vllm_gaudi/xpu/patches/lmcache_hma_connector.py`, `vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py`, `vllm_gaudi/xpu/patches/tq_spec_fix.py` |
| Quantization | FP8 weights with a block-quant **W8A16** route (vs stock per-token W8A8) | `vllm_gaudi/xpu/patches/fp8_block_w8a16.py` |
| Scheduler | Tail-split v4 for hybrid prefix-cache loads | `vllm_gaudi/xpu/patches/sched_tail_split.py` |
| Telemetry | Silent-off diagnostics instruments: per-step host timeline (`ARC_STEP_TRACE`, P6.20) + per-call distributed-collective census (`ARC_AR_TRACE`, P6.22) — registry-exempt, zero cost when off | `vllm_gaudi/xpu/patches/step_trace.py`, `vllm_gaudi/xpu/patches/ar_trace.py` |
| Comm (image overlay) | **e4m3-wire allreduce** for the TP2 fp16 large ring (oneCCL 2022.0.0 same-version rebuild + patch, default-OFF env `CCL_SYCL_ALLREDUCE_FP8`; cert10 ships it as a two-file lib overlay — opt-in 1.65× at the dominant AR) | `external_patches/oneccl/`, `tools/build_oneccl_arc.sh` |
| Structured output | XGrammar backend, compact-JSON posture | `tools/serve_arc_b70.sh` |
| Distribution | Certified Docker image with `IMAGE_MANIFEST` (fork SHA, pinned wheels, vendored lmcache closure) | `Dockerfile`, `tools/build_image.sh` |

## Certified performance (dual Arc Pro B70, TP2, qwen3.8-27b-fp8)

**Certified posture (P6.17): probabilistic drafting — the serve script's default.** The certified line is `ARC_SPEC_METHOD=probabilistic` (the default when unset). **All record values below are measured natively on the probabilistic lane** (comprehensive p617 campaign on `vllm-arc:v0.30.0-b70-cert6` — every face re-measured on fresh boots; earlier prob-lane faces `results/2026-11-03-p616b-probmatrix/` + `results/2026-11-03-p616c-spechook/`. The cert7–cert10 release gates re-verified the key faces on the tag-built images — t0 in-family (cert10 default-posture gate: 300.39 agg steady / 84.52–84.51 solo, FP8 env proven absent; cert9: 296.91 / 84.54–84.48; cert8: 300.65 / 84.53–84.57), integrity/determinism/spec green (cert10 spec gate at exactly the banked values: pos1 0.9258 — also byte-exact on the FP8=1 leg), long-return t0 lanes 100 % clean — `results/2026-11-03-p618-cert7/`, `results/2026-11-04-p619-tqverify/`, `results/2026-11-04-p620-prefill-opt/`, `results/2026-11-05-p624-fp8-ar/`. Methodology, face definitions, and per-face provenance in [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md)):

| Face | Value |
|---|---|
| t0 agg4x256 / solo200 | **291.4–303.1 / 83.7–84.9 tok/s** (cert10 gate: 300.39 / 84.52–84.51; cert9 gate: 296.91 / 84.54–84.48; cert8 gate: 300.65 / 84.53–84.57) |
| t1.0-xhigh agg4x256 | **218–241 tok/s steady** cross-boot (p616/p617 boots + the cert7/cert8/cert9 gates 218.29/221.36/240.06 — the face's highest-variance posture; cert9 first-sampled-face 133.47 transient, the documented warm-up face; cert10 gate t1: one boot-stochastic REP-LOOP face, dispositioned by the r99c fresh-boot annex — clean on a fresh boot and same-day cert9 stock, future bar REPORT-with-annex) |
| t1.0-xhigh solo | **64–71 tok/s, uniform (no collapse face in 20 measured pairs; cert9: 67.04/64.15)** |
| Temperature sweep agg (t0.2/0.4/0.6/0.8 steady) | **214.9 / 215.0 / 222.3 / 224.7 tok/s** |
| Alt sampling face (t0.7/tp0.80/pp1.5) agg / solo | **271.5–281.3 / 82.7–92.1 tok/s** |
| TTFT solo 8k / 16k / 32k / 64k | default posture **3.76 / 7.90 / 16.93 / 38.93 s** (cert9 gate — P6.20 L3 off-thread store; cert10 same-day control in-family); **opt-in `CCL_SYCL_ALLREDUCE_FP8=1` on cert10: 3.27 / — / 16.29 / 35.73 s** (r99 A/B/A/B; every face beats the cert9 baseline — U8k −13 %, U64k −8.2 %, U32k −3.8 %) |
| Warm prefix re-hit 8k / 16k / 32k (t2) | **0.99 / 1.08 / 1.09 s** (warm-hit face 0.80–0.85 s; cert9 C32k 1.09; cert10 FP8=1 C32k 1.04) |
| MTP k3 acceptance, t0 | **83.7 %** strict-gate window / **85.9 %** battery window (mean accepted length 3.51 / 2.58; hard floor PASS; cert9 gate pos1 0.9258 / acceptance 0.837 / mean length 3.51 — byte-exact cert8 match) |
| MTP over no-spec (same-boot A/B) | **t0 2.35–2.47× · t1.0 1.84–2.09× · alt 2.42–2.53×** |

The dtype face behind these records is `fp8_e4m3` — the certified default
and still the fastest aggregate. The TurboQuant presets are **servable
since cert8** (P6.19 spec-verify fix — section below; re-gated clean on
cert9): they trade aggregate speed for KV capacity (up to 3.68× at the
cert8 gate; 3.72× at the P6.14 campaign's memory state).

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
corruption gate re-proven on cert8 **and re-gated on cert9** (zero garble
all dtypes; dominant identity hashes byte-identical across the cert8 and
cert9 gates — same variant families, the per-boot draft lottery only
chooses which identity face shows); **k8v4 carry proven on cert10** (r99
gate: essay/code/simple hashes byte-identical to the stock cert9 k8v4 leg,
the oneCCL overlay absent from the decision path); fp8 row is the certified
default.
Comprehensive per-dtype guide: `docs/CONFIGURATION-arc.md` "KV-cache
dtype"):

| Preset | KV capacity (cert8) | U32k / U64k TTFT (P6.14) | t1.0 agg (P6.14) | Status |
|---|---|---|---|---|
| `fp8_e4m3` (default) | 400,739 tok | 23.16 / 50.77 s¹ | **241.10** | **the certified default — fastest aggregate; the TQ fix never engages (control-proven)** |
| `turboquant_k8v4` | 519,972 tok (1.98×) | 15.29 / 33.03 s | 172.95 | **servable (cert8, re-gated cert9)** — spec tps solo 54.4 / agg8 236.0 split (cert9 gate: 54.6 / 236.2); **285.4 MQ mode** (cert9 annex; cert8 284.8 — +21 %, experimental — `VLLM_ARC_TQ_VERIFY_MODE=mq`) |
| `turboquant_4bit_nc` | 748,549 tok (2.86×) | 15.91 / 34.02 s | 134.42 | **servable (cert8, re-gated cert9)** — zero garble both gates; identity face varies per boot (cert8 cleanest, cert9 essay coherent variants — same dominant hash `56a2e47a4325` both) |
| `turboquant_k3v4_nc` | 843,202 tok (3.22×) | 15.48 / 33.78 s | 137.11 | **servable (cert8, re-gated cert9)** — zero garble both gates; sporadic t1-seed pairs on TQ boots (r84b re-char: t0 boot-stable, variant set shared with cert8 — dominant `8ccfa3b73aa1` both) |
| `turboquant_3bit_nc` | **963,764 tok (3.68×)** | 16.02 / 35.74 s | 171.32 | **servable (cert8, re-gated cert9)** — widest t0 variant drift (8 V levels; coherent output, not corruption; same character both gates) |

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
SHA=$(git rev-parse rel/v0.30.0-b70-cert10)      # the certified commit
git archive --format=tar -o /tmp/fork.tar rel/v0.30.0-b70-cert10
scp /tmp/fork.tar root@$RIG:/root/build/
ssh root@$RIG 'tar -xOf /root/build/fork.tar tools/build_image.sh > /root/build/build_image.sh'
ssh root@$RIG bash /root/build/build_image.sh /root/build/fork.tar $SHA vllm-arc:v0.30.0-b70-cert10
#    -> wheel_guard PASSes (4/4 md5s), IMAGE_MANIFEST records fork_sha == $SHA
#       (verify: docker run --rm --entrypoint cat
#        vllm-arc:v0.30.0-b70-cert10 /etc/vllm-arc/IMAGE_MANIFEST)
#       ~15 min cold; ~1 min when only the source changed (stack layers
#       cached — layer-order law in the Dockerfile).
#
#    cert10's vllm_gaudi tree is byte-identical to cert9's — building the
#    tag gives cert9's engine; the P6.24 oneCCL overlay (fp8-wire
#    allreduce, default-off) rides on top as a two-file lib overlay built
#    by external_patches/oneccl (README + tools/build_oneccl_arc.sh;
#    recipe + provenance there; the shipped image's overlay sha is pinned
#    in its IMAGE_MANIFEST row oneccl=2022.0.0+arc1). EXACT-image
#    reproduction = that overlay step (the LIB_SHA-gated cert10.Dockerfile
#    recipe, oneccl README "Build law"; the 210 MB fat lib is sha-pinned,
#    NOT tracked — see its ARTIFACT note for the durable homes).
#
#    Tag lineage (full table: docs/PERFORMANCE-arc.md "Image lineage"):
#    cert10 = current (P6.24 oneCCL fp8-wire overlay: opt-in e4m3 AR
#    1.65× at the dominant size, every p65 TTFT face beats cert9).
#    Superseded: cert9 (P6.20 prefill host-path: L3 off-thread LMCache
#    store default-ON + L6 dispatch-late + step_trace diagnostics;
#    U32k/U64k/U8k TTFT wins; TQ dtypes re-gated clean), cert8 (P6.19 TQ
#    spec-verify fix — all four turboquant_* dtypes serve clean; fp8 TTFT
#    faces carry the pre-L3 store path), cert7 (its P6.18 refusal is
#    superseded by the P6.19 fix), cert6
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
  --entrypoint bash vllm-arc:v0.30.0-b70-cert10 -c "sleep infinity"

# 3. Serve — the certified line (P6.17: probabilistic drafting = the serve
#    script's default posture), byte-identical to
#    docs/CONFIGURATION-arc.md Step 2 (that file is canonical; per-var
#    semantics there + docs/envs.md). The rig pattern: script + docker cp
#    + exec; logs land in /root/serve_full.log.
#    (Best-TTFT opt-in on cert10: add export CCL_SYCL_ALLREDUCE_FP8=1 to
#    the env block — first bullet in Optional postures below.)
cat > serve_k3.sh <<'EOF'
#!/bin/bash
# --- serve environment: 23 of the certified 25-var chain (2 set at boot) ---
export PYTHONHASHSEED=0                 # init determinism
export ZE_AFFINITY_MASK=0,1             # one Arc Pro B70 per socket
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd        # pairs with --privileged (Step 1)
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
export LMCACHE_TRACK_USAGE=false        # P6.3: usage telemetry is default-ON;
                                        # this rig serves offline (pairs with
                                        # the e04 wheel patch gating the
                                        # per-request usage-context creation)
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
#    cert9 gates; ~290 s at the cert10 gate: model load + graph capture)
curl -s -o /dev/null -w "%{http_code}\n" -m 5 http://127.0.0.1:8000/health
```

Optional postures (all documented in full in
[`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md)):

- **Best TTFT / prefill — the recommended perf opt-in (P6.24, cert10)** — one env line in the serve block: `export CCL_SYCL_ALLREDUCE_FP8=1` (e4m3-wire allreduce on the dominant prefill ARs — every p65 TTFT face beats cert9: U8k −13 %, U64k −8.2 %, U32k −3.8 %, C32k −4.6 %; decode/spec neutral, task-level quality parity; silent no-op on images without the overlay; CONFIGURATION "Best-performance postures")
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

- [`docs/STATUS-arc.md`](docs/STATUS-arc.md) — release detail + the full phase/improvement-lane narrative (the front-page bullets above link here)
- [`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md) — how to run the certified posture (the canonical serve block, knob table, posture laws)
- [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md) — measured numbers: **current certified record (P6.17 prob-lane campaign on cert6, re-verified at the cert7–cert10 release gates; the P6.24 fp8-wire opt-in faces first), the comprehensive dtype faces (fp8 + all four TurboQuant presets + MQ experimental mode), the posture matrices, then pre-P6.17 history (posture-labeled)
- [`docs/TESTING-arc.md`](docs/TESTING-arc.md) — 14 test suites, gates, validation record
- [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) — swap an incumbent stack for this one (boot, verify, rollback) + the evidence table
- [`docs/envs.md`](docs/envs.md) — complete env/knob control surface + naming map

## Repository model

- **`main` is linear and protected**: changes land as squash commits from per-item lanes (`feat/arc-pNN-*`); direct pushes, force pushes, and deletions are blocked; PRs are required.
- **Certified releases are annotated tags** pointing at the certified `main` commit — current: `rel/v0.30.0-b70-cert21` (P6.31 online-MXFP8 W8A16 route on the cert20 engine: arc5 `vllm_xpu_kernels` wheel 0.1.14.1+arc5 (adds `_xpu_C.fp8_gemm_w8a16_mxfp8` — bf16 acts × f8 weights, group-32 e8m0 weight scales, no per-call act quant; also carries the W4A8 probe op `_xpu_C.fp4_gemm_w4a8`, a banked oneDNN-refused closed-negative) + `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` reroute behind `ARC_MXFP8_MODE` + the `mxfp4_w4a16.py` w4a8 probe branch + the platform entry, via `tools/cert21.Dockerfile`; image-validated (r124/r124b on the same bytes, r125 gate): qwen online mxfp8 **U8k 3.61 / U32k 17.07 s, decode 53.8–55.6 / 265.9 agg8 tok/s = 1.37–1.55× over stock W8A8**, spec 0.9804; gemma **1.57–1.85× over stock** at `ARC_MAXLEN=40960` (capacity law); checkpoint-fp8 stays the perf default, mxfp4-w4a16 the superior online lane, by adjudication; three-lane gate green = fp8 champion + w4a16 carry + the new lane); superseded: `rel/v0.30.0-b70-cert20` (P6.30 online-MXFP4 W4A16 route on the cert19 engine: arc4 `vllm_xpu_kernels` wheel 0.1.14.1+arc4 (adds `_xpu_C.fp4_gemm_w4a16` — bf16 acts × f4 weights, group-32 e8m0 weight scales → `jit:gemm` DPAS via weight decomposition) + `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` reroute behind `ARC_MXFP4_MODE` via `tools/cert20.Dockerfile`; image-validated (r120c/r120g on the same bytes, r121 gate): qwen online mxfp4 **U8k 3.66 / U32k 17.12 s, decode 62.5–65.2 / ~305 agg8 tok/s — superior to online mxfp8 on every axis (1.47–1.75×)**, 1.74–1.87× TTFT over the W4A4 control, spec 1.0000; gemma 1.90–2.21× over W4A4; qwen fp8 discriminator in-band = incumbent lanes untouched); superseded: `rel/v0.30.0-b70-cert19` (P6.29 sliding-window FA2 route on the cert18 engine: gemma-4's 50/60 sliding hd256 layers dispatch to the same ESVD FA2 kernel windowed — 18–25x microbench k-independent (r109a/r109a3), image-validated face8k 4.9 / face32k 17.2 (qwen-parity) / face120k 110.9 s, decode 28.4/195 tok/s (r111); qwen3.8 discriminator in-band; `ARC_ATTN_FA2=0` disables both routes); superseded: `rel/v0.30.0-b70-cert18` (P6.28 FA2 route on the cert17 engine: gemma-4's hd512 full-attention layers dispatch to the ESVD FA2 kernel — 34x microbench (r108ac/r108ad), image-validated 2.2–16.3x on the faces + 6 % decode (r108ae); qwen3.8 discriminator marker-0 green — the incumbent lane untouched; `ARC_ATTN_FA2=0` escape hatch); superseded: `rel/v0.30.0-b70-cert17` (P6.26 MXFP4 lane: `quark` + `supports_mx` on `ArcXpuPlatform` — overlay on cert10, all prior bytes carry; fp8 lane behavior identical, r105t/t2 same-day face-for-face proof); superseded: `rel/v0.30.0-b70-cert10` (P6.24 oneCCL fp8-wire overlay: same-version 2022.0.0 rebuild + e4m3-wire large-ring allreduce, default-OFF env `CCL_SYCL_ALLREDUCE_FP8`; opt-in 1.65× at the dominant AR and every p65 TTFT face beats cert9 — cert11's engine + overlay bytes carry unchanged), `rel/v0.30.0-b70-cert9` (P6.20 prefill host-path: L3 off-thread LMCache store default-ON + L6 dispatch-late + `step_trace` diagnostics — cert10's engine is byte-identical), `rel/v0.30.0-b70-cert8` (P6.19 TQ spec-verify fix — fp8 TTFT faces carry the pre-L3 inline store path), `rel/v0.30.0-b70-cert7` (cert6 engine + the P6.18 TQ refusal — superseded by the P6.19 fix: its refusal is lifted on cert8 and the TQ dtypes serve clean), `rel/v0.30.0-b70-cert6` (fp8 behavior byte-identical; its servable `turboquant_*` dtypes carried the P6.18-corrupted spec-verify path), `rel/v0.30.0-b70-cert5` (identical default behavior, lacks the P6.14 TQ-aware store hunk), `rel/v0.30.0-b70-cert4` (identical default behavior, lacks the default-OFF P6.13 slice hunk), `rel/v0.30.0-b70-cert3` (identical default behavior, lacks the p65 deferral hunk), `rel/v0.30.0-b70-cert2` (0.5.5-era lmcache line — valid, older), `rel/v0.30.0-b70-cert` (predates the r46u glue-alias fix — never build from it). The Docker image embeds the fork SHA in `/etc/vllm-arc/IMAGE_MANIFEST` for A/B verification.
- **Evidence is in-tree**: every validation leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT + raw outputs) — **115 evidence dirs, 102 VERDICTs of record**, P0 through P6.
- **No secrets or real host addresses in-tree (publishing-hygiene law)**: tracked == pushable, so credentials, rig/host addresses, and private identities never enter tracked files; banked logs carry redacted host addresses (each banking-time scrub documented in its `results/REDACTION-*.md` note). `tools/leak_sweep.sh` must exit 0 before every push.
- The `feat/arc-*` branches are kept deliberately: squash-merging leaves the per-item development history nowhere else.

## Upstream & license

Fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) @ tag `v0.30.0` (the vLLM hardware plugin for Intel® Gaudi®). Upstream's README and docs for the base plugin: [vllm-gaudi docs](https://vllm-gaudi.readthedocs.io/en/latest/index.html). License: Apache-2.0 (see [LICENSE](LICENSE)).
