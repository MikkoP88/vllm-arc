# Defect-b deep analysis + solution-source research + plan rev-4 analysis
Owner directive 2026-10-28: "do realy deep analyze and research of issue and
code, also check do solution live in any sources" + "VLLM-ARC-MASTER-PLAN
has been updated, do deep analyze and research of plan".
Branch feat/arc-p12b-graph-stream-hygiene @ 50c41b45. All legs banked in
results/; rig raws in /root/build/legs/arc-*.

---

## 1. THE DEFECT (current model, evidence-weighted)

**Symptom**: with XPU graph machinery live, the first real request's GDN
prefill leaves the device-default stream permanently reporting "recording to
a command graph" with no capture open (`torch.xpu.is_current_stream_capturing()`
True; host-side SYCL queue-state query — pytorch RFC #162143). Every later
request 500s; integrity guard kills the worker (fail-loud).

**Model v2** (r13+r14 refined): the stock **Qwen GDN XPU prefill path**
(`qwen_gdn_linear_attn.forward_xpu` → `torch.ops.vllm.gdn_attention_core_xpu`
→ `torch.ops._xpu_C.gdn_attention` SYCL chunk kernels) × **live XPU graph
machinery**. `forward_xpu` (qwen_gdn_linear_attn.py:976) calls the SYCL op
UNCONDITIONALLY — no backend routing exists on XPU (verified in the arc
image; `_resolve_gdn_prefill_backend`'s "triton" resolution only selects
flashinfer/cutedsl on CUDA).

**Model v3 (r16, FINAL — supersedes v2 and plan Appendix C)**: the wedge
is **inside the first eager LARGE TP allreduce** (the embedding AR of the
first real request; entry pre_rec=False → exit pre_rec=True, both ranks,
banked bracket), executed after decode-graph capture warmup on vllm 0.30
XPU TP2, which reports healthy without any real-size eager collective
(no warmup generate in 0.30). A small first request (≤ ~512 tok, AR <
8 MB) initializes cleanly and PERMANENTLY vaccinates the engine for later
large requests. The GDN kernel is fully exonerated: real-kwargs standalone
replay (captured hidden/weights/metadata/pool geometry) is clean ± graph
machinery — r15's synthetic DEVICE_LOST was a kwargs artifact. See
results/2026-10-28-p2-graph-p12b-r16-kwcapture/VERDICT.md.

### Eliminated suspects (r3–r15, each banked)

| Suspect | Leg | Result |
|---|---|---|
| Capture-stream placement / side-stream | r3–r9 | flip persists in every posture |
| UR immediate command lists | r10 | UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 — still flips |
| Async scheduling | r11 | still flips |
| Prefix caching write | r7 | ARC_PREFIX=0 — still flips |
| Compilation (inductor piecewise) | r13 | mode=NONE eager — STILL FLIPS; eager onset stack names in_proj_ba gemm |
| Graph posture (mode/sizes/pool count) | r14 | W14's exact FULL_AND_PIECEWISE + [4,8,16,32,64] + max 64 — STILL FLIPS |
| Kernel wheel missing released fixes | this session | 0.1.14.1 is the LATEST release; arc3 wheel carries all of them (below) |
| Decode path | r15 | decode-shaped call clean standalone |
| GDN kernel with REAL serve kwargs | r16 | CLEAN standalone ± graphs — kernel exonerated |
| Projection gemms / graph replays / 2-proc xccl chain | r16 | all clean standalone (wedge needs vllm runner context) |
| SYCL_CACHE_PERSISTENT=0 (arcitech env) | r16 leg B | large-as-first STILL flips — env not the cure |
| **First eager large AR (embedding, 8.5 MB)** | **r16 v7** | **THE WEDGE: entry clean → exit recording (both ranks)** |

### The three onset stacks converge

r12 (compiled FDO), r13 (eager mode=none), r14 (compiled FAP) — all land in
the GDN linear-attention prefill region: the `in_proj_ba` gemm (weight
(5120,48)) immediately adjacent to the SYCL `gdn_attention` call. **r16
retrospective: these thread positions were sampling artifacts** — the
10 ms watchdog catches the forward thread wherever it is when the
already-wedged queue is first queried; the actual wedge (the embedding
allreduce) precedes all of them.

## 2. WHEEL FORENSICS (closed this session — results/2026-10-28-p1-vxk-wheel-provenance)

- Our image runs `0.1.14.1+arc3` — a LOCAL, toolchain-generation-matched
  rebuild (oneAPI 2026.0 = venv runtime; arc1/arc2 failures banked) of the
  v0.1.14.1 lineage + the wedgefix-v88 int64 patcher (line-D donor kit,
  imported as FILES — fork law held). sha256 b4a6b201… cross-checked against
  the image's direct_url.json.
- **v_head_id guard present** (#438/#439, shipped v0.1.12) — the dual-B70
  TP2 GP-fault fix.
- **Device gates PASSED**: v88 repro on capture pin cap727.pt → CLEAN WHEEL,
  seeded AND noseed (gate4/gate5); stock PyPI 0.1.14.1 FAILS the same gate
  (gate6_bare: exit 1, dmesg engine reset) — the int32 pool-offset bug is
  real on stock and never upstreamed. Plan §6's optimistic branch ("audit
  clean, pin as-is") is refuted; the contingency branch was already
  correctly executed at image-build time. **P1 gate: satisfied, now banked.**
- **NEW: 9th bare site** — `causal_conv1d.hpp:684` uses `state_id`
  (singular; patcher regex matched `states_id` only): the init-state READ
  path (prefix-cache resume / chunked-prefill continuation). Exposed at pool
  id ≥ 4097 → **arc4 wheel owed before P3** (mamba_pool_ratio 2 pushes rows
  past 4096). NOT defect-b's mechanism: the flip lands at ids ≈ 0–2, two
  orders of magnitude below the int32 wrap range.

**Kernel-wheel lane for defect-b: CLOSED.** No released fix exists that we
lack; no newer release exists to take.

## 3. SOLUTION-SOURCE INVENTORY (external research, all read this session)

1. **vllm-xpu-kernels releases** (v0.1.12 → v0.1.14.1): GDN fixes #438/#439
   (v_head_id OOB — dual-B70 TP2!), #510 (mixed spec/non-spec split), #378
   (tiled conv1d prefill), #379 (l2_norm/chunk_fwd_o prefill), #411
   (write-after-read race NaN), #399 (GQA-ratio-3 z-drop). All in our base.
2. **vllm#49924** — the mechanism CLASS precedent: `chunk_prepare_kernel`
   computed v_head_id from HW sub-group counts; non-multiple geometry → OOB
   write past scratch → **silent corruption of whatever the allocator placed
   next** (victim varies run-to-run); on OUR exact hardware (dual B70 TP2).
   Kernel-side fix + guard shipped v0.1.12. Proves this kernel family HAS
   shipped allocator-adjacent OOB writes on our geometry before.
3. **vllm#62781** — GDN fp16 overflow class on Qwen3.5-family XPU
   (#54146/#62116/#62119); pending vllm fixes don't reach the XPU path;
   reporter's local fix = hand-written SYCL GDN kernel. Cites the vendor's
   **100% needle on Arc B70 at vLLM 0.30.0 + kernels 0.1.14.1 + two
   patches** — the base stack is capable.
4. **arcitech-psp/vllm-xpu-arc (branch holo4-v030)** — the vendor recipe,
   digested from their README + HF card (Holo4-27B = Qwen 3.8 dense 27B
   arch): stock vLLM 0.30.0 + stock kernels 0.1.14.1 + torch 2.13.0+xpu on
   Arc Pro B70, **graphs ON, MTP, prefix caching — 100% needle (21/21),
   stable**. Their delta over stock:
   - **SYCL persistent device cache OFF** (`SYCL_CACHE_PERSISTENT=0`): with
     the image default (1) the persistent cache **segfaulted on the FIRST
     REQUEST** (a first-request × kernel-JIT-cache interaction — our onset
     window);
   - guarded exact-sort fallback for XPU BF16 top-k (stock crash they
     retained a log of);
   - **cookbook partial-final-group fix** (Intel B70 cookbook, SergiioB) —
     included in their wheel build;
   - mixed-GDN split — already in our stock 0.30.0 (#510 mirror).
   **Crucially: their clean run is SINGLE-GPU (no TP)**. Our TP2 halves the
   per-rank head counts (24 v / 8 k) — exactly the sub-group-arithmetic
   regime #49924's bug lived in. Their evidence proves the stack, not our
   geometry.
5. **Radiance W14 (the port blueprint — sanctioned source)**:
   `ops/hpu_gdn_pytorch.py` (1464 lines, read in full) + the `qwen3_5.py`
   tail-swap (`QwenGatedDeltaNetAttention → HPUGatedDeltaNetAttention`).
   Decode: fully vectorized single-token fast path (shape-only detection, no
   host sync), fp32 state math, `ssm_state_indices` + `ssm_store_indices`
   pool handling. Prefill: 3-stage chunked GDR (batched phases A/A', Python
   loop only for the state recurrence), Neumann-14 approximate tril solve
   (exact-solve env for accuracy debugging; residual tuned per model —
   Qwen3.5-9B evidence; ours needs validation). **Speculative path:
   NotImplementedError (phase 1)** — a FULL swap breaks MTP ⇒ lane A must
   be PREFILL-ROUTING (pytorch prefill + SYCL decode/spec), or the ESIMD
   lane (plan §3 #16, flash-next-shaped — needs port+revalidation).

## 4. FIX LANES (recommendation — REVISED by r16)

| Lane | Content | Sanctioned? | Risk |
|---|---|---|---|
| **W (r16 outcome): boot-time 2-stage warmup** | platform patch: after graph-capture warmup, internally run small (≤512 tok) then max-batch prefill before serving traffic; fail-loud at boot | YES — our own platform layer, evidence = r16 controlled pair | warmup must not race external traffic (run inside engine readiness); re-validated by §10 P2 gates |
| **B: kernel wheel** | take a newer/fixed release | CLOSED — none exists; kernel exonerated anyway by r16 real-kwargs replay | — |
| **A: radiance pytorch-GDN PREFILL routing** | forward_xpu routes prefill tokens to the ported chunked-GDR; decode/spec stay on SYCL | sanctioned, but DEMOTED to perf-only (it cannot be the defect-b fix — the defect is not in the GDN path) | TTFT-parity gate; Neumann accuracy; mixed-batch split |
| **D: arcitech env** | SYCL_CACHE_PERSISTENT=0 | REFUTED as defect-b fix (r16 leg B: large-as-first still flips); arcitech stays evidence-only | — |
| **C: prefill→fla-triton routing** | patch forward_xpu to call forward_native | moot for defect-b | — |

**Recommended sequence**: lane W warmup patch (p12b continuation) →
validate large-as-first on the patched boot → P2 gates (graphs+MTP, TPOT,
acceptance) → arc4 wheel (9th site) before P3 → upstream issue text now
describes model v3 (first eager large AR after decode-graph capture
warps the default queue; owner sign-off before filing).

## 5. PLAN REV-4 ANALYSIS (owner's 2026-10-28 update, mtime 12:11)

Read in full (1019 lines). What changed and its impact:

1. **§6 vllm-xpu-kernels = the wedgefix-v88 law is now MANDATED** (8-site
   audit, v88 patcher contingency, "mamba_pool_ratio 2 … dense qwen3.8-27b
   IS exposed"). Impact on us: the audit is now a **P1 gate** — satisfied by
   the arc3 evidence (banked this session as
   results/2026-10-28-p1-vxk-wheel-provenance), with ONE open item the plan
   did not anticipate: the 9th site (upstream rename `state_id` vs the
   patcher's `states_id` pattern) ⇒ **arc4 before P3**. The plan's "expected
   outcome: audit clean" is refuted by gate6_bare — stock fails; our
   contingency-built wheel is the reason the image survives at all.
2. **§1/§2/Appendix J registration correction** (tag setup.py:66-72
   entry-points inherited; pyproject has none). No action — matches what we
   run; removes any residual registration work item.
3. **§3 #23 correction** (qwen3_5.py model class = upstream tag's own,
   radiance zero-delta; only MTP head arrives) — matches the r-series
   forensics; constrains lane A to the ATTENTION-SWAP pattern (radiance's
   tail-swap), not a model-file port.
4. **Appendix C defect-b section** was folded to the **r12 model** ("compiled
   prefill × live capture machinery"; ranked fix #1 = mode:none "IS the
   champion posture") — **now STALE on two points**: r13 refuted fix #1
   (eager mode=none still flips) and r14 refuted posture-sufficiency (FAP +
   W14 sizes flips). The banked VERDICTs (73a8f05f, 50c41b45) supersede;
   model v2 + the lane table above are the current truth. The plan's
   "Upstream gap" note (0.30 XPUPlatform never gates FULL_DECODE_ONLY ×
   compiled prefill) remains valid as an upstream-issue candidate — but the
   issue text must now describe the kernel-path mechanism, not the compile
   mechanism (owner sign-off still required before filing).
5. Unchanged and still governing: scope law (Appendix I §1 — arcitech is
   OUTSIDE the sanctioned source list; evidence-only), §0 superiority
   contract, §10 gates, §12 order, banking directive.

## 6. RIG STATE + NEXT

- r16 COMPLETE (mechanism identified; see
  results/2026-10-28-p2-graph-p12b-r16-kwcapture/). arc-test Up (dead v7
  engine), vxk-arc3-builder Up (arc4 build host), radiance-test stopped —
  **W14 restore owed at session end.**
- Queue: lane W warmup patch (p12b) → patched-boot large-as-first
  validation → P2 gates → arc4 wheel (9th site) before P3. Upstream issue
  (model v3 text) after owner sign-off.

## Sources

- vllm-xpu-kernels releases: https://github.com/vllm-project/vllm-xpu-kernels/releases
- vllm#49924 (GDN OOB corruption, dual B70 TP2): https://github.com/vllm-project/vllm/issues/49924
- vllm#62781 (GDN fp16 overflow + vendor recipe): https://github.com/vllm-project/vllm/issues/62781
- arcitech fork (holo4-v030): https://github.com/arcitech-psp/vllm-xpu-arc/tree/holo4-v030
- arcitech HF card: https://huggingface.co/arcitech-psp/Holo4-27B-AutoRound-W4A16
- pytorch RFC #162143 (stream-capture query semantics): https://github.com/pytorch/pytorch/pull/162143
- wedgefix-v88 donor kit (line D): C:\Projects\vllm-iexperimental\vllm\patches\prod\external-deps\wedgefix-v88\
