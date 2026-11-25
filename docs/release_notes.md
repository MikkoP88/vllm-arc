# Release notes — vllm-arc certified releases

Index of the certified releases, newest first. This file follows the
upstream vllm-gaudi release-notes practice: **one short section per
release** — tag, image, what changed for the user, and where the
evidence lives. The detailed per-release entries (full change lists,
fork SHAs, wheel pins, gate numbers) are in
[`../CHANGELOG.md`](../CHANGELOG.md); superseded doc content is
archived per-doc under [`history/`](history/). The current certified
release is **`rel/v0.30.0-b70-cert28`** (`vllm-arc:v0.30.0-b70-cert28`).

Every release below is based on upstream vllm-gaudi **v0.30.0** with
the pinned wheel stack (`tools/wheel_guard.sh`; per-release wheel shas
in the CHANGELOG entries) and ships the full certified posture for
qwen3.8-27b-fp8 on dual Intel Arc Pro B70, TP2. The upstream basis
itself (what vLLM 0.30.0 + the Gaudi plugin stack provide) is documented
in the upstream per-version detail files
[`release_notes_v0.30.0.md`](release_notes_v0.30.0.md) and predecessors.

---

## rel/v0.30.0-b70-cert28 — P6.47: seqused-tightened split clamp; the MX walk line closed (2026-11-24)

Current certified release. The arc7mx build28 clamp wheel
(md5-pinned, three installed-`.so` md5s asserted in-image):
`seq_num_kv_splits = min(num_kv_splits, max(1, windowed_k_blocks/4))` —
deep splits at short contexts no longer serialize tiles the attention
window cannot fill. **For the user**: the certified native KV-decode
lane (`ARC_KVDTYPE=mxfp8|mxfp4|mxk8v4` + `ARC_MXKV_NATIVE=1`) keeps
every correctness gate green (r233 §10 battery, 8 legs — qwen MX lanes
LONGRETURN/T1_SEED absolute PASS at native_lines=108, GEMMA_COMPARE
PASS) and gets the **first full shallow-through-long record**: native
mxfp4 ITL 43.4/45.2/48.5/52.2/58.1 ms at 2/4/8/16/32K, first shallow
rows for mxfp8 (44.4/46.5) and mxk8v4 (44.1/46.0), long-ctx rows
banked (r234).
The MX KV-dtype walk line is closed at cert28 — the ladder's
capture-point ledger is fully dispositioned by measurement (S1 knee
cured r185/r186 and re-confirmed on cert28 at 1.78–1.81
accepted/draft; aux/plan-build at noise; continuation-dequant
refuted); the always-loop directive continues on new evidence. Manifest
`cert28_fork_sha=43f84cf9` (the r231d evidence merge; close-out
commits are docs/banks only). Evidence:
`results/2026-11-24-p647-cert28-battery/`,
`results/2026-11-24-p647-r234-cert28-itl/`.

## rel/v0.30.0-b70-cert27 — P6.47: r225 wheel fixes; all three MX lanes native (2026-11-23)

Superseded by cert28 (2026-11-24).

cert26 + the r225-fixed `arc7mx` build25
wheel (md5-pinned + installed-`.so` md5 asserted in-image): four
kernel fixes — e2m1 subnormal mantissa decode, `sycl::exp2` split-KV
combine rescale, fp32 split partials, S==1 sentinel-LSE predicate —
plus the `ARC_MXKV_AB` diagnostic instrument in the overlay
(default-off). **For the user**: the certified native KV-decode lane
extends to **all three MX dtypes** —
`ARC_KVDTYPE=mxfp8|mxfp4|mxk8v4` + `ARC_MXKV_NATIVE=1`; the cert26
`mxk8v4` REP-LOOP refusal is eliminated (r228 battery LONGRETURN/
T1_SEED/longprobe CLEAN; offline replay 12/12 + 8/8 + 8/8 + 8/8 with
format-invariant full-population residual distributions). Native-lane
outputs drift vs cert26 by design (numerics fixes); generation gates
are per-boot self-consistency and all pass. The perf record is
**re-measured on cert27 itself** (r230: fp8 ctrl at 0.999–1.001× its
banked anchors; native mxfp4 +1.8–2.2% = the fix3 fp32 split-partials
cost; first native long-ctx rows for mxfp8/mxk8v4; agg8 m4q 0.86×
ctrl, native mkq +7.9% over its cert26 Triton lane). Evidence:
`results/2026-11-23-p647-cert27-battery/`,
`results/2026-11-23-p647-r229-format-replay/`,
`results/2026-11-23-p647-r230-itl-redisc/`.

## rel/v0.30.0-b70-cert26 — P6.47 Phase-8: arc7 MX-native decode line (2026-11-22)

Superseded by cert27 (2026-11-23). cert25zc content + the Phase-8 arc7
MX-native decode line: the `arc7mx` wheel (vllm_xpu_kernels
0.1.14.1+arc7mx, md5-pinned, BAKED) with the MX-native paged-decode
Xe2 kernel and the r219 C++ auto-split heuristic at the validated
optimum (`ARC_MXKV_NSPLIT` retired as posture — auto lands 128 by
itself), plus the fork overlay `native_verify` bridge. **For the
user**: `--kv-cache-dtype mxfp8|mxfp4` now decodes through the native
Xe2 kernel — pass `ARC_MXKV_NATIVE=1` (mxfp4 ITL 48.5/51.2/56.8 ms at
8/16/32K, −19.2% vs the prior posture; tps 65.4 @32K; battery agg8
0.84× ctrl, up from 0.74–0.78×). `mxk8v4` keeps the certified Triton
lanes (native mixed-fmt lane refused at the long-return gate —
deterministic REP-LOOP; diagnosis reopen path r224; refusal evidence
banked). Quality: qwen lanes absolute-clean 4/4; gemma comparative
PASS. Evidence: `results/2026-11-22-p647-cert26-battery/` + the
r216–r221 leg banks; details in the
[`../CHANGELOG.md`](../CHANGELOG.md) cert26 entry.

## rel/v0.30.0-b70-cert25zc — P6.47 Phase-7 MX decode closing rung (2026-11-19)

cert25zb engine + the Phase-7 closing rung of
the MX decode ladder: TILE_FLOOR launcher default 2 (`ARC_MXKV_V5_TFLOOR`),
G-adaptive mxfp4 verify-decoder dispatch (`ARC_MXKV_V5_G_MIN=8`), v7
research kernel dormant. MX lanes serve LMCache-free (Phase-6 auto-off
posture; ctrl unchanged). Headline: **mxfp4 n8 32K agg decode p50
1547→583 ms (−62%)**; m8q n1 32K ITL −10.4%. Existing deployments need
no action — the knobs ship as new defaults with certified-equivalent or
better decode, and the native `fp8_e4m3` path is untouched. The
remaining ~0.74–0.78× MX agg8 decode gap vs ctrl is structural (Triton
split-dot) and is the open Phase-8 arc7 MX-native kernel line. Full
battery evidence: `results/2026-11-19-p647-cert25zc-battery/` (qwen
absolute 4/4 + gemma comparative PASS; verdict re-derived under the
amended gate from banked markers).
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert25zb — P6.47 MX KV-cache dtype lane (2026-11-13)

cert23-lineage engine + the P6.47 lane:
`--kv-cache-dtype {mxfp4,mxfp8,mxk8v4}` (Triton MX store/decode in
`vllm_gaudi/xpu/patches/mx_kv/` + engine dispatch), with the K2
packed-fp16 dequant and K3 e2m1 fast-unpack lanes default-on
(bit-identical outputs; `ARC_MXKV_K2=0`/`ARC_MXKV_K3=0` revert).
Headline: **mxfp4 holds fp8-checkpoint solo decode parity (60.5–62.9
vs 60.1–61.8 tps) at 1.81× KV capacity** (721,994 vs 398,363 tokens);
mxk8v4 1.27×. The kernel ladder is measured-converged on all three
formats; the native `fp8_e4m3` bank is byte-identical across the bake
(r184 cross-cert), so existing deployments need no action — the lane
activates only when an MX dtype is selected. Interim chain (cert24
store-vstride fix, cert25z v5 decoder, cert25za v6 decoder) documented
in this entry's CHANGELOG section, never tagged. Evidence:
`results/2026-11-13-p647-cert25zb-certification/`.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert23 — graph-exec double-buffer (2026-11-10)

cert22 engine + one patch module: every XPU
graph captured ×2 into the manager's shared pool, replays alternate
execs — removes the torch-2.13 same-exec `XPUGraph.replay()` host block.
E2E-neutral at every measured posture (fp8-k3 94.5 vs 94.3 tok/s on/off),
+29 s boot, byte-identical semantics, `VLLM_ARC_GRAPH_EXEC_DB=0`
kill-switch. Existing cert22 deployments need no action.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert22 — skinny-M kernel family, banked OFF (2026-11-09)

arc6 wheel: three GEMV-regime weight-only GEMM kernels, correct and
faster in the eager regime, e2e-falsified under XPU-graph replay —
shipped default OFF (`ARC_SKINNY_M`). Engine otherwise carries cert21.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert21 — online-MXFP8 W8A16 route (2026-11-08)

arc5 wheel: `_xpu_C.fp8_gemm_w8a16_mxfp8` keeps activations bf16 against
f8 weights — online mxfp8 turns from the slow lane into a real one
(1.37–1.55× over stock W8A8 on every qwen face). `ARC_MXFP8_MODE` gate
knob. Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert20 — online-MXFP4 W4A16 route (2026-11-08)

arc4 wheel: `_xpu_C.fp4_gemm_w4a16` — online mxfp4 superior to every
online lane on every axis (1.47–1.75× over stock mxfp8; 1.74–1.87× TTFT
over the W4A4 control). `ARC_MXFP4_MODE` gate knob; with the later
probabilistic k5 posture this becomes the recommended decode lane
(123.9 tok/s, +52 %). Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert19 — sliding-window FA2 route (2026-11-07)

Wrapper dispatch of gemma-4's sliding-window hd256 layers to the
`vllm_xpu_kernels` ESVD FA2 kernel (18–25× prefill over the triton
path); gemma face32k reaches qwen parity (17.2 s). Gemma-4-31B is a
fully validated second model from this release line.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert18 — FA2 hd512 route (2026-11-07)

Wrapper dispatch of gemma-4's full-attention hd512 layers to the ESVD
FA2 kernel (34× over the triton path at production geometry).
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert17 — gemma-4 validation chain (2026-11-07)

cert11 engine + the three build-kit fixes that made gemma-4-31B
servable (this entry also covers the intermediate build-kit releases
**cert12** `a964e047` — B70 prefill launch-config table + warmup gate —
and **cert16** `5d895fef` — warm-gate v4, the triton cache-key fix that
closed the G2 long-prefill freeze; cert13–cert15 were process/kit
stepping stones inside the same chain). Adds the KVGATE boot gate +
`ARC_RPC_TIMEOUT=1200` to the certified gemma posture.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert11 — MXFP4/quark lane (2026-11-06)

Two-override `ArcXpuPlatform` patch (quark in `supported_quantization`,
truthful `supports_mx()`): pre-quantized MXFP4 checkpoints boot and
serve under the full certified posture — mxfp4 becomes the certified
weight-constrained lane (9.03 GiB weights/rank); fp8 stays the perf
default; fp8 bytes unchanged (proven face-for-face).
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert10 — oneCCL fp8-wire allreduce overlay (2026-11-05)

Patched-`libccl` overlay: `CCL_SYCL_ALLREDUCE_FP8=1` moves dominant
prefill allreduces to the e4m3 wire — the best prefill faces, opt-in
(U8k −13 %, U64k −8.2 % TTFT), decode/spec neutral, quality parity
proven. Default OFF. Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert9 — prefill host-path (2026-11-04)

LMCache off-thread store (`VLLM_ARC_LMC_ASYNC_STORE=1`, default ON):
the store's host wait moves off the engine thread — U32k TTFT −0.74 s,
U64k −1.64 s at zero cost. Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert8 — TurboQuant spec-verify FIX (2026-11-04)

Root-caused the TurboQuant corruption to three stacked spec-verify
defects (split-row dispatch, donor capture forcing, the MQ kernel's
SoA/AoS mismatch) and fixed them wheel-side (`tq_spec_fix`): every
`turboquant_*` preset serves clean again, all-dtype gate with the fp8
control. KV capacity up to 3.68× opt-in. The MQ experimental mode lands
behind `VLLM_ARC_TQ_VERIFY_MODE=mq` (+21 % agg8).
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert7 — TurboQuant interim refusal (2026-11-03)

Hard-refused the `turboquant_*` dtypes at boot pending the corruption
root cause (the honest interim posture; the full reproduction battery
and roundtest are banked from this release).
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert6 — TurboQuant presets (2026-11-02)

First certified TurboQuant battery: the five-dtype matrix (capacity,
TTFT, t0/t1.0 faces, bystander gaps) that remains the TQ record.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert5 — sliced final-span store, default OFF (2026-11-01)

`LMC_STORE_SLICE_TOKENS` ships default-OFF (experimental slice of the
deferred one-shot store; the gate record is banked).
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert4 — certified posture image (2026-10-31)

The first full certified-posture release: the frozen 25-var env chain +
CLI, the 12-cell load matrix, and the comprehensive battery of record
(soak GREEN, 4 h). Everything earlier was pre-certification hardening.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

## rel/v0.30.0-b70-cert3 / cert2 / cert — image-process milestones (2026-10-31 and earlier)

The build/boot/process milestones that led to cert4: wheel-pinning
guard, IMAGE_MANIFEST, container-freshness law, boot verification.
Full detail: [`../CHANGELOG.md`](../CHANGELOG.md).

---

## Serve-layer and analysis phases (no cert — engine bytes unchanged)

Between releases, serve-script postures and measurement campaigns land
without engine changes — probabilistic drafting (the certified default
posture), MTP-depth selection, the prefill/device-ledger closures, the
per-config performance reports, and the docs/history restructure. These
are listed per phase in [`../CHANGELOG.md`](../CHANGELOG.md) under
"Serve-layer and analysis phases".
