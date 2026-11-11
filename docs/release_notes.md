# Release notes — vllm-arc certified releases

Index of the certified releases, newest first. This file follows the
upstream vllm-gaudi release-notes practice: **one short section per
release** — tag, image, what changed for the user, and where the
evidence lives. The detailed per-release entries (full change lists,
fork SHAs, wheel pins, gate numbers) are in
[`../CHANGELOG.md`](../CHANGELOG.md); superseded doc content is
archived per-doc under [`history/`](history/). The current certified
release is **`rel/v0.30.0-b70-cert23`** (`vllm-arc:v0.30.0-b70-cert23`).

Every release below is based on upstream vllm-gaudi **v0.30.0** with
the pinned wheel stack (`tools/wheel_guard.sh`; per-release wheel shas
in the CHANGELOG entries) and ships the full certified posture for
qwen3.8-27b-fp8 on dual Intel Arc Pro B70, TP2. The upstream basis
itself (what vLLM 0.30.0 + the Gaudi plugin stack provide) is documented
in the upstream per-version detail files
[`release_notes_v0.30.0.md`](release_notes_v0.30.0.md) and predecessors.

---

## rel/v0.30.0-b70-cert23 — graph-exec double-buffer (2026-11-10)

Current certified release. cert22 engine + one patch module: every XPU
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
