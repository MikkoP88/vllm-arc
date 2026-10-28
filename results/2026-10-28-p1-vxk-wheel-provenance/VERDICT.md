# VERDICT — plan §6 vxk wedge-site audit: arc image wheel = the v88-law fixed wheel, device-gated CLEAN (seeded + noseed); stock PyPI 0.1.14.1 FAILS the same gate; ONE 9th bare site survives in our build (init-state read path) — arc4 owed before P3

## What the arc image actually runs

`vllm-arc:v0.30.0-b70-p12b` (and every earlier arc image) ships
`vllm_xpu_kernels-0.1.14.1+arc3` — a LOCAL rebuild, not the PyPI wheel:

- Provenance (arc image dist-info `direct_url.json`):
  `file:///wheels/arc/vllm_xpu_kernels-0.1.14.1+arc3-cp312-cp312-linux_x86_64.whl`,
  sha256 `b4a6b2018b170aeff258b63ab8d82aaa34a0fe313ff05d196593f4aaaa6718ed`.
- Builder: rig container `vxk-arc3-builder` (intel/omix:0.2.0-devel = oneAPI
  DPC++ 2026.0.0 — generation-matched to the venv runtime intel_sycl_rt
  2026.0.0; arc1 failed so.8-vs-so.9, arc2 failed `reportAdjustedRange`
  undefined-symbol, arc3 linked clean — compiler generation MUST equal
  runtime generation).
- Source tree: `/root/build/vxk-arc-src` = upstream v0.1.14.1 lineage +
  `patch_vxk_int64fix.py` (adapted wedgefix-v88 port, line D donor kit,
  read-only import — fork law held: files, never git).
- Base carries the #438/#439 v_head_id guard
  (`chunk_gated_delta_rule_kernels_xe2.hpp:80 if (v_head_id >= num_v_heads)
  return;`) — the dual-B70 TP2 GP-fault/corruption fix shipped v0.1.12.

## The §6 audit matrix (rig evidence, 2026-10-27 build logs re-verified 2026-10-28)

| Wheel | Sites patched | Device gate (v88 repro, cap727.pt, LO+HI+BND) |
|---|---|---|
| stock PyPI 0.1.14.1 | 0 — **never upstreamed** | **FAIL** (gate6_bare: REPRO_EXIT=1, dmesg engine resets 1→2) |
| 0.1.14.1+arc2 | 8 | FAIL (undefined symbol at import — toolchain, not sites) |
| **0.1.14.1+arc3 (OURS)** | **8 of 9** | **CLEAN WHEEL** seeded AND noseed (gate4/gate5/gate_fixed*) |

Plan §6's "expected outcome: audit clean, pin used as-is" branch is REFUTED —
stock 0.1.14.1 is still vulnerable; the contingency branch (v88 patcher +
torch-matched rebuild + device gate) was already executed correctly at image
build time. This leg banks that evidence (it lived only in /root/build logs).

## NEW FINDING — 9th bare site survived the arc3 patcher

`patch_vxk_int64fix.py` matches the variable name `states_id` (6 sites in the
0.1.14.1-refactored tree). Upstream's refactored `causal_conv1d.hpp` renamed
the init-state path variable to `state_id`, and line 684 still computes:

```cpp
? conv_states + state_id * conv_states_stride_0   // BARE — current tree
```

(the `.vxki64bak` backup confirms it was bare pre-patch too; v88's original
4th site in this file was `init_state_id * ...` — same path, renamed.)

- Exposure: init-state READ for sequences resuming from a cached mamba state
  (prefix-cache hits / chunked-prefill continuations) at pool id ≥ 4097 —
  reachable once mamba_pool_ratio 2 lands (P3). NOT the defect-b mechanism:
  defect-b flips at the FIRST request with pool ids ≈ 0–2, an order of
  magnitude below the int32 wrap range (id ≥ 4097 at stride 524288).
- Action owed: arc4 wheel = arc3 + widen line 684 (`state_id`), re-run both
  device gates, re-audit with a name-agnostic pattern
  (`conv_states + \w+ * conv_states_stride_0`). P3 blocker, not P2.

## Consequence for defect-b (r3–r15 chain)

The kernel-wheel lane for defect-b is CLOSED on current evidence: our wheel
has the int64 fix (gated), the v_head_id guard, and every GDN fix in
vllm-xpu-kernels releases through v0.1.14.1 (latest; no newer release
exists). Whatever poisons the stream at first prefill, it is not a missing
fix from the kernel repo's release history — remaining candidates: a
TP2-geometry bug never exercised upstream (arcitech's cookbook
partial-final-group fix class; their clean B70 run is SINGLE-GPU), or the
kernel × graph-machinery interaction itself. r16 (capture real serve kwargs
→ standalone replay ± graphs) is the decisive discriminator.

Raws: build_vxk_wheel_arc{,2,3}.sh, vxk_arc3_inner.sh, patch_vxk_int64fix.py,
vxk_gate{3,4,5,6_bare}.log, tree_audit.txt (live re-grep of the source tree
2026-10-28), wheel sha256 cross-checked against the running image's
direct_url.json.
