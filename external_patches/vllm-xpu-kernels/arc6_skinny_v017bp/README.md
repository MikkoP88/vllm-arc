# arc6 — skinny-M decode GEMM family + v0.1.17 Option-B backports (P6.33)

Sixth arc section of the `vllm-xpu-kernels` wheel lineage
(`0.1.14.1+arc6`). Built from the live post-arc5 tree with the same
builder toolchain (oneAPI 2026.0.0 == venv runtime, torch 2.13.0+xpu —
the arc1/arc2 ABI laws hold: compiler generation == runtime
generation, `.deps` kept).

## What arc6 adds

1. **`csrc/xpu/skinny_gemm/`** — the skinny-M decode GEMM family
   (`skinny_gemm_kernel.hpp` + `skinny_gemm_interface.cpp`): one kernel
   parameterized over weight format (f8 e4m3 / f8 e5m2 / packed f4
   e2m1) × scale mode (epilogue per-tensor / per-channel / in-kernel
   group-32 e8m0) × activation dtype (bf16 / f16), M ≤ 8. Design
   rationale + the r123 bytes-floor economics are in the kernel header
   comment and master plan §P6.32 item 5. Three torch ops with the
   stock signatures:
   - `skinny_fp8_gemm_w8a16(A, B, B_scale?, bias?)` (checkpoint-fp8
     champion classes)
   - `skinny_fp8_gemm_w8a16_mxfp8(A, B, B_scale, out_dtype?, bias?)`
     (online MXFP8)
   - `skinny_fp4_gemm_w4a16(A, B, B_scale, out_dtype?, bias?)`
     (online MXFP4)
   Python route: `vllm_gaudi/xpu/patches/skinny_m.py` (`ARC_SKINNY_M`,
   default on at M ≤ 8, chains after mxfp4/mxfp8, per-lane permanent
   fallback on any runtime failure — visibility law).

2. **W4A8 dispatch arms** (`onednn_ext.h`): two `case` arms for the
   arc5 enum values `f8_e4m3_mxfp4_bf16/_f16`. The arc5 wheel shipped
   the mappers but no dispatch arms, so the wrapper's own `default:`
   threw `"Only support int4 and fp8 gemm ..."` before oneDNN was ever
   asked — the r123 mis-attribution (master plan §P6.32 item 3; the
   lane is OPEN, not oneDNN-refused). r126 GATE-3 banks the answer.

3. **v0.1.17 Option-B backports** (`patches/`, csrc-only — tests/
   hunks dropped; apply order is load-bearing, 645's context contains
   611's lines):
   - `kit_611.patch` — FP8 block GEMM K==block_k routing (#611:
     `fp8_gemm_w8a8.h`, `fp8_gemm_w8a16.h`)
   - `kit_645.patch` — MXFP8 K==32 e8m0 → block-scale routing (#645:
     `fp8_gemm_w8a8.h`)
   - `kit_650.patch` — per-tensor fp8 quant scale rounding (#650:
     `quantization/fp8/fp8_quant.h`)
   - `kit_600.patch` — GDN ragged spec-token traversal (#600:
     `causal_conv1d.hpp`, `gated_delta_rule.hpp`,
     `gdn_attn_interface.cpp`)
   Deferred to the Phase-3 rebase: #658/#660/#683 (GDN
   causal-conv offsets / NULL_BLOCK / DeepSeek layout) — they refactor
   the `xe_2/chunk_causal_conv1d*` split absent from our 0.1.14.1
   base; hand-carrying the intermediate refactor chain is an unbounded
   cascade (§P6.32 item 8 future-release policy). Manifest law: when
   upstream supersedes, the backport drops from this kit.

## Rig flow (P6.33 rig sessions)

```
ship kit -> /root/build/arc6kit            (md5-verify; ship+launch in
                                             SEPARATE ssh calls)
python3 /root/build/arc6kit/apply_arc6.py  (dry-run validated locally
                                             against the fetched tree)
bash   /root/build/build_vxk_wheel_arc6.sh (v3 host-swap; arc5 kept at
                                             vxk-arc5-kept)
build arc6test image (arc6test.Dockerfile, FROM cert21)
r126_skinny_kbench.sh                       (GATE-1/2/3)
r127 e2e battery (both models, skinny legs vs banked refs)
cert22 close-out (tracked tools/cert22.Dockerfile + docs)
```

## Files

- `apply_arc6.py` — rig-side patcher (exact-match anchors, refuses on
  miss/double-apply; backports applied by `git apply` with marker-based
  idempotence)
- `csrc/xpu/skinny_gemm/` — kernel + interface sources
- `patches/kit_{611,645,650,600}.patch` — backports (csrc-only)
- `vxk_arc6_inner.sh`, `build_vxk_wheel_arc6.sh` — wheel build (arc5 v3
  host-swap strategy)
- `arc6test.Dockerfile` — test image layer (FROM cert21)

Companion repo-side files: `vllm_gaudi/xpu/patches/skinny_m.py`,
`tools/ar/r126_kbench.py` + `r126_skinny_kbench.sh` (gate window), and
the platform/envs registration entries (added on the P6.33 branch after
the P6.31 close-out merge).
