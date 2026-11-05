# P6.21b — dense-FA headroom: roofline, route falsification, v2 open (2026-11-05)

Item: the FA-lane decision ladder after the ar_quant sub-lane closed
negative (same owner directive, plan P6.21). The banked p72 record left
one decisive number unmeasured: the stock kernel's distance to the
silicon roofline — the "v2 economics = parity" note assumed stock sits
near the ceiling. Five legs (r87-r91, all throwaway cert9 containers,
GPUs quiet, W14 stop/restore per leg, FRANCE-verified restores) measured
it.

## Legs

| leg | what | verdict |
|---|---|---|
| r87 | stock `_vllm_fa2_C.varlen_fwd` at serve shapes (GQA 12/2, hd 256, fp16 q, paged fp8 KV page 832, causal bs=1; the 64-call U32k pattern) + copy ceilings + v1 ESIMD three-way | stock 1.0-1.7 GB/s effective on mandatory traffic (memory roofline S~97-265 = fantasy bound — the op is COMPUTE-bound); standalone U32k slice 2.83-2.86 s vs banked serve 3.12 s (cross-validate); v1 x23-26 slower (reproduces p72 ~26x); v1 corrupts adjacent allocations at big bs=1 shapes (OOB; the p72 probe's `.float()` ref-copy masked it) |
| r88 | fp16 XMX peaks at attention shape classes + dual roofline | peak 147 TF/s (8k square), PV-bmm 136.7, QK-bmm 67.7 (transposed-B class); stock runs at 24-26% of peak across the pattern -> **S_dual(U32k) = 3.96 on BOTH cards >= 2.0 -> v2 lane OPENS** (pre-registered rule); v1 OOB proof affirmative (ref mutated by v1 run) |
| r89 | on-image attention inventory + SDPA rate table | `vllm_xpu_kernels` = flash_attn_interface only (same vxk op); no vllm_flash_attn; torch SDPA at Lq==Lk real+correct fro 2.6e-4 at ~98-104 "TF/s"; at Lq<Lk `is_causal` is TOP-LEFT (fro 4.7 garbage) and the constant-4.2 ms across kv = the kernel skipping the fully-masked bulk (the "impossible TF/s" was the flop model, not fake timing) |
| r90 | SDPA explicit bottom-right mask route prototype | default-context SDPA with bool mask silently falls to a catastrophic path at serve shapes (>26 min for a handful of calls — the 12.9 GB fp32 attention intermediate); leg killed at 26 min; W14 restored + FRANCE verified (`qwen3.8-flashnext` incumbent model name) |
| r90b | forced-backend mask matrix (fast-fail) | **FLASH rejects every explicit mask on XPU** ("Flash Attention does not support non-null attn_mask" -> No available kernel); EFFICIENT same; MATH correct (fro 2.1e-4) but small-shape rates already 1.5-3.4 ms and memory-blows at scale (r90); warning trail: XPU fused kernels REQUIRE equal q/kv heads — `enable_gqa` unsupported (r89's 98 TF/s was the MATH backend all along) |
| r91 | aten `_scaled_dot_product_efficient_attention` (LSE-capable) for a two-part split route (causal self-block + dense prefix + exact online-softmax merge) | op EXISTS with logsumexp; **rate 4.7 TF/s causal 8kx8k, 12.5 TF/s dense 8kx24k — 3-8x SLOWER than stock** (naive XPU implementation); split-merge numerics moot at that rate |

## Verdict

**Every patch-level route is now measured dead.** The XPU attention
surface on this stack offers exactly one fast kernel (the vxk varlen op,
37 TF/s useful = 25% of the 147 TF/s practical peak) and no mask/no
GQA/no LSE-capable alternative within reach of torch/aten/vllm_xpu_-
kernels. The measured headroom (S_dual 3.96; blended bmm ceiling
~90 TF/s => ~2.4x over stock) is real and reachable only by a kernel.

**P6.21 FA lane -> ESIMD v2 build** (the r88 pre-registered open):
flash-class redesign of the p72 v1 base — the v1 launcher (grid
(qHeads, rows), work-group (1,1) = ONE THREAD per (row, head)) is the
smoking gun for both the 26x and the OOB. v2 spec: subgroup-cooperative
q-tiles with GQA-6 KV sharing (kills the 6x reread), lsc block loads
(not 16-B gathers), online softmax, pipelined KV chain, XMX/DPAS for
QK/PV if achievable in ESIMD — target >= 2x stock on the slice (U32k
TTFT -6..-11%, more at U64k where the slice share grows). The OOB fix
is mandatory regardless (v1 is default-off in-tree; never armed).

## Key artifacts

- `rig/r87_fa_headroom.py` + `.json/.out` — the S_lo (memory-only)
  decision leg; includes the v1 NaN variant matrix (v1_diag)
- `rig/r88_gemm_peak.py` + `.json/.out` — peaks + dual roofline + v1
  OOB proof (ref1-vs-snapshot around a v1 call)
- `rig/r89_attn_inventory.py` + `.json/.out` — inventory + the
  Lq==Lk/Lq<Lk alignment discovery
- `rig/r90_sdpa_route.py` + `r90.out` — the stalled mask-route attempt
  (killed at 26 min; failure IS the datum)
- `rig/r90b_backend_matrix.py` + `.json` (console in `r90b.out`) —
  forced-backend accept/reject matrix
- `rig/r91_split_route.py` + `.json` (console in `r91.out`) — the
  LSE-split falsification. NOTE: r90b/r91 wrappers (sed-derived)
  reused the r90 bench filename — their full console outputs live in
  the respective `.out` files
- Naming: leg consoles = r8X.out / r9X.out (wrapper cat), raw bench
  streams = r8X_bench.out where they survived container kills

## Disposition

**ROUTE lanes CLOSED by measurement (this evidence); ESIMD v2 kernel
build OPEN** as the P6.21 vehicle per the pre-registered r88 rule.
No engine change lands from r87-r91 (pure telemetry legs; zero code
touched in the ship path). Rig: W14 incumbent live, FRANCE-verified
after every window.
