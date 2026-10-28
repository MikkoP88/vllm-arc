# VERDICT — r14: the W14 incumbent's exact graph posture (FULL_AND_PIECEWISE, sizes [4,8,16,32,64], max 64, compile ON) does NOT save our stack — flip at first request anyway. Graph posture is exonerated; the GDN kernel-path delta between W14 and us is the mechanism

## The leg

arc-graph-w14posture-r14-20261028-074551, image c966fcf89a3f (fork
cbb6937e), all-CLI-knob reproduction of the W14 incumbent's serve posture on
OUR stack (plan Appendix C control group 1):

- `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE",
  "cudagraph_capture_sizes":[4,8,16,32,64],"max_cudagraph_capture_size":64}`
  (compile ON — PIECEWISE pieces need the compiled artifacts).
- Engine config dump confirms `<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>`,
  sizes [4, 8, 16, 32, 64], max 64 (serve_full.log line 60; platform echo
  line 43).
- PIECEWISE graphs captured 5/5 clean (line 226), FULL graphs 5/5 clean
  (line 227) — no capture-time death; defect (a) (speculator PIECEWISE boot
  death) not exercised here (MTP off).

## Result — control group 1's posture is NOT sufficient on our stack

- Serve ready 07:51:24 (line 249).
- **07:51:48 ARC_GRAPH_ZOMBIE_ONSET** (line 302): device-default stream
  reads RECORDING with no capture open — the same signature as r3–r13, now
  under the incumbent's exact posture. Onset stack (stack_extract.txt):
  `piecewise_backend.py:380 → standalone_compile → inductor cached artifact`
  — the compiled-prefill window, consistent with r12.
- 12 × HTTP 500 follow; engine dumps input and dies (line 1301, 07:51:51).

## Interpretation (with r13): the mechanism is narrowed to the kernel path

r13 (mode=NONE, eager) and r14 (FAP, compiled) BOTH flip at first request;
r13's eager onset stack names the GDN `in_proj_ba` prefill gemm. Graph
posture (mode, sizes, pool count) and compilation are both exonerated as
THE mechanism. What remains different between us and the two clean
controls:

- **W14 incumbent** (clean, same torch 2.13.0+xpu, same core 0.30.0, same
  qwen_gdn_linear_attn.py md5, same wheel base 0.1.14.1): its GDN does NOT
  run the stock XPU path — radiance `qwen3_5.py` tail-swaps
  `QwenGatedDeltaNetAttention → HPUGatedDeltaNetAttention` (pytorch GDN via
  `ops/hpu_gdn_pytorch.py`; ESIMD QSA for the flash-next lane) — the stock
  `torch.ops._xpu_C.gdn_attention` SYCL prefill is never called.
- **0.26 D champion** (clean, FULL graphs): different GDN code generation
  entirely (torch 2.12-era kernels).

Defect-b model v2 stands: **stock Qwen GDN XPU prefill path × live XPU
graph machinery**. Fix lane: replace/reroute the prefill path (radiance
pytorch-GDN port = plan-sanctioned §3 #16 lane; p10) or fix the kernel.
Discriminating evidence owed: r15 probe parked inconclusive (synthetic
prefill-shaped call DEVICE_LOST standalone — metadata suspect); r16
(capture REAL serve kwargs → standalone replay ± graphs) decides
"kernel bug on TP2 geometry" vs "kernel × graph interaction".

Posture note for P2: W14-posture FAP is NOT blocked from being our
production posture by this leg (it flips for the same reason FDO does);
defect-a (speculator PIECEWISE boot death) still gates FAP+MTP separately.

Raws: serve_full.log (onset line 302, stacks follow, dump line 1301),
stack_extract.txt, boot.txt, gates.txt, k1k_sanity.txt, spec_gate.txt,
matrix.log (rig leg dir arc-graph-w14posture-r14-20261028-074551).
