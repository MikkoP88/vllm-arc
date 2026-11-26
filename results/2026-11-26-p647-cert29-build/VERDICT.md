# r254 — cert29 image build (P6.47, the cert29 cut)

**VERDICT: PASS — `vllm-arc:v0.30.0-b70-cert29` built and verified
(R254_CERT29_BUILT 2026-11-26T21:56:16Z).** All in-image asserts green
(`#8 31.44 CERT29_WHEEL_PLUS_DELTA_LIVE asserts OK` — 22 asserts:
19 source-string checks + callable checks + 3 .so md5 pins), manifest
append verified (`cert29_fork_sha=16effb497cd36cb25df7c19cb6f1938d2d45cdc5`,
`cert29_built=2026-11-26T21:53:34Z`), image present on the rig
(ID `4a4e9b0dd244`).

## What the image is

`FROM vllm-arc:v0.30.0-b70-cert28` + two halves:

1. **Wheel (kernel side)**: the r245a build34 wheel
   `vllm_xpu_kernels-0.1.14.1+arc7mx` (md5 `95451e404b463d9d6efa4bd4a4939b26`,
   `.r245a.md5` sidecar, force-reinstalled over the cert28 wheel) —
   page-128 admission at the gemma full-attention geometry (D=512,
   G=8: the TORCH_CHECK admission + the head_size==512/block_size==128
   dispatch branch riding the _512/_16 sub-tile frame), carrying the
   r231d seqused-tightened split-clamp lineage. Installed .so pins
   asserted in-image: `libattn_kernels_xe_2`
   `c20ca75550f29513a0f73b70ebb1243a`, `_vllm_fa2_C.abi3.so`
   `046e915ebce0eb33a37832377f8c2aed`, `libmqa_logits`
   `190e261edc79b3b497d8d8e801e65757` (not relinked).
2. **vllm_gaudi (python side, the 16effb49 content commit, 6 files)**:
   - `mx_kv/engine.py` + `mx_kv/native_decode.py` (P-C2 r244): the
     fa_utils window shim + the sliding-window init refusal (windowed
     impls boot ONLY on the native decode lane, boot-loud) + the window
     riding the kernel as (W-1, 0) per row.
   - `xpu/platform.py` + `envs.py` (P-D): the lmc_mx_boot_gate factory
     exec-wrap hook + the `VLLM_ARC_LMC_MX_GATE` knob registration.
   - `lmcint/vllm_v1_adapter.py` (P-H2/P-H4): the packed-row MX interop
     fix (the 32 B/token cumulative misaddress behind the r168 garble
     and the r240b wedge) + the `ARC_LMC_MX_INTEROP = "packed-row/r249"`
     capability marker.
   - `lmc_mx_boot_gate.py` (P-H4): the posture-aware gate — interop
     present -> `gate=interop-live` allow with telemetry; absent ->
     `ARC_LMC_MX_REFUSED` (pre-r249 installs protected); knob bypass
     for A/B evidence only.

## Refuse set (all passed before the build started)

| refuse | gate |
| --- | --- |
| base image | cert28 manifest `cert28_fork_sha=43f84cf977f9444900ff21387715ca5865db4c1a` |
| wheel bytes | md5 `95451e404b463d9d6efa4bd4a4939b26` + `.r245a.md5` sidecar |
| wheel build rc | `VXK_ARC7MX_BUILD_EXIT rc=0` in `/root/build/arc7mx_build34_r245a.log` |
| offline oracle | `VERDICT: PASS` in `/root/build/p245a/out/r245a_summary.txt` (40 faces) |
| e2e grid | `R245_DONE failed=[] 2026-11-26T13:04:09Z` in `/root/build/logs/r245_e2e_launch.log` |
| P-D probe | `== p246 faces ok=11/11` in `/root/build/p246/out/r246_summary.txt` |
| P-H interop | `INTEROP_PASS` in `/root/build/p249/outcome.txt` |
| P-H4 probe | `== p253 faces ok=15/15` in `/root/build/p253/out/r253_summary.txt` |
| staged delta | 6-file md5 map (see `stage_provenance.txt`) |

## In-image asserts (the r232 recipe)

The Dockerfile RUN block imports the patched modules, `inspect.getsource`s
engine/native_decode/gate and `read_text`s platform/envs/adapter (the
adapter is NEVER imported bare — the r253 probe law: the vendored
`lmcache_integration/__init__` dies on lmcache 0.5.5 outside the p41b
src-override order), asserts the inherited certified line (native bridge
import, default-off gate, `_NATIVE_MINLEN`, AB instrument, NSPLIT
instrument, r212f capacity-priced walk, capture telemetry) + the six
cert29-delta strings (P-C2 x5, P-D x2, P-H4 x4, P-H2 x2) + the three
.so md5s, then prints `CERT29_WHEEL_PLUS_DELTA_LIVE asserts OK`.
`__pycache__` purged under `vllm_gaudi` after the copies.

## Banked files

| file | content |
| --- | --- |
| VERDICT.md | this verdict |
| r254_build_launch.log | full launch log (167 lines): refuses, docker build output, asserts, manifest cat (md5-verified both sides `285eb5ff81c0664dc51f2730827c98c8`) |
| manifest_tail.txt | the post-build manifest tail verification (`R254_CERT29_BUILT`, `cert29_fork_sha`, `cert29_built`) |
| stage_provenance.txt | the 6 staged-file md5s + the wheel pin + the launch mechanics |
| r254_build_cert29.sh.shipped | the builder bytes (md5 `b14d7d9d93858018910aef725a1b9f25`) |
| SHA256SUMS | local sha256 over the bank |

## Next rung

The section-10 gate `tools/ar/r255_cert29_gate.sh` (md5
`bb671486edcf8add69680fe09bbb390d`, shipped + verified both sides) —
launched 2026-11-26T22:09:03Z with FORK_SHA=16effb49. Refuse set green
(`R255_IMAGE_CONTENT_OK`, `R255_FORK_SHA_OK`). Battery: ctrl, m8q, m4q,
mkq, lmc (the P-H interop lane knob-free — `gate=interop-live` required),
gref (the P-C2 boot-loud refusal face), m8g/m4g/mkg (the r245-validated
gemma native+window posture), g0, gemma_compare. Gate results bank at
close-out in `results/2026-11-26-p647-cert29-gate/`.
