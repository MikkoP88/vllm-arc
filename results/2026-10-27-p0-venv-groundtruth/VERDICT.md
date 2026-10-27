# VERDICT — P0 venv ground-truth capture

**LEG**: 2026-10-27-p0-venv-groundtruth
**GATE** (plan §10 P0 row, capture step): pip freeze of the validated image
venv mirrored into the repo; pins md5-recorded; build interface reproduced.

| Check | Result |
|---|---|
| freeze captured (217 pins) | PASS — freeze.txt |
| vllm 0.30.0+xpu wheel md5 pinned (1f14d00b…) | PASS — external_patches/wheels.pins.md5 |
| triton 3.7.2+xpu wheel md5 pinned (708e93ed…) | PASS — same |
| base image present on rig (vllm-iexperimental:base-v0.26.0) | PASS |
| build interface documented + reproduced (Dockerfile) | PASS |
| verify-or-drop envs (Q40_LIB, OFFLOAD_WEIGHTS) | DROPPED with rationale (summary.txt) |

**OUTCOME: PASS** — P0 capture complete; fork scaffold branches p01–p06
built on top; image build authorized.
