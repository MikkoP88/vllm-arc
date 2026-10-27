# VERDICT — P1 platform leg

**LEG**: 2026-10-27-p1-platform
**IMAGE**: vllm-arc:v0.30.0-b70 (docker 404c29d83376, fork_sha 7d6498d4,
manifest `/etc/vllm-arc/IMAGE_MANIFEST`: vllm 0.30.0+xpu md5 1f14d00b…,
triton 3.7.2+xpu md5 708e93ed…)
**GATE** (plan §10 P1 row): qwen3.8-27b-fp8 serves on the arc image; no boot
death; ArcXpuPlatform resolved; mem-info patch live; k1k cells sane.

| Check | Result | Evidence |
|---|---|---|
| Image builds with in-build gates | PASS | build.log (this dir); BASE_IMAGE/layer-order fixes commits |
| Engine boots TP2, no boot death | PASS | boot.txt; health 13:24:13 (7 min cold) |
| ArcXpuPlatform resolved (probe) | PASS | gates.txt `PLATFORM=vllm_gaudi.xpu.platform.ArcXpuPlatform` |
| Xe2 detect (device-id 0xe223, count=2) | PASS | gates.txt `XE2=True`, `DEVICE=Intel(R) Graphics [0xe223] count=2` |
| mem-info patch live (free>0 after import vllm) | PASS | gates.txt `MEMINFO free=4.77GiB total=31.89GiB nonzero=True` |
| Posture lines in serve log | PASS | posture.log ×2 processes (`Xe2 posture active`, `graph posture cudagraph_mode=NONE … XPU_GRAPH=False`) |
| Model identity | PASS | gates.txt `MODEL_ID ['Qwen3_5ForConditionalGeneration'] fp8`; served-model-name in log |
| k1k sanity cells | PASS | k1k_sanity.txt — TPOT c8 92.7 ms flat (eager tax, pre-P2), per-req ~10.8 tok/s, c16 agg 137.6 tok/s, all cells ok |

**Defects found & fixed during the leg** (all banked on the branch stack):
1. Dockerfile global `ARG BASE_IMAGE` unbound in manifest step (`set -u`).
2. `logger.info_once` unhashable-list arg (capture_sizes) — boot death.
3. `info_once` is rank-gated → dead in APIServer process → plain `logger.info`.
4. vLLM dictConfig gates INFO to the `vllm` logger namespace → platform
   logger renamed `vllm.platforms.arc` (posture lines now in serve logs).
5. oneCCL `ze_fd_manager opendir` boot death on non-privileged container →
   `--privileged` + the universal validated CCL env block (W14 == D
   lsv-test, byte-compared 2026-10-27).
6. `docker exec` heredoc gate probe missing `-i` (empty stdin, false FAIL).
7. Health-probe race vs dying previous engine → pkill port-drain wait.

**OUTCOME: PASS** — §10 P1 gate met. Banking p01–p06 + build pipeline to
main authorized (plan §10/§11 banking directive). P2 (attention + graphs)
starts next.

> **Amendment 2026-10-27 (p13, audit leg)**: this gate table omitted the
> §10 P1 row item "vxk wedge-site audit clean (or v88 wheel rebuilt +
> device-gated, §6)". The audit was executed later the same day at owner
> prompt: **NOT CLEAN** — the pinned vllm-xpu-kernels==0.1.14.1 carries 6
> bare int32 id×stride sites; §6 contingency invoked. Evidence:
> results/2026-10-27-p1-vxk-wedge-audit/ (incl. the discrimination showing
> the P2 defect-b capture-zombie is a separate defect). The PASS above
> stands for the checks actually run; the wedge item is closed by the p13
> contingency, not by this leg.
