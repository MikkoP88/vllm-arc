# vllm-arc — status

Current-state holder: the certified release and the improvement-lane
disposition. History lives in dedicated archives:
[`history/STATUS_HISTORY.md`](history/STATUS_HISTORY.md) (prior certs +
phase ledger), the changelog [`../CHANGELOG.md`](../CHANGELOG.md)
(per-cert/per-phase entries), and per-release summaries
[`release_notes.md`](release_notes.md). Every claim traces
to banked `results/` legs.

**Currency: `rel/v0.30.0-b70-cert23` (`vllm-arc:v0.30.0-b70-cert23`).**
At every certification close-out the anchors repoint (checklist in
`tools/check_currency.sh`) and this page is rewritten in the same
change.

## Certified release — `rel/v0.30.0-b70-cert23`

Every gate GREEN with banked evidence. The image carries the certified
engine plus the **graph-exec double-buffer**
(`vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`): every XPU graph
captured ×2 into the manager's shared pool, replays alternate execs —
removes the torch-2.13 same-exec replay host block at zero cost
(+29 s boot, no extra memory, byte-identical semantics, E2E-neutral at
every measured posture; `VLLM_ARC_GRAPH_EXEC_DB=0` kill-switch).
Evidence: `results/2026-11-10-p635-telemetry/`.

What the certified stack delivers:

- **Decode postures** — MTP depth k5 = +41 % decode (115.2 tok/s solo
  U8k vs 81.6 at k3); **mxfp4-W4A16 + k5 = 123.9 tok/s (+52 %) with
  TTFT 3.98 s < the 4.55 s baseline** — the recommended decode posture
  (`ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16
  ARC_QUANTCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
  ARC_GMU=0.85 ARC_SPEC=5 ARC_FP8=0`, BASE checkpoint); project best
  **128.3 tok/s** in the full validation matrix. mxfp8-W8A16 is a
  decode dud (≈ baseline — no weight savings, bf16-activation tax).
- **Full validation matrix** — 11 cells, both models, all online
  quants × spec depths, uniform protocol, 0 tracebacks: qwen
  mx4-W4A16 wins every spec column; gemma mx4-W4A16 37.1 tok/s
  (+43 % vs mxfp8); gemma online fp8 boots + serves; mxfp8 deprecated
  for decode. Evidence: `results/2026-11-10-p635-telemetry/vmatrix/`.
- **Per-config performance reports** —
  [`docs/reports/`](reports/README.md): model × quant × MTP,
  decode/TTFT/prefill/ITL at 1K–128K context, the MTP-k3 temperature
  sweep, the kv-cache-dtype comparison, line charts, and the copyable
  configuration for every cell (bank `results/2026-11-10-p637-reports/`).
- **Host compatibility** — 64 GB-RAM hosts run the recommended configs
  (`--shm-size 16g`, 16 GB LMCache CPU tier).
- **KV-cache dtype knob** — `ARC_KVDTYPE`: `fp8` ≈ the incumbent;
  `auto` (fp16 KV) refuses at `max_model_len=262144` (8.73 > 6.78 GiB)
  but serves at 131072; `fp8_e5m2` hard-refused (no e5m2 KV path in
  esimd FA2 on Xe2); the four `turboquant_*` presets servable
  (comparison:
  [`reports/qwen3.8-27b-fp8-kv-cache-dtype.md`](reports/qwen3.8-27b-fp8-kv-cache-dtype.md)).
- **Patch inventory** — [`PATCHES-arc.md`](PATCHES-arc.md): every
  module with purpose, mechanism, knob, and evidence.

## Project status

All execution phases are complete and the close-out is delivered:
every phase closed with banked evidence, and no engine bytes changed
since the certified tag. The per-phase ledger:
[`history/STATUS_HISTORY.md`](history/STATUS_HISTORY.md); per-phase
changelog entries: [`../CHANGELOG.md`](../CHANGELOG.md).

## Improvement-lane status — the in-scope ledger is CLOSED

Every prefill device slice is dispositioned by measurement — GEMM
39.7 % (fastest on-image route deployed; ~99 % of the per-card dense
ceiling), allreduce ~30 % (quantized e4m3 wire available opt-in at
1.65× the dominant AR; 82 % of the measured wire floor), FA ~20 %
(ESIMD v2 ceiling-bound at 0.33× stock), GDN ~5 %, norms/act ~5 %,
host path clean. Cross-stream AR∥GEMM co-tenancy — the last unmeasured
door — closed by measurement (the device time-slices AR and GEMM
kernels; overlap is slower than serial at every k on both wires).
**No remaining lawful lever ≥ 10 %**; residual headroom bounded
≤1–2 % class. Transport proven clean (direct P2P, 10.23 GB/s D2D).
Measurements: [`PERFORMANCE-arc.md`](PERFORMANCE-arc.md);
route-by-route closure history:
[`../CHANGELOG.md`](../CHANGELOG.md) (per-phase entries).
