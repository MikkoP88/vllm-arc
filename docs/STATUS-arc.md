# vllm-arc — status

Current-state holder: the certified release and the improvement-lane
disposition. History lives in dedicated archives:
[`history/STATUS_HISTORY.md`](history/STATUS_HISTORY.md) (prior certs +
phase ledger), the changelog [`../CHANGELOG.md`](../CHANGELOG.md)
(per-cert/per-phase entries), and per-release summaries
[`release_notes.md`](release_notes.md). Every claim traces
to banked `results/` legs.

**Currency: `rel/v0.30.0-b70-cert28`** (`vllm-arc:v0.30.0-b70-cert28`).
At every certification close-out the anchors repoint (checklist in
`tools/check_currency.sh`) and this page is rewritten in the same
change.

## Certified release — `rel/v0.30.0-b70-cert28`

Every gate GREEN with banked evidence. The image carries the certified
engine plus the **MX KV decode lane** (P6.47, the owner-directed
decode goal): `--kv-cache-dtype {mxfp4, mxfp8, mxk8v4}` through the
`vllm_gaudi/xpu/patches/mx_kv/` family + the baked arc7mx build28
wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, `.so` md5s asserted
in-image; cert27 content + the r231d seqused split clamp). The
graph-exec double-buffer (cert23) and every earlier certified patch
ride along — full inventory:
[`PATCHES-arc.md`](PATCHES-arc.md).

What the certified stack delivers:

- **MX KV decode line (the P6.47 record, cert28-measured)** — all
  three MX dtypes native-certified (qwen postures `ARC_MXKV_NATIVE=1`):
  the n1 ITL five-face record (mxfp4 43.4→58.1 ms at 2→32K vs the fp8
  ctrl's flat 40.1–42.5 — +8.2 % shallow, 1.37× at 32K), battery agg8
  ctrl 307.8 / m4q 0.84× / m8q 0.82× / mkq 0.81×, n8 steady-state
  long-ctx 32K ss MX 1.10–1.18× ctrl (r237), and the acceptance
  K-precision law (e2m1 K −10 % acc/step, e2m1 V free; r238). Net:
  MX KV decode at 0.81–0.84× fp8 with 1.28–1.88× KV-token capacity;
  the MX hunt idles on closed channels (engine-shape, workload-shape,
  counters — the r236–r238 dispositions). Evidence:
  `results/2026-11-24-p647-cert28-battery/` + the r234–r238 legs.
- **Decode postures** — **MTP k3 is the certified, overall-best depth**
  (acceptance 83.7 % strict-gate at t0; deeper drafts measured and
  dropped — decision record in the CHANGELOG). Certified lanes: qwen
  fp8 + k3 = 93.8 tok/s @ 1K (the default posture, `ARC_SPEC=3
  ARC_FP8=1`); **qwen mxfp4-W4A16 + k3 = 103.3 tok/s @ 8K, the best
  online-quant lane** (`ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16
  ARC_QUANTCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
  ARC_GMU=0.85 ARC_SPEC=3 ARC_FP8=0`, BASE checkpoint); W4A16's edge
  widens monotonically with context. mxfp8-W8A16 is a decode dud
  (≈ baseline — no weight savings, bf16-activation tax). Evidence:
  `results/2026-11-10-p635-telemetry/` (+ `vmatrix/`) + the
  per-config report banks under `results/` (docs/reports).
- **Full validation matrix** — 11 cells, both models, online quants ×
  MTP k3/no-spec, uniform protocol, 0 tracebacks: qwen mx4-W4A16 wins
  both spec columns; gemma mx4-W4A16 37.1 tok/s (+43 % vs mxfp8);
  gemma online fp8 boots + serves; mxfp8 deprecated for decode.
  Evidence: `results/2026-11-10-p635-telemetry/vmatrix/`.
- **Per-config performance reports** —
  [`docs/reports/`](reports/README.md): model × quant × MTP,
  decode/TTFT/prefill/ITL at 1K–128K context, the MTP-k3 temperature
  sweep, the kv-cache-dtype comparison, line charts, and the copyable
  configuration for every cell (cert23-era battery — being re-run on
  cert28 by the 2026-11-24 currency campaign, see Project status;
  bank `results/2026-11-10-p637-reports/`).
- **Host compatibility** — 64 GB-RAM hosts run the recommended configs
  (`--shm-size 16g`, 16 GB LMCache CPU tier).
- **KV-cache dtype knob** — `ARC_KVDTYPE`: `fp8` ≈ the incumbent;
  the three MX lanes above; `auto` (fp16 KV) refuses at
  `max_model_len=262144` (8.73 > 6.78 GiB) but serves at 131072;
  `fp8_e5m2` hard-refused (no e5m2 KV path in esimd FA2 on Xe2); the
  four `turboquant_*` presets servable (comparison:
  [`reports/qwen3.8-27b-fp8-kv-cache-dtype.md`](reports/qwen3.8-27b-fp8-kv-cache-dtype.md)).
- **Patch inventory** — [`PATCHES-arc.md`](PATCHES-arc.md): every
  module with purpose, mechanism, knob, and evidence.

## Project status

P6.47 (the MX KV decode goal) is delivered and certified through
cert28: the §10 correctness battery (r233) plus the record legs
(r234–r238) are banked. The 2026-11-24 deep-sweep campaign: landed —
the record-lane re-discrimination (r239/r239b: checkpoint-lane TTFTs
1.5–3.5× faster than their era anchors, gemma L1 record
parity-or-better with the agg8 record 1.85–2.4× the era aggregate;
bank `results/2026-11-24-p647-r239-record-lanes/`) and the docs
hard-currency sweep (this rewrite, the [`PATCHES-arc.md`](PATCHES-arc.md)
MX KV family, and the `tools/check_currency.sh` anchors that close
the drift class). In flight: the full per-config report battery
re-run ([`docs/reports/`](reports/README.md)) on the current image —
r240a solo faces → r240b concurrency sweep → r240c gemma 128K annex
→ report regeneration. The per-phase ledger:
[`history/STATUS_HISTORY.md`](history/STATUS_HISTORY.md); per-phase
changelog entries: [`../CHANGELOG.md`](../CHANGELOG.md).

## Improvement-lane status

Prefill: the in-scope ledger is CLOSED — every device slice
dispositioned by measurement (GEMM 39.7 %, the fastest on-image route
deployed, ~99 % of the per-card dense ceiling; allreduce ~30 %
(quantized e4m3 wire available opt-in at 1.65× the dominant AR, 82 %
of the measured wire floor); FA ~20 % (ESIMD v2 ceiling-bound at
0.33× stock); GDN ~5 %, norms/act ~5 %, host path clean;
cross-stream AR∥GEMM co-tenancy closed measured — the device
time-slices the two kernel families, overlap slower than serial at
every k on both wires). **No remaining lawful lever ≥ 10 %**; residual
headroom bounded ≤1–2 % class. Transport proven clean (direct P2P,
10.23 GB/s D2D). Decode: the MX KV hunt idles on closed channels —
engine-shape (no new lineage), workload-shape (all face classes
measured), counters (the acceptance channel closed by r238);
reopening requires a NEW channel. Measurements:
[`PERFORMANCE-arc.md`](PERFORMANCE-arc.md); route-by-route closure
history: [`../CHANGELOG.md`](../CHANGELOG.md) (per-phase entries).
