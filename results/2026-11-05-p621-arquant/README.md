# P6.21 — W13H ar_quant donor port: quantized-transport TP2 all-reduce (2026-11-05)

Item: the next optimization lane after P6.20 closed the prefill host path
(owner directive 2026-11-04: code optimization, telemetry-driven, certified
image at the end). Lane chosen by the P6.21 deep-analysis re-read of the
banked r67b op-class capture: GEMM 39.7 % already optimal (P6.7 rung C
closed negative), **TP2 allreduce family = 30.4 % of rank0 device time at
U32k** (ring 2.25 s + ReduceCopy 2.26 s + barrier 0.15 s, c10d-verified) —
P6.7's own "top recommendation beyond the named rungs".

## Component

Line-D donor (scope-law review in plan P6.21 §): iexperimental W13H staging,
rig `/root/build/w13h/` — `ar_quant.py` (118 lines, md5
`b744f090369a53272b9733c425c9af0a`), banked verbatim at `donor/` with its
`patch_ar_quant_xpu` installer and RowParallelLinear call-site hook.
Mechanism (W13D law): oneCCL AR already runs at the P2P ceiling
(9.5–9.7 GB/s ≈ raw copy); the only byte-cut is payload precision — peer
partial as per-row-scaled e4m3 over the dtype-agnostic
`all_gather_into_tensor` (half the AR bytes), local partial exact, fp32
reconstruct `out = local + dequant(peer)`. NOT byte-exact by construction
(~1e-3…3e-2 relative on the peer term only). Guards: TP2-only, xpu-only,
numel ≥ 2 M (prefill-class only; decode/verify ARs stay plain), never
inside stream capture, sticky fail-soft to the original AR.

Port deltas (all cited in the plan): dtype gate {bf16, fp16} (our p44 W8A16
activations are fp16); capture probe resolved defensively; env names
`VLLM_ARC_AR_QUANT` / `VLLM_ARC_AR_QUANT_MIN_ELEMS` (r45g namespace law) +
registry rows; wiring as an in-place `_install()` patch module
(`vllm_gaudi/xpu/patches/ar_quant.py`, `_XPU_PATCH_MODULES` registration —
wrap-and-rebind of `vllm.distributed.tensor_model_parallel_all_reduce`,
original stashed as `_arc_ar_quant_orig`). Default OFF pending the r86
evidence.

Donor validation on the incumbent (same model law, banked at
`donor/w13h_validate_excerpt` refs in the plan record): lane installed +
active at 4096×2560 payloads, long-gate A/B determinism PASS, math gate
'42' correct under the lane — the incumbent's production posture runs it
armed.

## Legs (rig 10.20.3.REDACTED)

| leg | what | verdict |
|---|---|---|
| r85 | port written + py_compile in cert9 container + single-process arm smoke (wrap installs, orig stashed, lane inert without TP2 — fallthrough to the original AR is the stock symbol's own no-TP assert) | PASS |
| r86 | A/B/A/B on cert9 bind-mount: phases arq1/off1/arq2/off2 (VLLM_ARC_AR_QUANT 1/0/1/0) + cf1 counterfactual (armed, MIN_ELEMS=1e9 → installed-yes/active-no; cf outputs must hash-match the OFF phases). Per-phase: integrity 5/5 + det 3/3, longprobe full (coherence under e4m3 peer noise), spec gate, run7n t0, p65 timeline faces | **QUALITY GREEN / PERF NEGATIVE — lane NOT adopted** (table below) |

## r86 verdict detail

Engagement proof: both armed phases log `patch installed` at register-time
and `all-reduce active` at the first qualifying payload (65536×1152 =
75.5 M elements; both workers); unarmed phases log neither (posture-leak
gates green); cf1 logs installed-but-inert (`CF-OK`) — the sizing knob
cleanly gates the lane.

Quality (armed posture): integrity 5/5 + det 3/3; longprobe CLEAN — all
four classes t0-identical within posture (essay `1d415a15e11d`, code
`507d389dc258`, multi `636e4ffdbb04`, simple `84efc84cc142`), t1-seed
identical, zero garble; spec gate **byte-exact the cert8/cert9 banked
values** (pos1 0.9258 / acceptance 0.837 / mean-len 3.51) — the sizing
gate keeps every decode/verify AR on the plain collective (the simple
class, 38 tok, hash-identical ON and OFF, is the smallest proof); zero
fail-soft firings across all five phases.

Perf (TTFT, p65 v5.1 faces, seconds):

| face | ON-1 | ON-2 | OFF-1 | OFF-2 | cf (inert) | ON mean | OFF mean | Δ |
|---|---|---|---|---|---|---|---|---|
| U32k | 18.44 | 18.44 | 17.72 | 17.66 | 17.64 | 18.44 | 17.69 | **+0.75 (+4.2 %)** |
| C32k | 1.86 | 1.83 | 1.36 | 1.25 | 1.24 | 1.85 | 1.31 | +0.54 |
| U8k | 4.03 | 4.04 | 3.76 | 3.76 | 3.74 | 4.04 | 3.76 | **+0.28 (+7.3 %)** |
| U64k | 43.68 | 41.33 | 38.96 | 38.90 | 41.08 | 42.51 | 38.93 | **+3.58 (+9.2 %)** |

OFF phases reproduce the cert9 banked family (U64k 38.90–38.96 vs banked
38.93; U8k 3.76 exact). The ON regression is reproducible across both
interleave pairs and monotone in context length; the U64k face's own
boot variance (cf1 41.08; r81-era OFF range 38.9–41.5) overlaps the ON
tail but the ON-mean delta is +3.6 s vs a ±1.1 s face σ.

t0 run7n (lane-inert payloads): ON 269.68/294.84, solo 83.72/83.62;
OFF/cf 268.70–296.79, solo 84.33–84.53 — in-family, no decode-side
effect (expected: decode ARs never engage the lane).

**Why the donor's law does not transfer (disposition analysis):** the
W13D premise is that the plain AR runs at the P2P ceiling so halving
transported bytes must win. On our stack the lane ADDS per-AR device
work — abs/amax + quantize (read 151 MiB fp16 + fp32 intermediates +
write 75.5 MiB e4m3) + reconstruct (read local + dequant peer + write,
again with fp32 intermediates) — roughly three full-tensor passes with
two fp32 materializations per engaged AR, times every vocab-class and
hidden-class AR of every layer of every chunk. At U32k that is ~48
hidden-ARs/chunk plus the giant embedding-class ARs; the added compute
exceeds the transport saved, and it scales WITH the face (worst at
U64k, +9.2 %). The incumbent's win case (bf16 MoE posture, their AR
sizes, their CCL tuning) does not hold for our dense fp16 W8A16
activations under the certified CCL chain. CONFIRMED NEGATIVE — the
A/B/A/B interleave closes it; no adoption, no engine change lands
(the P6.7 rung-C precedent for closed-negative lanes: evidence banked,
code reverted from the ship path, `rig/files/` preserves the exact
staged bytes for any future re-test).

## Key artifacts

- `donor/ar_quant.py` — verbatim donor (md5 b744f090…)
- `donor/patch_ar_quant_xpu_installer.py`, `donor/hpu_row_parallel_linear_hook.py` — wiring excerpts
- `rig/files/` — the exact staged port bytes (ar_quant.py e1a12e7e…,
  platform.py, envs.py as bind-mounted over cert9)
- `rig/r86_arq_abab.sh` — the A/B leg script; `rig/r86/` — per-phase
  evidence (posture gates, integrity, longprobe, spec, run7n, timeline
  faces + bystander windows, serve logs, serve_used.sh);
  `rig/r86.out` — the leg console
- `../REDACTION-2026-11-05-p621-arquant.md` — byte-exact redaction record

## Disposition

**CLOSED NEGATIVE — lane not adopted; no engine change lands.** The port
proved SAFE (deterministic, coherent, spec untouched by the sizing gate)
but SLOWER on every prefill face (U64k +9.2 %, U8k +7.3 %, U32k +4.2 %):
the added quant/reconstruct device work with fp32 intermediates exceeds
the halved-transport saving under our dense fp16 W8A16 posture and
certified CCL chain — the donor's at-P2P-ceiling law does not transfer.
Working-tree code reverted after banking (`rig/files/` + this tree hold
the exact bytes). The P6.21 lane's remaining live lever is the FA ESIMD
kernel project (FA 20.3 % of rank0 device time at U32k; P6.7 port spec).

