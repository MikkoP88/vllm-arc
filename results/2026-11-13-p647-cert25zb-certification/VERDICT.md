# P6.47 — cert25zb §10 full battery (r184): certification verdict (2026-11-13)

Image `vllm-arc:v0.30.0-b70-cert25zb` (manifest fork_sha `2dca3455`, cert25za base +
K2/K3 packed-fp16 dequant lanes default-on). Gate: `tools/ar/r184_cert25zb_gate.sh`
(r170 pattern, 8 legs + gemma compare, 14:46–19:43 UTC). Branch
`feat/arc-p647-mxkv-engine-vstride-fix`. Raw evidence: `raw/cert25zb/<leg>/`,
verdict extract `raw/cert25zb/gate_verdicts.txt` (full log on rig:
`/root/build/p647/r184_gate.log`). Pre-redaction originals stay on rig under
`/root/build/p647/cert25zb/` (rig-side sums: `raw/cert25zb/rig_serve_sums.txt`).

## Battery verdicts

| leg | posture | solo tps | agg8 tps | KV tokens (×ctrl/native) | verdict |
|---|---|---|---|---|---|
| ctrl | qwen fp8-checkpoint chain, KV default | 60.1–61.8 | 305.0–308.5 | 398,363 (1.00×) | **PASS** (LONGRETURN + T1_SEED + erraudit clean) |
| m8q | qwen + `ARC_KVDTYPE=mxfp8` | 51.3–53.5 | 222.7–238.9 | 386,484 (0.97×) | **PASS** |
| m4q | qwen + `ARC_KVDTYPE=mxfp4` | 60.5–62.9 | 248.7–260.9 | 721,994 (1.81×) | **PASS** |
| mkq | qwen + `ARC_KVDTYPE=mxk8v4` | 48.7–54.4 | 227.9–229.0 | 504,832 (1.27×) | **PASS** |
| m8g | gemma W8A16 + mxfp8 KV | 25.1–25.2 | 144.9 | 51,218 (0.97× g0) | bank-only comparative — KNOWN-CLASS |
| m4g | gemma W8A16 + mxfp4 KV | 26.5–26.6 | 142.4–142.8 | 99,422 (1.88× g0) | bank-only comparative — KNOWN-CLASS |
| mkg | gemma W8A16 + mxk8v4 KV | 26.1 | 138.2–138.3 | 67,602 (1.28× g0) | bank-only comparative — KNOWN-CLASS |
| g0 | gemma native-KV baseline | 27.6–27.7 | 191.5–192.7 | 52,822 (1.00×) | baseline — CLEAN rehits |

`GEMMA_COMPARE: PASS (no new modes / no amplification; known-class churn banked
for review)` — gemma legs gate comparative per the REP-LOOP baseline law (gemma
think-spam is native-KV behavior; absolute LONGRETURN FAIL lines on gemma legs are
the expected class, not counted). Qwen legs gate absolute: all four PASS with
cross-cycle determinism (`LONGRETURN: PASS`, `T1_SEED: PASS`, `ERROR=0`,
`TRACEBACK=0`).

Headline qwen decode numbers: **m4q holds ctrl parity at solo (60.5–62.9 vs
60.1–61.8 tps) at 1.81× KV capacity** (721,994 vs 398,363 tokens); mkq carries
1.27× capacity at 0.75× agg8 throughput; m8q holds capacity parity (0.97×) at
0.75× agg8. Gemma comparative: MX legs run 0.72–0.96× native decode at
0.97–1.88× KV capacity, no new output modes.

## m4g ERRAUDIT: the single counted GATE_FAIL item — re-adjudicated to 0 real errors

The wrapper printed `P647_CERT25ZB_GATE_COMPLETE -- GATE_FAIL=1`; the one counted
item is `ERRAUDIT: FAIL (raw/cert25zb/m4g)` with `ERROR=4`. All four ERROR lines
(`raw/cert25zb/m4g/erraudit_readjudication.txt`, banked) are the identical class:

```text
(APIServer pid=13) ERROR 11-13 16:43:58 [serving.py:235] Error with model error=ErrorInfo(message='The model `qwen3.8-27b-fp8` does not exist.', type='NotFoundError', param='model', code=404)
```

Adjudication — external client, not the engine, not the gate:

- The leg serves `ARC_SERVED=gemma4-31b`; every wrapper probe in that leg posts
  the gemma name. A `qwen3.8-27b-fp8` model-name request cannot originate from
  the gate. The requests arrive in pairs ~9 s apart (16:43:58/16:44:07 and
  17:27:51/17:28:01) — a LAN client's retry cadence against the open port.
- `TRACEBACK=0` across the whole serve log; every local probe 200 OK; the leg
  otherwise passed all faces (garbage 15/15, T1_SEED PASS, deterministic
  rehits).
- The mkg leg, run later with no external hits in its window, closed with
  `ERROR=0` on the identical image and chain — the engine is not producing 404s.

Excluding the external-client 404 class, the corrected error count is **0**.
Wrapper refinement noted for the next battery revision: ERRAUDIT should exclude
the non-local `does not exist` 404 class from the ERROR count.

## Cross-cert evidence (K2/K3 lanes touch only the MX paths)

- **g0 (native KV): byte-identical to the r170/cert25z g0 bank** —
  `G0_XCERT_IDENTICAL` in `raw/cert25zb/gate_verdicts.txt` (diff of the four
  longprobe face lines: `873d8e07cdec` ×3 + `8d3c1d66f587`, every field equal;
  see `raw/cert25zb/g0/longprobe_full.out`). The native path is untouched by the
  K2/K3 bake.
- **m8g**: cold/attractor pair `12c0e71e5be4 → fc2dddcc7ce6` matches the r170
  bank exactly (K2 is bit-identical d=0.0 — r178 — so mxfp8 outputs cannot move).
- **m4g**: simple ladder `af456639ad9d → 815df9453920 → 4ef5307e6135` and essay
  4-stage drift `252ee3808838/5637e5e39e68/a084d4f2e35d → fc2dddcc7ce6` match
  the r170b amendment signatures (drift-into-attractor, KNOWN-CLASS).
- **mkg**: cold hashes byte-match m8g's cold (`12c0e71e5be4`, `3950cc634cad`,
  `d3c6a465ecc1`) — cross-dtype cold agreement; the e2m1 V-side does not change
  the attractor destination.

## Posture

- W14 incumbent restored by the wrapper trap at 19:43:21Z (`W14_RESTORED` in
  `gate_verdicts.txt`; radiance-test Up and serving).
- Kernel lane state at bake time: m8 walk CONVERGED at 4.88 ms @32K (r182
  kdiagA21: 66% dot+load floor + irreducible 1.62 ms f8→f16 cvt; scale and
  softmax free), LUT rung DEAD, join/shift WASH (r183 shiftabA22) — ladder
  closed on all three formats (plan §P6.47).
- **VERDICT: cert25zb CERTIFIED** — qwen absolute gates 4/4, gemma comparative
  PASS, single counted item re-adjudicated to 0 real errors with banked
  evidence. Release: tag `v0.30.0-b70-cert25zb`.

## Redaction note

`raw/cert25zb/**` banked logs carry the law-8 pass: `10.20.3.<host>` →
`10.20.3.REDACTED`, `10.100.8.<host>` → `10.100.8.REDACTED` (perl byte-exact;
46 client-address occurrences in `m4g/serve.log`, 4 in its readjudication
excerpt, remainder across the other serve logs' access lines). See
`results/REDACTION-2026-11-13-p647-cert25zb.md`. Pre-redaction bytes remain on
the rig; rig-side sums in `raw/cert25zb/rig_serve_sums.txt` and the rig bank's
own `SHA256SUMS` refer to those originals. Local `SHA256SUMS` at this bank's
root covers the redacted tracked files (authoritative).
