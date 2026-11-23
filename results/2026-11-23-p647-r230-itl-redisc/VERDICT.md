# r230 — cert27 long-ctx ITL re-discrimination (P6.47, 2026-11-23)

**VERDICT: PASS — R230_GATE: PASS.** The certified perf record's
long-ctx ITL faces re-measured on the shipping
`vllm-arc:v0.30.0-b70-cert27` image. The fp8 control reproduces the
banked anchors at 0.999–1.001x; native mxfp4 lands +1.8–2.2% (the r225
fix3 fp32 split-partials cost — within the +5% one-sided band,
oracle-band law); native mxfp8/mxk8v4 get their first banked long-ctx
rows. All quality gates clean (chunks <= 57 vs the 80 law floor, zero
errors/tracebacks, native telemetry 108 `MXKV native call` lines/cell
(`grep -c`, verified against both serve.log and telemetry.txt),
splits=auto).

## Result table (n1, MT=192, spec k3, XPU graph on)

| cell | 8K itl_ms | 16K itl_ms | 32K itl_ms | vs anchor | tps @8/16/32K |
|------|-----------|------------|------------|-----------|----------------|
| ctrl (fp8_e4m3) | 40.4 | 41.1 | 42.6 | 0.999/1.000/1.001 | 85.2/82.2/80.2 |
| m4q (mxfp4 native) | 49.4 | 52.1 | 58.1 | 1.018/1.018/1.022 | 70.5/72.4/65.2 |
| m8q (mxfp8 native) | 50.9 | 55.3 | 64.4 | — (first native row) | 72.7/62.3/53.4 |
| mkq (mxk8v4 native) | 50.0 | 53.9 | 61.9 | — (first native row) | 75.4/68.8/58.9 |

Anchors (banked cert26-content record, r219b auto-at-128):
fp8 {8192: 40.4, 16384: 41.1, 32768: 42.5}; mxfp4 {8192: 48.5,
16384: 51.2, 32768: 56.8}.

## Reading

- **ctrl parity 0.999–1.001x**: the fp8 bank is untouched by the r225
  wheel fixes, as designed — the strongest possible control.
- **m4q +1.8–2.2%**: fix3 moves split-KV partial accumulation to fp32
  (single bf16 quantization point) — the numerics correctness fix that
  un-refused mxk8v4 costs ~2% ITL on mxfp4. Accepted within the +5%
  band; record updated to the cert27-measured numbers.
- **Native mkq vs its cert26 Triton lane**: 61.9 ms @32K native; the
  agg8 half (banked in the cert27 battery) already showed native mkq
  +7.9% over the cert26 Triton lane (250.6 vs 232.2 agg8 tps) — native
  is the fastest way to run mxk8v4 at both short and long context.
- **Lane ordering at 32K**: ctrl 42.6 < mkq 61.9 < m4q 58.1 < m8q
  64.4 — mxfp4 stays the fastest MX lane at long context (1.36x ctrl
  @32K, was 1.34x on cert26), mkq close behind (1.45x), m8q the MX
  capacity lane (1.51x).
- **agg8 half (banked prior, `2026-11-23-p647-cert27-battery/`)**:
  ctrl 303.5/303.2 parity, m8q 249.7/249.9 parity, m4q 261.9/254.6
  (+2.9%), mkq native 250.6 vs Triton 232.2 (+7.9%).

## Method

One boot per cell (`tools/ar/r230_cert27_itl.sh`, md5
e712efd9ba03fae11a031ee50a937b20): refuses verified the image, the
manifest `cert27_fork_sha=d39ab5a26bf737d006d47a3f3125a404dca6e928`,
and the installed `libmqa_logits_kernels_xe_2.so` md5 190e261e…
(build25 pin). Posture = the r219b record posture exactly: ARC_GRAPH=1
ARC_SPEC=3 ARC_FP8=1, `ARC_KVDTYPE=<fmt> ARC_MXKV_NATIVE=1` on MX
cells, wheel+overlay baked in-image (no mounts). Faces via in-container
`p139_conc.py` (docker cp'd per boot, argv: ctx 192 1 tag model 0.0);
JSON polled with the r219b caps; report recomputed from the banked
JSONs (gate-script CWD law: verdict python reads banked artifacts, not
shell state). Warmup curl before the first face; W14 quiesced before,
restored via guarded trap after (HEALTH 200 at 17:11:49).

## Hygiene

Bank pulled AFTER a rig-side perl redaction of both private address
classes (rig + proxy-client) in the four serve.logs — the full-pattern
law from the cert27 battery close-out applied at bank time this round;
local post-pull scan zero across rfc1918. `r230_report_tail.txt` is
the report block from the launch log (pre-filtered for the address
classes; md5 of the filtered stream 81c8297cde31e07967debab6d4dbaa37).
Telemetry: `telemetry.txt` per cell (108 `MXKV native call` lines,
splits=auto); `erraudit.txt` ERROR=0 TRACEBACK=0 per cell (verified).
Count note: the in-log report prints `lines=156` — python
universal-newlines splits the tqdm `\r` fragments embedded in matched
serve lines into extra "lines"; `wc -l`/`grep -c` on the same bytes
give 108. The gate predicate (`lines >= 1`) is unaffected either way.

Artifacts: `r230/{ctrl,m4q,m8q,mkq}/` — boot.txt, serve.log
(redacted), telemetry.txt, erraudit.txt, and per-face
`c{8192,16384,32768}_n1/{metrics_before.txt,p139_conc_c*_n1.json}`.
