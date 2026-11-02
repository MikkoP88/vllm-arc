# p614 L3 — remaining TurboQuant preset legs on p614c2: ALL GATES PASS (2026-11-02)

**Context:** owner directive 2026-02 (fix + max out ALL turboquant
types). After L1v2 (4bit_nc) and L2 (default + slice faces), the three
remaining first-class TQ presets ran the same one-variable leg
(`--kv-cache-dtype` only; script `../2026-11-02-p614-l1v2/rig/p614_tq_leg.sh`,
post-hardened with the rm-race + corpse-port fixes, see below) on
`vllm-arc:v0.30.0-b70-p614c2`. Baseline fp8 = cert5 family.

## Chain-race failures (operational, NOT image defects) and reruns

The three legs were first launched back-to-back in one chain. k8v4 ran
clean, but its gate-summary crash (a script `set -u` echo bug, fixed)
fired the EXIT trap's W14 restore, and the next legs then raced that
W14 boot/shutdown: k3v4_nc and 3bit_nc workers died at
`init_device` (GPUs still held by the dying/booting W14 serve; 64
ERROR / 5 Traceback each; the 3bit health wait even passed against the
W14 corpse on :8000). Evidence banked in `l3_k3v4/` + `l3_3bit/`
(serve logs show only the WorkerProc crash loop). Fixes, now permanent
in the leg script: (1) remove arc-test and WAIT until `docker inspect`
says gone (the L2 slice rm-race lesson), (2) after `docker stop
radiance-test`, WAIT until :8000 stops answering before boot. Reruns
`l3r_k3v4/` + `l3r_3bit/` from a stable W14-up state: ALL GATES PASS.

## Per-preset matrix (same harness, same rig, same image)

| Preset | slot→HS | G1 err/tb | G2 kit lines | G4 GC | integ/det | spec pos1/acc | flip | t0 agg#2/solo | KV capacity | U32k/U64k TTFT | gap p95 U32k/U64k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fp8_e4m3 (cert5) | — | 0/0 | (r43 face) | 0 | 5/5+3/3 | 0.9258 | 8.3 s | 297.19/84.48 | 405,491 (1.55×) | 23.16/50.77 s | 1299/1569 ms |
| turboquant_4bit_nc (L1v2) | 262 | **0/0** | **2+2 @262** | **0** | **5/5+3/3** | 0.9533/0.9159 | 10.3 s | 300.06/86.31 | 757,641 (2.89×) | 15.91/34.02 s | 3218/3785 ms |
| turboquant_k8v4 | 388 | **0/0** | **2+2 @388** | **0** | **5/5+3/3** | 0.9450/0.8945 | 9.5 s | 302.58/88.05 | 526,445 (2.01×) | **15.29/33.03 s** | 3690/4385 ms |
| turboquant_k3v4_nc | 230 | **0/0** | **2+2 @230** | **0** | **5/5+3/3** | 0.9550/0.8649 | 9.6 s | 294.43/83.44 | **853,220 (3.25×)** | 15.48/33.78 s | 3736/4186 ms |
| turboquant_3bit_nc | 198 | **0/0** | **2+2 @198** | **0** | **5/5+3/3** | 0.9577/0.9218 | 12.6 s | 302.77/86.25 | **975,329 (3.72×)** | 16.02/35.74 s | 3188/3729 ms |

External tier participates in every preset (hit-rate max 5.9/6.2/7.6 %
through the integrity re-hits). t1.0 xhigh (the A/B's −53 % quality
face): **k8v4 agg 173.39 (−8.5 % vs fp8's 189.41 — by far the best TQ
preset)**; 4bit 93.05, k3v4 88.95, 3bit 89.58; every TQ preset shows
the ragged solo pair (~77–86 + ~26 tok/s) — sampled-throughput quality
cost of quantized KV, dtype-wide, owner-call material per the A/B
disposition.

## Reading

- The fix works for EVERY first-class TQ preset at its own live
  geometry (HS 262/388/230/198; k8v4 even merges to a different page
  size BS=1152 — the spec-derived page merge self-adapts).
- Capacity ladder behaves exactly as the slot math predicts: 3bit_nc
  3.72× > k3v4_nc 3.25× > 4bit_nc 2.89× > k8v4 2.01× (vs fp8 1.55×).
- TTFT wins hold across presets (−31…−34 % U32k, −30…−35 % U64k);
  best absolute TTFT on k8v4.
- Bystander gaps stay the TQ-inherent ~3.2–3.7 s / ~3.7–4.6 s p95
  (fp8: ~1.3/1.6 s) — unchanged by the fix, as in L1v2.
- Recommended per use: k8v4 = quality-safest (t1.0 within 9 % of fp8,
  acceptance above target); 3bit_nc/k3v4_nc = max capacity (3.7×/3.3×)
  where sampled-throughput quality is acceptable; 4bit_nc = middle.

Rig addresses/hostname redacted byte-exactly (deltas +125/+6/+6/+125/
+139 and +5×4 hostname; appended to `../REDACTION-2026-11-02-p614.md`).
W14 restored by trap on every leg (recreate law honored throughout).
