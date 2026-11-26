# P6.47 P-H2 — r249 LMC × MX interop validation: VERDICT

**INTEROP_PASS — the P-H2 fix is validated end-to-end: the LMC × MX
lane (ARC_LMC=1 over mxfp4, the exact r240b death posture) now stores,
retrieves, and restores byte-true on the real serve path, with zero
transfer errors. Two runs; run 1 passed the original four gates but
exposed one residual defect in its posture, fixed in the same session
(one line), and re-proven by run 2 with a fifth gate added.**

## The validated fix (two pieces, both in the fork-owned lmcint adapter)

1. **Metadata alignment** (`_init_lmcache_engine`, r249): the MX
   dtypes (`mxfp4`/`mxfp8`/`mxk8v4`) take the packed-row branch —
   `kv_dtype=uint8`, `kv_shape=(L, 2, chunk, 1, H_kv*slot_bytes(D,fmt)//2)`
   (= `(17, 2, 1600, 1, 272)` at D=256/mxfp4, H_kv=2), mirroring the
   p614 TurboQuant template with the slot width from
   `mx_layout.slot_bytes` — the single source of truth. This aligns
   the engine metadata with the byte-true pool view
   `_arc_attn_pool_view` already hands the connector
   (`(493, 1600, 1, 2, 272)`, 544 B/token). The r248 triage proved
   the old 512-vs-544 B/token disagreement misaddressed every
   transfer (the r168 re-hit garble, the r240b sustained-sweep
   wedge/device-lost).
2. **Connector hidden** (`_packed`, r249b): `hidden_dim_size` must be
   the packed row half for the MX lane too (`_packed = _tq or
   bool(_mx_fmt)` → 272). Run 1 caught this: the KV_2LTD
   `tmp_gpu_buffer` is built from `hidden_dim_size`, and leaving it
   at the standard `H_kv*head_size = 512` made every lagged
   `from_gpu` store throw `RuntimeError: tensor a (272) vs b (512)
   at dim 3` — swallowed by the catch-all as "Lagged attn store
   failed (cached chunk absent)". Pre-fix, both sides said 512 and
   the copy "succeeded" while misaddressing; post-fix it fails loudly
   until the hidden is aligned.

## The evidence chain (run 2 = this bank's root; run 1 = run1/)

| Gate | Readout | Run 1 | Run 2 |
|------|--------|-------|-------|
| G1 packed metadata | `use mla: False, kv shape: (17, 2, 1600, 1, 272)` | PASS | PASS |
| G2 clean face (c8192 n1) | n_ok=1 | PASS | PASS |
| G3 re-hit A/A | fixed ~8K-token prompt twice, one boot | texts equal, 6.15→1.88 s | **texts byte-equal, 6.18→1.89 s + metric-window proof** |
| G4 r240b death sequence (c32768: n1 → same-tag re-hit → n2 → n4, mt600) | n_ok full, no wedge/device-lost/timeout | PASS (1/1/2/4) | PASS (1/1/2/4) |
| G5 clean transfers | tracebacks / LMCache ERROR lines | **96/96 — the residual** | **0/0** |

**Store-size classes are engine-attributing telemetry.** The "Stored N
tokens, size: X GB" lines decompose per engine: attn chunks at the
correct new geometry = 0.0138 GB/1600-token chunk (17×2×1600×1×272 B
exactly), mamba chunks unchanged at 0.0379 GB (the seed-bytes path was
never affected). r248 (pre-fix): 0.0259-class attn (the misaddressed
rows) + 0.0379 mamba. Run 1: ONLY 0.0379 — zero attn stores survived.
Run 2: 0.0138-class singles + 2/4/5-chunk linear forms + 0.0379
mamba — both engines storing.

**The store-count arithmetic closes the residual.** Run 1: 72 stored
+ 96 lagged-dropped. Run 2: 168 stored, 0 dropped — 72+96=168
exactly; the fix recovered the entire lagged class.

**The re-hit hit-proof (beyond wall time).** LMCache retrieval is
log-quiet at INFO (retrieve_lines never moved off the 4 config-dump
baseline), so the G3 hit is proven by the engine metric windows in
serve.log: pass 1's 10 s window shows 3205.9 prompt-tokens/s (the
full ~8K prefill at ~3200 tok/s ≈ 2.5 s); pass 2's window shows
**0.0 prompt-tokens/s** while a 148-byte byte-equal completion
streams — the prefill was genuinely skipped over the retrieved
chunks. Byte-equal text at temp 0 over MTP k4 is the corruption
oracle: the restored state reproduced the exact decode trajectory
(the r168 garble class).

**Sustained survival.** The exact r240b kill sequence — c32768 n1 →
same-tag re-hit → n2 → n4 at mt600 on ONE boot — served n_ok 1/1/2/4
with zero wedge, zero DEVICE_LOST, zero watchdog timeouts (the class
that killed kvm4/kvmk after 274/306 store lines under the corrupted
restore path). Mean acceptance 3.95–4.00 throughout (the mxfp4 lane's
r238-mined class — unchanged by LMC).

## What ships

- `vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py` — both fix
  pieces (md5 bd7065c79fe14053c9c2a39fa6179226, the exact bytes
  validated by run 2 via the ARC_PD_LMCDIAG_AD mount).
- `tools/serve_arc_b70.sh` — byte-parity with the rig artifact
  (md5 93ac8477545352886f43fce92e84ac16): the two env-optional
  lmcint mounts + the `VLLM_ARC_LMC_MX_GATE` pass-through (unset =
  the certified gate-ON posture byte-identical).
- `tools/ar/r249_lmc_mx_interop_e2e.sh` — the five-gate validation
  runner (the G5 gate is the run-1 law made structural);
  `tools/ar/r249_pull.sh` — the bank pull with the law-8 redaction.
- The r248 triage bank + tools (P-H1) ride the same increment.

## Laws banked by this leg

1. **Packed-lane hidden law**: for EVERY packed-row lane the
   connector's `hidden_dim_size` MUST equal kv_shape's collapsed
   NH×HS — or the lagged `from_gpu` path builds its tmp buffer at the
   wrong width and the sync path silently diverges from it.
2. **Store-size classes are the transfer telemetry**: the per-store
   "size:" values attribute engine + geometry per chunk (attn 0.0138
   / mamba 0.0379 at this posture; 0.0259 = the old misaddressed
   geometry) — read them before reading anything else.
3. **LMCache retrieval is log-quiet at INFO**: hit-proof comes from
   the engine metric windows (prompt-throughput 0.0 during the
   would-be prefill), never from retrieve-line greps.
4. **The catch-all lie**: `except Exception: "cached chunk absent"`
   rewrites every transfer failure into a graceful-sounding skip —
   gate on traceback/error counts (G5), not on the message's tone.

## Posture / bank

cert28 + the P-D overlay trio + the patched adapter mounted over the
lmcint kit; gate bypassed for the leg (`VLLM_ARC_LMC_MX_GATE=0`,
the serve-script pass-through). Boot 17:10:24Z → done 17:19:50Z.
`boot.txt serve.log posture.txt outcome.txt failed.txt
kvshape_line.txt g1..g5.txt warmup.* aa1/aa2.{json,text}
aa_times.txt aaprompt.json p139_conc_*.json launch.log
SHA256SUMS(.rig)`; run 1's full artifact set under `run1/` (the
96-error evidence). W14 restored + health-200 verified post-leg.

## Open follow-ups (P-H4 / later)

- The interop is validated for **mxfp4**; mxk8v4/mxfp8 lanes take the
  same code path (the branch is dtype-generic via `mx_layout`) but get
  their own boot-face validation when their lanes are next exercised
  (P-I).
- gemma × LMC stays REFUSED (the P-D gate, ARC_LMC=0 posture — the
  two-head-size hybrid has no single-HS KV_2LTD mapping; cert17 law).
