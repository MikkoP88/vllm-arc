# P6.47 P-H1 — r248 LMC × MX interop triage: VERDICT

**TRIAGE COMPLETE — the root issue is located, pinned by instrumented
telemetry on the real serve path.** Leg: cert28 + the P-D overlay trio
(gate bypassed for the leg via the new serve-script knob pass-through),
ARC_LMC=1 over mxfp4 (the exact r240b death posture), two diag
instruments mounted over the lmcint kit (env-optional serve mounts —
the detector census + the adapter pool-view/registration census).
Two faces on one boot: the clean-store face (c8192 n1 mt64) and the
exact r240b death face (c32768 n4 mt600). Runner
`tools/ar/r248_lmc_mx_triage.sh`; serve extension
`tools/ar/r248_serve_patch.sh` (diag mounts + the
`VLLM_ARC_LMC_MX_GATE` pass-through; unset = certified gate-ON
posture byte-identical).

## The root issue (the telemetry chain, banked in diag_full.txt)

1. **The MX attn pool registers as a packed LBNHC uint8 tensor**
   `(12325, 2, 64, 272)` — strides `(34816, 272, 544, 1)`: logical
   `(NB, H_kv=2, N=64, slot=272)`, physical `(B, N, H, C)` — the
   SAME geometry family as the turboquant pools (the mx_dtype_map
   elem-class alias, P6.47 leg B).
2. **The pool-view side classifies the MX lane as PACKED**:
   `_arc_attn_pool_view` runs the TQ branch (`tq_attn=True
   tq_slot=272`) and hands lmcache the correct packed view
   `(493, 1600, 1, 2, 272)` — per-token row = H_kv × slot = **544 B**,
   `tq-page-merge` at the 1600-token chunk granularity. The view is
   byte-true.
3. **The metadata side classifies the SAME lane as STANDARD**:
   `_init_lmcache_engine` builds `kv_shape=(17, 2, 1600, 2, 256)`,
   `kv_dtype=uint8` — per-token row = kv_heads × head_size = **512 B**
   (the `_tq` test reads the raw dtype-string prefix `turboquant_`;
   `mxfp4` fails it, so the standard-shape branch runs).
4. **The 544-vs-512 disagreement is the misaddress**: every
   store/restore transfer walks the pool with 512-B rows over 544-B
   physical rows — a cumulative 32 B/token drift. Stored chunks are
   drift-scrambled; a restore writes the scrambled bytes back into
   live pool rows, clobbering E8M0 scale bytes (255 = NaN) and packed
   payload.
5. **The load dependence explains every prior observation**: my
   isolated death face SURVIVED (n_ok=4, 96 store lines, zero
   tracebacks, no wedges) — single-pass faces never re-hit, so the
   corrupt-restore path never fires. The r240b deaths (kvm4 watchdog
   wedge after 274 store lines; kvmk UR_RESULT_ERROR_DEVICE_LOST
   after 306) came in the SUSTAINED grid where the t1.0 re-hit faces
   restored drift-scrambled chunks into the pool mid-sweep; the r168
   quality garble (m4 essay re-hit REP-LOOP) is the same restore-path
   corruption at low load. The KVGATE passes throughout (rows=[272]
   uniform) — it checks uniformity, not consistency with the
   metadata's geometry, so it cannot see this class.

## The fix (P-H2, scoped by this leg)

Align `_init_lmcache_engine`'s classification with the pool-view
reality: MX dtypes (`mxfp4` / `mxfp8` / `mxk8v4`) take the packed-row
branch — `kv_dtype=uint8`, row = `H_kv × slot_bytes(D=256, fmt)`
(mxfp4 544 / mxk8v4 800 / mxfp8 1056 B/token, all even), `kv_shape =
(L, 2, chunk, 1, row//2)`, `hidden_dim_size = row//2` — exactly the
banked view geometry. The p614 TQ pattern is the template; the
`mx_layout.slot_bytes` function is the single source of truth for
the slot width. No lmcache-side change, no wheel change, no upstream
change — the whole fix is fork-owned (the lmcint adapter).

## P-H3 validation gates (from this leg's findings)

1. Offline round-trip oracle: store → restore → byte-diff vs the
   never-stored control (byte-exact), over the real lmcache native
   transfer (ref-harness law — no stubs).
2. e2e RE-HIT battery: store → re-query the same prefix → generation
   A/A equality vs LMC-off (the face-tag prompt-seed law governs the
   seeds), acceptance unchanged, zero LMCache-error lines.
3. The r240b-shaped sustained sweep (the full kvm4/kvmk face lists
   with the t1.0 re-hit faces) on the patched build: no wedge, no
   device-lost, n_ok full at every face.

## Bank

`diag_face1.txt` / `diag_full.txt` (the instrumented telemetry),
`serve.log` (full, redacted), `posture.txt` (boot posture + face
outcomes), `outcome.txt`, `p139_conc_r248f*.json` (the faces),
`boot.txt`, `warmup.*`, `launch{,2}.log` (run 1's knob-reach defect +
run 2), `SHA256SUMS`.

Run 1 defect (banked in launch.log + the runner header): the runner
exported `VLLM_ARC_LMC_MX_GATE=0` but the serve script's docker-exec
env block never passed host `VLLM_*` vars — the P-D gate refused the
boot at connector construction (16:27:32Z, zero diag lines). Fix:
serve patch rev 2 (the `${VLLM_ARC_LMC_MX_GATE:+export ...}` line).
