# VERDICT — r245a: the page-128 admission for the D=512 full-attention pool (P-C1 addendum)

**Date**: 2026-11-26 (r245 e2e death 06:38 UTC; oracle runs 11:20–11:57 UTC; e2e re-run launched 12:07 UTC)
**Image**: `vllm-arc:v0.30.0-b70-cert28` (offline harness, NO engine boot)
**fork_sha**: `a3d39523` (main at PR #148 — the P-C2 window surface)
**Wheel**: `vllm_xpu_kernels-0.1.14.1+arc7mx` **build6/r245a** md5 `95451e404b463d9d6efa4bd4a4939b26` (build34 = build5 + the admission; `build-launchers/r245a_build.sh`, gates + `.r245a.md5` sidecar)
**Harness**: `tools/ar/p147_mxkv/p245a_p128_offline.py` (rev 3) via `tools/ar/r245a_p128_offline.sh` (W14 quiesced, EXIT-trap restored)
**Cells**: d512p128 (the gemma full-attention pool geometry) + d512p64 / ctrl256p64 regression guards on the same wheel

## VERDICT: PASS — page 128 is an admission, not a new kernel surface

The mainloop's page indirection is runtime-generic at every site
(walk L918–926: `tiles_per_page = page_size/16`, `page_local_idx =
tile*16/page_size`; early-exit L1150–1155; prefetch L1465–1473), the
auto-split heuristic hard-codes the 16-row tile unit (`mx_get_num_splits`
called with `16`, the r212c2 law), and `k_blocks` is seq-derived — so
admitting `block_size == 128` needs only the TORCH_CHECK text + a
dispatch branch riding the certified `_512/_16` frame. The oracle
proves it: 40 faces green, zero fails.

- **exact** (fp64 on dequantized cache, all 3 formats): d 9.70–9.74e-04,
  cos 0.9999986 — identical to the r242 page-64 authority numbers; the
  v2 triangle composes (d_v2 1.41–1.94e-07 — the Triton decoder is
  clean at page 128 too).
- **v2** (native vs the certified Triton decoder, same cache): passes
  the 1-ulp output law (d 9.72–9.76e-04, d_rounded ≤ 4.88e-04).
- **bitwise** (seq_len==1 dequant bit law): 0/8192 mismatches × 3
  formats.
- **splits** (forced-8 + default at ctx 8192): d ≤ 4.24e-04 — the
  r242d 16-tile plan walks 512 page-local transitions per seq (64
  pages × 8 sub-tiles) and ReduceSplitK combines them clean.
- **boundary** (edge lens 1/63/64/127/128/129/255/256 through shuffled
  tables): the 127/128 and 255/256 lens cross the page-0→1 and
  page-1→2 boundaries where `page_local_idx` recomputes the physical
  row — clean at cos 0.9999998 (see the gate-history section: this
  face forced a face-law fix, not a kernel fix).
- **regression guards on the same wheel**: d512p64 exact/bitwise and
  ctrl256p64 (the certified qwen production lane) exact/bitwise all
  green with the r242-authority numbers — build6 carries the
  previously-banked lanes forward (kernel-binary-layout-dice law).
- **telemetry** (page-128 native decode): ctx 40960 — mxfp4 1.42 ms
  (3.92x v2, 250.7 GB/s), mxfp8 1.87 ms (3.34x, 369.3 GB/s), mxk8v4
  1.78 ms (3.51x, 294.0 GB/s) — within noise of the r242 page-64
  numbers (mxfp4 3.95x / 252.8 GB/s): the walk cost is page-size-
  neutral, exactly as the runtime-generic indirection predicts.

## The death this run closes — the r245 e2e grid on build5

The r245 gemma e2e grid (P-C's e2e authority attempt) booted all three
cells healthy on build5 + the P-C2 overlay — `pc2_windowed_impl=100`,
`lmc_lines=0`, KV pools 105669/54426/71856, native windowed calls
(`window=1024`) succeeding at capture — and **died at the first real
request**: `RuntimeError: mx_paged_decode: block_size 128 not yet
compiled (arc7 covers page 16/64)` (serve traceback `engine.py:1261 →
native_decode.py:135`). The gemma hybrid KV manager banks the D=512
full_attention group a **page-128 pool** while the sliding D=256 group
rides page 64 — the telemetry confirms the geometry: the death call is
`bt_width=320 stride=1088 window=-1` (40960/128 blocks; 544 B slot ×
HK=2) while the sliding calls (`bt_width=640`, window=1024) had all
succeeded. /health never runs decode, so the boot was green while the
first completion killed the engine worker.

All 37 banked faces were n_ok=0 (Connection refused) behind green
`CONC_OK` lines — the existence-only-green-gate lesson applied: the
hardened e2e runner now gates semantically (n_ok>0 per face;
a non-200 warmup banks `WARMUP_DEAD` with the fresh traceback, since
the warmup completion IS the earliest real-request probe). Failure
evidence preserved in `failrun/` (per-cell posture, traceback excerpts,
conc logs, native telemetry, the 37-face dead tally) and on the rig at
`/root/build/p647/r245_pre_r245a_failrun`.

## Gate history — two FAILs that were face-law gaps, not kernel defects

- **Run 1 (harness rev 1)**: everything green except `boundary` —
  d=2.27e-03, 13 pct over the 2e-3 absolute band. The mini ledger
  decomposed it as intrinsic: `d(F32,ref) == d` (an honest fp32
  pipeline deviates identically), the steps>1 census was **one**
  element — a 7e-7 near-cancellation sum where F32 sits at 7.8 ulps
  vs the kernel's 6 — and the absolute-d driver was a single bf16
  output ulp at a ~0.58-magnitude element.
- **Run 2 (rev 2 — the v2 triangle added)**: v2 itself banked
  d_v2=2.33e-07 on the payload (fp32 accumulation, no output
  quantization — clean), so the `+d_v2` composite could not widen,
  and the same two element classes tripped it. The finding: the 2e-3
  absolute band is the 1-output-ulp law at 0.5 magnitude (ulp(0.5) =
  1.95e-3) — a payload with max elements above 0.5 has its 1-ulp step
  **above** the band; the band is blind to magnitude. And near-
  cancellation outputs (|ref| far below the row's PV scale) violate
  the per-element ulp law under ANY accumulation order — every
  previously-banked payload contains them; faces passing via the
  absolute band never noticed.
- **Run 3 (rev 3 — the F32 discriminator promoted into the gate)**:
  the ulp law now excludes elements where the honest fp32 pipeline
  itself exceeds 1 ulp vs bf16(ref) — the mini-ledger's already-banked
  discriminator, promoted into `_gated` with zero free parameters.
  It cannot fire on corruption (a corrupted region deviates far
  beyond the fp32 order-noise class and cos collapses regardless).
  Result: PASS with `n_f32_excluded=1` — exactly the one census
  element — and every other element at ≤ 1 ulp. All runs' logs and
  JSONs are banked (`out/`, `logs/`); the face-law evolution is
  documented in the harness and runner headers.

This is gate calibration on evidence, not loosening: the absolute band,
the cos guard, the v2 triangle, and the population census are all
intact; splits walked 512 page transitions per seq at d ≤ 4.24e-04 — a
page-indirection defect corrupts rows, not one near-cancellation tail.

## What stands

- The e2e re-run on build6 launched 2026-11-26 12:07 UTC (the
  hardened runner: n_ok>0 semantic gates, warmup-death detector,
  build6 sidecar refuse) — its bank is the P-C phase close-out.
- The masked twin remains compiled for D=256 at page 16/64
  (qgroup ≤ 8); the D=512 classes stay windowless (same `!is_local`
  refusal as 512/64 — gemma full_attention layers carry
  sliding_window=None; a windowed D=512 model needs the masked twin
  instantiated on that model's evidence).
- The next dispatch-widening candidates refuse loudly at the
  TORCH_CHECK with the compiled list now "128/16, 256/16, 256/64,
  512/64, 512/128".

## Evidence

- `failrun/` — the r245 build5 death: 3 cells' posture/traceback/conc/
  telemetry + `dead_faces.json` (37 faces, 0 alive)
- `out/r245a_summary.txt` + `p245a_p128.json` — run 3 (PASS, rev 3)
- `out/r245a_run1_summary.txt` + `p245a_p128_run1.json` — run 1 (the
  boundary FAIL + mini ledger)
- `out/r245a_run2_summary.txt` + `p245a_p128_run2.json` — run 2 (the
  v2-triangle attempt)
- `logs/r245a_run{1,2,3}.out` — full harness logs
- `build-launchers/r245a_build.sh` — the build34 launcher (gates +
  wheel sidecar); `r243a_build.sh` for lineage
- `SHA256SUMS` at bank root covers all banked files
