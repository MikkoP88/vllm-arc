# Changelog — vllm-arc

Every certified release and closed phase, newest first. The living docs
(README, `docs/*`) carry **current state only**; superseded doc content
is archived per-doc under `docs/history/` (main-currency law —
`tools/check_currency.sh`). Facts in each entry are
the durable summary of the banked evidence; full narratives live in the
cited `results/` legs and the close-out PRs. Tags are annotated and point
at the certified `main` commit; `IMAGE_MANIFEST` inside each image carries
the fork SHA it was built from.

Phase labels P0–P6.38 refer to the plan's execution phases; every phase's
raw evidence is banked under `results/YYYY-MM-DD-<phase>-*/`.

## r246 — P-D LMC x MX boot gate: the r240b posture confound made structural (2026-11-26)

The engine itself now refuses the posture every r240b death shared:
an LMCache-named connector over an MX KV dtype. `patches/lmc_mx_boot_
gate.py` exec-hooks `KVConnectorFactory.create_connector` (beside the
stock factory's own HMA refusal — same method, same shape) and raises
`ARC_LMC_MX_REFUSED` at connector construction; `VLLM_ARC_LMC_MX_
GATE=0` is the A/B-evidence-only bypass knob (envs.py). Scope is
evidence-bounded: non-LMC connectors over MX pass through, and LMCache
over non-MX dtypes stays the lawful certified lane.

Validated on both layers (bank:
`results/2026-11-26-p647-pd-lmc-mx-boot-gate/`):

- **Offline probe (r246 run 4): 11/11** over the real factory wiring —
  hook fired, refusal over mxfp4/mxfp8/mxk8v4, the real registry
  resolves `LMCacheConnectorV1` (detection basis), non-LMC scope
  pass-through, lawful non-MX pass-through, knob bypass posture,
  stock None-config ValueError intact, idempotence.
- **e2e (r246b run 3): both cells green** — the exact r240b confound
  posture (ARC_LMC=1 over mxfp4) dies at construction with 5x
  `ARC_LMC_MX_REFUSED` in the serve log; the lawful posture
  (ARC_LMC=0) boots HEALTH_OK with the gate hook installed (4 install
  lines), zero LMCache lines, and serves a real face (CONC_OK n_ok=1).

Two defects found and fixed on the way, both banked in the runners'
headers — **the validation legs earned their keep twice**:

1. **Runner poll defect (r246b run 1)**: the refusal marker poll broke
   at iteration 1 on the not-yet-created container and pkilled a
   healthy 14-second boot — the connector constructs minutes into
   init, after model load. Fixed: wait for the container, then a slow
   poll (marker / health / EngineCore liveness); health=200 over the
   confound posture is the genuine-leak discriminator.
2. **The 0.30 cache_dtype rename (r246b run 2)**: the gate read the
   pre-rename attr `cache_config.kv_cache_dtype` — vllm 0.30 renamed
   the field to `cache_dtype` (config/cache.py:111, no legacy alias;
   the CLI kwarg and the AttentionSelectorConfig tuple kept the old
   name) — so the live engine surfaced `kv_dtype=None` and the gate
   passed the confound posture through to health. The offline probe
   masked it: its stub invented the pre-rename attr (the ref-harness
   law — only the e2e leg over the real config object catches
   attr-name drift). Fixed: the gate reads both surfaces (legacy
   first, then the 0.30 field); the probe models the real surface and
   pins the chain (B_legacy, 11 faces).

The gate rides cert29 from the build6 lineage. Posture law unchanged:
LMC off on every MX-KV lane — now enforced by the engine, not just the
harness.

## r242–r245 — P-C hd512 MX-native decode + sliding-window phase: gemma hybrid MX-KV served e2e at D=512/page-128 (2026-11-26)

The gemma-4-31b first-decode refusal named in the cert29 gate is
closed end-to-end. Four increments, PRs #147/#148/#149 + the r245 e2e
close-out:

- **r242 (P-C1)**: the hd512 kernel-instantiation gap. Build1's 16/16
  D=512 faces failed — the **V-slice law**: at D=512 q8 the output
  head size exceeds the single-slice V stage (`kDecodeMaxShapeOutV`
  256), every blk_v=1 workgroup re-staged slice 0's payload/scales
  (cos 0.49). Fix: `v_slice_bytes` staged once per slice at all three
  dense-frame sites; splits' plan tile units unified to the 16-row
  kernel tile (the r242d floor). 24/24 faces green on build5.
- **r243 (P-C2 kernel half)**: the MX-native paged-decode
  sliding-window surface (masked twin). Build4's splits face died
  `UR_RESULT_ERROR_DEVICE_LOST` — the **windowed work-list clamp**:
  split plans were built from FULL seq lens while the kernel walks
  the window-bounded list; r243a clamps the plan to the window's own
  bound. All faces green at both pages (16/64), W=64..2048.
- **r244 (P-C2 engine half)**: the stock TQ-lane prefill window. The
  vendor varlen op already threads `window_size` — the fork's engine
  never passed it. Proven on the vendor surface (windowed W=1024
  prefill **3.1x faster** at ctx 8192: 1.266 vs 3.898 ms); the
  engine override now passes it (50 sliding layers shed 75% of their
  prefill walk), boot-loud refusals guard the unsupported classes.
- **r245a (P-C1 addendum)**: the **page-128 admission**. The gemma
  hybrid KV manager banks the D=512 full_attention group a PAGE-128
  pool (sliding D=256 rides page 64) — the r245 grid on build5 died
  at the FIRST REAL REQUEST with `block_size 128 not yet compiled`
  while /health stayed green. The mainloop's page indirection is
  runtime-generic (tiles_per_page / page_local_idx computed from
  params.page_size at all three sites), so page 128 is admission
  only: TORCH_CHECK widening + a dispatch branch on the certified
  `_512/_16` frame. Offline oracle 40/40 (exact d 9.7e-4 / bitwise
  0/8192 / splits over 512 page transitions / boundary edge lens /
  same-wheel regression guards). The gate evolution banked the
  **F32 near-cancellation discriminator**: the 1-ulp law excludes
  elements where an honest fp32 pipeline itself exceeds 1 ulp (order
  noise at cancellation sums; zero free parameters; cannot mask
  corruption) — runs 1-2 failed on exactly that element class, run 3
  passed with n_f32_excluded=1, the census count.
- **r245 e2e (phase close-out)**: 3 cells (mxfp4/mxfp8/mxk8v4) × 15
  faces on the hardened runner. **45 scheduled = 37 CONC_OK (all
  n_ok>0) + 8 honest KV_CAP rows, 0 dead/timeout**; zero tracebacks
  and zero LMCache lines in every serve log; the death-signature call
  (`bt_width=320 window=-1`) executed clean in all three cells. The
  r245 failrun (37 n_ok=0 faces behind green gate lines) banked the
  **existence-only-green-gate law** that hardened the runner (n_ok>0
  per face; non-200 warmup banks WARMUP_DEAD). mxfp4-KV leads gemma
  tps at every depth (21.2 tps/user at 1K n1; 105K-token pool).
- **Wheel authority chain**: build4 → build5 `ead038b1` (r243a) →
  **build6** `95451e40` (r245a = build5 + page-128) — the cert29
  image builds on build6.

Evidence: `results/2026-11-26-p647-r242-hd512/`, `-r243-window/`,
`-r244-tq-window/`, `-r245-page128/` (oracle + failrun) and
`-r245-page128/e2e/` (the grid).

## r241 — r240b packed-KV finding re-classified posture-confounded and re-measured LMC-free: Finding 2 fully retracted (2026-11-25)

- **Root cause (banked evidence)**: the r240b qwen MX-KV cells ran
  `ARC_LMC=1` — inherited from the `$QF8` base env
  (`tools/ar/r240b_conc_cert28.sh`) — against the certified LMC-free
  MX posture and each boot's own WARN. LMCache actively stored in both
  death windows (kvm4 274 / kvmk 306 store lines; the surviving
  e4m3-V kvm8 logged 1206); every clean MX reference (r233 qwen MX
  cells, r234, r237 — zero LMCache lines) ran LMC-free. The deaths fit
  the r168-banked e2m1-V × LMCache interaction class, escalated to
  engine death under MT=600 sustained load.
- **Re-classification**: r240b Finding 2 ("qwen packed-KV
  sustained-decode stability boundary") is retracted as an
  *intrinsic-engine* claim — the evidence was posture-confounded. The
  banked artifacts stand; the ≥16K/low-concurrency packed-KV
  restriction is provisional pending the re-run. The certified law is
  unchanged (and was always the posture the deaths violated): LMC off
  on every MX-KV lane.
- **r241 re-measured clean — Finding 2 retracted in full**: the three
  MX grids (kvm4/kvm8/kvmk — exact r240b face lists, MT=600, cert28)
  re-ran under `ARC_LMC=0` with a semantic gate (per-face n_ok/n_fail,
  CONC_ENGINE_DEAD discrimination — the existence-only-gate law
  applied) and a posture refuse (zero LMCache lines or the leg
  aborts). Result: 68 real faces, zero dead or partial, zero
  tracebacks, zero LMCache lines in every serve log. Every r240b
  death face survived LMC-free — mxfp4-KV 32K × n4/n8 and 64K × n8,
  mxk8v4 32K × n4/n8 — with full n_ok at every face. The
  ≥16K/low-concurrency packed-KV restriction is RETRACTED: the
  lanes are capacity-limited only (mxfp4-KV serves 128K at n1–n4,
  pool 721,994 tokens; e4m3-V 386,484; mxk8v4 504,832). Evidence:
  `results/2026-11-25-p647-r241-mxkv-lmcfree/` (VERDICT +
  SHA256SUMS).
- **Corrections landed**: PERFORMANCE/STATUS/CONFIGURATION passages
  reworded to the re-classification; r240b VERDICT correction
  addendum; generator notes + regenerated reports.

## r240 — deep-sweep close-out: the concurrency sweep + gemma 128K annex re-audited; two new engine boundaries banked (2026-11-25)

- **Campaign legs r240b/r240c pulled, VERDICTed, and re-audited**
  (recompute law — the gate lines alone were insufficient): r240b 22
  cells, 436 faces, **373 real**; r240c 8 cells, 78 faces, **30
  real**. The legs' `group()` gate printed `CONC_OK` on face-JSON
  existence and never checked `n_ok` — **85 banked faces across 6
  cells are connection-refused artifacts behind green gate lines**.
  VERDICTs written into both banks with the full census; the
  generator's every numeric cell gates on `n_ok>0` and all-vacuous
  lanes render as banked refusal classes, never as gaps.
- **Finding 1 — gemma native MX-KV cannot decode on this image
  (hd512 gap)**: all six `gkv_m*` cells boot and bank their pools,
  then the first decode request fatals — gemma's 10 full-attention
  layers are head_size 512, outside the arc7 `mx_paged_decode`
  instantiations (D=128/256 only, sealed in
  `mx_paged_decode_xe2.cpp:198-203`; never surfaced before because
  certified gemma legs ran fp8_e4m3 KV). On gemma `fp8_e4m3` is the
  only servable KV dtype; the native MX-KV lane is qwen-scoped. An
  hd512 instantiation is recorded as a reopened improvement channel.
- **Finding 2 — qwen packed-KV sustained-decode stability boundary**:
  mxfp4-KV and mxk8v4-KV both died mid-sweep at 32K × n4 sustained
  decode (worker silent-hang → the 300 s watchdog `TimeoutError`;
  `UR_RESULT_ERROR_DEVICE_LOST`) — mxfp8-KV alone survived the full
  5-ctx × {1,2,4,8} grid (22/22). No contradiction of r237 (those
  cells were MT=128 short-decode; r240b is MT=600 sustained — a new
  cell class). Deployment law: packed 4-bit-V KV at ≥32K context only
  at low concurrency, or the mxfp8 lane.
- **The annex's serving record — gemma 128K is a W4A16 exclusive**:
  `gm4x` (MXFP4 W4A16 weights + fp8_e4m3 KV at maxlen 131072,
  26/26 faces) boots a 288,972-token pool where every 8-bit lane
  refuses and serves the full rung ladder: 40.1 / 37.3 / 35.0 / 32.3
  / **28.3 tok/s** n1 decode at 1K→128K (TTFT 175.6 s at 128K;
  1K n8 aggregate 227 tok/s; KV_CAP boundaries banked).
- **Real sweep results**: 3×2 qwen quant × spec matrix (MTP-k3
  multiplier 2.8–2.9× on every weights lane; the W4A16-over-fp8 delta
  −2.6 % @1K then monotone widening +5.7 % @8K → +8.9 % @128K);
  pool ladders (TQ 398,363 → 959,909 = +141 %; native mxfp4-KV
  721,994 = 1.81× ctrl; gemma W4A16 128,033 = 2.28× fp8);
  KV-dtype menu n1 8K: fp8 97.2 vs MX 78.8/76.9/78.0 (0.79–0.81×),
  TQ k3v4 frontier 70.2 at 2.10× pool; gemma gf8 conc ladder
  (1K agg 27→178); 64K closure gf8x64 24.1 / gm8x64 23.7; five
  by-design refusals banked (gkv_auto, gkv_auto16, gkv_e5m2, gm8x,
  gkv_m8x131).
- **Docs**: every report under `docs/reports/` regenerated on cert28
  from the three campaign banks (generator honesty-corrected:
  refusal-summary tables, re-audited outcome cells, refusal-class
  legends); PERFORMANCE (stability boundary + model scope + the
  W4A16 128K-exclusive passage), STATUS (campaign landed; the two
  boundaries), CONFIGURATION (the knob paragraph carries the
  qwen-only scope + the sustained-decode law).
- **Evidence**: `results/2026-11-24-p647-r240-reports/` (VERDICT),
  `results/2026-11-24-p647-r240b-conc/` (VERDICT + 22 cells),
  `results/2026-11-24-p647-r240c-gemma128k/` (VERDICT + 8 cells),
  `results/REDACTION-2026-11-24-r240a.md`; legs
  `tools/ar/r240_reports_cert28.sh`, `tools/ar/r240b_conc_cert28.sh`,
  `tools/ar/r240c_gemma128k_cert28.sh`, pull `tools/ar/r240_pull.sh`,
  fill `tools/ar/r240_fill_verdicts.py`, reports
  `tools/ar/p139/p139_report.py`.

## r240-docs6 — gemma agg8 revalidation: harness-numerator artifact corrected (2026-11-25)

- **Revalidation request**: the gemma L1 agg8 row ("48.5 / 59.2 /
  45.1 → 361–474 tok/s aggregate") sounded too high — confirmed and
  corrected. The r239b gemma battery harness scored each stream as
  requested-2048-budget / stream-wall with no `ignore_eos` and no
  finish check; on this face streams stop early (finish=stop at
  ~460–760 tokens; engine `Running:` drops 8→7→6→5 inside every
  banked rep), inflating the client means ×2.3–3.1.
- **Honest cert28 values, recomputed from the banked raw artifacts**
  (engine 10-s telemetry integrated over the client rep windows —
  `reval_agg8.py` banked in the r239 bank; client algebra and engine
  integration agree to the token): **24.6–26.4 tok/s per stream
  steady under 8-way (solo 29.0 — the expected batch tax), 196–211
  tok/s aggregate at full 8-way, 148–168 tok/s per-rep wall
  aggregate** (13,204 / 11,875 / 11,631 tokens over 78.4 / 77.4 /
  78.4 s spans). Same-lane 8×256-protocol battery (all streams
  complete): 195.9–196.6 aggregate — **parity held; the 1.85–2.4×
  improvement claim is retracted**.
- **Scope audit — every other aggregate in the docs is clean**:
  the qwen agg8 numbers (r233/r236 batteries, `agg8 0.84× ctrl`,
  307.8 ctrl) use `r113_battery.py` (numerator = actual
  `usage.completion_tokens`); the r240a/b/c conc probes use
  `p139_conc.py` (`ignore_eos: true`, `agg_tps_wall = conc·mt/wall`);
  gemma solo 29.0 validates against engine-side rates directly.
- Corrections landed: PERFORMANCE-arc gemma L1 row + accounting note
  + lane-parity passage (gemma agg8 is 1.55–1.70× qwen's no-spec
  aggregate and 0.64–0.69× qwen's MTP-k3 aggregate — the 2.8–3.7×
  claim is gone), STATUS-arc campaign bullet, this file's r239 entry
  (retraction marker), and the r239 bank VERDICT correction addendum
  (raw artifacts + SHA256SUMS untouched).
- Gates at file-write time: validate_docs 0 errors / 0 advisories
  (27 fork docs), leak_sweep clean, check_currency clean at cert28.

## r240-docs5 — deep-sweep pass 5: current-value presentation + owner-note removal (2026-11-25)

- **Improvement-vs-history framing removed from every current-facing
  doc** — documents present the current certified image's measured
  values, not how much better they are than past measurements. Every
  "re-discriminated / re-verified / beats its historical anchor /
  reproduces its banked anchors" clause is restated as the current
  record; the deltas live only in the history homes (this file's
  per-leg entries, `docs/history/`, superseded release-note sections):
  - PERFORMANCE-arc: the "Current certified image" preamble presents
    the build28 wheel as engine content and the r234 faces as the
    record (drops "cert27 + … carry through unchanged", the
    0.999–1.000×-banked-anchor and worst-ratio-1.001 clauses); the
    identity row's carried-record provenance becomes record provenance
    (measured natively on this image); the MXFP4 checkpoint-lane block
    drops "every face beats its historical anchor by 1.5–3.5×" (the
    p50-vs-ctrl-rows fact stays); the AutoRound and gemma-L1 lead-ins
    become plain "cert28 (r239/r239b)"; the gemma 131072 refusal drops
    "now".
  - README: the release bullet reads "measured on the shipping image
    itself"; the champion-faces preamble drops "E2E-neutral vs the
    measurement eras"; the Repository-model tag row and the build
    comment present build28 as the image content (no cert27-derivation
    phrasing).
  - STATUS-arc / MIGRATION-arc / CONFIGURATION-arc / release_notes
    (current section): the cert-release image identity rows present
    build28 directly (drops "cert27 content + the r231d …" in all five
    sites); MIGRATION's r234 line and CONFIGURATION's KV-dtype row drop
    their vs-banked-anchor clauses; release_notes' cert28 section drops
    "2K −7.6% vs cert27" and "worst anchored ratio 1.001".
  - envs.md / CONFIGURATION-arc: the CCL fp8-wire rows state "beats
    its OFF-phase control (A/B/A/B)" instead of "beats the banked
    baseline"; the CONFIGURATION recipe row drops the pre-cert10
    absence caveat; the drafting table cell drops "beats its greedy
    bank" (the vs-the-greedy-lane framing stays — the greedy lane is a
    same-image env opt-out, a contemporaneous A/B).
- **Owner-note language removed from tracked files** (the standing
  residue law): the two doc mentions ("the owner-directed decode
  goal/lane") and six CHANGELOG mentions ("(owner directive)", "owner
  question resolved") are gone — the docs read as project records.
  Grep-verified zero `owner` matches across current-facing docs.
- **Legitimate survivors (audit-verified)**: contemporaneous
  same-image lane comparisons (TurboQuant vs fp8, W4A16 vs mxfp8,
  interleave ON vs OFF, greedy vs probabilistic posture, GEMM-variant
  rankings), component-provenance stamps (per-patch "Cert: certNN",
  the r231d clamp leg name in the wheel identity), bank names with
  era tokens, and the designated history homes.
- Gates at file-write time: validate_docs 0 errors / 0 advisories
  (27 fork docs), leak_sweep clean, check_currency clean at cert28.

## r240-docs — MTP k3 unification + README feature-table completion (2026-11-25)

- **MTP depth presentation unified on k3** — the certified, overall-best
  depth — across every current-facing doc. The README
  performance table's two k5 rows are gone (the TTFT row now carries the
  banked k3 value: **3.76 s @ 8K** fresh-context, best-TTFT opt-in
  `CCL_SYCL_ALLREDUCE_FP8=1` 3.27 s); the spec-decoding row reads
  "k3: the certified, overall-best depth … deeper drafts measured and
  dropped". The STATUS-arc recommended decode posture block — the main
  violation, it still **recommended `ARC_SPEC=5`** with cert23-era k5
  numbers (115.2/123.9/128.3 tok/s) — is rewritten to the certified
  lanes (fp8+k3 93.8 tok/s @ 1K default; **mxfp4-W4A16+k3 103.3 tok/s
  @ 8K** best online-quant, `ARC_SPEC=3`). PERFORMANCE-arc identity row
  + close-out paragraph, PATCHES-arc champion-lane parenthetical (now
  the certified k3 read), CONFIGURATION serve-line lineage (stale
  `r43_serve_k5.sh` mention), MIGRATION golden-ref parenthetical, and
  the envs double-buffer E2E-note pair all follow. Deeper-draft values
  survive only in decision records (this CHANGELOG's history,
  `docs/release_notes.md`).
- **README "What this fork adds" completed to the full implemented
  feature set** (10 → 14 rows, grouped by subsystem): the MX-native KV
  cache lane (`mx_kv/` + `ARC_KVDTYPE`/`ARC_MXKV_NATIVE`), the
  graph-exec double-buffer, graph-safe output, speculator full-capture
  and first-request warmup graph patches, online MXFP4-W4A16 +
  MXFP8-W8A16 quantization routes, the attention image-overlay routing
  (launcher table / warmup gate / FA2 full + sliding-window dispatch)
  + GDN metadata fastpath, the TurboQuant MQ verify kernel (own row),
  the lmcint kit, and the wheel-pins/wheel-guard distribution row —
  every row cites its module paths. A dedicated **change-log bullet**
  links CHANGELOG.md from the README top block.
- **Currency sweep**: grep-verified every current-facing doc cites
  cert28 as the image to run (zero older-tag run recommendations);
  quick start / build / configuration sections read against the cert28
  serve line and image identity. No history rewritten — `docs/history/`
  and per-cert entries untouched.
- Gates at file-write time: validate_docs 0 errors / 0 advisories
  (27 fork docs; one table column-count fix — escaped pipes in the
  MX-native row), leak_sweep clean, check_currency clean at cert28.

## r240-docs2 — README best-practice alignment + deep staleness sweep (2026-11-25)

- **README restructured to GitHub README/DMS best practice**
  (deep-analyze best practices, align the fork): badge row
  under the title (release / license / Python / vLLM core), a Contents
  TOC, and a new **Dependencies** section — the full certified stack as
  a table (hardware, host runtime, Python 3.12, vLLM `0.30.0+xpu`,
  torch `2.13.0+xpu`, triton `3.7.2+xpu`, `vllm_xpu_kernels
  0.1.14.1+arc7mx` build28, LMCache `0.5.6+arc1`, oneCCL `2022.0.0`),
  grounded in `external_patches/wheels.pins.md5`, the Dockerfile stage
  docs, and MIGRATION-arc. Title/tagline/fork-line block untouched.
- **Release bullet cluster cleaned and simplified**:
  the four dense sub-bullets (engine content kernel delta, battery
  detail, re-discrimination record, evidence) collapsed to four scannable
  bullets — the release content detail lives in this CHANGELOG's cert28
  entry + MIGRATION-arc + PERFORMANCE-arc (verified present before
  removal; nothing lost, README links to all three). The project-status
  and improvement-lane bullets tightened to their claims + links.
- **Deep staleness sweep, second pass** — one real defect the k5-literal
  grep missed: the CONFIGURATION-arc identity row still recommended the
  weights-quant decode posture with **`ARC_SPEC=5`** spelled out (123.9
  tok/s, +52 % vs k3) — rewritten to the certified k3 recommendation
  (`ARC_SPEC=3`, **103.3 tok/s @ 8K**, best online-quant lane). Grep
  classes run: depth-5 spellings (`SPEC=5`/`depth 5`/`k=5`) now zero
  outside decision records; superseded numbers (81.6/115.2/123.9/128.3)
  survive only as probe-tax/decision-record context; TODO/TBD markers
  zero; currency claims all cert28.
- **Known-pending class, documented not hand-patched**: `docs/reports/*`
  + `METHODOLOGY.md` carry cert23 engine stamps — they are generator
  output (the r139-era regen) and the held generator already stamps
  cert28; the r240 close-out regen on the cert28 banks is the lawful
  fix (hand-editing would falsify config-parity: the banks' engine was
  cert23). `docs/README.md` is upstream's tracked index — untouched by
  fork law.
- Gates at file-write time: validate_docs 0 errors / 0 advisories
  (27 fork docs), leak_sweep clean, check_currency clean at cert28.

## r240-docs4 — deep-sweep pass 4: era-comparison removal from current-facing docs (2026-11-25)

- Current-facing docs present only the latest
  certified image's information; no comparisons against older-era
  images or values. Pass 4 removed/rewrote every era-comparison
  framing while keeping the measurements and evidence pointers:
  - PERFORMANCE-arc: the MX-decode paragraph drops the "vs cert27 /
    cert25zb-era / cert11c-era anchor" deltas (the record stands on
    its own; cross-era deltas were recorded bookkeeping, not lever
    claims); the AutoRound and gemma-L1 blocks drop parenthetical
    era columns ("era 26.67…", "cert19-era record (r111)" → "prior
    record"); the close-out matrix now names its banks generically
    (cert23-era close-out bank → the p637 bank) and points at the
    cert28 regeneration; the warm-face note drops "Earlier-era".
  - STATUS-arc: "era anchors / era aggregate" → "prior anchors /
    prior aggregate"; the per-config-reports bullet now states the
    cert28 regeneration directly (was "cert23-era battery — being
    re-run").
  - CONFIGURATION-arc: the quark image-floor row and the manifest row
    drop cert-numeric lineage framing ("cert11+ image", "base
    fork_sha from cert9 + cert10 rows" → "current certified image" /
    "current stamp: cert28_fork_sha"); the dropped-depth row states
    the sampling-collapse decision without era framing; the
    corruption-record note reworded.
  - envs.md: MX-knob provenance clauses rewritten ("since cert27 /
    cert26 refusal" → "the earlier REP-LOOP refusal root-caused and
    fixed"; "cert25zb-era kill switches" → "kill switches";
    "cert25zc closing rung" → "validated optimum").
  - MIGRATION-arc: the boot-time note drops the cert26/cert23/cert22
    chain ("the cert23-lineage ×2 graph captures add ~29 s over the
    cert22 base's 285–295 s" → "the ×2 graph captures add ~29 s").
  - README: champion-faces preamble ("the cert23 engine is
    E2E-neutral" → "re-discriminated on the cert28 engine") and the
    health-wait comment (cert23 double-buffer → the graph-exec
    double-buffer).
  - Legitimate survivors (audit-verified, not era comparisons):
    PATCHES-arc per-patch "Cert: certNN" stamps (component
    provenance — which release shipped a module), bank names that
    contain era tokens (the p627/p637 evidence paths), the gemma
    VM-era config-set name, and `docs/history/` + CHANGELOG
    (the designated history homes).
- Gates at file-write time: validate_docs 0 errors / 0 advisories
  (27 fork docs), leak_sweep clean, check_currency clean at cert28.

## r240-docs3 — deep-sweep pass 3: config-parity completion + evidence counts + Contributing (2026-11-25)

- **Config-parity completion (the deep pass's real catch)**: all three
  certified serve paste-blocks (README Quick start, CONFIGURATION Step
  2, CONFIGURATION's runtime/system reference) were missing
  **`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300`** — the serve script's
  `ARC_RPC_TIMEOUT` passthrough (script line 388, p628; r108y evidence:
  a face died at 300.5 s) — while their "23 of 25 / (13 runtime/system)"
  count claims were composition-wrong (they showed the graph-posture
  var, which rides `$GRAPH_ENV` gated by `ARC_GRAPH=1`, and omitted the
  watchdog). All three blocks now byte-match the script's resolved
  default-lane env: **24 serve-time exports** (13 runtime/system + the
  graph-posture var + the LMCache 10) + 2 boot-time; count claims
  rewritten to the verified composition. The watchdog line's comment
  states it equals the vLLM stock default and only the gemma-4 lanes
  override (1200 s) — resolved on record. Rig script
  parity re-verified: local vs rig md5 `72b1b4b0…` byte-identical.
- **Evidence counts refreshed to the verified tree** (counted, not
  carried): **233 evidence dirs / 203 VERDICTs** — README Repository
  model (232/202) and TESTING-arc Evidence layout (128/102, three eras
  stale) both corrected. Counts re-verified at write time against
  `find results/ -maxdepth 1 -type d` / `-name VERDICT.md`.
- **Contributing section added** (README best-practice completion):
  branch/commit/PR flow pointer to the Repository model, AGENTS.md as
  the conventions home, and the evidence + push-gate requirement; TOC
  entry added.
- Gates at file-write time: validate_docs 0 errors / 0 advisories
  (27 fork docs), leak_sweep clean, check_currency clean at cert28.

## r240-prep — docs: professional sweep — identity-row currency fix + gate anchor (2026-11-24)

- Editorial sweep of the fork-owned docs ahead of the r240 report
  battery close-out (the r240 campaign's local half). **One hard
  currency defect found and fixed**: the PERFORMANCE-arc
  "Rig, image, and posture identity" table's **Engine image** row
  still presented cert25zc — it had drifted across three certs
  (cert25zc→cert28) while both of the file's check_currency anchors
  stayed green (the same anywhere-in-file class the P6.38 row-anchor
  fix described for MIGRATION S1). The row now carries the cert28
  identity + re-discrimination provenance, and
  `tools/check_currency.sh` anchors the row itself
  (`| Engine image | **current**: …cert28`) — the drift class is
  closed at the gate.
- README professional restructure: the certified-release bullet
  (1,000+ chars, single run-on) became four scannable sub-bullets
  (engine content / certified native posture / the re-discriminated
  record / evidence); every fact and citation retained, the full
  `.so` md5 pin inventory consolidated to the CHANGELOG cert28 entry
  (byte-exact in CHANGELOG + MIGRATION + CONFIGURATION — no
  information loss). Evidence counts refreshed to **232 dirs / 202
  VERDICTs** (r239 bank).
- Gates at file-write time: validate_docs 0 errors / 0 advisories
  (27 fork docs), leak_sweep clean, check_currency clean at cert28.

## r239 — P6.47: record-lane re-discrimination on cert28 — every face beats its era anchor; docs hard-currency sweep (2026-11-24)

- The deep-sweep campaign's first leg (2026-11-24:
  re-run all analyses, validations and performance tests on the
  latest certified image): the era-stale record faces re-measured on
  cert28. VERDICT:
  `results/2026-11-24-p647-r239-record-lanes/` (r239 + r239b merged
  bank, 38 files).
- **Checkpoint-lane TTFTs 1.5–3.5× faster than their cert11c-era
  anchors** (the era banks pre-date the prefill-lane closure): qurk
  17.06 / 1.10 / 4.05 / 38.31 s vs 32.28 / 3.84 / 7.73 / 70.61;
  aur 16.95 / 1.19 / 3.73 / 38.49 vs 26.67 / 1.52 / 6.36 / 59.52
  (the era every-face TTFT ordering vs the mxfp4 lanes inverted);
  rhmxfp4 17.51 / 1.09 / 4.04 / 39.01 vs 29.01 / 1.68 / 7.37 /
  62.46. Acceptance 1.0 / 0.9804 / 1.0; erraudit 0/0; G1 3/3.
- **Gemma L1 record parity-or-better** at `ARC_MAXLEN=118784`:
  face8k 4.2, face32k 16.8, face120k 108.4, face32k_2 0.9 s, solo
  29.0 tok/s ×3 (era 4.9 / 17.2 / 110.9 / 0.9 / 28.4–28.5). The
  r239b agg8 gate FAIL was re-derived as a units artifact — but the
  "same-units 361–474 aggregate / 1.85–2.4× the era record" that
  this derivation produced is itself RETRACTED (correction
  2026-11-25, see the r240-docs6 entry: requested-budget-numerator
  artifact over early-stopped streams; honest cert28 agg8 = 24.6–26.4
  tok/s per stream, 196–211 tok/s aggregate at full 8-way — parity
  with the same-lane 195.9–196.6 8×256-protocol aggregate).
- Two boot-fail law-lessons banked: gemma maxlen 131072 no longer
  boots on cert28 (KV 5.83 > 5.64 GiB, est. max 120704 → 118784,
  face class preserved); a boot-fail branch that returns without
  `docker rm -f` cascades the container-name conflict into the next
  cell (r239b adds the rm to every branch).
- **Docs hard-currency sweep in the same PR**: STATUS-arc +
  PATCHES-arc rewritten to cert28 (the MX KV lane family, image
  manifest + .so pins — the deep-sweep audit found both docs five
  certs stale); two new `tools/check_currency.sh` anchors close the
  drift class. In flight: the r240 report-battery re-run
  (r240a → r240b → r240c → `docs/reports/` regeneration).

## r238 — P6.47: cert28 per-lane n8 acceptance mined; K-precision law, channels all closed (2026-11-24)

- No cert cut, no rig time (analysis-only, r236 precedent): the
  acceptance channel from the r237 idle disposition closed by
  mining the banked `metrics_before.txt` snapshots (16K-cell deltas,
  ~300 drafts per lane). VERDICT:
  `results/2026-11-24-p647-r238-n8-accept-mining/`.
- **First per-lane n8/16K acceptance rows on cert28** — all lanes
  healthy (pos0 ≥ 0.81, no knee; S1 stays cured under this traffic
  shape too): ctrl 2.306 acc/step, m4q 2.170, mkq 2.399, m8q 2.430.
- **Acceptance K-precision law (the factorial)**: e2m1 **K** costs
  ~0.23 acc/step (−10%, ~5 SE, concentrated at pos2); e2m1 **V** is
  acceptance-free (~0.03, sub-noise). Mechanism: the MTP draft is
  lane-invariant — only the target verify rides the KV lane, and
  deepest-position verification is where e2m1-K numerics diverge.
  Throughput consistency: acceptance does not dominate tps (m8q best
  acceptance, worst n1 ITL); the m4q aggregate position carries a
  ~6% acceptance tax on top of its ITL gap at n8 — the mkq split
  (e4m3 K + e2m1 V) avoids it while keeping half the KV savings.
- All three idle-disposition evidence channels now closed
  (engine-shape / workload-shape / counters); the MX hunt's idle
  stands.

## r237 — P6.47: cert28 n8 steady-state long-ctx faces; every face class re-discriminated (2026-11-24)

- The last un-re-measured face class on cert28: n8 steady-state
  long-context (16K/32K, MT=128, the r185-ladder cell shape), one
  boot per lane, 4 boots ~41 min, zero errors. VERDICT:
  `results/2026-11-24-p647-r237-n8ss-cert28/`.
- **Anchor comparison (shape-valid: era anchors recompute exactly
  from the r185 banked JSONs with the identical formula)**: every
  lane below its cert25zb-era anchor — ctrl 1671→1273.9 (−23.8%),
  m4q 1703→1440.4 (−15.4%), m8q 1743→1506.1 (−13.6%), mkq
  1823.7→1399.7 (−23.3%). Cross-era deltas recorded, not claimed as
  lever effects: cert26–28 changes are MX-kernel-scoped and the ctrl
  move has no release-notes attribution (ramp-weighted metric family,
  stack drift unbounded within-boot).
- **Within-leg ratios (the clean statements)**: ss @32K m4q 1.131×,
  m8q 1.182×, mkq 1.099× ctrl; @16K (first cert28 rows) MX lanes
  0.755–0.938× ctrl. Metric honesty banked: at MT=128 `itl_ms_p50`
  degenerates to the spec-burst cadence (identical across slots) and
  `ss` is ramp-weighted (TTFTs stagger 10→115 s, totals collapse) —
  recorded faces, not decode claims.
- **Disposition**: with this leg every face class is
  re-discriminated on cert28 (n1 shallow r234/r235, agg8 r236, n8-ss
  r237); no lever priced above noise — the MX hunt idles pending NEW
  evidence (plan §12).

## r236 — P6.47: cert28 agg8 re-discrimination by bank mining; clamp-neutral as designed (2026-11-24)

- No cert cut, no rig time (analysis-only leg, r231c precedent):
  the agg8 rows already existed in the r233 battery bank
  (`r113_battery.py decode` ran in every leg) — mined and compared
  to the cert27 values re-verified in r230. VERDICT:
  `results/2026-11-24-p647-r236-agg8-mining/`.
- **agg8 (conc8 ctok 2048, 3-rep mean)**: ctrl 307.8 (+1.4% vs
  cert27), m4q 258.4 (−1.3%), m8q 252.0 (+0.9%), mkq 249.2 (−0.6%)
  — every lane inside the cross-boot noise band; ratios m4q/m8q/
  mkq = 0.84/0.82/0.81 × ctrl. solo (ctok 256): ctrl 60.6, m4q
  51.2, m8q 47.7, mkq 46.7.
- **Mechanism**: the r231d clamp lives on the legacy scalar split
  path; conc8 decode rides the G-adaptive/v5 decoder lane — the
  −7.6% n1 @2K win is lane-scoped by construction and does not
  transfer to aggregate faces (and none was claimed). The
  aggregate record stands; no aggregate-face lever remains priced
  above noise on the clamp wheel.
- Docs: README record row + PERFORMANCE-arc agg8 line now cite the
  cert28-mined values.

## r235 — P6.47: the fp8-ctrl shallow record rows; exact same-context MX gaps (2026-11-24)

- No cert cut (measurement only — the record's control side
  completed; r230 precedent, zero engine bytes).
- The r230/r234 ctrl cells ran long faces only, so every MX-vs-fp8
  shallow comparison was cross-context. r235
  (`tools/ar/r235_ctrl_shallow.sh`, r234-verbatim harness, refuses =
  image + manifest + `.so` md5s; ONE ctrl boot on cert28): **ctrl
  40.1/40.0/40.4 ms at 2/4/8K** — the fp8 ctrl is FLAT through the
  shallow range (tps 92.4 @2K; c8192 same-boot sanity 1.000×;
  chunks 52–53, 0 errors). `R235_GATE: PASS`.
- **Exact same-context shallow gaps** (r234 MX rows vs r235 ctrl):
  mxfp4 **+8.2% @2K** (43.4 vs 40.1) and +13.0% @4K; mxfp8 +10.7%
  @2K; mxk8v4 +10.0% @2K — the residual shallow tax after the
  seqused-tightened clamp (the cross-context "within 8% of fp8 8K"
  phrasing retired). Docs updated to the same-context ratios
  (README cert paragraph + record row, PERFORMANCE-arc record
  block, envs.md `ARC_MXKV_NATIVE` row).
- Bank: `results/2026-11-24-p647-r235-ctrl-shallow/` (10-file bank;
  law-8 rig-side redaction before pull, strict post-pass 0; W14
  restored by the leg trap, HEALTH 200 05:43:54).

## rel/v0.30.0-b70-cert28 — P6.47: seqused-tightened split clamp certified; the MX walk line closed (2026-11-24)

- cert28 = cert27 + **the r231d clamp wheel** (arc7mx build28,
  `vllm_xpu_kernels-0.1.14.1+arc7mx` md5 `e6d8d337e43b907504537061414128f8`
  BAKED; three installed-`.so` md5 pins asserted in-image —
  `libattn_kernels_xe_2` `61d2f5a365f05084b3dd72c4ae1da3e0`,
  `_vllm_fa2_C.abi3.so` `6e6c48508d40cb4c5007f27f79df97cf`,
  `libmqa_logits` `190e261edc79b3b497d8d8e801e65757`; the latter two
  identical to cert27's — the patch is header-only in the split-lane
  TUs). Manifest `cert28_fork_sha=43f84cf977f9444900ff21387715ca5865db4c1a`
  = PR #123 (the r231d evidence merge the image was built and gated
  from; the cert28 close-out commits add zero engine bytes — docs +
  banks only). Build: `tools/ar/r232_build_cert28.sh` (refuses on
  base-identity drift; single-RUN install + in-image python asserts +
  manifest append; overlay inherited byte-identical from cert27).
- **The change** (provenance `tools/ar/r231d_apply.py`):
  `seq_num_kv_splits = min(num_kv_splits, max(1, windowed_k_blocks/4))`
  on the legacy scalar path, mirrored in `mx_paged_decode_kernel.hpp` +
  `mx_reduce_split.hpp` — deep splits at short contexts serialize
  chain tiles the attention window cannot fill. Line history: r231a
  decomposed the MX ITL gap (flat-cost signature — a format-independent
  constant = heuristic/reduce overhead, not walk cost); r231b q-prestage
  refuted (correct, no perf); r231c fixed-term decomposition closed the
  e2e per-call model and priced the split term; r231d offline gates
  green with 2K decode −43/−39/−51% record-neutral; r231de e2e face
  leg **−7.6% ITL at 2K with record parity exact** (long faces).
- **r233 §10 battery PASS** (`tools/ar/r233_cert28_gate.sh`, derived
  from r228 by the refusing transform `tools/ar/r233_transform.py`;
  stage 2 of the gated chain `tools/ar/r232_cert28_launch.sh`,
  `R232C_LAUNCH_START 23:39:39 → R232C_LAUNCH_DONE 05:10:16`): 8 legs,
  one boot each — m8q/m4q/mkq LONGRETURN + T1_SEED **absolute PASS at
  native_lines=108**, SUITE13 15/15, ctrl default-off (mx_backend=0)
  + LMC-on 4 lines intact, gemma legs bank-only (REP-LOOP = the
  confirmed native-KV baseline), **GEMMA_COMPARE PASS** (no new modes
  vs g0, topwin 0). `P647_CERT28_GATE_COMPLETE -- PASS 04:32:12`.
- **r234 record re-discrimination PASS**
  (`tools/ar/r234_cert28_itl.sh`, one boot per cell, the certified
  posture): every anchored face within **+0.1%** (worst ratio 1.001;
  ctrl 40.4/41.1/42.5 = 0.999–1.000×). **Native mxfp4 five-face record
  43.4/45.2/48.5/52.2/58.1 ms at 2/4/8/16/32K** (2K was 47.0 on
  cert27 — the clamp win now certified; tps 88.6→65.2). **First
  shallow rows for mxfp8 (44.4/46.5) and mxk8v4 (44.1/46.0)** — the
  clamp's shallow benefit is lane-uniform; MX 2K ITL within 8% of the
  fp8 ctrl's 8K ITL. Long-ctx parity exact (the no-op tail at
  capacity-priced walks, as designed). Native telemetry splits=auto,
  true count 108/cell (`grep -c`; the in-log `lines=156` is the
  tqdm universal-newlines artifact — r230 note). All 18 faces
  CONC_OK, chunks 51–57, 0 errors. `R234_GATE: PASS`.
- **Disposition: the MX walk line closes at cert28** — all three MX
  KV dtypes native, correctness-gated and at their measured record
  shallow-through-long context. The ladder's capture-point ledger is
  fully dispositioned: the S1 acceptance knee cured (r185/r186,
  KNEE=none; cert28 battery metrics re-confirm — 1.78–1.81
  accepted/draft on the MX lanes, pos0 ≈ 0.80), aux/plan-build at
  noise (T1), continuation-dequant refuted (p182). The always-loop
  directive continues on new evidence.
- Evidence: `results/2026-11-24-p647-cert28-battery/` (193-file bank
  + chain launch.log) + `results/2026-11-24-p647-r234-cert28-itl/`
  (53-file bank) + the r231a–r231de line banks. Law-8: 376 address
  occurrences redacted rig-side before pull (perl, byte-exact);
  `SHA256SUMS` regenerated post-pass.

## r230 — P6.47: cert27 perf re-discrimination; the record re-measured on the shipping image (2026-11-23)

- No cert cut (engine bytes unchanged — a measurement + docs campaign;
  the closing phase L1–L2 of the plan's closing-phases directive).
- The long-ctx ITL faces re-run on `vllm-arc:v0.30.0-b70-cert27`
  (r219b harness posture: ARC_GRAPH=1 spec k3 fp8 weights, one boot
  per cell, n1 MT=192; `tools/ar/r230_cert27_itl.sh`, refuses =
  image + manifest `cert27_fork_sha` + installed-`.so` md5):
  **ctrl 40.4/41.1/42.6 ms @8/16/32K = 0.999–1.001× its banked
  anchors** (fp8 bank untouched by the r225 fixes — the strongest
  control); **native mxfp4 49.4/52.1/58.1 (+1.8–2.2% vs the cert26
  measurement = the fix3 fp32 split-partials cost**, within the +5%
  one-sided band); **first native long-ctx rows for mxfp8
  (50.9/55.3/64.4) and mxk8v4 (50.0/53.9/61.9)**. Quality: chunks
  ≤ 57 (80 law floor), 0 errors/tracebacks, native telemetry 108
  lines/cell, splits=auto. R230_GATE: PASS.
- agg8 half (banked prior in the r228 battery, re-verified from the
  banks): ctrl 303.5 (cert26 303.2 — parity), m8q 249.7 (249.9 —
  parity), m4q 261.9 (254.6, +2.9%), mkq native 250.6 vs its cert26
  Triton lane 232.2 (+7.9%) — native is the fastest mxk8v4 posture at
  both short and long context.
- Telemetry-count note: the in-log report's `lines=156` is a python
  universal-newlines artifact (tqdm `\r` fragments split into extra
  "lines"); `grep -c` on the same bytes = 108.
- Docs brought to cert27-measured values (closing phase L3):
  `PERFORMANCE-arc.md` record block, `envs.md` `ARC_MXKV_NATIVE` row
  (was still cert26 posture: mkq "REFUSED / Triton only"),
  `CONFIGURATION-arc.md` image row, `release_notes.md` cert27 section,
  README headline + evidence-dir counts (drifted 130/104 → verified
  220/189).
- Bank: `results/2026-11-23-p647-r230-itl-redisc/` (43 artifacts;
  rig-side dual-class perl redaction BEFORE pull, post-pull
  rfc1918 scan zero).

## rel/v0.30.0-b70-cert27 — P6.47: r225 wheel fixes certified; all three MX lanes native, mxk8v4 un-refused (2026-11-23)

- cert27 = cert26 + **the r225-fixed `arc7mx` wheel** (build25,
  `vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
  `aeca45daa740e9973a1c51bfc5ea2bc4`, byte-pinned; installed
  `libmqa_logits_kernels_xe_2.so` md5 `190e261edc79b3b497d8d8e801e65757`,
  asserted in-image) BAKED over the cert26 wheel, plus the fork overlay
  with the `ARC_MXKV_AB` diagnostic instrument (default-off,
  self-guarding under capture). Build: `tools/ar/r227_build_cert27.sh`
  (inline Dockerfile on the cert26 base; manifest
  `cert27_fork_sha=d39ab5a26bf737d006d47a3f3125a404dca6e928`).
- **Kernel fixes over the cert26 wheel** (fork-owned tree, provenance
  `tools/ar/r225_fix*_apply.py`): fix1 e2m1 subnormal mantissa (implicit
  bit + OCP `m*0.5` decode), fix2 `sycl::exp2` in the split-KV combine
  rescale (evicting `native::exp2`'s ~1e-3 relative error), fix3 fp32
  split partials (`ElementOAccum` float — the reduce combine is the
  single bf16 quantization point), fix5 the S==1 sentinel-LSE predicate
  aligned to the O-normalize arm.
- **Certified posture change: `mxk8v4` native UN-REFUSED.** The cert26
  mkq refusal (REP-LOOP all classes) root-caused to the four defects
  above (r224 in-vivo A/B + offline factor bisection; r225 offline
  12/12; r226a e2e re-entry 8/8 fresh COVERED payloads, engine raw lane
  gap p50 3.05e-1 → 5.5e-2, nan 0/120). r228 battery on cert27:
  m8q/m4q/mkq LONGRETURN + T1_SEED PASS, longprobes CLEAN on full +
  3 rehits, native_lines=108 each, tracebacks 0; gemma legs = the
  documented native-KV baseline verbatim (GEMMA_COMPARE PASS, no new
  modes / no amplification); g0 deterministic on rehits.
  `P647_CERT27_GATE_COMPLETE -- PASS`.
- **r229 format-replay value legs**: mxfp8 8/8 + mxfp4 8/8 at the
  full-population-calibrated q-cast-matched criterion; the r229s/r229s2
  sweeps (86 COVERED cells per format) measured **format-invariant
  residual distributions** (native-vs-walk max 1.216e-1 vs 1.208e-1) —
  the r225 coverage-note claim (one ReduceSplitK path serves all three
  formats) is measured, not argued. Band law banked: bands calibrated
  on a lexical-prefix sample of a heavy-tailed distribution are not
  calibrated (the old 2.5e-2/5e-2/16 bands failed ~2/3 of the full
  population on BOTH formats; recalibrated 2.0e-1 / 4.0e-1 / 64.0,
  corruption signatures still gated tight).
- **Cross-image law**: native-lane outputs drift cert26 → cert27 (the
  fixes change native numerics by design); the certified gate is
  per-boot self-consistency (3-cycle single-hash + T1_SEED + CLEAN
  rehits), not cross-image hash stability. The perf record carries the
  banked cert26-measured numbers (numerics-only wheel delta; the
  `fp8_e4m3` bank is untouched).
- Wrapper/infra deltas: `serve_arc_b70.sh` gains the `ARC_MXKV_AB*`
  passthrough trio (r229a run1 abort fingerprint: native armed,
  instrument not) and the tracked copy resynced to the live rig script
  (it had drifted pre-r212f).
- Evidence: `results/2026-11-23-p647-cert27-battery/` (196 files,
  serve.logs redacted rig-side 176 → 0),
  `results/2026-11-23-p647-r229-format-replay/`, and the r224–r226
  leg banks cited therein.

## rel/v0.30.0-b70-cert26 — P6.47 Phase-8: arc7 MX-native decode line certified (2026-11-22)

- cert26 = cert25zc content + **the Phase-8 arc7 MX-native decode
  line**: the `arc7mx` wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
  `62f234c8d0aa89c4aec3e8d2c6fa22ad`, byte-pinned) BAKED into the image
  — MX-native paged-decode Xe2 kernel
  (`vllm_xpu_kernels.mx_attn_interface.mx_paged_decode`), the r219 C++
  auto-split heuristic at the validated optimum (chain 341 bw → 130
  reduce → 128 policy; `ARC_MXKV_NSPLIT` RETIRED as a posture
  requirement, kept as probe instrument), ELSE-branch policy cap
  64→128 + block16 alignment, the r220a3 corrected WG-canonical
  coordinate frames, and the value-correct env-gated `P64NATIVE`
  instrument (default OFF) — plus the fork `mx_kv` overlay
  `native_verify` bridge (`ARC_MXKV_NATIVE` gate default-off;
  `_NATIVE_MINLEN` per-lane thresholds mxfp8 16384 / mxfp4
  12288:8192 / mxk8v4 8192; r212f capacity-priced walk; capture-time
  bucket telemetry).
- Built by r222 (`tools/ar/r222_build_cert26.sh` — staged-tree
  content refuses + wheel md5 refuse + evidence refuses; inline
  Dockerfile on the cert25zc base) with build-time asserts;
  `IMAGE_MANIFEST` `cert26_fork_sha=cac6393159bdf1f6356e744159163fd679726221`.
- **Certified native posture: `ARC_KVDTYPE=mxfp8|mxfp4` +
  `ARC_MXKV_NATIVE=1`** (NSPLIT + P64NATIVE unset). Split-depth surface
  8..256 fully measured (r216–r219), optimum 128 A/A-confirmed 4×,
  then r219b proved C++ auto lands it by itself (within 0.08%).
  Long-ctx faces (r219b): mxfp4 ITL 48.5/51.2/56.8 ms at 8/16/32K
  (−19.2% vs auto-40, flattening the long-ctx droop; tps 65.4 @32K;
  1.34× fp8 walk-parallelism). r223 battery smoke face: agg8 medians
  m8q 254.0 (0.84× ctrl, was 0.78×), m4q 256.1 (0.84×, was 0.75×).
- **Native mxk8v4 REFUSED at the battery's absolute long-return gate**
  (first mkq cell: REP-LOOP garble all classes, deterministic; T1_SEED
  PASS) while the uniform-dtype native lanes are clean — attributed to
  the mixed-fmt lane (e4m3 K + e2m1 V; per-tensor K/V layout
  divergence the r220c offline gate's contiguous synthetic pool
  structurally could not see). **mkq serves the cert25zc Triton
  lanes**; the failed cell banked verbatim as
  `results/2026-11-22-p647-cert26-battery/mkq_native_refused/`;
  **r224** = the reopen path (engine-stride-faithful value gate +
  native-vs-triton mkq A/B).
- **The 16→64 KV-tile lever CLOSED by the e2e arbiter** (r220):
  offline −46..−50% @1K inverted e2e +0.86/+5.63/+13.40% ITL monotone
  in ctx, not rescued at any split depth (r220d); r221a/r221b/r221d
  eliminated shape/row-count/cache-state/knob families as
  offline-transferable mechanisms (residual: TP2 dual-GPU lockstep /
  engine step-graph structure). Posture stays `_16` + auto-128.
  r221d decomposed the +14–20% @1K engine-shape cost: BTWIDTH is the
  sole carrier (flat +99 µs = auto-split over-pricing reduce overhead
  at short ctx; pool/pad no-ops) — priced as the r222
  skip-empty-splits candidate (~1.6 ms/step at 1K-class ctx).
- **r223 full §10 battery: CERTIFIED**
  (`results/2026-11-22-p647-cert26-battery/` — 218 files +
  SHA256SUMS-verified + law-8 redaction pass [366 serve-log hits,
  perl byte-exact, `REDACTION-2026-11-22-p647-cert26-battery.md`]).
  Qwen absolute gates 4/4 clean (ctrl/m8q/m4q/mkq — LONGRETURN +
  T1_SEED + garbage + erraudit); gemma comparative PASS (GEMMA_COMPARE:
  no new modes / no amplification). Verdict of record under the
  amended gate (mkq triton cell rerun post-refusal;
  `P647_CERT26_GATE_COMPLETE -- PASS`).

## rel/v0.30.0-b70-cert25zc — P6.47: Phase-7 MX decode closing rung (2026-11-19)

- cert25zc = cert25zb content + **the Phase-7 closing rung of the MX
  decode ladder**: TILE_FLOOR launcher default 4→2 (`ARC_MXKV_V5_TFLOOR`;
  kernel constexpr stays 4 — certified-signature freeze), G-adaptive mxfp4
  verify-decoder dispatch (`ARC_MXKV_V5_G_MIN=8` — small-G legs keep the
  split path, big-G flip to the verify decoder), v7 research kernel DORMANT
  (`ARC_MXKV_V7=0`). MX lanes now serve **LMCache-free** (Phase-6 auto-off
  posture; r176 A/B neutral-to-positive), ctrl keeps the certified LMC-on.
- Built by r196 (`tools/ar/r196_build_cert25zc.sh` — inline Dockerfile on
  the cert25zb base) with build-time asserts; `IMAGE_MANIFEST`
  `cert25zc_fork_sha=dd74715079dc6c23fd4ad2feb1a700294515d423`.
- **Closing-rung gains** (r195 tfe2 / r188 B1, held e2e at r196b/r197):
  m8q n1 32K ITL 106→95 ms (−10.4%); m4q n8 32K steady-state agg −14%;
  headline mxfp4 n8 32K agg decode p50 **1547→583 ms (−62%)**
  (`results/2026-11-19-p647-zc-fastcycle/`).
- **r197 full §10 battery: CERTIFIED**
  (`results/2026-11-19-p647-cert25zc-battery/`, 192 files banked +
  SHA256SUMS-verified + law-8 redaction pass). Qwen absolute gates 4/4
  (ctrl/m8q/m4q/mkq — LONGRETURN + T1_SEED + garbage + erraudit clean);
  gemma comparative PASS (REP-LOOP native baseline class, no new modes).
  Verdict of record re-derived from banked markers under the amended gate
  7fe72efc (m8q asserts the Phase-6 LMC auto-off posture) per the
  recompute-from-banked-artifacts law — `verdict_recompute.sh` banked.
- Decode smoke (256-tok solo / 2048-tok agg8, tps medians): ctrl 59.9/304.1;
  m8q 54.2/236.7 (0.78×); m4q 57.2/229.0 (0.75×); mkq 55.9/226.6 (0.74×).
  The remaining ~0.74–0.78× agg8 gap is the Triton split-dot structural
  deficit — the **arc7 MX-native kernel (Phase 8)** is the open line.
- Evidence: `results/2026-11-19-p647-v7a-attempt1/`,
  `results/2026-11-19-p647-v7a-attempt2-tfloor/`,
  `results/2026-11-19-p647-v7a-attempt3-bnsweep/`,
  `results/2026-11-19-p647-tf2-e2e/`, `results/2026-11-18-p647-b1-v5flip/`,
  `results/2026-11-14-p647-s1-acceptknee/`,
  `results/2026-11-18-p647-t1-attribution/`,
  `results/2026-11-19-p647-zc-fastcycle/`,
  `results/2026-11-19-p647-cert25zc-battery/`.

## rel/v0.30.0-b70-cert25zb — P6.47: MX KV-cache dtype lane, K2/K3 packed dequant (2026-11-13)

- `--kv-cache-dtype {mxfp4,mxfp8,mxk8v4}` on qwen3.8-27b-fp8 (dual B70, TP2):
  Triton MX store/decode primitives in `vllm_gaudi/xpu/patches/mx_kv/` +
  engine dispatch. Interim certified chain on this lane (never tagged, per
  the close-out release law): cert24 (store separate-strides fix — spec/eager
  text-corruption root cause), cert25z (v5 split-dot decoder default), cert25za
  (v6 GQA-batched split decoder; m8 32K/n1 ITL 173.3 → 95.8 ms).
- cert25zb content: **K2 packed-fp16 dequant + K3 e2m1 fast-unpack default-on**
  (`ARC_MXKV_K2=0` / `ARC_MXKV_K3=0` revert). K2: e4m3→fp16 exact + e8m0 pow2
  scale in fp16 — single rounding point unchanged, bit-identical d=0.0; m8
  kernel 5.94 → 4.87 ms @32K. K3: nibble→e4m3 byte via ~10 int ops riding K2's
  convert; m4 8.64 → 7.62, mkv4 8.62 → 7.62.
- **Kernel ladder CLOSED on all three formats** (r182/r183): m8 walk = 66 %
  dot+load floor + irreducible 1.62 ms f8→f16 hardware convert (Xe2 has no
  native fp8 dot; persistent-cache variant rejected — would surrender the
  capacity product); LUT rung and join/shift rung measured dead/wash.
- **r184 full §10 battery: CERTIFIED** (`results/2026-11-13-p647-cert25zb-
  certification/`). Qwen absolute gates 4/4 (ctrl/m8q/m4q/mkq). Headline:
  mxfp4 holds ctrl solo decode parity (60.5–62.9 vs 60.1–61.8 tps) at
  **1.81× KV capacity** (721,994 vs 398,363 tokens); mxk8v4 1.27×; mxfp8
  capacity parity. Gemma comparative PASS (REP-LOOP native baseline class,
  no new modes). g0 native bank byte-identical to cert25z — the bake touches
  nothing on the native path. The battery's single counted gate item (m4g
  erraudit ERROR=4) re-adjudicated to 0 real errors: external-client 404s
  against the gemma-served leg (banked evidence + redaction note).
- Evidence: `results/2026-11-11-p647-mxkv-research/`, `results/2026-11-11-
  p647-r147-mxkv-lega/`, `results/2026-11-12-p647-cert24-gate/`,
  `results/2026-11-12-p647-decode-perf-research/`, `results/2026-11-12-
  p647-mxkv-vstride-fix/`, `results/2026-11-13-p647-cert25zb-certification/`.

## rel/v0.30.0-b70-cert23 — P6.34/P6.35: graph-exec double-buffer (2026-11-10)

- cert22 engine + one patch module, `vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py`:
  every XPU graph captured ×2 into the manager's shared pool, replays
  alternate execs — removes the torch-2.13 same-exec `XPUGraph.replay()`
  host block (p135i microbench: 13.96 ms same-graph vs 8–12 µs
  alternating). Replay max 24 → 0.37 ms; engine serial work 40.8 →
  12.7 ms/step; **E2E-neutral at every measured posture** (fp8-k3 94.5 vs
  94.3 tok/s on/off; mx4k5 121.0 vs 123.9) — a costless driver-level
  hygiene fix: +29 s boot, zero extra memory. Knob
  `VLLM_ARC_GRAPH_EXEC_DB` (default 1; `=0` kill-switch proven).
- The P6.35 decode gains are **posture** (serve-line choices on the same
  engine): spec k5 +41 % (115.2 tok/s); mxfp4-W4A16 + k5 = 123.9 tok/s
  (+52 %) with TTFT 3.98 s < the 4.55 s baseline — the recommended decode
  posture. mxfp8-W8A16 is a decode dud.
- P6.35-V full validation matrix (r136/r136b, 11 cells, both models, all
  online quants × spec depths, 0 tracebacks): qwen mx4-W4A16 k5 re-read
  **128.3 tok/s = +36 % vs the clean fp8-k3 base 94.5 (project best)**;
  gemma mx4-W4A16 **37.1 tok/s (+43 % vs mxfp8)**; gemma online fp8
  boots + serves (26.6, new capability); mxfp8 deprecated for decode;
  best TTFT fp8-k5 3.48 s. Rate accounting corrected (spec SSE streams
  multi-token chunks — numerator = max_tokens).
- Evidence: `results/2026-11-10-p635-telemetry/` (legs A–K + sweep +
  `cert23/` gate + `vmatrix/`).

## rel/v0.30.0-b70-cert22 — P6.33: skinny-M kernel family, banked OFF (2026-11-09)

- cert21 engine + the arc6 `vllm_xpu_kernels 0.1.14.1+arc6` wheel
  (three GEMV-regime M≤8 weight-only GEMM ops —
  `_xpu_C.skinny_fp8_gemm_w8a16` / `skinny_fp8_gemm_w8a16_mxfp8` /
  `skinny_fp4_gemm_w4a16`: TN weight streaming, register dequant, fp32
  accumulate, deterministic fixed-order split-K; plus W4A8 dispatch arms
  and the v0.1.17 Option-B backports #611/#645/#650/#600; sha256
  `484e06c1…`, ~352 MB — rig-only per the 100 MB artifact law) +
  `vllm_gaudi/xpu/patches/skinny_m.py` route behind `ARC_SKINNY_M`.
- Kernels correct (r126 GATE-1/2/3) and winning the eager kbench regime,
  but the r127 five-leg e2e battery **falsified the reroute**: under
  FULL_DECODE_ONLY XPU-graph replay stock oneDNN runs 420–680 GB/s
  device-side (r127f 50-call amortized probe; the motivating ~197 GB/s
  was an eager-launch host-rate artifact; the size-1 decode graph pads
  M→2), so gemma solo regressed −38 %/−42 % while every over-cap path sat
  on ref. r127c `ARC_SKINNY_M=off` reproduced the r124b ref exactly.
- **Route ships default OFF** (unset/off = stock routes = the certified
  default); serve behavior byte-identical to cert21 (r127g2: gemma
  27.7/192.9 exact, zero route markers). TC (joint-matrix) redesign is
  the open future lane behind the r127f gate.
- Content fork SHA `e2457389`; evidence `results/2026-11-08-p633-skinny-m/`.

## rel/v0.30.0-b70-cert21 — P6.31: online-MXFP8 W8A16 route (2026-11-08)

- cert20 engine + the arc5 `vllm_xpu_kernels 0.1.14.1+arc5` wheel
  (`_xpu_C.fp8_gemm_w8a16_mxfp8`: bf16 activations × f8 weights,
  group-32 e8m0 weight scales, zero per-call act quant; carries the
  oneDNN-refused `fp4_gemm_w4a8` probe op; sha256 `219268cf…`) +
  `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` reroute behind `ARC_MXFP8_MODE`
  (unset/`w8a16` = route, default; `w8a8` = stock control).
- Same-image A/B (r124/r124b): **qwen 1.37–1.55× over stock W8A8 on
  every face** (U8k 3.61 s / U32k 17.07 s, 53.8–55.6 solo / 265.9 agg8
  tok/s, spec pos1 0.9804); **gemma 1.57–1.85×** (3.87 / 20.94 s,
  27.7 / 192.4) at `ARC_MAXLEN=40960` (the 8-bit-weights capacity law:
  ~29 GiB TP2 weights leave 5.31 GiB KV).
- Adjudication (pre-registered): checkpoint-fp8 stays the perf default;
  mxfp4-w4a16 stays the superior online lane; W4A8 (e4m3×f4) banked
  closed-negative (oneDNN refuses the joint dtype at primitive-create).
- r125 three-lane gate green (fp8 discriminator + w8a16 + w4a16 carry).
  Content fork SHA `c8fdf32a`; evidence
  `results/2026-11-08-p631-mxfp8-w8a16/`.

## rel/v0.30.0-b70-cert20 — P6.30: online-MXFP4 W4A16 route (2026-11-08)

- cert19 engine + the arc4 `vllm_xpu_kernels 0.1.14.1+arc4` wheel
  (`_xpu_C.fp4_gemm_w4a16`: bf16 activations × f4 weights, group-32
  e8m0 weight scales → oneDNN `jit:gemm` DPAS via weight decomposition;
  sha256 `c675f1a3…`) + `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` reroute
  behind `ARC_MXFP4_MODE`.
- Root-caused the stock W4A4 penalty (r113→r120d): per-call activation
  quant ≈ GEMM cost at decode Ms, and the (f4,f4) joint dtype can never
  pass oneDNN gemmstone's JIT gate → `matmul:ref`, structurally. W4A16
  drops both (verbose-proven `jit:gemm`, numerics rel 1.66e-3).
- Same-image A/B (r120c, qwen): **U8k 3.66 s / U32k 17.12 s, 62.5–65.2
  solo / 301–311 agg8 tok/s — superior to online mxfp8 on every axis
  1.47–1.75×**, 1.74–1.87× TTFT over the W4A4 control, spec pos1 1.0000;
  gemma (r120g) 1.90–2.21× over W4A4. The v3.13 online-fp8 question
  closed mechanistically (group-scale handling inside `jit:gemm`); plain
  qwen fp8-online is an upstream interface gap (r120h traceback banked).
- Engineering laws banked: `_xpu_C` needs explicit submodule import
  (package import does not dlopen); ≤24 icpx jobs when a W14 window is
  live; pip validates wheel filenames; gemma legs need `ARC_LMC=0` +
  `ARC_RPC_TIMEOUT=1200` + dir `/models/gemma4-31b` (no dash).
- r121 gate green. Content fork SHA `a918bee9`; evidence
  `results/2026-11-08-p630-mxfp4-perf/`.

## rel/v0.30.0-b70-cert19 — P6.29: sliding-window FA2 route (2026-11-07)

- cert18 engine + `external_patches/vllm/patch_arc_xpu_fa2_slide_route.py`
  (`tools/cert19.Dockerfile`): sliding-window head-256 calls (gemma-4's
  50/60 local layers) dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel
  windowed in the same triton unified-attention wrapper. `ARC_ATTN_FA2=0`
  disables both routes (this and cert18's).
- Microbench (r109a/r109a3): triton 0.147–0.158 s/call, k-INDEPENDENT vs
  FA2 windowed 0.006–0.009 s = **18–25× prefill, 11–15× decode**. The
  r109a4–a6 "ref misses" were a per-chunk-softmax bug in the TEST refs
  (both kernels correct).
- Image validation (r111): gemma faces 4.9 / 17.2 (qwen-parity, 1.016×
  of the 16.93 record) / 110.9 / 0.9 s; decode 28.4 solo / 195 agg8
  tok/s (+45 % / 2.64× over cert18); both markers live per rank; qwen
  discriminator in-band (3.83/17.58 s). Spec parity for gemma-4
  dispositioned closed: no MTP head in the checkpoint; ngram
  structurally refused by the upstream v1 spec path on pure-attention
  hybrids (r110/r111 banked) — decode posture stays `ARC_SPEC=0`.
- Content fork SHA `904a3de8`; evidence `results/2026-11-07-p629/`.

## rel/v0.30.0-b70-cert18 — P6.28: FA2 hd512 route (2026-11-07)

- cert17 engine + `external_patches/vllm/patch_arc_xpu_fa2_route.py` v3
  (`tools/cert18.Dockerfile`): full-attention head-512 calls (gemma-4's
  10 global layers) dispatch to the ESVD FA2 kernel in the triton
  unified-attention wrapper. `ARC_ATTN_FA2=0` escape hatch.
- r108ac/r108ad: 0.3197 s vs 10.81 s per k32k fp8-KV call = **34×**
  (k98k ~30×, decode 1.5 ms/call); the fp16-KV control refuted the
  earlier L2-bandwidth-wall reading — the triton kernel is slow per
  K-tile iteration; the engine shipped the fast kernel all along.
- Image validation (r108ae): gemma faces 2.2–16.3× faster (face32k
  283.4→41.5 s, face120k 2950.3→181.5 s), decode +6 %, qwen
  discriminator marker-0 in-band (3.77/17.49 s) — incumbent lane
  untouched. The v2 boot crash (warm-gate full-storage descale form) was
  root-caused and fixed in v3.
- Content fork SHA `3136adfe`; evidence
  `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/`.

## rel/v0.30.0-b70-cert17 — P6.28: gemma-4 validation chain (2026-11-07)

- The cert11 engine + three build-kit fixes that made gemma-4-31B
  servable: cert12 `a964e047` (B70 prefill launch-config table in the
  triton unified-attention launcher + engine warmup gate), cert16
  `5d895fef` (warm-gate v4 — the triton cache-key int-softcap fix that
  closed the G2 long-prefill freeze layer 1), cert17 `e5e725b4` (lmcint
  kit hardening — KV-format detector BS floor 16 + the fail-loud
  `[ARC-P628-LMC-KVGATE]` two-head-size boot gate).
- G2 freeze closed in three layers (warm-gate cache key; ~31 s probe
  compile accepted; LMCache store wedge → KVGATE + `ARC_LMC=0` posture).
  Layer-2b: the face120k 300 s cliff was the stock execute-model RPC
  watchdog — `ARC_RPC_TIMEOUT=1200` joins the certified gemma posture.
- Both gemma lanes validated end-to-end (faces + decode + quality
  probes, erraudit 0/0); prefill root cause diagnosed at r108aa/r108ab
  (hd512 wall reading, later refined by r108ac — see cert18).
- Evidence: `results/2026-11-06-p628-gemma4/`.

## rel/v0.30.0-b70-cert11 — P6.26: MXFP4/quark lane (2026-11-06)

- cert10 engine + a two-override `ArcXpuPlatform` patch (`quark` in
  `supported_quantization` + `supports_mx() → True`, commit `7ea0f27b`) —
  the ONLY byte changes vs cert10 are `vllm_gaudi/xpu/platform.py` +pyc
  (r105b walk 294→294); the fp8 default posture is byte-identical,
  proven face-for-face same-day (r105t/r105t2, thermal hypothesis
  refuted with xpu-smi telemetry).
- Both pre-quantized MXFP4 checkpoints validated end-to-end under the
  full posture with the native `XPUMxFp4LinearKernel`; spec gates above
  the fp8 incumbent (0.9853/0.9951 vs 0.9258); decode penalty 1.21× /
  1.29× under the 2× trigger — mxfp4 = certified weight-constrained
  lane, fp8 stays the performance default.
- Tag `c987c2cd`; evidence `results/2026-11-06-p626-mxfp4/`.

## rel/v0.30.0-b70-cert10 — P6.24: oneCCL fp8-wire allreduce overlay (2026-11-05)

- cert9 engine (exhaustive `vllm_gaudi/` md5 walk: diff 0) + a two-file
  oneCCL 2022.0.0 same-version rebuild overlay with the e4m3-wire
  large-ring allreduce patch (u32-packed peer stores per the sector law,
  branchless converters, min-count threshold 2M elems). **Default OFF**;
  `CCL_SYCL_ALLREDUCE_FP8=1` opts in.
- r96 rebuild fidelity PASS (ABI/perf-identical); r97 gate **1.65×** at
  the dominant 76,677,120 B AR (7.801→4.731 ms); r98 task-level quality
  parity; r99 release gate — the opt-in beats cert9 on every p65 TTFT
  face (U8k −13 %, U64k −8.2 %, U32k −3.8 %, C32k −4.6 %); r100 closed
  every remaining AR door with wire-rate measurement (v4 = 82 % of the
  wire floor, final).
- Base fork SHA `aa2ee17d` (carried from cert9); evidence
  `results/2026-11-05-p624-fp8-ar/`.

## rel/v0.30.0-b70-cert9 — P6.20: prefill host-path (2026-11-04)

- cert8 engine + `patches/step_trace.py` diagnostics (silent-off),
  **L3 off-thread LMCache store (default ON**, `arc-lmc-store` worker —
  device ops stay on the engine stream) and L6 dispatch-late. TTFT
  −0.74 s U32k / −1.64 s U64k / −0.08 s U8k; spec/identity faces
  byte-exact cert8. The r83 window trace closed the lane: U64k/U32k are
  device-compute-bound (~89 % of TTFT device-side; host path clean).
- Fork SHA `aa2ee17d`; evidence `results/2026-11-04-p620-prefill-opt/`.

## rel/v0.30.0-b70-cert8 — P6.19: TurboQuant spec-verify FIX (2026-11-04)

- The P6.18 TQ refusal lifted: the corruption reported under
  `turboquant_k8v4` was three stacked spec-verify defects — RC1 captured
  verify graphs baking the raw-KV flash fast-path, RC2 a replay-broken
  multi-row synthetic-decode dispatch (fix: **split-row dispatch**),
  RC3 the MQ-kernel draft targeting the wheel's never-used SoA ops tree
  (fix: `tq_mq_aos.py` re-port to the engine's AoS layout). Fix ships
  wheel-side as `vllm_gaudi/xpu/patches/tq_spec_fix.py`, default ON.
- All-dtype long-return gate: **zero garble on all four TQ presets AND
  the fp8 control** (the hooked module never imports on fp8). fp8 faces
  byte-family cert7 (KV pool 400,739; t0 300.65 / 84.53–84.57; spec
  pos1 0.9258). TQ lanes servable again with capacity multipliers
  1.98×–3.68× (k8v4 spec tps 54.4 solo / 236 agg8; MQ experimental mode
  284.8 agg8, +21 %).
- Fork SHA `e47739b0`; evidence `results/2026-11-04-p619-tqverify/`.

## rel/v0.30.0-b70-cert7 — P6.18: TurboQuant interim refusal (2026-11-03)

- The corruption report (k8v4 corrupt returns) reproduced and split into
  Class B (deterministic REP-LOOP attractors, all presets) and Class A
  (t0 warm-resume trajectory flips); p618c roundtest proved the SoA
  store→decode pair numerically exact on XPU, pinning the then-diagnosis
  to a 4-bit-class precision ceiling → interim fix = config-time refusal
  of every `turboquant_*` dtype. **Superseded next day by cert8** — the
  diagnosis was wrong, the refusal is lifted, and the dtype serves
  clean. Long-return gate + determinism-semantics law + reproducibility
  audit banked here.
- Fork SHA `5bf4192d`; evidence `results/2026-11-03-p618b-repro/`,
  `…-p618c-roundtest/`, `…-p618-cert7/`.

## rel/v0.30.0-b70-cert6 — P6.14: TurboQuant presets (2026-11-02)

- The TQ-aware store fix (`_arc_attn_pool_view` TQ branch first,
  kernel-block→page merge, uint8 KV_2LTD init) + all four presets
  validated per-preset: KV capacity 526k/757k/853k/975k tokens
  (2.01/2.89/3.25/3.72×), U32k/U64k TTFT −30…−35 % vs fp8, spec ≥
  target, integrity green. (The TQ spec-verify defects surfaced later —
  see cert7/cert8.)
- Fork SHA `986e97d6`; evidence `results/2026-11-02-p614-*/`.

## rel/v0.30.0-b70-cert5 — P6.13: sliced final-span store, default OFF (2026-11-01)

- `LMC_STORE_SLICE_TOKENS` (default OFF): the sliced final-span store —
  safety/regression faces GREEN, mechanism landed exactly
  (63232-class = 0; 38 = 2×19 windows; U64k TTFT −1.5…−3.5 s); the
  pre-registered ≤600 ms max-gap bar was mis-set and recorded as such —
  merged default-OFF, enabling profile experimental/uncertified.
- Fork SHA `3e268627`; evidence `results/2026-11-01-p65-cert5/`.

## rel/v0.30.0-b70-cert4 — P6.5: certified posture image (2026-11-10-31 era; 2026-10-31)

- The p65 stall-capture campaign + the candidate battery on the shipped
  image; the long-context serve profile (`--long-prefill-token-threshold
  2048` + LMC deferral pair) and the stall-lever menu that still
  document CONFIGURATION's optional postures.
- Evidence: `results/2026-10-31-p65-cert4/`, `…-p65-r48a-stall-capture/`.

## rel/v0.30.0-b70-cert3 / cert2 / cert — image-process milestones (2026-10-31 and earlier)

- cert3: the tag-built image process (manifest A/B + image-native 28/28)
  — `results/2026-10-31-p6-r47e-cert3/`.
- cert2: 0.5.5-era lmcache line — valid, older.
- cert: predates the r46u glue-alias fix — never build from it.

## Serve-layer and analysis phases (no cert — engine bytes unchanged)

- **P6.46 performance-testing context ladder (2026-11-11)**:
  a bank audit (r138/r139/r140) proved every scheduled rung was swept
  to its KV wall — 128K walls hit exactly on all 11 long-context
  cells (n2–n7 by quant), gemma main hit its shallow 8K/32K walls —
  but 64K was in NO concurrency or annex matrix: the gates hardcoded
  CTXS {1K, 8K, 32K, 128K} while the rung was feasible on all 11
  boots (walls n4–n14 from pools 288,972–959,909). The r138 SOLO
  gate did carry the full ladder — the six qwen 64K solo faces
  are the only cert23-era 64K data (older r29/r46-era 64K probes are
  pre-cert image/harness: 54.8 s vs 38–42 s TTFT — archived under the
  comparability law, never merged). The gm8x refusal banks the mxfp8
  long-context boundary (5.83 GiB KV needed > 5.3 available;
  engine-estimated max length 103,296 — 128K genuinely needs MXFP4
  weights). LANDED RULE: `docs/TESTING-arc.md` gains the standing
  **performance-testing context ladder** section — canonical rung
  ladder {1K, 8K, 32K, 64K, 128K} on every boot where `max_model_len`
  and the booted pool permit, wall sweep `n_max = pool // (ctx + 664)`
  at conc {1,2,3,4,8,wall} + the t0.0/t1.0 pair, KV_CAP banking, the
  solo/conc rung-set equality, the same-image/same-harness merge law,
  and the banked feasibility table; both gates now carry 65536
  (`tools/ar/r139_conc.sh`, `tools/ar/r140_gemma128k.sh`; gemma main
  stays 3-rung — maxlen 40960; long gemma boots go through r140). A
  64K fill leg is ~60 faces across the 11 reachable cells on the
  updated gates, and `p139_report.py` picks the faces up with no
  generator change (P6.45 ladder rows + P6.44 connect lines already
  render the empty slots). No rig work run — the rule fires when
  a leg is needed. Docs + gate scripts only; no engine bytes changed —
  cert23 stands.
- **P6.45 canonical rung rows in every context table (2026-11-11)**: extends
  the ladder law from the chart axes (P6.43/P6.44) to the TABLES: every
  context-row table now renders the full canonical rung ladder up to its
  highest measured context — an unmeasured rung (64K on every boot here)
  shows a `-` row instead of the table jumping 32K→128K. A single helper
  `_ctx_axis` (extracted from the queue table's proven `qxd` expansion,
  which the queue and efficiency builders now share) drives the row sets:
  `allctx` (per-user speed, TTFT, KV pool), the prefill table + series,
  and the solo detail sweep; the per-context concurrency-sweep section
  keeps its heading at an unscheduled rung with an explicit not-measured
  note (a table whose rows are concurrency levels cannot carry dash
  rows). The KV-pool capacity column still computes at 64K from the
  banked pool arithmetic (the table is the whole-pool bound; the
  measured-green column reads `-`), so the gemma mxfp4 64K row shows the
  real 1-session bound. The embedded METHODOLOGY copy in the generator
  is synced to the hand-edited tree file (the regen copy had gone stale
  at P6.44) and both now state the table-row law; future regens emit the
  current text byte-identically. Oracle diff vs P6.44 output: all 175
  charts byte-identical (no chart bytes touched), 7 report mds changed —
  the six qwen configs gain only the 64K concurrency-sweep stub (their
  tables already carried 64K solo values from the r138 bank), gemma
  mxfp4 gains the five dash rows (speed, TTFT, KV pool, prefill, solo
  detail) + stub; gemma fp8/mxfp8 (max measured 32K) and the TQ
  per-preset blocks correctly gain nothing — the axis law stops at the
  highest measured rung. Same law, no fabrication: a table cell and a
  chart marker alike exist only where something was measured or
  derivable. Generator + docs only; no engine bytes changed — cert23
  stands.
- **P6.44 measured points connect across empty rung slots (2026-11-11)**:
  completes the P6.43 ladder law with the line rule a full-chart read
  demands: every line now CONNECTS its measured points, drawn straight
  across a labeled empty rung slot (64K) — standard uneven-x line
  plotting, no marker ever at the unmeasured rung. P6.43's rung
  padding (`(rung, None)` injection) had every line BREAK at the empty
  slot, leaving measured 128K points isolated after 32K (visible on
  the gemma queue decomposition: the n2 stagger point 89.0 s sat
  detached). The padding is removed; the axis keeps the honest 5-rung
  ladder. The break survives only where a face RAN and failed
  (banked `n_fail > 0`) or a per-n KV_CAP wall ends a line —
  `_conc_ctx_series`, `_ctx_edge`, and the queue/eff/prefill/tq-solo
  builders now distinguish rung-never-scheduled (no point, line
  connects) from ran-but-failed (None, honest break); `_solo_fill_n1`/
  `_solo_fill_edge` additionally ADD the solo-measured rungs (qwen 64K
  single-request point) instead of only filling failed slots. Band
  shading follows the same rule: one polygon over the both-measured
  domain, each edge following its own measured points (qwen top edge
  keeps its 64K point, the loaded bottom edge bridges 32K→128K
  straight). METHODOLOGY states the connect law. Oracle diff vs
  P6.43 output: 113 charts byte-identical, 62 changed — every chart
  whose series cross the 64K slot (gemma mxfp4 15, gemma-family 2,
  qwen 44, tq 1); the max-32K charts are untouched. Point-attachment
  audit: 1296 markers, 1276 on their line, 20 isolated by data (single-
  point series — gemma n8@1K and n5@8K on fp8/mxfp8, mxfp4-k0 n5@128K —
  one measurement, nothing to connect); ink checkers still 0
  overlapping pairs across 1603 value labels and 313 band tags. A full
  64K-presence deep audit
  against the three banks confirms the reports render 64K exactly as
  measured: the six qwen configs carry complete solo 64K faces in the
  r138 bank (TTFT 38–42 s; decode derived by the harness loader, 75–94
  tok/s; ITL; prefill ~1.7K tok/s) and every solo-family chart, table
  row, solo-completed n1 line, single-request band edge, and
  all-configs qwen series shows them; no concurrency face exists at
  64K anywhere (qwen sweep skipped it; gemma never planned it; the
  r140 annex is 1K/8K/32K/128K; the tq preset cells have no solo
  bank), so loaded lines bridge and the queue/eff/aggregate rows read
  "-" — the two remaining isolated 128K markers (mxfp4-k0 speed/ttft)
  are the genuinely single-point n5@128K series. Generator only; no
  engine bytes changed — cert23 stands.

- **P6.43 canonical rung ladder + no-chord padding + gemma 128K completion
  (2026-11-11)**: every context axis now keeps the canonical rung ladder
  (1K, 8K, 32K, 64K, 128K) up to its highest measured x, centralized in the
  two plotters. (1) **Labeled empty 64K slot on gemma**: the gemma boots
  never measured 64K, so the 17 gemma charts that reach 128K previously
  jumped 32K→128K with no 64K position; they now render the full 5-rung
  ladder with an empty, still-labeled 64K slot. (2) **Rung padding
  enforces no-chord centrally**: `chart()`/`band_chart()` inject an
  explicit `(rung, None)` pair for every rung a series did not measure,
  so the existing run-splitting breaks the line at the empty slot — this
  also fixed a latent defect where 4 of 5 `tq_speed_ctx` series had been
  chording 32K→128K across the unmeasured 64K rung since P6.42 (invisible
  to the tick-only audit). (3) **Gemma 128K completion**: the solo detail
  table gains its 131072 row (annex-boot faces, t1.0 decode 26.9 tok/s;
  `r140` faces carry `None` fields, cells are now None-safe), and the
  annex boot's t1.0@128K face joins the temperature comparison (both bars
  at 128K: 28 vs 27 tok/s) with the exec-success counts updated. Queue
  and efficiency tables carry the 64K "-" row with a generalized empty-slot
  note; the annex intro's stale "not merged into the charts above" is now
  the merge provenance sentence. METHODOLOGY gains the canonical-ladder
  law. (4) **Band-tag de-collision**: the picture-capture validation pass
  exposed that all 26 TTFT charts drew the right-margin class-threshold
  tags (instant/responsive/noticeable/slow) at 1.9–6.3 px baseline gaps —
  the four bands compress under a 0–300 s axis. The new `_hline_tags`
  helper (shared by all three plotters) spreads the stack order-
  preservingly (11.5 px pitch, centred, clamped inside the plot) only
  when a collision exists; uncompressed sets keep the line-hugging
  position, so the other 149 charts are byte-identical. Oracle diff vs
  P6.42 output: 136 charts byte-identical, 39 changed (19 ladder-set +
  26 band-tag-set, 6 in both) — qwen/fp8/mxfp4 outputs outside those
  sets untouched. All 175 charts rasterized at 2x (Edge headless) and
  validated: vision checks on the ladder/chord/band-tag/bar fixes, plus
  exact-box checkers — 0 overlapping pairs across 1603 value labels,
  0 across 313 band tags, 0 tag-vs-tick.
  Generator only; no engine bytes changed — cert23 stands.

- **P6.42 no-jump full-ladder axes + gemma 128K annex merge (2026-11-11)**:
  closes the last x-axis defects found by a full 175-chart tick audit.
  (1) **Break, never chord**: a `None` inside a series (or at a band edge)
  now splits the polyline into contiguous measured runs — a line can no
  longer bridge an unmeasured context with a straight segment that
  implies data there. (2) **Empty slots instead of axis jumps**: derived
  metrics that are unmeasurable at a context (queue wait, efficiency at
  the solo-only 64K rung) render that rung as a labeled empty slot on
  the full ladder (`xdomain`), instead of dropping it and making the
  axis jump 32K→128K; rows absent from a table show "-". (3) **Gemma
  128K annex merge**: `gemma-4-31b-mxfp4.md` main tables and charts no
  longer stop at 32K — the 128K annex boot's faces merge into the main
  report's data layer (cross-boot equivalence already delta-table-proven
  in the annex section: same image, same quantization posture), so the
  exec summary, per-user speed/TTFT/aggregate charts, solo decode/TTFT/
  ITL charts, queue table, KV-pool table (128K row uses the annex boot's
  pool, footnoted), use-case bars, and the cross-model all-chart now all
  run 1K→128K, with a provenance sentence under the exec summary. The
  tq per-user speed line is completed to the full five-rung ladder the
  same way. Oracle diff vs P6.41 output: 115 charts byte-identical,
  60 changed, all in the intended families; 1,602 value labels,
  0 overlaps. No engine bytes changed; cert23 stands.
- **P6.41 linear x-axis + full measured prefill coverage (2026-11-11)**:
  every context-length line chart re-spaced from log2 to a **linear
  equal-slot axis** — one evenly spaced slot per measured context (no
  log compression, no value-proportional bunching), all middle labels
  kept, and the axis stops at the highest context actually measured.
  The qwen concurrency sweep skipped 64K (sweep = 1K/8K/32K/128K) while
  the solo faces measured it: the "1 request" line, its band edge, and
  the prefill table/chart are now completed from each configuration's
  solo faces (the same single-request measurement on the same certified
  image — cross-boot drift bounded near zero by the same-contexts delta
  tables), so the prefill chart spans the full measured 1K-128K with
  64K restored; loaded lines keep honest gaps and band shading is now
  drawn per contiguous measured run (never bridges an unmeasured
  context). 12 reports / 175 charts (unchanged counts); 1,581 value
  labels, 0 overlaps. No engine bytes changed; cert23 stands.
- **P6.40 presentation overhaul — context-length x-axis + range bands
  (2026-11-11)**: every line chart re-anchored so the x-axis measures
  **context length** and concurrency becomes the plotted series (the
  prior x = concurrency charts retired); per-user speed and TTFT gain
  millstone-style **Range band charts** (the measured envelope: solo
  edge vs heaviest tested load) with dashed quality-threshold guide
  lines on the use-case bars; every chart now ships a data-precise
  *Interpretation* paragraph and sits directly under its table in the
  same section. Near-duplicate reports eliminated:
  `gemma-4-31b-mxfp4-128k.md` merged into `gemma-4-31b-mxfp4.md` as a
  full 128K annex (same-contexts delta table proves both boots agree)
  and `gemma-4-31b-mxfp8-128k.md` into `gemma-4-31b-mxfp8.md` as the
  refused-131K control — 12 reports / 175 charts (was 14 / 108).
  Generator refactor (`p139_report.py`: label-placement extraction,
  `band_chart`, per-concurrency series builders) proven
  behavior-preserving by oracle diff — 41 untouched charts
  byte-identical; 1,545 point labels, 0 overlaps across all 175.
  No engine bytes changed; cert23 stands. Reanalysis of the banked
  P6.37/P6.39/P6.39c data, no new banks.
- **P6.39 concurrency reports + gemma 128K annex (2026-11-11)**: every
  per-config report gains the measured concurrency 1–8 series and the
  paired temperature 0.0/1.0 faces; all line-chart points carry value
  labels (staggered collision-free placement — 1,360 labels, 0 overlaps
  across 108 charts); the gemma MXFP4 128K annex (max_model_len 131072,
  KV pool 288,972 tokens: solo 28.4 tok/s @ 128K, TTFT 175.9 s,
  capacity 2 concurrent requests) with the mxfp8 131K boot refusal
  banking the 8-bit boundary; the TurboQuant KV presets and the gemma
  fp8 auto-KV wall (~13.3K ctx) measured. Engineered boundary find: the
  API server enforces **prompt_len + max_tokens ≤ max_model_len**
  (probe-banked verbatim) — gemma's sentencepiece round-trip inflates
  1.026x, so the harness now sizes server-side prompt budgets
  (ctx − max_tokens − 64). Docs tooling hardened: `validate_docs.py`
  (0 errors across 29 fork docs) and a two-pass `leak_sweep.sh`. No
  engine bytes changed; cert23 stands. Banks
  `results/2026-11-11-p637b-conc/` + `results/2026-11-11-r140-gemma128k/`.
- **P6.38 docs restructure (2026-11-11)**: current-state-only living docs
  + the two-tier history architecture — `CHANGELOG.md` as the pure
  changelog (the consolidated-history section removed), per-doc archives
  `docs/history/{STATUS,TESTING,MIGRATION,PERFORMANCE,CONFIGURATION}_HISTORY.md`
  (every outgoing table/narrative banked verbatim), and
  `docs/release_notes.md` as the upstream-style release index. Era
  labels (rNN leg IDs in prose, cert-era tags, phase prefixes) stripped
  from README + all living docs; two latent staleness bugs fixed
  (CONFIGURATION Step-1 docker image cert19→cert23; draft-sampling
  ARC_IMAGE example cert19→cert23) plus the envs.md currency bugs
  (`LMCACHE_MAX_LOCAL_CPU_SIZE` 100→16, host-var count 23→24 with
  `ARC_KVDTYPE`). No engine bytes changed; cert23 stands.
- **P6.37 close-out (2026-11-10/11)**: the per-config performance reports
  (`docs/reports/`), 64 GB-RAM host defaults (`--shm-size 16g`,
  `LMCACHE_MAX_LOCAL_CPU_SIZE=16`), the `ARC_KVDTYPE` knob + measured
  dtype boundaries, `docs/PATCHES-arc.md`. Bank
  `results/2026-11-10-p637-reports/`.
- **P6.36 prefill-window decomposition (2026-11-10)**: the prefill
  window is device drain (80 % of U32k = engine-core waits on chunk
  device completion); rank skew ZERO; every B discriminator negative;
  decode D2 ceiling +8–10 % → **no remaining lawful lever ≥ 10 %**,
  cert23 stands. `results/2026-11-10-p636-legA/`.
- **P6.34 joint-matrix skinny-kernel research (2026-11-09)**: the
  fragment-op serialization law + SLM-staging law banked; ladder
  numerics never certify — lane closed research-only.
  `results/2026-11-09-p634-jm-ladder/`.
- **P6.32 Xe2 FP8-nativeity verdict (2026-11-08)**: no native FP8 dot on
  Xe2 (counter-operand conversion only; B70 = BMG-G31, 256 XMX); the
  kernel-feasibility map + Option-B backport policy.
- **P6.27 MXFP8 AutoRound validation (2026-11-06)**: the intel-ai MXFP8
  checkpoint validated zero-change on cert11 (native
  `XPUMxFp8LinearKernel`, quantized MTP drafter, spec 0.9626/0.9174) —
  a config-driven model route, no cert. `results/2026-11-06-p627-mxfp8/`.
- **P6.25 overlap-door audit (2026-11-06)**: cross-stream AR∥GEMM
  co-tenancy closed MEASURED (overlap slower than serial at every k on
  both wires; the device time-slices AR and GEMM) — every prefill door
  now measured-closed. `results/2026-11-06-p625-overlap-door/`.
- **P6.23/P6.24b/P6.24d/P6.24e doc + patch audits (2026-11-05/06)**:
  docs-currency audit, P6.24 patch audit, STATUS-ledger move,
  best-performance-postures consolidation.
- **P6.21/P6.22 device-compute lanes (2026-11-05)**: ar_quant donor
  (−4…−9 %, unadopted), FA headroom ladder (stock at 25 % of peak but
  every patch route falsified), ESIMD dense-FA v2 (0.33× stock, ceiling
  ~0.83× — under bar), TP2 one-shot allreduce (stock ring 2.06× faster
  than the one-shot ceiling) — all closed NEGATIVE.
  `results/2026-11-05-p621*/`, `…-p622-ar-lane/`.
- **P6.15 dtype/stall probes + collapse mechanism (2026-11-02/03)**:
  bystander-stall taxonomy + MNBT Pareto; the t1.0-xhigh
  sampled-collapse mechanism pinned; 12-face master matrix.
- **P6.17 prob-lane certification (2026-11-03)**: the certified line is
  `ARC_SPEC_METHOD=probabilistic` (serve-script default); every record
  face re-measured on that lane; the P6.16 opt-in hook became the
  default. `results/2026-11-03-p617a-probcert/` (+p616 lineage).
- **P6.16 draft-sampling posture (2026-11-03)**: the probabilistic
  serve hook measured (t1.0 agg +29 % fp8 … +91 % 3bit) — merged
  serve-layer. `results/2026-11-03-p616*/`.
- **P6.1–P6.13 (2026-10-30 … 2026-11-01)**: external-hit-rate,
  0.5.6+arc1 lmcache gates, p65 stall capture + cert4, p66 prefill
  anatomy (device-bound, MNBT-16384 not adopted), p67 compile REVAL
  rejected, p70 cert4-native comprehensive re-measurement + 4 h soak,
  p71 P2P transport proven clean, p73 END-stall analysis + the sliced
  store. Evidence per `results/` phase dirs.
- **P0–P5 (2026-10-27 … 2026-10-30)**: venv ground truth, platform +
  first serve, spec/graph defect fixes, cache lane (connector, hybrid
  state, elide/memo, tail-split), certification battery + certified
  image + 12-cell matrix, env/doc/lint audits + 8 h soak + the r46t
  real-world campaign (k3 recommended). Evidence per `results/` phase
  dirs.
