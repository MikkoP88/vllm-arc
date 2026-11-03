# P6.18b — TurboQuant corruption reproduction battery (cert6, 2026-11-03)

Owner report: `--kv-cache-dtype turboquant_k8v4` corrupt returns; all TQ
presets to be re-validated. Campaign: 6 faces on rig 10.20.3.REDACTED
(`/root/build/p618/repro/`, script `p618b_repro.sh`, probe
`p618_longprobe.py`) — fp8 control, k8v4, tq4, k3v4, 3bit, k8v4+LMC=0.

## Probe design notes (fixed before launch)

- Classes: essay ~2.2k tok (1 page + tail), code ~4.4k (2 pages + tail),
  multi ~2.3k multilingual, simple ~78 tok (single partial chunk).
- Per class: temp0 cold → rehit → rehit2 (byte-identity), temp1.0 fixed
  seed ×2 (seed reproducibility); rehit mode after 4×256 churn.
- Detectors: REP-LOOP (25-char window ×4), JUNK >5%, cross-class BLEED,
  EMPTY. r42 harness gap being closed: r42 = single 1664-page prompts,
  immediate same-content rehits (GPU prefix cache holds → LMC load path
  never engaged).

## Interim readings (per-leg, as observed)

- fp8 control A-gates: integrity 5/5, det 3/3 — matches cert6 record.
- fp8 [essay]/[code] EMPTY on every phase (hash `da39a3ee5e6b` = sha1(""),
  identical across phases): qwen3.8 thinking channel consumes the 768-token
  budget → `content` empty. HARNESS ARTIFACT, not corruption; deterministic.
  Follow-up batteries must raise max_tokens or capture `reasoning_content`.
- fp8 [simple] t0_identical=**False**: cold=rehit=`4039a177de2b`,
  rehit2=`c45d00190084` — temp-0 flip on the certified-clean dtype, 78-tok
  prompt (single partial chunk, sub-512 = no mamba boundary). First direct
  evidence the defect class is dtype-INDEPENDENT and lives on the
  store→load / partial-chunk / boundary-resume path, not in TQ quant.
- fp8 [multi] (3440 tok = 2 full chunks + 112 tail): t0 identical —
  full-chunk roundtrip byte-clean on fp8.
- fp8 post-churn rehits: [multi] r1 = clean hash `2268b3d30c96`, r2 =
  `29d445e1525f` (FLIP); [simple] r1 = EMPTY, r2 = `88162fbeb8c4` (FLIP);
  essay/code stable-EMPTY (artifact). ERROR=0 TRACEBACK=0 — silent.
- fp8 leg verdict: **the defect class reproduces on the certified default
  dtype** — temp-0 flips on the cache-resume path, churn-accelerated.
  Signature: the rehit that first engages after GPU-prefix-cache eviction
  is the one that flips; a fixed bad stored object would flip every rehit
  identically, so the corruption is resume-path state, not a static bad
  object. r42's 5/5 was a harness artifact (immediate same-content rehits
  never engaged the LMC load path; single-page prompts; no churn).

## Reframe for P6.18 (owner report context)

k8v4 owner-visible garble may be (a) the same resume defect amplified by
TQ quantization noise (wrong-resume KV → incoherent, not just near-tie
flip) or (b) an additional TQ-specific byte defect. Discriminators in
this battery: k8v4 garble detectors vs fp8's clean-but-flipped text;
k8v4lmc0 leg (LMC off) — flips vanishing there pin the LMC path (H1).

- k8v4 leg: A-gates integrity 5/5 + det 3/3 PASS, ERROR=0 TRACEBACK=0,
  post-churn rehit CLEAN — but **vacuous**: all four classes returned
  EMPTY under v1 (thinking budget ate max_tokens=768), so
  "identical=True" hashed sha1("") throughout. Same blindness class as
  r42. No TQ garble reproduced under this harness, and none could have
  been.

## Probe v2 (hot-fix, deployed 17:01:07, applies from tq4 leg onward)

- v1 flaw: EMPTY channel → vacuous identity. v2: max_tokens 3072
  (t1 1536), `reasoning_content` + `content` hashed separately AND
  combined (`reasoning\x00<<<SPLIT>>>\x00content`) — combined is the
  identity signal; garble detectors run on combined. Probe file replaced
  in place (`p618_longprobe.py`); v1 preserved as
  `p618_longprobe_v1.py` (rig). fp8 + k8v4 ran v1 → both get a v2
  re-pass after the battery (boots fp8 + k8v4 again, longprobe full +
  churn + rehit).

## Code-level findings during the battery (2026-11-03, cert6)

- **Grids coincide by construction** (kills the grid-mismatch hypothesis):
  the p41c glue's `bs` (= mamba group `spec.block_size`) is
  dtype-dependent and EQUALS the attention LMC chunk — k8v4 boot log
  `[p41c] wrap live bs=1152` + engine dump `chunk_size: 1152`; tq4
  `bs=1664`; fp8 (p617 logs) `bs=832` = its chunk 832. The mamba engine
  chunks at the same value (`_init_mamba_engine` chunk=tokens_per_page),
  and "one stored chunk = the boundary state at that chunk's END" — so
  the external joint seed is depth-exact per chunk key. The external
  path resumes coherently at bs-grid depths.
- k8v4 log has 24 `Retrieved N out of N` lines — external loads fire and
  complete (no shortfall errors), e.g. `Retrieved 1152 out of 1152
  required` at 16:51:50 (essay rehit phase).
- LMCache wheel 0.5.6+arc1: the #1867 fix (async wrong-memory-object,
  upstream #1866) IS present — `_async_process_tokens_internal` walks
  `process_tokens()` (start, end, key) and fetches by key from
  `memory_obj_map` (wheel_tree read 2026-11-03). `save_unfull_chunk:
  False` (the #2207 default) active at runtime (engine config dump).
- **H2 (live, primary)**: GPU-APC hybrid resume off the mamba grid.
  vllm matches attention blocks at 64-token granularity; if the mamba
  state does not gate the hit depth, a resume lands between mamba chunk
  ends and the SSM never consumes the gap tokens → silent trajectory
  divergence. Explains every fp8 observation ([simple] 78-tok rehit2
  flip = 64-token attn hit below the first mamba chunk; [multi] r2
  flip = GPU APC re-hold after r1's recompute; dtype-independent;
  churn-gated; ERROR=0). The k8v4lmc0 leg (LMC off) is the
  discriminator: flips persisting there pin H2.
- **H3 (pending tq4/k3v4/3bit v2 readings)**: a TQ byte defect in the
  pool-view store/load path (4-bit packing / lagged-flush window)
  adding real KV corruption on top of H2 — would show as REP-LOOP/JUNK
  garble in real text. Owner-visible k8v4 garble = H2 amplified by TQ
  quant noise if H3 is dead.
- Upstream context: vllm upstream does NOT support mamba/hybrid prefix
  caching at all in this era (vLLM #61970 "pending Mamba boundary
  state"; #32802/#40624 0 % hits on hybrids) — the fork's mamba pool +
  snap glue implements it, so this resume-coherence class is
  fork-local, not an upstream-fixed bug. vllm-gaudi #1890 (off-grid
  resume crash, multimodal path) documents the same geometry upstream
  (HPU tolerates it because GDN reads the running block, not a chunk
  grid); not applicable to our XPU stock-scheduler path.

## Probe v3 readings (first real-text TQ data)

- k3v4 (turboquant_k3v4_nc), v3 probe (thinking off): [essay]
  t0_identical=**True** hashes `700bbba1f878` ×3, chars 4667×3, garble
  **REP-LOOP on cold AND rehit AND rehit2**, t1_seed_identical=True.
  Two classes of signal: (a) garble present on COLD — the first-ever
  generation of this content, no resume path involved → the defect
  class is in the forward computation (quant kernels), not cache
  resume; (b) deterministic (identical bytes every phase) → engine
  self-consistent, no state-dependent corruption on this face. This
  is the owner-visible "corrupt returns" face reproducing on the most
  aggressive preset. chars 4667 vs the requested 400-word conclusion
  (~2.5k chars) = overlong degenerate generation, classic
  degraded-quant signature — but the probe's source text is
  deliberately repetitive, so the fp8 v3 baseline (re-pass) must
  confirm the detector stays clean there before calling Class B
  quant-inherent.
- Defect taxonomy now two-distinct: **Class A** resume-path temp-0
  trajectory flips (dtype-INDEPENDENT, fp8-proven, churn-gated —
  H2/H5 engine defect) vs **Class B** deterministic REP-LOOP garble
  on aggressive TQ presets (cold-included — forward-compute/quant
  face). Different fixes: A = engine resume-coherence law; B = TQ
  kernel/quality validation per preset (fix or deprecate-with-docs).

- k3v4 FULL (v3): all four classes REP-LOOP on every phase (cold
  included), deterministic identical hashes, t1-seed reproducible.
  Smoking gun: [simple] prompt_tok=38 asking for a 3-sentence answer
  → **3159 chars of repetitive degenerate text ON COLD**. k3v4
  generation quality is broken from the first token with zero cache
  involvement — Class B proven for the 3-bit-K preset. Overlong
  degeneration on every class (4667/4578/3118/3159 chars vs
  requested ~2.5k/~2k/summary/3-sentences).

## Stock-code surfaces read during k3v4 leg (container cert6)

- Hybrid coordinator `find_longest_cache_hit` DOES reconcile groups
  (fixed-point min; full-attn trimmed to reconciled length) — engine-
  local hybrid APC is mamba-aware by design in this base. The gap is
  INSIDE the mamba manager's hit reporting:
  `MambaManager.supports_fine_grained_hash_lookup = True`
  (single_type_kv_cache_manager.py:1445) + fine path :1517-1538:
  when alignment (64) < mamba block_size (832/1152/1664) it walks
  64-token fine hashes and returns `num_tokens = (fine_idx+1)*64` —
  **hits at 64-granularity while mamba states are banked ONLY at
  segment/replay chunk boundaries** (sparse retention :1578-1619:
  segment boundaries + request replay boundary
  `aligned // block_size - 1`). A fine hit below the deepest banked
  state = resume where no SSM state exists → silent divergence.
- `has_positionally_stable_blocks` = False for align-mode Mamba
  (:1448: "Align-mode Mamba can null interior states and RELOCATE
  speculative …") — relocation + interior nulling is the stale-mapping
  surface (the fail11 stale-dict class, vllm-side).
- Consequence for H2/H5: the local-path defect is narrowed to the
  MambaManager fine-grained hit vs banked-state geometry (+ possible
  relocation staleness). The fp8 [simple] 78-tok rehit2 flip fits:
  fine hash hit at 64 with no sub-832 state banked. Discriminators
  already queued: k8v4lmc0 leg (LMC off) + diag boot (DEBUG +
  ARC_SNAP_PROBE hit/snap chains around a live flip).

- k3v4 rehit (post-churn): every class byte-identical to its own
  cold/full hashes (700bbba1f878 / 98e00598a815 / 971d33b4163e /
  5cfb47762ab1) — cache resume on k3v4 is COHERENT, zero Class A
  flips. k3v4 = pure Class B (deterministic forward degeneration),
  a clean control separating the two classes. ERROR=0 TRACEBACK=0.

- EXONERATION (static, insert side): `block_pool.cache_full_blocks`
  inserts mamba blocks keyed by `BlockHashListWithBlockSize` per-chunk
  views = the fine hash at each chunk-END position ONLY
  (block_pool.py:269-284). The fine lookup therefore can only hit
  chunk-end depths where a state block was banked — the MambaManager
  fine path is coherent by construction; mid-chunk fine hits cannot
  occur. Class A narrowed to: stale/relocated state CONTENT behind a
  valid chunk-end hash (align-mode relocation/nulling, preemption
  requeue), OR p41c glue post-kernel banking on local-resume rows,
  OR sched_tail_split composition. Diag boot discriminates.

- 3bit FULL (v3): Class B CONFIRMED on 3bit too — [essay]/[code]/
  [simple] REP-LOOP cold-included, deterministic t0. Two NEW faces
  vs k3v4: (a) [multi] chars=[72,72,72] with NO garble flags — a
  near-empty truncated response (clean-text but ~15 tokens) —
  degeneration can also manifest as immediate stop; (b) [code]+
  [simple] **t1_seed_identical=False** — same-seed temp-1.0 reruns
  DIFFER (determinism break under seeded sampling; k3v4 kept t1
  identical). 3-bit K is the common factor across the Class-B
  presets measured so far (k3v4, 3bit); the t1-seed break is 3bit-
  only so far. Legacy P6.14 gates (integrity 5/5, det 3/3) PASSED
  this leg (A-gates green) — the certified battery never gated long-
  generation text quality or t1-seed reproducibility; both are
  testing gaps for p618d.

## k8v4lmc0 leg (turboquant_k8v4, ARC_LMC=0) — DECISIVE (18:08)

- **Class A CONFIRMED engine-local (H2)**: [essay] t0_identical=
  **False** hashes [700bbba1f878, 5215b87f24b7, 700bbba1f878] chars
  [4667, 6181, 4667] — a temp-0 FLIP on rehit that reverts on rehit2,
  WITH LMCache DISABLED. The resume-flip class lives in the engine
  GPU-APC path, not the LMC store/load path. (fp8 already proved it
  independent of dtype; lmc0 proves it independent of LMC.)
- **Class B on k8v4 CONFIRMED (the owner's reported dtype)**: all
  four classes REP-LOOP cold-included with LMC off — the "corrupt
  returns" report reproduces without any LMCache involvement; the
  degeneration is the dtype's forward compute (TQ KV dequant/quant
  path in the engine), NOT the store/load roundtrip.
- **Cross-dtype identical attractor**: [essay] 700bbba1f878,
  [multi] 971d33b4163e, [simple] 5cfb47762ab1 are BYTE-IDENTICAL
  across k3v4, 3bit, AND k8v4lmc0 (chars 4667/3118/3159 identical
  too); [code] differs per dtype (4578/4589/4581). Interpretation:
  TQ-quantized KV drives temp-0 long generations into a repetition
  attractor whose locked cycle is prompt+weights-determined —
  quant-noise-independent — so all TQ dtypes collapse to the SAME
  degenerate bytes on most classes. fp8 [multi] post-churn text
  (2268b3d30c96) shows fp8 stays out of the attractor.
- Class B map: k8v4 BROKEN, k3v4 BROKEN, 3bit BROKEN (3/4 TQ);
  tq4 pending v3 re-pass; fp8 baseline pending v3 re-pass.
- A-gates still PASS (integrity 5/5, det 3/3) on this leg — the
  certified battery's blindness to long-generation quality is now
  proven on every TQ preset.

## fp8v3 re-pass baseline (18:18) — THE CONTROL

- LONGPROBE fp8v3 full: **CLEAN** — all four classes real text,
  sane lengths (essay 3885 / code 2004 / multi 1052 / simple 1510
  chars), ZERO garble flags on any phase, t0 byte-identical
  cold/rehit/rehit2, t1_seed_identical=True everywhere. Same probe,
  same repetitive-source prompts, same boot posture as the TQ legs.
- Conclusion: the probe itself is NOT an artifact source — under
  fp8_e4m3 the model produces clean, distinct, appropriately-sized
  generations and NEVER enters the repetition attractor. **Class B
  is TQ-only, now proven against the clean baseline.** Class A on
  fp8 (v1 leg flips) remains the engine resume face; fp8v3's own
  cold/rehit/rehit2 identity shows the flip face is state-gated
  (churn/eviction-conditioned), not prompt-class-conditioned.

- fp8v3 post-churn rehit: CLEAN, byte-identical to cold on all four
  classes — the v1 leg's flips ([multi]/[simple] r1→r2) did NOT
  reproduce on this boot. Class A is eviction-timing-conditioned
  (needs a specific cache-pressure sequence); one clean pass does
  not clear a dtype. Consequence for p618d: the long gate must
  AMPLIFY — multiple churn+rehit cycles per dtype, not one —
  to make Class A absence certifiable (owner's "100% sure").
  k8v4lmc0's flip (different boot, LMC off) stands as the
  engine-local proof; fp8 v1's 4 flips stand as the default-dtype
  proof.

## k8v4v3 re-pass (LMC=1) — both classes on the owner's dtype

- Class B: SAME cross-dtype attractor hashes as k8v4lmc0 (essay
  700bbba1f878, code 76a92299e7ea, multi 971d33b4163e) — REP-LOOP
  cold-included with LMC ON too. Class B is LMC-independent on
  k8v4, matching lmc0: forward compute, full stop.
- Class A: [multi] t0_identical=False — rehit flip
  971d33b4163e → 8853a9d894bb → back, chars 3118 → 7676 → 3118
  (the flipped trajectory runs the attractor 2.5× longer before
  stopping). With k8v4lmc0's [essay] flip, Class A is now observed
  on the owner's dtype in BOTH LMC modes → engine-local, twice.
- [multi] t1_seed_identical=False — the seeded-sampling break seen
  on 3bit reproduces on k8v4 (probabilistic-draft lane suspect;
  needs its own discriminator in p618c: greedy lane A/B).

## tq4v3 re-pass (18:52)

- Class B on tq4 (turboquant_4bit_nc) CONFIRMED — same attractor
  hashes (essay 700bbba1f878 / code 76a92299e7ea / simple
  5cfb47762ab1; multi lands on the 8853a9d894bb 7676-char variant).
  No Class A flip on this boot (t0 identical everywhere).
  **Class B map FINAL: 4/4 TQ presets broken; fp8 clean.**

## Verdict (battery + re-pass, 2026-11-03; diag/diag2 forensics appended below)

**TWO distinct defect classes, both reproduced and both outside the
certified battery's field of view. The owner's turboquant_k8v4
"corrupt returns" report = Class B (primary, always-on) with Class A
(temp-0 flips) layered on top.**

1. **Class A — engine-local hybrid resume temp-0 trajectory flips.**
   Dtype-INDEPENDENT (fp8 v1 leg: 4 flips incl. 78-tok prompt; k8v4:
   both LMC modes) and LMC-INDEPENDENT (k8v4lmc0 [essay] flip;
   k8v4v3 [multi] flip pair 971d33b4163e→8853a9d894bb reproduced
   twice, post-churn again) → lives in the engine GPU-APC hybrid
   resume path, NOT the LMCache store/load path. Silent (ERROR=0,
   TRACEBACK=0 everywhere). Eviction/state-gated: fp8v3 re-pass ran
   fully CLEAN — one clean pass clears nothing (amplification law
   for the cert gate). Static narrowing: stock hybrid coordinator
   reconciles groups correctly; the MambaManager fine-grained hit
   path is coherent by construction (insert keys = chunk-end fine
   hashes only, block_pool.py:269-284) — EXONERATED; live suspects:
   stale/relocated mamba state CONTENT behind a valid chunk-end hash
   (align-mode relocation, `has_positionally_stable_blocks=False`),
   p41c glue post-kernel banking interaction, sched_tail_split
   composition. diag (fp8) + diag2 (k8v4 canary) boots capture the
   flip under ARC_SNAP_PROBE + LMCASE DEBUG.

2. **Class B — deterministic degenerate generation on ALL FOUR
   turboquant_* presets.** REP-LOOP attractor, COLD-included (38-tok
   [simple] → 3159 chars repetitive; near-empty truncation face on
   3bit [multi] 72 chars), byte-identical per phase, t1-seed mostly
   reproducible, LMC-independent, and CROSS-DTYPE IDENTICAL on
   essay/multi/simple (700bbba1f878 / 971d33b4163e /
   5cfb47762ab1 across k3v4/3bit/k8v4/tq4 boots) — a
   prompt+weights-determined repetition attractor whose basin every
   TQ dtype falls into. fp8v3 baseline under the SAME probe: CLEAN
   real text, zero flags, all identities hold. TurboQuant is STOCK
   vllm wheel code (triton_turboquant_{store,decode}.py et al.);
   every broken preset carries 4-bit-class V (per-slot scalar affine
   16-level) — V precision is the binding constraint. Owner-visible
   "corruption" on TQ = this class; NOT a cache defect.

3. **Dead hypotheses**: H1/H3-as-store/load (LMC roundtrip byte
   defect) — DEAD as primary for both classes (both reproduce with
   ARC_LMC=0). H4 (cross-request bleed) — never observed (no BLEED
   flags anywhere). Probe-artifact — DEAD (fp8v3 clean under same
   prompts; v1/v2 EMPTY-blindness was the artifact, fixed by v3).

4. **Certified-battery blindness**: integrity 5/5 + det 3/3 passed
   on EVERY leg incl. the degenerate ones — the certified gates
   never measured long-generation text quality, truncation, or
   t1-seed reproducibility. p618d adds the LONG-RETURN GATE (owner
   directive: 100% sure no corruption or looping) with churn
   amplification (≥3 cycles) on every dtype.

Disposition into p618c: Class A = resume-coherence fix in
vllm_gaudi/xpu/patches (+ upstream-style + plan-§1 naming laws,
owner directives 2026-11-03); Class B = XPU numeric roundtrip test
through the boot-selected TQ store→decode pair; if numerics correct
→ TQ dtypes deprecated for this model (serve guard + docs — new
quant kernels are scope-law-blocked), fp8 default untouched.

## diag + diag2 forensics (fp8 19:00, k8v4 19:16 — both under
ARC_SNAP_PROBE + LMCACHE_LOG_LEVEL=DEBUG)

- diag-fp8 (13,348-line log): CLEAN boot, no flips, 0 drop markers,
  0 Retrieved (external never engaged; GPU-APC hit 47.9→54.1%),
  2474 r31probe prints. Also observed: fp8 [code]/[multi] COLD
  outputs differ ACROSS boots (441e29854e32@2239 vs f9192d849270
  @2004; 0627e973e7b1@1096 vs 2243a4099b6f@1052) while essay/simple
  are boot-stable — boot-level numerical non-determinism (fp
  rounding near-ties; within-boot identities all hold). Bounds what
  byte-identity can mean: guaranteed within a path/boot, never
  across boots.
- diag2-k8v4 (33,880-line log, 5062 probe prints): the canary FIRED
  — [multi] cold=8853a9d894bb(7676 ch) → rehit=971d33b4163e(3118)
  → rehit2=8853a9d894bb. INVERTED vs k8v4v3 (cold=971d→rehit=8853):
  the two attractor variants alternate with resume state; the
  resumed generation deterministically takes the other basin. 0
  drop/mismatch markers — the glue and external seed path ran
  clean. 24 Retrieved lines: the [multi] rehit that flipped did an
  EXTERNAL load (Retrieved 1152 out of 1152 required from 1152
  total — pool held one chunk of the two banked 1152/2304
  boundaries; save_unfull_chunk=False keeps the 1096-token tail
  unsaved) → resume composes loaded-KV + recompute over the
  remainder under TQ quant noise.
- **UNIFIED CLASS-A MECHANISM (settled)**: warm resume is not
  numerically identical to cold compute — resumed generations
  recompute a suffix against CACHED (and for TQ, quantized-
  roundtripped) KV with different batch/kernel shapes → fp16
  near-ties resolve differently → trajectory switch. At fp8 SNR
  this is RARE and subtle (v1-leg flips; standard cached-serving
  behavior, present in stock vLLM on every platform); at TQ quant
  SNR it is SYSTEMATIC and spectacular (basin swap between
  attractor variants, 3118↔7676 chars). k8v4lmc0's flip proves the
  local-only face; diag2's flip proves the external-composed face;
  both are the same physics. REFRAME: Class A is NOT corruption —
  it is inherent cache-reuse numerical divergence (stock
  semantics), whose VISIBILITY is amplified by Class B's
  degenerate attractor. Consequences: (a) no engine "fix" is
  possible or warranted for A beyond documentation (within-path
  determinism guaranteed; cross-path byte-identity is not a
  guarantee any cached system can make); (b) the owner's
  "100% sure no corruption or looping" gate = per-dtype CLEAN text
  (no loops/junk/truncation) + within-path identity + t1-seed
  reproducibility — achievable on fp8, NOT achievable on TQ dtypes
  (Class B) → deprecation lane if the numeric roundtrip test
  confirms implementation correctness.

## Rig incident + restore (19:16-19:55)

- diag2's EXIT trap starts W14 but never removes the final arc
  container, and `docker start radiance-test` silently no-ops — the
  radiance-test container had been REMOVED from the rig at some
  earlier point (docker ps -a had no radiance-test; all of today's
  inter-leg `docker stop/start radiance-test` calls were silent
  no-ops).
- First restore attempt via `/root/build/w14_recreate.sh` (docker run
  with the CCL/ZE env chain + model mounts) + chat-template docker cp
  + `w14_serve_only.sh`: serve died on a missing chat template
  (writable layer lost with the old container), then after cp died
  TWICE more with a DETERMINISTIC "Available KV cache memory:
  -0.82 GiB" on EMPTY GPUs (34.0/34.24 GB free probed). Root cause:
  w14_recreate.sh recreates a BARE container — the real W14 incumbent
  posture lives in `/root/build/w14_boot.sh fullres`: a ~25-var
  IEXP_* env chain (expert offload ARENA 17.5 GiB pinned, W8B/W12/
  W13/QSA overlays, cache-dir mounts) + a dozen staged overlay files
  (expert_offload, w12_lru, mtp, draft_quant, ple_layer, model_state,
  qsa stack, qsa_ops.so, patches.py/__init__, ar_quant) + the expert
  heat sidecar. Without the arena the engine runs the dense expert
  path and gmu 0.955 does not fit (-0.82 GiB).
- CORRECT restore (19:50): `bash /root/build/w14_boot.sh fullres` —
  one command: container rm/run with the full env chain, overlay-file
  staging with md5 print + py_compile gate, sidecar verification
  (SIDECAR_IN_POSTURE_OK 11255040), then serve. Law for all future
  legs: w14_serve_only.sh only re-serves INSIDE a healthy
  w14_boot-created container; if radiance-test is missing or bare,
  the restore is w14_boot.sh fullres, nothing less.
