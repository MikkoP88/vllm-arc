# P6.34 — JM skinny: g16 gate-ref root cause + per-step cost ladder (V6)

Date: 2026-11-09 (rig `arc-jm7` on build6 wheel, builder `vxk-arc3-builder`)

Chain: two questions banked here.

1. **Why did g16 fail the gate while every isolation probe passed?**
   Answer: the GATE's f64 reference was wrong, not the kernel.
2. **Where do the measured ~2.9–6.4 μs per 16-k-step of the JM kernel
   go, and what structural change removes them?**
   Answer: 98.7% is the same-step staging→barrier→load serialization;
   the r128k V6 ping-pong restructure removes 41–108× of it.
   *(Perf answer stands; the V6 CORRECTNESS claim is retracted by
   Part 4 — r128k2 laws 1+2.)*

## Part 1 — g16 gate false-fail: `torch.stack` dim semantics

`r128b_jm_gate.py` GATE-1 failed 10/10 g16 cells (rel ~1.76, "out =
multiples of 0.5", deterministic) on build6 while the SAME kernel
passed every r128i/r128j telemetry probe (one-hot coefficients exact,
block sweeps clean, x-scaling exact, determinism + poison immunity,
clone/address-order invariance).

Probe chain (all in `tools/ar/`, banked at rig `/root/build/r128a/`):

- `r128j_gcell.py` — 2×2×2 discriminator: cells A/B/C FAIL
  deterministic, D–G need padded scales → not construction-dependent.
- `r128j2_toggle.py` — 6 construction toggles all PASS → exonerates
  repack construction.
- `r128j3_mixmatch.py` — P-set and F-set pass verbatim in one process;
  "mixed combos fail" rows were MY harness bug (refs mismatched to the
  tensor combinations) — the ref-harness-bug-audit law, second
  instance.
- `r128j4_scratch.py` — P-set deterministic + immune to scratch poison
  (0x3C/0xFF/0x00) → no uninitialized-scratch reads.
- `r128j5_xread.py` — out(2x) = 2·out(x) exact; device==host bytes.
- `r128j6_coef.py` — one-hot k=0..15 coefficients EXACT vs true
  LUT[nib]·scale; all eight 128-k blocks clean.
- `r128j7_addr.py` — original tensors, clones, empty_cache-shaken
  heap: identical rel; out[:4] = −836 family = the CPU ref of
  r128j7 itself. The kernel was right; earlier "refs" disagreed.
- `r128j8_order.py` — both op-first and copy-first arms FAIL against
  `torch.stack((lo, hi), dim=1)` — isolating the ref formula.

Root cause: the gate's g16 f64 ref decoded nibbles with
`torch.stack((lo, hi), dim=1)` on `[N, K/2]` tensors → `[N, 2, K/2]`
→ reshape = block-concat `[lo_half | hi_half]`, NOT k-interleaved
nibbles. `dim=2` → `[N, K/2, 2]` → k-ascending interleave, which is
what `nib_vals`/`repack_f4` (and the kernel) actually do. One-line fix
(dim=1 → dim=2) → **GATE-1 PASS 40/40** (cut 7, rig
`/root/build/r128a/r128b_cut7.log`). The kernel needed NO correctness
change; the prior session's r128i8 "1416 XPU f64 divergence" and
r128i4 "uniform-scale 1.316" were the same ref bug.

Law (memory-updated): a mirror test that copies the same buggy code to
both sides is vacuous; audit BOTH the kernel and the ref — wrapper
signatures can structurally eliminate hypotheses; value-coverage
arguments cannot exonerate a harness bug.

## Part 2 — per-step cost ladder (r128k) and the V6 restructure

Ladder `tools/ar/r128k_ladder.cpp`: torch-free SYCL replica of the
exact `jm_launch` geometry (lsz 64, subs = tiles·splits, golden-ratio
swizzle, per-global-sg scratch slot), template variants V0..V10.
Measured on arc-jm7 (`ladder_runs.log`):

| variant | qwen.up 5120×8704 (160 steps) | gemma.o 5376×5376 |
|---|---|---|
| staging A only (V9) | 60.1 ns/step | — |
| staging B only (V10) | 24.4 ns/step | — |
| **V5 = current kernel sequence** | **6445.9 ns/step** | **3221.3 ns/step** |
| **V6 = ping-pong + zero-once** | **59.7 ns/step (108×)** | **78.0 ns/step (41×)** |
| V7 = V6 + alt-target accum | 63.8 ns/step | — |
| V8 = two-tile shared-ja | 91.8 ns/step (45.9/tile) | — |

Attribution law: staging is 1.3% of the current cost; the other 98.7%
is the same-step chain `[stage; bar; load; bar; zero; zero; mad;
read]` — the matrix-engine fragment ops serialize behind the scratch
written THIS step. Loading LAST step's staging behind ONE barrier
(ping-pong slots) collapses the step cost to a ~60–80 ns/step floor
(bandwidth/launch bound).

Stack-fragility law extended (r128e3 lineage): the ladder's
intermediate shapes are not runnable on this stack — V1/V2 (staging
without fragment ops, both staging families in one iteration) =
deterministic DEVICE_LOST; V3 (loads+mad without the zero-applies) =
hang. Only the exact proven fragment sequence (V5) and the reordered
V6 are JIT-stable. A cumulative ladder is impossible on this stack;
only same-harness variant ranking is meaningful.

**Fix implemented** (`jm_gemm_kernel.hpp`, build7): stage-1 hot loop
rewritten to V6 — `c0` zeroed once (read-only addend; `mad` fully
overwrites `c1`), pre-stage step 0, per step `{load A/B; mad(c1, ja,
jb, c0); stage k+16 into the other slot; barrier; swap; apply-read
with the group scale}`. Ordering proof: `load(cur)@i < bar@end_i <
stage(cur)@i+1`, and slot-other's staging@i is visible to its
load@i+1 through the same barrier. r128e2/e3 fragment laws hold
exactly (no A/B apply-writes; one mad per freshly-loaded pair;
D ≠ C). Scratch doubles to 2×768 B per sg (launcher `subs*1536`).
V7/V8 rejected: no gain over V6 in-harness, more aliasing risk.

Projection (calibrated on cut7 qwen.up 2898 ns/step real-kernel):
jm should collapse toward the DRAM floor (~44.6 MB / 589 GB/s ≈
76 μs ≈ 475 ns/step at qwen.up) vs stock oneDNN ~88–110 μs — pending
gate cut 8 correctness + stage-B measurement on build7.

Artifacts: `ladder_runs.log`; sources `tools/ar/r128k_ladder.cpp`,
`r128j*.py`; rig `/root/build/r128a/` (bin + logs + probes);
gate logs `r128b_cut6.log` (fails), `r128b_cut7.log` (GATE1_PASS).

## Part 3 — jm serving-route glue (r128l/r128m) and the bugs they caught

The serving route (`vllm_gaudi/xpu/patches/skinny_m.py`, mode
`ARC_SKINNY_M=jm`) adds `_jm_lane_m_max`/`_jm_pack` (per-layer cache
keyed (lane, weight.data_ptr(), shape); eager repack lands in pre-capture
warmup so XPU-graph capture records only the jm kernel + slice view) and
`_try_jm` (lane detect by weight dtype/scale form; M cap 8 per lane;
bias pad to `[N_pad]`; `out[..., :N]`; exception → per-lane permanent
fallback). Two AST-extraction probes validate it against the wheel
without importing vllm (host Windows is torchless; probes run on rig):

- `r128l_route_pack_test.py` — route repack/pad/cache glue vs the gate's
  builders, byte-parity at 4 shapes (1003/8704/10752/517 odd-N mix) per
  lane + cache-hit identity + scale parity. **PASS**
  (`r128l_route_pack.log`).
- `r128m_route_through.py` — FULL `_try_jm` dispatch on stub layers in
  the exact serving forms (weight `[N,K]` f8 / `[N,K/2]` f4x2; scale
  `[1]`/`[1,N]`/`[N,K/32]`), f64 dequant refs incl. bias, 4 shapes ×
  M∈{1,2,4,8} × 4 lanes = **64/64 cells PASS** on the build6 wheel
  (`r128m_route_through.log`).

Bugs the probes caught before any serving boot (probe value = the
r128j7/r128j8 lesson, but at route level):

1. `_jm_pad_n` padded only the last dim — weights `[N, K]` need dim 0 →
   negative-size zeros crash. Fix: `dim` parameter (weights dim 0,
   scales/bias last dim).
2. `_jm_pad_n` matched `i == dim` with 0-based `i` against `dim=-1` —
   never matched, scales silently became full-width garbage
   (`[32, 2006]`). Fix: normalize `dim = dim if dim >= 0 else t.dim()+dim`.
3. bias contract: jm ops require contiguous fp32 `[N_pad]`; route pads
   `bias` when `bias.numel()==N and bias.dim()==1`.

Probe-side (not product) bugs fixed in the probes themselves: m816
scale stub built `[K/32,N]` instead of the layer form `[N,K/32]`; `x`
built with 2 rows so M=4/8 cells compared wrong shapes (→ 8 rows);
AST extraction missed `ast.AnnAssign` (`_state`) and plain assignments
(`_JM_LANE_M_MAX`). Cut 8 re-runs r128m against the build7 (V6) wheel:
`PYTHONPATH=/tmp/arc7wheel7`.

## Part 4 — build7 cut-8 GATE-1 catastrophe + the r128k2 laws (V6 retraction, V14/build8)

**Cut 8 (build7 wheel = V6 ping-pong) FAILED GATE-1 catastrophically**
(`r128b_cut8.log`): every cell garbage — big shapes exact zeros
(rel≈1.0), odd shapes e+18 junk. Stage-B jm rows were physically
impossible (up to ~8 TB/s vs the 589 GB/s DRAM ceiling — a kernel that
exits without doing the work). The cut-8 stage-B `st`/`sk` rows are
valid and bankable (stock/scalar); every `jm` row is INVALID. This
retracts Part 2's V6 projection: the ladder's 41–108× speedup was real
*timing* of a kernel that was no longer computing the right thing.

`r128k2_ladder.cpp` (numerics ladder: real LCG bf16 activations,
`((lcg>>8)%7-3)*0.25f` self-consistent byte order; LCG u8 codes; host
f64 ref with an e4m3 decoder + decoder selftest 0 mismatches; variants
V5/V6/V11/V12/V13/V14 with per-variant NUMERICS-PASS/FAIL) pinned two
hard facts after MY OWN harness bugs were cleared first (ref bug 1:
split slices last-writer-wins — run numerics at SGF=1/splits=1; ref
bug 2: outputs close-but-wrong AND varying between identical runs —
the tell that sent the search to codegen):

**LAW 1 — alternating fragment-load pointers read ZERO fragments.**
Every variant whose `joint_matrix_load` address VARIES across loop
iterations — V6 (swap), V11 (+barrier-after-load), V12 (+per-step
zeros), V13 (both) — produced exact zeros, while the pointer-fixed V5
slot read real data in the SAME binary. Barriers and zero-applies do
NOT rescue it: the lowering cannot take a varying fragment-load
address. The wheel's cut-8 zeros-at-big-shapes / junk-at-odd-shapes is
exactly this (big shapes = more loop iterations living on the broken
path; odd shapes also hit the garbage epilogue state).

**LAW 2 — standalone-harness numerics are a CODEGEN LOTTERY.**
Identical V5 source flipped exact-pass (maxrel 8.7e-8) ↔ garbage
between ladder binaries differing ONLY by an unrelated template
instantiation or one debug store; runtime sub-group size was 16 in
both (not the cause). r128e3's fragility law extends to AOT binaries:
code layout randomizes fragment-op correctness. Consequences: (a) a
ladder/probe binary can RANK PERF but can NEVER certify numerics for
this family; (b) instrumenting a harness changes its answer; (c) only
the full wheel build + gate is ground truth — and even the proven V5
sequence could flip in a future wheel, so it must always ship beside
any new variant as the control.

**Build8 response (this is the dual-variant wheel):**
`jm_gemm_kernel.hpp` now instantiates BOTH variants, selected per call
by `ARC_JM_VAR` (env, read per call; default 14):

- **V14 (`PP=true`)**: unroll-by-2 so each load SITE is a lexically
  CONSTANT pointer (`sa0/sb0`, `sa1/sb1`; slot pitch 384 halves,
  1536 B/sg scratch). Guarded one-step-ahead staging
  (`(kn < k_hi)` — with K%32==0 and 64-mult slices every step count is
  even, so the odd-tail loop is a dead safety net), per half-step the
  V5-EXACT op sequence `[load; bar; zero c0; zero c1; mad; apply(s)]`,
  final barrier per body. 1.5 barriers/step + per-half-step zeros is
  the price of Law-1-safe addressing; whether that still beats stock
  is exactly what cut-9 stage B measures.
- **V5 (`PP=false`)**: the gate-proven sequence with the rig's exact
  inline scale expression (no shared-lambda refactor — a lottery
  control must not change source text), 768 B/sg of the 1536 B/sg
  scratch. If THIS arm fails on build8 while build6 passed, the
  failure is codegen (Law 2), not source.

Gate cut 9 (`r128b_jm_gate.py`, dual-variant): stage A runs the
40-cell matrix under ARC_JM_VAR∈{5,14} (80 cells); stage B measures
jm5 AND jm14 per (shape × lane × M × SGF), stock/scalar on the default
pass. Analyzer `r128p_cut9_report.py` renders per-variant verdicts
(its cut-8 predecessor's GB/s column had a 1000× units bug — MB/ms is
already GB/s; fixed, verdicts were never affected).

Artifacts: `r128k2_ladder.cpp` (source, this repo `tools/ar/`), rig
`/root/build/r128a/r128k2.bin` + `r128k2_ladder.cpp` + `r128b_cut8.log`;
build7 wheel sha256 `c9de61ec…59f4c` (broken V6, archived for
provenance); build8 launched from `vxk_arc8_inner.sh` (builder
`vxk-arc3-builder`, kernel md5 `fc54923f…d976`).

## Part 5 — the A-direct/SLM decomposition (r128k4..k7, 2026-11-09 evening)

**Correcting Part 4's V14 read first.** SGF=1 re-runs of every surviving
binary (k2/k3/k4/k5) show V5 failing with a few-% signature
(maxrel 0.23–0.30, ~99% cols, values in-ballpark) in ALL of them at
splits=1 — including k2, the "original". There never was a ladder
exact-pass; the remembered 40/40 was the WHEEL gate. Re-running the
stage-A slice TODAY against the surviving wheel `arc7k9.whl`
(sha256 `0df7d098…255c`, cut-7's product; its kernel predates
ARC_JM_VAR so both var passes ran V5): **80/80 cells, rel 2e-3..7.7e-3**
— V5-wheel exactness reproduces. Two harness facts follow:
(a) the ladder's few-% V5 error is a HARNESS-ONLY race (one SG exact,
its neighbor few-% off, varying per binary — a global-scratch
store→load visibility race the wheel build does not share); (b) cut-7's
per-cell rels (4e-4..7.7e-3 vs REL_TOL 3e-2) were genuine exactness,
not racing under a loose bar. Law 2 stands, sharpened: ladder numerics
are WORSE than useless — they manufacture failures; only the wheel gate
certifies, and only the wheel gate's failures are real.

**LAW 3 (occupancy-scaled staging cost).** Ladder per-step cost of a
form depends on resident-WG count; compare forms only at matched SGF.

**The probes (r128k4→k7, `r128k2_ladder.cpp` grew V18–V24):**

| form | ns/step @SGF2048 | µs/call | numerics @splits=1 (qwen.up) |
|---|---|---|---|
| V5 A+B staged global | 5435 | 870 | fail (harness race) |
| V6/V11–V16 ping-pong family | 26–28 | ~4 | **exact zeros — dead, 4th binary** |
| V18 A-direct stride-K probe | 2091 | 334 | garbage by design (bf16-as-f16) |
| V19 A-direct-16 + B staged global | 2136 | 342 | fail (harness race) |
| V20 2-tile shared-ja | 3916 | 627 | race (flashed 1.6e-7 once — race not form) |
| V21 A staged + B-direct f16 | 676 | 108 | near (2.3%, 309/8704 — race) |
| **V22 all-direct f16 (no scratch, no bars)** | **585** | **93.6** | **EXACT 8.7e-7, 0/8704** |
| **V23 A-direct + B staged SLM (V5 ops)** | 3133* | 501* | **EXACT 8.7e-7, 0/8704** |
| **V24 V23 minus post-load bar** | 1511* | 242* | **EXACT 8.7e-7, 0/8704** |

\* at splits=2 occupancy; at splits=1 V23/V24 are 556/550 ns/step
(176 µs) — Law 3. Gemma 5376²: 884 ns/step @splits1 (336 sgs =
under-occupied) but 177 µs wall @splits4 — small-tile shapes WANT
splits. deep-N 3584×18944: 993 ns/step, 222 µs, splits forced 1
(tiles 1184 ≥ floor). M=2 identical to M=1 (A-direct amortizes M).

**Conclusions (perf decomposition, qwen.up, splits-matched):**
1. Fragment loads themselves are CHEAP: V22's two direct loads + mad +
   zeros + apply = 585 ns/step at 2× weight bytes — at stock parity
   (~90–110 µs stock) before bandwidth correction. The whole V5 wall is
   the global-scratch store→barrier→load round trip: A-staging
   ~3.3 µs/step of V5's 5.4, B-staging ~1.55 µs/step of V19's 2.1.
2. **r128e3's "SLM fragment loads = dead kernels" is REFUTED** (AOT
   ladder, local_accessor via handler submission): V23/V24 are EXACT on
   every shape at splits=1 — SLM store→barrier→load is
   hardware-ordered, no race, ever.
3. B-in-SLM costs ~130 ns/step over the all-direct floor (V24 550 vs
   V22 420 @splits1) — the cheapest LAWFUL B path (weights stay f8/f4
   in DRAM; pre-decoded-f16 serving is the 2×-DRAM dead end, V21/V22
   exist only as the floor measurement).
4. Best-form wall-clocks: qwen.up 176 µs, gemma.o 177 µs, deep-N
   222 µs ≈ 1.5–1.7× stock on ladder numbers — and the ladder has
   historically UNDER-estimated wheel stage-1 by ~1.5× (wheel V5 464 µs
   vs ladder 700+). Striking distance: **build9 = the SLM wheel.**

**Build9 design (V24-in-wheel):** A loads direct at stride 16 from a
route-built k-blocked f16 activation buffer `x16[(K/16), 8, 16]`
(rows ≥M zero-padded; `x.to(f16).view(M,K/16,16).permute(1,0,2)` into a
zeroed 8-row buffer — bf16→f16 exact for activations, ~0.3–0.5 µs/call
under graph replay, 80 KB max); B staged per-step into per-sg SLM slots
(1024 halves/WG = 4 sg × B-tile; launcher moves to `q.submit` +
`local_accessor`); no post-load barrier; V5 control stays as
`ARC_JM_VAR=5`. V14 is retained in the wheel only long enough for
cut-9 to formally bury it (4 binaries of zeros predict it).

Artifacts: `r128k2_ladder.cpp` (md5 `b15d6c12…3035`, repo + rig
`/root/build/r128a/`), binaries rig `/tmp/r128k{4,5,6,7}.bin` +
banked `/root/build/r128a/r128k{4,5,6,7}*.log`, k7 = the SLM probe
(862904 B); `r128q_stageA6.py` (stage-A slice) + `arc7k9.whl`
re-validation log `/root/build/r128a/r128q_stageA6_arc7k9.log`.

## Part 6 — build8 post-mortem + build9 (2026-11-09 21:04–21:20 UTC)

**Build8 (arc8, the V5+V14 dual wheel) FAILED — and could never have
built**: `JM_LAUNCH_G` passed the RUNTIME `pp` bool (the per-call
`ARC_JM_VAR` env read) as a template non-type argument —
`error: non-type template argument is not a constant expression`,
2h into the build at TU 1356/1403 (in-container `/tmp/build8.log`,
marker `VXK_ARC8_BUILD_EXIT rc=1 21:04:02Z`). Design bug in the
dual-variant launcher, present from the first arc8 edit. The arc8 wheel
never existed; no arc8 gate data exists. Lesson banked: runtime
variant selection must branch OUTSIDE the template instantiation —
`if (pp) JM_LAUNCH(...,true) else JM_LAUNCH(...,false)`.

**Build9 (arc9) = the SLM wheel, launched 21:15Z in RESUME mode**
(`VXK_ARC9_CLEAN=0` reuses build8's 1356 surviving objects — only the
jm TUs + link rebuild): kernel `jm_gemm_kernel.hpp` (md5
`33330004…b627` on rig `/root/build/r128a/jm_gemm_kernel_b9.hpp` and
builder `/src/.../jm_gemm_kernel.hpp`): PP arm replaced V14 with the
r128k7 V24 SLM form (A-direct from `a16_` + B in per-sg SLM slots, one
barrier, no post-load bar); launcher builds `a16` in-op
(`from_blob → zeros(8,K) → narrow copy_ (bf16→f16, rows≥M zero) →
view/permute/contiguous` = [K/16, 8, 16]) — torch schema and python
route untouched; V5 arm byte-unchanged (lottery control); scratch
shrunk to 768 B/sg (V5 only); `JM_LAUNCH` moved to `q.submit` +
`local_accessor(1024)`; `JM_LAUNCH_G` fixed per the lesson above.
Interface TU pre-flighted clean in the builder (RC=0, one known
`get_pointer()` deprecation warning — the ladder-verified form).
Gate: `r128b_jm_gate.py` unchanged (vars 5/14 = V5/SLM); runbook
`run_cut10.sh`; watcher `watch9.sh` (rig `r128a/`), build log
in-container `/tmp/build9.log`.

## Part 7 — cut-10 VERDICT: lane CLOSED as definitive negative (21:20–21:27 UTC)

Build9 resumed cleanly (16 min, wheel 355 MB, sha256
`54aa858f…d187`, rig-only per the 100 MB law) and the cut-10 gate ran
to completion on arc-jm7 (`run_cut10.sh`, exit 0):

**GATE1_PASS — 80/80 correctness cells for BOTH variants.** The SLM
form is WHEEL-EXACT (v14 rels 1.2e-3…7.8e-3, same band as the V5
control, tol 3e-2) on all 10 shapes × 4 lanes — the r128e3 "SLM
fragment loads = dead kernels" claim is now refuted IN-WHEEL, not just
in the ladder. Every law from Part 5 held in the wheel: A-direct from
the launcher-built `a16` exact, B-in-SLM exact, no post-load barrier
needed.

**Stage B — 0 WIN / 0 TIE / 56 LOSS (both variants), and it is not
close.** Best-of-(var,SGF) jm14 lands at 27–47% of the DRAM floor
while stock rides 60–90%:

| cell class | st (ms) | jm14 best (ms) | ratio | floor% |
|---|---|---|---|---|
| qwen.up 5120×8704 w16 M=1 | 0.0899 | 0.1933 | 0.47× | 39% |
| gemma.up 5376² w16 M=1 | 0.1096 | 0.2075 | 0.53× | 47% |
| qwen.o w16 M=1 (L2-resident) | 0.0283 | 0.1009 | 0.28× | 44% |
| qwen.o g16 M=1 | 0.0224 | 0.0794 | 0.28× | 28% |

Per-lane medians (SGF 2048): jm14 0.49×/0.49×/0.50×/0.28× st for
w16pt/w16pc/m816/g16; jm5 0.09–0.21×. No cell wins, no cell ties at a
ceiling — the stop condition fails outright. **The jm lane is CLOSED.**

**Why (root cause, final):** stock oneDNN M=1 GEMM on B70 is already
near-optimal for these shapes — at qwen.up it runs 0.0899 ms against a
0.076 ms fp8-bytes DRAM floor (84%). A joint_matrix skinny kernel must
pay a per-k-step B staging cost (SLM store→barrier→load, the ONLY
exact lawful form — global scratch races/costs more, cross-iteration
staging reads zeros, pre-decoded-f16 B doubles weight bytes) on top of
the same DRAM traffic, capping it at ~2× stock. The decomposition is
closed: fragment loads are cheap (V22), A-direct is free (V18/19),
SLM staging is exact but not free enough (V24), and the wheel confirms
the ladder's best-form numbers almost exactly (wheel jm14 193 µs vs
ladder V24 176 µs at qwen.up splits-matched — **the earlier "wheel runs
1.5× faster than ladder" calibration was V5-specific** (V5's
global-scratch round trip was worse in the ladder harness than in the
wheel), not a general factor; ladder V24 ≈ wheel SLM within 10%).

**Consequences:**
1. No cert23 from this lane; the certified image remains **cert22**
   (arc6 wheel). No default flips — the working-tree jm mode in
   `skinny_m.py`/`envs.py` is REVERTED to cert22 state; the validated
   route file is banked at
   `external_patches/vllm-xpu-kernels/arc7_jm_gemm/skinny_m_jm_route.py`
   (md5 `c7021e7a…5705` = rig `r128a/skinny_m_route.py`, the exact
   file cut-10 exercised).
2. The arc9 wheel (arc7 lineage + b9 SLM kernel) is banked as
   negative-evidence artifact, never shipped; the not-built arc9test
   kit (`tools/ar/arc9test.Dockerfile`, `stage_arc9test.sh`) is banked
   for provenance.
3. The lane's positive yield: the four staging laws (Part 5) + the
   wheel-vs-ladder calibration correction + stock-is-near-floor
   measurement — these govern any future custom-GEMM work (e.g. a
   register-DPAS lane would face the same B-staging bill).
4. P6.34 closes: jm joins P6.33 skinny-decode as a banked OFF lane;
   the certified skinny stack stays arc6 scalar + FA2 routes
   (P6.28/6.29) + online MXFP4/MXFP8 (P6.30/6.31).

Artifacts (all md5-verified rig==repo): `r128b_cut10.log`
(`dc3d5262…b2ab`), `r128b_cut10_rows.json` (`d2d6f653…8eab`),
`r128p_cut9_report.txt` (`940204dd…17d7`), `arc9_wheel.sha256`
(`06b344f6…cbaa`; wheel at rig `/root/build/r128a/`, 354920052 B);
runbook/watcher/inner at `tools/ar/{run_cut10.sh,watch9.sh,vxk_arc9_inner.sh}`
(`42df6176…`, `5a1136db…`, `bae2cdce…`).


## Part 8 — vehicle (a) register-DPAS: instruction unreachable, lane CLOSED (22:00–22:15 UTC)

The cut-10 write-up left one theoretical escape: a **register-fed dpas**
(vehicle (a)) — decode f8/f4 weights straight into registers, feed
`dpas` with register operands, never touch memory between decode and
MMA. That kills vehicle (b)'s staging bill by construction. Probe
`tools/ar/r129a_regdpas.cpp` (cutlass-sycl cute ops, Marlin repack
`[nt][kc][t16][16B/8B]`, branchless packed decoders, one SG per
n-tile, M=8-pad/1-row-store, 3 warmup + 50 profiled launches).

**Verdict: the register-fed instruction does not exist on this stack.**
Both lowering forms of `SubgroupMatrixMultiplyAccumulate` kill the
thread at the first fma — every configuration tried:

| form | compile | behavior |
|---|---|---|
| raw-asm `XE_DPAS_TT` (`mma_xe.hpp`) | AOT bmg | thread dies at kc=0 fma; kernel "completes" ~4.8 µs, out all zeros, maxrel 1.00 |
| `XE_8x16x16_F32F16F16F32_TT` → `__spirv_SubgroupMatrixMultiplyAccumulateINTEL` (`mma_xe_legacy.hpp` → `_spirv.hpp`, oneAPI ≥2025.2 route) | AOT bmg | identical: pre-fma DBG prints fire, post-fma never |
| same + `[[sycl::reqd_sub_group_size(16)]]` | AOT bmg | worse: process **segfaults (exit 139)** |
| same | JIT (spir64, no AOT flags) | **segfault mid-run** (log ends at a gemma DBG line) |

Controls: the same flags are production's (vxk `CMakeLists.txt:291`
spirv-ext list incl. `subgroup_matrix_multiply_accumulate`); the
non-mma control build (`-DR129_NOMMA`, loads kept alive) completes
fine, so loads/geometry are not the killer; and `joint_matrix`
(`OpJointMatrixMadINTEL` — a *different* instruction) is proven
wheel-exact on this exact stack/container at cut-10. XE2/bmg +
oneAPI 2026.0 simply has no working lowering for the legacy
subgroup-mma instruction; Intel's own guidance now points Xe2+ at
joint_matrix, which has **no register-input API** (fragments are
loaded from memory only) — so "register-DPAS + inline dequant" is
unimplementable, not merely slow. The perf question (≥516 GB/s?) is
moot.

**Consequences:**
1. Vehicle (a) CLOSED, completing the pair: (a) register-fed dpas —
   unreachable; (b) joint_matrix staging — reachable and 2× stock
   (cut-10). **Custom skinny-GEMM is CLOSED on all vehicles.** The
   arc7-README law stands with a harder edge: any future custom-GEMM
   lane must beat stock's 60–90%-of-floor bar *through joint_matrix
   only* — i.e. pay the measured ~130 ns/step SLM B-staging — before
   it earns a wheel build.
2. CORRECTION (initially posted as "next session's probe", withdrawn
   same evening): **stock w8a8 vs w16 at M=1 was already measured** —
   r40 (`results/2026-10-29-p4-r40-forward-gemm-capture/`) captured
   the stock block path (input quant + `_xpu_C.fp8_gemm` W8A8) at a
   FIXED ~0.33 ms per GEMM at decode Ms (48–268 GB/s) on every layer
   shape vs W8A16's 0.064–0.167 ms (247–579 GBs) — 3–8× slower at
   decode Ms, and the champion's own W8A8 probe was the same slow
   class (routing, not a torch delta). The winning route (fp16
   activations → `fp8_gemm_w8a16`) has been in our certified stack
   since the p44 patch (`vllm_gaudi/xpu/patches/fp8_block_w8a16.py`).
   Nothing left to probe there; the P6.26 "w8a8 refusal" memory was
   about the W4A8 (MXFP4-weight) lane, a different question.
3. `mqa_logits_xe2.cpp` (production vxk, raw-asm `XE_DPAS_TT`) is
   presumptively dead on bmg for the same reason — never exercised by
   our serving (mqa route unused at cert22). Noted, not actionable.

Artifacts (md5 rig==repo): `r129a_run.log` (`49348c54…43db`, AOT
intrinsic full run), `r129a_jit.log` (`b234d472…ef07`, JIT segfault),
`r129a_sg16_aot.log` (`f0b09242…4b906`, exit-139 run); source
`tools/ar/r129a_regdpas.cpp` (`d3225889…e1b` pre-verdict-header);
binary path rig `/root/build/r129a/`, builder `/tmp/`. Source md5s:
executed variant `d3225889a55eef69916b40e49553ae1b` (rig
`/root/build/r129a/r129a_regdpas.cpp`), repo final `2476ae6e30dc2e9d
aec03c1cfef95d7d` (= rig `/root/build/r129a/r129a_final.cpp`, adds the
verdict header). Compile:

```
source /opt/intel/oneapi/setvars.sh --force   # in vxk-arc3-builder
icpx -fsycl -O3 -DNDEBUG -DCUTLASS_ENABLE_SYCL -DR129_DEBUG   -fsycl-targets=spir64_gen   -Xsycl-target-backend=spir64_gen "-device bmg,bmg-g31-a0"   -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"   -I/src/vllm-xpu-kernels/.deps/cutlass-sycl-src/include   /tmp/r129a_regdpas.cpp -o /tmp/r129a.bin
```

(JIT arm: drop the `-fsycl-targets`/backend/device/spirv-translator
flag groups.)
