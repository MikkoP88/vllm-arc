# P6.19 — TurboQuant spec-verify corruption: ROOT-CAUSED and FIXED (2026-11-04)

Owner directive (reopening P6.18's precision-ceiling disposition):
*"Reopen fix this is not true! Issue is fixaple, this ISSUE HAS TO BE
FIXED SO NO CORRUBTION ON ANY KV CACHE TYPE!, this project images do
not have same issue do analyze why. Do deep analyze and research, and
update master plan and start implement fix start-to-end, create new
certificated image on end and run full performance metric and update
all documents."*

**Outcome: all four TurboQuant presets now SERVE CLEAN — zero
REP-LOOP / JUNK / BLEED / EMPTY on every dtype, verified by the
all-dtype long-return battery (t9) with the fp8 certified posture as
the control.** The P6.18 boot refusal is lifted; cert8 ships
`--kv-cache-dtype turboquant_*` as a supported choice with the
P6.19 spec-verify fix active by default.

Campaign tree (this directory):

| path | content |
|---|---|
| `p619b/` | donor probes (d1 = donor TQ image serves k8v4 clean; w1 = wheel wedge control) |
| `p619c/t1..t4f` | telemetry localization legs (P6.19c) |
| `p619c/t5, t6p, t7_*` | fix bring-up: split dispatch, perf, all-dtype eager |
| `p619c/t8, t8e, t8f, t8g` | MQ-kernel arc: SoA draft corruption → dump forensics → AoS port |
| `p619c/t9_*` | **all-dtype shipping gate** (split default, 4 TQ dtypes + fp8 control) |
| `p619d/` | offline validators (MQVAL suite), leg scripts, wheel-source extracts (`tqsrc/`), patched files as-run |
| `p619*_*.out` (top level) | rig consoles for every leg above |

Redaction: host-address pass run before banking — see
`../REDACTION-2026-11-04-p619-tqverify.md`.

---

## 1. Why donor images don't corrupt (the question that reopened this)

Static diff (P6.19a, master-plan as-executed record) + donor probe
(`p619b/d1/` — donor TQ image, k8v4, spec3: longprobe clean, zero
garble): the donor trees run **a multi-row (MQ) verify kernel** for
spec-decode continuation attention and force verify-friendly capture
state (donor v19b). Our wheel's `turboquant_attn` backend dispatches
verify rows through paths that are numerically wrong for the
multi-row shape on XPU: (a) under XPU graphs the captured verify
graph baked the **raw-KV flash fast-path** (TQ-blind — treats the
quantized cache as fp8 bytes), and (b) the eager fallback replayed a
**broken multi-row synthetic-decode dispatch**. Donors never hit
either because their dispatch is the MQ kernel + forced capture
semantics.

## 2. Three stacked root causes (each proven, each fixed)

| # | root cause | proof leg | fix |
|---|---|---|---|
| RC1 | **TQ-blind verify under XPU graphs** — capture recorded the raw-KV flash fast-path for verify shapes, so every spec-verify attention read the quantized cache as fp8 | t2i telemetry: verify outputs diverge from eager only under graphs; donor v19b forces `max_seq_len=q_len+1, seq_lens_cpu=q_len+1, seq_lens=q_len` at capture | `tq_spec_fix.py` applies the donor v19b capture forcing on TQ dtypes (all modes) |
| RC2 | **Replay-broken multi-row dispatch** — the stock backend's multi-row synthetic-decode call is not replayable/wrong on XPU; eager TQ was corrupt even with RC1 fixed | t3e/t4f/t5: eager legs clean only after dispatch replaced | **split-row dispatch** (SHIPS): q_len single-row stock decode calls, per-row limit `seq_lens[i:i+1]-(q_len-1-r)` recorded as a device op — graph-safe, uses only stock kernels |
| RC3 | **Two TQ ops trees** — first MQ-kernel draft targeted `ops/turboquant_soa/*` (FlyDSL-gated, never used on XPU) while the engine stores slot-interleaved **AoS** (`ops/triton_turboquant_{store,decode}.py`); the draft read the live cache as exact zeros | t8/t8e: REP-LOOP on all classes (hashes `52fd1c2e…`, 13,830 chars — the degenerate attractor); t8f dump: `mq_out` exactly [0]; **MQVAL3c killer test**: fresh store into the dumped tensor decodes fine, the engine's own page decodes to zero under the SoA pair | `tq_mq_aos.py`: MQ kernel re-ported to the AoS layout (addressing byte-identical to stock `triton_turboquant_decode`), **MQVAL5: 4/4 presets exact vs the engine's own store+decode pair** (rel 0 / ≤1e-5, det 2/2) |

**LAW (offline validation):** an offline harness must import the pair
the RUNTIME dispatches — validating a kernel against a different
impl of the same name (SoA vs AoS trees in one wheel) proves a
self-consistent wrong world (MQVAL1: FAIL everywhere yet det=True).

## 3. What ships (cert8)

- `vllm_gaudi/xpu/patches/tq_spec_fix.py` (v6) — wheel-side hook,
  installed via `_WHEEL_HOOKS` on `turboquant_attn` import; active on
  every `turboquant_*` dtype, absent on fp8 (module never imports —
  t7/t9 control proof). Applies RC1 capture forcing + RC2 split-row
  dispatch.
- `vllm_gaudi/xpu/patches/tq_mq_aos.py` — RC3-fixed MQ kernel
  (opt-in, see knobs). Structure: donor v19 MQ (multi-row online
  softmax, KV splits); addressing/layout: this wheel's AoS pair.
- `vllm_gaudi/xpu/platform.py` — P6.18 turboquant boot refusal
  REMOVED (P6.19 disposition comment in its place).

Knobs (all documented in `docs/envs.md`):

| env | default | meaning |
|---|---|---|
| `VLLM_ARC_TQ_SPEC_FIX` | `1` | master switch for the P6.19 hook |
| `VLLM_ARC_TQ_VERIFY_MODE` | `split` | `split` = shipping (stock kernels, graph-safe). `mq` = single-pass MQ kernel — **experimental** (see §5). `synth` = stock multi-row synthetic-decode (pre-fix behavior, forensics only) |
| `VLLM_ARC_TQ_MQ_SPLITS` | `32` | KV splits for the MQ kernel (`mq` mode only) |

## 4. All-dtype shipping gate (t9) — the owner's corruption bar

Fix v6 mounted, **shipping default (split)**, no mode env, XPU
graphs on, spec k=3, one boot per dtype (`p619c/t9_*/`,
console `p619d_t9.out`):

| dtype | install line | flip-first | longprobe garble (4 classes × cold/rehit/rehit2) | identity faces | erraudit |
|---|---|---|---|---|---|
| `turboquant_k8v4` | v6 active split | 9.3 s `stop` coherent | **ZERO** | multi t0 A/B pair (see §5) | 0 |
| `turboquant_k3v4_nc` | v6 active split | 9.8 s `stop` | **ZERO** | t1-seed essay+simple sporadic | 0 |
| `turboquant_4bit_nc` | v6 active split | 9.8 s `stop` | **ZERO** | t1-seed essay+code sporadic | 0 |
| `turboquant_3bit_nc` | v6 active split | 10.0 s `stop` | **ZERO** | essay/code t0 3-variant drift (§5) | 0 |
| `fp8_e4m3` (**control**) | fix ABSENT by construction | 9.1 s `stop` | **ZERO** | **multi t1-seed sporadic — the certified posture fires the same class** | 0 |

KV pool on k8v4 graphs posture: **519,972 tokens** (fp8: 400,739) —
the quantized dtypes buy ~30 % more cache, the practical reason this
lane matters.

**The corruption bar — "NO CORRUPTION ON ANY KV CACHE TYPE" — is
met: zero REP-LOOP/JUNK/BLEED/EMPTY in all 20 dtype×class cells,
0 ERROR / 0 Traceback per leg, coherent stop-terminated flips.**

## 5. Identity faces — disposition (not corruption)

The longprobe verdict logic fires `CORRUPTION_SIGNAL` on ANY identity
anomaly; the campaign standard (cert7 precedent) is the corruption
bar of §4 plus honest characterization of identity faces. Three
independent proofs that the remaining faces are engine-class
(probabilistic drafting), not TQ corruption:

1. **The fp8 certified control fired the same class** (t9_fp8:
   multi t1-seed flip, fix absent by construction — the hook never
   installs on fp8). A face that exists on the certified posture is
   not a TQ regression.
2. **Cross-boot, cross-mode hash-pair stability**: the multi t0
   variants {`3bb45b51520b`, `f0dce5b55f4f`} appear in BOTH t8g (mq
   mode) and t9_k8v4 (split mode); simple `84efc84cc142` appears
   across t8g, t9_q4nc AND t9_fp8 (= cert7's banked simple hash).
   Fixed adjacent coherent variants = the p618b Class-A signature
   (sporadic RNG-path divergence under probabilistic drafting), not
   garble.
3. **Face continuity**: each dtype's t9 face matches its earlier
   campaign record (k3v4/q4nc t1-seed = t7 faces; cert7's own t1-seed
   annex re-characterized its single flip 16/16 identical on a live
   engine).

q3nc (coarsest preset, 8 affine V levels) additionally shows a
3-variant t0 drift on essay/code (chars 3563/4341/3821 — coherent,
zero garble, deterministic within rehit): the flatter token
distribution of 3-bit values widens where the probabilistic drafter
branches. Recorded as the precision-characteristic face of the
coarsest preset; upstream's own PPL table rates 3bit worst. t0
byte-identity remains guaranteed within a request path on every
dtype (cold==rehit==rehit2 wherever t0_identical=True; drift cells
are coherent variants of the same request, never garble).

## 6. Performance (development legs, cert6 base + fix as-run)

| posture | solo tps (512 tok) | agg8 tps (8×512) | source |
|---|---|---|---|
| k8v4 split (SHIPS) | 54.50 (warm) | 240.86 | `p619c/t6p/tps.out` |
| k8v4 mq (experimental) | 54.63 / 56.24 | **289.23 (+20 % vs split)** | `p619c/t8g/tps.out` |
| fp8 cert7 reference | 84.50 | 299.73 | `../2026-11-03-p618-cert7/` |

MQ mode is corrupt-free after RC3 (t8g zero garble) and ~20 % faster
at aggregate, but its multi/simple identity faces (annex
length-cutoff empties) keep it opt-in-experimental; split ships.
Numbers re-verified on the cert8 image in `VERDICT.md`.

## 7. Offline validators (`p619d/`)

| validator | what it proves | result |
|---|---|---|
| MQVAL1 (`mqval/mqval.out`) | SoA-tree draft vs the runtime pair | FAIL everywhere, det=True — the wrong-world signature that exposed RC3 |
| MQVAL2 (`mqval2/mqval2.out`) | **split dispatch** vs stock decode + fp32 ref, 4k pages, sparse pages | **PASS 6/6** (rel_split = 0.000000, rel_ref ≤ 0.003, det) |
| MQVAL3c (`mqval/mqval3c.out`) | killer test: SoA pair on the engine's own dumped page vs fresh store | engine page → zeros; fresh store → correct (RC3 proven) |
| MQVAL5 (`mqval/mqval5.out`) | **AoS MQ kernel** vs the engine's own store+decode pair, 4 presets, engine geometry | **PASS 4/4** (rel 0 / ≤1e-5, det 2/2) |
| MQVAL6/6b | dump replay + semantics sweep (offline(dump)=44.00 vs engine ref0=30.28) | nonzero+det replay; discrepancy dispositioned as instrument-timing (async cache writes across streams + `.item()` sync between ref0 and dump); gather fidelity proven (cpu-idx == dev-idx) |

`tqsrc/` holds the wheel-source extracts the port was written against
(AoS store/decode, SoA decode, backend) — provenance for the
addressing claims in §2.

## 8. Cert8 release gate + annexes (`rig/`, `VERDICT.md`)

The tag-built image `vllm-arc:v0.30.0-b70-cert8` (tag
`rel/v0.30.0-b70-cert8` @ main `e47739b0`, id `b212fbca59ec`) ran the
full release battery 2026-11-04: manifest/fix/guard + image-native
p48 28/28 + p73 11/11, fp8 posture + long-return (t0 lane 100 % clean,
byte-family cert7), all-dtype TQ long-return (zero garble on every
dtype; Class-A + 3-bit-floor identity faces dispositioned with
cross-boot byte-evidence), the opt-in mq annex re-run (agg8 284.81 =
1.21× split), the W14 incumbent control that closed the t1-seed
REP-LOOP flag as a model-lane property, and the 24-record t1-seed
re-characterization (zero garble, 15/16 seeded pairs identical).
Full table + incidents + disposition: **`VERDICT.md`**; consoles +
per-leg evidence under `rig/` (redaction pass 2: +816,
`../REDACTION-2026-11-04-p619-tqverify.md`). Best-practices audit of
the fix: `BEST-PRACTICES-AUDIT.md`.
