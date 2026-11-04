# BEST-PRACTICES-AUDIT — P6.19 TQ spec-verify fix (cert8, 2026-11-04)

Owner directive (P6.19e, 2026-11-04): *"do deep research and analyze all
fix and changes are included in line best practise project patches."*
Same standard as the P6.18c audit (`../2026-11-03-p618-cert7/` tree):
every change patterned against (a) the wheel's own code — the upstream
truth this fork runs, (b) donor-admitted idioms (scope law §3: line C
llm-scaler v19/v19b/v21, line D iexperimental), (c) the project's
established patch-layer laws. Method: source-level comparison against
the wheel extracts banked in `p619d/tqsrc/` (`turboquant_attn.py`,
`triton_turboquant_{store,decode}.py`, `turboquant_soa/*`, plus
`flash_attn.py` extracted for this audit) and the shipped files
`vllm_gaudi/xpu/patches/tq_spec_fix.py` (v6) and
`vllm_gaudi/xpu/patches/tq_mq_aos.py`.

## Change 1 — `vllm_gaudi/xpu/platform.py`: refusal lifted, hook entry added

- The P6.18 guard (`check_and_update_config` raising on every
  `turboquant_*` dtype) was **our addition**, not upstream's. Removing
  it restores the upstream passthrough: the wheel's platform layer
  accepts these dtypes (the whole TQ lane exists upstream), and dtype
  validation remains the wheel's own config machinery. A local
  platform narrowing the accepted-dtype set is an upstream-sanctioned
  pattern only when the platform genuinely cannot serve the dtype —
  P6.19 proved it can (the corruption was spec-verify dispatch, not
  the dtype).
- The new `_WHEEL_HOOKS` entry
  `("vllm.v1.attention.backends.turboquant_attn", "tq_spec_fix")`
  follows the exact mechanism every existing patch uses (p41, p41b,
  p41c, r43 rows in the same tuple): target module + patches
  submodule, installed by the meta-path hook, identity proven by the
  one-time install line in serve logs. No new mechanism was invented.
- **In line.** One comment disposition replaces the raise; one tuple
  row. No behavioral delta on any non-TQ path (fp8 boots never import
  `turboquant_attn` — t7/t9 control proof).

## Change 2 — `tq_spec_fix.py` (v6), the wheel-side hook

### 2a. Install pattern

- Marker attribute on the wrapped method (`_forward._arc_p619_tq_spec_fix`)
  making install idempotent; loud one-time `logger.warning` install
  line; graceful absent-symbol no-op (`Builder/Impl is None → return`).
  This is the established pattern of every `_WHEEL_HOOKS` patch
  (visibility law: every patch logs its install; install is
  idempotent; a patch must never hard-crash a boot it cannot apply).
- Env-gated with default-active + `0` passthrough
  (`VLLM_ARC_TQ_SPEC_FIX`), matching the knob-surface conventions
  (`docs/envs.md` r45g truth table: default column, `=0` disable
  column, evidence column). **In line.**

### 2b. Verify-capture forcing — enforces the wheel's OWN capture contract

The wheel's `TurboQuantMetadataBuilder.build_for_cudagraph_capture`
(`tqsrc/turboquant_attn.py:264-271`) states the contract in its own
comment:

```python
attn_metadata = self.build(0, common_attn_metadata)
# Set seq_lens to 1 so CUDA graph capture is fast
# (real seq_lens are filled at replay time).
attn_metadata.seq_lens.fill_(1)
```

*"Real seq_lens are filled at replay time"* — i.e. the captured branch
must read the dynamic buffers, never bake capture-time lengths. For
single-token decode captures (`max_query_len == 1`) the stock builder
satisfies this. For **multi-token spec-verify captures** it does not:
with `seq_lens == 1 …` the derived `max_seq_len == max_query_len ==
q_len`, and `_prefill_attention`'s raw-KV flash fast path condition
(`max_query_len == max_seq_len`, `tqsrc/turboquant_attn.py`) is TRUE
at capture — the fast path attends only in-batch tokens and the
recorded graph never reads the KV cache. The wheel's FA2 backend
shows the same upstream awareness in the other direction
(`tqsrc/flash_attn.py:545-560`: capture shapes are enumerated per
query-length class precisely because a graph is only valid for the
shape class it was captured for).

The patch's forcing — `max_seq_len=q_len+1`, `seq_lens_cpu=q_len+1`,
`seq_lens=q_len` for `max_query_len > 1` captures only — restores the
builder's own contract by the minimal mechanism: it makes the
capture take the continuation branch, whose kernels read the dynamic
`seq_lens`/`block_table` buffers refreshed at replay. The idiom is
donor-admitted (llm-scaler v19b hit the identical defect and shipped
this forcing — line C provenance). Single-token captures are
untouched (stock `fill_(1)` stays). **In line: the change enforces
the upstream comment's stated law using a donor-proven idiom, with
no stock-path modification when spec is off.**

### 2c. Split-row continuation dispatch (the shipping fix)

- Uses **only the wheel's own stock decode kernel**, in the wheel's
  own single-row call shape (`B=1`, normal-stride one-row block
  table) — the exact call the stock decode path makes per request
  (`tqsrc/turboquant_attn.py:1205` decode dispatch). No new kernel on
  the shipping path.
- Per-row causal limits `seq_lens[i:i+1] - (q_len-1-r)` are computed
  as **recorded tensor ops on the LIVE `seq_lens` buffer** — never
  host-side values. This is the graph-capture law the wheel itself
  follows everywhere ("real seq_lens are filled at replay time";
  replay-refreshed device buffers, no host data-dependence inside a
  captured region). The values equal the stock synthetic-decode
  tensor's values (`cached+1 … cached+q`) — identical semantics,
  replay-safe construction.
- Replaces the stock multi-row synthetic-decode dispatch (stride-0
  expanded block table + synthesized seq_lens + `B=q_len` grid,
  `tqsrc/turboquant_attn.py:871`), which is numerically identical in
  eager (t3e) but computes wrong under graph REPLAY (t4f root-cause
  leg). The stock path is retained verbatim behind
  `VLLM_ARC_TQ_VERIFY_MODE=synth` for differential diagnosis —
  default-off, A/B-evidenced (law 6).
- The per-row-limit construction mirrors the donor's synthetic
  fallback (llm-scaler v21 lines 1394-1404 — line C provenance,
  recorded in the file docstring). **In line.**

### 2d. Bounded instrument

`_log_shape` is bounded (one line per shape class, hard cap
`_LOUD_CLASSES=12`, never raises) — matching the project
telemetry-off posture law (P6.6: serving logs stay bounded; the
instrument was the t1i/t2i campaign tool and stays for field
diagnosis). **In line.**

## Change 3 — `tq_mq_aos.py` (opt-in `mq` mode, experimental)

- **Structure provenance:** donor llm-scaler v19 MQ kernel (line C):
  multi-row online-softmax over KV splits, two-stage combine, fixed
  grid with kernel-side split ranges (the graph-replay-safe shape).
- **Addressing provenance:** byte-level the wheel's own
  `ops.triton_turboquant_decode` slot-interleaved layout
  (`tqsrc/triton_turboquant_decode.py:148-152` slot-base arithmetic,
  K fp8/MSE paths, V nibble unpack, inline fp16 scale/zero) — the
  engine's DEFAULT XPU path (the first draft's SoA-tree mistake and
  its proof are banked: t8/t8e/t8f + MQVAL3c; the two-trees law).
- **Validation law (new, this campaign):** an offline harness must
  import the pair the RUNTIME dispatches — MQVAL5 validates the
  kernel against the engine's own store+decode pair at engine
  geometry: 4/4 presets exact (rel 0 / ≤1e-5, det 2/2).
- **Posture:** default-OFF (`split` ships), documented experimental
  with its serve-leg evidence (t8g zero-garble, agg8 +20 %) and its
  known identity-face caveat. A perf-mode knob that can never break
  serving (absent module → loud fallback to split). **In line.**

## Knobs (all three)

`VLLM_ARC_TQ_SPEC_FIX`, `VLLM_ARC_TQ_VERIFY_MODE`,
`VLLM_ARC_TQ_MQ_SPLITS` follow the `VLLM_ARC_*` surface conventions
(default + disable column + evidence link in `docs/envs.md`); each
carries leg evidence (t4f/t5/t6p/t8g/t9/MQVAL5 — law 6: no
unevidenced customization). **In line.**

## Divergences from upstream — recorded, justified

1. **We hook the wheel instead of rebuilding it.** The fork law
   (plan §13 / CLAUDE.local.md): components enter as wheel-side
   patch files with provenance; the ground-truth wheels are pinned
   (md5-guarded) and never rebuilt per-release. This is the
   project's own established layer — every prior fix (p41 family,
   r43 kit, P6.14 store hunk) ships the same way.
2. **The MQ kernel is new code** (no upstream counterpart on the AoS
   tree). Its admission is scope-law-clean (donor v19 structure,
   wheel's own addressing) and it is opt-in-experimental; the
   certified path runs stock kernels only.
3. **The instrument lines (`P619I …`)** are campaign telemetry kept
   bounded per the telemetry law; they add no serving-path work
   beyond a dict lookup per shape class.

## Post-audit correction

None required for P6.19 at audit time — unlike P6.18c (whose
CLI-flag-vs-pydantic-field drift was caught post-audit), every
P6.19 code path was exercised on the live engine before this audit
(t3e/t4f/t5/t6p/t8*/t9 legs + MQVAL suite re-run and banked
2026-11-04). The one near-miss recorded as a campaign LAW instead:
the first MQ draft validated against the wrong ops tree (SoA) —
offline validation must import the pair the runtime dispatches
(master plan, P6.19 as-executed record).

## Verdict

Every P6.19 change is in line with (1) the wheel's own documented
contracts — the capture forcing literally enforces the stock
builder's "real seq_lens are filled at replay time" comment; (2)
donor-admitted idioms with recorded provenance (v19b forcing, v21
limit construction, v19 MQ structure); (3) the project's patch-layer
laws (hook mechanism, install visibility, idempotence, bounded
telemetry, evidenced knobs, default-safe posture). The shipping
default path runs stock kernels exclusively; the only new kernel is
opt-in and validated against the engine's own pair.
