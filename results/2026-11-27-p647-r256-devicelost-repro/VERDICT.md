# r256 — the DEVICE_LOST minimal repro (P6.47, run2)

**VERDICT: NO CRASH REPRODUCED under the tested conditions — and the run
EXPOSED the repro gap: the r255 crash request rode the chat template's
OFF branch (`chat_template_kwargs {"enable_thinking": False}`), making
it byte-different from every request this leg sent (no kwarg = the ON
branch, +47 template tokens). The negative result narrows the trigger
to the OFF-branch prompt stream and/or the suite-state allocator
layout; r257 replays the exact r255 sequence.**

## What ran

`tools/ar/r256_devicelost_repro.sh` (driver, rig-side
`/root/build/r256_devicelost_repro.sh`, md5 `3cf3331f2c546e9de20398e63af35c20`)
+ `tools/ar/r256_devlost_probe.py` (probe v3, rig-side
`/root/build/p647/r256_probe.py`, md5 `d7aca2345c8a97a6307ad16ee859c188`)
on `vllm-arc:v0.30.0-b70-cert29` (cert29_fork_sha
`16effb497cd36cb25df7c19cb6f1938d2d45cdc5`). Launched 2026-11-27T01:37Z
(run1, killed after two probe defects) and ~02:03Z (run2, full leg);
`P647_R256_REPRO_COMPLETE -- R256_SUMMARY repro=0 ladder=0 control=0`
at ~02:11Z. One gemma m8 boot per phase (the r255 m8g crash posture:
`GEMMA_NAT` + `ARC_KVDTYPE=mxfp8` + `ARC_MXKV_NATIVE=1`), a 2 s-cadence
devcoredump watcher (`/sys/class/drm/*/device/devcoredump/data` +
dmesg the moment a GPU reset lands — the r255 dumps were eaten by the
retention window), and the `/tokenize`-oracle probe.

Posture evidence (all three boots): `mx_backend_lines=1`,
`native_lines=54`, `pc2_windowed_impl=100`, `lmcache_lines=0`,
`tracebacks=0`, KV pool 54,426 tok @1.33x — byte-identical to the r255
m8g leg.

## The two probe defects (run1, banked in `run1_probe_defects/`)

1. **The /models remap**: in-container `/models/` holds ONLY `target` —
   the serve script remaps ARC_MODEL's host dir to that fixed internal
   mount point; the host dir name (`/models/gemma4-31b`) never appears
   in-container. A local `AutoTokenizer.from_pretrained` on the host
   path died with `HFValidationError`.
2. **No embedded chat template**: even at the right path, the model dir
   ships NO chat template (`tokenizer.chat_template is not set` — the
   engine gets its template from the serve script's
   `--chat-template` flag). A local tokenizer can never reproduce the
   engine's prompt stream.

**Fix**: the count oracle is now the server's own `POST /tokenize`
with the messages payload — the serve's template resolution,
byte-for-byte the chat-completion preprocessing. Runtime
self-calibration: `CALIB tokenize=93 engine_prompt_tokens=93 MATCH` on
every boot (the oracle count == a live completion's
`usage.prompt_tokens`).

## Results (run2 — all under the ON-branch template, no kwarg)

| phase | requests | result |
| --- | --- | --- |
| A repro | the exact r255 LENGTH (3,446 tok): multi text truncated to 3446, fresh -> A/A resend | CLEAN (fresh 2.9 s, resend 1.7 s, alive) |
| B bisect | tails 24/48/63/64/65/96/118 with the 3,328-tok prefix held constant; every rung a real fresh+resend pair (engine counts recorded: 3352/3376/3391/3392/3393/3424/3446) | ALL CLEAN — no rung died, `LADDER_SURVIVED_ALL_RUNGS` |
| C control | English-only content truncated to 3,446 tok, fresh -> A/A resend | CLEAN (content-geometry without the multilingual bytes does not crash the ON-branch stream either) |

`erraudit` per phase: `DEVICE_LOST=0 TRACEBACK=0`. No devcoredumps
landed (0 banked). The banked dmesg ring contains NO resets in the
run2 window (02:03-02:11 on 11-27).

## THE DISCOVERY — the r255 request was never byte-tested by this leg

- The engine's non-default args confirm **the gemma serves run the
  QWEN template**: `chat_template: '/root/chat_template_qwen38_high.jinja'`
  (GEMMA_NAT sets no ARC_TEMPLATE -> the serve script's
  `__qwen_default__` case mounts it). All gemma legs since the
  template knob exists served this way — the crash study and r255
  share the template, so it is not a confounder; but it IS the reason
  the r255 probe's kwarg matters.
- The r255 longprobe sends every face with
  `chat_template_kwargs {"enable_thinking": False}` — the template's
  OFF branch. This leg's probes sent no kwarg — the ON branch.
- Measured: the UNCUT multi text is **3,493 tok ON-branch** vs the
  r255 crash prompt's **3,446 tok OFF-branch** — a constant 47-token
  template delta. Phase A landed 3,446 by truncating the text under
  the ON branch instead of switching branches: **same length, entirely
  different bytes** (a 47-token head shift — the tail-prefill window
  at positions 3328..3445 held different content than r255's).
- The run2 negative is therefore honest but does NOT clear the defect:
  the r255 byte stream was never tested. The trigger hypotheses are
  now (a) the OFF-branch prompt bytes (the 118-token tail content the
  window twin actually reads), and/or (b) the suite-state allocator
  layout — the r255 multi rehit ran after the essay + code faces
  (cold/rehit/rehit2 each) had seeded the block allocator, so the
  26-block prefix hit walked a different block-address layout than
  any run2 rung.

## The banked r255 reset record

The run2 dmesg snapshots (39 reset lines in the ring) hard-copy the
r255 crash resets for the bank — both GPUs, all three legs:
`ccs` resets at 23:34:17 / 23:53:22 / 00:14:13 with the `bcs`
follow-ups ~11 s later (matching the r255 VERDICT's account exactly),
plus context: earlier resets on Nov 26 at 04:51:46 (single-GPU ccs)
and 13:19:30 (ccs+bcs) — outside the r255 gate window (22:19-01:12),
i.e. ccs resets are not unique to the multi crash; the r255 TRIPLE
(byte-identical request + same request-state echo) remains the
deterministic crash set.

## Next rung (r257)

Replay the r255 sequence faithfully: one m8-posture boot, the devcore
watcher live, then (A) the faithful minimal — UNCUT multi text + the
enable_thinking kwarg, fresh -> A/A resend (the r255 request
byte-exact modulo max_tokens); then on the same boot (B) the actual
`p647_longprobe.py` full suite (the identical binary probe that
crashed r255: essay -> code -> multi, cold/rehit/rehit2 per face) —
crash expected at the multi second sample; capture the devcoredump
this time. If (B) crashes where (A) does not, the state delta is
isolated and the allocator-layout hypothesis wins.
