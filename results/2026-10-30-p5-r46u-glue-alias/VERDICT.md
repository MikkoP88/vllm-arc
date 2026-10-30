# r46u — p41c glue-alias defect: root cause, fix, validation

**Date:** 2026-10-30 · **Phase:** P5 (post-cert defect fix) · **Image proven:** vllm-arc:v0.30.0-b70-p46 (48b00dc4…) · **Outcome: GREEN — fix validated at import level AND on a live serve**

## Defect

Operator running the recommended configuration reported a log flood:

```
(Worker_TP0 pid=1027) arc-p41c glue-unavailable: RuntimeError('arc_p41_patch (p41 mamba glue) not installed')
(Worker_TP1 pid=1028) arc-p41c glue-unavailable: RuntimeError('arc_p41_patch (p41 mamba glue) not installed')
```

repeated every engine step, both TP workers.

Beyond the noise, the flood was the visible symptom of a **silent
posture regression**: `preprocess_state` fell back to the original
(zero-snap) implementation on every call — no LMCache mamba boundary
snapshots, no seed writes — i.e. every boot on r45-wheel-hook wiring ran
the p41c wrap DEGRADED, the exact disease p41c exists to cure.

## Root cause (proven — evidence/prefix_repro_import.txt)

The r45 `_WHEEL_HOOKS` rewiring (platform.py) loads the p41 glue as
`vllm_gaudi.xpu.patches.mamba_lmcache_glue` and attaches its symbols to
`vllm.v1.worker.mamba_utils`. The legacy name
`sys.modules["arc_p41_patch"]` — an artifact of the r30-cap sitecustomize
era, which loaded the glue file standalone under that name — was never
recreated. `mamba_hybrid_snap_glue._glue()` resolves the glue **only**
through that name, so it always raised. In-image repro: glue attached ✓,
wrap live ✓, `arc_p41_patch in sys.modules: False` (8×), and an
image-wide grep found no creator of the alias. The flood itself:
`_warn()` latched only its stdout print (3/tag); the `logger.warning`
fired every call.

## Fix (2 files, 3 edits)

1. `patches/mamba_lmcache_glue.py::_install` — register
   `sys.modules.setdefault("arc_p41_patch", mod)` (the patched
   mamba_utils module: the LIVE instance — never a fresh import of the
   file, which would split registry state across two module objects),
   and attach `_FP16_SEED_REPAIR` to `mod` (the fourth symbol the p41c
   glue handle reads; previously only reachable on the old standalone
   module).
2. `patches/mamba_hybrid_snap_glue.py::_glue` — fallback: if the alias
   is absent, resolve the glue from the patched `mamba_utils` itself
   (guarded by `hasattr(_lmc_snap_boundary)`).
3. `patches/mamba_hybrid_snap_glue.py::_warn` — latch the
   `logger.warning` to the same first-3-per-tag cadence as the stdout
   print (no more per-step spam even if a glue ever is missing again).

## Validation

- **Import level** (throwaway GPU-visible container, fixed files
  bind-mounted over the installed ones — evidence/postfix_verify_import.txt):
  13/13 PASS, `R46U_VERIFY: ALL_GREEN` — alias registered, all 5 glue
  symbols on the module, `_glue()` resolves primary AND fallback,
  p41c wrap live, post-wrap `_glue()` resolves.
- **Serve level** (operator's arc-test, hot-patched, relaunched with the
  full 22-var certified env chain — evidence/serve_log_digest.txt):
  HEALTH=200 in 180 s; `glue-unavailable` count **0** (previously every
  step ×2 workers); `[p41c] wrap live (bs=832 gids=3)` in both workers —
  the wrap's first-live-call print, proving `_glue()` resolved inside
  the engine; greedy completion GEN_OK (446 chars, coherent), count
  still 0 post-generation; 0 p41c warn/ERR lines.

## Consequences & follow-ups

- **All p45/p46-era boots ran p41c degraded** (wrap installed, glue
  resolving never): certified legs r42–r46t banked no `glue-unavailable`
  line because the curated keyline excerpts didn't include it and the
  stdout latch printed 3× early in boot — the per-step spam lived in the
  full logs' WARNING stream. Performance numbers for those legs include
  the degraded posture on the mamba-snap face (attn-side LMCache/APC
  unaffected); warm-prefix re-hit (740 ms) is attn-dominated. Re-run the
  cache-face benchmark on an r46u image before quoting mamba-snap gains.
- **rel/v0.30.0-b70-cert predates this fix** — building from that tag
  reproduces the defect. Until a re-certified tag is cut, build from
  `main` (≥ the r46u commit); README build step updated accordingly.
- Operator's posture deltas (recorded, unvalidated, operator's choice):
  model checkpoint `swift-qwen3.8-27b` (not the certified
  `qwen3.8-27b-fp8`); the earlier interactive serve also ran WITHOUT the
  22-var env chain (no `VLLM_XPU_ENABLE_XPU_GRAPH=1`, no `LMCACHE_*`,
  no oneCCL block) — restarted from `/root/serve_cert.sh` with the full
  chain.
- Rig state after the leg: arc-test left serving (fixed posture) at the
  operator's request-in-progress; W14 radiance-test untouched.
