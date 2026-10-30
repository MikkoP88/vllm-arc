# p45 — wheel-hook registry (in-image patch loading, no PYTHONPATH overlay)

## What

`vllm_gaudi/xpu/platform.py`:

- **`_WHEEL_HOOKS`** — exec-wrap registry for patches whose target is a
  WHEEL module (vllm-core or lmcache files the fork must never edit in
  place): 7 targets → patch modules (mamba_utils, mamba_hybrid
  model_states, lmcache_connector, lmc device_ops, lmc kvfmt detector, lmc
  torch_ops, sched.scheduler). A one-shot meta-path finder inserted at
  `register()` time (earliest per-process moment, imports nothing,
  circular-safe) wraps the first import of each target to call
  `patches.<mod>._install(module)`. Idempotent; per-hook visibility log
  (`wheel hook <target> <- <patch> OK`); the finder retires after the last
  hook fires. In-place patches (`gdn_metadata_fastpath`,
  `fp8_block_w8a16`) stay on `_XPU_PATCH_MODULES`.
- **`_load_wheel_patch(patch_name)`** (p45b fix, 00af061a) — sub-package
  patch names load STANDALONE via `spec_from_file_location` under qualname
  `vllm_gaudi.xpu.patches.<name>` (sys.modules-cached, idempotent);
  non-dotted names keep normal import. Reason: `lmcint/__init__` IS the
  lmcache_integration entry (heavy adapter import) and must never execute
  mid-lmcache-init — the first p45 boot died on exactly that re-entry in
  both workers. This reproduces the r30 cap-sitecustomize semantics
  (standalone file exec) that legs r30–r41 ran on; files byte-identical
  cap↔wheel.

## Why

The certified image must be self-contained: no cap `PYTHONPATH` overlay, no
runtime installs (Appendix H self-containment law). The P3 glues existed
as cap-sitecustomize hooks; p45 is their in-image product form so the
serve line is a plain `vllm serve`.

## Test

Every boot greps: `wheel hook .* OK` = 7/7 per process, zero `FAILED`;
reroute/kvfmt/TWO_HS lines present; hooks visible in all TP worker
processes (each process registers its own finder).

## Validation

- 7fd210d7 (p45 wiring) + 00af061a (p45b standalone loader — root-caused
  circular ImportError ×2, E-law capture in the commit).
- r42: first fully standalone certified-image boot (v0.30.0-b70-p45b) —
  hooks 7/7 every process zero-FAILED, full battery green, golden re-bank.
- r43/r44: k5 adoption + 12-cell matrix on the same loader.
- Appendix H law (banked r42): sub-package wheel hooks ALWAYS load through
  the standalone loader; future sub-package patches must stay
  standalone-importable (only relative import allowed is the package
  `__init__` itself).
