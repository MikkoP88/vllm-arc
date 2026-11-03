# P6.18 best-practices audit — are all fix changes in line with upstream practice?

**Date**: 2026-11-03 (owner directive: "deep research and analyze all fix
changes are included in line best practise project patches") · **Scope**:
every artifact the P6.18 campaign changed — the cert7 platform guard
(`vllm_gaudi/xpu/platform.py`, commit `848dd7ef`), the evidence trees
(`results/2026-11-03-p618b-repro/`, `results/2026-11-03-p618c-roundtest/`),
and the interim docs warning (`f3cc4681`). References are the pinned stock
wheel (`vllm-0.30.0+xpu`, the image's own `vllm` package — extracted from
`/root/build/arc_build/wheels/arc/vllm-0.30.0+xpu-*.whl`) and the upstream
plugin repo (mirror @ v0.30.0).

## 1. The guard's placement — the upstream-sanctioned hook

`vllm/platforms/interface.py:576-582` documents the contract of
`check_and_update_config`:

> "It can raise an exception if the configuration is not compatible with
> the current platform, or it can update the configuration to make it
> compatible with the current platform."

The guard is a raise of exactly that kind, in exactly that hook.
Upstream core platforms use it the same way for hardware-specific dtype
policy — `vllm/platforms/cuda.py:680-684` refuses `bfloat16` on sub-8.0
compute capability with `ValueError` at config time (constraint + hardware
fact + remedy: "You can use float16 instead by explicitly setting …").

**Artifact-level claim verified**: the hook has ONE core call site —
`vllm/config/vllm.py:1842`, at the tail of `VllmConfig` construction.
Every entry point (CLI `vllm serve`, python `LLM`, scripts, raw image
runs) constructs a `VllmConfig` before any worker or model load, so the
refusal fires on all of them before GPU allocation. Not a serve-script
guard.

## 2. The idiom — upstream itself gates turboquant exactly this way

`vllm/platforms/rocm.py:636-640` (stock wheel), backend selection:

```python
# … For any other dtype the explicit selection is genuinely invalid -> fail loud.
kv_dtype = attn_selector_config.kv_cache_dtype
if not (kv_dtype is not None and str(kv_dtype).startswith("turboquant")):
    raise ValueError(
        f"Selected backend {selected_backend} is not valid for …")
```

- the **`startswith("turboquant")` prefix match** over the dtype string is
  upstream's own idiom for this exact family of presets (the cert7 guard
  uses the same match, with an `isinstance(str)` guard instead of
  `str()`-cast — equivalent robustness for a field typed as the
  `CacheDType` Literal);
- **fail-loud ValueError with a why-comment** immediately above is the
  upstream pattern;
- the remedy-in-message practice (rocm:680-689 lists "Supported
  kv_cache_dtype values" for discoverability; cuda offers the float16
  alternative) is mirrored by the guard's closing line — "Serve with the
  certified default 'fp8_e4m3'".

## 3. Layering — why platform, not cache-config or backend

`vllm/config/cache.py:33-58` types `CacheDType` as a `Literal` that
includes all four `turboquant_*` presets: the strings are globally valid
ENGINE values; hardware compatibility is decided downstream. Upstream
decides it at the platform/backend layer (rocm at backend selection).
The guard decides at config time — the same layer upstream's bfloat16
refusal uses — which is strictly earlier (before worker spawn and GPU
allocation), i.e. fail-fast at the first layer that owns the decision.
The upstream Gaudi plugin's own `check_and_update_config`
(`vllm_gaudi/platform.py:143-163`) establishes the plugin-platform
pattern this fork's `ArcXpuPlatform` follows (subclass refines core
posture at construction time).

## 4. Project-law conformance (naming / scope / fork-model)

| Law | Status |
|---|---|
| Scope (§3/App I): donors admitted for shared components only; no new quant kernels | **OK** — the fix is a pure in-fork refusal; the roundtest proved no kernel defect exists, and new kernels exist in NO admitted donor, so deprecation is the only scope-legal disposition |
| Naming: no new env vars / flags / files for the fix | **OK** — one raise inside the existing platform hook; the naming audit in `848dd7ef` recorded zero new surfaces |
| Fork-model (§13): upstream tracked files untouched | **OK** — the guard lives in `vllm_gaudi/xpu/platform.py` (fork-owned); the stock wheel is byte-identical to cert6's (manifest md5s unchanged, gate phase [0]) |
| Marker-free commits, Conventional Commits, branch-per-item, squash after gates | **OK** — `feat/arc-p618c-tq-deprecation` → squash `848dd7ef` (`fix(p618c): …`) |
| Evidence banking: date-campaign results dirs | **OK** — `2026-11-03-p618b-repro/`, `2026-11-03-p618c-roundtest/`, cert7 gate tree; REDACTION note with byte-exact deltas |
| Comment style | **OK** — the guard's provenance comment (campaign, evidence pointer, mechanism, decision) matches `platform.py`'s house style (cf. the mem-info patch comment, `_WheelHookFinder` docstring) |

## 5. Documentation posture

The interim warning (`f3cc4681`: README + CONFIGURATION warning box,
"under re-validation, default unaffected") followed the repo's
certification-warning conventions and was merged BEFORE the fix landed —
correct sequencing for a corruption report. The final deprecation text
(warning box → platform-layer refusal statement with evidence pointers)
is the p618e deliverable once cert7 gates pass; no doc currently
contradicts the guard.

## Verdict

**Every P6.18 fix change is in line with upstream practice and project
law.** The guard's hook, idiom (prefix match + loud ValueError +
why-comment + remedy), layer (platform config time, earliest owner of the
hardware decision), and artifact-level reach are each directly patterned
on the pinned upstream wheel's own code; no upstream tracked file was
touched; no new config surface was introduced.

## Post-audit correction (2026-11-03, same evening)

The cert7 G0 guard-proof leg caught a correctness bug the audit above
could not see statically: the first build of the guard read
`cache_config.kv_cache_dtype`, but the wheel's pydantic field is
`cache_dtype` (`config/cache.py:111`; `--kv-cache-dtype` is only the CLI
flag name, and rocm's `kv_cache_dtype` lives on `AttentionSelectorConfig`,
a different object). The boot died on `AttributeError` at
`platform.py:276` before reaching the intended `ValueError` — for EVERY
dtype, fp8 included. Fixed the same evening (read `cache_dtype`, field
provenance comment added), runtime-verified against the real class:
`CacheConfig(cache_dtype=…)` × {4 TQ presets → REFUSE, `fp8_e4m3`/`auto`
→ allow}. Lesson recorded in the plan: a config-hook change needs a
runtime config-construction smoke — `py_compile` cannot catch
attribute-name drift between CLI flags and pydantic fields. The idiom,
layer, and reach analysis above is unaffected; the tag
`rel/v0.30.0-b70-cert7` moved to the fixed commit before any gate ran
green (first build failed G0 loudly; no broken image was ever certified).
