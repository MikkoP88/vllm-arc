# VERDICT — p2r3e-zombie-noshow (2026-10-27)

**LEG: diagnostic (hot-patch canary overlay, image ce0ee65f3766 @ 17ae84a7
+ commit 20536a5d content). The zombie DID NOT FIRE in two consecutive
boots — including a forced fresh-compile boot. Boot-nondeterministic; the
compile-vs-cache theory is FALSIFIED as the sole cause. Canary stays armed.**

## Boot 1 — cache-hit (16:04, container from p2r3d, AOT cache warm)

- "Directly load AOT compilation" (both ranks), compile 6.25 s, boot 39 s,
  campaign FULL 27/27 at 16:05:57–16:06:02.
- Warmup outputs log `applied — device-sync barrier` ×6 (budget 3/rank).
- Requests: 5-tok (Paris — coherent, matches the cache-hit-generation
  baseline EXACTLY) and ~838-tok quick-brown-fox ×93 (coherent repetition
  + summary onset). Both 200; health 200 after each; ZERO defer lines.
- Engine alive throughout.

## Boot 2 — forced fresh compile (16:10, same container, cache wiped)

- /root/.cache/vllm/torch_compile_cache removed (522 MiB) → recompile:
  "saved AOT compiled function" ×1/rank, boot 340 s, campaign 27/27.
- 5-tok request: 200, text BIT-IDENTICAL to boot 1 (" Paris.\nThe capital
  of Germany is Berlin...") — same weights, same seeds, deterministic.
- ~838-tok request: 200, finish stop, health 200. ZERO defer lines, zero
  canary, zero deaths.

## Reading

- p2r3b/c/d: three consecutive boots ALL poisoned (defer + death).
- p2r3e: two consecutive boots (cache-hit AND fresh-compile) both clean,
  same image, same container, same env, same request shapes (incl. the
  832-tok prefill that killed p2r3b).
- ⇒ The zombie is NOT deterministic in compile mode, container freshness,
  or request shape. Remaining suspects: a race in the boot tail (campaign
  exit ordering), a rare path in the first-step scheduler (e.g. chunked
  prefill interplay), or state left by an earlier process in the same
  container/device session. The canary instrument (20536a5d) answers the
  execution question the moment a poisoned boot recurs.
- Separately this session: the vxk wedge audit (owner check-request)
  found the pinned wheel BARE — p13 contingency running in parallel
  (results/2026-10-27-p1-vxk-wedge-audit/). Defect-b and the wedge are
  discriminated (five independent discriminators, see that VERDICT).

## Artifacts

serve_full.log (boot 2 full log: cache-cleared boot, saved-AOT, campaign,
both clean requests, health 200s), gso_hotpatch_canary.py (the overlay
content = commit 20536a5d).
