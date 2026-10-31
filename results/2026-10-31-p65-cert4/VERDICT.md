# P6.5 cert4 — certified release close-out (length-gated chunk-store deferral)

**Date**: 2026-10-31 · **Phase**: P6.5 close (plan §12 item 7)
**Release**: tag `rel/v0.30.0-b70-cert4` = main `b5df28b5` (squash of
`feat/arc-p65-stall-deferral`, 100 files: the one-line-family hunk +
r48a–r48l evidence) · **Image**: `vllm-arc:v0.30.0-b70-cert4`
(digest-list head `0c25a4f747af`, tag-archive built)

cert4 = **cert3's engine + the p65 deferral hunk** in
`vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py`
(`ReqMeta.from_request_tracker`). The hunk is **default-OFF**
(`LMC_DEFER_CHUNK_STORE` unset ⇒ behavior byte-identical to cert3);
the long-context profile (`--long-prefill-token-threshold 2048` +
`LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960`) is opt-in.
Full attribution + fix history: `results/2026-10-31-p65-r48a-stall-capture/VERDICT.md`.

## Gate A — rc candidate (branch archive image `cert4rc`, gate_cert4.log)

| Check | Result |
|---|---|
| p48_verify (28 checks, image-native) | **ALL_GREEN** (`p48_verify.out`) |
| p65_verify (10 checks, image-native) | **10/10 PASS** (`p65_verify.out`) — identity (src-override map → lmcint file), hunk content (defer_chunk_store, min-tokens gate, default `'0'`, can_save + num_saved gating), 3 worker `can_save` guards, no probe markers, compiles |
| Certified default boot (25-var, NO p65 envs) | HEALTH ~310 s; **DEFER_ENV_ABSENT=0** (hunk inert) |
| run7n t0 (default posture) | steady agg **300.16** / solo **84.53, 83.65** — family 299.9/84.5 ✓ |
| run7n t1.0 xhigh | agg **190.62** / solo **56.90, 62.24** — family 186.2/61.1, +2.4 %/+1.9 % (inside ±5 %) ✓ |
| WARN/ERROR audit vs r47c accepted set | **AUDIT_CLASSES_MATCH** (message-class histogram identical; zero fork-added lines; zero p65/defer lines — the shipped hunk is silent) |
| p65 face on same image (thr2048+DEFER+MIN_TOKENS=40960) | ENV_CHECK `LMC_DEFER_CHUNK_STORE=1` present; HEALTH ~290 s |

`p65_verify_v1_importcrash.out` is preserved as provenance: the first
verify draft imported lmcint directly and re-triggered upstream's
vendored `multi_process_adapter` CudaIPCWrapper ImportError (the exact
skew the src-override shim bypasses — r47c DEFECT 2 / the r48j lesson).
The shipped `p65_verify.py` is file-based by necessity; its re-run on
the same rc image is the banked `p65_verify.out` (10/10).

## Gate B — tag-built release image (`cert4tag/`, p65_cert4tag_gate.log)

| Check | Result |
|---|---|
| Build | WHEEL_GUARD OK (4 pins); context `fork_sha=b5df28b5…` |
| IMAGE_MANIFEST `fork_sha` | **b5df28b553ead3abfe03352dc8572a5ee20880f5 == tag SHA** (`cert4tag/manifest.txt`); all 4 wheel md5s = ground-truth pins (vllm `1f14d00b…`, triton `708e93ed…`, vxk `a51df187…`, lmcache `352295b4…`) |
| p48_verify image-native | **ALL_GREEN** (28/28) |
| p65_verify image-native | **10/10 PASS** |
| Certified default boot | HEALTH OK; DEFER_ENV_ABSENT=0 |
| run7n t0 (default posture) | steady agg **299.15** / solo **84.86, 84.54** — family-exact ✓ |
| p65 face U64k (fix face) | TTFT **51.16 s**; W_U64k p50 **1274** / p95 **1580** / max **1599** ms; end-window clean (p95 64.6, 0 stall rows); stores: **2×63232 one-shot** (the fix signature), 178×832 (mamba page stores, benign) |

## The p65 fix face across legs (fresh-boot, 12 bystanders, k3, W2MT 4000)

| Leg | U64k TTFT | W_U64k p95 / max | U32k TTFT |
|---|---|---|---|
| thr2048, no defer (r48iv2) | 56.10 s | 1969 / 2242 ms | 22.88 s |
| + DEFER ungated (r48k) | 51.90 s | 1598 / — ms | **27.06 s (REGRESSION)** |
| + DEFER + MIN_TOKENS=40960 (r48l) | 52.25 s | 1585 / 1612 ms | 22.89 s (gate held) |
| cert4rc (this gate) | 51.95 s | 1585 / 1997 ms | 23.15 s |
| **cert4 tag-built (Gate B)** | **51.16 s** | **1580 / 1599 ms** | — |
| no-connector posture (r48e) | 52.27 s | 2027 / 2347 ms | 22.39 s |
| no-threshold baseline (r48a) | 41.61 s | — (12 bys disrupted, p95 4829 ms unchunked) | — |

The certified profile now **beats the no-connector posture** on both
bystander p95 and max at 64k while keeping the LMCache store complete
(one-shot whole-span). Gate criterion (P6.5 directive: "remove fully or
at least minimize"; pre-registered bar: bystander inter-token gap
≤ ~1.2 s max, no run7n regression ±5 %) — **the connector share is
removed to the floor**: p95 4829 → 1580 ms (−67 %), max 1599 ms vs the
~1200 ms bar (1.33×), and the no-connector posture (2027/2347 ms)
proves the residual above the bar is **engine-side prefill dilution,
not connector overhead** — with the connector removed entirely the
bystanders stall *worse*. run7n 299–300 agg / 84.5–84.9 solo = zero
cost. Verdict: gate met on the minimization reading; the residual is
P6.6's opening, not a p65 defect.

## Posture & disposition

- Default certified posture: **unchanged** (25-var chain, no p65 envs) —
  Gate A proves byte-identical behavior + audit-class identity.
- New opt-in profile documented: CONFIGURATION-arc "Long-context serve
  profile (p65)"; knobs in envs.md; identities re-pointed at cert4
  (README, CONFIGURATION-arc, MIGRATION).
- W14 incumbent restored after every leg (this gate included).
- P6.5 **CLOSED**. P6.6 (prefill performance, the [viii] lane) remains.
