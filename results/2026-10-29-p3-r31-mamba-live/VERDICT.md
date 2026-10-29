# r31 — mamba half LIVE, first external-load exercise — VERDICT: CHAIN PROVEN, INTEGRITY FAIL (root cause #1 fixed, #2 scoped)

Date: 2026-10-29 (rig legs 01:40–03:10). Posture: p2 product (mode=none +
FULL_DECODE_ONLY, MTP k4, fp8, kv fp8_e4m3, ssm fp16, APC on) + LMCache stack
p40+p41+p41b+p41c + lmcint kit with r31 fork deltas. Serve
`/root/r31_serve_dbg.sh` (PYTHONHASHSEED=0, VLLM_ARC_LOOKUP_ELIDE=0,
ARC_SNAP_PROBE=1, DEBUG logs), boot ~3 min, health 200.

## What was fixed this leg (deploy chain)

1. **Orphaned-worker GPU holds** (VLLM:: titled procs survive
   `pkill -f python` — kill by pid; setproctitle hides them).
2. **The 256-byte floor family** (4 ceil sites + prefix-copy/zero-tail +
   load-side seed slice in the lmcint adapter — see
   `external_patches/lmcache/README.md` r31 section). Symptom pre-fix:
   `Mamba store seed size mismatch: 41189376 vs obj 41189120` ×56, then
   HTTP 500 `Mamba seed size 41190016 != expected 41189376` on first load.
3. **p41c V2-path re-anchor** (mamba_hybrid_snap_glue.py, untracked→banked
   this commit): wraps `MambaHybridModelState.preprocess_state`; snaps the
   deepest COMPLETE boundary after the fused pre-copy launch; consumes
   adapter-parked seeds before it. 56→44 snaps/leg fire with exact
   predicted math (B = (ncc//bs)*bs).

## The chain is LIVE (first time anywhere in the donor lines)

- 44 snaps → 44 `LMC mamba store` (2 ranks), registry deepest per req.
- Joint lookups book `min(attn_hit, mamba_hit)` on every ladder face:
  6272/4480/2688 (mamba keys reach L−896: the final boundary never snaps —
  no post-final-step preprocess; p43: final-boundary snap at finish).
- 12 seeds parked/leg (B=7168, bytes=41,189,376 exact — ceil+slice working),
  12 seed writes, 12 sidx corrections (E2), zero seed-retrieve failures,
  zero size mismatches, engine stable through ladder + battery + 3 boots.

## Gates (battery, `battery/` dir)

| cell | result | verdict |
|---|---|---|
| survival | 200/200, onset count present | PASS |
| spec pos1 | 0.872 (hard 0.5, target 0.85) | PASS (P2 band low was 0.877 — noted) |
| golden vs banked ref | **len=4096 DRIFT** (4/5 match) | **FAIL** |
| warm-hit | cold 5855.4 / warm 1055.4 ms | FAIL (bar 750; p43 lane) |
| integrity 5×8k | **0/5 cold-vs-rehit sha** | **FAIL** |
| churn | 24 colds ~5860 ms; rehit 1058/1046 ms both warm-class | N/A (no pool pressure — redesign pending) |
| run7n agg | 193–211 tok/s | band-consistent |
| stepprobe | script path stale (r22a) | skipped, non-gate |

Ladder (1-token, 5 faces ×4): warm==cold sha on 4/5 faces; L=7168 face
deterministically adc83b19≠4345cb1f on all warms. Cross-boot cold shas
themselves drift on 2/5 integrity prompts (random-id prompts are
knife-edge-heavy; the certified golden set is the stable reference).

## Root cause #1 — FOUND and FIXED (upstream defect, first live path)

`vllm/v1/worker/gpu/model_states/mamba_hybrid.py:121` (0.30):

```python
self._mamba_state_idx_gpu[req_index].fill_(
    (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
)
```

`cache_config.block_size` is the ATTENTION block size (64); the column
units are mamba blocks (896). A remote-loaded resume (B=6272) seeds the
running-state column at **97** instead of 6 → the fused pre-copy carries a
FOREIGN page and the parked seed is never read. Upstream never exercises
num_computed>0 at add_request (fresh reqs seed −1; vllm-APC resumes
reattach cached mamba blocks via the hybrid KV-cache manager) — the
external-load resume is the first live path. Fix (p41c wrapper, probe-print
`p41c-sidx`): `self._mamba_state_idx_gpu[slot] = prev` for seed rows before
orig(). Confirmed firing (10/leg). **Integrity remained 0/5 → this defect
was necessary but not sufficient.**

## Root cause #2 — SCOPED (open)

Key realization: the joint clamp (`mamba_hit=0 → 0`) means **r30's "hits"
were vllm-APC shadows — the external-load path (attn AND mamba) was never
exercised before r31** (r30 engine-stat "External prefix cache hit rate
0.0%"; r31 battery reads 14.8%). The corruption persists with fp16 repair
off (E1) and with the sidx fix (E2), so at least one more defect lives in
the never-before-live external-load path:

- (a) the ATTN external load (C's V2 paged-connector load onto 0.30 attn
  blocks — fp8, block 64) — unmodified by us, never live in C's own
  champion either (their mamba half was equally dead → their 750 ms bar was
  also APC-only);
- (b) the mamba SEED CONTENT (snap bytes ≠ the state the cold chain
  carries — conv-ring `token_bias` semantics of `_copy_mamba_state_block`
  are the prime suspect; the migration writes boundary blocks at a biased
  slot, and a verbatim restore + bias-0 read may double-rotate the conv
  window).

Ruled out this leg: fp16 seed-repair asymmetry (E1: repair off → 0/5);
seed byte accounting (parks exact 41,189,376; roundtrip is pure CPU);
`gpu` num_computed zeros in probe lines (StagedWriteTensor materializes at
data_ptr — measurement artifact, kernel reads real values).

**Next-session plan (instrumented dumps, one boot):** for one 8k prompt,
(a) md5 the snap bytes at pack time (probe print) and at seed write;
(b) dump the attn KV blocks [0,B) of the warm req after load and of the
cold req at finish, compare; (c) dump the carried-from block content at the
resume step vs the cold chain's same-depth block. First divergence localizes
(a) vs (b). If (b): the C adapter's 0.30 attn-load port is defective → fix
or fall back to APC-only posture (joint clamp keeps that safe by design —
`VLLM_ARC_LMC_MAMBA_SNAP=0` is the interim safe knob: mamba keys never
bank → joint 0 → APC-only, r30-equivalent behavior).

## Warm-perf regression (p43 lane, secondary to correctness)

Cold 5855 vs r30 5556 (+300 ms: 44×41 MB snap D2H syncs + CPU pool packs).
Warm 1055 vs bar 750: snap D2H stalls + seed load (41 MB retrieve/park/H2D
×2 ranks) + the −896 joint-clamp recompute (missing final boundary).
p43 design: async pinned-buffer snaps (event-synced at store), final-
boundary snap at request finish (kills the −896), seed load overlap.

## Rig state

arc-test STOPPED; radiance-test (W14) restored + health 200 (see session
log tail). Never left on arc image overnight.

## Files

- `ladder10_probe.txt` — 5-face ladder (cold+3 warms, shas, ms)
- `seedchain_excerpt.txt` — stores/snaps/parks/joint/hits log excerpt
- `serve_ladder10.log` — preserved full serve log (ladder run)
- `serve_e2.log` — preserved full serve log (E2 sidx-fix integrity run)
- `r31_integrity_E2.txt` — E2 integrity replicate (0/5)
- `battery/` — full battery cells (spec/golden/lmc/run7n/statsmon)
