# r33 — N2/N4 warm re-hit integrity break ROOT-CAUSED + FIXED: attn-KV pool is per-token K/V interleaved; byte-true NL_X_NB_BS_NH_TWO_HS path

Date: 2026-10-29 (rig legs 14:40–16:00). Posture: r32 product (p2 + p40/p41/
p41b/p41c + lmcint kit) + the r43 attn-KV format fix banked here. Serves:
probe boots `/root/build/r30/r30_serve.sh` with ARC probes (removed at E4),
gate boots the same script clean (timers-off, first-traffic battery).

## Symptom (carried from r31/r32)

N4 (8064-token seeded prompt, 896 decode): cold sha always
`9c1623f0d38e28e9`; warm re-hits byte-stable WITHIN a boot but WRONG and
boot-varying among a small set (`36a9e7f1c95b82ff`, `d4735e3a265e16ee`,
`b0ad6bc193119775`). r32 fixed the MAMBA half (exact-boundary snap law,
integrity 5/5 on its cells) — the attn half stayed broken. Round-trip was
self-consistently byte-identical (store misparse == load misparse), so r31's
"5/5 byte-identical re-hits" measured the ROUNDTRIP, not the CONTENT.

## Capture (r43 probes, this leg)

- v10 full-pool snapshot diff + per-page md5 on the store of a fresh 8k
  request: slot mapping proven correct (v11: `pages896 == bt row` exactly);
  the pages LMCache WROTE were aliased — predicted fold `slot P*896 lands
  at page P//2` matched **8/8 aliases** ({7→3, 8→4, 10→5, 12→6, 15→7,
  25→12, 26→13, 38→19}).
- diag5 (live strides at the store seam): the engine's FA-XPU attn pool is
  ONE stacked tensor per layer `(476, 2, 896, 512)` with strides
  `(917504, 512, 1024, 1)` — **NOT contiguous in its logical shape**:
  physical order is `(NB, BS tokens, K/V, kvdim)` — per-token K/V
  interleaving (element `(b,kv,t,d)` at `b*917504 + kv*512 + t*1024 + d`).
  Earlier "pool = (page, K-half, V-half)" model was WRONG; a K-plane
  tensor view of this pool is impossible (interior padding).

## Root cause chain (pinned ~100%)

1. **Detection.** The connector's contiguity recovery permutes the raw
   rank-4 pool (stride-sorted, zero-copy) to `(NB, BS, 2, kvdim)`; the
   VLLM detector's rank-4 "fused-KV" branch returns `NL_X_NB_BS_NH_CS`
   (NH=2 = our K/V axis, CS=512 = our kvdim). Byte strides match — but
   the transfer's slot model for that layout counts K/V as separate slot
   units → the P//2 fold (verified 8/8).
2. **Transfer backends.** Two candidate executors, NEITHER can express
   per-token interleaved K/V on XPU:
   - native `lmcache.xpu_ops multi_layer_kv_transfer` (csrc/sycl/
     mem_kernels_sycl.cpp) implements only `NB_NL_TWO_BS_NH_HS`,
     `NL_X_TWO_NB_BS_NH_HS`, `NL_X_NB_TWO_BS_NH_HS` (+ MLA) and raises
     `Unsupported non-MLA EngineKVFormat` otherwise — this is exactly
     why p40d keeps the four KV-transfer ops on the torch baseline
     (verified again by synthetic call this leg: TWO_HS → throw, both
     directions). The CUDA kernel (csrc/cuda/mem_kernels.cu:310) HAS the
     `NL_X_NB_BS_NH_TWO_HS` branch — the SYCL port never got it.
   - the torch fallback (`v1/platform/torch_ops.py:598`, the e01 C-delta
     file — the LIVE path under p40d) has MLA / flash_infer / generic
     branches only; TWO_HS fell into the generic else-branch, which
     wraps the pool pointer as `(2, page_buffer_size, hidden)` K/V
     half-planes. On this interleaved pool that view computes physical
     element `s*512` for slot `s` → page `s//1792` = **P//2 — the same
     fold**, which is why fix2's warm sha (`d4735e3a265e16ee`) matched
     the pre-fix CS-world boot exactly.

## Fix (three iterations, one final change)

- fix1 (rejected): 5-D reshape `(NB, 2, BS, kvh, 128)` → detector has no
  matching branch → `ValueError: unsupported kv_caches structure` at
  `_initialize_pointers` → **connector poisoning** (pointers cached
  before normalize; later calls early-exit; every store dies at
  from_gpu:265 → "Lagged attn store failed" ×80 → 0 Retrieved; warms rode
  vLLM APC recompute — correct shas, not LMCache).
- fix2 (necessary, not sufficient): adapter hands a contiguous ZERO-COPY
  view `pool.permute(0,2,1,3).reshape(NB, BS, 1, TWO, kvdim)` (permute of
  this non-contiguous pool IS contiguous in (NB,BS,K/V,kvdim) order) +
  kit detector override mapping it to `NL_X_NB_BS_NH_TWO_HS` (NH=1,
  HS=kvdim). Detection then perfect (`r43fmt`: fmt=NL_X_NB_BS_NH_TWO_HS,
  bs=896 nb=476; strides (917504,1024,1024,512,1) contiguous; 17 layers) —
  but the transfer still folded (else-branch, above).
- **fix3 (the fix)**: kit source-override of
  `lmcache.v1.platform.torch_ops` wrapping `multi_layer_kv_transfer` with
  a true TWO_HS branch — pool viewed physically `(NB, BS, 2, HS)`,
  vectorized advanced indexing per layer:
  `D2H: kv[:, l, valid, :] = pool[bidx, boff, :, :].T`;
  `H2D: pool[bidx, boff, :, :] = kv[:, l, valid, :].T` — byte-true per
  token (`two*HS + hs`), one slot per token, both K and V per slot. All
  other formats pass through to the original implementation unchanged.

Files (rig + repo, identical): `patches/lmcint/lmc_kvfmt_detector.py`
(detector override), `patches/lmcint/lmc_twohs_transfer.py` (transfer
branch), `patches/lmcint/vllm_v1_adapter.py` (`_arc_attn_pool_view` +
seam transforms at register/store/load sites).

## Evidence (fix3 boot, probe build, timers-off cells)

- N4: cold `9c1623f0d38e28e9`; warm[0..2] ALL `9c1623f0d38e28e9`;
  `Retrieved 7168 out of 7168` ×6 (real LMCache loads, not APC);
  `[r43xfer] TWO_HS interleaved branch live: NB=476 BS=896 L=17 H=512
  nslots=896 dir=D2H`; 0 "Lagged attn store failed"; 0 "Unsupported".
- N3 (1k face): shas all identical cold==warm (`8a5edab282632443`) —
  integrity holds; timing 9.2 s ≈ cold (KNOWN open: the 1k face's mamba
  half never banks (m-lookups res=0) → joint lookup misses → recompute;
  this is the p43 warm lane, NOT a regression of this fix).

## E4 (instrumentation removed)

Deleted: `r43_scan_probe.py`, `r31_key_probe.py`,
`r31_mamba_hybrid_probe.py` + their 4+1 sitecustomize entries + the 3
probe envs in `r30_serve.sh`; adapter diag prints (r43meta2/r43load/
r43fmt/r43fmtkeys/r43store/r43norm) stripped; `mamba_hybrid_snap_glue.py`
restored to the repo HEAD blob (md5 286c6105). The one-shot
`[r43kvfmt]`/`[r43xfer]` boot prints stay (product self-identification,
same pattern as the p40c/p40d hook prints).

## Re-gate battery (clean boot, first-traffic, timers-off; logs in `battery/`)

| cell | integrity | cold | warm | note |
|---|---|---|---|---|
| N4 8064 (9×896 exact) | 4/4 `9c1623f0d38e28e9` | 14,336.7 ms | 1,298–1,346 ms | 11× speedup; `Retrieved 7168`×6 |
| Golden 5×8k (r31_integrity) | **5/5 match** | ~6.8–8.2 s | ~2.0–3.4 s | zero drift |
| N2 7168 | 6/6 `2b4c342f5433ebe5` | 5,617.7 ms | med 1,257.1 ms | |
| N5 7680 (partial-tail) | 3/3 `cc7e4412564ba8a7` | 5,599.2 ms | ~1,290 ms | partial tails MISS by design (save_unfull_chunk=False) → recompute = correct bytes |
| N5 8960 (exact) | 3/3 `7eb70257593da06f` | 6,413.2 ms | ~1,303 ms | |
| WARM5 8192 | 6/6 `d03502c43d74a30b` | (5,962.8 first-warm) | **med 1,021.5 ms** | gate ≤750 FAIL → p43 carry-over (was 1,047 ms at r32 — improved) |
| N1 8192 | 6/6 `01ba4719c80b6fe9` | 5,878.1 ms | med 1,034.9 ms | gate ≤750 FAIL → p43 carry-over |

Engine: 0 "Lagged attn store failed" (r32-era ×80 storm gone), 0
"Unsupported", `[r43kvfmt]`/`[r43xfer]` boot lines present on both ranks
(`r33_engine_evidence.txt`). Clean cold (14.3 s vs 94 s on probe boots)
confirms the E1 telemetry cost — gate numbers only ever from clean boots.

Warm-hit gap decomposition (p43 lane, unchanged mechanism): warm ≈ load
(~0.2 s) + recompute of the LAST partial chunk (8192−7168 = 1024 tokens
≈ 0.8 s) + 1-token decode → the r32-named final-boundary-at-finish snap
(bank the last partial chunk at request finish) is the 300 ms closer.
N3 1k face (9.2 s warm) = mamba half never banks on single-chunk prompts
(m-lookups res=0) — joint lookup misses → full recompute; same p43 lane.

## Follow-ups (REVAL ledger additions)

- native SYCL TWO_HS branch (CUDA-parity) — perf follow-up only; the
  torch-baseline branch is C's posture and is correct.
- 1k-face mamba banking (m-lookups res=0) — p43 warm-hit lane with the
  ≤750 ms gate.
- `External-prefix-cache-hit-rate` counter still 0.0% during verified
  hits — not gate-usable (truth = sha + timing), unchanged.
