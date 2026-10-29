# r34 — warm-hit via `disable_eagle_block_drop`: mechanism CONFIRMED, integrity BROKEN — falsified and reverted

Date: 2026-10-29 (rig legs 16:24–17:20). Posture: r33 product + owner-pinned
`--mamba-cache-mode align` (plan Appendix F). Boot script
`/root/build/r30/r30_serve.sh` (rig); the single leg change was
`"disable_eagle_block_drop":true` inside the speculative-config JSON.

## Hypothesis (from r33 decomposition, static + scheduler code read)

`use_eagle_block_drop` (active for MTP: `use_eagle()` includes "mtp";
speculative.py:1909–1917) backs `last_cache_position` one 896-block off in
`_mamba_block_aligned_split` (scheduler.py:413–521) AND prunes the trailing
attn hash block → an 8192-prompt banks only 7168 → warm re-hit recomputes the
1024-token residual (~770 ms) → warm 1021 ms vs 750 gate. The 1024-token
1k-face prompt's only full block is always dropped → zero mamba banks →
warm 9.2 s ≈ cold. Disabling the drop should bank 8064 / 896 respectively.

## A/B (probe boots, ARC_SNAP_PROBE=1, timers ON — orientation only)

| face | drop ON (baseline) | drop OFF | delta |
|---|---|---|---|
| N1 8192 warm med | 1390.5 ms (r33 clean: 1034.9) | **836.2 ms** | −554 ms mechanism-real |
| N3 1k warm med | ~9200 ms (r33) | **418.8 ms — GATE PASS** | 1k face banks 896 ✓ |
| N4 8064 | 9c1623f0 stable | 9c1623f0 stable, ~1330 ms | unchanged (exact-aligned already exempt) |
| snap boundary | B=7168 | **B=8064** (ncc=8064, idx=8) | scheduler half works |
| LMCache stores | Stored 7168 | Stored 8064 + 896 | as designed |
| scheduler log | (warning absent) | "EAGLE trailing prefix-cache block dropping is disabled…" (scheduler.py:289) | flag live |

Engine-side capture (fix boot): `[r31probe p41c-snap … ncc=8064 B=8064 idx=8
bs=896]` per request; `Stored 8064 out of total 8064` ×2 ranks; zero
`Retrieved` lines — warms ride GPU-resident hybrid APC (+ residual), the
external pool is not consulted when APC already covers the cacheable extent.

## The break (E-law capture, deterministic)

Golden battery after any N4-class cell (order identical to r33's passing
battery):

- `N4 → golden` (run 1): INTEGRITY **3/5**; rerun: 5/5; `N4 → golden` (run 2):
  **4/5** — golden[2] cold=`37a00b32da76` vs warm=`8fa239e95ea3` (the stable
  value); `N4 → N2 → golden` (run 3): **4/5** — golden[2] same wrong value
  `37a00b32` (deterministic) AND golden[3] moved to a new stable value
  `d9f601923574` (from `a15739023154`).
- Discriminator (r34_disc.py, same prompt g2=seed 30303002, mt=1 vs mt=64,
  N4 cell in between): `A1=ca978112 → A2=56ee5b50` (**PREFILL_FLIP** — same
  warm-cached 1-token request, different bytes after an unrelated prompt's
  cell), `B1==B2==37a00b32` (DECODE_STABLE). Outputs are engine-HISTORY-
  dependent: a re-hit's served/recomputed bytes depend on what ran before.

Attribution control (flag reverted, fresh boot, same failing sequence):
`N4 → golden` = **5/5** (golden[2] back to `8fa239e9`), N1 warm med 1033.1 —
the break was caused by the flag, exactly the hazard upstream documents:
"EAGLE trailing prefix-cache block dropping is disabled. This is experimental
and may affect speculative-token acceptance rates." (scheduler.py:289). With
spec decode on, retained trailing blocks carry draft-token contamination;
re-hits that read them become history-dependent.

Not-a-regression note: cross-BOOT marginal flips (single-token argmax ties)
pre-exist — r33 saw N1 flip `01ba4719`↔other across boots; today's clean boot
gives N1=`7eb70257` with within-boot cold==warm 6/6 intact (and equals
N5-8960's sha trivially: both 1-token outputs emit the same token text). The
flag's break was qualitatively worse: within-RUN cold≠warm, deterministic
wrong values, mt=1 re-hit flips.

## Re-gate battery (E4 clean boot, first-traffic, timers-off)

ARC_SNAP_PROBE stripped; align flag kept (owner pin — exonerated: the +356 ms
on today's probe boots was telemetry cost, clean boot returns to r33 numbers).

| cell | r33 | r34 reverted | verdict |
|---|---|---|---|
| N4 8064 | 1298–1346, `9c1623f0` | 1287–1293, `9c1623f0` | parity ✓ |
| golden 5×8k | 5/5 | **5/5** | integrity restored ✓ |
| N2 7168 | med 1257.1, `2b4c342f` | med 1257.9, `2b4c342f` | parity ✓ |
| N5 8960 | ~1303, `7eb70257` | 1256–1348, `7eb70257` | parity ✓ |
| WARM5 8192 | med 1021.5, `d03502c4` | med 1021.6, `d03502c4` | parity ✓ |
| N1 8192 | med 1034.9 | med 1033.1 | gate ≤750 still FAIL → r35 lane |

Engine posture (r34_engine_posture.txt): `mamba_cache_mode: 'align'` +
`num_speculative_tokens: 4`, 0 probe prints, 0 "disabled" warnings, block
size 896 both ranks.

## Verdict + next (r35)

`disable_eagle_block_drop` is FALSIFIED for this contract: it delivers the
warm-hit mechanism (−554 ms on N1, N3 pass) but violates the 0-drift
integrity gate deterministically. The path to ≤750 ms WITHOUT touching the
spec-protective drop is the r32-named **final-boundary-at-finish snap**
(p41c glue side): bank the deepest full 896-block of the final partial chunk
at request finish — N1 banks 8064 (residual 128), N3 banks 896. Open design
question captured this leg: the warm re-hit needs the ATTN side to serve
[7168,8064) too — GPU APC won't hash the trailing block with drop ON, so the
finish-snap must make the external (LMCache) pool the load source for that
span (external complements APC in the D-lineage load design), or the residual
stays 1024. Also carried: first-warm blip (~5.9 s) on every cell's first
re-hit; N4/N2 warm ~1290/1260 (decode-window dominated, mt=1 — separate
decomposition); load-cost (~210 ms/8064) reduction follow-up.
